# ------------------------------------------------------------
# SMAFormerV2: Synergistic Multi-Attention Transformer V2
# Based on IEEE BIBM 2024 SMAFormer with Swin Transformer backbone
# 
# Key Features:
# 1. Swin Transformer backbone (loads pretrained weights)
# 2. SMA (Synergistic Multi-Attention) blocks as per paper
# 3. E-MLP (Enhanced MLP with depthwise conv) as per paper
# 4. Symmetric U-shaped encoder-decoder with skip connections
# 5. Edge enhancement in final layer
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.swin_transformer import SwinTransformer
import math
import os

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


# ============== SMA Components (Following Paper Description) ==============

class PixelAttention(nn.Module):
    """Pixel-wise Attention - focuses on each pixel's importance"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn


class ChannelAttention(nn.Module):
    """Channel Attention - SE-like mechanism for channel-wise importance"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_dim = max(dim // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention - focuses on spatial importance"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class SynergisticMultiAttention(nn.Module):
    """
    Synergistic Multi-Attention (SMA) as described in IEEE BIBM 2024 paper
    
    Combines channel attention, pixel attention, and spatial attention:
    1. Pixel and channel attention outputs are combined via matrix multiplication
    2. Result is further processed by spatial attention
    3. All outputs are fused synergistically
    """
    def __init__(self, dim):
        super().__init__()
        self.pixel_attn = PixelAttention(dim)
        self.channel_attn = ChannelAttention(dim, reduction=16)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        
        # Projection for synergistic fusion
        self.fusion_proj = nn.Conv2d(dim, dim, 1)
        
        # Final output projection with residual
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, C, H, W]
        """
        # Parallel attention branches
        pixel_out = self.pixel_attn(x)      # Pixel-wise attention
        channel_out = self.channel_attn(x)  # Channel attention
        
        # Synergistic fusion via element-wise multiplication (matrix multiplication in paper)
        synergy = pixel_out * channel_out
        synergy = self.fusion_proj(synergy)
        
        # Spatial attention on synergistic features
        spatial_out = self.spatial_attn(synergy)
        
        # Output with residual connection
        out = self.out_proj(spatial_out) + x
        
        return out


class EnhancedMLP(nn.Module):
    """
    Enhanced MLP (E-MLP) as described in IEEE BIBM 2024 paper
    
    Incorporates depth-wise and pixel-wise convolutions for local context:
    1. Linear projection to higher dimension
    2. Reshape to 2D and apply pixel-wise + depth-wise conv
    3. Reshape back and project to original dimension
    """
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        # First linear projection
        self.fc1 = nn.Linear(dim, hidden_dim)
        
        # Depthwise + Pointwise convolutions for local context
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.pwconv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # Second linear projection
        self.fc2 = nn.Linear(hidden_dim, dim)
        
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        """
        x: [B, N, C] where N = H * W
        Returns: [B, N, C]
        """
        B, N, C = x.shape
        
        # FC1
        x = self.fc1(x)
        x = self.act(x)
        
        # Reshape for conv
        x = x.transpose(1, 2).view(B, -1, H, W)
        
        # Depthwise + Pointwise conv
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.act(x)
        
        # Reshape back
        x = x.flatten(2).transpose(1, 2)
        
        # FC2
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class SMATransformerBlock(nn.Module):
    """
    SMA Transformer Block as described in IEEE BIBM 2024 paper
    
    Computation:
    X' = SMA(LN(X)) + X
    X_out = E-MLP(LN(X')) + X'
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # SMA module
        self.sma = SynergisticMultiAttention(dim)
        
        # E-MLP module
        self.emlp = EnhancedMLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x, H, W):
        """
        x: [B, N, C] where N = H * W
        Returns: [B, N, C]
        """
        B, N, C = x.shape
        
        # SMA branch: need to reshape to 2D for SMA
        x_norm = self.norm1(x)
        x_2d = x_norm.transpose(1, 2).view(B, C, H, W)
        sma_out = self.sma(x_2d)
        sma_out = sma_out.flatten(2).transpose(1, 2)
        x = x + self.drop_path(sma_out - x_norm)  # Residual connection
        
        # E-MLP branch
        x = x + self.drop_path(self.emlp(self.norm2(x), H, W))
        
        return x


# ============== Residual Convolution Block ==============

class ResidualConvBlock(nn.Module):
    """Residual convolution block for downsampling/upsampling"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


# ============== Edge Enhancement Module ==============

class EdgeEnhancement(nn.Module):
    """Edge enhancement for final output refinement"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Sobel-like edge detection
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        
        # Initialize with edge-detecting kernel
        with torch.no_grad():
            sobel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            sobel = sobel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
            self.edge_conv.weight.data = sobel / 8.0
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        edges = self.edge_conv(x)
        enhanced = self.refine(torch.cat([x, edges], dim=1))
        out = self.out_conv(enhanced)
        return out


# ============== Decoder Stage ==============

class DecoderStage(nn.Module):
    """
    Decoder stage with upsampling, skip connection, and SMA blocks
    Following the paper: transposed conv for upsampling, then SMA transformer blocks
    """
    def __init__(self, in_channels, out_channels, num_blocks=2, mlp_ratio=4., drop_path=0.):
        super().__init__()
        
        # Upsampling via transposed convolution
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # Fusion after skip connection
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # SMA Transformer blocks
        self.blocks = nn.ModuleList([
            SMATransformerBlock(
                dim=out_channels,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path
            ) for _ in range(num_blocks)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, skip):
        """
        x: [B, C_in, H, W]
        skip: [B, C_out, 2H, 2W]
        Returns: [B, C_out, 2H, 2W]
        """
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch by interpolation
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        
        # SMA transformer blocks
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        for blk in self.blocks:
            x = blk(x, H, W)
        
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x


# ============== Main Model ==============

class SMAFormerV2(nn.Module):
    """
    SMAFormerV2: Synergistic Multi-Attention Transformer V2
    
    Based on IEEE BIBM 2024 paper with Swin Transformer backbone.
    
    Architecture (following paper Figure 1):
    1. Input projection layer (3x3 conv + ReLU)
    2. Four-stage Swin Transformer encoder (loads pretrained weights)
    3. SMA enhancement after each encoder stage
    4. Four-stage symmetric decoder with skip connections
    5. Edge enhancement final layer
    
    Key differences from original SMAFormer:
    - Uses Swin Transformer as backbone instead of ViT
    - Enables loading of Swin pretrained weights
    - Maintains SMA blocks for attention fusion
    """
    
    def __init__(
        self,
        args,
        img_size=256,
        num_classes=9,
        embed_dims=[96, 192, 384, 768],  # Swin Tiny channels
        depths=[2, 2, 6, 2],  # Swin Tiny depths
        num_heads=[3, 6, 12, 24],  # Swin Tiny heads
        window_size=7,
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.2,
        decoder_blocks=2,
        use_edge_enhancement=True,
        pretrained_path='pre_trained_weights/swin_tiny_patch4_window7_224.pth'
    ):
        super().__init__()
        self.args = args
        self.use_edge_enhancement = use_edge_enhancement
        
        # Determine number of classes from dataset
        if hasattr(args, 'dataset'):
            if args.dataset == 'LiTS2017':
                num_classes = 3
            elif args.dataset == 'Synapse':
                num_classes = 9
            elif args.dataset == 'ACDC':
                num_classes = 3
        self.num_classes = num_classes
        
        # ========== Input Projection (as per paper) ==========
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # ========== Swin Transformer Encoder (Pretrained) ==========
        # Adjust image size for window compatibility
        base = window_size * 4  # patch_size=4
        self.init_img_size = int(math.ceil(img_size / base) * base)
        self.orig_img_size = img_size
        self.pad_h = self.init_img_size - self.orig_img_size
        self.pad_w = self.init_img_size - self.orig_img_size
        
        self.encoder = SwinTransformer(
            img_size=self.init_img_size,
            patch_size=4,
            in_chans=3,
            num_classes=0,  # No classification head
            embed_dim=embed_dims[0],
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=0.,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=True
        )
        
        # ========== SMA Enhancement after each encoder stage ==========
        self.sma_stages = nn.ModuleList([
            SynergisticMultiAttention(dim) for dim in embed_dims
        ])
        
        # ========== Symmetric Decoder ==========
        # Decoder stage 4: 768 -> 384
        self.decoder4 = DecoderStage(embed_dims[3], embed_dims[2], num_blocks=decoder_blocks)
        # Decoder stage 3: 384 -> 192
        self.decoder3 = DecoderStage(embed_dims[2], embed_dims[1], num_blocks=decoder_blocks)
        # Decoder stage 2: 192 -> 96
        self.decoder2 = DecoderStage(embed_dims[1], embed_dims[0], num_blocks=decoder_blocks)
        
        # Final upsampling to original resolution
        # After patch_embed: 1/4 resolution. After decoder2: back to 1/4. Need 4x upsample.
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[0], embed_dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # ========== Output Layer ==========
        if use_edge_enhancement:
            self.output = EdgeEnhancement(embed_dims[0], num_classes)
        else:
            self.output = nn.Conv2d(embed_dims[0], num_classes, 1)
        
        # ========== Load Pretrained Weights ==========
        self.pretrained_path = pretrained_path
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        else:
            print(f"⚠ Pretrained weights not found at {pretrained_path}")
            print("  Model will be trained from scratch")
    
    def _load_pretrained_weights(self, pretrained_path):
        """Load and report pretrained weight statistics"""
        print("\n" + "=" * 70)
        print("SMAFormerV2 预训练权重加载报告")
        print("=" * 70)
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        
        # Get model state dict
        model_dict = self.encoder.state_dict()
        
        # Statistics
        pretrained_keys = set(pretrained_dict.keys())
        model_keys = set(model_dict.keys())
        
        print(f"\n📦 预训练权重文件: {pretrained_path}")
        print(f"   预训练权重总层数: {len(pretrained_keys)}")
        print(f"   模型Encoder总层数: {len(model_keys)}")
        
        # Filter matching weights
        matched_dict = {}
        matched_params = 0
        total_pretrained_params = 0
        
        for k, v in pretrained_dict.items():
            total_pretrained_params += v.numel()
            
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    matched_dict[k] = v
                    matched_params += v.numel()
                elif 'relative_position_bias_table' in k:
                    # Handle position bias interpolation
                    resized = self._resize_position_bias(v, model_dict[k].shape)
                    if resized is not None:
                        matched_dict[k] = resized
                        matched_params += resized.numel()
        
        # Load matched weights
        model_dict.update(matched_dict)
        msg = self.encoder.load_state_dict(model_dict, strict=False)
        
        # Calculate statistics
        encoder_total_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_total_params = sum(p.numel() for p in self.decoder4.parameters()) + \
                              sum(p.numel() for p in self.decoder3.parameters()) + \
                              sum(p.numel() for p in self.decoder2.parameters())
        sma_total_params = sum(p.numel() for p in self.sma_stages.parameters())
        
        encoder_matched_rate = matched_params / encoder_total_params * 100 if encoder_total_params > 0 else 0
        pretrained_used_rate = matched_params / total_pretrained_params * 100 if total_pretrained_params > 0 else 0
        
        print(f"\n📊 权重加载统计:")
        print(f"   ├─ 成功匹配的层数: {len(matched_dict)} / {len(model_keys)}")
        print(f"   ├─ 成功加载的参数量: {matched_params:,}")
        print(f"   ├─ Encoder总参数量: {encoder_total_params:,}")
        print(f"   └─ Encoder预训练权重覆盖率: {encoder_matched_rate:.2f}%")
        
        print(f"\n📈 模型各部分参数统计:")
        print(f"   ├─ Encoder (Swin): {encoder_total_params:,} ({encoder_total_params/1e6:.2f}M)")
        print(f"   ├─ SMA Stages: {sma_total_params:,} ({sma_total_params/1e6:.2f}M)")
        print(f"   ├─ Decoder: {decoder_total_params:,} ({decoder_total_params/1e6:.2f}M)")
        print(f"   └─ 总参数量: {sum(p.numel() for p in self.parameters()):,} ({sum(p.numel() for p in self.parameters())/1e6:.2f}M)")
        
        print(f"\n✅ Encoder权重加载完成!")
        print(f"   - 预训练权重利用率: {pretrained_used_rate:.2f}%")
        print(f"   - Decoder权重: 随机初始化 (需要训练)")
        print("=" * 70 + "\n")
        
        return len(matched_dict), len(model_keys)
    
    def _resize_position_bias(self, tensor, target_shape):
        """Resize relative position bias table"""
        nH = tensor.shape[1]
        src_len = tensor.shape[0]
        tgt_len = target_shape[0]
        src_size = int(math.sqrt(src_len))
        tgt_size = int(math.sqrt(tgt_len))
        
        if src_size * src_size != src_len or tgt_size * tgt_size != tgt_len:
            return None
        if src_size == tgt_size:
            return tensor
        
        bias = tensor.transpose(0, 1).view(1, nH, src_size, src_size)
        bias = F.interpolate(bias, size=(tgt_size, tgt_size), mode='bicubic', align_corners=False)
        bias = bias.view(nH, tgt_size * tgt_size).transpose(0, 1)
        return bias
    
    def forward_encoder(self, x):
        """
        Forward through Swin encoder with SMA enhancement
        Returns multi-scale features
        """
        B, C, H, W = x.shape
        
        # Pad input if needed
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        
        # Patch embedding
        x = self.encoder.patch_embed(x)
        
        # Handle different timm versions output format
        if x.dim() == 4:
            B, h, w, C = x.shape
        else:
            B, L, C = x.shape
            h = w = int(L ** 0.5)
            x = x.view(B, h, w, C)
        
        # Position drop
        if hasattr(self.encoder, 'pos_drop') and self.encoder.pos_drop is not None:
            x = self.encoder.pos_drop(x)
        
        # Get features from each stage
        features = []
        for idx, layer in enumerate(self.encoder.layers):
            x = layer(x)
            B, H, W, C = x.shape
            
            # Convert to BCHW for SMA
            feat = x.permute(0, 3, 1, 2).contiguous()
            
            # Apply SMA enhancement
            feat = self.sma_stages[idx](feat)
            
            features.append(feat)
            
            # Continue with enhanced features
            x = feat.permute(0, 2, 3, 1).contiguous()
        
        return features
    
    def forward(self, x):
        """
        Forward pass
        x: [B, 3, H, W]
        Returns: [B, num_classes, H, W]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Encoder with SMA enhancement
        features = self.forward_encoder(x)
        # features: [f1, f2, f3, f4] with channels [96, 192, 384, 768]
        # resolutions: [1/4, 1/8, 1/16, 1/32] relative to patch_embed output
        
        f1, f2, f3, f4 = features
        
        # Decoder with skip connections
        d3 = self.decoder4(f4, f3)  # 768 -> 384
        d2 = self.decoder3(d3, f2)  # 384 -> 192
        d1 = self.decoder2(d2, f1)  # 192 -> 96
        
        # Final upsampling to original resolution
        out = self.final_upsample(d1)
        
        # Crop to original size if padded
        if self.pad_h > 0 or self.pad_w > 0:
            out = out[:, :, :self.orig_img_size, :self.orig_img_size]
        
        # Output layer
        out = self.output(out)
        
        return out


# ============== Testing ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Synapse', help='dataset name')
    args = parser.parse_args()
    
    # Get pretrained weights path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    pretrained_path = os.path.join(project_root, 'pre_trained_weights', 'swin_tiny_patch4_window7_224.pth')
    
    # Create model
    model = SMAFormerV2(
        args,
        img_size=256,
        num_classes=9,
        use_edge_enhancement=True,
        pretrained_path=pretrained_path
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate FLOPs
    if get_model_complexity_info is not None:
        try:
            flops, params = get_model_complexity_info(
                model, input_res=(3, 256, 256),
                as_strings=True, print_per_layer_stat=False
            )
            print(f"FLOPs: {flops}")
            print(f"Params: {params}")
        except Exception as e:
            print(f"Could not compute FLOPs: {e}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of classes: {model.num_classes}")
    
    print("\n✅ SMAFormerV2 test passed!")
