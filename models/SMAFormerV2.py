# ------------------------------------------------------------
# SMAFormerV2 V2.3: Synergistic Multi-Attention Transformer V2
# Based on IEEE BIBM 2024 SMAFormer with Swin Transformer backbone
# 
# ========== Version 2.3 (2025-12-22) ==========
# 核心改进：Decoder镜像Encoder结构，双向加载Swin预训练权重
# 
# V2.3 vs V2.1 的关键差异：
# 1. Decoder采用镜像的Swin Transformer Block结构
# 2. Decoder可以加载Swin预训练权重（反向映射）
# 3. Encoder和Decoder的预训练权重加载率都会被打印
# 4. 去除复杂的额外模块，保持简洁架构
# 5. 预期通过最大化预训练权重利用率来提升性能
#
# 设计原理：
# - Encoder: Swin Tiny stage0→stage3 (下采样路径)
# - Decoder: 镜像 stage3→stage0 (上采样路径)
# - Encoder的pretrained weights正向加载
# - Decoder的pretrained weights反向映射加载
#   (encoder.layers.3 → decoder.stage0, encoder.layers.2 → decoder.stage1, ...)
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import os

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


# ============== Window Attention (from Swin Transformer) ==============

def window_partition(x, window_size):
    """
    Partition feature map into windows.
    x: (B, H, W, C)
    window_size: int
    Returns: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    windows: (num_windows*B, window_size, window_size, C)
    Returns: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA) from Swin Transformer"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinMlp(nn.Module):
    """MLP from Swin Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block - Compatible with pretrained weights"""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwinMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Calculate attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} != {H * W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer (Downsampling) - Compatible with pretrained weights"""
    
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """Patch Expanding Layer (Upsampling) - Mirror of PatchMerging for weight loading"""
    
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # Linear to expand channels: dim -> 2*dim, then reshape to 2x spatial, dim/2 channels
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        x = x.view(B, H, W, -1)
        
        # Pixel shuffle style: (B, H, W, 4*C/2) -> (B, 2H, 2W, C/2)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)
        
        x = self.norm(x)
        return x


class BasicEncoderLayer(nn.Module):
    """
    Basic Encoder Layer - One stage of Swin Transformer
    Structure matches Swin Transformer for weight loading compatibility
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Downsample layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        x_before_down = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_before_down


class BasicDecoderLayer(nn.Module):
    """
    Basic Decoder Layer - Mirror of BasicEncoderLayer for weight loading
    Structure mirrors Swin Transformer encoder layer
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, concat_dim=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Upsample layer (applied first)
        if upsample is not None:
            self.upsample = upsample(
                (input_resolution[0] // 2, input_resolution[1] // 2),  # Input resolution before upsample
                dim=dim * 2,  # Input dim before upsample
                norm_layer=norm_layer)
        else:
            self.upsample = None
        
        # Skip connection fusion
        self.skip_fusion = nn.Linear(dim * 2, dim)

        # Build blocks (mirror structure of encoder)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, skip=None):
        # Upsample first
        if self.upsample is not None:
            x = self.upsample(x)
        
        # Fuse with skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=-1)
            x = self.skip_fusion(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding - Compatible with Swin pretrained weights"""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class FinalExpand(nn.Module):
    """Final 4x Upsample to original resolution"""
    
    def __init__(self, dim, num_classes):
        super().__init__()
        self.expand1 = nn.Linear(dim, dim * 4, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.expand2 = nn.Linear(dim, dim * 4, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, num_classes)

    def forward(self, x, H, W):
        B, L, C = x.shape
        
        # First 2x upsample
        x = self.expand1(x)
        x = x.view(B, H, W, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C)
        x = self.norm1(x)
        
        # Second 2x upsample
        x = x.view(B, -1, C)
        x = self.expand2(x)
        x = x.view(B, H * 2, W * 2, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 4, W * 4, C)
        x = self.norm2(x)
        
        # Output projection
        x = self.output(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        
        return x


# ============== SMA Components (Lightweight from V2.1) ==============

class SynergisticMultiAttention(nn.Module):
    """
    Synergistic Multi-Attention (SMA) as described in IEEE BIBM 2024 paper
    Lightweight version - minimal additional parameters
    """
    def __init__(self, dim):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(dim // 16, 8)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(dim, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, dim, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        # Output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_attn = self.sigmoid(avg_out + max_out)
        x = x * channel_attn
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid(self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        x = x * spatial_attn
        
        return x


# ============== Main Model ==============

class SMAFormerV2(nn.Module):
    """
    SMAFormerV2 V2.3: Symmetric Swin Encoder-Decoder with Pretrained Weights
    
    核心架构：
    1. Encoder: 标准Swin Transformer (加载ImageNet预训练权重)
    2. Decoder: 镜像Swin结构 (反向加载预训练权重)
    3. Skip Connections: 对应层级连接
    4. SMA: 轻量级注意力增强
    
    预训练权重映射：
    - encoder.layers.0 → 正向加载 layers.0
    - encoder.layers.1 → 正向加载 layers.1
    - encoder.layers.2 → 正向加载 layers.2
    - encoder.layers.3 → 正向加载 layers.3
    - decoder.layers.0 ← 反向加载 layers.2 (对应resolution)
    - decoder.layers.1 ← 反向加载 layers.1
    - decoder.layers.2 ← 反向加载 layers.0
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
        norm_layer=nn.LayerNorm,
        use_sma=True,
        pretrained_path='pre_trained_weights/swin_tiny_patch4_window7_224.pth',
        load_pretrained=True
    ):
        super().__init__()
        self.args = args
        self.use_sma = use_sma
        self.load_pretrained = load_pretrained
        
        # Determine number of classes from dataset
        if hasattr(args, 'dataset'):
            if args.dataset == 'LiTS2017':
                num_classes = 3
            elif args.dataset == 'Synapse':
                num_classes = 9
            elif args.dataset == 'ACDC':
                num_classes = 3
        self.num_classes = num_classes
        
        # Image size adjustment for window compatibility
        patch_size = 4
        # 对于256输入，patch后是64x64
        # stage分辨率: 64->32->16->8
        # window_size=7时，8不能被7整除，需要padding
        # 最简单的方法：使用224作为init size（Swin默认），56->28->14->7都能被7整除
        # 或者padding到能整除的大小
        
        # 方案1：使用能被window完美整除的尺寸
        # 对于window_size=7，patch_size=4：需要 H/4/8 = H/32 能被7整除
        # H = 224 -> 224/32 = 7 ✓
        # H = 256 -> 256/32 = 8，8不能被7整除 ✗
        
        # 使用224并在输入时resize
        self.init_img_size = 224
        self.orig_img_size = img_size
        self.pad_h = self.init_img_size - self.orig_img_size
        self.pad_w = self.init_img_size - self.orig_img_size
        
        patches_resolution = [self.init_img_size // patch_size, self.init_img_size // patch_size]
        self.patches_resolution = patches_resolution
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # ========== Patch Embedding ==========
        self.patch_embed = PatchEmbed(
            img_size=self.init_img_size, patch_size=patch_size, in_chans=3,
            embed_dim=embed_dims[0], norm_layer=norm_layer)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # ========== Encoder Layers ==========
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(4):
            layer = BasicEncoderLayer(
                dim=embed_dims[i_layer],
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < 3) else None)
            self.encoder_layers.append(layer)
        
        # Encoder final norm
        self.encoder_norm = norm_layer(embed_dims[3])
        
        # ========== Decoder Layers (Mirror Structure) ==========
        # Decoder processes in reverse: 768 -> 384 -> 192 -> 96
        self.decoder_layers = nn.ModuleList()
        decoder_depths = [depths[2], depths[1], depths[0]]  # [6, 2, 2] - use encoder depths for matching
        decoder_dims = [384, 192, 96]  # Output dims
        decoder_heads = [num_heads[2], num_heads[1], num_heads[0]]  # [12, 6, 3]
        decoder_input_dims = [768, 384, 192]  # Input dims (before upsample)
        
        # Decoder resolutions (for 224x224 input, patches_resolution=56)
        decoder_resolutions = [
            (patches_resolution[0] // 4, patches_resolution[1] // 4),   # 14x14 for dim=384
            (patches_resolution[0] // 2, patches_resolution[1] // 2),   # 28x28 for dim=192
            (patches_resolution[0], patches_resolution[1]),             # 56x56 for dim=96
        ]
        
        # Reverse drop path rates for decoder
        dpr_decoder = dpr[::-1]
        
        for i_layer in range(3):
            layer = BasicDecoderLayer(
                dim=decoder_dims[i_layer],
                input_resolution=decoder_resolutions[i_layer],
                depth=decoder_depths[i_layer],
                num_heads=decoder_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=0.,
                drop_path=dpr_decoder[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpanding)
            self.decoder_layers.append(layer)
        
        # Decoder norms
        self.decoder_norms = nn.ModuleList([norm_layer(dim) for dim in decoder_dims])
        
        # ========== SMA Enhancement ==========
        if use_sma:
            self.sma_encoder = nn.ModuleList([SynergisticMultiAttention(dim) for dim in embed_dims])
            self.sma_decoder = nn.ModuleList([SynergisticMultiAttention(dim) for dim in decoder_dims])
        
        # ========== Final Upsample and Output ==========
        self.final_expand = FinalExpand(embed_dims[0], num_classes)
        
        # ========== Initialize and Load Pretrained Weights ==========
        self.apply(self._init_weights)
        
        self.pretrained_path = pretrained_path
        self.encoder_load_rate = 0.0
        self.decoder_load_rate = 0.0
        
        if load_pretrained and pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        elif load_pretrained:
            print(f"⚠ Pretrained weights not found at {pretrained_path}")
            print("  Model will be trained from scratch")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights for both encoder and decoder"""
        print("\n" + "=" * 80)
        print("SMAFormerV2 V2.3 - 双向预训练权重加载报告")
        print("=" * 80)
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        
        # ========== Load Encoder Weights ==========
        print("\n📦 [1/2] 加载 Encoder 预训练权重...")
        encoder_matched, encoder_total = self._load_encoder_weights(pretrained_dict)
        
        # ========== Load Decoder Weights (Reverse Mapping) ==========
        print("\n📦 [2/2] 加载 Decoder 预训练权重 (反向映射)...")
        decoder_matched, decoder_total = self._load_decoder_weights(pretrained_dict)
        
        # ========== Summary ==========
        print("\n" + "=" * 80)
        print("📊 预训练权重加载总结")
        print("=" * 80)
        
        self.encoder_load_rate = encoder_matched / encoder_total * 100 if encoder_total > 0 else 0
        self.decoder_load_rate = decoder_matched / decoder_total * 100 if decoder_total > 0 else 0
        
        total_params = sum(p.numel() for p in self.parameters())
        total_loaded = encoder_matched + decoder_matched
        
        print(f"\n   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Encoder 加载率: {self.encoder_load_rate:6.2f}%  ({encoder_matched:,}/{encoder_total:,} 参数)")
        print(f"   │  Decoder 加载率: {self.decoder_load_rate:6.2f}%  ({decoder_matched:,}/{decoder_total:,} 参数)")
        print(f"   │  ─────────────────────────────────────────────────── ")
        print(f"   │  模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   │  预训练覆盖率: {total_loaded/total_params*100:.2f}%")
        print(f"   └─────────────────────────────────────────────────────┘")
        print("=" * 80 + "\n")

    def _load_encoder_weights(self, pretrained_dict):
        """Load encoder weights from Swin pretrained model"""
        model_dict = {}
        matched_params = 0
        total_params = 0
        
        # Patch embed
        for name, param in self.patch_embed.named_parameters():
            total_params += param.numel()
            pretrained_key = f"patch_embed.{name}"
            if pretrained_key in pretrained_dict and pretrained_dict[pretrained_key].shape == param.shape:
                model_dict[f"patch_embed.{name}"] = pretrained_dict[pretrained_key]
                matched_params += param.numel()
        
        # Encoder layers
        for layer_idx in range(4):
            for name, param in self.encoder_layers[layer_idx].named_parameters():
                total_params += param.numel()
                # Map to Swin pretrained key
                pretrained_key = f"layers.{layer_idx}.{name}"
                if pretrained_key in pretrained_dict:
                    if pretrained_dict[pretrained_key].shape == param.shape:
                        model_dict[f"encoder_layers.{layer_idx}.{name}"] = pretrained_dict[pretrained_key]
                        matched_params += param.numel()
        
        # Encoder norm
        for name, param in self.encoder_norm.named_parameters():
            total_params += param.numel()
            pretrained_key = f"norm.{name}"
            if pretrained_key in pretrained_dict and pretrained_dict[pretrained_key].shape == param.shape:
                model_dict[f"encoder_norm.{name}"] = pretrained_dict[pretrained_key]
                matched_params += param.numel()
        
        # Load matched weights
        current_dict = self.state_dict()
        current_dict.update(model_dict)
        self.load_state_dict(current_dict, strict=False)
        
        print(f"   ├─ Encoder参数总量: {total_params:,}")
        print(f"   ├─ 成功加载参数量: {matched_params:,}")
        print(f"   └─ 加载成功率: {matched_params/total_params*100:.2f}%")
        
        return matched_params, total_params

    def _load_decoder_weights(self, pretrained_dict):
        """Load decoder weights from Swin pretrained model (reverse mapping)"""
        model_dict = {}
        matched_params = 0
        total_params = 0
        
        # Mapping: decoder layer 0 <- encoder layer 2 (same depth=6)
        # decoder layer 1 <- encoder layer 1 (same depth=2)
        # decoder layer 2 <- encoder layer 0 (same depth=2)
        reverse_mapping = {0: 2, 1: 1, 2: 0}
        
        for decoder_idx in range(3):
            encoder_idx = reverse_mapping[decoder_idx]
            
            for name, param in self.decoder_layers[decoder_idx].named_parameters():
                total_params += param.numel()
                
                # Skip non-transferable layers
                if 'skip_fusion' in name or 'upsample' in name:
                    continue
                
                # Map decoder blocks to encoder blocks
                pretrained_key = f"layers.{encoder_idx}.{name}"
                if pretrained_key in pretrained_dict:
                    pretrained_param = pretrained_dict[pretrained_key]
                    if pretrained_param.shape == param.shape:
                        model_dict[f"decoder_layers.{decoder_idx}.{name}"] = pretrained_param
                        matched_params += param.numel()
        
        # Load matched weights
        current_dict = self.state_dict()
        current_dict.update(model_dict)
        self.load_state_dict(current_dict, strict=False)
        
        print(f"   ├─ Decoder参数总量: {total_params:,}")
        print(f"   ├─ 成功加载参数量: {matched_params:,}")
        print(f"   └─ 加载成功率: {matched_params/total_params*100:.2f}%")
        
        return matched_params, total_params

    def forward(self, x):
        """
        Forward pass
        x: [B, 3, H, W]
        Returns: [B, num_classes, H, W]
        """
        B, C, H_orig, W_orig = x.shape
        
        # Resize to init_img_size if needed (for window compatibility)
        if H_orig != self.init_img_size or W_orig != self.init_img_size:
            x = F.interpolate(x, size=(self.init_img_size, self.init_img_size), 
                            mode='bilinear', align_corners=False)
        
        # ========== Encoder ==========
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        encoder_features = []
        for idx, layer in enumerate(self.encoder_layers):
            x, x_before_down = layer(x)
            
            # Apply SMA if enabled
            if self.use_sma:
                B, L, C = x_before_down.shape
                h = w = int(math.sqrt(L))
                feat = x_before_down.transpose(1, 2).view(B, C, h, w)
                feat = self.sma_encoder[idx](feat)
                x_before_down = feat.flatten(2).transpose(1, 2)
            
            encoder_features.append(x_before_down)
        
        # Final encoder norm
        x = self.encoder_norm(x)
        
        # ========== Decoder ==========
        # encoder_features: [f0, f1, f2, f3] with dims [96, 192, 384, 768]
        # Skip connections: decoder_layer[0] uses f2 (dim=384)
        #                   decoder_layer[1] uses f1 (dim=192)
        #                   decoder_layer[2] uses f0 (dim=96)
        
        for idx, layer in enumerate(self.decoder_layers):
            skip_idx = 2 - idx  # 2, 1, 0
            skip = encoder_features[skip_idx]
            
            x = layer(x, skip)
            x = self.decoder_norms[idx](x)
            
            # Apply SMA to decoder output
            if self.use_sma:
                B, L, C = x.shape
                h = w = int(math.sqrt(L))
                feat = x.transpose(1, 2).view(B, C, h, w)
                feat = self.sma_decoder[idx](feat)
                x = feat.flatten(2).transpose(1, 2)
        
        # Final 4x upsample
        h = w = self.patches_resolution[0]  # 56 for 224x224 internal size
        out = self.final_expand(x, h, w)
        
        # Resize back to original size if needed
        if H_orig != self.init_img_size or W_orig != self.init_img_size:
            out = F.interpolate(out, size=(H_orig, W_orig), 
                               mode='bilinear', align_corners=False)
        
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
    
    print("=" * 80)
    print("SMAFormerV2 V2.3 测试")
    print("=" * 80)
    
    # Create model
    model = SMAFormerV2(
        args,
        img_size=256,
        num_classes=9,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_sma=True,
        pretrained_path=pretrained_path,
        load_pretrained=True
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(x)
    
    print(f"\n✅ 前向传播测试通过!")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out.shape}")
    print(f"   类别数: {model.num_classes}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 参数统计:")
    print(f"   总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    print(f"\n   Encoder加载率: {model.encoder_load_rate:.2f}%")
    print(f"   Decoder加载率: {model.decoder_load_rate:.2f}%")
    
    print("\n✅ SMAFormerV2 V2.3 测试完成!")
