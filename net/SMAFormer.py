import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 添加额外的卷积层来调整residual张量的维度
        self.adjust_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.adjust_residual(x)  # 调整residual张量的维度
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(in_channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return self.norm(x * attention)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return self.norm(x * avg_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(1)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return self.norm(x * attention)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.pixel_attention = PixelAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        pa_out = self.pixel_attention(x)
        ca_out = self.channel_attention(pa_out)
        sa_out = self.spatial_attention(ca_out)
        return self.norm(sa_out + x)

class Modulator(nn.Module):
    def __init__(self, feature_size, num_embeddings=128):
        super(Modulator, self).__init__()
        # Modulator部分
        self.embeddings = nn.Embedding(num_embeddings, feature_size)
        self.norm = nn.LayerNorm(feature_size)
        # FeatureFusionModulator部分
        self.conv_fusion = nn.Conv2d(feature_size * 3, feature_size, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, feature_size, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Modulator部分
        batch_size, seq_len, feature_size = x.size()
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        pos_embedding = self.embeddings(positions)
        x = self.norm(x + pos_embedding)

        return x

    def fuse_features(self, ca_out, sa_out, pa_out):
        # FeatureFusionModulator部分
        fusion = torch.cat([ca_out, sa_out, pa_out], dim=1)  # Concatenate along the channel dimension
        fusion = self.conv_fusion(fusion)
        fusion = self.relu(fusion + self.bias)
        return fusion

'''
Synergistic Multi-Attention
'''
class SMA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(SMA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.attention_block = AttentionBlock(feature_size)
        self.combined_modulator = Modulator(feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(attention) + query

        # 通过CombinedModulator进行多尺度特征融合
        x = self.combined_modulator(x)

        # 将输出转换为适合AttentionBlock的输入格式
        batch_size, seq_len, feature_size = x.shape
        x = x.permute(0, 2, 1).view(batch_size, feature_size, int(seq_len**0.5), int(seq_len**0.5))

        # 通过AttentionBlock
        ca_out = self.attention_block.channel_attention(x)
        sa_out = self.attention_block.spatial_attention(x)
        pa_out = self.attention_block.pixel_attention(x)

        # 通过CombinedModulator进行特征融合
        fusion_out = self.combined_modulator.fuse_features(ca_out, sa_out, pa_out)

        # 将输出转换回 (batch_size, seq_len, feature_size) 格式
        x = fusion_out.view(batch_size, feature_size, -1).permute(0, 2, 1)

        return x

class E_MLP(nn.Module):
    def __init__(self, feature_size, forward_expansion, dropout):
        super(E_MLP, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, forward_expansion * feature_size),
            nn.GELU(),
            nn.Linear(forward_expansion * feature_size, feature_size)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        forward = self.feed_forward(x)
        out = self.dropout(forward) + x
        return out

class SMAFormerBlock(nn.Module):
    def __init__(self, feature_size, heads, dropout, forward_expansion):
        super(SMAFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.synergistic_multi_attention = SMA(feature_size, heads, dropout)
        self.e_mlp = E_MLP(feature_size, forward_expansion, dropout)

    def forward(self, value, key, query):
        query = self.norm1(query)
        x = self.synergistic_multi_attention(value, key, query)
        x = self.norm2(x)
        out = self.e_mlp(x)
        return out

class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNetPlusPlus, self).__init__()
        self.encoder1 = ResBlock(in_channels, 64)
        self.encoder2 = ResBlock(64, 128)
        self.encoder3 = ResBlock(128, 256)
        self.encoder4 = ResBlock(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder1 = ResBlock(512 + 256, 256)
        self.decoder2 = ResBlock(256 + 128, 128)
        self.decoder3 = ResBlock(128 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        d1 = self.upconv1(e4)
        d1 = torch.cat((d1, e3), dim=1)
        d1 = self.decoder1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat((d3, e1), dim=1)
        d3 = self.decoder3(d3)

        output = self.final_conv(d3)
        return output

class SMAFormer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SMAFormer, self).__init__()
        self.backbone = ResUNetPlusPlus(in_channels, num_classes)
        self.sma_block1 = SMAFormerBlock(64)
        self.sma_block2 = SMAFormerBlock(128)
        self.sma_block3 = SMAFormerBlock(256)
        self.sma_block4 = SMAFormerBlock(512)

        self.feature_fusion_mod1 = FeatureFusionModulator(64)
        self.feature_fusion_mod2 = FeatureFusionModulator(128)
        self.feature_fusion_mod3 = FeatureFusionModulator(256)
        self.feature_fusion_mod4 = FeatureFusionModulator(512)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with SMA blocks
        e1 = self.backbone.encoder1(x)
        e1 = self.sma_block1(e1)
        e2 = self.backbone.encoder2(self.backbone.pool(e1))
        e2 = self.sma_block2(e2)
        e3 = self.backbone.encoder3(self.backbone.pool(e2))
        e3 = self.sma_block3(e3)
        e4 = self.backbone.encoder4(self.backbone.pool(e3))
        e4 = self.sma_block4(e4)

        # Decoder with fusion modulator
        d1 = self.backbone.upconv1(e4)
        d1 = torch.cat((d1, e3), dim=1)
        d1 = self.feature_fusion_mod3(d1)
        d1 = self.backbone.decoder1(d1)

        d2 = self.backbone.upconv2(d1)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.feature_fusion_mod2(d2)
        d2 = self.backbone.decoder2(d2)

        d3 = self.backbone.upconv3(d2)
        d3 = torch.cat((d3, e1), dim=1)
        d3 = self.feature_fusion_mod1(d3)
        d3 = self.backbone.decoder3(d3)

        output = self.final_conv(d3)
        return output

# Example usage
if __name__ == "__main__":
    model = SMAFormer(in_channels=3, num_classes=2)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)