import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import argparse
from dataset import dataset
import math
from calflops import calculate_flops




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ResUformer', default=None,
                        help='model name: (default: arch+timestamp)')

    args = parser.parse_args()

    return args


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
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(SpatialAttention, self).__init__()

        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.spatial_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.spatial_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.spatial_block1(x)
        x2 = self.spatial_block2(x)
        x3 = self.spatial_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.pixel_attention = PixelAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(in_channels, out_channels)
        self.norm = nn.BatchNorm2d(in_channels)

    # def forward(self, x):
    #     pa_out = self.pixel_attention(x)
    #     ca_out = self.channel_attention(pa_out)
    #     sa_out = self.spatial_attention(ca_out)
    #     return self.norm(sa_out + x)

class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class Modulator(nn.Module):
    def __init__(self, feature_size):
        super(Modulator, self).__init__()
        self.feature_size = feature_size
        self.norm = nn.LayerNorm(feature_size)
        # FeatureFusionModulator部分
        self.conv_fusion = nn.Conv2d(feature_size * 3, feature_size, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, feature_size, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Modulator部分
        batch_size, seq_len, feature_size = x.size()
        embeddings = nn.Embedding(seq_len, feature_size).to(x.device)  # 动态创建嵌入层
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        pos_embedding = embeddings(positions)
        x = self.norm(x + pos_embedding)
        return x

    def fuse_features(self, ca_out, sa_out, pa_out):
        # # FeatureFusionModulator部分
        # batch_size, seq_len, feature_size = ca_out.size()
        # height_width = int(seq_len ** 0.5)  # 假设输入是方形的
        #
        # # 将输入从 [B, N, C] 转换为 [B, C, H, W]
        # ca_out = ca_out.view(batch_size, feature_size, height_width, height_width)
        # sa_out = sa_out.view(batch_size, feature_size, height_width, height_width)
        # pa_out = pa_out.view(batch_size, feature_size, height_width, height_width)

        # Concatenate along the channel dimension
        '''[b,c,h,w]->[b,3c,h,w]'''
        fusion = torch.cat([ca_out, sa_out, pa_out], dim=1)
        '''[b,3c,h,w]->[b,c,h,w]'''
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
        self.attention_block = AttentionBlock(feature_size, feature_size)
        self.combined_modulator = Modulator(feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(attention) + query
        B,N,C = x.size()

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

class ResEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, heads, dropout, forward_expansion, num_layers):
        super(ResEncoderBlock, self).__init__()
        self.layers = nn.ModuleList([
            SMAFormerBlock(in_ch, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.in_ch = in_ch
        self.out_ch = out_ch
        # self.patch_embedding = P_Embed(in_ch=in_ch, out_ch=out_ch, patch_size=1)


    def forward(self, x):
        residual = x
        '''[B, H*W, C]'''
        for layer in self.layers:
            x = layer(x, x, x)  # 在自注意力中 key, query 和 value 都是相同的输入
        x += residual

        return x


class SMAFormer(nn.Module):
    def __init__(self, args):
        super(SMAFormer, self).__init__()
        self.args = args
        in_channels = 3
        n_classes = dataset.num_classes
        patch_size = 1
        filters = [16, 32, 64, 128, 256, 512]
        encoder_layer = 2
        decoder_layer = 2
        self.patch_size = patch_size
        self.filters = filters
        # layer 1 + embedding
        # Licensed under the Apache License 2.0 [see LICENSE for details]
        # Written by FuChen Zheng
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )
        self.patch_embedding1 = P_Embed(in_ch=filters[0], out_ch=filters[0], patch_size=1)
        self.encoder_block1 = ResEncoderBlock(in_ch=filters[0], out_ch=filters[1], heads=8, dropout=0.,
                                              forward_expansion=4, num_layers=encoder_layer)
        self.Res_DownSampling1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.patch_embedding2 = P_Embed(in_ch=filters[1], out_ch=filters[1], patch_size=1)


        self.encoder_block2 = ResEncoderBlock(in_ch=filters[1], out_ch=filters[1], heads=16, dropout=0.,
                                              forward_expansion=8, num_layers=encoder_layer)
        self.Res_DownSampling2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.patch_embedding3 = P_Embed(in_ch=filters[2], out_ch=filters[2], patch_size=1)


        self.encoder_block3 = ResEncoderBlock(in_ch=filters[2], out_ch=filters[2], heads=32, dropout=0.,
                                              forward_expansion=16, num_layers=encoder_layer)
        self.Res_DownSampling3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.patch_embedding4 = P_Embed(in_ch=filters[3], out_ch=filters[3], patch_size=1)


        self.encoder_block4 = ResEncoderBlock(in_ch=filters[3], out_ch=filters[3], heads=64, dropout=0.,
                                              forward_expansion=32, num_layers=encoder_layer)
        self.Res_DownSampling4 = ResidualConv(filters[3], filters[4], 2, 1)
        self.patch_embedding5 = P_Embed(in_ch=filters[4], out_ch=filters[4], patch_size=1)


        self.encoder_block5 = ResEncoderBlock(in_ch=filters[4], out_ch=filters[4], heads=128, dropout=0.,
                                              forward_expansion=64, num_layers=encoder_layer)
        self.Res_DownSampling5 = ResidualConv(filters[4], filters[5], 2, 1)
        # self.patch_embedding6 = P_Embed(in_ch=filters[5], out_ch=filters[5], patch_size=1)


        self.UpSampling = Upsample_(2)

        self.Res_UpSampling1 = ResidualConv(filters[5]+filters[4], filters[4], 1, 1)
        self.patch_embedding6 = P_Embed(in_ch=filters[4], out_ch=filters[4], patch_size=1)
        self.decoder_block1 = ResEncoderBlock(in_ch=filters[4], out_ch=filters[4], heads=64, dropout=0.,
                                              forward_expansion=32, num_layers=encoder_layer)


        self.Res_UpSampling2 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)
        self.patch_embedding7 = P_Embed(in_ch=filters[3], out_ch=filters[3], patch_size=1)
        self.decoder_block2 = ResEncoderBlock(in_ch=filters[3], out_ch=filters[3], heads=32, dropout=0.,
                                              forward_expansion=16, num_layers=encoder_layer)


        self.Res_UpSampling3 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
        self.patch_embedding8 = P_Embed(in_ch=filters[2], out_ch=filters[2], patch_size=1)
        self.decoder_block3 = ResEncoderBlock(in_ch=filters[2], out_ch=filters[2], heads=16, dropout=0.,
                                              forward_expansion=8, num_layers=encoder_layer)


        self.Res_UpSampling4 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
        self.patch_embedding9 = P_Embed(in_ch=filters[1], out_ch=filters[1], patch_size=1)
        self.decoder_block4 = ResEncoderBlock(in_ch=filters[1], out_ch=filters[1], heads=8, dropout=0.,
                                              forward_expansion=4, num_layers=encoder_layer)


        self.Res_UpSampling5 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        self.patch_embedding10 = P_Embed(in_ch=filters[0], out_ch=filters[0], patch_size=1)
        self.decoder_block5 = ResEncoderBlock(in_ch=filters[0], out_ch=filters[0], heads=8, dropout=0.,
                                              forward_expansion=4, num_layers=encoder_layer)
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1))


    def forward(self, x):
        ''''''
        '''Input Projection layer: #[16,3,512,512]->[16,16,512,512]'''
        Pj1 = self.input_layer(x) + self.input_skip(x)
        '''[B,C,H,W]->[B,H*W,C]:[16,16,512,512]->[16,512x2,16]'''
        Pj2 = self.patch_embedding1(Pj1)

        '''Encoder with SMA blocks'''
        '''[16,512x2,16]->[16,512x2,16]'''
        e1 = self.encoder_block1(Pj2)
        '''[B, N, C] -> [B, C, H, W]:[16,512x2,16]->[16,16,512,512]'''
        b, num_patch, c = e1.size()
        e1 = e1.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))
        '''[B, C, H, W] -> [B, 2C, H//2, W//2]:[16,16,512,512]->[16,32,256,256]'''
        c1 = self.Res_DownSampling1(e1)
        '''[16,32,256,256]->[16,256*256,32]'''
        e1 = self.patch_embedding2(c1)

        '''[16,256*256,32]->[16,256*256,32]'''
        e2 = self.encoder_block2(e1)
        '''[B, N, C] -> [B, C, H, W]:[16,256*256,32]->[16,32,256,256]'''
        b, num_patch, c = e2.size()
        e2 = e2.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))
        '''[B, C, H, W] -> [B, 2C, H//2, W//2]:[16,32,256,256]->[16,64,128,128]'''
        c2 = self.Res_DownSampling2(e2)
        '''[16,64,128,128]->[16,128*128,64]'''
        e2 = self.patch_embedding3(c2)

        '''[16,128*128,64]->[16,128*128,64]'''
        e3 = self.encoder_block3(e2)
        '''[B, N, C] -> [B, C, H, W]:[16,128*128,64]->[16,64,128,128]'''
        b, num_patch, c = e3.size()
        e3 = e3.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))
        '''[B, C, H, W] -> [B, 2C, H//2, W//2]:[16,64,128,128]->[16,128,64,64]'''
        c3 = self.Res_DownSampling3(e3)
        '''[16,128,64,64]->[16,64*64,128]'''
        e3 = self.patch_embedding4(c3)

        '''[16,64*64,128]->[16,64*64,128]'''
        e4 = self.encoder_block4(e3)
        '''[B, N, C] -> [B, C, H, W]:[16,64*64,128]->[16,128,64,64]'''
        b, num_patch, c = e4.size()
        e4 = e4.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))
        '''[B, C, H, W] -> [B, 2C, H//2, W//2]:[16,128,64,64]->[16,256,32,32]'''
        c4 = self.Res_DownSampling4(e4)
        '''[16,256,32,32]->[16,32*32,256]'''
        e4 = self.patch_embedding5(c4)

        '''[16,32*32,256]->[16,32*32,256]'''
        e5 = self.encoder_block5(e4)
        '''[B, N, C] -> [B, C, H, W]:[16,32*32,256]->[16,256,32,32]'''
        b, num_patch, c = e5.size()
        e5 = e5.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))
        '''[B, C, H, W] -> [B, 2C, H//2, W//2]:[16,256,32,32]->[16,512,16,16]'''
        e5 = self.Res_DownSampling5(e5)
        # '''[16,512,16,16]->[16,16*16,512]'''
        # e5 = self.patch_embedding6(e5)

        '''Decoder with fusion modulator'''
        '''[16,512,16,16]->[16,512,32,32]'''
        d1 = self.UpSampling(e5)
        '''[16,512,32,32]+[16,256,32,32]->[16,512+256,32,32]'''
        d1 = torch.cat((d1, c4), dim=1)
        '''[16,512+256,32,32]->[16,256,32,32]'''
        d1 = self.Res_UpSampling1(d1)
        '''[16,256, 32, 32]->[16,32*32,256]'''
        d1 = self.patch_embedding6(d1)
        '''[16,32*32,256]->[16,32*32,256]'''
        d1 = self.decoder_block1(d1)
        '''[B, N, C] -> [B, C, H, W]:[16,32*32,256]->[16,256,32,32]'''
        b, num_patch, c = d1.size()
        d1 = d1.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))

        '''[16,256,32,32]->[16,256,64,64]'''
        d2 = self.UpSampling(d1)
        '''[16,256,64,64]+[16,128,64,64]->[16,256+128,64,64]'''
        d2 = torch.cat((d2, c3), dim=1)
        '''[16,256+128,64,64]->[16,128,64,64]'''
        d2 = self.Res_UpSampling2(d2)
        '''[16,128,64,64]->[16,64*64,128]'''
        d2 = self.patch_embedding7(d2)
        '''[16,64*64,128]->[16,64*64,128]'''
        d2 = self.decoder_block2(d2)
        '''[B, N, C] -> [B, C, H, W]:[16,64*64,128]->[16,128,64,64]'''
        b, num_patch, c = d2.size()
        d2 = d2.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))

        '''[16,128,64,64]->[16,256,128,128]'''
        d3 = self.UpSampling(d2)
        '''[16,256,128,128]+[16,64,128,128]->[16,256+64,128,128]'''
        d3 = torch.cat((d3, c2), dim=1)
        '''[16,256+64,128,128]->[16,64,128,128]'''
        d3 = self.Res_UpSampling3(d3)
        '''[16,64,128,128]->[16,128*128,64]'''
        d3 = self.patch_embedding8(d3)
        '''[16,128*128,64]->[16,128*128,64]'''
        d3 = self.decoder_block3(d3)
        '''[B,N,C]->[B,C,H,W]:[16,128*128,64]->[16,64,128,128]'''
        b, num_patch, c = d3.size()
        d3 = d3.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))

        '''[16,64,128,128]->[16,64,256,256]'''
        d4 = self.UpSampling(d3)
        '''[16,64,256,256]+[16,32,256,256]->[16,64+32,256,256]'''
        d4 = torch.cat((d4, c1), dim=1)
        '''[16,64+32,256,256]->[16,32,256,256]'''
        d4 = self.Res_UpSampling4(d4)
        '''[16,32,256,256]->[16,256*256,32]'''
        d4 = self.patch_embedding9(d4)
        '''[16,256*256,32]->[16,256*256,32]'''
        d4 = self.decoder_block4(d4)
        '''[B,N,C]->[B,C,H,W]:[16,256*256,32]->[16,32,256,256]'''
        b, num_patch, c = d4.size()
        d4 = d4.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))

        '''[16,32,256,256]->[16,32,512,512]'''
        d5 = self.UpSampling(d4)
        '''[16,32,512,512]+[16,16,512,512]->[16,32+16,512,512]'''
        d5 = torch.cat((d5, Pj1), dim=1)
        '''[16,32+16,512,512]->[16,16,512,512]'''
        d5 = self.Res_UpSampling5(d5)
        '''[16,16,512,512]->[16,512*512,16]'''
        d5 = self.patch_embedding10(d5)
        '''[16,512*512,16]->[16,512*512,16]'''
        d5 = self.decoder_block5(d5)
        '''[B,N,C]->[B,C,H,W]:[16,512*512,16]->[16,16,512,512]'''
        b, num_patch, c = d5.size()
        d5 = d5.view(b, c, int(math.sqrt(num_patch)), int(math.sqrt(num_patch)))
        '''[16,16,512,512]->[16,3,512,512]'''
        output = self.output_layer(d5)

        return output

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class P_Embed(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size, padding=patch_size // 2)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        # B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# Example usage
# # 检查模型是否能够创建并输出期望的维度
# args = parse_args()
# model = SMAFormer(args)
# #calculate Flops
# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=(16,3,512,512),
#                                       output_as_string=True,
#                                       print_results=True,
#                                       print_detailed=True,
#                                       output_unit='M'
#                                       )
# print('%s -- FLOPs:%s  -- MACs:%s   -- Params:%s \n'%(args.model_name, flops, macs, params))
# x = torch.randn(16, 3, 512, 512)  # 假设输入是256x256的RGB图像
# with torch.no_grad():  # 在不计算梯度的情况下执行前向传播
#     out = model(x)
# print('Final Output:')
# print(out.shape)  # 输出预期是与分类头的输出通道数匹配的特征图