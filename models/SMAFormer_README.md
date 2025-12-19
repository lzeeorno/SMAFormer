# SMAFormer: Spatial-Mamba Attention Transformer for Medical Image Segmentation

**IEEE BIBM 2024**

## 概述

SMAFormer是一个创新的医学图像分割架构，将Vision Transformer (ViT)的全局建模能力与Spatial-Mamba Attention (SMA)的高效空间感知机制相结合。该架构专为多器官医学图像分割设计，在Synapse多器官分割数据集上取得了优异性能。

## 核心创新

### 1. Spatial-Mamba Attention (SMA)
SMA模块整合了两个关键组件：
- **Pixel-wise Spatial Attention**: 捕获空间维度的局部特征和位置信息
- **Channel-wise Mamba**: 建模通道间的长程依赖关系

SMA通过高效的门控融合机制平衡空间和通道信息，实现了优于传统注意力的性能。

### 2. 并行双路径架构
不同于传统的串行后处理方式，SMAFormer采用并行双路径设计：

```
Input Features
    ├──→ Self-Attention Branch  →  Global Context
    │                              ↓
    └──→ SMA Branch            →  Spatial-Channel Awareness
                                   ↓
                        Gated Fusion (α·SA + β·SMA)
                                   ↓
                              Output Features
```

这种设计允许模型同时处理全局上下文和空间细节，并通过自适应门控机制动态融合。

### 3. 多尺度特征提取 (DPT-Style)
借鉴DPT (Dense Prediction Transformer)的思想，从ViT的中间层提取多尺度特征：

```
ViT Blocks      Feature      Reassemble       Output Scale
-----------     -------      ----------       ------------
Blocks 0-2   →  F1(768D)  →  Upsample×2   →  C1: 96×64×64
Blocks 3-5   →  F2(768D)  →  Keep         →  C2: 192×32×32
Blocks 6-8   →  F3(768D)  →  Downsample×2 →  C3: 384×16×16
Blocks 9-11  →  F4(768D)  →  Downsample×4 →  C4: 768×8×8
```

这种分层特征金字塔为后续解码器提供了丰富的多尺度信息。

### 4. 增强型Decoder
集成多种先进技术的解码器设计：

- **ASPP (Atrous Spatial Pyramid Pooling)**: 多尺度上下文聚合
- **SE (Squeeze-and-Excitation) Attention**: 通道注意力增强
- **UNet-style Skip Connections**: 跨层特征融合

解码器逐步上采样并融合多尺度特征，最终恢复到原始分辨率的分割掩码。

## 架构设计

### 整体流程

```
Input Image (H×W×3)
        ↓
    Patch Embedding (16×16 patches)
        ↓
    Position Embedding
        ↓
┌───────────────────────────────┐
│  Multi-Scale ViT Encoder      │
│  ┌─────────────────────────┐  │
│  │ Parallel SMA-Transformer │  │
│  │ Blocks (×12)            │  │
│  │  ┌──────────────────┐   │  │
│  │  │ Layer Norm       │   │  │
│  │  ├──────────────────┤   │  │
│  │  │ ┌──────┬────────┐│   │  │
│  │  │ │  SA  │  SMA   ││   │  │
│  │  │ └──┬───┴───┬────┘│   │  │
│  │  │    └─Gate──┘     │   │  │
│  │  ├──────────────────┤   │  │
│  │  │ Layer Norm       │   │  │
│  │  ├──────────────────┤   │  │
│  │  │ MLP (FFN)        │   │  │
│  │  └──────────────────┘   │  │
│  └─────────────────────────┘  │
│    ↓      ↓      ↓      ↓     │
│   C1     C2     C3     C4     │
│ 64×64  32×32  16×16   8×8     │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│  Enhanced Decoder             │
│  ┌─────────────────────────┐  │
│  │ ASPP on C4              │  │
│  ├─────────────────────────┤  │
│  │ Decoder Block 3         │  │
│  │  ├─ Skip from C3        │  │
│  │  └─ SE Attention        │  │
│  ├─────────────────────────┤  │
│  │ Decoder Block 2         │  │
│  │  ├─ Skip from C2        │  │
│  │  └─ SE Attention        │  │
│  ├─────────────────────────┤  │
│  │ Decoder Block 1         │  │
│  │  ├─ Skip from C1        │  │
│  │  └─ SE Attention        │  │
│  ├─────────────────────────┤  │
│  │ Progressive Upsample    │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
        ↓
Output Segmentation (H×W×9)
```

### SMA模块详细设计

```python
# Improved SMA Module with Matrix Fusion
class ImprovedSMAModule(nn.Module):
    """
    空间-通道混合注意力模块
    
    工作流程:
    1. Pixel-wise Spatial: Conv → 空间注意力权重
    2. Channel-wise Mamba: Linear → Mamba → 通道建模
    3. Gated Fusion: 门控机制动态融合两路输出
    """
    def forward(self, x):
        # x: [B, N, C]
        pixel_out = self.spatial_branch(x)    # 空间处理
        channel_out = self.channel_branch(x)   # 通道处理
        gate = self.gate(torch.cat([pixel_out, channel_out], dim=-1))
        return gate * pixel_out + (1 - gate) * channel_out
```

### 并行SMA-Transformer Block

```python
class ImprovedSMATransformerBlock(nn.Module):
    """
    并行处理SA和SMA，门控融合
    
    前向过程:
    1. 标准化输入
    2. 并行计算:
       - Self-Attention分支
       - SMA分支
    3. 门控融合两路输出
    4. 残差连接
    5. FFN处理
    """
    def forward(self, x, H, W):
        # Norm + Parallel Dual-Path
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        sma_out = self.sma(x_norm, H, W)
        
        # Gated Fusion
        gate = self.gate_mlp(torch.cat([attn_out, sma_out], dim=-1))
        fused = gate * attn_out + (1 - gate) * sma_out
        
        # Residual + FFN
        x = x + self.drop_path(fused)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

## 模型配置

### ViT-Base Backbone
- **Patch Size**: 16×16
- **Embedding Dimension**: 768
- **Depth**: 12 Transformer blocks
- **Heads**: 12 attention heads
- **MLP Ratio**: 4.0
- **Pretrained**: ImageNet-21K ViT-Base

### Multi-Scale Encoder
- **Extract Layers**: [2, 5, 8, 11] (after blocks 2, 5, 8, 11)
- **Output Channels**: [96, 192, 384, 768]
- **Scale Factors**: [2.0, 1.0, 0.5, 0.25]

### Enhanced Decoder
- **ASPP**: dilation rates [1, 6, 12, 18]
- **Decoder Stages**: 3 stages (256→128→64 channels)
- **SE Reduction**: 16
- **Final Upsampling**: 4×

## 性能特点

### 参数量与计算量
- **Total Parameters**: 138.89M
- **Pretrained Parameters**: 85.80M (61.8%)
- **FLOPs**: ~125 GFLOPs (at 256×256 input)

### 训练配置
- **Input Size**: 256×256
- **Batch Size**: 28
- **Optimizer**: AdamW (lr=0.0001)
- **Loss Function**: CE + Dice
- **Data Augmentation**: Pseudo-HDR preprocessing

### 推理速度
- **RTX 4090**: ~30ms per image (256×256)
- **Memory**: ~2GB GPU memory for inference

## 使用方法

### 基本使用

```python
from models.SMAFormer import SMAFormer

# 创建模型实例
class Args:
    def __init__(self):
        self.dataset = 'Synapse'

args = Args()

model = SMAFormer(
    args=args,
    img_size=256,                      # 输入图像尺寸
    in_chans=3,                        # 输入通道数
    num_classes=9,                     # 输出类别数
    embed_dim=768,                     # ViT-Base固定
    depth=12,                          # ViT-Base固定
    num_heads=12,                      # ViT-Base固定
    pretrained=True,                   # 使用预训练权重
    pretrained_path='pre_trained_weights',
    
    # 架构配置
    use_multi_scale=True,              # 启用多尺度特征
    use_enhanced_decoder=True,         # 启用增强Decoder
    sma_mode='parallel',               # 并行SMA模式
)

# 前向传播
import torch
x = torch.randn(1, 3, 256, 256).cuda()
output = model(x)  # [1, 9, 256, 256]
```

### 配置选项

```python
# 方案A: 仅并行SMA
model = SMAFormer(
    args=args,
    img_size=256,
    sma_mode='parallel',           # 启用并行SMA
    use_multi_scale=False,
    use_enhanced_decoder=False
)

# 方案B: 多尺度特征
model = SMAFormer(
    args=args,
    img_size=256,
    sma_mode='disabled',           # 禁用SMA
    use_multi_scale=True,          # 多尺度特征
    use_enhanced_decoder=False
)

# 方案C: 增强Decoder
model = SMAFormer(
    args=args,
    img_size=256,
    sma_mode='disabled',
    use_multi_scale=False,
    use_enhanced_decoder=True      # 增强Decoder
)

# 完整版 (推荐): A+B+C
model = SMAFormer(
    args=args,
    img_size=256,
    sma_mode='parallel',           # 方案A
    use_multi_scale=True,          # 方案B
    use_enhanced_decoder=True      # 方案C
)
```

## 预训练权重

### 下载权重
ViT-Base预训练权重会自动下载到`pre_trained_weights/`目录：
```
pre_trained_weights/
└── jx_vit_base_patch16_224-8ee2ff3e.pth
```

### 权重加载统计
模型初始化时会显示详细的权重加载信息：
```
============================================================
✅ SMAFormer预训练权重加载成功！
============================================================
📊 加载统计:
   - 成功加载的参数层数: 149
   - 成功加载的参数量: 85,797,120
   - ViT-Base有效参数量: 85,797,120
   - 🎯 预训练权重加载率: 100.00%
   - 模型参数覆盖率: 61.8%
   - 随机初始化层数: 660 (decoder部分)
============================================================
```

## 训练

### 数据准备
请参考项目根目录的数据准备文档。Synapse数据集应组织为：
```
data/Synapse/
├── train_npz/
│   ├── case0001_slice000.npz
│   ├── case0001_slice001.npz
│   └── ...
└── test_vol/
    ├── case0001.npy.h5
    ├── case0002.npy.h5
    └── ...
```

### 开始训练

```bash
# 激活环境
conda activate seg

# 训练SMAFormer (完整版)
python train_synapse.py --max_epochs 150

# 训练特定配置
# 修改 configs/config_setting_synapse.py 中的参数
```

### 配置文件

修改`configs/config_setting_synapse.py`:

```python
smaformer_config = {
    'num_classes': 9,
    'input_channels': 3,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'pretrained': True,
    
    # 架构开关
    'sma_mode': 'parallel',              # 'parallel', 'disabled', 'original'
    'use_multi_scale': True,             # 多尺度特征
    'use_enhanced_decoder': True,        # 增强Decoder
}
```

## 评估

```bash
# 测试模型
python test_synapse.py --model_path checkpoints/best_model.pth
```

## 技术细节

### 伪HDR预处理
输入图像被转换为三通道伪HDR表示：
- **Channel 1**: 原始图像
- **Channel 2**: 增强对比度 (×1.2)
- **Channel 3**: 平滑版本 (avg_pool 3×3)

### 位置编码插值
ViT预训练权重使用224×224输入(14×14 patch grid)，训练时自动插值到256×256(16×16 grid)：
```python
# 自动插值位置编码
pos_embed_old = [1, 197, 768]  # 1 cls + 196 patches
pos_embed_new = [1, 257, 768]  # 1 cls + 256 patches
# 通过双线性插值实现
```

### 损失函数
混合损失函数结合CE和Dice：
```python
Loss = CrossEntropy + DiceLoss
```

## 消融研究

不同配置的参数量对比：

| 配置 | 参数量 | 说明 |
|------|--------|------|
| 基线 (ViT-only) | 89.77M | 禁用SMA |
| +Original SMA | 104.85M | 串行SMA |
| +Parallel SMA (A) | 126.12M | 并行门控SMA |
| +Multi-Scale (B) | 97.51M | 多尺度特征 |
| +Enhanced Decoder (C) | 103.13M | 增强解码器 |
| **完整版 (A+B+C)** | **138.89M** | **所有改进** |

## 依赖项

```
torch>=1.10.0
timm>=0.6.0
einops>=0.6.0
numpy>=1.21.0
scipy>=1.7.0
```

## 引用

如果您使用了SMAFormer,请引用：

```bibtex
@inproceedings{smaformer2024,
  title={SMAFormer: Spatial-Mamba Attention Transformer for Medical Image Segmentation},
  author={Your Name},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2024}
}
```

## 致谢

本工作基于以下优秀项目：
- Vision Transformer (ViT) - Google Research
- Mamba - State Space Models
- DPT - Dense Prediction Transformer
- VMUNet - Vision Mamba UNet

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，欢迎提Issue或Pull Request。

---

**Last Updated**: December 2024
