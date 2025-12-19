# 🛎 Citation

If you find our work helpful for your research, please cite:

```bib
@inproceedings{zheng2024smaformer,
  title={Smaformer: Synergistic multi-attention transformer for medical image segmentation},
  author={Zheng, Fuchen and Chen, Xuhang and Liu, Weihuang and Li, Haolun and Lei, Yingtie and He, Jiahui and Pun, Chi-Man and Zhou, Shoujun},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={4048--4053},
  year={2024},
  organization={IEEE}
}

```
# 📋SMAFormer

SMAFormer: Synergistic Multi-Attention Transformer for Medical Image Segmentation
[Vedio introduction](https://www.bilibili.com/video/BV1FLDsYqExZ/)

[Fuchen Zheng](https://lzeeorno.github.io/),  [Xuhang Chen](https://cxh.netlify.app/), Weihuang Liu, Haolun Li, Yingtie Lei, Jiahui He, [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) 📮and [Shoujun Zhou](https://people.ucas.edu.cn/~sjzhou?language=en) 📮( 📮 Corresponding authors)

**University of Macau, SIAT CAS, Huizhou University, University of Nottingham Ningbo China**

2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

## 🚧 Installation 
Requirements: `Ubuntu 20.04`

1. Create a virtual environment: `conda create -n your_environment python=3.8 -y` and `conda activate your_environment `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) :`pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118`
Or you can use Tsinghua Source for installation
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```
3. `pip install tqdm scikit-learn albumentations==1.0.3 pandas einops axial_attention`
4. `pip install xlsxwriter`


## 1. 概述

SMAFormer 是一个专为医学图像分割设计的深度学习架构，结合了 Swin Transformer 的分层特征提取能力和 Synergistic Multi-Attention (SMA) 机制的细节感知能力。该模型在保持高精度的同时，显著降低了参数量和计算复杂度，特别适合多器官医学图像分割任务。

### 核心特点

- 🏗️ **分层架构**: 基于 Swin Transformer 的四阶段编码器，提供多尺度特征金字塔
- 🎯 **SMA 增强**: 协同融合像素、通道和空间三种注意力机制
- ⚡ **高效设计**: 42.66M 参数，30.50 GFLOPs，训练和推理速度快
- 🔧 **预训练加载**: 94.35% 的编码器参数加载 ImageNet 预训练权重
- 🎨 **边缘增强**: 内置 Sobel 边缘检测提升分割边界精度

---

## 架构设计

### 整体流程

```
输入图像 (3 × 256 × 256)
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Input Projection Layer                                     │
│  3×3 Conv + ReLU                                            │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Swin Transformer Encoder (预训练: 94.35%)                  │
│                                                              │
│  ┌──────────┐  SMA  ┌──────────┐  SMA  ┌──────────┐  SMA   │
│  │ Stage 1  │  ───→ │ Stage 2  │  ───→ │ Stage 3  │  ───→  │
│  │ 96×64×64 │   1   │192×32×32 │   2   │384×16×16 │   3   │
│  └────┬─────┘       └────┬─────┘       └────┬─────┘       │
│       │                  │                  │              │
│       F1                 F2                 F3             │
│                                                              │
│  ┌──────────┐  SMA                                          │
│  │ Stage 4  │  ───→ (Bottleneck)                           │
│  │768×8×8   │   4                                           │
│  └────┬─────┘                                               │
│       │                                                      │
│       F4                                                     │
└───────┼──────────────────────────────────────────────────────┘
        │
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Symmetric Decoder (随机初始化)                              │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │  Decoder 4   │ ←─  │  Decoder 3   │ ←─  │  Decoder 2   ││
│  │  768→384     │ F3  │  384→192     │ F2  │  192→96      ││
│  │+ 2×SMA Block │     │+ 2×SMA Block │     │+ 2×SMA Block ││
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘│
│         │ Upsample×2         │ Upsample×2         │        │
│         └────────────────────┴────────────────────┘        │
│                             ↓                                │
│                    96 × 64 × 64                             │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Final Upsampling + Edge Enhancement                        │
│  ConvTranspose2d (×4) + Sobel Edge Detection                │
└─────────────────────────────────────────────────────────────┘
        ↓
输出分割图 (num_classes × 256 × 256)
```

### 编码器 - Swin Transformer Backbone

采用 Swin Transformer Tiny 作为特征提取器：

| Stage | Input Size | Depth | Heads | Channels | Output Size | 预训练 |
|-------|-----------|-------|-------|----------|-------------|--------|
| Patch Embed | 256×256×3 | - | - | - | 96×64×64 | ✅ |
| Stage 1 | 96×64×64 | 2 | 3 | 96 | 96×64×64 | ✅ |
| Stage 2 | 96×64×64 | 2 | 6 | 192 | 192×32×32 | ✅ |
| Stage 3 | 192×32×32 | 6 | 12 | 384 | 384×16×16 | ✅ |
| Stage 4 | 384×16×16 | 2 | 24 | 768 | 768×8×8 | ✅ |

**关键特性**：
- **窗口注意力**: 7×7 窗口大小，计算复杂度 O(n) 而非 O(n²)
- **Shifted Window**: 交替使用常规窗口和偏移窗口，实现跨窗口信息交互
- **层次特征**: 四个不同分辨率的特征层，自然适配 U-Net 风格解码器

---

## 项目结构

SMAFormerV2 相关的文件组织：

```
AFFSegNnet_VMUnetVis/
│
├── models/                                    # 模型定义
│   ├── SMAFormerV2.py                        # ✨ SMAFormerV2 主模型
│   ├── SMAFormerV2_README.md                 # 本文档
│   ├── SMAFormerV2_arch.html                 # ✨ 架构可视化 HTML
│   └── ...
│
├── configs/                                   # 配置文件
│   ├── config_setting_synapse.py             # ✨ Synapse 数据集配置
│   │   └── smaformerv2_config {...}          #    SMAFormerV2 配置段
│   ├── config_setting_lits2017.py            # LiTS2017 配置
│   └── config_setting_ACDC.py                # ACDC 配置
│
├── train_synapse.py                          # ✨ 训练脚本
├── test_synapse.py                           # ✨ 测试脚本
├── engine_synapse.py                         # 训练/验证引擎
│
├── datasets/                                  # 数据集加载
│   ├── dataset.py                            # Synapse 数据集
│   └── ...
│
├── utils.py                                   # 工具函数
│   ├── cal_params_flops()                    # 参数量/FLOPs 计算
│   ├── test_single_volume()                  # 3D 体积测试
│   └── calculate_metric_percase()            # Dice/HD95 计算
│
├── data/                                      # 数据目录
│   └── Synapse/                              # ✨ Synapse 数据集
│       ├── train_npz/                        #    训练数据 (NPZ)
│       │   ├── case0001_slice000.npz
│       │   ├── case0001_slice001.npz
│       │   └── ...
│       ├── test_vol_h5/                      #    测试数据 (H5)
│       │   ├── case0001.npy.h5
│       │   └── ...
│       └── lists/lists_Synapse/              #    数据列表
│           ├── train.txt                     #    训练集列表
│           ├── test_vol.txt                  #    测试体积列表
│           └── test_slice.txt                #    测试切片列表
│
├── pre_trained_weights/                       # 预训练权重
│   └── swin_tiny_patch4_window7_224.pth      # ✨ Swin-Tiny 预训练
│
├── results/                                   # 训练结果
│   └── SMAFormerV2_Synapse/                  # ✨ SMAFormerV2 实验
│       ├── checkpoints/                      #    模型权重
│       │   ├── best.pth                      #    最佳模型
│       │   ├── best_dice.pth                 #    最佳 Dice
│       │   └── latest.pth                    #    最新检查点
│       ├── train_record.csv                  #    训练记录
│       ├── val_record.csv                    #    验证记录
│       ├── log/                              #    训练日志
│       └── outputs/                          #    预测可视化
│
├── test_result/                               # 测试结果
│   └── SMAFormerV2_Synapse/
│       ├── test_results_detailed.json        # 详细结果
│       ├── test_results_summary.csv          # 结果汇总
│       └── visualizations/                   # 可视化
│
└── SMAFORMERV2_README.md                     # V2 总体文档
```

### 关键文件说明

| 文件/目录 | 用途 | 重要性 |
|----------|------|--------|
| `models/SMAFormerV2.py` | 模型定义：SMA、E-MLP、DecoderStage、EdgeEnhancement | ⭐⭐⭐ |
| `models/SMAFormerV2_arch.html` | 交互式架构可视化，包含流程图、模块图 | ⭐⭐⭐ |
| `configs/config_setting_synapse.py` | 模型配置、训练超参数、数据路径 | ⭐⭐⭐ |
| `train_synapse.py` | 训练主循环、模型创建、优化器设置 | ⭐⭐⭐ |
| `test_synapse.py` | 测试脚本、指标计算、结果保存 | ⭐⭐⭐ |
| `pre_trained_weights/swin_tiny_*.pth` | Swin Transformer 预训练权重 | ⭐⭐⭐ |
| `results/SMAFormerV2_Synapse/checkpoints/` | 训练产生的模型权重 | ⭐⭐⭐ |

---


## 2. Prepare the pre_trained weights and Data

数据遵守SwinUnet的划分格式，将 Synapse 数据集组织为以下结构：

```
data/Synapse/
├── train_npz/          # 训练数据 (2D 切片)
│   ├── case0001_slice000.npz
│   └── ...
├── test_vol_h5/        # 测试数据 (3D 体积)
│   ├── case0001.npy.h5
│   └── ...
└── lists/lists_Synapse/
    ├── train.txt       # 训练集文件名
    └── test_vol.txt    # 测试集文件名
```
- The weights of the pre-trained SMAFormer could be downloaded from [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth).


### 3. 配置模型

编辑 `configs/config_setting_synapse.py`：

```python
# 选择 SMAFormerV2
network = 'SMAFormerV2'

# SMAFormerV2 配置
smaformerv2_config = {
    'num_classes': 9,                          # Synapse: 9 类
    'input_channels': 3,                       # RGB 输入
    'img_size': (256, 256),                    # 输入尺寸
    'swin_pretrained_path': 'pre_trained_weights/swin_tiny_patch4_window7_224.pth',
    'use_edge_enhancement': True,              # 启用边缘增强
}
```

### 4. 训练模型

```bash
# 激活环境
conda activate seg

# 开始训练
python train_synapse.py

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python train_synapse.py
```

**训练参数**（在 config 中配置）：
- Batch Size: 28
- Learning Rate: 3e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Epochs: 150
- Loss: CrossEntropy + Dice

### 5. 测试模型

```bash
# 测试（自动加载 best.pth）
python test_synapse.py

# 测试并保存可视化
python test_synapse.py --save_vis
```

### 6. 代码调用

```python
from models.SMAFormerV2 import SMAFormerV2
import torch

# 创建 args 对象
class Args:
    def __init__(self):
        self.dataset = 'Synapse'

args = Args()

# 创建模型
model = SMAFormerV2(
    args=args,
    img_size=256,                              # 输入尺寸
    num_classes=9,                             # 类别数
    pretrained_path='pre_trained_weights/swin_tiny_patch4_window7_224.pth',
    use_edge_enhancement=True                  # 边缘增强
).cuda()

# 前向传播
x = torch.randn(2, 3, 256, 256).cuda()
with torch.no_grad():
    output = model(x)  # [2, 9, 256, 256]

print(f"Input: {x.shape}, Output: {output.shape}")
```

---

## 模型性能

### 参数量与计算量

| 指标 | 数值 | 说明 |
|------|------|------|
| **总参数量** | 42.66M | 比原版 SMAFormer 减少 69% |
| **GFLOPs** | 30.50 | 比原版 SMAFormer 减少 76% |
| **Encoder 参数** | 27.52M (64.5%) | 94.35% 加载预训练 |
| **SMA 参数** | 2.46M (5.8%) | 随机初始化 |
| **Decoder 参数** | 12.52M (29.3%) | 随机初始化 |

### 预训练权重加载

运行时输出的权重加载报告：

```
======================================================================
SMAFormerV2 预训练权重加载报告
======================================================================

📦 预训练权重文件: pre_trained_weights/swin_tiny_patch4_window7_224.pth
   预训练权重总层数: 190
   模型Encoder总层数: 171

📊 权重加载统计:
   ├─ 成功匹配的层数: 162 / 171
   ├─ 成功加载的参数量: 25,965,690
   ├─ Encoder总参数量: 27,519,354
   └─ Encoder预训练权重覆盖率: 94.35%

📈 模型各部分参数统计:
   ├─ Encoder (Swin): 27,519,354 (27.52M)
   ├─ SMA Stages: 2,455,976 (2.46M)
   ├─ Decoder: 12,518,028 (12.52M)
   └─ 总参数量: 42,661,643 (42.66M)

✅ Encoder权重加载完成!
   - 预训练权重利用率: 90.99%
   - Decoder权重: 随机初始化 (需要训练)
======================================================================
```

### 训练性能

基于 RTX 4090 的训练速度：

| 指标 | 数值 |
|------|------|
| 训练速度 | ~5.8 it/s (batch_size=28) |
| 单 epoch 时间 | ~125 秒 |
| 验证速度 (slice) | ~72 slices/s |
| GPU 显存 | ~8GB (训练) / ~2GB (推理) |


---

## 训练技巧

### 1. 学习率调度

使用 CosineAnnealingLR：

```python
# 初始学习率
initial_lr = 3e-4

# Cosine 退火
lr_t = lr_min + 0.5 * (initial_lr - lr_min) * (1 + cos(π * t / T))
```

### 2. 数据增强

```python
transform_train = transforms.Compose([
    RandomGenerator(output_size=[256, 256])  # 随机裁剪、旋转、翻转
])
```

### 3. 损失函数

混合损失：

```python
loss = ce_loss + dice_loss

# CE Loss: 逐像素交叉熵
ce = CrossEntropyLoss()(pred, target)

# Dice Loss: 优化 Dice 系数
dice = DiceLoss()(pred, target)
```

### 4. 梯度裁剪

防止梯度爆炸：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 可视化

打开 `models/SMAFormerV2_arch.html` 查看交互式架构可视化，包括：

- 🏗️ 完整的网络架构图
- 📊 参数量和 FLOPs 统计
- 🔧 SMA 模块详解
- ⚡ E-MLP 结构说明
- 📞 完整的调用关系图（Call Graph）
- 🔄 数据流向图（Flow Chart）
- 📋 权重加载详情表

---

## 常见问题

### Q1: 如何切换数据集？

修改 `configs/config_setting_synapse.py`：

```python
datasets_name = 'Synapse'  # 或 'LiTS2017', 'ACDC'
```

模型会自动根据数据集调整类别数。

### Q2: 如何调整模型大小？

目前使用 Swin-Tiny，可以通过修改 `SMAFormerV2.py` 中的 `embed_dims` 参数来调整：

```python
# Swin-Tiny (默认)
embed_dims = [96, 192, 384, 768]

# 更小的模型
embed_dims = [64, 128, 256, 512]

# 更大的模型 (Swin-Small)
embed_dims = [96, 192, 384, 768]
depths = [2, 2, 18, 2]  # 增加 depth
```

### Q3: 训练显存不足怎么办？

减小 batch size：

```python
# configs/config_setting_synapse.py
batch_size = 16  # 从 28 减少到 16
```

或使用梯度累积：

```python
accumulation_steps = 2
loss = loss / accumulation_steps
loss.backward()
if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Q4: 如何只使用部分改进？

修改配置（虽然当前版本都是默认开启的）：

```python
# 关闭边缘增强
model = SMAFormerV2(
    ...,
    use_edge_enhancement=False
)
```

### Q5: 可以加载 HBFormer 的权重吗？

可以部分加载编码器权重，因为都使用 Swin Transformer：

```python
# 加载 HBFormer checkpoint
checkpoint = torch.load('hbformer_checkpoint.pth')

# 提取 encoder 权重
encoder_weights = {k: v for k, v in checkpoint.items() 
                   if k.startswith('encoder.')}

# 加载到 SMAFormerV2
model.encoder.load_state_dict(encoder_weights, strict=False)
```

---

## 依赖项

```txt
torch>=1.10.0
torchvision>=0.11.0
timm>=0.6.0
einops>=0.6.0
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
h5py>=3.6.0
medpy>=0.4.0
SimpleITK>=2.1.0
```

---

# 🧧 Acknowledgement

This work was supported in part by the National Key R\&D Project of China (2018YFA0704102, 2018YFA0704104), in part by Natural Science Foundation of Guangdong Province (No. 2023A1515010673), and in part by Shenzhen Technology Innovation Commission (No. JSGG20220831110400001), in part by Shenzhen Development and Reform Commission (No. XMHT20220104009), in part by the Science and Technology Development Fund, Macau SAR, under Grant 0141/2023/RIA2 and 0193/2023/RIA3.


