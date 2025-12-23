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

@article{zheng2025hbformer,
  title={HBFormer: A Hybrid-Bridge Transformer for Microtumor and Miniature Organ Segmentation},
  author={Zheng, Fuchen and Chen, Xinyi and Li, Weixuan and Li, Quanjun and Zhou, Junhua and Guo, Xiaojiao and Chen, Xuhang and Pun, Chi-Man and Zhou, Shoujun},
  journal={arXiv preprint arXiv:2512.03597},
  year={2025}
}

```
# 📋SMAFormer

SMAFormer: Synergistic Multi-Attention Transformer for Medical Image Segmentation
[Vedio introduction](https://www.bilibili.com/video/BV1FLDsYqExZ/)

[Fuchen Zheng](https://lzeeorno.github.io/),  [Xuhang Chen](https://cxh.netlify.app/), Weihuang Liu, Haolun Li, Yingtie Lei, Jiahui He, [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) 📮and [Shoujun Zhou](https://people.ucas.edu.cn/~sjzhou?language=en) 📮( 📮 Corresponding authors)

**University of Macau, SIAT CAS, Huizhou University, University of Nottingham Ningbo China**

2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

---

## 1. 概述

SMAFormer V2.3 是基于 Swin Transformer 的医学图像分割架构，其中BIBM2024年最初版本SMAFormer基于ViT架构使用了大量参数（138M）才能再Synapse数据集达到86%的Dice。因此我们参考BIBM2025HBFormer的设计，切换成Swin Transformer的Backbone，在使用仅仅40M的参数量达到了89%的Dice。SMAFomerV2.3采用创新的**镜像Encoder-Decoder设计**，实现了**双向预训练权重加载**。该版本通过让Decoder完全镜像Encoder的结构，将预训练权重覆盖率从传统的65%大幅提升至**94.84%**，为医学图像分割任务提供了更强的初始化基础。

### 核心设计理念

V2.3版本遵循"**最大化预训练权重利用**"的设计哲学：
- 与其设计复杂的Decoder模块（如V2.2的ASPP、CAFF等），不如让Decoder镜像Encoder结构
- Encoder可加载Swin Tiny预训练权重（100%覆盖）
- Decoder通过反向映射也可加载预训练权重（85.87%覆盖）
- 总预训练覆盖率达到**94.84%**，远超传统U-Net架构（仅Encoder有预训练）

## 2.🌳 版本演进树

```
                           SMAFormer
                               │
           ┌───────────────────┴───────────────────┐
           │                                       │
    [Branch: ViT-Base]                    [Branch: Swin-Tiny]
           │                                       │
    ┌──────┴──────┐                        ┌──────┴──────┐
    │  SMAFormer  │                        │ SMAFormerV2 │
    │  (ViT-Base) │                        │ (Swin-Tiny) │
    └──────┬──────┘                        └──────┬──────┘
           │                                       │
    v1.1 原始版本                           v2.1 原始版本
    Dice: 74%                              Dice: 78%
           │                                       │
    v1.2 [2024-12-21]                      v2.2 [2025-12-21]
    ├─ Adapter风格SMA                      ├─ 改进SMA模块
    ├─ 正弦余弦位置编码                     ├─ DynamicMultiScaleASPP
    ├─ DPT多尺度特征                       ├─ CrossAttention融合
    └─ UNet Decoder                        └─ BoundaryAwareHead
           │                                       │
           ▼                                       │
    Dice: 80%                                     │
                                                   │
                                           v2.2 复杂特征版本
                                           (ASPP+CAFF+边界头)
                                           Dice: 80%
                                                   │
                                           v2.3 ★ 镜像架构 [最新]
                                           ├─ Decoder镜像Encoder
                                           ├─ 双向预训练权重加载
                                           ├─ Encoder: 100%加载
                                           └─ Decoder: 86%加载
                                                   │
                                                   ▼
                                           Dice: 89%
```

---


#### 🆚 版本对比

| 特性 | V2.1 | V2.2 | V2.3 |
|------|------|------|------|
| Decoder类型 | ConvNext | Complex (ASPP+CAFF) | **Mirror Swin** |
| Decoder预训练 | ❌ 无 | ❌ 无 | **✅ 86%** |
| 总预训练覆盖率 | ~65% | ~65% | **95%** |
| 参数量 | ~42M | ~55M | **41.4M** |
| 设计复杂度 | 低 | 高 | **低** |
| Dice (验证) | 78% | 80% | 89% |



---

## 3. 📈 性能对比表

| 模型 | Backbone | 切片Dice | 3D Dice | HD95 | 状态 |
|------|----------|----------|---------|------|------|
| HBFormer | Swin-Tiny | 98.56% | **87.82%** | **6.92** | ✅ 基准 |
| SMAFormer v1.1 | ViT-Base | 83.36% | 70% | 35+ | ❌ 已废弃 |
| SMAFormer v1.2 | ViT-Base | 98.12% | 86% | - | ❌ 已废弃 |
| SMAFormerV2 v2.1| Swin-Tiny | 95.32 | 78.66% | 27.51 | ❌ 已废弃 |
| SMAFormerV2 v2.2 | Swin-Tiny | 98.39 | 78.24% | 24.47 | ⚠️ 复杂特征 |
| SMAFormerV2 v2.3 | Swin-Tiny | 98.36 | 89.41% | 6.64 | 🌟 **镜像架构** |

---

## 4. 🔬 改进方法参考文献

### 注意力机制
- **CBAM** (ECCV 2018): Convolutional Block Attention Module
- **SE-Net** (CVPR 2018): Squeeze-and-Excitation Networks  
- **ECA-Net** (CVPR 2020): Efficient Channel Attention

### 动态网络
- **ODConv** (ICLR 2022): Omni-Dimensional Dynamic Convolution
- **CondConv** (NeurIPS 2019): Conditionally Parameterized Convolutions

### Transformer改进
- **ViT-Adapter** (ICLR 2023): Vision Transformer Adapter
- **DPT** (ICCV 2021): Vision Transformers for Dense Prediction
- **Swin-UNet** (arXiv 2021): Swin Transformer for Medical Segmentation

### 分割方法
- **SegNeXt** (NeurIPS 2022): Rethinking Convolutional Attention
- **ConvNeXt V2** (CVPR 2023): Co-designing and Scaling ConvNets
- **NAFNet** (ECCV 2022): Simple Baselines for Image Restoration

### 医学影像
- **TransUNet** (arXiv 2021): Transformers for Medical Image Segmentation
- **Swin-UNet** (arXiv 2021): Swin Transformer for Medical Segmentation
- **HBFormer**: BIBM2025

---

## 5. 性能特点

### 参数量与计算量
- **Total Parameters**: 41.43M


### 训练配置
- **Input Size**: 256×256
- **Batch Size**: 40 (a6000)
- **Optimizer**: AdamW (lr=0.0001)
- **Loss Function**: CE + Dice
- **Data Augmentation**: Pseudo-HDR preprocessing



## 6. 预训练权重

### 下载权重
预训练权重下载到`pre_trained_weights/`目录：
```
pre_trained_weights/swin_tiny_patch4_window7_224.pth
```




## 7. SMAFormerV2 各版本差异

| 特性 | V2.1 | V2.2 | V2.3 (当前) |
|------|------|------|-------------|
| **Encoder** | Swin-Tiny | Swin-Tiny | Swin-Tiny |
| **Decoder** | ConvNeXt | Complex (ASPP+CAFF+边界头) | **Mirror Swin** |
| **Encoder预训练** | ✅ 100% | ✅ 100% | ✅ **100%** |
| **Decoder预训练** | ❌ 0% | ❌ 0% | ✅ **85.87%** |
| **总预训练覆盖** | ~65% | ~65% | ✅ **94.84%** |
| **参数量** | ~42M | ~55M | ✅ **41.4M** |
| **设计复杂度** | 低 | 高 | ✅ **低** |
| **Dice (Synapse)** | ~78% | ~80% | ~89.41% |

### 关键优势

1. **预训练覆盖率最高**: 94.84% vs 传统UNet架构的50-60%
2. **参数效率**: 41.4M参数，相比V2.2减少25%
3. **设计简洁**: 避免过度设计，专注于预训练权重利用
4. **训练稳定**: Decoder也有良好初始化，收敛更快


----

## 8. 训练SMAFormer (完整版)

```
python train_synapse.py --max_epochs 300 --batch_size 40 --model SMAFormerV2 --config configs/config_setting_synapse.py
```


## 9. 评估

```bash
# 测试模型
python test_synapse.py --model_path checkpoints/best.pth
```


### 查看权重加载情况

模型初始化时会自动打印预训练权重加载报告：

```
================================================================================
SMAFormerV2 V2.3 - 双向预训练权重加载报告
================================================================================

📦 [1/2] 加载 Encoder 预训练权重...
   ├─ Encoder参数总量: 27,519,354
   ├─ 成功加载参数量: 27,519,354
   └─ 加载成功率: 100.00%

📦 [2/2] 加载 Decoder 预训练权重 (反向映射)...
   ├─ Decoder参数总量: 13,712,778
   ├─ 成功加载参数量: 11,775,402
   └─ 加载成功率: 85.87%

================================================================================
📊 预训练权重加载总结
================================================================================
   ┌─────────────────────────────────────────────────────┐
   │  Encoder 加载率: 100.00%  (27,519,354/27,519,354)   │
   │  Decoder 加载率:  85.87%  (11,775,402/13,712,778)   │
   │  ─────────────────────────────────────────────────── │
   │  模型总参数量: 41,432,027 (41.43M)                   │
   │  预训练覆盖率: 94.84%                                 │
   └─────────────────────────────────────────────────────┘
```

## 10. 依赖项

```bash
torch>=1.10.0
timm>=0.6.0
numpy>=1.21.0
scipy>=1.7.0
h5py>=3.1.0
```



---

## 11.项目结构

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





---

## 12. 训练技巧

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

## 13. 可视化

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

### 为什么选择镜像架构？

传统的医学图像分割模型（如UNet、TransUNet）通常只在Encoder部分使用预训练权重，Decoder部分从随机初始化开始训练。这导致：
1. 预训练权重利用率低（通常<60%）
2. Decoder需要从头学习，收敛慢
3. 容易过拟合（Decoder随机初始化）

V2.3通过**让Decoder镜像Encoder结构**，实现了：
1. Decoder也可以加载预训练权重（反向映射）
2. 预训练覆盖率提升至94.84%
3. 训练更稳定，收敛更快
4. 避免过度设计复杂模块

### 为什么Decoder加载率是85.87%而非100%？

Decoder无法100%加载的原因：
1. **PatchExpanding模块**是新设计的（PatchMerging的逆操作），没有对应的预训练权重
2. **Skip connection的projection层**需要处理concatenation后的特征，与预训练不匹配
3. **输入维度不同**：Decoder各stage的输入维度与Encoder不完全对称

但即便如此，85.87%的Decoder加载率已经远超传统架构的0%。


---

# 🧧 Acknowledgement

This work was supported in part by the National Key R\&D Project of China (2018YFA0704102, 2018YFA0704104), in part by Natural Science Foundation of Guangdong Province (No. 2023A1515010673), and in part by Shenzhen Technology Innovation Commission (No. JSGG20220831110400001), in part by Shenzhen Development and Reform Commission (No. XMHT20220104009), in part by the Science and Technology Development Fund, Macau SAR, under Grant 0141/2023/RIA2 and 0193/2023/RIA3.


