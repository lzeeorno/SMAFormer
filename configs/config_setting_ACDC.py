from pickle import FALSE
from torchvision import transforms
from datasets.dataset_ACDC import *
from utils import *

from datetime import datetime
import ml_collections

class setting_config:
    """
    ACDC数据集训练配置（心脏分割）- 3分类不包括背景
    """
    # 模型选择配置 - 新增功能
    available_models = ['DWSegNet', 'AFFSegNet', 'SMAFormerV2']  # 可用的模型列表
    network = 'AFFSegNet'  # 默认使用AFFSegNet，可通过set_model方法切换
    
    affsegnet_config = {
        'num_classes': 3,  # 右心室、心肌、左心室（不包括背景）
        'input_channels': 3, 
        'feature_size': 48,
        'use_boundary_refinement': False,
        'load_ckpt_path': '',  # 不使用预训练权重
    }
    
    # SMAFormerV2 配置 - V2.3版本 (基于Swin-Tiny，镜像Encoder-Decoder架构)
    # 核心改进：Decoder镜像Encoder结构，实现双向预训练权重加载
    # - Encoder加载率：100%
    # - Decoder加载率：~86%
    # - 总预训练覆盖率：~95%
    smaformerv2_config = {
        'num_classes': 3,  # 右心室、心肌、左心室（不包括背景）
        'input_channels': 3,
        'img_size': 224,  # ACDC图像尺寸
        'embed_dims': [96, 192, 384, 768],  # Swin Tiny通道数
        'depths': [2, 2, 6, 2],  # Swin Tiny层深度
        'num_heads': [3, 6, 12, 24],  # Swin Tiny注意力头数
        'window_size': 7,  # Swin窗口大小
        'mlp_ratio': 4.,
        'drop_rate': 0.,
        'drop_path_rate': 0.2,
        'use_sma': True,  # 是否使用轻量级SMA增强
        'load_pretrained': True,  # 是否加载预训练权重
        'pretrained_path': 'pre_trained_weights/swin_tiny_patch4_window7_224.pth',
    }
    
    # 根据选择的网络设置model_config
    if network == 'AFFSegNet':
        model_config = affsegnet_config
    elif network == 'SMAFormerV2':
        model_config = smaformerv2_config
    else:
        model_config = affsegnet_config
    datasets_name = 'ACDC'
    input_size_h = 224  # ACDC图像通常较小
    input_size_w = 224
    
    # 数据路径配置
    data_path = './data/ACDC/'
    datasets = ACDC_dataset
    
    pretrained_path = ''
    num_classes = 3  # 改为3分类：右心室、心肌、左心室
    loss_weight = [0.33, 0.33, 0.34]  # CE, Dice, Focal weights
    criterion = CeDiceFocalLoss(num_classes, loss_weight)  # 使用正确的3分类
    z_spacing = 1
    input_channels = 3

    # 分布式训练配置
    distributed = False
    local_rank = -1
    num_workers = 8
    seed = 2050
    world_size = None
    rank = None
    amp = False

    # 训练参数
    batch_size = 2  # ACDC图像较小，可以用较大的batch size
    epochs = 3
    work_dir = 'results/' + network + '_' + datasets_name 
    print_interval = 20
    val_interval = 1  # 每1个epoch验证一次，便于测试
    test_weights_path = ''

    threshold = 0.5

    # 优化器配置
    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    
    if opt == 'AdamW':
        lr = 3e-4
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    
    # 学习率调度器配置
    sch = 'CosineAnnealingLR'
    if sch == 'CosineAnnealingLR':
        T_max = 300  # 与epochs相同
        eta_min = 1e-6
        last_epoch = -1 

    # 新增保存间隔配置  
    save_interval = 10  # 每10个epoch保存一次权重

    @classmethod
    def set_model(cls, model_name):
        """设置使用的模型"""
        if model_name not in cls.available_models:
            raise ValueError(f"模型必须是以下之一: {cls.available_models}")
        cls.network = model_name
        print(f"设置模型为: {model_name}")
        
        # 更新工作目录
        cls.work_dir = f'results/{cls.network}_{cls.datasets_name}'
    
    @classmethod
    def set_save_interval(cls, interval):
        """设置保存间隔"""
        if interval <= 0:
            raise ValueError("保存间隔必须大于0")
        cls.save_interval = interval
        print(f"设置保存间隔为每{interval}个epoch保存一次") 