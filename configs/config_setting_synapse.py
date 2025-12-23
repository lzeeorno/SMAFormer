from pickle import FALSE
from torchvision import transforms
from datasets.dataset import *
from utils import *

from datetime import datetime
import ml_collections

class setting_config:
    """
    the config of training setting.
    """
    # 模型选择配置 - 新增功能
    available_models = ['DWSegNet', 'AFFSegNet', 'vmunet', 'HBFormer', 'SMAFormer', 'SMAFormerV2']  # 可用的模型列表
    network = 'SMAFormerV2'  # 切换到SMAFormer进行训练
    
    affsegnet_config = {
        'num_classes': 9, 
        'input_channels': 3, 
        'feature_size': 48,  # AFFSegNet的特征维度
        'use_boundary_refinement': False,
        # 可选的预训练权重路径
        'load_ckpt_path': '',  # 暂时不使用预训练权重
        # vmunet特定配置
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 9, 2, 2],
        'drop_path_rate': 0.2,
    }

    dwsegnet_config = {
        'num_classes': 9, 
        'input_channels': 3, 
        'feature_size': 48,  # AFFSegNet的特征维度
        'use_boundary_refinement': False,
        # 可选的预训练权重路径
        'load_ckpt_path': '',  # 暂时不使用预训练权重
        # vmunet特定配置
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 9, 2, 2],
        'drop_path_rate': 0.2,
    }

    hbformer_config = {
        'num_classes': 9,
        'input_channels': 3,
        'feature_size': 48,
        'use_boundary_refinement': False,
    }

    # SMAFormer 配置 - 2025-12-21改进版本 (基于ViT-Base)
    # 核心改进：Adapter风格SMA + 混合位置编码 + DPT多尺度特征 + UNet解码器
    smaformer_config = {
        'num_classes': 9,
        'input_channels': 3,
        'embed_dim': 768,  # ViT-Base固定768
        'depth': 12,  # ViT-Base固定12个blocks
        'num_heads': 12,  # ViT-Base固定12个heads
        'mlp_ratio': 4.,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'pretrained': True,
        'pretrained_model': 'vit_base_patch16_224',  # 使用ViT-Base
        'pretrained_path': 'pre_trained_weights',
        'use_sma': True,  # 是否使用SMA Adapter模块
    }

    # SMAFormerV2 配置 - V2.3版本 (基于Swin-Tiny，镜像Encoder-Decoder架构)
    # 核心改进：Decoder镜像Encoder结构，实现双向预训练权重加载
    # - Encoder加载率：100%
    # - Decoder加载率：~86%
    # - 总预训练覆盖率：~95%
    smaformerv2_config = {
        'num_classes': 9,
        'input_channels': 3,
        'img_size': 256,  # 输入图像尺寸（内部会resize到224处理，输出resize回256）
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

    # VMUNet 配置 - 恢复原版设置
    vmunet_config = {
        'num_classes': 9, 
        'input_channels': 3,
        'model_name': 'vmunet_s',
        'depths': [2, 2, 2, 2],  # 恢复原版VMUNet层深度配置
        'depths_decoder': [2, 2, 2, 1],  # 恢复原版VMUNet解码器层深度配置
        'drop_path_rate': 0.2,  # VMUNet需要的drop path rate
        'load_ckpt_path': None,
        'pretrained_path': 'pre_trained_weights/vmamba_small_e238_ema.pth',  # VMUNet预训练权重
        'load_pretrained': True,  # 启用预训练权重加载
    }

     # 根据选择的网络设置model_config
    if network == 'vmunet':
        model_config = vmunet_config
    elif network == 'HBFormer':
        model_config = hbformer_config
    elif network == 'DWSegNet':
        model_config = dwsegnet_config
    elif network == 'AFFSegNet':
        model_config = affsegnet_config
    elif network == 'SMAFormer':
        model_config = smaformer_config
    elif network == 'SMAFormerV2':
        model_config = smaformerv2_config
    else:
        raise ValueError(f"不支持的网络类型: {network}")



    datasets_name = 'Synapse'  # 保持数据集名称为Synapse，但会转换为大写
    # input_size_h = 224
    # input_size_w = 224
    input_size_h = 256
    input_size_w = 256
    if datasets_name == 'synapse' or datasets_name == 'Synapse':
        data_path = './data/Synapse/train_npz/'
        datasets = Synapse_dataset
        list_dir = './data/Synapse/lists/lists_Synapse/'
        volume_path = './data/Synapse/test_vol_h5/'
        # 新增：支持slice-by-slice的测试
        test_slice_path = './data/Synapse/train_npz/'  # 测试slice也使用训练数据路径
        test_list_file = 'test_slice.txt'  # 使用slice级别的测试列表
    else:
        raise Exception('datasets in not right!')
    
    pretrained_path = '' # if using pretrained, please enter the path of weights
    num_classes = 9
    # 强化Dice和Focal在总损失中的权重，以更关注小器官
    loss_weight = [0.2, 0.4, 0.4]  # CE, Dice, Focal weights
    criterion = CeDiceFocalLoss(num_classes, loss_weight)
    z_spacing = 1
    input_channels = 3

    distributed = False
    local_rank = -1
    num_workers = 1 #16
    seed = 2050
    world_size = None
    rank = None
    amp = False

    batch_size = 60  # batch size (4090: 26 HBFormer, 48for SMAFormerV1, a6000: 118 for SMAFormerV1, 75 for SMAFormerV2.3)
    epochs = 300  # 暂时设置为5个epoch测试修复效果
    resume_training = False  # 是否恢复训练，False表示重新开始训练
    work_dir = 'results/' + network + '_' + datasets_name 
    # 'D:/CODES/MedSeg/BIBM22/results/datrm2_isic18_Sunday_04_September_2022_12h_04m_10s/'
    print_interval = 20  # 更频繁的输出
    val_interval = 2   # 每2个epoch验证一次
    # 新增保存间隔配置
    save_interval = 50  # 每10个epoch保存一次权重

    test_weights_path = './results/SMAFormerV2_Synapse/checkpoints/best.pth'  # 使用AFFSegNet的300epoch权重路径

    threshold = 0.5

    opt = 'AdamW'  # Changed to AdamW as requested
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.0001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 3e-4 # 降低学习率避免梯度爆炸
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-3 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.003 # – learning rate
        momentum = 0.98 # default: 0 – momentum factor
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'  # Changed to CosineAnnealingLR as requested
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 300 # Set to 300 as requested
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: 'min'.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn't improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: 'rel'.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20

    
    
    # 3D预测相关配置
    pred_3d_config = {
        'img_size': (256, 256),  # 模型输入尺寸
        'output_size': [256, 256],  # 测试时的输出尺寸
        'device': 'cuda',  # 使用的设备，可选 'cuda' 或 'cpu'
        'use_auto_device': True,  # 是否自动检测设备
    }

    @classmethod
    def set_model(cls, model_name):
        """设置使用的模型"""
        if model_name not in cls.available_models:
            raise ValueError(f"模型必须是以下之一: {cls.available_models}")
        cls.network = model_name
        print(f"设置模型为: {model_name}")
        
        # 更新工作目录
        if hasattr(cls, 'current_fold') and cls.current_fold is not None:
            cls.work_dir = f'results/{cls.network}_{cls.datasets_name}_fold{cls.current_fold}'
        else:
            cls.work_dir = f'results/{cls.network}_{cls.datasets_name}'
    
    @classmethod
    def set_save_interval(cls, interval):
        """设置保存间隔"""
        if interval <= 0:
            raise ValueError("保存间隔必须大于0")
        cls.save_interval = interval
        print(f"设置保存间隔为每{interval}个epoch保存一次")
