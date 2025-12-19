from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import torch.nn.functional as F


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 几何增强：随机旋转/翻转 + 轻微缩放
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 随机缩放（0.9~1.1），保持CT解剖结构的合理性
        scale = random.uniform(0.9, 1.1)
        if abs(scale - 1.0) > 1e-3:
            x, y = image.shape
            image_scaled = zoom(image, (scale, scale), order=3)
            label_scaled = zoom(label, (scale, scale), order=0)

            # 记录实际缩放后的大小，避免整数截断导致的broadcast错误
            new_h, new_w = image_scaled.shape

            # 再padding回至少目标分辨率，后面仍会统一resize到output_size
            pad_h = max(self.output_size[0], new_h)
            pad_w = max(self.output_size[1], new_w)
            pad_img = np.zeros((pad_h, pad_w), dtype=image_scaled.dtype)
            pad_lab = np.zeros((pad_h, pad_w), dtype=label_scaled.dtype)
            pad_img[:new_h, :new_w] = image_scaled
            pad_lab[:new_h, :new_w] = label_scaled
            image, label = pad_img, pad_lab
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # 轻微强度扰动：高斯噪声 + Gamma校正，控制在小范围
        image = image.astype(np.float32)
        # 归一化到[0,1]以便做强度变换
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_norm = (image - img_min) / (img_max - img_min)
        else:
            image_norm = image

        # 高斯噪声（std 0.01~0.03）
        if random.random() < 0.5:
            noise_std = random.uniform(0.01, 0.03)
            noise = np.random.normal(0.0, noise_std, size=image_norm.shape).astype(np.float32)
            image_norm = np.clip(image_norm + noise, 0.0, 1.0)

        # Gamma校正（0.9~1.1），模拟对比度变化
        if random.random() < 0.5:
            gamma = random.uniform(0.9, 1.1)
            image_norm = np.power(image_norm, gamma)

        # 恢复到原始强度范围，后续再做伪HDR三通道
        image = image_norm * (img_max - img_min) + img_min
        
        # 伪HDR三通道预处理
        image_tensor = torch.from_numpy(image.astype(np.float32))
        
        # 创建三个通道：原始、增强对比度、平滑
        channel1 = image_tensor  # 原始图像
        channel2 = torch.clamp(image_tensor * 1.2, 0, 1)  # 增强对比度
        channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                               kernel_size=3, stride=1, padding=1).squeeze()  # 平滑
        
        # 合并为三通道
        image = torch.stack([channel1, channel2, channel3], dim=0)
        
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        # 为训练split预计算每个slice的器官信息，用于简单的小器官重采样策略
        self.has_small_organ = None
        if self.split == "train":
            small_organ_labels = {4, 5, 8}  # 例如胆囊(4)、食管/小器官(5)和胰腺/主动脉(8)等
            self.has_small_organ = []
            for name in self.sample_list:
                slice_name = name.strip('\n')
                data_path = os.path.join(self.data_dir, slice_name + '.npz')
                if not os.path.exists(data_path):
                    self.has_small_organ.append(False)
                    continue
                try:
                    data = np.load(data_path)
                    label = data['label']
                    unique_labels = np.unique(label)
                    # 是否包含任一“小器官”标签
                    self.has_small_organ.append(len(set(unique_labels.tolist()) & small_organ_labels) > 0)
                except Exception:
                    self.has_small_organ.append(False)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 训练阶段：优先重采样包含小器官的slice
        if self.split == "train" and self.has_small_organ is not None:
            if np.random.rand() < 0.5:  # 50% 概率重采样小器官slice
                small_indices = [i for i, flag in enumerate(self.has_small_organ) if flag]
                if small_indices:
                    idx = np.random.choice(small_indices)

        if self.split in ["train", "test_slice"]:
            # 训练与 slice 验证均使用 .npz 文件（位于 train_npz）
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"找不到切片数据文件: {data_path}")
            data = np.load(data_path)
            image, label = data['image'], data['label']
        elif self.split in ["test_vol", "val_vol"]:
            # 体数据验证/测试使用 .npy.h5 文件（位于 test_vol_h5）
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, f"{vol_name}.npy.h5")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"找不到体数据文件: {filepath}")
            with h5py.File(filepath, 'r') as data:
                image, label = data['image'][:], data['label'][:]
        else:
            raise ValueError(f"不支持的split类型: {self.split}")

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    