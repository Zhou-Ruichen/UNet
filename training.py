#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强海底地形精化训练 - 完整版本

集成功能：
1. 多重损失函数组合（像素损失、梯度损失、频谱损失）
2. TID权重支持
3. WandB实验跟踪（可选）
4. 命令行接口
5. 完整的可视化输出
6. 改进的tqdm输出管理


使用方法:
python train.py --config baseline --epochs 20
python train.py --config baseline --epochs 20 --use_wandb  # 使用WandB
python train.py --config baseline --epochs 20 --quiet      # 安静模式


作者: 周瑞宸
日期: 2025-05-25
版本: v4.0 (完整集成版本)
"""

import argparse
import sys
import os
import platform
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
from tqdm import tqdm
import warnings

# 检查是否在subprocess中运行
RUNNING_IN_SUBPROCESS = not sys.stdout.isatty()

# 设置matplotlib后端（避免GUI问题）
if RUNNING_IN_SUBPROCESS:
    matplotlib.use('Agg')
matplotlib.use('Agg')

# 尝试导入wandb（可选）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB not available. Install with: pip install wandb")

warnings.filterwarnings('ignore')


def setup_chinese_font():
    """配置matplotlib支持中文字体"""
    system = platform.system()
    if system == 'Linux':
        plt.rcParams['font.sans-serif'] = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
            'SimHei', 'SimSun', 'DejaVu Sans'
        ]
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

setup_chinese_font()


def setup_logging(quiet=False):
    """设置日志配置"""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_model_training_complete_lt135km.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('enhanced_training_complete')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EnhancedTrainingConfig:
    """增强训练配置参数 - 支持多重损失函数"""
    # 数据路径
    DATA_DIR = "/mnt/data5/06-Projects/05-unet/output/1-refined_seafloor_data_preparation"
    
    FEATURES_PATH = os.path.join(DATA_DIR, "Features_SWOT_LT135km_Standardized.nc")
    LABELS_PATH = os.path.join(DATA_DIR, "Label_Target_Bathy_LT135km_Standardized.nc")
    TID_WEIGHTS_PATH = os.path.join(DATA_DIR, "TID_Loss_Weights.nc")

    # 输出路径
    OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-enhanced_multi_loss_outputs_lt135km"

    # 数据参数
    PATCH_SIZE = 128
    STRIDE = 32
    MIN_VALID_RATIO = 0.25

    # 数据集划分
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # 模型参数
    IN_CHANNELS = 4
    OUT_CHANNELS = 1
    INIT_FEATURES = 48

    # 训练参数
    BATCH_SIZE = 12
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 15
    WEIGHT_DECAY = 5e-5

    # 其他参数
    NUM_WORKERS = 4
    RANDOM_SEED = 42

    # 数据增强参数
    USE_DATA_AUGMENTATION = True
    AUGMENTATION_PROB = 0.2

    # 损失函数配置
    LOSS_CONFIG = {
        'loss_combination': 'all',
        'pixel_loss_weight': 1.0,
        'gradient_loss_weight': 0.3,
        'spectral_loss_weight': 0.2,
        'pixel_loss_type': 'mse',
        'use_tid_weights': True,
        'tid_weight_scale': 1.0,
        'gradient_method': 'sobel',
        'spectral_focus_high_freq': True,
        'high_freq_weight_factor': 2.0,
    }


class TIDWeightedLoss(nn.Module):
    """带TID权重的基础损失函数"""
    
    def __init__(self, loss_type='mse', use_tid_weights=True, tid_weight_scale=1.0):
        super(TIDWeightedLoss, self).__init__()
        self.loss_type = loss_type
        self.use_tid_weights = use_tid_weights
        self.tid_weight_scale = tid_weight_scale
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, pred, target, tid_weights=None):
        loss_map = self.base_loss(pred, target)
        
        if self.use_tid_weights and tid_weights is not None:
            weighted_loss_map = loss_map * (tid_weights * self.tid_weight_scale)
            valid_mask = (tid_weights > 0)
            if valid_mask.sum() > 0:
                loss = weighted_loss_map[valid_mask].mean()
            else:
                loss = loss_map.mean()
        else:
            loss = loss_map.mean()
        
        return loss


class GradientLoss(nn.Module):
    """梯度损失函数 - 增强边缘和细节感知"""
    
    def __init__(self, method='sobel'):
        super(GradientLoss, self).__init__()
        self.method = method
        
        if method == 'sobel':
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
            self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target, tid_weights=None):
        if self.method == 'sobel':
            sobel_x = self.sobel_x.to(pred.device)
            sobel_y = self.sobel_y.to(pred.device)
            
            pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
            
            target_grad_x = F.conv2d(target, sobel_x, padding=1)
            target_grad_y = F.conv2d(target, sobel_y, padding=1)
            
        elif self.method == 'diff':
            pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            
            target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
            target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x, reduction='none')
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y, reduction='none')
        
        if self.method == 'sobel':
            grad_loss = grad_loss_x + grad_loss_y
        else:
            if tid_weights is not None:
                tid_weights_x = tid_weights[:, :, :, 1:]
                tid_weights_y = tid_weights[:, :, 1:, :]
            
                weighted_grad_loss_x = grad_loss_x * tid_weights_x
                weighted_grad_loss_y = grad_loss_y * tid_weights_y
                
                valid_mask_x = (tid_weights_x > 0)
                valid_mask_y = (tid_weights_y > 0)
                
                loss_x = weighted_grad_loss_x[valid_mask_x].mean() if valid_mask_x.sum() > 0 else grad_loss_x.mean()
                loss_y = weighted_grad_loss_y[valid_mask_y].mean() if valid_mask_y.sum() > 0 else grad_loss_y.mean()
                
                return (loss_x + loss_y) / 2
            else:
                return (grad_loss_x.mean() + grad_loss_y.mean()) / 2
        
        if tid_weights is not None:
            weighted_grad_loss = grad_loss * tid_weights
            valid_mask = (tid_weights > 0)
            if valid_mask.sum() > 0:
                return weighted_grad_loss[valid_mask].mean()
            else:
                return grad_loss.mean()
        else:
            return grad_loss.mean()


class SpectralLoss(nn.Module):
    """频谱损失函数 - 直接约束频域表现"""
    
    def __init__(self, focus_high_freq=True, high_freq_weight_factor=2.0):
        super(SpectralLoss, self).__init__()
        self.focus_high_freq = focus_high_freq
        self.high_freq_weight_factor = high_freq_weight_factor
    
    def forward(self, pred, target, tid_weights=None):
        batch_size = pred.size(0)
        spectral_losses = []
        
        for i in range(batch_size):
            pred_single = pred[i, 0]
            target_single = target[i, 0]
            
            pred_fft = torch.fft.fft2(pred_single)
            target_fft = torch.fft.fft2(target_single)
            
            pred_magnitude = torch.abs(pred_fft)
            target_magnitude = torch.abs(target_fft)
            
            eps = 1e-8
            pred_log_magnitude = torch.log(pred_magnitude + eps)
            target_log_magnitude = torch.log(target_magnitude + eps)
            
            spectral_diff = F.mse_loss(pred_log_magnitude, target_log_magnitude, reduction='none')
            
            if self.focus_high_freq:
                H, W = spectral_diff.shape
                center_h, center_w = H // 2, W // 2
                
                y, x = torch.meshgrid(torch.arange(H, device=pred.device), 
                                    torch.arange(W, device=pred.device), indexing='ij')
                freq_distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
                
                max_distance = torch.sqrt(torch.tensor(center_h ** 2 + center_w ** 2, device=pred.device))
                freq_weights = 1.0 + (freq_distance / max_distance) * (self.high_freq_weight_factor - 1.0)
                
                spectral_diff = spectral_diff * freq_weights
            
            if tid_weights is not None:
                tid_weight_single = tid_weights[i, 0]
                tid_weight_factor = tid_weight_single.mean()
                spectral_diff = spectral_diff * tid_weight_factor
            
            normalized_loss = spectral_diff.mean() / (H * W)
            spectral_losses.append(normalized_loss)
        
        final_loss = torch.stack(spectral_losses).mean()
        return final_loss * 0.01


class MultiLossFunction(nn.Module):
    """多重损失函数组合"""
    
    def __init__(self, config: Dict):
        super(MultiLossFunction, self).__init__()
        self.config = config
        
        self.pixel_loss = TIDWeightedLoss(
            loss_type=config['pixel_loss_type'],
            use_tid_weights=config['use_tid_weights'],
            tid_weight_scale=config['tid_weight_scale']
        )
        
        self.gradient_loss = GradientLoss(method=config['gradient_method'])
        
        self.spectral_loss = SpectralLoss(
            focus_high_freq=config['spectral_focus_high_freq'],
            high_freq_weight_factor=config['high_freq_weight_factor']
        )
        
        self.pixel_weight = config['pixel_loss_weight']
        self.gradient_weight = config['gradient_loss_weight']
        self.spectral_weight = config['spectral_loss_weight']
        self.loss_combination = config['loss_combination']
    
    def forward(self, pred, target, tid_weights=None):
        losses = {}
        total_loss = 0.0
        
        # 像素级损失
        if self.loss_combination == 'baseline':
            pixel_loss = self.pixel_loss(pred, target, None)
        else:
            pixel_loss = self.pixel_loss(pred, target, tid_weights)
        
        losses['pixel'] = pixel_loss.item()
        total_loss += self.pixel_weight * pixel_loss
        
        # 梯度损失
        if self.loss_combination in ['weighted_mse_grad', 'all'] and self.gradient_weight > 0:
            gradient_loss = self.gradient_loss(pred, target, tid_weights)
            losses['gradient'] = gradient_loss.item()
            total_loss += self.gradient_weight * gradient_loss
        else:
            losses['gradient'] = 0.0
        
        # 频谱损失
        if self.loss_combination in ['weighted_mse_fft', 'all'] and self.spectral_weight > 0:
            spectral_loss = self.spectral_loss(pred, target, tid_weights)
            losses['spectral'] = spectral_loss.item()
            total_loss += self.spectral_weight * spectral_loss
        else:
            losses['spectral'] = 0.0
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


class EnhancedDataAugmentation:
    """增强数据增强类"""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, features, labels, tid_weights=None):
        if np.random.random() > self.prob:
            if tid_weights is not None:
                return features, labels, tid_weights
            else:
                return features, labels

        if np.random.random() > 0.5:
            features = torch.flip(features, dims=[2])
            labels = torch.flip(labels, dims=[2])
            if tid_weights is not None:
                tid_weights = torch.flip(tid_weights, dims=[2])
        else:
            features = torch.flip(features, dims=[1])
            labels = torch.flip(labels, dims=[1])
            if tid_weights is not None:
                tid_weights = torch.flip(tid_weights, dims=[1])

        if tid_weights is not None:
            return features, labels, tid_weights
        else:
            return features, labels


class EnhancedBathymetryDataset(Dataset):
    """增强海底地形数据集类 - 支持TID权重"""

    def __init__(self, patches_list: List[Dict], config, is_training=False):
        self.patches = patches_list
        self.config = config
        self.is_training = is_training

        if is_training and config.USE_DATA_AUGMENTATION:
            self.augmentation = EnhancedDataAugmentation(config.AUGMENTATION_PROB)
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]

        features = patch_info['features'].astype(np.float32)
        label = patch_info['label'].astype(np.float32)
        tid_weights = patch_info.get('tid_weights', None)

        if label.ndim == 2:
            label = label[np.newaxis, ...]
        
        if tid_weights is not None and tid_weights.ndim == 2:
            tid_weights = tid_weights[np.newaxis, ...]

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        
        if tid_weights is not None:
            tid_weights = np.nan_to_num(tid_weights, nan=0.0, posinf=0.0, neginf=0.0)

        features_tensor = torch.from_numpy(features)
        label_tensor = torch.from_numpy(label)
        
        if tid_weights is not None:
            tid_weights_tensor = torch.from_numpy(tid_weights)
        else:
            tid_weights_tensor = None

        if self.augmentation is not None:
            if tid_weights_tensor is not None:
                features_tensor, label_tensor, tid_weights_tensor = self.augmentation(
                    features_tensor, label_tensor, tid_weights_tensor
                )
            else:
                features_tensor, label_tensor = self.augmentation(features_tensor, label_tensor)

        if tid_weights_tensor is not None:
            return features_tensor, label_tensor, tid_weights_tensor
        else:
            return features_tensor, label_tensor


class OptimizedUNet(nn.Module):
    """优化的U-Net模型架构"""

    def __init__(self, in_channels=4, out_channels=1, init_features=48):
        super(OptimizedUNet, self).__init__()

        features = init_features

        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)

        self.decoder4 = self._block((features * 8) + features * 16, features * 8)
        self.decoder3 = self._block((features * 4) + features * 8, features * 4)
        self.decoder2 = self._block((features * 2) + features * 4, features * 2)
        self.decoder1 = self._block(features + features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = F.interpolate(bottleneck, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05)
        )


class EnhancedDataPreparation:
    """增强数据准备类 - 支持TID权重"""

    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def load_data(self) -> Tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
        logger = logging.getLogger('enhanced_training_complete')
        logger.info("加载数据（包括TID权重）...")

        for path, name in [(self.config.FEATURES_PATH, "特征文件"), 
                          (self.config.LABELS_PATH, "标签文件"),
                          (self.config.TID_WEIGHTS_PATH, "TID权重文件")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name}未找到: {path}")

        features_ds = xr.open_dataset(self.config.FEATURES_PATH)
        labels_da = xr.open_dataarray(self.config.LABELS_PATH)
        tid_weights_da = xr.open_dataarray(self.config.TID_WEIGHTS_PATH)

        logger.info(f"特征数据变量: {list(features_ds.data_vars)}")
        logger.info(f"标签数据形状: {labels_da.shape}")
        logger.info(f"TID权重数据形状: {tid_weights_da.shape}")

        return features_ds, labels_da, tid_weights_da

    def extract_patches(self, features_ds: xr.Dataset, labels_da: xr.DataArray, 
                       tid_weights_da: xr.DataArray) -> List[Dict]:
        logger = logging.getLogger('enhanced_training_complete')
        logger.info("提取数据块（包含TID权重）...")

        patches = []
        lat_size = len(labels_da.lat)
        lon_size = len(labels_da.lon)
        patch_size = self.config.PATCH_SIZE
        stride = self.config.STRIDE

        margin = patch_size // 8

        logger.info(f"数据网格大小: {lat_size} x {lon_size}")
        logger.info(f"数据块大小: {patch_size}, 步长: {stride}, 边距: {margin}")

        for lat_start in range(margin, lat_size - patch_size - margin + 1, stride):
            for lon_start in range(margin, lon_size - patch_size - margin + 1, stride):
                lat_end = lat_start + patch_size
                lon_end = lon_start + patch_size

                label_patch = labels_da.isel(
                    lat=slice(lat_start, lat_end),
                    lon=slice(lon_start, lon_end)
                ).values

                tid_weights_patch = tid_weights_da.isel(
                    lat=slice(lat_start, lat_end),
                    lon=slice(lon_start, lon_end)
                ).values

                valid_ratio = np.sum(~np.isnan(label_patch)) / label_patch.size
                if valid_ratio < self.config.MIN_VALID_RATIO:
                    continue

                label_std = np.nanstd(label_patch)
                if label_std < 1e-6:
                    continue

                valid_weight_ratio = np.sum(tid_weights_patch > 0) / tid_weights_patch.size
                if valid_weight_ratio < self.config.MIN_VALID_RATIO * 0.5:
                    continue

                feature_patches = []
                valid_features = True

                expected_vars = ['GA', 'DOV_EW', 'DOV_NS', 'VGG']
                
                for var_name in expected_vars:
                    if var_name in features_ds:
                        feature_patch = features_ds[var_name].isel(
                            lat=slice(lat_start, lat_end),
                            lon=slice(lon_start, lon_end)
                        ).values

                        feature_valid_ratio = np.sum(~np.isnan(feature_patch)) / feature_patch.size
                        if feature_valid_ratio < self.config.MIN_VALID_RATIO:
                            valid_features = False
                            break

                        feature_patches.append(feature_patch)
                    else:
                        logger.warning(f"未找到特征变量: {var_name}")
                        valid_features = False
                        break

                if not valid_features or len(feature_patches) != 4:
                    continue

                features_array = np.stack(feature_patches, axis=0)

                patches.append({
                    'features': features_array,
                    'label': label_patch,
                    'tid_weights': tid_weights_patch,
                    'lat_start': lat_start,
                    'lon_start': lon_start
                })

        logger.info(f"提取了 {len(patches)} 个有效数据块")
        return patches

    def split_patches(self, patches: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger = logging.getLogger('enhanced_training_complete')
        logger.info("进行数据集划分...")

        np.random.seed(self.config.RANDOM_SEED)

        indices = np.arange(len(patches))
        np.random.shuffle(indices)

        n_train = int(len(patches) * self.config.TRAIN_RATIO)
        n_val = int(len(patches) * self.config.VAL_RATIO)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_patches = [patches[i] for i in train_indices]
        val_patches = [patches[i] for i in val_indices]
        test_patches = [patches[i] for i in test_indices]

        logger.info(f"数据集划分: 训练={len(train_patches)}, 验证={len(val_patches)}, 测试={len(test_patches)}")

        return train_patches, val_patches, test_patches


class WandBTrainer:
    """集成WandB的训练器"""
    
    def __init__(self, model, config, use_wandb=False, project_name="seafloor-topography"):
        self.model = model.to(device)
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=f"{config.LOSS_CONFIG['loss_combination']}_experiment",
                config=dict(config.LOSS_CONFIG),
                tags=[config.LOSS_CONFIG['loss_combination']],
                notes=f"Loss combination: {config.LOSS_CONFIG['loss_combination']}"
            )
            wandb.watch(model, log="all", log_freq=100)
        
        self.criterion = MultiLossFunction(config.LOSS_CONFIG)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
        
        self.history = {
            'train_loss_total': [],
            'train_loss_pixel': [],
            'train_loss_gradient': [],
            'train_loss_spectral': [],
            'val_loss_total': [],
            'val_loss_pixel': [],
            'val_loss_gradient': [],
            'val_loss_spectral': [],
            'val_correlation': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
    
    def train_epoch(self, train_loader, quiet=False):
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'pixel': 0.0,
            'gradient': 0.0,
            'spectral': 0.0
        }
        
        if quiet or RUNNING_IN_SUBPROCESS:
            progress_bar = train_loader
        else:
            progress_bar = tqdm(train_loader, desc='训练中', leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            
            if len(batch_data) == 3:
                features, labels, tid_weights = batch_data
                features = features.to(device)
                labels = labels.to(device)
                tid_weights = tid_weights.to(device)
            else:
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
                tid_weights = None

            outputs = self.model(features)
            loss, loss_components = self.criterion(outputs, labels, tid_weights)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for key in epoch_losses:
                if key in loss_components:
                    epoch_losses[key] += loss_components[key]

            if not quiet and not RUNNING_IN_SUBPROCESS and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({
                    'loss': f'{loss_components["total"]:.4f}',
                    'pixel': f'{loss_components["pixel"]:.4f}',
                    'grad': f'{loss_components["gradient"]:.4f}',
                    'spec': f'{loss_components["spectral"]:.4f}'
                })
            
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'batch_loss': loss_components["total"],
                    'batch_pixel_loss': loss_components["pixel"],
                    'batch_gradient_loss': loss_components["gradient"],
                    'batch_spectral_loss': loss_components["spectral"],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

        n_batches = len(train_loader)
        return {key: val / n_batches for key, val in epoch_losses.items()}

    def validate(self, val_loader, quiet=False):
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'pixel': 0.0,
            'gradient': 0.0,
            'spectral': 0.0
        }

        all_predictions = []
        all_targets = []

        if quiet or RUNNING_IN_SUBPROCESS:
            progress_bar = val_loader
        else:
            progress_bar = tqdm(val_loader, desc='验证中', leave=False)

        with torch.no_grad():
            for batch_data in progress_bar:
                
                if len(batch_data) == 3:
                    features, labels, tid_weights = batch_data
                    features = features.to(device)
                    labels = labels.to(device)
                    tid_weights = tid_weights.to(device)
                else:
                    features, labels = batch_data
                    features = features.to(device)
                    labels = labels.to(device)
                    tid_weights = None

                outputs = self.model(features)
                loss, loss_components = self.criterion(outputs, labels, tid_weights)

                for key in epoch_losses:
                    if key in loss_components:
                        epoch_losses[key] += loss_components[key]

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

        n_batches = len(val_loader)
        results = {key: val / n_batches for key, val in epoch_losses.items()}
        results['correlation'] = correlation

        return results

    def train(self, train_loader, val_loader, quiet=False):
        logger = logging.getLogger('enhanced_training_complete')
        
        if not quiet:
            logger.info(f"开始训练 - 损失函数组合: {self.config.LOSS_CONFIG['loss_combination']}")

        for epoch in range(self.config.NUM_EPOCHS):
            if not quiet:
                logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            train_metrics = self.train_epoch(train_loader, quiet)
            val_metrics = self.validate(val_loader, quiet)

            self.scheduler.step(val_metrics['total'])
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss_total'].append(train_metrics['total'])
            self.history['train_loss_pixel'].append(train_metrics['pixel'])
            self.history['train_loss_gradient'].append(train_metrics['gradient'])
            self.history['train_loss_spectral'].append(train_metrics['spectral'])
            
            self.history['val_loss_total'].append(val_metrics['total'])
            self.history['val_loss_pixel'].append(val_metrics['pixel'])
            self.history['val_loss_gradient'].append(val_metrics['gradient'])
            self.history['val_loss_spectral'].append(val_metrics['spectral'])
            
            self.history['val_correlation'].append(val_metrics['correlation'])
            self.history['learning_rates'].append(current_lr)

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_total': train_metrics['total'],
                    'train_loss_pixel': train_metrics['pixel'],
                    'train_loss_gradient': train_metrics['gradient'],
                    'train_loss_spectral': train_metrics['spectral'],
                    'val_loss_total': val_metrics['total'],
                    'val_loss_pixel': val_metrics['pixel'],
                    'val_loss_gradient': val_metrics['gradient'],
                    'val_loss_spectral': val_metrics['spectral'],
                    'val_correlation': val_metrics['correlation'],
                    'learning_rate': current_lr
                })

            if not quiet:
                logger.info(f"训练 - 总损失: {train_metrics['total']:.6f}, 像素: {train_metrics['pixel']:.6f}, "
                          f"梯度: {train_metrics['gradient']:.6f}, 频谱: {train_metrics['spectral']:.6f}")
                logger.info(f"验证 - 总损失: {val_metrics['total']:.6f}, 像素: {val_metrics['pixel']:.6f}, "
                          f"梯度: {val_metrics['gradient']:.6f}, 频谱: {val_metrics['spectral']:.6f}, "
                          f"相关: {val_metrics['correlation']:.6f}")
            else:
                print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}: "
                      f"Train: {train_metrics['total']:.4f}, "
                      f"Val: {val_metrics['total']:.4f}, "
                      f"Corr: {val_metrics['correlation']:.4f}")

            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, val_metrics['total'])
                if not quiet:
                    logger.info("保存最佳模型")
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                if not quiet:
                    logger.info(f"早停触发，在epoch {epoch + 1}")
                break

        if not quiet:
            logger.info("训练完成!")
        
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'loss_config': self.config.LOSS_CONFIG,
            'history': self.history
        }

        path = os.path.join(self.config.OUTPUT_DIR, 
                           f'best_model_enhanced_multi_loss_{self.config.LOSS_CONFIG["loss_combination"]}_lt135km.pth')
        torch.save(checkpoint, path)


def create_training_visualization(trainer, config, output_dir):
    """创建训练可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(1, len(trainer.history['train_loss_total']) + 1)
    
    # 总损失曲线
    axes[0, 0].plot(epochs, trainer.history['train_loss_total'], 'b-', label='训练总损失', alpha=0.7)
    axes[0, 0].plot(epochs, trainer.history['val_loss_total'], 'r-', label='验证总损失', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('总损失')
    axes[0, 0].set_title('总损失变化')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 像素损失曲线
    axes[0, 1].plot(epochs, trainer.history['train_loss_pixel'], 'b-', label='训练像素损失', alpha=0.7)
    axes[0, 1].plot(epochs, trainer.history['val_loss_pixel'], 'r-', label='验证像素损失', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('像素损失')
    axes[0, 1].set_title('像素损失变化')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 梯度损失曲线
    axes[0, 2].plot(epochs, trainer.history['train_loss_gradient'], 'b-', label='训练梯度损失', alpha=0.7)
    axes[0, 2].plot(epochs, trainer.history['val_loss_gradient'], 'r-', label='验证梯度损失', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('梯度损失')
    axes[0, 2].set_title('梯度损失变化')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 频谱损失曲线
    axes[1, 0].plot(epochs, trainer.history['train_loss_spectral'], 'b-', label='训练频谱损失', alpha=0.7)
    axes[1, 0].plot(epochs, trainer.history['val_loss_spectral'], 'r-', label='验证频谱损失', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('频谱损失')
    axes[1, 0].set_title('频谱损失变化')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 相关系数
    axes[1, 1].plot(epochs, trainer.history['val_correlation'], 'g-', label='验证相关系数', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('相关系数')
    axes[1, 1].set_title('相关系数变化')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 2].plot(epochs, trainer.history['learning_rates'], 'orange', alpha=0.7)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('学习率')
    axes[1, 2].set_title('学习率变化')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    
    plt.suptitle(f'训练历史 - {config.LOSS_CONFIG["loss_combination"]} 损失组合', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'training_history_{config.LOSS_CONFIG["loss_combination"]}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def create_prediction_examples(model, test_loader, config, output_dir, device):
    """创建预测示例图"""
    model.eval()
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                features, labels, tid_weights = batch_data
                features = features.to(device)
                labels = labels.to(device)
            else:
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
            
            predictions = model(features)
            break
    
    n_examples = min(4, predictions.size(0))
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 5*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        pred = predictions[i, 0].cpu().numpy()
        true = labels[i, 0].cpu().numpy()
        error = pred - true
        
        im1 = axes[i, 0].imshow(true, cmap='RdBu_r', aspect='auto')
        axes[i, 0].set_title(f'真值 {i+1}')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(pred, cmap='RdBu_r', aspect='auto')
        axes[i, 1].set_title(f'预测 {i+1}')
        plt.colorbar(im2, ax=axes[i, 1])
        
        im3 = axes[i, 2].imshow(error, cmap='RdBu_r', aspect='auto')
        axes[i, 2].set_title(f'误差 {i+1}')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.suptitle(f'预测示例 - {config.LOSS_CONFIG["loss_combination"]}', fontsize=14)
    plt.tight_layout()
    
    examples_path = os.path.join(output_dir, f'prediction_examples_{config.LOSS_CONFIG["loss_combination"]}.png')
    plt.savefig(examples_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return examples_path


def create_scatter_plot(model, test_loader, config, output_dir, device):
    """创建散点图"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                features, labels, _ = batch_data
                features = features.to(device)
                labels = labels.to(device)
            else:
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
            
            predictions = model(features)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # 计算统计指标
    mse = np.mean((pred_flat - target_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - target_flat))
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # 创建散点图
    fig, ax = plt.subplots(figsize=(10, 10))
    
    n_samples = min(50000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_samples, replace=False)
    pred_sample = pred_flat[indices]
    target_sample = target_flat[indices]
    
    ax.scatter(target_sample, pred_sample, alpha=0.3, s=1, c='blue')
    
    min_val = min(target_sample.min(), pred_sample.min())
    max_val = max(target_sample.max(), pred_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测')
    
    ax.set_xlabel('真值')
    ax.set_ylabel('预测值')
    ax.set_title(f'预测 vs 真值散点图\n'
                f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, 相关系数: {correlation:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    scatter_path = os.path.join(output_dir, f'scatter_plot_{config.LOSS_CONFIG["loss_combination"]}.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return scatter_path, {'rmse': float(rmse), 'mae': float(mae), 'correlation': float(correlation)}


# 配置类定义
class BaselineConfig(EnhancedTrainingConfig):
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-baseline_outputs_lt135km"
        self.LOSS_CONFIG = {
            'loss_combination': 'baseline',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.0,
            'spectral_loss_weight': 0.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': False,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }

# 2. get_config函数
def get_config(config_name):
    config_map = {
        'baseline': BaselineConfig,
        'weighted_mse': WeightedMSEConfig,
        'mse_grad': MSEGradientConfig,
        'mse_fft': MSESpectralConfig,
        'all': AllCombinedConfig,
    }
    if config_name not in config_map:
        available_configs = list(config_map.keys())
        raise ValueError(f"未知配置: {config_name}. 可用配置: {available_configs}")
    return config_map[config_name]()

class WeightedMSEConfig(EnhancedTrainingConfig):
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-weighted_mse_outputs_lt135km"
        self.LOSS_CONFIG = {
            'loss_combination': 'weighted_mse',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.0,
            'spectral_loss_weight': 0.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }

class MSEGradientConfig(EnhancedTrainingConfig):
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-mse_grad_outputs_lt135km"
        self.LOSS_CONFIG = {
            'loss_combination': 'weighted_mse_grad',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.3,
            'spectral_loss_weight': 0.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }

class MSESpectralConfig(EnhancedTrainingConfig):
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-mse_fft_outputs_lt135km"
        self.LOSS_CONFIG = {
            'loss_combination': 'weighted_mse_fft',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.0,
            'spectral_loss_weight': 1.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }

class AllCombinedConfig(EnhancedTrainingConfig):
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-all_combined_outputs_lt135km"
        self.LOSS_CONFIG = {
            'loss_combination': 'all',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.3,
            'spectral_loss_weight': 1.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强海底地形精化训练 - 完整版本')
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline', 'weighted_mse', 'mse_grad', 'mse_fft', 'all'],
                        help='选择训练配置')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖默认的训练轮数')
    parser.add_argument('--use_wandb', action='store_true', help='使用WandB进行实验跟踪')
    parser.add_argument('--wandb_project', type=str, default='seafloor-topography', help='WandB项目名称')
    parser.add_argument('--quiet', action='store_true', help='安静模式，减少输出')

    
    args = parser.parse_args()
    
    logger = setup_logging(args.quiet)
    
    config = get_config(args.config)
    
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    
    if not args.quiet:
        logger.info(f"使用配置: {args.config}")
        logger.info(f"损失函数组合: {config.LOSS_CONFIG['loss_combination']}")
        logger.info(f"输出目录: {config.OUTPUT_DIR}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.quiet:
        logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 数据准备
    data_prep = EnhancedDataPreparation(config)

    try:
        # 加载数据
        features_ds, labels_da, tid_weights_da = data_prep.load_data()
        patches = data_prep.extract_patches(features_ds, labels_da, tid_weights_da)

        if len(patches) == 0:
            logger.error("没有提取到有效的数据块！")
            return

        # 划分数据集
        train_patches, val_patches, test_patches = data_prep.split_patches(patches)

        # 创建数据集
        train_dataset = EnhancedBathymetryDataset(train_patches, config, is_training=True)
        val_dataset = EnhancedBathymetryDataset(val_patches, config, is_training=False)
        test_dataset = EnhancedBathymetryDataset(test_patches, config, is_training=False)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                                num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                              num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                                num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())

        # 创建模型
        model = OptimizedUNet(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            init_features=config.INIT_FEATURES
        )

        total_params = sum(p.numel() for p in model.parameters())
        if not args.quiet:
            logger.info(f"模型总参数数量: {total_params:,}")

        # 创建训练器
        trainer = WandBTrainer(model, config, use_wandb=args.use_wandb, project_name=args.wandb_project)
        
        # 训练模型
        trainer.train(train_loader, val_loader, quiet=args.quiet)

        # 创建可视化
        if not args.quiet:
            logger.info("创建可视化...")
        
        plot_path = create_training_visualization(trainer, config, config.OUTPUT_DIR)
        examples_path = create_prediction_examples(model, test_loader, config, config.OUTPUT_DIR, device)
        scatter_path, test_metrics = create_scatter_plot(model, test_loader, config, config.OUTPUT_DIR, device)

        # 保存实验摘要
        experiment_summary = {
            'experiment_name': f'enhanced_multi_loss_{config.LOSS_CONFIG["loss_combination"]}',
            'loss_config': config.LOSS_CONFIG,
            'model_config': {
                'in_channels': config.IN_CHANNELS,
                'out_channels': config.OUT_CHANNELS,
                'init_features': config.INIT_FEATURES
            },
            'training_config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'num_epochs': len(trainer.history['train_loss_total']),
                'early_stopping_patience': config.EARLY_STOPPING_PATIENCE
            },
            'best_val_loss': float(trainer.best_val_loss),  # 确保转换为Python float
            'total_params': int(total_params),  # 确保转换为Python int
            'trainable_params': int(total_params),  # 确保转换为Python int
            'test_metrics': {
                'rmse': float(test_metrics['rmse']),  # 转换为Python float
                'mae': float(test_metrics['mae']),  # 转换为Python float
                'correlation': float(test_metrics['correlation'])  # 转换为Python float
            },
            'use_wandb': args.use_wandb and WANDB_AVAILABLE
        }

        summary_path = os.path.join(config.OUTPUT_DIR, 
                                f'experiment_summary_{config.LOSS_CONFIG["loss_combination"]}_lt135km.json')
        with open(summary_path, 'w') as f:
            json.dump(experiment_summary, f, indent=2)

        # 输出结果摘要
        if not args.quiet:
            logger.info(f"✅ 训练完成！损失组合: {config.LOSS_CONFIG['loss_combination']}")
            logger.info(f"📊 最佳验证损失: {trainer.best_val_loss:.6f}")
            logger.info(f"📈 测试RMSE: {test_metrics['rmse']:.6f}")
            logger.info(f"📈 测试相关系数: {test_metrics['correlation']:.6f}")
            logger.info(f"📁 结果保存在: {config.OUTPUT_DIR}")
        else:
            print(f"✅ {args.config}: Val={trainer.best_val_loss:.4f}, RMSE={test_metrics['rmse']:.4f}, Corr={test_metrics['correlation']:.4f}")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()