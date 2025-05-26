#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强海底地形精化训练 - 多重损失函数版本 (适配0-135km波段数据)

基于前版本的经验，引入多种损失函数组合以优化高频细节恢复：
1. 带TID权重的MSE/MAE损失（基线改进）
2. 梯度损失（Gradient Loss）- 增强边缘和细节感知
3. 频率域损失（Spectral Loss）- 直接约束特定频率表现

主要特性:
- 支持多种损失函数组合的实验配置
- 集成TID权重掩膜用于质量加权训练
- 可配置的损失函数权重系数
- 详细的训练监控和可视化

作者: 周瑞宸
日期: 2025-05-25
版本: v3.2 (多重损失函数集成)
"""

import os
import platform
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
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


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_model_training_multi_loss_lt135km.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enhanced_multi_loss_training_lt135km')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")


class EnhancedTrainingConfig:
    """增强训练配置参数 - 支持多重损失函数"""
    # 数据路径
    DATA_DIR = "/mnt/data5/06-Projects/05-unet/output/1-refined_seafloor_data_preparation"
    
    FEATURES_PATH = os.path.join(DATA_DIR, "Features_SWOT_LT135km_Standardized.nc")
    LABELS_PATH = os.path.join(DATA_DIR, "Label_Target_Bathy_LT135km_Standardized.nc")
    TID_WEIGHTS_PATH = os.path.join(DATA_DIR, "TID_Loss_Weights.nc")  # 新增TID权重路径

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

    # *** 新增：损失函数配置 ***
    LOSS_CONFIG = {
        # 损失函数组合选择（可选：'weighted_mse', 'weighted_mse_grad', 'weighted_mse_fft', 'all'）
        'loss_combination': 'all',
        
        # 各损失函数权重
        'pixel_loss_weight': 1.0,      # 像素级损失权重（MSE/MAE）
        'gradient_loss_weight': 0.3,    # 梯度损失权重
        'spectral_loss_weight': 0.2,    # 频谱损失权重
        
        # 像素损失类型（'mse' 或 'mae'）
        'pixel_loss_type': 'mse',
        
        # TID权重相关
        'use_tid_weights': True,        # 是否使用TID权重
        'tid_weight_scale': 1.0,        # TID权重缩放因子
        
        # 梯度损失配置
        'gradient_method': 'sobel',     # 梯度计算方法（'sobel' 或 'diff'）
        
        # 频谱损失配置
        'spectral_focus_high_freq': True,  # 是否重点关注高频
        'high_freq_weight_factor': 2.0,    # 高频权重因子
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
        """
        计算带权重的损失
        
        参数:
            pred: 预测值 (B, C, H, W)
            target: 目标值 (B, C, H, W)
            tid_weights: TID权重 (B, C, H, W) 或 None
        """
        # 计算基础损失
        loss_map = self.base_loss(pred, target)
        
        if self.use_tid_weights and tid_weights is not None:
            # 应用TID权重
            weighted_loss_map = loss_map * (tid_weights * self.tid_weight_scale)
            
            # 计算有效区域的平均损失
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
            # Sobel算子
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
            self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target, tid_weights=None):
        """
        计算梯度损失
        
        参数:
            pred: 预测值 (B, C, H, W)
            target: 目标值 (B, C, H, W)
            tid_weights: TID权重 (B, C, H, W) 或 None
        """
        if self.method == 'sobel':
            # 确保Sobel卷积核在正确的设备上
            sobel_x = self.sobel_x.to(pred.device)
            sobel_y = self.sobel_y.to(pred.device)
            
            # 使用Sobel算子计算梯度
            pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
            
            target_grad_x = F.conv2d(target, sobel_x, padding=1)
            target_grad_y = F.conv2d(target, sobel_y, padding=1)
            
        elif self.method == 'diff':
            # 使用差分计算梯度
            pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            
            target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
            target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # 计算梯度差异
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x, reduction='none')
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y, reduction='none')
        
        # 合并x和y方向的梯度损失
        if self.method == 'sobel':
            grad_loss = grad_loss_x + grad_loss_y
        else:
            # 对于差分方法，需要处理尺寸不一致
            # 裁剪tid_weights以匹配梯度尺寸
            if tid_weights is not None:
                tid_weights_x = tid_weights[:, :, :, 1:]  # 匹配x梯度尺寸
                tid_weights_y = tid_weights[:, :, 1:, :]  # 匹配y梯度尺寸
            
            # 分别计算加权损失
            if tid_weights is not None:
                weighted_grad_loss_x = grad_loss_x * tid_weights_x
                weighted_grad_loss_y = grad_loss_y * tid_weights_y
                
                valid_mask_x = (tid_weights_x > 0)
                valid_mask_y = (tid_weights_y > 0)
                
                loss_x = weighted_grad_loss_x[valid_mask_x].mean() if valid_mask_x.sum() > 0 else grad_loss_x.mean()
                loss_y = weighted_grad_loss_y[valid_mask_y].mean() if valid_mask_y.sum() > 0 else grad_loss_y.mean()
                
                return (loss_x + loss_y) / 2
            else:
                return (grad_loss_x.mean() + grad_loss_y.mean()) / 2
        
        # 对于Sobel方法，应用TID权重
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
        """
        计算频谱损失
        
        参数:
            pred: 预测值 (B, C, H, W)
            target: 目标值 (B, C, H, W)
            tid_weights: TID权重 (B, C, H, W) 或 None
        """
        # 对每个batch项分别计算FFT
        batch_size = pred.size(0)
        spectral_losses = []
        
        for i in range(batch_size):
            pred_single = pred[i, 0]  # (H, W)
            target_single = target[i, 0]  # (H, W)
            
            # 计算2D FFT
            pred_fft = torch.fft.fft2(pred_single)
            target_fft = torch.fft.fft2(target_single)
            
            # 计算振幅谱并进行对数变换以减少动态范围
            pred_magnitude = torch.abs(pred_fft)
            target_magnitude = torch.abs(target_fft)
            
            # 添加小的常数避免log(0)，然后取对数
            eps = 1e-8
            pred_log_magnitude = torch.log(pred_magnitude + eps)
            target_log_magnitude = torch.log(target_magnitude + eps)
            
            # 计算频谱差异
            spectral_diff = F.mse_loss(pred_log_magnitude, target_log_magnitude, reduction='none')
            
            if self.focus_high_freq:
                # 创建高频权重掩膜
                H, W = spectral_diff.shape
                center_h, center_w = H // 2, W // 2
                
                # 创建频率权重（距离中心越远权重越高）
                y, x = torch.meshgrid(torch.arange(H, device=pred.device), 
                                    torch.arange(W, device=pred.device), indexing='ij')
                freq_distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
                
                # 归一化频率距离并应用权重
                max_distance = torch.sqrt(torch.tensor(center_h ** 2 + center_w ** 2, device=pred.device))
                freq_weights = 1.0 + (freq_distance / max_distance) * (self.high_freq_weight_factor - 1.0)
                
                # 应用频率权重
                spectral_diff = spectral_diff * freq_weights
            
            # 如果有TID权重，可以进一步加权（需要从空间域转换到频域权重）
            if tid_weights is not None:
                tid_weight_single = tid_weights[i, 0]  # (H, W)
                # 简单起见，使用空间权重的平均值作为频域权重因子
                tid_weight_factor = tid_weight_single.mean()
                spectral_diff = spectral_diff * tid_weight_factor
            
            # 归一化：除以patch大小，使损失与patch尺寸无关
            normalized_loss = spectral_diff.mean() / (H * W)
            spectral_losses.append(normalized_loss)
        
        final_loss = torch.stack(spectral_losses).mean()
        
        # 进一步缩放到合理范围，使其与MSE损失量级相当
        return final_loss * 0.01  # 缩放因子，可根据实验调整


class MultiLossFunction(nn.Module):
    """多重损失函数组合"""
    
    def __init__(self, config: Dict):
        super(MultiLossFunction, self).__init__()
        self.config = config
        
        # 初始化各个损失函数
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
        
        # 损失权重
        self.pixel_weight = config['pixel_loss_weight']
        self.gradient_weight = config['gradient_loss_weight']
        self.spectral_weight = config['spectral_loss_weight']
        
        # 损失组合类型
        self.loss_combination = config['loss_combination']
    
    def forward(self, pred, target, tid_weights=None):
        """
        计算组合损失
        
        参数:
            pred: 预测值 (B, C, H, W)
            target: 目标值 (B, C, H, W)
            tid_weights: TID权重 (B, C, H, W) 或 None
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 像素级损失（总是计算）
        # 对于baseline配置，强制不使用TID权重
        if self.loss_combination == 'baseline':
            pixel_loss = self.pixel_loss(pred, target, None)  # baseline不使用TID权重
        else:
            pixel_loss = self.pixel_loss(pred, target, tid_weights)
        
        losses['pixel'] = pixel_loss.item()
        total_loss += self.pixel_weight * pixel_loss
        
        # 2. 梯度损失
        if self.loss_combination in ['weighted_mse_grad', 'all'] and self.gradient_weight > 0:
            gradient_loss = self.gradient_loss(pred, target, tid_weights)
            losses['gradient'] = gradient_loss.item()
            total_loss += self.gradient_weight * gradient_loss
        else:
            losses['gradient'] = 0.0
        
        # 3. 频谱损失
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
        """应用轻量级数据增强（包括TID权重）"""
        if np.random.random() > self.prob:
            if tid_weights is not None:
                return features, labels, tid_weights
            else:
                return features, labels

        # 随机选择翻转方向
        if np.random.random() > 0.5:
            # 水平翻转
            features = torch.flip(features, dims=[2])
            labels = torch.flip(labels, dims=[2])
            if tid_weights is not None:
                tid_weights = torch.flip(tid_weights, dims=[2])
        else:
            # 垂直翻转
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

        # 轻量级数据增强
        if is_training and config.USE_DATA_AUGMENTATION:
            self.augmentation = EnhancedDataAugmentation(config.AUGMENTATION_PROB)
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]

        # 加载数据
        features = patch_info['features'].astype(np.float32)
        label = patch_info['label'].astype(np.float32)
        tid_weights = patch_info.get('tid_weights', None)  # 新增TID权重

        # 确保维度正确
        if label.ndim == 2:
            label = label[np.newaxis, ...]
        
        if tid_weights is not None and tid_weights.ndim == 2:
            tid_weights = tid_weights[np.newaxis, ...]

        # 简化的NaN处理
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        
        if tid_weights is not None:
            tid_weights = np.nan_to_num(tid_weights, nan=0.0, posinf=0.0, neginf=0.0)

        # 转换为张量
        features_tensor = torch.from_numpy(features)
        label_tensor = torch.from_numpy(label)
        
        if tid_weights is not None:
            tid_weights_tensor = torch.from_numpy(tid_weights)
        else:
            tid_weights_tensor = None

        # 数据增强
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

        # 编码器路径
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈层
        self.bottleneck = self._block(features * 8, features * 16)

        # 解码器路径
        self.decoder4 = self._block((features * 8) + features * 16, features * 8)
        self.decoder3 = self._block((features * 4) + features * 8, features * 4)
        self.decoder2 = self._block((features * 2) + features * 4, features * 2)
        self.decoder1 = self._block(features + features * 2, features)

        # 输出层
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool4(enc4))

        # 解码器
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
        """优化的双卷积块"""
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
        """加载数据 - 包括TID权重"""
        logger.info("加载数据（包括TID权重）...")

        # 检查文件是否存在
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
        """增强数据块提取 - 包含TID权重"""
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

                # 提取标签块
                label_patch = labels_da.isel(
                    lat=slice(lat_start, lat_end),
                    lon=slice(lon_start, lon_end)
                ).values

                # 提取TID权重块
                tid_weights_patch = tid_weights_da.isel(
                    lat=slice(lat_start, lat_end),
                    lon=slice(lon_start, lon_end)
                ).values

                # 质量控制
                valid_ratio = np.sum(~np.isnan(label_patch)) / label_patch.size
                if valid_ratio < self.config.MIN_VALID_RATIO:
                    continue

                # 检查数据变化范围
                label_std = np.nanstd(label_patch)
                if label_std < 1e-6:
                    continue

                # 检查TID权重的有效性
                valid_weight_ratio = np.sum(tid_weights_patch > 0) / tid_weights_patch.size
                if valid_weight_ratio < self.config.MIN_VALID_RATIO * 0.5:  # 权重要求可以稍微放松
                    continue

                # 提取特征块
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

                # 堆叠特征通道
                features_array = np.stack(feature_patches, axis=0)

                patches.append({
                    'features': features_array,
                    'label': label_patch,
                    'tid_weights': tid_weights_patch,  # 新增TID权重
                    'lat_start': lat_start,
                    'lon_start': lon_start
                })

        logger.info(f"提取了 {len(patches)} 个有效数据块")
        return patches

    def split_patches(self, patches: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """空间感知的数据集划分"""
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


class EnhancedTrainer:
    """增强模型训练器 - 支持多重损失函数"""

    def __init__(self, model: nn.Module, config: EnhancedTrainingConfig):
        self.model = model.to(device)
        self.config = config

        # 多重损失函数
        self.criterion = MultiLossFunction(config.LOSS_CONFIG)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )

        # 训练历史 - 扩展以包含各种损失
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

        # 最佳模型状态
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'pixel': 0.0,
            'gradient': 0.0,
            'spectral': 0.0
        }

        progress_bar = tqdm(train_loader, desc='训练中')
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

            # 前向传播
            outputs = self.model(features)
            loss, loss_components = self.criterion(outputs, labels, tid_weights)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录损失
            for key in epoch_losses:
                if key in loss_components:
                    epoch_losses[key] += loss_components[key]

            progress_bar.set_postfix({
                'total': f'{loss_components["total"]:.4f}',
                'pixel': f'{loss_components["pixel"]:.4f}',
                'grad': f'{loss_components["gradient"]:.4f}',
                'spec': f'{loss_components["spectral"]:.4f}'
            })

        n_batches = len(train_loader)
        return {key: val / n_batches for key, val in epoch_losses.items()}

    def validate(self, val_loader: DataLoader) -> Dict:
        """验证模型"""
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'pixel': 0.0,
            'gradient': 0.0,
            'spectral': 0.0
        }

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc='验证中'):
                
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

                # 记录损失
                for key in epoch_losses:
                    if key in loss_components:
                        epoch_losses[key] += loss_components[key]

                # 收集预测结果
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        # 计算相关系数
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

        n_batches = len(val_loader)
        results = {key: val / n_batches for key, val in epoch_losses.items()}
        results['correlation'] = correlation

        return results

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        logger.info(f"开始增强训练 - 损失函数组合: {self.config.LOSS_CONFIG['loss_combination']}")

        for epoch in range(self.config.NUM_EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step(val_metrics['total'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
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

            # 打印指标
            logger.info(f"训练 - 总损失: {train_metrics['total']:.6f}, 像素: {train_metrics['pixel']:.6f}, "
                      f"梯度: {train_metrics['gradient']:.6f}, 频谱: {train_metrics['spectral']:.6f}")
            logger.info(f"验证 - 总损失: {val_metrics['total']:.6f}, 像素: {val_metrics['pixel']:.6f}, "
                      f"梯度: {val_metrics['gradient']:.6f}, 频谱: {val_metrics['spectral']:.6f}, "
                      f"相关: {val_metrics['correlation']:.6f}")
            logger.info(f"学习率: {current_lr:.6f}")

            # 保存最佳模型
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, val_metrics['total'])
                logger.info("保存最佳模型")
            else:
                self.early_stopping_counter += 1

            # 早停检查
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"早停触发，在epoch {epoch + 1}")
                break

            # 定期保存训练曲线
            if (epoch + 1) % 10 == 0:
                self.plot_training_history()

        logger.info("训练完成!")

    def save_checkpoint(self, epoch: int, val_loss: float):
        """保存模型检查点"""
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

    def plot_training_history(self):
        """绘制增强训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        epochs = range(1, len(self.history['train_loss_total']) + 1)

        # 总损失曲线
        axes[0, 0].plot(epochs, self.history['train_loss_total'], 'b-', label='训练总损失', alpha=0.7)
        axes[0, 0].plot(epochs, self.history['val_loss_total'], 'r-', label='验证总损失', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('总损失')
        axes[0, 0].set_title('总损失变化')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 像素损失曲线
        axes[0, 1].plot(epochs, self.history['train_loss_pixel'], 'b-', label='训练像素损失', alpha=0.7)
        axes[0, 1].plot(epochs, self.history['val_loss_pixel'], 'r-', label='验证像素损失', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('像素损失')
        axes[0, 1].set_title('像素损失变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 梯度损失曲线
        axes[0, 2].plot(epochs, self.history['train_loss_gradient'], 'b-', label='训练梯度损失', alpha=0.7)
        axes[0, 2].plot(epochs, self.history['val_loss_gradient'], 'r-', label='验证梯度损失', alpha=0.7)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('梯度损失')
        axes[0, 2].set_title('梯度损失变化')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 频谱损失曲线
        axes[1, 0].plot(epochs, self.history['train_loss_spectral'], 'b-', label='训练频谱损失', alpha=0.7)
        axes[1, 0].plot(epochs, self.history['val_loss_spectral'], 'r-', label='验证频谱损失', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('频谱损失')
        axes[1, 0].set_title('频谱损失变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 相关系数
        axes[1, 1].plot(epochs, self.history['val_correlation'], 'g-', label='验证相关系数', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('相关系数')
        axes[1, 1].set_title('相关系数变化')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 学习率曲线
        axes[1, 2].plot(epochs, self.history['learning_rates'], 'orange', alpha=0.7)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('学习率')
        axes[1, 2].set_title('学习率变化')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')

        plt.suptitle(f'增强训练历史 - {self.config.LOSS_CONFIG["loss_combination"]} 损失组合', fontsize=14)
        plt.tight_layout()
        
        filename = f'enhanced_training_history_{self.config.LOSS_CONFIG["loss_combination"]}_lt135km.png'
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, filename), dpi=300)
        plt.close()


def main():
    """增强主函数 - 支持多重损失函数实验"""
    config = EnhancedTrainingConfig()

    logger.info("=== 增强海底地形精化训练 - 多重损失函数版本 ===")
    logger.info(f"损失函数组合: {config.LOSS_CONFIG['loss_combination']}")
    logger.info(f"数据目录: {config.DATA_DIR}")
    logger.info(f"输出目录: {config.OUTPUT_DIR}")

    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 数据准备
    data_prep = EnhancedDataPreparation(config)

    try:
        # 加载数据（包括TID权重）
        features_ds, labels_da, tid_weights_da = data_prep.load_data()

        # 提取数据块（包括TID权重）
        patches = data_prep.extract_patches(features_ds, labels_da, tid_weights_da)

        if len(patches) == 0:
            logger.error("没有提取到有效的数据块！请检查数据质量和参数设置。")
            return

        # 划分数据集
        train_patches, val_patches, test_patches = data_prep.split_patches(patches)

        # 创建数据集
        train_dataset = EnhancedBathymetryDataset(train_patches, config, is_training=True)
        val_dataset = EnhancedBathymetryDataset(val_patches, config, is_training=False)
        test_dataset = EnhancedBathymetryDataset(test_patches, config, is_training=False)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        # 创建模型
        model = OptimizedUNet(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            init_features=config.INIT_FEATURES
        )

        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数数量: {total_params:,}")
        logger.info(f"可训练参数数量: {trainable_params:,}")

        # 训练模型
        trainer = EnhancedTrainer(model, config)
        trainer.train(train_loader, val_loader)

        # 保存实验配置和结果摘要
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
            'best_val_loss': trainer.best_val_loss,
            'total_params': total_params,
            'trainable_params': trainable_params
        }

        summary_path = os.path.join(config.OUTPUT_DIR, 
                                  f'experiment_summary_{config.LOSS_CONFIG["loss_combination"]}_lt135km.json')
        with open(summary_path, 'w') as f:
            json.dump(experiment_summary, f, indent=2)

        logger.info(f"增强训练完成！损失组合: {config.LOSS_CONFIG['loss_combination']}")
        logger.info(f"最佳验证损失: {trainer.best_val_loss:.6f}")
        logger.info(f"结果保存在: {config.OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()