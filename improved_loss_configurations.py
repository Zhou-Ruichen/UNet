#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的损失函数配置管理 - 统一GAN和U-Net的损失函数选择

提供多种预定义的损失函数组合，支持：
1. Baseline配置（标准MSE）
2. TID加权配置 
3. 梯度增强配置
4. GAN专用配置（平衡的权重设置）
5. 自定义配置

作者: 周瑞宸
日期: 2025-05-26
版本: v1.0
"""

import os
import logging
from typing import Dict, Any
from enhanced_training_multi_loss import EnhancedTrainingConfig
from conditional_gan_bathymetry import ConditionalGANConfig

logger = logging.getLogger(__name__)


class LossConfigManager:
    """损失函数配置管理器"""
    
    # U-Net损失函数配置预设
    UNET_LOSS_PRESETS = {
        'baseline': {
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
        },
        
        'tid_weighted': {
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
        },
        
        'tid_gradient': {
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
        },
        
        'enhanced_gradient': {
            'loss_combination': 'weighted_mse_grad',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.5,  # 增强梯度权重
            'spectral_loss_weight': 0.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.2,  # 增强TID权重
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }
    }
    
    # GAN损失函数配置预设
    GAN_LOSS_PRESETS = {
        'balanced_gan': {
            'adversarial_weight': 1.0,
            'reconstruction_weight': 10.0,  # 大幅降低
            'perceptual_weight': 5.0,       # 降低
            'gradient_weight': 3.0,         # 降低
            'spectral_weight': 0.0,         # 关闭FFT
            'n_discriminator_steps': 1,
            'n_generator_steps': 1,
            'discriminator_delay': 5,
            'label_smoothing': 0.1,
        },
        
        'reconstruction_focused': {
            'adversarial_weight': 0.5,
            'reconstruction_weight': 20.0,  # 重建为主
            'perceptual_weight': 2.0,
            'gradient_weight': 1.0,
            'spectral_weight': 0.0,
            'n_discriminator_steps': 1,
            'n_generator_steps': 2,        # 生成器多训练
            'discriminator_delay': 8,      # 延迟更长
            'label_smoothing': 0.1,
        },
        
        'adversarial_focused': {
            'adversarial_weight': 2.0,      # 增强对抗
            'reconstruction_weight': 5.0,   # 减少重建
            'perceptual_weight': 8.0,       # 增强感知
            'gradient_weight': 5.0,
            'spectral_weight': 0.0,
            'n_discriminator_steps': 2,     # 判别器多训练
            'n_generator_steps': 1,
            'discriminator_delay': 3,       # 更早开始
            'label_smoothing': 0.05,        # 减少平滑
        },
        
        'detail_enhanced': {
            'adversarial_weight': 1.5,
            'reconstruction_weight': 8.0,
            'perceptual_weight': 12.0,      # 高权重感知损失
            'gradient_weight': 8.0,         # 高权重梯度损失
            'spectral_weight': 0.0,
            'n_discriminator_steps': 1,
            'n_generator_steps': 1,
            'discriminator_delay': 5,
            'label_smoothing': 0.1,
        }
    }
    
    @classmethod
    def get_unet_config(cls, config_name: str, base_config_class=None) -> EnhancedTrainingConfig:
        """获取U-Net损失配置"""
        if config_name not in cls.UNET_LOSS_PRESETS:
            available = list(cls.UNET_LOSS_PRESETS.keys())
            raise ValueError(f"未知的U-Net配置: {config_name}. 可用配置: {available}")
        
        if base_config_class is None:
            base_config_class = EnhancedTrainingConfig
        
        config = base_config_class()
        config.LOSS_CONFIG = cls.UNET_LOSS_PRESETS[config_name].copy()
        
        # 根据配置调整输出目录
        config.OUTPUT_DIR = f"/mnt/data5/06-Projects/05-unet/output/2-{config_name}_outputs_lt135km"
        
        return config
    
    @classmethod
    def get_gan_config(cls, config_name: str) -> 'ImprovedConditionalGANConfig':
        """获取GAN损失配置"""
        if config_name not in cls.GAN_LOSS_PRESETS:
            available = list(cls.GAN_LOSS_PRESETS.keys())
            raise ValueError(f"未知的GAN配置: {config_name}. 可用配置: {available}")
        
        config = ImprovedConditionalGANConfig()
        
        # 更新GAN特定配置
        preset = cls.GAN_LOSS_PRESETS[config_name]
        config.GAN_CONFIG.update(preset)
        
        # 同步到LOSS_CONFIG（为了兼容性）
        config.LOSS_CONFIG.update({
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'pixel_loss_type': 'mse',
            'gradient_method': 'sobel',
        })
        
        # 调整输出目录
        config.OUTPUT_DIR = f"/mnt/data5/06-Projects/05-unet/output/3-gan_{config_name}_outputs_lt135km"
        
        return config
    
    @classmethod
    def list_available_configs(cls):
        """列出所有可用配置"""
        print("=== 可用的U-Net损失配置 ===")
        for name, config in cls.UNET_LOSS_PRESETS.items():
            use_tid = "✓" if config['use_tid_weights'] else "✗"
            use_grad = "✓" if config['gradient_loss_weight'] > 0 else "✗"
            print(f"  {name:15s} | TID权重: {use_tid} | 梯度损失: {use_grad}")
        
        print("\n=== 可用的GAN损失配置 ===")
        for name, config in cls.GAN_LOSS_PRESETS.items():
            adv_w = config['adversarial_weight']
            rec_w = config['reconstruction_weight']
            print(f"  {name:18s} | 对抗: {adv_w:4.1f} | 重建: {rec_w:4.1f}")


class ImprovedConditionalGANConfig(ConditionalGANConfig):
    """改进的条件GAN配置"""
    
    def __init__(self):
        super().__init__()
        
        # 使用更平衡的默认配置
        self.GAN_CONFIG = {
            # 学习率配置
            'generator_lr': 2e-4,
            'generator_beta1': 0.5,
            'generator_beta2': 0.999,
            'discriminator_lr': 2e-4,
            'discriminator_beta1': 0.5,
            'discriminator_beta2': 0.999,
            
            # 平衡的损失权重
            'adversarial_weight': 1.0,
            'reconstruction_weight': 10.0,  # 从100降到10
            'perceptual_weight': 5.0,       # 从10降到5
            'gradient_weight': 3.0,         # 从10降到3
            'spectral_weight': 0.0,         # 关闭FFT
            
            # 训练策略
            'n_discriminator_steps': 1,
            'n_generator_steps': 1,
            'discriminator_delay': 5,
            
            # 正则化
            'gradient_penalty_weight': 10.0,
            'spectral_norm': True,
            'label_smoothing': 0.1,
        }


class ImprovedSwinUNetConfig:
    """改进的Swin U-Net配置类"""
    
    def __init__(self, loss_config_name='tid_gradient'):
        # 导入Swin U-Net配置
        from swin_unet_training import SwinUNetConfig
        
        # 基础配置
        self.base_config = SwinUNetConfig()
        
        # 应用指定的损失配置
        if loss_config_name in LossConfigManager.UNET_LOSS_PRESETS:
            self.base_config.LOSS_CONFIG = LossConfigManager.UNET_LOSS_PRESETS[loss_config_name].copy()
            self.base_config.OUTPUT_DIR = f"/mnt/data5/06-Projects/05-unet/output/3-swin_{loss_config_name}_outputs_lt135km"
        else:
            logger.warning(f"未知损失配置 {loss_config_name}，使用默认配置")
    
    def get_config(self):
        return self.base_config


def create_training_script_with_loss_selection():
    """创建支持损失函数选择的训练脚本"""
    
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一训练脚本 - 支持损失函数配置选择

使用方法:
python unified_training.py --model unet --loss_config baseline
python unified_training.py --model unet --loss_config tid_weighted  
python unified_training.py --model unet --loss_config tid_gradient
python unified_training.py --model gan --loss_config balanced_gan
python unified_training.py --model swin --loss_config tid_gradient
"""

import argparse
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_loss_configurations import LossConfigManager
from enhanced_training_multi_loss import EnhancedTrainer, EnhancedDataPreparation, OptimizedUNet
from conditional_gan_bathymetry import ConditionalGANTrainer, EnhancedUNetGenerator, MultiScalePatchDiscriminator


def main():
    parser = argparse.ArgumentParser(description='统一训练脚本')
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet', 'gan', 'swin'],
                        help='模型类型')
    parser.add_argument('--loss_config', type=str, required=True,
                        help='损失函数配置名称')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖默认值）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖默认值）')
    parser.add_argument('--list_configs', action='store_true',
                        help='列出所有可用配置')
    
    args = parser.parse_args()
    
    if args.list_configs:
        LossConfigManager.list_available_configs()
        return
    
    # 根据模型类型和损失配置创建训练
    if args.model == 'unet':
        config = LossConfigManager.get_unet_config(args.loss_config)
        if args.epochs: config.NUM_EPOCHS = args.epochs
        if args.batch_size: config.BATCH_SIZE = args.batch_size
        
        # 创建标准U-Net训练
        # ... (具体实现)
        
    elif args.model == 'gan':
        config = LossConfigManager.get_gan_config(args.loss_config)
        if args.epochs: config.NUM_EPOCHS = args.epochs
        if args.batch_size: config.BATCH_SIZE = args.batch_size
        
        # 创建GAN训练
        # ... (具体实现)
        
    elif args.model == 'swin':
        from improved_loss_configurations import ImprovedSwinUNetConfig
        swin_config_wrapper = ImprovedSwinUNetConfig(args.loss_config)
        config = swin_config_wrapper.get_config()
        if args.epochs: config.NUM_EPOCHS = args.epochs
        if args.batch_size: config.BATCH_SIZE = args.batch_size
        
        # 创建Swin U-Net训练
        # ... (具体实现)


if __name__ == "__main__":
    main()
'''
    
    return script_content


# 使用示例和测试
if __name__ == "__main__":
    # 展示所有可用配置
    print("测试损失函数配置管理器...")
    LossConfigManager.list_available_configs()
    
    print("\n=== 测试U-Net配置 ===")
    baseline_config = LossConfigManager.get_unet_config('baseline')
    print(f"Baseline配置: {baseline_config.LOSS_CONFIG}")
    
    print("\n=== 测试GAN配置 ===") 
    balanced_gan_config = LossConfigManager.get_gan_config('balanced_gan')
    print(f"平衡GAN配置: {balanced_gan_config.GAN_CONFIG}")
    
    print("\n配置管理器测试完成！")