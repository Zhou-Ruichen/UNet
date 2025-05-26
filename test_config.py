#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置测试脚本 - 验证baseline配置问题是否解决

使用方法:
python test_config.py --config baseline
python test_config.py --config weighted_mse

作者: 周瑞宸
日期: 2025-05-25
"""

import argparse
import sys
import os

# 将父目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_training_multi_loss import EnhancedTrainingConfig


class BaselineConfig(EnhancedTrainingConfig):
    """基线配置 - 仅使用标准MSE损失（不使用TID权重）"""
    
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-baseline_outputs_lt135km"
        
        self.LOSS_CONFIG = {
            'loss_combination': 'baseline',  # 正确设置为baseline
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.0,
            'spectral_loss_weight': 0.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': False,     # 不使用TID权重
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }


class WeightedMSEConfig(EnhancedTrainingConfig):
    """带TID权重的MSE损失配置"""
    
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


def get_config(config_name):
    """根据配置名称获取配置对象"""
    config_map = {
        'baseline': BaselineConfig,
        'weighted_mse': WeightedMSEConfig,
    }
    
    if config_name not in config_map:
        available_configs = list(config_map.keys())
        raise ValueError(f"未知配置: {config_name}. 可用配置: {available_configs}")
    
    return config_map[config_name]()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='配置测试脚本')
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline', 'weighted_mse'],
                        help='选择测试配置')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.config)
    
    print(f"测试配置: {args.config}")
    print(f"损失函数组合: {config.LOSS_CONFIG['loss_combination']}")
    print(f"使用TID权重: {config.LOSS_CONFIG['use_tid_weights']}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(f"数据目录: {config.DATA_DIR}")
    
    # 验证路径
    print(f"\n路径验证:")
    print(f"输出目录是否包含正确路径: {'05-unet' in config.OUTPUT_DIR}")
    print(f"数据目录是否包含正确路径: {'05-unet' in config.DATA_DIR}")
    
    # 验证baseline配置
    if args.config == 'baseline':
        print(f"\nBaseline配置验证:")
        print(f"损失组合应为'baseline': {config.LOSS_CONFIG['loss_combination'] == 'baseline'}")
        print(f"不使用TID权重: {not config.LOSS_CONFIG['use_tid_weights']}")
        print(f"不使用梯度损失: {config.LOSS_CONFIG['gradient_loss_weight'] == 0}")
        print(f"不使用频谱损失: {config.LOSS_CONFIG['spectral_loss_weight'] == 0}")


if __name__ == "__main__":
    main()