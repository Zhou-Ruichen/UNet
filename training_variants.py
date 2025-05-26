#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练配置变体 - 修正版本

修正了以下问题：
1. baseline配置显示错误的损失函数组合
2. 所有配置的输出路径错误

作者: 周瑞宸
日期: 2025-05-25
版本: 修正版
"""

import argparse
import sys
import os

# 将父目录添加到Python路径，以便导入主训练模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_training_multi_loss import EnhancedTrainingConfig
import logging

logger = logging.getLogger('training_variants')


class BaselineConfig(EnhancedTrainingConfig):
    """基线配置 - 仅使用标准MSE损失（不使用TID权重）"""
    
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-baseline_outputs_lt135km"
        
        self.LOSS_CONFIG = {
            'loss_combination': 'baseline',  # 正确设置为baseline
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.0,  # 不使用梯度损失
            'spectral_loss_weight': 0.0,  # 不使用频谱损失
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
            'use_tid_weights': True,      # 使用TID权重
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }


class MSEGradientConfig(EnhancedTrainingConfig):
    """MSE + 梯度损失配置"""
    
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-mse_grad_outputs_lt135km"
        
        self.LOSS_CONFIG = {
            'loss_combination': 'weighted_mse_grad',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.3,  # 使用梯度损失
            'spectral_loss_weight': 0.0,
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',   # 推荐使用Sobel算子
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }


class MSESpectralConfig(EnhancedTrainingConfig):
    """MSE + 频谱损失配置"""
    
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-mse_fft_outputs_lt135km"
        
        self.LOSS_CONFIG = {
            'loss_combination': 'weighted_mse_fft',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.0,
            'spectral_loss_weight': 1.0,  # 使用频谱损失（已归一化）
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,  # 重点关注高频
            'high_freq_weight_factor': 2.0,    # 高频权重因子
        }


class AllCombinedConfig(EnhancedTrainingConfig):
    """全组合损失配置 - MSE + 梯度 + 频谱"""
    
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-all_combined_outputs_lt135km"
        
        self.LOSS_CONFIG = {
            'loss_combination': 'all',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.3,
            'spectral_loss_weight': 1.0,  # 更新频谱权重
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 1.0,
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 2.0,
        }


class ConservativeConfig(EnhancedTrainingConfig):
    """保守配置 - 较小的损失权重，适合初期实验"""
    
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-conservative_outputs_lt135km"
        
        self.LOSS_CONFIG = {
            'loss_combination': 'all',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.1,  # 较小的梯度权重
            'spectral_loss_weight': 0.5,  # 更新频谱权重
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 0.5,      # 较小的TID权重缩放
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 1.5,  # 较温和的高频权重
        }


class AggressiveConfig(EnhancedTrainingConfig):
    """激进配置 - 较大的损失权重，强调高频恢复"""
    
    def __init__(self):
        super().__init__()
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/2-aggressive_outputs_lt135km"
        
        self.LOSS_CONFIG = {
            'loss_combination': 'all',
            'pixel_loss_weight': 1.0,
            'gradient_loss_weight': 0.5,  # 较大的梯度权重
            'spectral_loss_weight': 1.5,  # 更新频谱权重
            'pixel_loss_type': 'mse',
            'use_tid_weights': True,
            'tid_weight_scale': 2.0,      # 较大的TID权重缩放
            'gradient_method': 'sobel',
            'spectral_focus_high_freq': True,
            'high_freq_weight_factor': 3.0,  # 强调高频
        }


def get_config(config_name):
    """根据配置名称获取配置对象"""
    config_map = {
        'baseline': BaselineConfig,
        'weighted_mse': WeightedMSEConfig,
        'mse_grad': MSEGradientConfig,
        'mse_fft': MSESpectralConfig,
        'all_combined': AllCombinedConfig,
        'conservative': ConservativeConfig,
        'aggressive': AggressiveConfig,
    }
    
    if config_name not in config_map:
        available_configs = list(config_map.keys())
        raise ValueError(f"未知配置: {config_name}. 可用配置: {available_configs}")
    
    return config_map[config_name]()


def main():
    """主函数 - 支持命令行参数选择配置"""
    parser = argparse.ArgumentParser(description='海底地形精化训练 - 多种损失函数配置')
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline', 'weighted_mse', 'mse_grad', 'mse_fft', 
                                'all_combined', 'conservative', 'aggressive'],
                        help='选择训练配置')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖默认的训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖默认的批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='覆盖默认的学习率')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.config)
    
    # 应用命令行参数覆盖
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    
    logger.info(f"使用配置: {args.config}")
    logger.info(f"损失函数组合: {config.LOSS_CONFIG['loss_combination']}")
    logger.info(f"输出目录: {config.OUTPUT_DIR}")
    
    # 直接调用enhanced_training_multi_loss模块
    import enhanced_training_multi_loss
    
    # 创建一个临时的主函数，使用我们的配置
    def run_training_with_config():
        """使用指定配置运行训练"""
        import torch
        import numpy as np
        
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
        data_prep = enhanced_training_multi_loss.EnhancedDataPreparation(config)

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
            train_dataset = enhanced_training_multi_loss.EnhancedBathymetryDataset(train_patches, config, is_training=True)
            val_dataset = enhanced_training_multi_loss.EnhancedBathymetryDataset(val_patches, config, is_training=False)

            # 创建数据加载器
            from torch.utils.data import DataLoader
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

            # 创建模型
            model = enhanced_training_multi_loss.OptimizedUNet(
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
            trainer = enhanced_training_multi_loss.EnhancedTrainer(model, config)
            trainer.train(train_loader, val_loader)

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
                'best_val_loss': trainer.best_val_loss,
                'total_params': total_params,
                'trainable_params': trainable_params
            }

            import json
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
    
    # 运行训练
    run_training_with_config()


if __name__ == "__main__":
    main()