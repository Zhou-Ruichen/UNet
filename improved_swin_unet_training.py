#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的Swin U-Net训练 - 支持损失函数配置选择

新增功能：
1. 支持多种预设损失配置（baseline、TID加权、梯度增强）
2. 移除FFT损失选项（根据用户要求）
3. 优化训练参数和监控
4. 简化的配置选择界面

使用方法:
python improved_swin_unet_training.py --loss_config baseline
python improved_swin_unet_training.py --loss_config tid_weighted
python improved_swin_unet_training.py --loss_config tid_gradient
python improved_swin_unet_training.py --loss_config enhanced_gradient

作者: 周瑞宸
日期: 2025-05-26
版本: v1.2 (损失函数配置化)
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import logging
from datetime import datetime
from tqdm import tqdm

# 导入现有模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_training_multi_loss import (
    EnhancedTrainingConfig, EnhancedDataPreparation,
    EnhancedBathymetryDataset, MultiLossFunction,
    device, setup_chinese_font
)

# 导入Swin U-Net模型（从用户提供的代码）
from swin_unet_training import (
    SwinUNet, SwinTransformerBlock, BasicLayer, PatchMerging,
    WindowAttention, DecoderBlock, DropPath, drop_path,
    window_partition, window_reverse
)

setup_chinese_font()
logger = logging.getLogger('improved_swin_unet')


# Swin U-Net损失函数配置预设
SWIN_LOSS_CONFIGS = {
    'baseline': {
        'name': 'Baseline',
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
        'description': '标准MSE损失，不使用权重和额外损失'
    },
    
    'tid_weighted': {
        'name': 'TID加权',
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
        'description': '使用TID权重的MSE损失，重点关注高质量区域'
    },
    
    'tid_gradient': {
        'name': 'TID权重+梯度',
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
        'description': 'TID权重MSE + 梯度损失，增强细节和边缘'
    },
    
    'enhanced_gradient': {
        'name': '增强梯度',
        'loss_combination': 'weighted_mse_grad',
        'pixel_loss_weight': 1.0,
        'gradient_loss_weight': 0.5,
        'spectral_loss_weight': 0.0,
        'pixel_loss_type': 'mse',
        'use_tid_weights': True,
        'tid_weight_scale': 1.2,
        'gradient_method': 'sobel',
        'spectral_focus_high_freq': True,
        'high_freq_weight_factor': 2.5,
        'description': '增强的梯度损失权重，最大化细节恢复能力'
    }
}


class ImprovedSwinUNetConfig(EnhancedTrainingConfig):
    """改进的Swin U-Net配置 - 支持损失函数选择"""
    
    def __init__(self, loss_config_name='tid_gradient'):
        super().__init__()
        
        # 验证损失配置
        if loss_config_name not in SWIN_LOSS_CONFIGS:
            available = list(SWIN_LOSS_CONFIGS.keys())
            raise ValueError(f"未知的损失配置: {loss_config_name}. 可用配置: {available}")
        
        # 基础配置
        self.OUTPUT_DIR = f"/mnt/data5/06-Projects/05-unet/output/3-swin_{loss_config_name}_outputs_lt135km"
        
        # 应用选择的损失配置
        loss_config = SWIN_LOSS_CONFIGS[loss_config_name]
        self.LOSS_CONFIG = loss_config.copy()
        
        # Swin Transformer特定参数
        self.SWIN_CONFIG = {
            'patch_size': 4,
            'in_chans': 4,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'mlp_ratio': 4.0,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'use_checkpoint': False,
            'decoder_channels': [512, 256, 128, 64],
            'skip_connection_type': 'concat',
            'final_upsample': 'expand_first',
        }
        
        # 训练参数优化
        self.BATCH_SIZE = 6  # 适合Swin U-Net的批次大小
        self.LEARNING_RATE = 1e-4
        self.NUM_EPOCHS = 150
        self.WEIGHT_DECAY = 0.05
        self.EARLY_STOPPING_PATIENCE = 20
        
        # 保存配置信息
        self.loss_config_name = loss_config_name
        self.loss_config_description = loss_config['description']
    
    def to_dict(self):
        """转换为字典用于保存"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }


class ImprovedSwinUNetTrainer:
    """改进的Swin U-Net训练器"""
    
    def __init__(self, model: nn.Module, config: ImprovedSwinUNetConfig):
        self.model = model.to(device)
        self.config = config
        
        # 使用现有的多重损失函数
        self.criterion = MultiLossFunction(config.LOSS_CONFIG)
        
        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.NUM_EPOCHS, 
            eta_min=config.LEARNING_RATE * 0.01
        )
        
        # 训练历史
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
        self.best_val_correlation = -1.0
        self.early_stopping_counter = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0, 
            'pixel': 0.0, 
            'gradient': 0.0, 
            'spectral': 0.0
        }
        
        progress_bar = tqdm(train_loader, desc=f'训练 Epoch {epoch}', leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # 获取数据
            features = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            tid_weights = batch_data[2].to(device) if len(batch_data) == 3 and self.config.LOSS_CONFIG['use_tid_weights'] else None
            
            # 前向传播
            outputs = self.model(features)
            
            # 确保输出尺寸匹配
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            # 计算损失
            loss, loss_components = self.criterion(outputs, labels, tid_weights)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            for key in epoch_losses:
                if key in loss_components:
                    value = loss_components[key]
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key] += value.item()
                    elif isinstance(value, (int, float)):
                        epoch_losses[key] += value
            
            # 更新进度条
            postfix_dict = {}
            for k, v in loss_components.items():
                if k in ['total', 'pixel', 'gradient']:
                    if isinstance(v, torch.Tensor):
                        postfix_dict[k] = f'{v.item():.4f}'
                    elif isinstance(v, (int, float)):
                        postfix_dict[k] = f'{v:.4f}'
            progress_bar.set_postfix(postfix_dict)
        
        # 计算平均损失
        n_batches = len(train_loader)
        return {key: val / n_batches if n_batches > 0 else 0.0 for key, val in epoch_losses.items()}
    
    def validate(self, val_loader: DataLoader) -> dict:
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
            for batch_data in tqdm(val_loader, desc='验证中', leave=False):
                features = batch_data[0].to(device)
                labels = batch_data[1].to(device)
                tid_weights = batch_data[2].to(device) if len(batch_data) == 3 and self.config.LOSS_CONFIG['use_tid_weights'] else None
                
                # 预测
                outputs = self.model(features)
                
                # 确保尺寸匹配
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = F.interpolate(outputs, size=labels.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                
                # 计算损失
                loss, loss_components = self.criterion(outputs, labels, tid_weights)
                
                # 记录损失
                for key in epoch_losses:
                    if key in loss_components:
                        value = loss_components[key]
                        if isinstance(value, torch.Tensor):
                            epoch_losses[key] += value.item()
                        elif isinstance(value, (int, float)):
                            epoch_losses[key] += value
                
                # 收集预测结果
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        # 计算相关系数
        correlation = 0.0
        if all_predictions and all_targets:
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            if len(pred_flat) > 1 and len(target_flat) > 1:
                correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
        
        # 计算平均损失
        n_batches = len(val_loader)
        results = {key: val / n_batches if n_batches > 0 else 0.0 for key, val in epoch_losses.items()}
        results['correlation'] = correlation
        
        return results
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        logger.info(f"开始改进的Swin U-Net训练")
        logger.info(f"损失配置: {self.config.loss_config_name} - {self.config.loss_config_description}")
        
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新历史记录
            self.history['train_loss_total'].append(train_metrics.get('total', 0.0))
            self.history['train_loss_pixel'].append(train_metrics.get('pixel', 0.0))
            self.history['train_loss_gradient'].append(train_metrics.get('gradient', 0.0))
            self.history['train_loss_spectral'].append(train_metrics.get('spectral', 0.0))
            
            self.history['val_loss_total'].append(val_metrics.get('total', 0.0))
            self.history['val_loss_pixel'].append(val_metrics.get('pixel', 0.0))
            self.history['val_loss_gradient'].append(val_metrics.get('gradient', 0.0))
            self.history['val_loss_spectral'].append(val_metrics.get('spectral', 0.0))
            
            self.history['val_correlation'].append(val_metrics.get('correlation', 0.0))
            self.history['learning_rates'].append(current_lr)
            
            # 打印指标
            train_log = f"总损失: {train_metrics.get('total', 0):.6f}"
            if train_metrics.get('pixel', 0) > 0:
                train_log += f", 像素: {train_metrics.get('pixel', 0):.6f}"
            if train_metrics.get('gradient', 0) > 0:
                train_log += f", 梯度: {train_metrics.get('gradient', 0):.6f}"
            
            val_log = f"总损失: {val_metrics.get('total', 0):.6f}, 相关系数: {val_metrics.get('correlation', 0):.6f}"
            
            logger.info(f"训练 - {train_log}")
            logger.info(f"验证 - {val_log}")
            logger.info(f"学习率: {current_lr:.6e}")
            
            # 保存最佳模型（基于验证损失）
            val_loss = val_metrics.get('total', float('inf'))
            val_correlation = val_metrics.get('correlation', 0.0)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_correlation = val_correlation
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, val_loss, val_correlation)
                logger.info(f"✓ 保存最佳模型 (验证损失: {val_loss:.6f}, 相关系数: {val_correlation:.6f})")
            else:
                self.early_stopping_counter += 1
            
            # 早停检查
            if (self.config.EARLY_STOPPING_PATIENCE > 0 and 
                self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE):
                logger.info(f"早停触发，在epoch {epoch}")
                break
        
        logger.info("改进的Swin U-Net训练完成!")
        logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")
        logger.info(f"最佳验证相关系数: {self.best_val_correlation:.6f}")
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_correlation: float):
        """保存检查点"""
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config).copy()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_correlation': val_correlation,
            'config_dict': config_dict,
            'history': self.history
        }
        
        path = os.path.join(self.config.OUTPUT_DIR, f'best_swin_unet_{self.config.loss_config_name}.pth')
        torch.save(checkpoint, path)


def list_available_configs():
    """列出所有可用的Swin U-Net损失配置"""
    print("=== 可用的Swin U-Net损失配置 ===")
    for name, config in SWIN_LOSS_CONFIGS.items():
        use_tid = "✓" if config['use_tid_weights'] else "✗"
        grad_weight = config['gradient_loss_weight']
        print(f"\n{name}:")
        print(f"  名称: {config['name']}")
        print(f"  描述: {config['description']}")
        print(f"  TID权重: {use_tid}")
        print(f"  梯度损失权重: {grad_weight}")


def main():
    parser = argparse.ArgumentParser(description='改进的Swin U-Net训练')
    parser.add_argument('--loss_config', type=str, default='tid_gradient',
                        choices=list(SWIN_LOSS_CONFIGS.keys()),
                        help='选择损失函数配置')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--list_configs', action='store_true',
                        help='列出所有可用配置')
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_available_configs()
        return
    
    # 创建配置
    config = ImprovedSwinUNetConfig(args.loss_config)
    
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    logger.info("=== 改进的Swin U-Net海底地形精化训练 ===")
    logger.info(f"损失配置: {config.loss_config_name} - {config.loss_config_description}")
    logger.info(f"运行设备: {device}")
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
        # 加载数据
        features_ds, labels_da, tid_weights_da = data_prep.load_data()
        
        if features_ds is None:
            logger.error("数据加载失败")
            return
        
        # 提取数据块
        patches = data_prep.extract_patches(features_ds, labels_da, tid_weights_da)
        
        if not patches:
            logger.error("没有提取到有效的数据块！")
            return
        
        logger.info(f"提取到 {len(patches)} 个数据块")
        
        # 划分数据集
        train_patches, val_patches, _ = data_prep.split_patches(patches)
        
        if not train_patches or not val_patches:
            logger.error("数据集划分失败！")
            return
        
        # 创建数据集和加载器
        train_dataset = EnhancedBathymetryDataset(train_patches, config, is_training=True)
        val_dataset = EnhancedBathymetryDataset(val_patches, config, is_training=False)
        
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
        model = SwinUNet(config.SWIN_CONFIG)
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        # 训练模型
        trainer = ImprovedSwinUNetTrainer(model, config)
        trainer.train(train_loader, val_loader)
        
        # 保存实验摘要
        experiment_summary = {
            'experiment_name': f'improved_swin_unet_{config.loss_config_name}',
            'loss_config_name': config.loss_config_name,
            'loss_config_description': config.loss_config_description,
            'model_type': 'Swin U-Net',
            'swin_config': config.SWIN_CONFIG,
            'loss_config': config.LOSS_CONFIG,
            'training_config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'num_epochs_run': len(trainer.history['train_loss_total']),
                'target_num_epochs': config.NUM_EPOCHS,
                'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR'
            },
            'best_val_loss': float(trainer.best_val_loss) if trainer.best_val_loss != float('inf') else None,
            'best_val_correlation': float(trainer.best_val_correlation),
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'final_epoch_completed': len(trainer.history['train_loss_total'])
        }
        
        summary_path = os.path.join(config.OUTPUT_DIR, f'improved_swin_summary_{config.loss_config_name}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        history_path = os.path.join(config.OUTPUT_DIR, f'improved_swin_history_{config.loss_config_name}.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(trainer.history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 改进的Swin U-Net训练完成！")
        logger.info(f"📊 最佳验证损失: {trainer.best_val_loss:.6f}")
        logger.info(f"📊 最佳验证相关系数: {trainer.best_val_correlation:.6f}")
        logger.info(f"📁 结果保存在: {config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()