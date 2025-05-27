#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强训练脚本 - 集成WandB和改进输出管理

新增功能：
1. WandB实验跟踪（可选）
2. 改进的tqdm输出管理
3. 更清洁的日志输出
4. 更好的subprocess兼容性

使用方法:
python enhanced_training_wandb.py --config baseline --epochs 20
python enhanced_training_wandb.py --config baseline --epochs 20 --use_wandb  # 使用WandB
python enhanced_training_wandb.py --config baseline --epochs 20 --quiet      # 安静模式

作者: 周瑞宸
日期: 2025-05-25
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
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
import matplotlib
matplotlib.use('Agg')

# 将父目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_training_multi_loss import EnhancedTrainingConfig
import enhanced_training_multi_loss

# 尝试导入wandb（可选）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB not available. Install with: pip install wandb")

# 设置日志
def setup_logging(quiet=False):
    """设置日志配置"""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('enhanced_training_wandb')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class WandBTrainer:
    """集成WandB的训练器"""
    
    def __init__(self, model, config, use_wandb=False, project_name="seafloor-topography"):
        self.model = model.to(enhanced_training_multi_loss.device)
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # 初始化WandB
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=f"{config.LOSS_CONFIG['loss_combination']}_experiment",
                config=dict(config.LOSS_CONFIG),
                tags=[config.LOSS_CONFIG['loss_combination']],
                notes=f"Loss combination: {config.LOSS_CONFIG['loss_combination']}"
            )
            wandb.watch(model, log="all", log_freq=100)
        
        # 损失函数和优化器
        self.criterion = enhanced_training_multi_loss.MultiLossFunction(config.LOSS_CONFIG)
        self.optimizer = enhanced_training_multi_loss.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = enhanced_training_multi_loss.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
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
        self.early_stopping_counter = 0
    
    def train_epoch(self, train_loader, quiet=False):
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'pixel': 0.0,
            'gradient': 0.0,
            'spectral': 0.0
        }
        
        # 设置进度条
        if quiet or RUNNING_IN_SUBPROCESS:
            # 在安静模式或subprocess中使用简化的进度显示
            progress_bar = train_loader
            desc = '训练中'
        else:
            progress_bar = tqdm(train_loader, desc='训练中', leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            
            if len(batch_data) == 3:
                features, labels, tid_weights = batch_data
                features = features.to(enhanced_training_multi_loss.device)
                labels = labels.to(enhanced_training_multi_loss.device)
                tid_weights = tid_weights.to(enhanced_training_multi_loss.device)
            else:
                features, labels = batch_data
                features = features.to(enhanced_training_multi_loss.device)
                labels = labels.to(enhanced_training_multi_loss.device)
                tid_weights = None

            # 前向传播
            outputs = self.model(features)
            loss, loss_components = self.criterion(outputs, labels, tid_weights)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 记录损失
            for key in epoch_losses:
                if key in loss_components:
                    epoch_losses[key] += loss_components[key]

            # 更新进度条（只在非安静模式下）
            if not quiet and not RUNNING_IN_SUBPROCESS and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({
                    'loss': f'{loss_components["total"]:.4f}',
                    'pixel': f'{loss_components["pixel"]:.4f}',
                    'grad': f'{loss_components["gradient"]:.4f}',
                    'spec': f'{loss_components["spectral"]:.4f}'
                })
            
            # WandB实时日志
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

        # 设置进度条
        if quiet or RUNNING_IN_SUBPROCESS:
            progress_bar = val_loader
        else:
            progress_bar = tqdm(val_loader, desc='验证中', leave=False)

        with torch.no_grad():
            for batch_data in progress_bar:
                
                if len(batch_data) == 3:
                    features, labels, tid_weights = batch_data
                    features = features.to(enhanced_training_multi_loss.device)
                    labels = labels.to(enhanced_training_multi_loss.device)
                    tid_weights = tid_weights.to(enhanced_training_multi_loss.device)
                else:
                    features, labels = batch_data
                    features = features.to(enhanced_training_multi_loss.device)
                    labels = labels.to(enhanced_training_multi_loss.device)
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

    def train(self, train_loader, val_loader, quiet=False):
        """完整训练流程"""
        logger = logging.getLogger('enhanced_training_wandb')
        
        if not quiet:
            logger.info(f"开始训练 - 损失函数组合: {self.config.LOSS_CONFIG['loss_combination']}")

        for epoch in range(self.config.NUM_EPOCHS):
            if not quiet:
                logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            # 训练
            train_metrics = self.train_epoch(train_loader, quiet)

            # 验证
            val_metrics = self.validate(val_loader, quiet)

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

            # WandB日志
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

            # 打印指标（只在非安静模式下）
            if not quiet:
                logger.info(f"训练 - 总损失: {train_metrics['total']:.6f}, 像素: {train_metrics['pixel']:.6f}, "
                          f"梯度: {train_metrics['gradient']:.6f}, 频谱: {train_metrics['spectral']:.6f}")
                logger.info(f"验证 - 总损失: {val_metrics['total']:.6f}, 像素: {val_metrics['pixel']:.6f}, "
                          f"梯度: {val_metrics['gradient']:.6f}, 频谱: {val_metrics['spectral']:.6f}, "
                          f"相关: {val_metrics['correlation']:.6f}")
            else:
                # 安静模式下只打印关键信息
                print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}: "
                      f"Train: {train_metrics['total']:.4f}, "
                      f"Val: {val_metrics['total']:.4f}, "
                      f"Corr: {val_metrics['correlation']:.4f}")

            # 保存最佳模型
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, val_metrics['total'])
                if not quiet:
                    logger.info("保存最佳模型")
            else:
                self.early_stopping_counter += 1

            # 早停检查
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                if not quiet:
                    logger.info(f"早停触发，在epoch {epoch + 1}")
                break

        if not quiet:
            logger.info("训练完成!")
        
        # 结束WandB运行
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, val_loss):
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


# 配置类定义（保持与之前相同）
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


def get_config(config_name):
    """根据配置名称获取配置对象"""
    config_map = {
        'baseline': BaselineConfig,
        'weighted_mse': WeightedMSEConfig,
        'mse_grad': MSEGradientConfig,
        'mse_fft': MSESpectralConfig,
        'all_combined': AllCombinedConfig,
    }
    
    if config_name not in config_map:
        available_configs = list(config_map.keys())
        raise ValueError(f"未知配置: {config_name}. 可用配置: {available_configs}")
    
    return config_map[config_name]()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强海底地形精化训练 - 集成WandB')
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline', 'weighted_mse', 'mse_grad', 'mse_fft', 'all_combined'],
                        help='选择训练配置')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖默认的训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='覆盖默认的批次大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='覆盖默认的学习率')
    parser.add_argument('--use_wandb', action='store_true', help='使用WandB进行实验跟踪')
    parser.add_argument('--wandb_project', type=str, default='seafloor-topography', help='WandB项目名称')
    parser.add_argument('--quiet', action='store_true', help='安静模式，减少输出')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.quiet)
    
    # 获取配置
    config = get_config(args.config)
    
    # 应用命令行参数覆盖
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    
    if not args.quiet:
        logger.info(f"使用配置: {args.config}")
        logger.info(f"损失函数组合: {config.LOSS_CONFIG['loss_combination']}")
        logger.info(f"输出目录: {config.OUTPUT_DIR}")
        logger.info(f"使用WandB: {args.use_wandb and WANDB_AVAILABLE}")
    
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
    data_prep = enhanced_training_multi_loss.EnhancedDataPreparation(config)

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
        train_dataset = enhanced_training_multi_loss.EnhancedBathymetryDataset(train_patches, config, is_training=True)
        val_dataset = enhanced_training_multi_loss.EnhancedBathymetryDataset(val_patches, config, is_training=False)
        test_dataset = enhanced_training_multi_loss.EnhancedBathymetryDataset(test_patches, config, is_training=False)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                                num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                              num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                                num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())

        # 创建模型
        model = enhanced_training_multi_loss.OptimizedUNet(
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