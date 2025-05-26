#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强训练脚本 - 包含完整可视化功能

这个脚本在原有训练基础上添加了：
1. 完整的训练历史可视化
2. 预测示例图
3. 散点图对比
4. 详细的实验报告

使用方法:
python enhanced_training_with_viz.py --config baseline --epochs 20

作者: 周瑞宸
日期: 2025-05-25
"""

import argparse
import sys
import os
import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import json
import logging
from datetime import datetime

# 将父目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_training_multi_loss import EnhancedTrainingConfig
import enhanced_training_multi_loss


def convert_to_python_types(data):
    if isinstance(data, dict):
        return {key: convert_to_python_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, numpy.float32) or isinstance(data, numpy.float64): # 你也可以用 numpy.floating
        return float(data)
    elif isinstance(data, numpy.int32) or isinstance(data, numpy.int64): # 你也可以用 numpy.integer
        return int(data)
    # 可以根据需要添加其他 numpy 类型的转换
    return data


logger = logging.getLogger('enhanced_training_with_viz')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_training_visualization(trainer, config, output_dir):
    """创建完整的训练可视化"""
    logger.info("创建训练历史可视化...")
    
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
    
    # 保存图像
    plot_path = os.path.join(output_dir, f'training_history_{config.LOSS_CONFIG["loss_combination"]}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练历史图保存到: {plot_path}")


def create_prediction_examples(model, test_loader, config, output_dir, device):
    """创建预测示例图"""
    logger.info("创建预测示例...")
    
    model.eval()
    
    # 获取一个批次的测试数据
    with torch.no_grad():
        for batch_data in test_loader:
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
            
            predictions = model(features)
            break
    
    # 选择几个示例进行可视化
    n_examples = min(4, predictions.size(0))
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 5*n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        pred = predictions[i, 0].cpu().numpy()
        true = labels[i, 0].cpu().numpy()
        error = pred - true
        
        # 真值
        im1 = axes[i, 0].imshow(true, cmap='RdBu_r', aspect='auto')
        axes[i, 0].set_title(f'真值 {i+1}')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 预测
        im2 = axes[i, 1].imshow(pred, cmap='RdBu_r', aspect='auto')
        axes[i, 1].set_title(f'预测 {i+1}')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 误差
        im3 = axes[i, 2].imshow(error, cmap='RdBu_r', aspect='auto')
        axes[i, 2].set_title(f'误差 {i+1}')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.suptitle(f'预测示例 - {config.LOSS_CONFIG["loss_combination"]}', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    examples_path = os.path.join(output_dir, f'prediction_examples_{config.LOSS_CONFIG["loss_combination"]}.png')
    plt.savefig(examples_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"预测示例图保存到: {examples_path}")


def create_scatter_plot(model, test_loader, config, output_dir, device):
    """创建预测vs真值散点图"""
    logger.info("创建散点图...")
    
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
    
    # 合并所有数据
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
    
    # 采样数据点（避免过多数据点）
    n_samples = min(50000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_samples, replace=False)
    pred_sample = pred_flat[indices]
    target_sample = target_flat[indices]
    
    ax.scatter(target_sample, pred_sample, alpha=0.3, s=1, c='blue')
    
    # 绘制理想线
    min_val = min(target_sample.min(), pred_sample.min())
    max_val = max(target_sample.max(), pred_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测')
    
    ax.set_xlabel('真值')
    ax.set_ylabel('预测值')
    ax.set_title(f'预测 vs 真值散点图\n'
                f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, 相关系数: {correlation:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置相同的坐标轴范围
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # 保存图像
    scatter_path = os.path.join(output_dir, f'scatter_plot_{config.LOSS_CONFIG["loss_combination"]}.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"散点图保存到: {scatter_path}")
    
    return {'rmse': rmse, 'mae': mae, 'correlation': correlation}


def create_detailed_report(config, trainer, test_metrics, output_dir, total_params):
    """创建详细的实验报告"""
    logger.info("生成详细实验报告...")
    
    report_lines = []
    report_lines.append(f"# 海底地形精化实验报告")
    report_lines.append(f"## 配置: {config.LOSS_CONFIG['loss_combination']}")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 实验配置
    report_lines.append("## 1. 实验配置")
    report_lines.append(f"- **损失函数组合**: {config.LOSS_CONFIG['loss_combination']}")
    report_lines.append(f"- **使用TID权重**: {'是' if config.LOSS_CONFIG['use_tid_weights'] else '否'}")
    report_lines.append(f"- **像素损失权重**: {config.LOSS_CONFIG['pixel_loss_weight']}")
    report_lines.append(f"- **梯度损失权重**: {config.LOSS_CONFIG['gradient_loss_weight']}")
    report_lines.append(f"- **频谱损失权重**: {config.LOSS_CONFIG['spectral_loss_weight']}")
    report_lines.append(f"- **批次大小**: {config.BATCH_SIZE}")
    report_lines.append(f"- **学习率**: {config.LEARNING_RATE}")
    report_lines.append(f"- **权重衰减**: {config.WEIGHT_DECAY}")
    report_lines.append("")
    
    # 2. 模型信息
    report_lines.append("## 2. 模型信息")
    report_lines.append(f"- **总参数数量**: {total_params:,}")
    report_lines.append(f"- **输入通道数**: {config.IN_CHANNELS}")
    report_lines.append(f"- **输出通道数**: {config.OUT_CHANNELS}")
    report_lines.append(f"- **初始特征数**: {config.INIT_FEATURES}")
    report_lines.append("")
    
    # 3. 训练结果
    report_lines.append("## 3. 训练结果")
    report_lines.append(f"- **训练轮数**: {len(trainer.history['train_loss_total'])}")
    report_lines.append(f"- **最佳验证损失**: {trainer.best_val_loss:.6f}")
    report_lines.append(f"- **最终训练损失**: {trainer.history['train_loss_total'][-1]:.6f}")
    report_lines.append(f"- **最终验证相关系数**: {trainer.history['val_correlation'][-1]:.6f}")
    report_lines.append("")
    
    # 4. 测试集评估
    report_lines.append("## 4. 测试集评估")
    report_lines.append(f"- **RMSE**: {test_metrics['rmse']:.6f}")
    report_lines.append(f"- **MAE**: {test_metrics['mae']:.6f}")
    report_lines.append(f"- **相关系数**: {test_metrics['correlation']:.6f}")
    report_lines.append("")
    
    # 5. 损失函数分析
    if len(trainer.history['train_loss_total']) > 0:
        report_lines.append("## 5. 损失函数分析")
        final_pixel_loss = trainer.history['val_loss_pixel'][-1]
        final_gradient_loss = trainer.history['val_loss_gradient'][-1]
        final_spectral_loss = trainer.history['val_loss_spectral'][-1]
        
        report_lines.append(f"- **最终像素损失**: {final_pixel_loss:.6f}")
        report_lines.append(f"- **最终梯度损失**: {final_gradient_loss:.6f}")
        report_lines.append(f"- **最终频谱损失**: {final_spectral_loss:.6f}")
        report_lines.append("")
    
    # 6. 输出文件
    report_lines.append("## 6. 输出文件")
    report_lines.append(f"- **模型权重**: best_model_enhanced_multi_loss_{config.LOSS_CONFIG['loss_combination']}_lt135km.pth")
    report_lines.append(f"- **训练历史图**: training_history_{config.LOSS_CONFIG['loss_combination']}.png")
    report_lines.append(f"- **预测示例图**: prediction_examples_{config.LOSS_CONFIG['loss_combination']}.png")
    report_lines.append(f"- **散点图**: scatter_plot_{config.LOSS_CONFIG['loss_combination']}.png")
    report_lines.append(f"- **实验摘要**: experiment_summary_{config.LOSS_CONFIG['loss_combination']}_lt135km.json")
    
    report_content = "\n".join(report_lines)
    
    # 保存报告
    report_path = os.path.join(output_dir, f'detailed_report_{config.LOSS_CONFIG["loss_combination"]}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"详细报告保存到: {report_path}")


# 配置类定义（与之前相同）
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
    """主函数 - 包含完整可视化的训练流程"""
    parser = argparse.ArgumentParser(description='增强海底地形精化训练 - 包含完整可视化')
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline', 'weighted_mse', 'mse_grad', 'mse_fft', 'all_combined'],
                        help='选择训练配置')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖默认的训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='覆盖默认的批次大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='覆盖默认的学习率')
    
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
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 运行完整的训练和可视化流程
    logger.info("=== 增强海底地形精化训练 - 包含完整可视化 ===")
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
        # 加载数据
        features_ds, labels_da, tid_weights_da = data_prep.load_data()

        # 提取数据块
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
        logger.info(f"模型总参数数量: {total_params:,}")

        # 训练模型
        trainer = enhanced_training_multi_loss.EnhancedTrainer(model, config)
        trainer.train(train_loader, val_loader)

        # 创建完整的可视化
        create_training_visualization(trainer, config, config.OUTPUT_DIR)
        
        # 创建预测示例图
        create_prediction_examples(model, test_loader, config, config.OUTPUT_DIR, device)
        
        # 创建散点图并获取测试指标
        test_metrics = create_scatter_plot(model, test_loader, config, config.OUTPUT_DIR, device)
        
        # 创建详细报告
        create_detailed_report(config, trainer, test_metrics, config.OUTPUT_DIR, total_params)

        # 保存实验摘要（扩展版本）
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
            'trainable_params': total_params,
            'test_metrics': test_metrics
        }

        summary_path = os.path.join(config.OUTPUT_DIR, 
                                  f'experiment_summary_{config.LOSS_CONFIG["loss_combination"]}_lt135km.json')

        experiment_summary_serializable = convert_to_python_types(experiment_summary)

        with open(summary_path, 'w') as f:
            # json.dump(experiment_summary, f, indent=2)
            json.dump(experiment_summary_serializable, f, indent=2)

        logger.info(f"✅ 增强训练完成！损失组合: {config.LOSS_CONFIG['loss_combination']}")
        logger.info(f"📊 最佳验证损失: {trainer.best_val_loss:.6f}")
        logger.info(f"📈 测试RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"📈 测试相关系数: {test_metrics['correlation']:.6f}")
        logger.info(f"📁 结果保存在: {config.OUTPUT_DIR}")
        
        # 列出生成的文件
        logger.info("📄 生成的文件:")
        output_files = [
            f"best_model_enhanced_multi_loss_{config.LOSS_CONFIG['loss_combination']}_lt135km.pth",
            f"experiment_summary_{config.LOSS_CONFIG['loss_combination']}_lt135km.json",
            f"training_history_{config.LOSS_CONFIG['loss_combination']}.png",
            f"prediction_examples_{config.LOSS_CONFIG['loss_combination']}.png",
            f"scatter_plot_{config.LOSS_CONFIG['loss_combination']}.png",
            f"detailed_report_{config.LOSS_CONFIG['loss_combination']}.md"
        ]
        
        for file_name in output_files:
            file_path = os.path.join(config.OUTPUT_DIR, file_name)
            if os.path.exists(file_path):
                logger.info(f"   ✓ {file_name}")
            else:
                logger.info(f"   ✗ {file_name} (未找到)")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()