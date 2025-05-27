#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海底地形精化模型预测与评估脚本

该脚本用于加载不同损失函数训练的模型，对测试数据进行预测，并通过多种指标评估模型性能。
主要功能：
1. 加载多个训练好的模型
2. 在测试集上进行预测
3. 计算多种评估指标（MSE、MAE、相关系数、梯度损失、频谱损失等）
4. 可视化对比不同模型的预测结果
5. 生成综合评估报告

作者: 周瑞宸
日期: 2025-05-30
"""

import os
import sys
import json
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import logging
from pathlib import Path
import argparse
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import platform

# 导入项目模块
sys.path.append('/mnt/data5/06-Projects/05-unet')
from enhanced_training_multi_loss import (
    EnhancedTrainingConfig, OptimizedUNet, EnhancedDataPreparation,
    EnhancedBathymetryDataset, TIDWeightedLoss, GradientLoss, SpectralLoss
)
# 添加以下导入语句
from training_variants import (
    BaselineConfig, WeightedMSEConfig, MSEGradientConfig, 
    MSESpectralConfig, AllCombinedConfig
)


# 设置matplotlib支持中文
def setup_chinese_font():
    """配置matplotlib支持中文字体"""
    system = platform.system()
    if system == 'Linux':
        plt.rcParams['font.sans-serif'] = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
            'Noto Sans CJK TC', 'Droid Sans Fallback', 'AR PL UMing CN',
            'AR PL UKai CN', 'SimHei', 'SimSun', 'FangSong', 'DejaVu Sans'
        ]
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

setup_chinese_font()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_prediction_comparison.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_prediction_comparison')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")


class ModelEvaluator:
    """模型评估器 - 用于评估和比较不同模型的性能"""
    
    def __init__(self, base_dir="/mnt/data5/06-Projects/05-unet"):
        self.base_dir = base_dir
        self.output_base = os.path.join(base_dir, "output")
        self.models_dir = os.path.join(base_dir, "output")
        self.results_dir = os.path.join(base_dir, "prediction_results")
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 实验配置定义
        self.experiment_configs = [
            ('baseline', '基线实验 - 标准MSE', '2-baseline_outputs_lt135km'),
            ('weighted_mse', 'TID权重MSE', '2-weighted_mse_outputs_lt135km'),
            ('mse_grad', 'MSE + 梯度损失', '2-mse_grad_outputs_lt135km'),
            ('mse_fft', 'MSE + 频谱损失', '2-mse_fft_outputs_lt135km'),
            ('all_combined', '全损失组合', '2-all_combined_outputs_lt135km'),
        ]
        
        # 评估指标
        self.metrics = {
            'mse': self._compute_mse,
            'mae': self._compute_mae,
            'correlation': self._compute_correlation,
            'gradient_loss': self._compute_gradient_loss,
            'spectral_loss': self._compute_spectral_loss,
            'ssim': self._compute_ssim
        }
    
    def load_model(self, model_path):
        """加载模型"""
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 获取模型配置
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            else:
                # 使用默认配置
                model_config = {
                    'in_channels': 4,
                    'out_channels': 1,
                    'init_features': 48
                }
            
            # 创建模型实例
            model = OptimizedUNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                init_features=model_config['init_features']
            )
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            logger.info(f"成功加载模型: {model_path}")
            return model, checkpoint
        
        except Exception as e:
            logger.error(f"加载模型失败 {model_path}: {e}")
            return None, None
    
    def prepare_test_data(self, config=None):
        """准备测试数据"""
        if config is None:
            config = EnhancedTrainingConfig()
        
        logger.info("准备测试数据...")
        
        # 数据准备
        data_prep = EnhancedDataPreparation(config)
        
        # 加载数据
        features_ds, labels_da, tid_weights_da = data_prep.load_data()
        
        # 提取数据块
        patches = data_prep.extract_patches(features_ds, labels_da, tid_weights_da)
        
        # 划分数据集
        _, _, test_patches = data_prep.split_patches(patches)
        
        # 创建测试数据集
        test_dataset = EnhancedBathymetryDataset(test_patches, config, is_training=False)
        
        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"测试数据准备完成，共 {len(test_dataset)} 个样本")
        return test_loader, test_patches
    
    def _compute_mse(self, pred, target):
        """计算MSE"""
        return F.mse_loss(pred, target).item()
    
    def _compute_mae(self, pred, target):
        """计算MAE"""
        return F.l1_loss(pred, target).item()
    
    def _compute_correlation(self, pred, target):
        """计算相关系数"""
        pred_flat = pred.flatten().cpu().numpy()
        target_flat = target.flatten().cpu().numpy()
        return pearsonr(pred_flat, target_flat)[0]
    
    def _compute_gradient_loss(self, pred, target):
        """计算梯度损失"""
        gradient_loss = GradientLoss(method='sobel')
        return gradient_loss(pred, target).item()
    
    def _compute_spectral_loss(self, pred, target):
        """计算频谱损失"""
        spectral_loss = SpectralLoss(focus_high_freq=True)
        return spectral_loss(pred, target).item()
    
    def _compute_ssim(self, pred, target):
        """计算结构相似性"""
        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()
        
        # 获取图像的最小边长
        min_side = min(pred_np.shape)
        
        # 计算适合的窗口大小（必须是奇数且小于等于最小边长）
        win_size = min(7, min_side - (min_side % 2 == 0))  # 确保是奇数且不超过图像尺寸
        
        # 如果图像太小，无法应用SSIM
        if win_size < 3:
            logger.warning(f"图像太小 ({pred_np.shape})，无法计算SSIM，返回0")
            return 0.0
            
        try:
            return ssim(pred_np, target_np, 
                       data_range=target_np.max() - target_np.min(),
                       win_size=win_size)
        except Exception as e:
            logger.warning(f"计算SSIM时出错: {e}，返回0")
            return 0.0
    
    def evaluate_model(self, model, test_loader):
        """评估单个模型"""
        model.eval()
        
        metrics_results = {metric: [] for metric in self.metrics}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="评估中"):
                if len(batch_data) == 3:
                    features, labels, tid_weights = batch_data
                    features = features.to(device)
                    labels = labels.to(device)
                else:
                    features, labels = batch_data
                    features = features.to(device)
                    labels = labels.to(device)
                
                # 前向传播
                outputs = model(features)
                
                # 计算各项指标
                for metric_name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(outputs, labels)
                    metrics_results[metric_name].append(metric_value)
                
                # 收集预测结果
                all_predictions.append(outputs.cpu())
                all_targets.append(labels.cpu())
        
        # 计算平均指标
        avg_metrics = {metric: np.mean(values) for metric, values in metrics_results.items()}
        
        # 合并预测结果
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return avg_metrics, predictions, targets
    
    def compare_models(self):
        """比较多个模型的性能"""
        logger.info("开始比较模型性能...")
        
        # 准备测试数据
        test_loader, test_patches = self.prepare_test_data()
        
        results = []
        predictions_dict = {}
        
        for config_name, description, output_dir in self.experiment_configs:
            logger.info(f"\n评估模型: {description} ({config_name})")
            
            # 查找模型文件
            model_dir = os.path.join(self.output_base, output_dir)
            model_pattern = f"best_model_enhanced_multi_loss_*_lt135km.pth"
            model_files = list(Path(model_dir).glob(model_pattern))
            
            if not model_files:
                logger.warning(f"未找到模型文件: {model_dir}/{model_pattern}")
                continue
            
            model_path = str(model_files[0])
            
            # 加载模型
            model, checkpoint = self.load_model(model_path)
            if model is None:
                continue
            
            # 评估模型
            metrics, predictions, targets = self.evaluate_model(model, test_loader)
            
            # 保存结果
            result = {
                'experiment': config_name,
                'description': description,
                **metrics
            }
            
            results.append(result)
            predictions_dict[config_name] = predictions
            
            logger.info(f"评估结果: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "  
                      f"相关系数={metrics['correlation']:.6f}, SSIM={metrics['ssim']:.6f}")
        
        # 保存最后一个目标值用于可视化
        if targets is not None:
            predictions_dict['target'] = targets
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存结果
        results_df.to_csv(os.path.join(self.results_dir, 'model_comparison_results.csv'), index=False)
        
        # 可视化结果
        self.visualize_results(results_df, predictions_dict, test_patches)
        
        # 生成报告
        self.generate_report(results_df)
        
        return results_df
    
    def visualize_results(self, results_df, predictions_dict, test_patches):
        """可视化结果"""
        logger.info("生成可视化结果...")
        
        # 1. 性能指标对比图
        plt.figure(figsize=(15, 10))
        
        metrics_to_plot = ['mse', 'mae', 'correlation', 'ssim', 'gradient_loss', 'spectral_loss']
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # 排序以便更好地可视化
            if metric in ['correlation', 'ssim']:
                # 这些指标越高越好
                sorted_df = results_df.sort_values(metric, ascending=False)
            else:
                # 这些指标越低越好
                sorted_df = results_df.sort_values(metric)
            
            bars = ax.bar(sorted_df['description'], sorted_df[metric])
            ax.set_title(f'{metric} 对比')
            ax.set_xticklabels(sorted_df['description'], rotation=45, ha='right')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
        
        # 2. 样本预测对比图
        # 随机选择几个测试样本进行可视化
        num_samples = min(5, len(test_patches))
        sample_indices = np.random.choice(len(predictions_dict['target']), num_samples, replace=False)
        
        for idx in sample_indices:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            # 获取目标值
            target = predictions_dict['target'][idx, 0].numpy()
            
            # 绘制目标值
            im = axes[0].imshow(target, cmap='viridis')
            axes[0].set_title('真实值')
            plt.colorbar(im, ax=axes[0])
            
            # 绘制各模型预测值
            for i, (model_name, description) in enumerate([(k, v[1]) for k, v, _ in self.experiment_configs]):
                if model_name in predictions_dict:
                    pred = predictions_dict[model_name][idx, 0].numpy()
                    im = axes[i+1].imshow(pred, cmap='viridis')
                    axes[i+1].set_title(f'{description}')
                    plt.colorbar(im, ax=axes[i+1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'sample_{idx}_comparison.png'), dpi=300)
            plt.close()
        
        # 3. 误差分布图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # 选择一个样本进行误差可视化
        idx = sample_indices[0]
        target = predictions_dict['target'][idx, 0].numpy()
        
        for i, (model_name, description) in enumerate([(k, v[1]) for k, v, _ in self.experiment_configs]):
            if model_name in predictions_dict:
                pred = predictions_dict[model_name][idx, 0].numpy()
                error = np.abs(pred - target)
                
                im = axes[i].imshow(error, cmap='hot')
                axes[i].set_title(f'{description} - 误差')
                plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'error_distribution.png'), dpi=300)
        plt.close()
    
    def generate_report(self, results_df):
        """生成评估报告"""
        logger.info("生成评估报告...")
        
        report_lines = []
        report_lines.append("# 海底地形精化模型评估报告")
        report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. 模型概况
        report_lines.append("## 1. 模型概况")
        report_lines.append(f"- 评估模型数量: {len(results_df)}")
        report_lines.append(f"- 模型列表: {', '.join(results_df['description'].tolist())}")
        report_lines.append("")
        
        # 2. 性能指标对比
        report_lines.append("## 2. 性能指标对比")
        report_lines.append("")
        report_lines.append("### 2.1 均方误差 (MSE)")
        mse_df = results_df.sort_values('mse')
        for i, row in mse_df.iterrows():
            report_lines.append(f"- **{row['description']}**: {row['mse']:.6f}")
        report_lines.append("")
        
        report_lines.append("### 2.2 平均绝对误差 (MAE)")
        mae_df = results_df.sort_values('mae')
        for i, row in mae_df.iterrows():
            report_lines.append(f"- **{row['description']}**: {row['mae']:.6f}")
        report_lines.append("")
        
        report_lines.append("### 2.3 相关系数")
        corr_df = results_df.sort_values('correlation', ascending=False)
        for i, row in corr_df.iterrows():
            report_lines.append(f"- **{row['description']}**: {row['correlation']:.6f}")
        report_lines.append("")
        
        report_lines.append("### 2.4 结构相似性 (SSIM)")
        ssim_df = results_df.sort_values('ssim', ascending=False)
        for i, row in ssim_df.iterrows():
            report_lines.append(f"- **{row['description']}**: {row['ssim']:.6f}")
        report_lines.append("")
        
        report_lines.append("### 2.5 梯度损失")
        grad_df = results_df.sort_values('gradient_loss')
        for i, row in grad_df.iterrows():
            report_lines.append(f"- **{row['description']}**: {row['gradient_loss']:.6f}")
        report_lines.append("")
        
        report_lines.append("### 2.6 频谱损失")
        spec_df = results_df.sort_values('spectral_loss')
        for i, row in spec_df.iterrows():
            report_lines.append(f"- **{row['description']}**: {row['spectral_loss']:.6f}")
        report_lines.append("")
        
        # 3. 最佳模型
        report_lines.append("## 3. 最佳模型分析")
        
        # 根据综合排名确定最佳模型
        results_df['mse_rank'] = results_df['mse'].rank()
        results_df['mae_rank'] = results_df['mae'].rank()
        results_df['corr_rank'] = results_df['correlation'].rank(ascending=False)
        results_df['ssim_rank'] = results_df['ssim'].rank(ascending=False)
        results_df['grad_rank'] = results_df['gradient_loss'].rank()
        results_df['spec_rank'] = results_df['spectral_loss'].rank()
        
        results_df['avg_rank'] = (results_df['mse_rank'] + results_df['mae_rank'] + 
                                 results_df['corr_rank'] + results_df['ssim_rank'] + 
                                 results_df['grad_rank'] + results_df['spec_rank']) / 6
        
        best_model = results_df.loc[results_df['avg_rank'].idxmin()]
        
        report_lines.append(f"### 3.1 综合最佳模型: {best_model['description']}")
        report_lines.append(f"- MSE: {best_model['mse']:.6f}")
        report_lines.append(f"- MAE: {best_model['mae']:.6f}")
        report_lines.append(f"- 相关系数: {best_model['correlation']:.6f}")
        report_lines.append(f"- SSIM: {best_model['ssim']:.6f}")
        report_lines.append(f"- 梯度损失: {best_model['gradient_loss']:.6f}")
        report_lines.append(f"- 频谱损失: {best_model['spectral_loss']:.6f}")
        report_lines.append(f"- 平均排名: {best_model['avg_rank']:.2f}")
        report_lines.append("")
        
        # 4. 结论与建议
        report_lines.append("## 4. 结论与建议")
        
        # 根据结果给出建议
        if best_model['experiment'] == 'all_combined':
            report_lines.append("- 综合损失函数组合表现最佳，建议在实际应用中采用")
        elif best_model['experiment'] == 'mse_grad':
            report_lines.append("- MSE与梯度损失的组合表现良好，在计算资源有限时可以考虑使用")
        elif best_model['experiment'] == 'mse_fft':
            report_lines.append("- MSE与频谱损失的组合在频域表现突出，适合需要保留高频细节的场景")
        
        # 添加详细的指标分析
        report_lines.append("")
        report_lines.append("### 详细指标分析")
        report_lines.append("")
        report_lines.append("#### 均方误差 (MSE) 分析")
        if results_df['mse'].min() < results_df.loc[results_df['experiment'] == 'baseline', 'mse'].values[0]:
            improvement = (results_df.loc[results_df['experiment'] == 'baseline', 'mse'].values[0] - 
                          results_df['mse'].min()) / results_df.loc[results_df['experiment'] == 'baseline', 'mse'].values[0] * 100
            report_lines.append(f"- 最佳模型相比基线模型MSE降低了 {improvement:.2f}%")
        
        report_lines.append("")
        report_lines.append("#### 相关系数分析")
        if results_df['correlation'].max() > results_df.loc[results_df['experiment'] == 'baseline', 'correlation'].values[0]:
            improvement = (results_df['correlation'].max() - 
                          results_df.loc[results_df['experiment'] == 'baseline', 'correlation'].values[0]) / \
                          results_df.loc[results_df['experiment'] == 'baseline', 'correlation'].values[0] * 100
            report_lines.append(f"- 最佳模型相比基线模型相关系数提高了 {improvement:.2f}%")
        
        report_lines.append("")
        report_lines.append("## 5. 完整结果表格")
        report_lines.append("```")
        report_lines.append(results_df[['experiment', 'description', 'mse', 'mae', 'correlation', 
                                     'ssim', 'gradient_loss', 'spectral_loss', 'avg_rank']].to_string(index=False))
        report_lines.append("```")
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_path = os.path.join(self.results_dir, 'model_evaluation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"评估报告已保存到: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="海底地形精化模型预测与评估")
    parser.add_argument('--base_dir', type=str, default="/mnt/data5/06-Projects/05-unet",
                        help="项目基础目录")
    args = parser.parse_args()
    
    logger.info("=== 海底地形精化模型预测与评估 ===")
    
    # 创建评估器
    evaluator = ModelEvaluator(base_dir=args.base_dir)
    
    # 比较模型性能
    results = evaluator.compare_models()
    
    logger.info("评估完成!")
    
    # 打印最佳模型
    best_model = results.loc[results['mse'].idxmin()]
    logger.info(f"\n最佳模型 (MSE): {best_model['description']}")
    logger.info(f"MSE: {best_model['mse']:.6f}, MAE: {best_model['mae']:.6f}, "  
              f"相关系数: {best_model['correlation']:.6f}, SSIM: {best_model['ssim']:.6f}")


if __name__ == "__main__":
    main()