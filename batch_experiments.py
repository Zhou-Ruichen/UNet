#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验运行器和结果对比分析器

这个脚本可以：
1. 批量运行多种损失函数配置的训练实验
2. 自动收集和对比不同实验的结果
3. 生成综合的对比分析报告和可视化

使用方法:
# 运行所有预定义的实验配置
python batch_experiments.py --run_all

# 运行特定的实验配置
python batch_experiments.py --configs baseline weighted_mse mse_grad

# 只进行结果分析（假设实验已完成）
python batch_experiments.py --analyze_only

# 运行实验并分析
python batch_experiments.py --run_and_analyze --configs all_combined conservative aggressive

作者: 周瑞宸
日期: 2025-05-25
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import time
import logging
from typing import Dict, List, Tuple
import glob

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_experiments.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('batch_experiments')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BatchExperimentRunner:
    """批量实验运行器"""
    
    def __init__(self, base_output_dir="/mnt/data5/06-Projects/05-unet"):
        self.base_output_dir = base_output_dir
        self.results_dir = os.path.join(base_output_dir, "experiment_results")
        self.comparison_dir = os.path.join(base_output_dir, "comparison_analysis")
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # 预定义的实验配置
        self.experiment_configs = [
            'baseline',      # 基线 - 标准MSE
            'weighted_mse',  # 带TID权重的MSE
            'mse_grad',      # MSE + 梯度损失
            'mse_fft',       # MSE + 频谱损失
            'all_combined',  # 全组合
            'conservative',  # 保守配置
            'aggressive'     # 激进配置
        ]
    
    def run_single_experiment(self, config_name: str, max_epochs: int = 50) -> bool:
        """运行单个实验配置"""
        logger.info(f"开始运行实验: {config_name}")
        
        try:
            # 构造命令
            cmd = [
                'python', 'training_variants.py',
                '--config', config_name,
                '--epochs', str(max_epochs)
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 运行实验
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
            end_time = time.time()
            
            if result.returncode == 0:
                logger.info(f"实验 {config_name} 成功完成，用时: {end_time - start_time:.2f}秒")
                
                # 保存运行日志
                log_path = os.path.join(self.results_dir, f"{config_name}_run.log")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"实验配置: {config_name}\n")
                    f.write(f"开始时间: {datetime.fromtimestamp(start_time)}\n")
                    f.write(f"结束时间: {datetime.fromtimestamp(end_time)}\n")
                    f.write(f"用时: {end_time - start_time:.2f}秒\n\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
                
                return True
            else:
                logger.error(f"实验 {config_name} 失败，返回码: {result.returncode}")
                logger.error(f"错误输出: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"实验 {config_name} 超时")
            return False
        except Exception as e:
            logger.error(f"运行实验 {config_name} 时出错: {e}")
            return False
    
    def run_batch_experiments(self, config_names: List[str], max_epochs: int = 50):
        """批量运行实验"""
        logger.info(f"开始批量运行 {len(config_names)} 个实验配置")
        
        results = {}
        for config_name in config_names:
            success = self.run_single_experiment(config_name, max_epochs)
            results[config_name] = success
            
            # 短暂休息，避免系统过载
            time.sleep(10)
        
        # 保存批量运行结果摘要
        summary_path = os.path.join(self.results_dir, "batch_run_summary.json")
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(config_names),
            'successful_experiments': sum(results.values()),
            'failed_experiments': len(config_names) - sum(results.values()),
            'results': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"批量实验完成。成功: {summary['successful_experiments']}, 失败: {summary['failed_experiments']}")
        return results


class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, base_output_dir="/mnt/data5/06-Projects/05-unet"):
        self.base_output_dir = base_output_dir
        self.comparison_dir = os.path.join(base_output_dir, "comparison_analysis")
        os.makedirs(self.comparison_dir, exist_ok=True)
    
    def collect_experiment_results(self) -> pd.DataFrame:
        """收集所有实验的结果"""
        logger.info("收集实验结果...")
        
        results_data = []
        
        # 查找所有实验输出目录
        output_pattern = os.path.join(self.base_output_dir, "output", "2-*_outputs_lt135km")
        output_dirs = glob.glob(output_pattern)
        
        for output_dir in output_dirs:
            experiment_name = os.path.basename(output_dir).replace('2-', '').replace('_outputs_lt135km', '')
            
            # 查找实验摘要文件
            summary_files = glob.glob(os.path.join(output_dir, "experiment_summary_*_lt135km.json"))
            
            if summary_files:
                summary_file = summary_files[0]
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    # 提取关键指标
                    result_entry = {
                        'experiment': experiment_name,
                        'loss_combination': summary['loss_config']['loss_combination'],
                        'use_tid_weights': summary['loss_config']['use_tid_weights'],
                        'pixel_weight': summary['loss_config']['pixel_loss_weight'],
                        'gradient_weight': summary['loss_config']['gradient_loss_weight'],
                        'spectral_weight': summary['loss_config']['spectral_loss_weight'],
                        'best_val_loss': summary['best_val_loss'],
                        'total_params': summary['total_params'],
                        'num_epochs': summary['training_config']['num_epochs'],
                        'batch_size': summary['training_config']['batch_size'],
                        'learning_rate': summary['training_config']['learning_rate']
                    }
                    
                    results_data.append(result_entry)
                    logger.info(f"收集到实验结果: {experiment_name}")
                    
                except Exception as e:
                    logger.warning(f"读取实验摘要失败 {summary_file}: {e}")
            else:
                logger.warning(f"未找到实验摘要文件: {output_dir}")
        
        if results_data:
            df = pd.DataFrame(results_data)
            return df
        else:
            logger.warning("未找到任何实验结果")
            return pd.DataFrame()
    
    def analyze_training_histories(self) -> Dict:
        """分析训练历史曲线"""
        logger.info("分析训练历史...")
        
        # 查找所有训练历史图像
        history_pattern = os.path.join(self.base_output_dir, "output", "2-*_outputs_lt135km", 
                                     "enhanced_training_history_*_lt135km.png")
        history_files = glob.glob(history_pattern)
        
        training_data = {}
        
        # 尝试加载训练历史数据（如果保存了JSON格式）
        for output_dir in glob.glob(os.path.join(self.base_output_dir, "output", "2-*_outputs_lt135km")):
            experiment_name = os.path.basename(output_dir).replace('2-', '').replace('_outputs_lt135km', '')
            
            # 查找模型检查点文件
            checkpoint_files = glob.glob(os.path.join(output_dir, "best_model_*.pth"))
            
            if checkpoint_files:
                try:
                    import torch
                    checkpoint = torch.load(checkpoint_files[0], map_location='cpu', weights_only=False)
                    if 'history' in checkpoint:
                        training_data[experiment_name] = checkpoint['history']
                        logger.info(f"加载训练历史: {experiment_name}")
                except Exception as e:
                    logger.warning(f"加载训练历史失败 {experiment_name}: {e}")
        
        return training_data
    
    def create_comparison_plots(self, results_df: pd.DataFrame, training_data: Dict):
        """创建对比分析图表"""
        logger.info("创建对比分析图表...")
        
        if results_df.empty:
            logger.warning("没有结果数据，跳过图表创建")
            return
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 最佳验证损失对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1.1 验证损失柱状图
        sorted_df = results_df.sort_values('best_val_loss')
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(sorted_df)), sorted_df['best_val_loss'])
        ax1.set_xlabel('实验配置')
        ax1.set_ylabel('最佳验证损失')
        ax1.set_title('各实验配置的最佳验证损失对比')
        ax1.set_xticks(range(len(sorted_df)))
        ax1.set_xticklabels(sorted_df['experiment'], rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 1.2 损失权重配置热力图
        ax2 = axes[0, 1]
        weight_cols = ['pixel_weight', 'gradient_weight', 'spectral_weight']
        weight_data = results_df[['experiment'] + weight_cols].set_index('experiment')
        sns.heatmap(weight_data.T, annot=True, cmap='YlOrRd', ax=ax2, cbar_kws={'label': '权重值'})
        ax2.set_title('损失函数权重配置对比')
        ax2.set_xlabel('实验配置')
        ax2.set_ylabel('损失函数类型')
        
        # 1.3 训练效率对比（训练轮数 vs 最佳损失）
        ax3 = axes[1, 0]
        scatter = ax3.scatter(results_df['num_epochs'], results_df['best_val_loss'], 
                            s=100, alpha=0.7, c=results_df.index, cmap='tab10')
        ax3.set_xlabel('训练轮数')
        ax3.set_ylabel('最佳验证损失')
        ax3.set_title('训练效率分析')
        
        # 添加实验名称注释
        for i, row in results_df.iterrows():
            ax3.annotate(row['experiment'], (row['num_epochs'], row['best_val_loss']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 1.4 TID权重使用效果对比
        ax4 = axes[1, 1]
        tid_comparison = results_df.groupby('use_tid_weights')['best_val_loss'].agg(['mean', 'std']).reset_index()
        tid_labels = ['不使用TID权重', '使用TID权重']
        x_pos = [0, 1]
        bars = ax4.bar(x_pos, tid_comparison['mean'], yerr=tid_comparison['std'], 
                      capsize=5, alpha=0.7, color=['lightcoral', 'lightblue'])
        ax4.set_xlabel('TID权重使用情况')
        ax4.set_ylabel('平均最佳验证损失')
        ax4.set_title('TID权重使用效果对比')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(tid_labels)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        comparison_path = os.path.join(self.comparison_dir, 'loss_function_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 训练历史对比
        if training_data:
            self._plot_training_histories_comparison(training_data)
        
        logger.info(f"对比分析图表已保存到: {self.comparison_dir}")
    
    def _plot_training_histories_comparison(self, training_data: Dict):
        """绘制训练历史对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 总损失对比
        ax1 = axes[0, 0]
        for exp_name, history in training_data.items():
            if 'val_loss_total' in history:
                epochs = range(1, len(history['val_loss_total']) + 1)
                ax1.plot(epochs, history['val_loss_total'], label=exp_name, alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('验证总损失')
        ax1.set_title('验证总损失收敛对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 像素损失对比
        ax2 = axes[0, 1]
        for exp_name, history in training_data.items():
            if 'val_loss_pixel' in history:
                epochs = range(1, len(history['val_loss_pixel']) + 1)
                ax2.plot(epochs, history['val_loss_pixel'], label=exp_name, alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('验证像素损失')
        ax2.set_title('验证像素损失收敛对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 相关系数对比
        ax3 = axes[1, 0]
        for exp_name, history in training_data.items():
            if 'val_correlation' in history:
                epochs = range(1, len(history['val_correlation']) + 1)
                ax3.plot(epochs, history['val_correlation'], label=exp_name, alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('验证相关系数')
        ax3.set_title('验证相关系数变化对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 学习率对比
        ax4 = axes[1, 1]
        for exp_name, history in training_data.items():
            if 'learning_rates' in history:
                epochs = range(1, len(history['learning_rates']) + 1)
                ax4.semilogy(epochs, history['learning_rates'], label=exp_name, alpha=0.8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('学习率 (对数尺度)')
        ax4.set_title('学习率调度对比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        training_comparison_path = os.path.join(self.comparison_dir, 'training_histories_comparison.png')
        plt.savefig(training_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self, results_df: pd.DataFrame) -> str:
        """生成分析报告"""
        logger.info("生成分析报告...")
        
        report_lines = []
        report_lines.append("# 海底地形精化实验 - 损失函数对比分析报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if results_df.empty:
            report_lines.append("## 错误: 未找到实验结果数据")
            return "\n".join(report_lines)
        
        # 1. 实验概况
        report_lines.append("## 1. 实验概况")
        report_lines.append(f"- 总实验数量: {len(results_df)}")
        report_lines.append(f"- 实验配置: {', '.join(results_df['experiment'].tolist())}")
        report_lines.append("")
        
        # 2. 最佳结果
        best_result = results_df.loc[results_df['best_val_loss'].idxmin()]
        report_lines.append("## 2. 最佳实验结果")
        report_lines.append(f"- **最佳配置**: {best_result['experiment']}")
        report_lines.append(f"- **最佳验证损失**: {best_result['best_val_loss']:.6f}")
        report_lines.append(f"- **损失函数组合**: {best_result['loss_combination']}")
        report_lines.append(f"- **使用TID权重**: {'是' if best_result['use_tid_weights'] else '否'}")
        report_lines.append(f"- **训练轮数**: {best_result['num_epochs']}")
        report_lines.append("")
        
        # 3. 损失函数效果分析
        report_lines.append("## 3. 损失函数效果分析")
        
        # 3.1 TID权重效果
        tid_true = results_df[results_df['use_tid_weights'] == True]['best_val_loss'].mean()
        tid_false = results_df[results_df['use_tid_weights'] == False]['best_val_loss'].mean()
        
        if not np.isnan(tid_true) and not np.isnan(tid_false):
            improvement = (tid_false - tid_true) / tid_false * 100
            report_lines.append("### 3.1 TID权重效果")
            report_lines.append(f"- 使用TID权重平均损失: {tid_true:.6f}")
            report_lines.append(f"- 不使用TID权重平均损失: {tid_false:.6f}")
            report_lines.append(f"- 改善程度: {improvement:.2f}%")
            report_lines.append("")
        
        # 3.2 各损失函数组合效果排名
        report_lines.append("### 3.2 损失函数组合效果排名")
        sorted_results = results_df.sort_values('best_val_loss')
        for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
            report_lines.append(f"{i}. **{row['experiment']}** - 损失: {row['best_val_loss']:.6f}")
        report_lines.append("")
        
        # 4. 训练效率分析
        report_lines.append("## 4. 训练效率分析")
        avg_epochs = results_df['num_epochs'].mean()
        report_lines.append(f"- 平均训练轮数: {avg_epochs:.1f}")
        
        # 找出最快收敛的实验
        fastest_convergence = results_df.loc[results_df['num_epochs'].idxmin()]
        report_lines.append(f"- 最快收敛: {fastest_convergence['experiment']} ({fastest_convergence['num_epochs']} 轮)")
        report_lines.append("")
        
        # 5. 建议
        report_lines.append("## 5. 实验建议")
        report_lines.append("基于当前实验结果的建议:")
        
        if best_result['use_tid_weights']:
            report_lines.append("- ✓ 建议使用TID权重，可以有效提升模型性能")
        
        if best_result['gradient_weight'] > 0:
            report_lines.append("- ✓ 建议使用梯度损失，有助于恢复高频细节")
        
        if best_result['spectral_weight'] > 0:
            report_lines.append("- ✓ 建议使用频谱损失，可以增强频域约束")
        
        report_lines.append("")
        report_lines.append("## 6. 详细结果表格")
        report_lines.append("```")
        report_lines.append(results_df.to_string(index=False))
        report_lines.append("```")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.comparison_dir, 'experiment_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"分析报告已保存到: {report_path}")
        return report_content
    
    def run_full_analysis(self):
        """运行完整的结果分析"""
        logger.info("开始完整的结果分析...")
        
        # 收集结果
        results_df = self.collect_experiment_results()
        training_data = self.analyze_training_histories()
        
        # 创建图表
        self.create_comparison_plots(results_df, training_data)
        
        # 生成报告
        report = self.generate_analysis_report(results_df)
        
        # 保存结果数据
        if not results_df.empty:
            results_csv_path = os.path.join(self.comparison_dir, 'experiment_results.csv')
            results_df.to_csv(results_csv_path, index=False)
            logger.info(f"结果数据已保存到: {results_csv_path}")
        
        logger.info("完整分析完成!")
        return results_df, report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量实验运行器和结果分析器')
    parser.add_argument('--run_all', action='store_true', help='运行所有预定义的实验配置')
    parser.add_argument('--configs', nargs='+', help='指定要运行的实验配置')
    parser.add_argument('--analyze_only', action='store_true', help='只进行结果分析')
    parser.add_argument('--run_and_analyze', action='store_true', help='运行实验并分析结果')
    parser.add_argument('--max_epochs', type=int, default=50, help='最大训练轮数')
    
    args = parser.parse_args()
    
    runner = BatchExperimentRunner()
    analyzer = ExperimentAnalyzer()
    
    if args.analyze_only:
        # 只进行分析
        logger.info("执行结果分析...")
        analyzer.run_full_analysis()
        
    elif args.run_all:
        # 运行所有实验
        logger.info("运行所有预定义的实验配置...")
        results = runner.run_batch_experiments(runner.experiment_configs, args.max_epochs)
        
        if args.run_and_analyze:
            # 运行完毕后分析
            logger.info("实验完成，开始分析结果...")
            analyzer.run_full_analysis()
    
    elif args.configs:
        # 运行指定的实验配置
        logger.info(f"运行指定的实验配置: {args.configs}")
        
        # 验证配置名称
        valid_configs = set(runner.experiment_configs)
        invalid_configs = set(args.configs) - valid_configs
        if invalid_configs:
            logger.error(f"无效的配置名称: {invalid_configs}")
            logger.error(f"可用配置: {valid_configs}")
            return
        
        results = runner.run_batch_experiments(args.configs, args.max_epochs)
        
        if args.run_and_analyze:
            # 运行完毕后分析
            logger.info("实验完成，开始分析结果...")
            analyzer.run_full_analysis()
    
    else:
        # 显示帮助
        parser.print_help()


if __name__ == "__main__":
    main()