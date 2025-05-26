#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验管理器 - 系统化的损失函数对比实验

这个脚本提供了完整的实验管理功能：
1. 逐步执行实验配置
2. 实时监控训练进度
3. 自动收集和对比结果
4. 生成分析报告

使用方法:
python experiment_manager.py --mode sequential --epochs 20  # 逐步运行实验
python experiment_manager.py --mode analyze                 # 分析已有结果
python experiment_manager.py --mode quick_test             # 快速测试所有配置

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
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('experiment_manager')

# 设置matplotlib中文字体  
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, base_dir="/mnt/data5/06-Projects/05-unet"):
        self.base_dir = base_dir
        self.output_base = os.path.join(base_dir, "output")
        self.results_dir = os.path.join(base_dir, "experiment_results")
        self.analysis_dir = os.path.join(base_dir, "analysis_results")
        
        # 创建目录
        for dir_path in [self.results_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 实验配置定义（按推荐顺序）
        self.experiment_configs = [
            ('baseline', '基线实验 - 标准MSE', 'baseline'),
            ('weighted_mse', 'TID权重MSE', 'weighted_mse'),
            ('mse_grad', 'MSE + 梯度损失', 'weighted_mse_grad'),
            ('mse_fft', 'MSE + 频谱损失', 'weighted_mse_fft'),
            ('all_combined', '全损失组合', 'all'),
        ]
        
        # 实验状态跟踪
        self.experiment_status = {}
        self.load_experiment_status()
    
    def load_experiment_status(self):
        """加载实验状态"""
        status_file = os.path.join(self.results_dir, 'experiment_status.json')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                self.experiment_status = json.load(f)
        else:
            self.experiment_status = {config[0]: 'not_started' for config in self.experiment_configs}
    
    def save_experiment_status(self):
        """保存实验状态"""
        status_file = os.path.join(self.results_dir, 'experiment_status.json')
        with open(status_file, 'w') as f:
            json.dump(self.experiment_status, f, indent=2)
    
    def run_single_experiment(self, config_name: str, epochs: int = 50, force_restart: bool = False) -> bool:
        """运行单个实验"""
        if not force_restart and self.experiment_status.get(config_name) == 'completed':
            logger.info(f"实验 {config_name} 已完成，跳过。使用 --force 强制重新运行。")
            return True
        
        logger.info(f"开始运行实验: {config_name}, 训练轮数: {epochs}")
        self.experiment_status[config_name] = 'running'
        self.save_experiment_status()
        
        try:
            # 构造命令
            cmd = [
                'python', 'training_variants.py',
                '--config', config_name,
                '--epochs', str(epochs)
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 运行实验 - 修复tqdm显示问题
            start_time = time.time()
            
            # 使用实时输出，保持tqdm工作
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # 行缓冲
            )
            
            # 保存日志并实时显示
            log_file = os.path.join(self.results_dir, f"{config_name}_training.log")
            with open(log_file, 'w', encoding='utf-8') as f:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())  # 实时显示
                        f.write(output)
                        f.flush()
            
            process.wait()
            end_time = time.time()
            
            if process.returncode == 0:
                logger.info(f"实验 {config_name} 成功完成，用时: {end_time - start_time:.2f}秒")
                self.experiment_status[config_name] = 'completed'
                self.save_experiment_status()
                return True
            else:
                logger.error(f"实验 {config_name} 失败，返回码: {process.returncode}")
                self.experiment_status[config_name] = 'failed'
                self.save_experiment_status()
                return False
                
        except Exception as e:
            logger.error(f"运行实验 {config_name} 时出错: {e}")
            self.experiment_status[config_name] = 'failed'
            self.save_experiment_status()
            return False
    
    def run_sequential_experiments(self, epochs: int = 50, start_from: str = None, force_restart: bool = False):
        """按序列运行实验"""
        logger.info("开始按序列运行实验...")
        
        start_idx = 0
        if start_from:
            for i, (config_name, _, _) in enumerate(self.experiment_configs):
                if config_name == start_from:
                    start_idx = i
                    break
        
        results = {}
        for i, (config_name, description, loss_type) in enumerate(self.experiment_configs[start_idx:], start_idx):
            logger.info(f"\n{'='*60}")
            logger.info(f"实验 {i+1}/{len(self.experiment_configs)}: {description}")
            logger.info(f"配置: {config_name}, 损失类型: {loss_type}")
            logger.info(f"{'='*60}")
            
            success = self.run_single_experiment(config_name, epochs, force_restart)
            results[config_name] = success
            
            if success:
                logger.info(f"✓ 实验 {config_name} 完成")
                # 运行快速分析
                self.quick_analysis_single(config_name)
            else:
                logger.error(f"✗ 实验 {config_name} 失败")
                
                # 询问是否继续
                try:
                    response = input(f"实验 {config_name} 失败。是否继续下一个实验？(y/n): ")
                    if response.lower() != 'y':
                        logger.info("用户选择停止实验")
                        break
                except KeyboardInterrupt:
                    logger.info("用户中断实验")
                    break
            
            # 短暂休息
            if i < len(self.experiment_configs) - 1:
                logger.info("等待10秒后继续下一个实验...")
                time.sleep(10)
        
        # 运行综合分析
        logger.info("\n开始综合分析...")
        self.comprehensive_analysis()
        
        return results
    
    def quick_test_all(self, epochs: int = 10):
        """快速测试所有配置（用于验证脚本正确性）"""
        logger.info(f"快速测试所有配置，每个配置运行 {epochs} 轮...")
        
        results = {}
        for config_name, description, _ in self.experiment_configs:
            logger.info(f"快速测试: {config_name} - {description}")
            success = self.run_single_experiment(config_name, epochs)
            results[config_name] = success
            
            if not success:
                logger.error(f"快速测试失败: {config_name}")
        
        return results
    
    def quick_analysis_single(self, config_name: str):
        """单个实验的快速分析"""
        try:
            # 查找实验结果目录
            output_pattern = os.path.join(self.output_base, f"2-{config_name.replace('_', '-')}-*")
            output_dirs = glob.glob(output_pattern)
            
            if not output_dirs:
                output_pattern = os.path.join(self.output_base, f"2-*{config_name}*")
                output_dirs = glob.glob(output_pattern)
            
            if output_dirs:
                output_dir = output_dirs[0]
                
                # 查找实验摘要
                summary_files = glob.glob(os.path.join(output_dir, "experiment_summary_*.json"))
                if summary_files:
                    with open(summary_files[0], 'r') as f:
                        summary = json.load(f)
                    
                    logger.info(f"📊 {config_name} 实验结果:")
                    logger.info(f"   最佳验证损失: {summary['best_val_loss']:.6f}")
                    logger.info(f"   训练轮数: {summary['training_config']['num_epochs']}")
                    logger.info(f"   损失组合: {summary['loss_config']['loss_combination']}")
            
        except Exception as e:
            logger.warning(f"快速分析失败 {config_name}: {e}")
    
    def collect_all_results(self) -> pd.DataFrame:
        """收集所有实验结果"""
        logger.info("收集所有实验结果...")
        
        results_data = []
        
        # 查找所有实验输出目录
        output_pattern = os.path.join(self.output_base, "2-*_outputs_lt135km")
        output_dirs = glob.glob(output_pattern)
        
        for output_dir in output_dirs:
            experiment_name = os.path.basename(output_dir).replace('2-', '').replace('_outputs_lt135km', '')
            
            # 查找实验摘要文件
            summary_files = glob.glob(os.path.join(output_dir, "experiment_summary_*_lt135km.json"))
            
            if summary_files:
                try:
                    with open(summary_files[0], 'r') as f:
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
                    logger.warning(f"读取实验摘要失败 {summary_files[0]}: {e}")
        
        if results_data:
            df = pd.DataFrame(results_data)
            return df
        else:
            logger.warning("未找到任何实验结果")
            return pd.DataFrame()
    
    def create_comparison_plots(self, results_df: pd.DataFrame):
        """创建对比分析图表"""
        if results_df.empty:
            logger.warning("没有结果数据，跳过图表创建")
            return
        
        logger.info("创建对比分析图表...")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 最佳验证损失对比
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
        
        # 2. TID权重使用效果对比
        ax2 = axes[0, 1]
        if 'use_tid_weights' in results_df.columns:
            tid_comparison = results_df.groupby('use_tid_weights')['best_val_loss'].agg(['mean', 'std', 'count']).reset_index()
            
            if len(tid_comparison) > 1:
                tid_labels = ['不使用TID权重', '使用TID权重']
                x_pos = [0, 1]
                bars = ax2.bar(x_pos, tid_comparison['mean'], yerr=tid_comparison['std'], 
                              capsize=5, alpha=0.7, color=['lightcoral', 'lightblue'])
                ax2.set_xlabel('TID权重使用情况')
                ax2.set_ylabel('平均最佳验证损失')
                ax2.set_title('TID权重使用效果对比')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(tid_labels)
                
                # 添加数值标签
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom')
        
        # 3. 训练效率分析
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
        
        # 4. 损失函数组合效果
        ax4 = axes[1, 1]
        loss_comparison = results_df.groupby('loss_combination')['best_val_loss'].mean().sort_values()
        bars = ax4.bar(range(len(loss_comparison)), loss_comparison.values)
        ax4.set_xlabel('损失函数组合')
        ax4.set_ylabel('平均最佳验证损失')
        ax4.set_title('不同损失函数组合效果对比')
        ax4.set_xticks(range(len(loss_comparison)))
        ax4.set_xticklabels(loss_comparison.index, rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.analysis_dir, 'experiment_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"对比图表已保存到: {plot_path}")
    
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
        
        # 3. 实验排名
        report_lines.append("## 3. 实验效果排名")
        sorted_results = results_df.sort_values('best_val_loss')
        for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
            report_lines.append(f"{i}. **{row['experiment']}** - 损失: {row['best_val_loss']:.6f} ({row['loss_combination']})")
        report_lines.append("")
        
        # 4. 关键发现
        report_lines.append("## 4. 关键发现")
        
        # TID权重效果分析
        if len(results_df['use_tid_weights'].unique()) > 1:
            tid_true = results_df[results_df['use_tid_weights'] == True]['best_val_loss'].mean()
            tid_false = results_df[results_df['use_tid_weights'] == False]['best_val_loss'].mean()
            
            if not np.isnan(tid_true) and not np.isnan(tid_false):
                improvement = (tid_false - tid_true) / tid_false * 100
                report_lines.append("### 4.1 TID权重效果")
                report_lines.append(f"- 使用TID权重平均损失: {tid_true:.6f}")
                report_lines.append(f"- 不使用TID权重平均损失: {tid_false:.6f}")
                report_lines.append(f"- 改善程度: {improvement:.2f}%")
                report_lines.append("")
        
        # 损失函数组合效果
        report_lines.append("### 4.2 损失函数组合效果")
        loss_comparison = results_df.groupby('loss_combination')['best_val_loss'].agg(['mean', 'count']).sort_values('mean')
        for loss_type, stats in loss_comparison.iterrows():
            report_lines.append(f"- **{loss_type}**: 平均损失 {stats['mean']:.6f} (实验数: {stats['count']})")
        report_lines.append("")
        
        # 5. 建议
        report_lines.append("## 5. 下一步建议")
        report_lines.append("基于当前实验结果的建议:")
        
        if best_result['use_tid_weights']:
            report_lines.append("- ✓ 继续使用TID权重，确认有效提升模型性能")
        
        if best_result['gradient_weight'] > 0:
            report_lines.append("- ✓ 梯度损失表现良好，建议在后续实验中保留")
        
        if best_result['spectral_weight'] > 0:
            report_lines.append("- ✓ 频谱损失有效，建议进一步优化权重配置")
        
        report_lines.append("")
        report_lines.append("## 6. 详细结果表格")
        report_lines.append("```")
        report_lines.append(results_df.round(6).to_string(index=False))
        report_lines.append("```")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.analysis_dir, 'experiment_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"分析报告已保存到: {report_path}")
        return report_content
    
    def comprehensive_analysis(self):
        """综合分析"""
        logger.info("开始综合分析...")
        
        # 收集结果
        results_df = self.collect_all_results()
        
        if results_df.empty:
            logger.warning("没有找到实验结果，无法进行分析")
            return
        
        # 保存结果数据
        results_csv_path = os.path.join(self.analysis_dir, 'experiment_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"结果数据已保存到: {results_csv_path}")
        
        # 创建对比图表
        self.create_comparison_plots(results_df)
        
        # 生成分析报告
        report = self.generate_analysis_report(results_df)
        
        # 显示摘要
        logger.info("\n" + "="*60)
        logger.info("实验摘要:")
        logger.info("="*60)
        
        if not results_df.empty:
            best_result = results_df.loc[results_df['best_val_loss'].idxmin()]
            logger.info(f"🏆 最佳配置: {best_result['experiment']}")
            logger.info(f"📊 最佳损失: {best_result['best_val_loss']:.6f}")
            logger.info(f"🔧 损失组合: {best_result['loss_combination']}")
            logger.info(f"⚖️  使用TID权重: {'是' if best_result['use_tid_weights'] else '否'}")
        
        logger.info("="*60)
        logger.info("综合分析完成！")
    
    def show_status(self):
        """显示实验状态"""
        logger.info("当前实验状态:")
        logger.info("-" * 40)
        
        for config_name, description, _ in self.experiment_configs:
            status = self.experiment_status.get(config_name, 'not_started')
            status_symbol = {
                'not_started': '⚪',
                'running': '🟡',
                'completed': '🟢',
                'failed': '🔴'
            }.get(status, '❓')
            
            logger.info(f"{status_symbol} {config_name}: {description} - {status}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实验管理器 - 系统化的损失函数对比实验')
    parser.add_argument('--mode', type=str, default='sequential',
                        choices=['sequential', 'analyze', 'quick_test', 'status'],
                        help='运行模式')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--start_from', type=str, help='从指定配置开始运行')
    parser.add_argument('--force', action='store_true', help='强制重新运行已完成的实验')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.mode == 'sequential':
        # 按序列运行实验
        logger.info(f"按序列运行实验，每个配置 {args.epochs} 轮训练")
        if args.start_from:
            logger.info(f"从配置 {args.start_from} 开始")
        
        manager.run_sequential_experiments(
            epochs=args.epochs, 
            start_from=args.start_from,
            force_restart=args.force
        )
        
    elif args.mode == 'analyze':
        # 只进行分析
        logger.info("分析已有实验结果...")
        manager.comprehensive_analysis()
        
    elif args.mode == 'quick_test':
        # 快速测试
        logger.info(f"快速测试所有配置，每个配置 {args.epochs} 轮训练")
        manager.quick_test_all(epochs=args.epochs)
        
    elif args.mode == 'status':
        # 显示状态
        manager.show_status()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()