#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大区域海底地形预测器
基于04-UNet_SWOT项目的成功经验，实现多损失函数模型的大区域预测
"""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# 导入现有模块
from enhanced_training_multi_loss import OptimizedUNet, EnhancedTrainingConfig
from training_variants import BaselineConfig, WeightedMSEConfig, MSEGradientConfig, MSESpectralConfig, AllCombinedConfig

logger = logging.getLogger('large_area_predictor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LargeAreaPredictionConfig:
    """大区域预测配置"""
    # 数据路径
    DATA_DIR = "/mnt/data5/06-Projects/05-unet/output/1-refined_seafloor_data_preparation"
    FEATURES_PATH = os.path.join(DATA_DIR, "Features_SWOT_LT135km_Standardized.nc")
    LONGWAVE_PATH = os.path.join(DATA_DIR, "Bathy_True_Longwave.nc")
    TRUTH_PATH = "/mnt/data5/00-Data/nc/GEBCO_2024.nc"
    
    # 预测参数 - 借鉴成功经验
    PATCH_SIZE = 128
    STRIDE = PATCH_SIZE // 16  # 93.8%重叠率
    BATCH_SIZE = 16
    
    # 模型参数
    IN_CHANNELS = 4
    OUT_CHANNELS = 1
    INIT_FEATURES = 48
    
    # 预测区域
    PREDICTION_REGION = {
        "lat_range": slice(0, 50),  # 扩大预测区域
        "lon_range": slice(100, 130),
        "name": "南海扩展区域"
    }
    
    # 输出目录
    OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/large_area_predictions"

class LargeAreaPredictor:
    """大区域预测器"""
    
    def __init__(self, config: LargeAreaPredictionConfig):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # 模型配置映射 - 根据实际目录结构更新
        self.model_configs = {
            'baseline': ('2-baseline_outputs_lt135km', '基线MSE'),
            'weighted_mse': ('2-weighted_mse_outputs_lt135km', 'TID权重MSE'),
            'mse_grad': ('2-mse_grad_outputs_lt135km', 'MSE+梯度损失'),
            'mse_fft': ('2-mse_fft_outputs_lt135km', 'MSE+频谱损失'),
            'all_combined': ('2-all_combined_outputs_lt135km', '全损失组合'),
            'enhanced_multi_loss': ('2-enhanced_multi_loss_outputs_lt135km', '增强多损失')
        }
    
    def create_gaussian_weight_kernel(self, patch_size: int) -> np.ndarray:
        """创建高斯权重核 - 借鉴成功方法"""
        y, x = np.ogrid[:patch_size, :patch_size]
        center = patch_size / 2.0
        sigma = center * 0.5
        
        if sigma <= 0:
            sigma = 1.0
        
        weight_kernel = np.exp(-0.5 * ((x - (center - 0.5))**2 + (y - (center - 0.5))**2) / sigma**2)
        return weight_kernel.astype(np.float32)
    
    def load_model(self, model_path: str) -> Optional[nn.Module]:
        """加载模型"""
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            model = OptimizedUNet(
                in_channels=self.config.IN_CHANNELS,
                out_channels=self.config.OUT_CHANNELS,
                init_features=self.config.INIT_FEATURES
            )
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            logger.info(f"✓ 模型加载成功: {model_path}")
            return model
        
        except Exception as e:
            logger.error(f"模型加载失败 {model_path}: {e}")
            return None
    
    def load_data(self) -> Tuple[xr.Dataset, xr.DataArray]:
        """加载预测数据"""
        logger.info("加载数据...")
        
        # 加载特征
        features = xr.open_dataset(self.config.FEATURES_PATH)
        
        # 加载长波背景
        longwave = xr.open_dataarray(self.config.LONGWAVE_PATH)
        
        # 应用区域选择
        if self.config.PREDICTION_REGION:
            region = self.config.PREDICTION_REGION
            features = features.sel(
                lat=region['lat_range'],
                lon=region['lon_range']
            )
            longwave = longwave.sel(
                lat=region['lat_range'],
                lon=region['lon_range']
            )
        
        logger.info(f"数据形状: features={features.sizes}, longwave={longwave.shape}")
        return features, longwave
    
    def predict_large_area(self, model: nn.Module, features: xr.Dataset) -> np.ndarray:
        """大区域预测 - 借鉴成功的拼接策略"""
        logger.info("开始大区域预测...")
        
        lat_size = len(features.lat)
        lon_size = len(features.lon)
        
        # 初始化输出
        prediction = np.zeros((lat_size, lon_size), dtype=np.float32)
        count_map = np.zeros((lat_size, lon_size), dtype=np.float32)
        
        # 创建高斯权重核
        weight_kernel = self.create_gaussian_weight_kernel(self.config.PATCH_SIZE)
        
        # 收集patches
        patches = []
        positions = []
        
        for i in range(0, lat_size - self.config.PATCH_SIZE + 1, self.config.STRIDE):
            for j in range(0, lon_size - self.config.PATCH_SIZE + 1, self.config.STRIDE):
                lat_start, lon_start = i, j
                lat_end = lat_start + self.config.PATCH_SIZE
                lon_end = lon_start + self.config.PATCH_SIZE
                
                # 提取特征patch
                feature_patches = []
                all_valid = True
                
                for var in ['GA', 'DOV_EW', 'DOV_NS', 'VGG']:
                    if var in features:
                        patch = features[var].isel(
                            lat=slice(lat_start, lat_end),
                            lon=slice(lon_start, lon_end)
                        ).values
                        
                        if np.all(np.isnan(patch)):
                            all_valid = False
                            break
                        
                        feature_patches.append(patch)
                    else:
                        all_valid = False
                        break
                
                if all_valid and len(feature_patches) == 4:
                    feature_array = np.stack(feature_patches, axis=0)
                    feature_array = np.nan_to_num(feature_array, 0)
                    
                    patches.append(feature_array)
                    positions.append((lat_start, lon_start, lat_end, lon_end))
        
        logger.info(f"提取了 {len(patches)} 个有效patch")
        
        # 批量预测
        with torch.no_grad():
            for i in tqdm(range(0, len(patches), self.config.BATCH_SIZE), desc="预测进度"):
                batch_patches = patches[i:i + self.config.BATCH_SIZE]
                batch_positions = positions[i:i + self.config.BATCH_SIZE]
                
                # 转换为张量
                batch_tensor = torch.from_numpy(
                    np.array(batch_patches)
                ).float().to(device)
                
                # 预测
                batch_pred = model(batch_tensor)
                batch_pred = batch_pred.cpu().numpy()
                
                # 权重累积
                for j, (lat_start, lon_start, lat_end, lon_end) in enumerate(batch_positions):
                    pred_patch = batch_pred[j, 0]
                    
                    # 处理异常值
                    pred_patch = np.nan_to_num(pred_patch, 0)
                    
                    # 权重累积
                    weighted_output = pred_patch * weight_kernel
                    
                    prediction[lat_start:lat_end, lon_start:lon_end] += weighted_output
                    count_map[lat_start:lat_end, lon_start:lon_end] += weight_kernel
        
        # 安全归一化
        count_map_safe = np.where(count_map > 1e-9, count_map, 1.0)
        prediction_averaged = np.divide(prediction, count_map_safe)
        
        logger.info("✓ 大区域预测完成")
        return prediction_averaged
    
    def predict_all_models(self):
            """使用所有模型进行大区域预测"""
            logger.info("=== 开始多模型大区域预测 ===")
            
            features, longwave = self.load_data()
            results = {}
            
            for model_name, (output_dir, description) in self.model_configs.items():
                logger.info(f"\n处理模型: {description}")
                
                # --- MODIFICATION START ---
                # Adjust model_file_key_part to match the varying parts of the actual filenames
                model_file_key_part = model_name 
                if model_name == 'mse_grad':
                    model_file_key_part = 'weighted_mse_grad'
                elif model_name == 'mse_fft':
                    model_file_key_part = 'weighted_mse_fft'
                elif model_name == 'all_combined':
                    model_file_key_part = 'all'
                # For 'enhanced_multi_loss', if its filename was, for example,
                # 'best_model_enhanced_multi_loss_lt135km.pth' (without repeating 'enhanced_multi_loss'),
                # a specific rule would be needed here.
                # If model_name == 'enhanced_multi_loss':
                # model_file_key_part = '' # Or some other specific identifier

                # Construct the most likely filename based on observed patterns
                # e.g., best_model_enhanced_multi_loss_baseline_lt135km.pth
                specific_expected_filename = f'best_model_enhanced_multi_loss_{model_file_key_part}_lt135km.pth'
                
                # Special case for 'enhanced_multi_loss' if its naming is simpler and known
                # This is an example if the 'enhanced_multi_loss' model filename was simpler,
                # and assuming 'enhanced_multi_loss' is the unique identifier part.
                # if model_name == 'enhanced_multi_loss' and model_file_key_part == 'enhanced_multi_loss':
                #     # Example: if filename was just 'best_model_enhanced_multi_loss_lt135km.pth'
                #     specific_expected_filename = 'best_model_enhanced_multi_loss_lt135km.pth'


                possible_model_files = [
                    specific_expected_filename,  # Try the most accurately constructed name first
                    # Original fallbacks, in case some models use simpler naming
                    'best_model.pth',
                    'model_best.pth', 
                    'final_model.pth',
                    f'best_model_{model_name}.pth' # Original generic pattern based on model_name
                ]
                # --- MODIFICATION END ---
                
                model_path = None
                # Construct the full path to the directory where models for the current configuration are stored
                model_directory_path = f"/mnt/data5/06-Projects/05-unet/output/{output_dir}"

                for filename_to_check in possible_model_files:
                    candidate_path = os.path.join(model_directory_path, filename_to_check)
                    if os.path.exists(candidate_path):
                        model_path = candidate_path
                        logger.info(f"Found model file: {model_path}") # Log successful finding
                        break
                
                if model_path is None:
                    logger.warning(f"在 {output_dir} 中未找到模型文件. Tried patterns: {possible_model_files}")
                    # 列出目录中的所有文件以便调试 (already present and helpful)
                    try:
                        files = os.listdir(model_directory_path)
                        logger.info(f"目录中的文件: {files}")
                    except FileNotFoundError:
                        logger.warning(f"目录不存在: {model_directory_path}")
                    except Exception as e:
                        logger.warning(f"无法访问目录: {model_directory_path}. Error: {e}")
                    continue
                
                # 加载模型
                model = self.load_model(model_path)
                if model is None:
                    continue
                
                # 预测
                prediction = self.predict_large_area(model, features)
                
                # 保存结果 (remainder of the loop is likely fine)
                result_da = xr.DataArray(
                    prediction,
                    coords=features.coords,
                    dims=['lat', 'lon'],
                    name=f'prediction_{model_name}'
                )
                
                output_path = os.path.join(
                    self.config.OUTPUT_DIR, 
                    f'large_area_prediction_{model_name}.nc'
                )
                result_da.to_netcdf(output_path)
                
                results[model_name] = {
                    'prediction': result_da,
                    'description': description,
                    'output_path': output_path
                }
                
                logger.info(f"✓ {description} 预测完成，保存到: {output_path}")
            
            # 创建对比可视化
            if results:
                self.create_comparison_visualization(results, features)
                logger.info(f"\n🎉 部分或所有模型大区域预测完成！") # Adjusted message
                logger.info(f"结果保存在: {self.config.OUTPUT_DIR}")
            else:
                logger.error("❌ 没有成功加载并处理任何模型") # Adjusted message
            
            return results
    
    def create_comparison_visualization(self, results: Dict, features: xr.Dataset):
        """创建多模型对比可视化"""
        logger.info("创建对比可视化...")
        
        n_models = len(results)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 显示原始重力异常
        if 'GA' in features:
            im0 = axes[0].pcolormesh(
                features.lon, features.lat, features['GA'].values,
                cmap='RdBu_r', shading='auto'
            )
            axes[0].set_title('SWOT重力异常')
            plt.colorbar(im0, ax=axes[0])
        
        # 显示各模型预测结果
        for i, (model_name, result_info) in enumerate(results.items()):
            if i + 1 >= len(axes):
                break
            
            prediction = result_info['prediction']
            im = axes[i + 1].pcolormesh(
                prediction.lon, prediction.lat, prediction.values,
                cmap='terrain', shading='auto'
            )
            axes[i + 1].set_title(f"{result_info['description']}")
            plt.colorbar(im, ax=axes[i + 1])
        
        # 隐藏多余的子图
        for i in range(len(results) + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.config.OUTPUT_DIR, 'large_area_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 对比图保存到: {output_path}")

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('large_area_prediction.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    config = LargeAreaPredictionConfig()
    predictor = LargeAreaPredictor(config)
    
    try:
        results = predictor.predict_all_models()
        logger.info("✅ 大区域预测任务成功完成！")
        return 0
    except Exception as e:
        logger.error(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())