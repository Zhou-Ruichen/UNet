#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强海底地形数据准备 - 集成TID权重版本 (预测0-135km波段)

在原有数据准备流程基础上，新增基于TID的损失权重生成功能：
- 读取GEBCO_2024_TID.nc文件
- 生成基于数据质量的权重掩膜
- 为船测数据点（TID=10,11）分配高权重
- 为其他区域分配默认权重
- 保存权重掩膜供训练时使用

主要更新:
1. 新增TID权重生成模块
2. 优化变量命名和文档结构
3. 增加权重掩膜可视化
4. 保留原有的完整数据处理流程

作者: 周瑞宸
日期: 2025-05-25
版本: v2.1 (集成TID权重)
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import logging
from typing import Dict, Tuple, Optional, Union, List
import platform

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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_bathymetry_prep')

# 配置参数
INPUT_DIR = "/mnt/data5/00-Data/nc"
DATA_ROOT_DIR = "/mnt/data5/00-Data"
OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/1-refined_seafloor_data_preparation"
TRUTH_PATH = os.path.join(DATA_ROOT_DIR, "GEBCO_2024.nc")
TID_PATH = os.path.join(DATA_ROOT_DIR, "GEBCO_2024_TID.nc")

# SWOT数据路径
SWOT_PATHS = {
    "GA": os.path.join(INPUT_DIR, "grav_SWOT_02.nc"),
    "DOV_EW": os.path.join(INPUT_DIR, "east_SWOT_02.nc"),
    "DOV_NS": os.path.join(INPUT_DIR, "north_SWOT_02.nc"),
    "VGG": os.path.join(INPUT_DIR, "curv_SWOT_02.nc")
}

# 滤波参数
LONG_CUTOFF_WAVELENGTH = 135.0  # 长波截止波长(公里)

# TID权重配置
TID_WEIGHT_CONFIG = {
    "weight_ship_track": 5.0,    # 船测数据权重
    "weight_default": 1.0,       # 默认权重
    "weight_invalid": 0.0,       # 无效区域权重
    "ship_track_tids": [10, 11], # 船测数据的TID值
    "land_tids": [0, 1, 2]       # 陆地区域的TID值
}

# 可视化区域
VIS_REGION = {
    "lat_range": slice(0, 25),
    "lon_range": slice(105, 125)
}

# 南海区域定义
SOUTH_CHINA_SEA_REGION = {
    "lat_range": slice(0, 25),
    "lon_range": slice(105, 125),
    "name": "南海"
}


class TIDWeightGenerator:
    """TID权重生成器类"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def generate_weight_mask(self, tid_data: xr.DataArray) -> xr.DataArray:
        """
        基于TID数据生成权重掩膜
        
        参数:
            tid_data: TID DataArray
            
        返回:
            权重掩膜 DataArray
        """
        logger.info("基于TID数据生成损失权重掩膜...")
        
        # 初始化权重掩膜
        weights = np.full_like(tid_data.values, self.config["weight_default"], dtype=np.float32)
        
        # 为船测数据分配高权重
        for tid_val in self.config["ship_track_tids"]:
            ship_mask = (tid_data.values == tid_val)
            weights[ship_mask] = self.config["weight_ship_track"]
            ship_count = np.sum(ship_mask)
            logger.info(f"TID={tid_val}的船测数据点: {ship_count} 个，权重: {self.config['weight_ship_track']}")
        
        # 为陆地/无效区域分配零权重
        for tid_val in self.config["land_tids"]:
            land_mask = (tid_data.values == tid_val)
            weights[land_mask] = self.config["weight_invalid"]
            land_count = np.sum(land_mask)
            logger.info(f"TID={tid_val}的陆地/无效数据点: {land_count} 个，权重: {self.config['weight_invalid']}")
        
        # 创建权重DataArray
        weight_da = xr.DataArray(
            weights,
            coords=tid_data.coords,
            dims=tid_data.dims,
            name='tid_loss_weight_map'
        )
        
        # 添加元数据
        weight_da.attrs.update({
            'description': '基于TID的损失权重掩膜',
            'ship_track_weight': self.config["weight_ship_track"],
            'default_weight': self.config["weight_default"],
            'invalid_weight': self.config["weight_invalid"],
            'ship_track_tids': str(self.config["ship_track_tids"]),
            'land_tids': str(self.config["land_tids"])
        })
        
        # 统计信息
        total_points = weights.size
        ship_points = np.sum(np.isin(tid_data.values, self.config["ship_track_tids"]))
        land_points = np.sum(np.isin(tid_data.values, self.config["land_tids"]))
        other_points = total_points - ship_points - land_points
        
        logger.info(f"权重掩膜统计:")
        logger.info(f"  总点数: {total_points}")
        logger.info(f"  船测数据点: {ship_points} ({ship_points/total_points*100:.2f}%)")
        logger.info(f"  陆地/无效点: {land_points} ({land_points/total_points*100:.2f}%)")
        logger.info(f"  其他数据点: {other_points} ({other_points/total_points*100:.2f}%)")
        
        return weight_da
    
    def visualize_weights(self, weight_da: xr.DataArray, tid_data: xr.DataArray, 
                         region: Optional[Dict] = None) -> plt.Figure:
        """
        可视化权重掩膜和TID分布
        
        参数:
            weight_da: 权重掩膜
            tid_data: 原始TID数据
            region: 可视化区域
            
        返回:
            matplotlib图形
        """
        logger.info("创建TID权重可视化...")
        
        # 定义可视化区域
        if region is None:
            lat_range = slice(-30, 30)
            lon_range = slice(130, 180)
        else:
            lat_range = region['lat_range']
            lon_range = region['lon_range']
        
        # 提取区域数据
        weights_region = weight_da.sel(lat=lat_range, lon=lon_range)
        tid_region = tid_data.sel(lat=lat_range, lon=lon_range)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # TID分布图
        im1 = axes[0, 0].imshow(
            tid_region.values, 
            extent=[tid_region.lon.min(), tid_region.lon.max(), 
                   tid_region.lat.min(), tid_region.lat.max()],
            cmap='tab10', aspect='auto', origin='lower'
        )
        axes[0, 0].set_title('TID数据分布')
        axes[0, 0].set_xlabel('经度')
        axes[0, 0].set_ylabel('纬度')
        plt.colorbar(im1, ax=axes[0, 0], label='TID值')
        
        # 权重分布图
        im2 = axes[0, 1].imshow(
            weights_region.values,
            extent=[weights_region.lon.min(), weights_region.lon.max(),
                   weights_region.lat.min(), weights_region.lat.max()],
            cmap='Reds', aspect='auto', origin='lower'
        )
        axes[0, 1].set_title('损失权重分布')
        axes[0, 1].set_xlabel('经度')
        axes[0, 1].set_ylabel('纬度')
        plt.colorbar(im2, ax=axes[0, 1], label='权重值')
        
        # TID值统计直方图
        tid_flat = tid_region.values.flatten()
        tid_unique, tid_counts = np.unique(tid_flat, return_counts=True)
        axes[1, 0].bar(tid_unique, tid_counts, alpha=0.7)
        axes[1, 0].set_xlabel('TID值')
        axes[1, 0].set_ylabel('像素数量')
        axes[1, 0].set_title('TID值分布统计')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 权重值统计直方图
        weights_flat = weights_region.values.flatten()
        axes[1, 1].hist(weights_flat, bins=20, alpha=0.7, color='red')
        axes[1, 1].set_xlabel('权重值')
        axes[1, 1].set_ylabel('像素数量')
        axes[1, 1].set_title('权重值分布统计')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class EnhancedBathymetryProcessor:
    """
    增强的海底地形处理器 - 集成TID权重功能
    """

    def __init__(self, region=None):
        """
        初始化处理器
        
        参数:
            region: 可选的区域字典，包含'lat_range'和'lon_range'
        """
        self.long_cutoff_wavelength = LONG_CUTOFF_WAVELENGTH
        self.input_dir = INPUT_DIR
        self.output_dir = OUTPUT_DIR
        self.truth_path = TRUTH_PATH
        self.tid_path = TID_PATH
        self.swot_paths = SWOT_PATHS

        # 设置处理区域
        self.region = region if region is not None else SOUTH_CHINA_SEA_REGION
        logger.info(f"设置处理区域: {self.region['name']}")
        logger.info(f"纬度范围: {self.region['lat_range']}, 经度范围: {self.region['lon_range']}")

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录: {self.output_dir}")
        
        # 初始化TID权重生成器
        self.tid_weight_generator = TIDWeightGenerator(TID_WEIGHT_CONFIG)

    def load_truth_bathymetry(self) -> xr.DataArray:
        """加载真值海底地形数据"""
        logger.info(f"从以下路径加载真值海底地形: {self.truth_path}")

        try:
            ds = xr.open_dataset(self.truth_path)

            # 尝试识别海底地形变量
            bathy_var = None
            candidates = ['depth', 'elevation', 'bathymetry', 'filtered_elevation']

            for var in candidates:
                if var in ds:
                    bathy_var = var
                    break

            if bathy_var is None:
                for var in ds.data_vars:
                    if any(keyword in var.lower() for keyword in ['depth', 'bathy', 'elev']):
                        bathy_var = var
                        break

            if bathy_var is None:
                bathy_var = list(ds.data_vars)[0]
                logger.warning(f"无法识别海底地形变量。使用: {bathy_var}")
            else:
                logger.info(f"已识别海底地形变量: {bathy_var}")

            bathy_da = ds[bathy_var]

            # 应用区域选择
            if self.region is not None:
                logger.info(f"选择区域: {self.region['name']}")
                bathy_da = bathy_da.sel(lat=self.region['lat_range'], lon=self.region['lon_range'])

            bathy_da.name = 'bathy_true_original'
            return bathy_da

        except Exception as e:
            logger.error(f"加载真值海底地形失败: {e}")
            raise

    def load_tid_data(self) -> xr.DataArray:
        """加载TID数据"""
        logger.info(f"从以下路径加载TID数据: {self.tid_path}")
        
        try:
            ds = xr.open_dataset(self.tid_path)
            
            # 尝试识别TID变量
            tid_var = None
            candidates = ['tid', 'TID', 'type_id', 'type_identifier']
            
            for var in candidates:
                if var in ds:
                    tid_var = var
                    break
            
            if tid_var is None:
                # 如果没有找到明确的TID变量，使用第一个数据变量
                tid_var = list(ds.data_vars)[0]
                logger.warning(f"无法识别TID变量。使用: {tid_var}")
            else:
                logger.info(f"已识别TID变量: {tid_var}")
            
            tid_da = ds[tid_var]
            
            # 应用区域选择
            if self.region is not None:
                logger.info(f"选择TID区域: {self.region['name']}")
                tid_da = tid_da.sel(lat=self.region['lat_range'], lon=self.region['lon_range'])
            
            tid_da.name = 'tid_original'
            
            # 输出TID数据统计
            tid_unique, tid_counts = np.unique(tid_da.values, return_counts=True)
            logger.info("TID数据统计:")
            for tid_val, count in zip(tid_unique, tid_counts):
                percentage = count / tid_da.size * 100
                logger.info(f"  TID={tid_val}: {count} 个点 ({percentage:.2f}%)")
            
            return tid_da
            
        except Exception as e:
            logger.error(f"加载TID数据失败: {e}")
            raise

    def calculate_sigma_pixels(self, cutoff_wavelength_km: float, data_array: xr.DataArray) -> Tuple[float, float]:
        """根据波长计算高斯滤波的sigma值（像素单位）"""
        lat_coords = data_array['lat'].values
        lon_coords = data_array['lon'].values

        d_lat = np.abs(float(np.mean(np.diff(lat_coords))))
        d_lon = np.abs(float(np.mean(np.diff(lon_coords))))

        mean_lat_rad = np.radians(np.mean(lat_coords))

        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(mean_lat_rad)

        cutoff_lat_deg = cutoff_wavelength_km / km_per_deg_lat
        cutoff_lon_deg = cutoff_wavelength_km / km_per_deg_lon

        cutoff_lat_pixels = cutoff_lat_deg / d_lat
        cutoff_lon_pixels = cutoff_lon_deg / d_lon

        sigma_lat = cutoff_lat_pixels / 2.355
        sigma_lon = cutoff_lon_pixels / 2.355

        logger.info(f"网格分辨率: {d_lat:.6f}°纬度, {d_lon:.6f}°经度")
        logger.info(f"{cutoff_wavelength_km}公里波长 - Sigma值: {sigma_lat:.2f}像素(纬度), {sigma_lon:.2f}像素(经度)")

        return sigma_lat, sigma_lon

    def apply_gaussian_filter(self, data: xr.DataArray, sigma_lat: float, sigma_lon: float) -> xr.DataArray:
        """对DataArray应用二维高斯滤波，正确处理NaN值"""
        data_values = data.values.copy().astype(float)
        nan_mask = np.isnan(data_values)
        data_values[nan_mask] = 0

        filtered_values = gaussian_filter(
            data_values,
            sigma=[sigma_lat, sigma_lon],
            mode='reflect'
        )

        filtered_values[nan_mask] = np.nan

        filtered_da = xr.DataArray(
            filtered_values,
            coords=data.coords,
            dims=data.dims,
            name=f"{data.name or 'data'}_filtered"
        )

        filtered_da.attrs.update(data.attrs)
        return filtered_da

    def generate_longwave_background(self, bathy_true: xr.DataArray) -> xr.DataArray:
        """通过应用低通滤波器生成长波长背景（>135km）"""
        logger.info(f"生成长波长背景(>={self.long_cutoff_wavelength}公里)")

        sigma_lat, sigma_lon = self.calculate_sigma_pixels(
            self.long_cutoff_wavelength,
            bathy_true
        )

        bathy_true_longwave = self.apply_gaussian_filter(bathy_true, sigma_lat, sigma_lon)

        bathy_true_longwave.name = 'bathy_true_longwave'
        bathy_true_longwave.attrs['description'] = f'长波长背景海底地形(>={self.long_cutoff_wavelength}公里)'
        bathy_true_longwave.attrs['cutoff_wavelength_km'] = self.long_cutoff_wavelength
        bathy_true_longwave.attrs['sigma_lat_pixels'] = float(sigma_lat)
        bathy_true_longwave.attrs['sigma_lon_pixels'] = float(sigma_lon)
        bathy_true_longwave.attrs['filter_type'] = 'gaussian'

        return bathy_true_longwave

    def calculate_target_residual(self, bathy_true: xr.DataArray, 
                                 bathy_true_longwave: xr.DataArray) -> xr.DataArray:
        """计算目标残差海底地形（<135km）作为模型预测标签"""
        logger.info(f"计算目标残差海底地形(<{self.long_cutoff_wavelength}公里)")

        bathy_true_residual_lt135km = bathy_true - bathy_true_longwave
        bathy_true_residual_lt135km.name = 'bathy_true_residual_lt135km'
        bathy_true_residual_lt135km.attrs['description'] = f'目标残差海底地形(小于{self.long_cutoff_wavelength}公里的所有成分)'
        bathy_true_residual_lt135km.attrs['target_wavelength_range'] = f'0-{self.long_cutoff_wavelength}km'
        bathy_true_residual_lt135km.attrs['long_cutoff_wavelength_km'] = self.long_cutoff_wavelength

        return bathy_true_residual_lt135km

    def generate_target_label(self, bathy_true_residual_lt135km: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """生成最终的训练标签（标准化的目标残差地形）"""
        logger.info("生成模型训练的目标标签")

        # 直接使用残差作为目标标签
        label_target_bathy_lt135km = bathy_true_residual_lt135km.copy(deep=True)
        label_target_bathy_lt135km.name = 'label_target_bathy_lt135km'
        label_target_bathy_lt135km.attrs['description'] = f'目标标签海底地形(小于{self.long_cutoff_wavelength}公里的所有成分)'

        # 标准化目标标签
        logger.info("标准化目标标签(Z-score)")

        mean_val = float(np.nanmean(label_target_bathy_lt135km.values))
        std_val = float(np.nanstd(label_target_bathy_lt135km.values))

        logger.info(f"目标标签统计: 均值 = {mean_val:.6f}, 标准差 = {std_val:.6f}")

        standard_values = label_target_bathy_lt135km.values.copy()
        non_nan_mask = ~np.isnan(standard_values)
        standard_values[non_nan_mask] = (standard_values[non_nan_mask] - mean_val) / std_val

        label_target_bathy_lt135km_standardized = xr.DataArray(
            standard_values,
            coords=label_target_bathy_lt135km.coords,
            dims=label_target_bathy_lt135km.dims,
            name='label_target_bathy_lt135km_standardized'
        )

        label_target_bathy_lt135km_standardized.attrs.update(label_target_bathy_lt135km.attrs)
        label_target_bathy_lt135km_standardized.attrs['description'] = f'标准化目标标签(小于{self.long_cutoff_wavelength}公里, Z-score归一化)'
        label_target_bathy_lt135km_standardized.attrs['normalization'] = 'z-score'
        label_target_bathy_lt135km_standardized.attrs['mean'] = mean_val
        label_target_bathy_lt135km_standardized.attrs['std'] = std_val

        return label_target_bathy_lt135km, label_target_bathy_lt135km_standardized

    def load_swot_data(self) -> Dict[str, xr.DataArray]:
        """加载SWOT数据(GA, DOV_EW, DOV_NS, VGG)"""
        logger.info("加载SWOT数据...")

        swot_data = {}

        for key, path in self.swot_paths.items():
            try:
                logger.info(f"加载{key}，路径: {path}")
                ds = xr.open_dataset(path)

                if key in ds:
                    var_name = key
                else:
                    candidates = [var for var in ds.data_vars if key.lower() in var.lower()]
                    if candidates:
                        var_name = candidates[0]
                    else:
                        var_name = list(ds.data_vars)[0]
                        logger.warning(f"无法识别{key}变量。使用: {var_name}")

                swot_data[key] = ds[var_name]

                # 应用区域选择
                if self.region is not None:
                    logger.info(f"选择{key}区域: {self.region['name']}")
                    swot_data[key] = swot_data[key].sel(lat=self.region['lat_range'], lon=self.region['lon_range'])

                swot_data[key].name = f'{key.lower()}_original'

            except Exception as e:
                logger.error(f"加载{key}失败: {e}")
                swot_data[key] = None

        return swot_data

    def preprocess_swot_features(self, swot_data: Dict[str, xr.DataArray],
                               target_grid: xr.DataArray) -> Dict[str, xr.DataArray]:
        """预处理SWOT特征，提取<135km成分并与目标网格对齐"""
        logger.info("预处理SWOT特征...")

        processed_features = {}

        for key, data in swot_data.items():
            if data is None:
                logger.warning(f"跳过{key} - 数据不可用")
                continue

            logger.info(f"处理{key}...")

            if data.name is None:
                data.name = f'{key.lower()}_original'

            # 1. 重新网格化以匹配目标（如需要）
            if not (data.lon.equals(target_grid.lon) and data.lat.equals(target_grid.lat)):
                logger.info(f"将{key}重新网格化以匹配目标网格")

                ds = data.to_dataset(name=key)
                target_lat = target_grid.lat.values
                target_lon = target_grid.lon.values

                try:
                    regridded_ds = ds.interp(
                        lat=target_lat,
                        lon=target_lon,
                        method='linear'
                    )
                    data = regridded_ds[key]
                    logger.info(f"{key}重新网格化完成")
                except Exception as e:
                    logger.error(f"{key}重新网格化失败: {e}")
                    continue

            # 2. 提取<135km成分（移除长波成分）
            logger.info(f"提取{key}的<{self.long_cutoff_wavelength}km成分")

            # 计算长截止的sigma值
            sigma_lat_long, sigma_lon_long = self.calculate_sigma_pixels(
                self.long_cutoff_wavelength,
                data
            )

            # 应用低通滤波器提取长波长成分
            longwave_component = self.apply_gaussian_filter(data.copy(deep=True), sigma_lat_long, sigma_lon_long)
            longwave_component.name = f"{key.lower()}_longwave"

            # 相减得到<135km成分
            lt135km_data = data - longwave_component
            lt135km_data.name = f"{key.lower()}_lt135km"

            # 3. 标准化(Z-score)
            logger.info(f"标准化{key}")

            mean_val = float(np.nanmean(lt135km_data.values))
            std_val = float(np.nanstd(lt135km_data.values))

            logger.info(f"{key}统计: 均值 = {mean_val:.6f}, 标准差 = {std_val:.6f}")

            standard_values = lt135km_data.values.copy()
            non_nan_mask = ~np.isnan(standard_values)
            standard_values[non_nan_mask] = (standard_values[non_nan_mask] - mean_val) / std_val

            standardized_feature = xr.DataArray(
                standard_values,
                coords=lt135km_data.coords,
                dims=lt135km_data.dims,
                name=f'{key.lower()}_lt135km_std'
            )

            # 添加元数据
            standardized_feature.attrs['description'] = f'处理后的{key}(小于{self.long_cutoff_wavelength}公里, Z-score)'
            standardized_feature.attrs['target_wavelength_range'] = f'0-{self.long_cutoff_wavelength}km'
            standardized_feature.attrs['long_cutoff_wavelength_km'] = self.long_cutoff_wavelength
            standardized_feature.attrs['normalization'] = 'z-score'
            standardized_feature.attrs['mean'] = mean_val
            standardized_feature.attrs['std'] = std_val

            processed_features[key] = standardized_feature

        return processed_features

    def create_visualization(self, bathy_true: xr.DataArray,
                           bathy_true_longwave: xr.DataArray,
                           bathy_true_residual_lt135km: xr.DataArray,
                           label_target_bathy_lt135km: xr.DataArray,
                           label_target_bathy_lt135km_standardized: xr.DataArray,
                           processed_features: Dict[str, xr.DataArray],
                           tid_weight_map: xr.DataArray,
                           region: Optional[Dict] = None) -> plt.Figure:
        """创建数据准备步骤的可视化（包含TID权重）"""
        logger.info("创建增强可视化（包含TID权重）...")

        # 定义可视化区域
        if region is None:
            lat_range = slice(-30, 30)
            lon_range = slice(130, 180)
        else:
            lat_range = region['lat_range']
            lon_range = region['lon_range']

        # 提取区域
        bt = bathy_true.sel(lat=lat_range, lon=lon_range)
        btl = bathy_true_longwave.sel(lat=lat_range, lon=lon_range)
        btr = bathy_true_residual_lt135km.sel(lat=lat_range, lon=lon_range)
        lbl = label_target_bathy_lt135km.sel(lat=lat_range, lon=lon_range)
        lbls = label_target_bathy_lt135km_standardized.sel(lat=lat_range, lon=lon_range)
        weights = tid_weight_map.sel(lat=lat_range, lon=lon_range)

        # 创建图形
        n_cols = 3
        n_rows = 3 + (len(processed_features) + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(20, 5 * n_rows))

        # 真值海底地形
        ax1 = fig.add_subplot(n_rows, n_cols, 1)
        bt.plot(ax=ax1, cmap='viridis', add_colorbar=True)
        ax1.set_title('真值海底地形')

        # 长波长背景
        ax2 = fig.add_subplot(n_rows, n_cols, 2)
        btl.plot(ax=ax2, cmap='viridis', add_colorbar=True)
        ax2.set_title(f'长波长背景\n(≥{self.long_cutoff_wavelength}公里)')

        # 目标残差（<135km）
        ax3 = fig.add_subplot(n_rows, n_cols, 3)
        btr.plot(ax=ax3, cmap='RdBu_r', add_colorbar=True, center=0)
        ax3.set_title(f'目标残差海底地形\n(<{self.long_cutoff_wavelength}公里)')

        # 目标标签（物理单位）
        ax4 = fig.add_subplot(n_rows, n_cols, 4)
        lbl.plot(ax=ax4, cmap='RdBu_r', add_colorbar=True, center=0)
        ax4.set_title(f'目标标签(物理单位)\n(<{self.long_cutoff_wavelength}公里)')

        # 标准化目标标签
        ax5 = fig.add_subplot(n_rows, n_cols, 5)
        lbls.plot(ax=ax5, cmap='RdBu_r', add_colorbar=True, center=0)
        ax5.set_title('标准化目标标签(Z-score)')

        # TID权重掩膜
        ax6 = fig.add_subplot(n_rows, n_cols, 6)
        weights.plot(ax=ax6, cmap='Reds', add_colorbar=True)
        ax6.set_title('TID损失权重掩膜')

        # 目标标签的直方图
        ax7 = fig.add_subplot(n_rows, n_cols, 7)

        lbl_flat = lbl.values.flatten()
        lbl_flat = lbl_flat[~np.isnan(lbl_flat)]

        lbls_flat = lbls.values.flatten()
        lbls_flat = lbls_flat[~np.isnan(lbls_flat)]

        if len(lbl_flat) > 0:
            ax7.hist(lbl_flat, bins=50, alpha=0.5, label='物理单位', color='blue')

        if len(lbls_flat) > 0:
            ax7_twin = ax7.twinx()
            ax7_twin.hist(lbls_flat, bins=50, alpha=0.5, label='标准化', color='red')
            ax7_twin.axvline(x=0, color='k', linestyle='--')
            ax7_twin.axvline(x=1, color='k', linestyle=':')
            ax7_twin.axvline(x=-1, color='k', linestyle=':')
            ax7_twin.set_ylabel('频率(标准化)')

        ax7.set_xlabel('值')
        ax7.set_ylabel('频率(物理单位)')
        ax7.set_title('目标标签值分布')

        lines1, labels1 = ax7.get_legend_handles_labels()
        if len(lbls_flat) > 0:
            lines2, labels2 = ax7_twin.get_legend_handles_labels()
            ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax7.legend(loc='upper right')

        # 权重统计直方图
        ax8 = fig.add_subplot(n_rows, n_cols, 8)
        weights_flat = weights.values.flatten()
        ax8.hist(weights_flat, bins=20, alpha=0.7, color='red')
        ax8.set_xlabel('权重值')
        ax8.set_ylabel('像素数量')
        ax8.set_title('TID权重分布统计')
        ax8.grid(True, alpha=0.3)

        # SWOT特征可视化
        start_idx = 9
        for i, (key, feature) in enumerate(processed_features.items()):
            ax = fig.add_subplot(n_rows, n_cols, start_idx + i)
            feature.sel(lat=lat_range, lon=lon_range).plot(
                ax=ax, cmap='RdBu_r', add_colorbar=True, center=0
            )
            ax.set_title(f'处理后的{key}\n(<{self.long_cutoff_wavelength}公里, Z-score)')

        plt.tight_layout()
        return fig

    def save_outputs(self, bathy_true: xr.DataArray,
                    bathy_true_longwave: xr.DataArray,
                    bathy_true_residual_lt135km: xr.DataArray,
                    label_target_bathy_lt135km: xr.DataArray,
                    label_target_bathy_lt135km_standardized: xr.DataArray,
                    processed_features: Dict[str, xr.DataArray],
                    tid_weight_map: xr.DataArray,
                    visualization_fig: plt.Figure) -> None:
        """将所有输出保存到磁盘（包含TID权重）"""
        logger.info("保存输出（包含TID权重）...")

        # 保存海底地形组件
        longwave_path = os.path.join(self.output_dir, 'Bathy_True_Longwave.nc')
        residual_path = os.path.join(self.output_dir, 'Bathy_True_Residual_LT135km.nc')
        target_path = os.path.join(self.output_dir, 'Label_Target_Bathy_LT135km.nc')
        target_std_path = os.path.join(self.output_dir, 'Label_Target_Bathy_LT135km_Standardized.nc')

        logger.info(f"保存长波长背景到: {longwave_path}")
        bathy_true_longwave.to_netcdf(longwave_path)

        logger.info(f"保存目标残差海底地形到: {residual_path}")
        bathy_true_residual_lt135km.to_netcdf(residual_path)

        logger.info(f"保存目标标签到: {target_path}")
        label_target_bathy_lt135km.to_netcdf(target_path)

        logger.info(f"保存标准化目标标签到: {target_std_path}")
        label_target_bathy_lt135km_standardized.to_netcdf(target_std_path)

        # 保存处理后的SWOT特征
        features_path = os.path.join(self.output_dir, 'Features_SWOT_LT135km_Standardized.nc')
        logger.info(f"保存处理后的SWOT特征到: {features_path}")

        features_ds = xr.Dataset()
        for key, feature in processed_features.items():
            features_ds[key] = feature

        features_ds.attrs['description'] = f'用于海底地形预测的处理后SWOT特征(小于{self.long_cutoff_wavelength}公里, Z-score)'
        features_ds.attrs['target_wavelength_range'] = f'0-{self.long_cutoff_wavelength}km'
        features_ds.attrs['long_cutoff_wavelength_km'] = self.long_cutoff_wavelength

        features_ds.to_netcdf(features_path)

        # *** 新增：保存TID权重掩膜 ***
        tid_weights_path = os.path.join(self.output_dir, 'TID_Loss_Weights.nc')
        logger.info(f"保存TID权重掩膜到: {tid_weights_path}")
        tid_weight_map.to_netcdf(tid_weights_path)

        # 保存可视化
        viz_path = os.path.join(self.output_dir, 'Enhanced_Data_Preparation_Visualization_LT135km.png')
        logger.info(f"保存可视化到: {viz_path}")
        visualization_fig.savefig(viz_path, dpi=300, bbox_inches='tight')

        # 创建包含关键参数的摘要文件
        summary_path = os.path.join(self.output_dir, 'enhanced_data_preparation_summary_lt135km.txt')
        logger.info(f"保存摘要到: {summary_path}")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 增强海底地形数据准备摘要 (集成TID权重版本 - 预测0-135km波段) ===\n\n")
            f.write(f"日期: {np.datetime64('now')}\n\n")

            f.write("参数:\n")
            f.write(f"  - 长波长截止: {self.long_cutoff_wavelength} 公里\n")
            f.write(f"  - 目标预测波段: 0-{self.long_cutoff_wavelength} 公里\n")
            f.write(f"  - 处理区域: {self.region['name']}\n\n")

            f.write("TID权重配置:\n")
            f.write(f"  - 船测数据权重: {TID_WEIGHT_CONFIG['weight_ship_track']}\n")
            f.write(f"  - 默认数据权重: {TID_WEIGHT_CONFIG['weight_default']}\n")
            f.write(f"  - 无效区域权重: {TID_WEIGHT_CONFIG['weight_invalid']}\n")
            f.write(f"  - 船测TID值: {TID_WEIGHT_CONFIG['ship_track_tids']}\n")
            f.write(f"  - 陆地TID值: {TID_WEIGHT_CONFIG['land_tids']}\n\n")

            f.write("输出:\n")
            f.write(f"  1. 长波长背景: {os.path.basename(longwave_path)}\n")
            f.write(f"  2. 目标残差海底地形: {os.path.basename(residual_path)}\n")
            f.write(f"  3. 目标标签(物理单位): {os.path.basename(target_path)}\n")
            f.write(f"  4. 标准化目标标签: {os.path.basename(target_std_path)}\n")
            f.write(f"  5. 处理后的SWOT特征: {os.path.basename(features_path)}\n")
            f.write(f"  6. TID权重掩膜: {os.path.basename(tid_weights_path)}\n")
            f.write(f"  7. 可视化: {os.path.basename(viz_path)}\n\n")

            f.write("标准化统计:\n")
            f.write(f"  - 目标标签均值: {label_target_bathy_lt135km_standardized.attrs['mean']:.6f}\n")
            f.write(f"  - 目标标签标准差: {label_target_bathy_lt135km_standardized.attrs['std']:.6f}\n\n")

            f.write("SWOT特征:\n")
            for key, feature in processed_features.items():
                f.write(f"  - {key}: 均值={feature.attrs['mean']:.6f}, 标准差={feature.attrs['std']:.6f}\n")

            # 添加TID权重统计
            f.write("\nTID权重统计:\n")
            weights_flat = tid_weight_map.values.flatten()
            f.write(f"  - 权重范围: {np.min(weights_flat):.1f} - {np.max(weights_flat):.1f}\n")
            f.write(f"  - 平均权重: {np.mean(weights_flat):.3f}\n")
            f.write(f"  - 高权重点数: {np.sum(weights_flat == TID_WEIGHT_CONFIG['weight_ship_track'])}\n")
            f.write(f"  - 零权重点数: {np.sum(weights_flat == 0)}\n")

    def process(self) -> None:
        """执行完整的增强数据准备过程（包含TID权重生成）"""
        logger.info("开始增强数据准备过程（集成TID权重）...")

        # 1. 加载真值海底地形
        bathy_true = self.load_truth_bathymetry()

        # 2. 加载TID数据
        tid_data = self.load_tid_data()

        # 3. 生成TID权重掩膜
        tid_weight_map = self.tid_weight_generator.generate_weight_mask(tid_data)

        # 4. 创建TID权重可视化
        tid_viz_fig = self.tid_weight_generator.visualize_weights(
            tid_weight_map, tid_data, region=VIS_REGION
        )
        
        # 保存TID权重可视化
        tid_viz_path = os.path.join(self.output_dir, 'TID_Weights_Visualization.png')
        logger.info(f"保存TID权重可视化到: {tid_viz_path}")
        tid_viz_fig.savefig(tid_viz_path, dpi=300, bbox_inches='tight')
        plt.close(tid_viz_fig)

        # 5. 生成长波长背景
        bathy_true_longwave = self.generate_longwave_background(bathy_true)

        # 6. 计算目标残差海底地形（<135km）
        bathy_true_residual_lt135km = self.calculate_target_residual(bathy_true, bathy_true_longwave)

        # 7. 生成目标标签
        label_target_bathy_lt135km, label_target_bathy_lt135km_standardized = self.generate_target_label(bathy_true_residual_lt135km)

        # 8. 加载和处理SWOT数据
        swot_data = self.load_swot_data()
        processed_features = self.preprocess_swot_features(swot_data, label_target_bathy_lt135km_standardized)

        # 9. 创建综合可视化
        visualization_fig = self.create_visualization(
            bathy_true,
            bathy_true_longwave,
            bathy_true_residual_lt135km,
            label_target_bathy_lt135km,
            label_target_bathy_lt135km_standardized,
            processed_features,
            tid_weight_map,
            region=VIS_REGION
        )

        # 10. 保存输出
        self.save_outputs(
            bathy_true,
            bathy_true_longwave,
            bathy_true_residual_lt135km,
            label_target_bathy_lt135km,
            label_target_bathy_lt135km_standardized,
            processed_features,
            tid_weight_map,
            visualization_fig
        )

        logger.info("增强数据准备过程成功完成！（集成TID权重）")


def main():
    """主函数，运行增强数据准备过程"""
    try:
        processor = EnhancedBathymetryProcessor(region=SOUTH_CHINA_SEA_REGION)
        processor.process()

    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    main()