#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的Swin U-Net海底地形精化训练 - 支持多种损失函数配置

集成功能：
1. 完整的Swin U-Net模型实现（Transformer+CNN混合架构）
2. 多种预设损失配置选择（baseline、TID加权、梯度增强等）
3. 命令行接口支持
4. TID权重支持
5. 窗口注意力机制和移动窗口

主要特性:
- 分层的Swin Transformer编码器 - 捕获多尺度特征和全局上下文
- CNN解码器 - 精确的空间定位和细节恢复
- 跳跃连接增强 - 结合浅层和深层特征
- 可配置的损失函数组合
- 完全兼容现有训练框架

使用方法:
python swin_unet_complete.py --loss_config baseline
python swin_unet_complete.py --loss_config tid_weighted --epochs 100
python swin_unet_complete.py --loss_config tid_gradient --batch_size 4
python swin_unet_complete.py --list_configs

作者: 周瑞宸
日期: 2025-05-26
版本: v2.0 (完整合并版本)
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
import math
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple, List

# 导入现有模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_training_multi_loss import (
    EnhancedTrainingConfig, EnhancedDataPreparation,
    EnhancedBathymetryDataset, MultiLossFunction,
    device, setup_chinese_font
)

setup_chinese_font()
logger = logging.getLogger('swin_unet_complete')


# ==================== 损失函数配置预设 ====================
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
    },
    
    'all_combined': {
        'name': '全损失组合',
        'loss_combination': 'all',
        'pixel_loss_weight': 1.0,
        'gradient_loss_weight': 0.5,
        'spectral_loss_weight': 0.3,
        'pixel_loss_type': 'mse',
        'use_tid_weights': True,
        'tid_weight_scale': 1.0,
        'gradient_method': 'sobel',
        'spectral_focus_high_freq': True,
        'high_freq_weight_factor': 2.5,
        'description': '使用所有损失组合：MSE + 梯度 + 频谱，全面优化'
    }
}


# ==================== Swin Transformer 核心组件 ====================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将特征图分割为窗口
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # 确保 H 和 W 至少等于 window_size
    if not (H >= window_size and W >= window_size):
        logger.warning(f"window_partition called with H={H} or W={W} < window_size={window_size}. Errors may occur.")

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将窗口合并回特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    if num_windows_h == 0 or num_windows_w == 0:
        logger.error(f"window_reverse: num_windows_h or num_windows_w is 0. H={H}, W={W}, ws={window_size}")
        raise ValueError(f"window_reverse: H ({H}) or W ({W}) is too small for window_size ({window_size}) leading to zero windows.")

    B = int(windows.shape[0] / (num_windows_h * num_windows_w))
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        if isinstance(window_size, int):
            self.window_size_tuple = (window_size, window_size)
        else:
            self.window_size_tuple = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size_tuple[0] - 1) * (2 * self.window_size_tuple[1] - 1), num_heads))

        # 获取每个token的相对位置索引
        coords_h = torch.arange(self.window_size_tuple[0])
        coords_w = torch.arange(self.window_size_tuple[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size_tuple[0] - 1
        relative_coords[:, :, 1] += self.window_size_tuple[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_tuple[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size_tuple[0] * self.window_size_tuple[1], self.window_size_tuple[0] * self.window_size_tuple[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.H = None  # Set by BasicLayer
        self.W = None  # Set by BasicLayer

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, attn_mask_input):
        """
        Args:
            x: input features with shape of (B, L, C)
            attn_mask_input: attention mask
        """
        H_in, W_in = self.H, self.W
        B, L, C = x.shape
        assert L == H_in * W_in, f"Input L={L} != H_in*W_in={H_in*W_in}"

        shortcut = x
        x_norm1 = self.norm1(x)
        x_spatial = x_norm1.view(B, H_in, W_in, C)

        # 如果需要填充以满足窗口大小要求
        H_eff, W_eff = H_in, W_in
        pad_b_op, pad_r_op = 0, 0
        needs_padding = False

        if H_in < self.window_size:
            pad_b_op = self.window_size - H_in
            H_eff = self.window_size
            needs_padding = True
        if W_in < self.window_size:
            pad_r_op = self.window_size - W_in
            W_eff = self.window_size
            needs_padding = True

        x_to_shift = x_spatial
        if needs_padding:
            x_permuted = x_spatial.permute(0, 3, 1, 2)  # B, C, H_in, W_in
            x_padded_permuted = F.pad(x_permuted, (0, pad_r_op, 0, pad_b_op))  # Pad R, B
            x_to_shift = x_padded_permuted.permute(0, 2, 3, 1)  # B, H_eff, W_eff, C

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x_to_shift, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_to_shift

        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask_input)

        # 合并窗口
        attn_windows_merged = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x_reverted = window_reverse(attn_windows_merged, self.window_size, H_eff, W_eff)

        # 反向循环移位
        if self.shift_size > 0:
            x_rolled_back = torch.roll(shifted_x_reverted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_rolled_back = shifted_x_reverted

        # 如果进行了填充，则裁剪回原始大小
        if needs_padding:
            x_unpadded_spatial = x_rolled_back[:, :H_in, :W_in, :].contiguous()
        else:
            x_unpadded_spatial = x_rolled_back

        x_final_seq = x_unpadded_spatial.view(B, H_in * W_in, C)

        # 残差连接和FFN
        x = shortcut + self.drop_path(x_final_seq)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H: int, W: int):
        """
        Args:
            x: input features with shape of (B, L, C)
            H, W: spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "PatchMerging: Input L mismatch"
        assert C == self.dim, "PatchMerging: Input C mismatch"

        x_spatial = x.view(B, H, W, C)

        # 如果H或W是奇数，进行填充
        H_padded, W_padded = H, W
        if H % 2 == 1:
            H_padded += 1
        if W % 2 == 1:
            W_padded += 1

        x_to_sample = x_spatial
        if H_padded != H or W_padded != W:
            x_perm = x_spatial.permute(0, 3, 1, 2)  # B, C, H, W
            x_padded_perm = F.pad(x_perm, (0, W_padded - W, 0, H_padded - H))  # pad right, bottom
            x_to_sample = x_padded_perm.permute(0, 2, 3, 1)  # B, H_padded, W_padded, C

        # 下采样
        x0 = x_to_sample[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x_to_sample[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x_to_sample[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x_to_sample[:, 1::2, 1::2, :]  # B H/2 W/2 C

        merged_x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        merged_x_seq = merged_x.view(B, -1, 4 * C)
        merged_x_seq_norm = self.norm(merged_x_seq)
        output_seq = self.reduction(merged_x_seq_norm)
        return output_seq


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # 构建blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function"""
        attn_mask = None
        if self.shift_size > 0:
            # 计算SW-MSA的注意力掩码
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h_s in h_slices:
                for w_s in w_slices:
                    img_mask[:, h_s, w_s, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        x_processed_blocks = x
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x_processed_blocks = blk(x_processed_blocks, attn_mask)

        if self.downsample is not None:
            x_next_layer_input = self.downsample(x_processed_blocks, H, W)
            H_next = (H + 1) // 2
            W_next = (W + 1) // 2
            return x_processed_blocks, H, W, x_next_layer_input, H_next, W_next
        else:
            return x_processed_blocks, H, W, x_processed_blocks, H, W


class DecoderBlock(nn.Module):
    """解码器块"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        upsampled_x_channels = in_channels // 2
        self.upsample = nn.ConvTranspose2d(in_channels, upsampled_x_channels, kernel_size=2, stride=2)

        conv_block_input_ch = upsampled_x_channels
        if skip_channels is not None and skip_channels > 0:
            # 将跳跃连接特征投影到与上采样特征相同的通道数
            self.skip_projection = nn.Conv2d(skip_channels, upsampled_x_channels, kernel_size=1, bias=False)
            conv_block_input_ch += upsampled_x_channels  # 拼接后的总通道数
        else:
            self.skip_projection = None

        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_block_input_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_from_prev_decoder, skip_from_encoder=None):
        x_upsampled = self.upsample(x_from_prev_decoder)

        if skip_from_encoder is not None and self.skip_projection is not None:
            skip_projected = self.skip_projection(skip_from_encoder)
            # 确保空间尺寸匹配
            if x_upsampled.shape[2:] != skip_projected.shape[2:]:
                skip_projected = F.interpolate(
                    skip_projected, size=x_upsampled.shape[2:],
                    mode='bilinear', align_corners=False)
            x_to_conv_block = torch.cat([x_upsampled, skip_projected], dim=1)
        else:
            x_to_conv_block = x_upsampled

        output = self.conv_block(x_to_conv_block)
        return output


# ==================== Swin U-Net 模型 ====================

class SwinUNet(nn.Module):
    """Swin U-Net模型"""

    def __init__(self, config_dict):
        super().__init__()
        self.patch_size_val = config_dict['patch_size']
        self.in_chans = config_dict['in_chans']
        self.embed_dim_val = config_dict['embed_dim']
        self.depths = config_dict['depths']
        self.num_heads = config_dict['num_heads']
        self.window_size = config_dict['window_size']
        self.mlp_ratio = config_dict['mlp_ratio']
        self.drop_rate = config_dict['drop_rate']
        self.attn_drop_rate = config_dict['attn_drop_rate']
        self.drop_path_rate = config_dict['drop_path_rate']
        self.num_layers = len(self.depths)
        self.patch_norm = True

        # Patch embedding
        self.patch_embed = nn.Conv2d(self.in_chans, self.embed_dim_val,
                                     kernel_size=self.patch_size_val, stride=self.patch_size_val)
        if self.patch_norm:
            self.norm_embed = nn.LayerNorm(self.embed_dim_val)
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # 构建编码器层
        self.layers = nn.ModuleList()
        current_dim_encoder = self.embed_dim_val
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=current_dim_encoder,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                current_dim_encoder = int(current_dim_encoder * 2)

        # 构建解码器
        self.decoder_channels_conf = config_dict['decoder_channels']
        self.decoders = nn.ModuleList()
        bottleneck_dim = current_dim_encoder  # 最后一个编码器层的维度

        decoder_input_dim = bottleneck_dim
        for i in range(len(self.decoder_channels_conf)):
            skip_encoder_stage_idx = self.num_layers - 2 - i
            skip_channels_dim = None
            if skip_encoder_stage_idx >= 0:
                skip_channels_dim = int(self.embed_dim_val * (2 ** skip_encoder_stage_idx))

            decoder = DecoderBlock(
                in_channels=decoder_input_dim,
                skip_channels=skip_channels_dim,
                out_channels=self.decoder_channels_conf[i])
            self.decoders.append(decoder)
            decoder_input_dim = self.decoder_channels_conf[i]

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.decoder_channels_conf[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x_in_spatial):
        """特征提取"""
        x_after_embed = self.patch_embed(x_in_spatial)  # (B, embed_dim, H_p, W_p)
        H_curr, W_curr = x_after_embed.size(2), x_after_embed.size(3)

        x_seq = x_after_embed.flatten(2).transpose(1, 2)  # (B, L_p, embed_dim)
        if self.patch_norm:
            x_seq = self.norm_embed(x_seq)
        x_seq = self.pos_drop(x_seq)

        skip_connections_spatial_list = []

        for layer_module in self.layers:
            x_block_out_seq, H_block_out, W_block_out, \
            x_next_in_seq, H_next_in, W_next_in = layer_module(x_seq, H_curr, W_curr)

            # 保存跳跃连接
            skip_C = x_block_out_seq.size(-1)
            skip_spatial = x_block_out_seq.transpose(1, 2).contiguous().view(-1, skip_C, H_curr, W_curr)
            skip_connections_spatial_list.append(skip_spatial)

            x_seq = x_next_in_seq
            H_curr = H_next_in
            W_curr = W_next_in

        return x_seq, skip_connections_spatial_list, H_curr, W_curr

    def forward(self, x_input_to_model):
        """前向传播"""
        bottleneck_seq, skips_spatial, H_bottle, W_bottle = self.forward_features(x_input_to_model)

        B, L_bottle, C_bottle = bottleneck_seq.shape
        if H_bottle == 0 or W_bottle == 0:
            logger.error(f"Zero dimension in bottleneck: H={H_bottle}, W={W_bottle}. Input: {x_input_to_model.shape}")
            # 返回零张量作为fallback
            dummy_output_shape = (B, 1, x_input_to_model.shape[2], x_input_to_model.shape[3])
            return torch.zeros(dummy_output_shape, device=bottleneck_seq.device, dtype=bottleneck_seq.dtype)

        assert L_bottle > 0 or (H_bottle == 0 and W_bottle == 0), "Bottleneck L is 0 but H/W are not."

        x_dec_in_spatial = bottleneck_seq.transpose(1, 2).contiguous().view(B, C_bottle, H_bottle, W_bottle)

        for i, decoder_module in enumerate(self.decoders):
            skip_idx_for_decoder = self.num_layers - 2 - i
            skip_to_use = None
            if skip_idx_for_decoder >= 0:
                if skip_idx_for_decoder < len(skips_spatial):
                    skip_to_use = skips_spatial[skip_idx_for_decoder]
                else:
                    logger.warning(f"Decoder {i}: skip_idx_for_decoder {skip_idx_for_decoder} out of bounds for skips_spatial (len {len(skips_spatial)})")

            x_dec_in_spatial = decoder_module(x_dec_in_spatial, skip_to_use)

        output_final = self.final_conv(x_dec_in_spatial)
        return output_final


# ==================== 配置类 ====================

class SwinUNetConfig(EnhancedTrainingConfig):
    """完整的Swin U-Net配置类 - 支持损失函数选择"""

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


# ==================== 训练器类 ====================

class SwinUNetTrainer:
    """完整的Swin U-Net训练器"""

    def __init__(self, model: nn.Module, config: SwinUNetConfig):
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
        logger.info(f"开始完整的Swin U-Net训练")
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

        logger.info("完整的Swin U-Net训练完成!")
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


# ==================== 工具函数 ====================

def list_available_configs():
    """列出所有可用的Swin U-Net损失配置"""
    print("=== 可用的Swin U-Net损失配置 ===")
    for name, config in SWIN_LOSS_CONFIGS.items():
        use_tid = "✓" if config['use_tid_weights'] else "✗"
        grad_weight = config['gradient_loss_weight']
        spectral_weight = config['spectral_loss_weight']
        print(f"\n{name}:")
        print(f"  名称: {config['name']}")
        print(f"  描述: {config['description']}")
        print(f"  TID权重: {use_tid}")
        print(f"  梯度损失权重: {grad_weight}")
        print(f"  频谱损失权重: {spectral_weight}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='完整的Swin U-Net海底地形精化训练')
    parser.add_argument('--loss_config', type=str, default='tid_gradient',
                        choices=list(SWIN_LOSS_CONFIGS.keys()),
                        help='选择损失函数配置')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率')
    parser.add_argument('--list_configs', action='store_true',
                        help='列出所有可用配置')

    args = parser.parse_args()

    if args.list_configs:
        list_available_configs()
        return

    # 创建配置
    config = SwinUNetConfig(args.loss_config)

    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate

    logger.info("=== 完整的Swin U-Net海底地形精化训练 ===")
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
        trainer = SwinUNetTrainer(model, config)
        trainer.train(train_loader, val_loader)

        # 保存实验摘要
        experiment_summary = {
            'experiment_name': f'swin_unet_complete_{config.loss_config_name}',
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

        summary_path = os.path.join(config.OUTPUT_DIR, f'swin_complete_summary_{config.loss_config_name}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, indent=2, ensure_ascii=False)

        # 保存训练历史
        history_path = os.path.join(config.OUTPUT_DIR, f'swin_complete_history_{config.loss_config_name}.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(trainer.history, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 完整的Swin U-Net训练完成！")
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
