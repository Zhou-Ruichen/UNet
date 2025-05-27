#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swin U-Net 海底地形精化训练 - Transformer+CNN混合架构

基于Swin Transformer的U-Net变体，专门针对海底地形高频细节恢复优化：
1. 分层的Swin Transformer编码器 - 捕获多尺度特征和全局上下文
2. CNN解码器 - 精确的空间定位和细节恢复
3. 跳跃连接增强 - 结合浅层和深层特征
4. 兼容现有损失函数组合和TID权重

主要特性:
- 窗口注意力机制增强局部-全局特征学习
- 移动窗口提升跨窗口连接
- 可配置的patch大小、窗口大小、注意力头数等
- 完全兼容现有训练框架

作者: 周瑞宸
日期: 2025-05-25 (修改于 2025-05-26)
版本: v1.1 (修复维度传递和窗口处理错误)
"""

import os
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from typing import Optional, Tuple, List
import logging
from datetime import datetime
import json

# 导入现有模块
from enhanced_training_multi_loss import (
    EnhancedTrainingConfig, EnhancedDataPreparation,
    EnhancedBathymetryDataset, MultiLossFunction,
    device, setup_chinese_font, logger
)

setup_chinese_font()


class SwinUNetConfig(EnhancedTrainingConfig):
    """Swin U-Net 特定配置"""

    def __init__(self):
        super().__init__()

        # 覆盖输出目录
        self.OUTPUT_DIR = "/mnt/data5/06-Projects/05-unet/output/3-swin_unet_outputs_lt135km"

        # Swin Transformer 特定参数
        self.SWIN_CONFIG = {
            # 基础参数
            'patch_size': 4,
            'in_chans': 4,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8, # 确保输入特征图在各阶段处理时 H, W >= window_size
            'mlp_ratio': 4.0,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'use_checkpoint': False,

            # 解码器参数
            'decoder_channels': [512, 256, 128, 64],
            'skip_connection_type': 'concat', # 当前DecoderBlock为concat
            'final_upsample': 'expand_first', # 当前未使用
        }

        self.BATCH_SIZE = 8
        self.LEARNING_RATE = 1e-4
        self.NUM_EPOCHS = 200
        self.WEIGHT_DECAY = 0.05
        self.EARLY_STOPPING_PATIENCE = 20 # 增加了早停耐心

        self.LOSS_CONFIG = {
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
        }
    
    def to_dict(self): # Helper to save config
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    # 此函数期望 H 和 W 至少等于 window_size, 并且最好是 window_size 的倍数
    # SwinTransformerBlock 中的填充逻辑应确保 H, W >= window_size
    if not (H >= window_size and W >= window_size):
        # This should ideally not happen if SwinTransformerBlock pads correctly
        logger.warning(f"window_partition called with H={H} or W={W} < window_size={window_size}. Errors may occur.")

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    if num_windows_h == 0 or num_windows_w == 0: # Should not happen if H, W >= window_size
        logger.error(f"window_reverse: num_windows_h or num_windows_w is 0. H={H}, W={W}, ws={window_size}")
        # This would lead to division by zero for B if not careful
        # Fallback, though this indicates a serious upstream issue
        if windows.shape[0] == 0 : # If num_windows is 0, then windows.shape[0] could be 0
             return torch.zeros((0,H,W,windows.shape[-1]), device=windows.device, dtype=windows.dtype) # Placeholder for B=0 case
        # This case is problematic, means H or W likely 0
        # B = 0 # Avoid division by zero
        # For now, assume H,W are valid from upstream
        raise ValueError(f"window_reverse: H ({H}) or W ({W}) is too small for window_size ({window_size}) leading to zero windows.")


    B = int(windows.shape[0] / (num_windows_h * num_windows_w))
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
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

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size_tuple[0] - 1) * (2 * self.window_size_tuple[1] - 1), num_heads))

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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
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
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size # int
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.H = None # Set by BasicLayer
        self.W = None # Set by BasicLayer

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim), nn.Dropout(drop)
        )

    def forward(self, x, attn_mask_input): # x is (B, L, C)
        H_in, W_in = self.H, self.W
        B, L, C = x.shape
        assert L == H_in * W_in, f"Input L={L} != H_in*W_in={H_in*W_in}"

        shortcut = x
        x_norm1 = self.norm1(x)
        x_spatial = x_norm1.view(B, H_in, W_in, C)

        # Pad if H_in or W_in is less than window_size for window operation
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
            x_permuted = x_spatial.permute(0, 3, 1, 2) # B, C, H_in, W_in
            x_padded_permuted = F.pad(x_permuted, (0, pad_r_op, 0, pad_b_op)) # Pad R, B
            x_to_shift = x_padded_permuted.permute(0, 2, 3, 1) # B, H_eff, W_eff, C
        
        # Cyclic shift on (potentially padded) H_eff, W_eff
        if self.shift_size > 0:
            shifted_x = torch.roll(x_to_shift, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_to_shift
        
        # Partition windows using H_eff, W_eff
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA. attn_mask_input is computed in BasicLayer for H_in, W_in dimensions
        # (or Hp, Wp if H_in/W_in were smaller than window_size, aligning with H_eff/W_eff)
        attn_windows = self.attn(x_windows, mask=attn_mask_input)

        # Merge windows
        attn_windows_merged = attn_windows.view(-1, self.window_size, self.window_size, C)
        # Reverse windowing using H_eff, W_eff
        shifted_x_reverted = window_reverse(attn_windows_merged, self.window_size, H_eff, W_eff)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_rolled_back = torch.roll(shifted_x_reverted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_rolled_back = shifted_x_reverted

        # Crop to original H_in, W_in if padding was applied
        if needs_padding:
            x_unpadded_spatial = x_rolled_back[:, :H_in, :W_in, :].contiguous()
        else:
            x_unpadded_spatial = x_rolled_back
            
        x_final_seq = x_unpadded_spatial.view(B, H_in * W_in, C)
        
        x = shortcut + self.drop_path(x_final_seq) # Residual to original x (before norm1)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # Apply norm2 to the sum

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H: int, W: int): # x: (B, L, C), C = self.dim
        B, L, C = x.shape
        assert L == H * W, "PatchMerging: Input L mismatch"
        assert C == self.dim, "PatchMerging: Input C mismatch"

        x_spatial = x.view(B, H, W, C)
        
        H_padded, W_padded = H, W
        if H % 2 == 1: H_padded +=1
        if W % 2 == 1: W_padded +=1
        
        x_to_sample = x_spatial
        if H_padded != H or W_padded != W:
            x_perm = x_spatial.permute(0,3,1,2) # B, C, H, W
            x_padded_perm = F.pad(x_perm, (0, W_padded - W, 0, H_padded - H)) # pad right, bottom
            x_to_sample = x_padded_perm.permute(0,2,3,1) # B, H_padded, W_padded, C

        x0 = x_to_sample[:, 0::2, 0::2, :]
        x1 = x_to_sample[:, 1::2, 0::2, :]
        x2 = x_to_sample[:, 0::2, 1::2, :]
        x3 = x_to_sample[:, 1::2, 1::2, :]
        
        merged_x = torch.cat([x0, x1, x2, x3], -1) # B, H_padded/2, W_padded/2, 4*C
        merged_x_seq = merged_x.view(B, -1, 4 * C)
        merged_x_seq_norm = self.norm(merged_x_seq)
        output_seq = self.reduction(merged_x_seq_norm)
        return output_seq


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H, W): # x: (B, L, C), H, W are spatial for x
        attn_mask = None
        if self.shift_size > 0: # Create mask only for SW-MSA
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


class SwinUNet(nn.Module):
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

        self.patch_embed = nn.Conv2d(self.in_chans, self.embed_dim_val,
                                   kernel_size=self.patch_size_val, stride=self.patch_size_val)
        if self.patch_norm:
            self.norm_embed = nn.LayerNorm(self.embed_dim_val)
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        
        self.layers = nn.ModuleList()
        current_dim_encoder = self.embed_dim_val
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=current_dim_encoder, depth=self.depths[i_layer], num_heads=self.num_heads[i_layer],
                window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=True,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                current_dim_encoder = int(current_dim_encoder * 2)
        
        self.decoder_channels_conf = config_dict['decoder_channels']
        self.decoders = nn.ModuleList()
        bottleneck_dim = current_dim_encoder # Dim after last encoder layer
        
        decoder_input_dim = bottleneck_dim
        for i in range(len(self.decoder_channels_conf)):
            skip_encoder_stage_idx = self.num_layers - 2 - i
            skip_channels_dim = None
            if skip_encoder_stage_idx >= 0:
                skip_channels_dim = int(self.embed_dim_val * (2 ** skip_encoder_stage_idx))

            decoder = DecoderBlock(
                in_channels=decoder_input_dim, skip_channels=skip_channels_dim,
                out_channels=self.decoder_channels_conf[i])
            self.decoders.append(decoder)
            decoder_input_dim = self.decoder_channels_conf[i]
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.decoder_channels_conf[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             if m.weight is not None: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None: nn.init.constant_(m.bias, 0)


    def forward_features(self, x_in_spatial): # (B, C_in, H_img, W_img)
        x_after_embed = self.patch_embed(x_in_spatial) # (B, embed_dim, H_p, W_p)
        H_curr, W_curr = x_after_embed.size(2), x_after_embed.size(3)
        
        x_seq = x_after_embed.flatten(2).transpose(1, 2) # (B, L_p, embed_dim)
        if self.patch_norm:
            x_seq = self.norm_embed(x_seq)
        x_seq = self.pos_drop(x_seq)

        skip_connections_spatial_list = []
        
        for layer_module in self.layers:
            x_block_out_seq, H_block_out, W_block_out, \
            x_next_in_seq, H_next_in, W_next_in = layer_module(x_seq, H_curr, W_curr)
            
            # x_block_out_seq is at H_curr, W_curr resolution (before layer_module's own downsampler)
            skip_C = x_block_out_seq.size(-1)
            skip_spatial = x_block_out_seq.transpose(1, 2).contiguous().view(-1, skip_C, H_curr, W_curr)
            skip_connections_spatial_list.append(skip_spatial)

            x_seq = x_next_in_seq
            H_curr = H_next_in
            W_curr = W_next_in
            
        # x_seq is now bottleneck_features_seq; H_curr, W_curr are bottleneck H, W
        return x_seq, skip_connections_spatial_list, H_curr, W_curr

    def forward(self, x_input_to_model):
        bottleneck_seq, skips_spatial, H_bottle, W_bottle = self.forward_features(x_input_to_model)
        
        B, L_bottle, C_bottle = bottleneck_seq.shape
        if H_bottle == 0 or W_bottle == 0: # Should be caught by assertions or padding earlier
             logger.error(f"Zero dimension in bottleneck: H={H_bottle}, W={W_bottle}. Input: {x_input_to_model.shape}")
             # Attempt to return a zero tensor of expected output shape to prevent hard crash during training loop
             # This is a fallback, the root cause (likely too small input) must be addressed.
             # Assuming output is single channel and same H,W as input (typical for U-Net like tasks)
             dummy_output_shape = (B, 1, x_input_to_model.shape[2], x_input_to_model.shape[3])
             return torch.zeros(dummy_output_shape, device=bottleneck_seq.device, dtype=bottleneck_seq.dtype)

        # If L_bottle is unexpectedly 0 while H_bottle/W_bottle are not, it's an issue.
        # But if H_bottle/W_bottle are >0, L_bottle = H_bottle*W_bottle should also be >0.
        assert L_bottle > 0 or (H_bottle==0 and W_bottle==0), "Bottleneck L is 0 but H/W are not."


        x_dec_in_spatial = bottleneck_seq.transpose(1, 2).contiguous().view(B, C_bottle, H_bottle, W_bottle)
        
        for i, decoder_module in enumerate(self.decoders):
            # Skips: [out_L0, out_L1, out_L2, out_L3_block_out] for num_layers=4
            # Decoder 0 (i=0) uses skip from L2 (idx 2). Formula: num_layers - 2 - i
            # The last element of skips_spatial (skips_spatial[num_layers-1]) is the block output of the bottleneck stage.
            # It's not used as a "skip" in the traditional sense for the first decoder stage,
            # as x_dec_in_spatial *is* derived from it.
            # The skips should come from stages *before* the bottleneck.
            
            skip_idx_for_decoder = self.num_layers - 2 - i 
            skip_to_use = None
            if skip_idx_for_decoder >= 0:
                 if skip_idx_for_decoder < len(skips_spatial): # Ensure index is valid
                    skip_to_use = skips_spatial[skip_idx_for_decoder]
                 else:
                    logger.warning(f"Decoder {i}: skip_idx_for_decoder {skip_idx_for_decoder} out of bounds for skips_spatial (len {len(skips_spatial)})")

            x_dec_in_spatial = decoder_module(x_dec_in_spatial, skip_to_use)
        
        output_final = self.final_conv(x_dec_in_spatial)
        return output_final


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        upsampled_x_channels = in_channels // 2
        self.upsample = nn.ConvTranspose2d(in_channels, upsampled_x_channels, kernel_size=2, stride=2)
        
        conv_block_input_ch = upsampled_x_channels
        if skip_channels is not None and skip_channels > 0 : # Check if skip_channels is valid
            # Project skip features to match channel depth of upsampled_x_channels for concatenation
            self.skip_projection = nn.Conv2d(skip_channels, upsampled_x_channels, kernel_size=1, bias=False)
            conv_block_input_ch += upsampled_x_channels # Total after concat
        else:
            self.skip_projection = None # No skip or skip_channels is 0/None
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_block_input_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x_from_prev_decoder, skip_from_encoder=None):
        x_upsampled = self.upsample(x_from_prev_decoder)
        
        if skip_from_encoder is not None and self.skip_projection is not None:
            skip_projected = self.skip_projection(skip_from_encoder)
            if x_upsampled.shape[2:] != skip_projected.shape[2:]:
                skip_projected = F.interpolate(
                    skip_projected, size=x_upsampled.shape[2:], 
                    mode='bilinear', align_corners=False)
            x_to_conv_block = torch.cat([x_upsampled, skip_projected], dim=1)
        else:
            x_to_conv_block = x_upsampled
        
        output = self.conv_block(x_to_conv_block)
        return output


class SwinUNetTrainer:
    def __init__(self, model: nn.Module, config: SwinUNetConfig):
        self.model = model.to(device)
        self.config = config
        self.criterion = MultiLossFunction(config.LOSS_CONFIG)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=config.LEARNING_RATE * 0.01)
        self.history = {
            'train_loss_total': [], 'train_loss_pixel': [], 'train_loss_gradient': [], 'train_loss_spectral': [],
            'val_loss_total': [], 'val_loss_pixel': [], 'val_loss_gradient': [], 'val_loss_spectral': [],
            'val_correlation': [], 'learning_rates': []
        }
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.current_epoch_display = "1" # For tqdm

    def train_epoch(self, train_loader: DataLoader) -> dict:
        self.model.train()
        epoch_losses = {'total': 0.0, 'pixel': 0.0, 'gradient': 0.0, 'spectral': 0.0}
        from tqdm import tqdm
        # Make sure self.current_epoch_display is set before this, e.g., in train() method
        progress_bar = tqdm(train_loader, desc=f'训练 Epoch {getattr(self, "current_epoch_display", "")}', leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            features = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            tid_weights = batch_data[2].to(device) if len(batch_data) == 3 and self.config.LOSS_CONFIG['use_tid_weights'] else None

            outputs = self.model(features)
            if outputs.shape[-2:] != labels.shape[-2:]: # Ensure spatial dims match for loss
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            loss, loss_components = self.criterion(outputs, labels, tid_weights)

            self.optimizer.zero_grad()
            loss.backward() # loss should be a tensor
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for key_loss_component in epoch_losses: # Use a different variable name for clarity
                if key_loss_component in loss_components:
                    value = loss_components[key_loss_component]
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key_loss_component] += value.item()
                    elif isinstance(value, float): # If it's already a float
                        epoch_losses[key_loss_component] += value
                    # else: You might want to log a warning if it's neither
            
            # For progress bar, ensure values are numbers
            postfix_dict = {}
            for k, v_comp in loss_components.items():
                if k in ['total', 'pixel', 'gradient', 'spectral']:
                     if isinstance(v_comp, torch.Tensor):
                        postfix_dict[k] = f'{v_comp.item():.4f}'
                     elif isinstance(v_comp, float):
                        postfix_dict[k] = f'{v_comp:.4f}'
            progress_bar.set_postfix(postfix_dict)


        n_batches = len(train_loader)
        return {key_val: val / n_batches if n_batches > 0 else 0.0 for key_val, val in epoch_losses.items() }


    def validate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        epoch_losses = {'total': 0.0, 'pixel': 0.0, 'gradient': 0.0, 'spectral': 0.0}
        all_predictions_list, all_targets_list = [], []
        with torch.no_grad():
            from tqdm import tqdm
            for batch_data in tqdm(val_loader, desc='验证中', leave=False):
                features = batch_data[0].to(device)
                labels = batch_data[1].to(device)
                tid_weights = batch_data[2].to(device) if len(batch_data) == 3 and self.config.LOSS_CONFIG['use_tid_weights'] else None
                
                outputs = self.model(features)
                if outputs.shape[-2:] != labels.shape[-2:]:
                     outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)

                loss, loss_components = self.criterion(outputs, labels, tid_weights) # loss should be a tensor
                
                for key_loss_component in epoch_losses: # Use a different variable name
                    if key_loss_component in loss_components:
                        value = loss_components[key_loss_component]
                        if isinstance(value, torch.Tensor):
                            epoch_losses[key_loss_component] += value.item()
                        elif isinstance(value, float): # If it's already a float
                            epoch_losses[key_loss_component] += value
                        # else: Log warning
                            
                all_predictions_list.append(outputs.cpu().numpy())
                all_targets_list.append(labels.cpu().numpy())
        
        correlation = 0.0
        if all_predictions_list and all_targets_list:
            predictions_flat = np.concatenate([p.flatten() for p in all_predictions_list])
            targets_flat = np.concatenate([t.flatten() for t in all_targets_list])
            if len(predictions_flat) > 1 and len(targets_flat) > 1 and len(predictions_flat) == len(targets_flat):
                correlation = np.corrcoef(predictions_flat, targets_flat)[0, 1]
        
        n_batches = len(val_loader)
        results = {key_val: val / n_batches if n_batches > 0 else 0.0 for key_val, val in epoch_losses.items()}
        results['correlation'] = correlation
        return results

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info(f"开始Swin U-Net训练 - 损失: {self.config.LOSS_CONFIG['loss_combination']}")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch_display = f"{epoch + 1}/{self.config.NUM_EPOCHS}"
            logger.info(f"\nEpoch {self.current_epoch_display}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history (make sure keys match)
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
            
            train_log = ", ".join([f"{k}: {v:.6f}" for k,v in train_metrics.items()])
            val_log = ", ".join([f"{k}: {v:.6f}" for k,v in val_metrics.items()])
            logger.info(f"训练 - {train_log}")
            logger.info(f"验证 - {val_log}")
            logger.info(f"学习率: {current_lr:.6e}")

            if val_metrics.get('total', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, self.best_val_loss) # Pass current best loss
                logger.info(f"保存最佳模型 (验证损失: {self.best_val_loss:.6f})")
            else:
                self.early_stopping_counter += 1
            
            if self.config.EARLY_STOPPING_PATIENCE > 0 and \
               self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"早停触发，在epoch {epoch + 1}")
                break
        logger.info("Swin U-Net训练完成!")

    def save_checkpoint(self, epoch: int, val_loss: float):
        # Ensure config can be serialized (e.g. by having a to_dict method)
        config_to_save = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config).copy()
        # Remove non-serializable items if any (though vars() usually okay for simple configs)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config_dict': config_to_save, # Save config as dict
            'history': self.history
        }
        path = os.path.join(self.config.OUTPUT_DIR, 'best_swin_unet_model_lt135km.pth')
        torch.save(checkpoint, path)


def main():
    config = SwinUNetConfig()
    logger.info("=== Swin U-Net 海底地形精化训练 ===")
    logger.info(f"运行设备: {device}")
    try:
        logger.info(f"Swin配置: {json.dumps(config.SWIN_CONFIG, indent=2, ensure_ascii=False)}")
        logger.info(f"损失配置: {json.dumps(config.LOSS_CONFIG, indent=2, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"JSON序列化配置时出错: {e}")
        logger.info(f"Swin配置 (raw): {config.SWIN_CONFIG}")
        logger.info(f"损失配置 (raw): {config.LOSS_CONFIG}")

    logger.info(f"输出目录: {config.OUTPUT_DIR}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    data_prep = EnhancedDataPreparation(config)
    try:
        features_ds, labels_da, tid_weights_da = data_prep.load_data()
        if features_ds is None:
            logger.error("数据加载失败")
            return

        # extract_patches 返回 List[Dict]
        extracted_patches_list = data_prep.extract_patches(features_ds, labels_da, tid_weights_da)
        
        # 检查提取到的数据块列表是否有效
        if not extracted_patches_list:  # 检查列表是否为空或None
            logger.error("没有提取到有效的数据块！(列表为空)")
            return
        
        logger.info(f"提取到 {len(extracted_patches_list)} 个数据块.")
        # 记录第一个数据块的特征形状以供参考
        if len(extracted_patches_list) > 0:
            first_patch_features = extracted_patches_list[0].get('features') # 安全地获取特征
            if first_patch_features is not None and hasattr(first_patch_features, 'shape'):
                logger.info(f"单个数据块特征图形状: {first_patch_features.shape}")
            else:
                logger.warning("提取到的第一个数据块中未找到'features'或其不具备shape属性。")

        # split_patches 接收 List[Dict] 并返回 Tuple[List[Dict], List[Dict], List[Dict]]
        train_patches_list, val_patches_list, _ = data_prep.split_patches(extracted_patches_list) # test_patches_list 未使用，用 _ 接收

        # 检查划分后的训练集和验证集列表是否为空
        if not train_patches_list:
            logger.error("训练数据块列表为空!")
            return
        if not val_patches_list:
            logger.error("验证数据块列表为空!")
            return

        # EnhancedBathymetryDataset 接收 List[Dict]
        train_dataset = EnhancedBathymetryDataset(train_patches_list, config, is_training=True)
        val_dataset = EnhancedBathymetryDataset(val_patches_list, config, is_training=False)

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available() )

        model = SwinUNet(config.SWIN_CONFIG)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

        trainer = SwinUNetTrainer(model, config)
        trainer.train(train_loader, val_loader)

        experiment_summary = {
            'experiment_name': 'swin_unet_seafloor_topography_lt135km',
            'model_type': 'Swin U-Net',
            'swin_config': config.SWIN_CONFIG, 'loss_config': config.LOSS_CONFIG,
            'training_config': {
                'batch_size': config.BATCH_SIZE, 'learning_rate': config.LEARNING_RATE,
                'num_epochs_run': len(trainer.history['train_loss_total']),
                'target_num_epochs': config.NUM_EPOCHS,
                'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
                'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR'
            },
            'best_val_loss': trainer.best_val_loss if trainer.best_val_loss != float('inf') else None,
            'total_params': total_params, 'trainable_params': trainable_params,
            'final_epoch_completed': len(trainer.history['train_loss_total'])
        }
        summary_path = os.path.join(config.OUTPUT_DIR, 'swin_unet_experiment_summary_lt135km.json')
        with open(summary_path, 'w', encoding='utf-8') as f: json.dump(experiment_summary, f, indent=2, ensure_ascii=False)
        
        history_path = os.path.join(config.OUTPUT_DIR, 'swin_unet_training_history_lt135km.json')
        with open(history_path, 'w', encoding='utf-8') as f: json.dump(trainer.history, f, indent=2, ensure_ascii=False)

        logger.info(f"Swin U-Net训练完成！最佳验证损失: {trainer.best_val_loss if trainer.best_val_loss != float('inf') else 'N/A'}")
        logger.info(f"结果保存在: {config.OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"训练过程中出现严重错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()