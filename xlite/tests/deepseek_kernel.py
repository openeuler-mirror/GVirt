#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from typing import Tuple

import torch


def weight_dequant(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (1, N).

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    y = (x.float() * s).to(torch.bfloat16)
    return y

def weight_dequant_preblock(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.
 
    Args:
        weight (torch.Tensor): The quantized weight tensor of shape (M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.
 
    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.
 
    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """
    # Get the original dimensions of weight
    M, N = weight.shape
 
    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    assert scale_m == (M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."
 
    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)
 
    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
 
    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:M, :N]
 
    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded
 
    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())
 
    return dequantized_weight