#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from typing import Optional


def weight_dequant(x: torch.Tensor, s: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale (and optional offset).

    Formula: y = (x - offset) * scale  (asymmetric); y = x * scale (symmetric, offset=None).

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N), int8.
        s (torch.Tensor): The per-row scale tensor of shape (M, 1) or (1, N).
        offset (Optional[torch.Tensor]): The per-row offset tensor of shape (M, 1). Defaults to None.

    Returns:
        torch.Tensor: The dequantized weight tensor (bf16), same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    y = x.float()
    if offset is not None:
        y = y - offset.float()
    y = (y * s).to(torch.bfloat16)
    return y
