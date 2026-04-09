#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import
import logging
import os
import torch
import torch.nn.functional as F
from xlite._C import Runtime, layernorm

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

def stand_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.layer_norm(x, (x.size(-1),), weight, bias, eps)

BATCH_SIZE = 64
DIM = 128
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype)
        x_standard = x.clone()
        weight = torch.randn(DIM, dtype=test_dtype)
        bias = torch.randn(DIM, dtype=test_dtype)
        y = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)

    # standard
    standard = stand_layernorm(x_standard, weight, bias, NORMEPS)

    # xlite
    torch.npu.synchronize()
    layernorm(rt, x, weight, bias, y, NORMEPS, DIM)
    torch.npu.synchronize()

    logging.info(f'layernorm({test_dtype}) executed!')

    try:
        torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {standard}')
        logging.error(f'xlite: {y}')