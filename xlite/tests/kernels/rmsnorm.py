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
import torch.distributed as dist
from xlite._C import Runtime, rmsnorm

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

def stand_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(input_dtype)

BATCH_SIZE = 64
DIM = 4096
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype)
        x_standard = x.clone()
        weight = torch.randn(DIM, dtype=test_dtype)
        y = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)

    # standard
    standard = stand_rmsnorm(x_standard, weight, NORMEPS)

    # xlite
    torch.npu.synchronize()
    rmsnorm(rt, x, weight, y, NORMEPS)
    torch.npu.synchronize()

    logging.info(f'rmsnorm({test_dtype}) executed!')

    try:
        torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {standard}')
        logging.error(f'xlite: {y}')

BATCH_SIZE = 64
DIM = 128
CNT = 8
SIZE1 = DIM * (CNT + 2)
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        x = torch.randn(BATCH_SIZE, SIZE1, dtype=test_dtype)
        x_standard = x.clone()
        weight = torch.randn(DIM, dtype=test_dtype)
        y = x.clone()

    x_split1, x_split2, x_split3 = x_standard.split([CNT * DIM, DIM, DIM], dim = 1)
    x_split1 = x_split1.view(BATCH_SIZE, CNT, DIM)
    x_split2 = x_split2.view(BATCH_SIZE, 1, DIM)
    # standard
    x_split1 = stand_rmsnorm(x_split1, weight, NORMEPS)
    x_split2 = stand_rmsnorm(x_split2, weight, NORMEPS)
    x_split1 = x_split1.view(BATCH_SIZE, CNT * DIM)
    x_split2 = x_split2.view(BATCH_SIZE, DIM)
    standard = torch.cat([x_split1, x_split2, x_split3], dim=-1)

    # xlite
    torch.npu.synchronize()
    rmsnorm(rt, x, weight, y, NORMEPS, DIM, CNT)
    rmsnorm(rt, x, weight, y, NORMEPS, DIM, 1, DIM * CNT)
    torch.npu.synchronize()

    logging.info(f'rmsnorm with stride ({test_dtype}) executed!')

    try:
        torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {standard}')
        logging.error(f'xlite: {y}')