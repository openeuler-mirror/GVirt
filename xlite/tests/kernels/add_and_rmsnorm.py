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
from xlite._C import Runtime, add_and_rmsnorm

logging.getLogger().setLevel(logging.INFO)

world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("hccl")

rt = Runtime(local_rank, 500, rank, world_size)
torch.npu.set_device(local_rank)

BATCH_SIZE = 64
DIM = 4096
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype) / 10
        x_standard = x.clone()
        weight = torch.randn(DIM, dtype=test_dtype) / 10
        y = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype) / 10
        y_standard = y.clone()

    # standard
    x_standard = x_standard.to(torch.float32)
    y_standard = y_standard.to(torch.float32)
    x_standard = x_standard + y_standard
    x_add_standard = x_standard
    variance = x_standard.pow(2).mean(-1, keepdim=True)
    x_standard = x_standard * torch.rsqrt(variance + NORMEPS)
    standard = (weight.float() * x_standard).to(test_dtype)
    x_add_standard = x_add_standard.to(test_dtype)

    # xlite
    torch.npu.synchronize()
    add_and_rmsnorm(rt, x, y, weight, y, NORMEPS)
    torch.npu.synchronize()

    if rank == 0:
        logging.info(f'rmsnorm({test_dtype}) executed!')

    try:
        torch.testing.assert_close(x_add_standard, x, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {x_add_standard}')
        logging.error(f'xlite: {x}')

    try:
        torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {standard}')
        logging.error(f'xlite: {y}')
