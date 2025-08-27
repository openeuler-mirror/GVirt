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
        x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype)
        x_standard = x.clone()
        weight = torch.randn(DIM, dtype=test_dtype)
        y = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)

    # standard
    input_dtype = x_standard.dtype
    x_standard = x_standard.to(torch.float32)
    variance = x_standard.pow(2).mean(-1, keepdim=True)
    x_standard = x_standard * torch.rsqrt(variance + NORMEPS)
    standard = (weight.float() * x_standard).to(input_dtype)

    # xlite
    torch.npu.synchronize()
    rmsnorm(rt, x, weight, y, NORMEPS)
    torch.npu.synchronize()

    if rank == 0:
        logging.info(f'rmsnorm({test_dtype}) executed!')

    try:
        torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {standard}')
        logging.error(f'xlite: {y}')
