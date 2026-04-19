#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
# Usage:
#   2 cards: torchrun --nproc_per_node=2 --master_port=29500 rmsnorm_full.py
#   4 cards: torchrun --nproc_per_node=4 --master_port=29500 rmsnorm_full.py
# ===============================================================================
from __future__ import absolute_import
import logging
import os
import torch
import torch_npu
import torch.distributed as dist
from xlite._C import Runtime, rmsnorm, all_reduce


world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("hccl")

rt = Runtime(local_rank, 500, rank, world_size)
torch.npu.set_device(local_rank)

logging.getLogger().setLevel(logging.INFO)


def stand_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2)
    variance = variance.mean(-1, keepdim=True)
    if world_size > 1:
        dist.all_reduce(variance, op=dist.ReduceOp.SUM)
    variance = variance / world_size
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(input_dtype)

BATCH_SIZE = 64
DIM = 384
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
        with torch.device("npu"):
            x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype)
            x_standard = x.clone()
            weight = torch.randn(DIM, dtype=test_dtype)
            y = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)
            variance_xlite = torch.zeros(BATCH_SIZE, 1, dtype=torch.float32)

        # standard
        standard = stand_rmsnorm(x_standard, weight, NORMEPS)

        # xlite
        torch.npu.synchronize()
        rmsnorm(rt, x, weight, variance_xlite, NORMEPS, norm_dim=DIM, cnt_per_token=1, use_norm=False)
        all_reduce(rt, variance_xlite, variance_xlite)
        rmsnorm(rt, x, weight, y, NORMEPS, norm_dim=DIM, cnt_per_token=1, use_norm=True, variance=variance_xlite)
        torch.npu.synchronize()
        if rank == 0:
            logging.info(f'rmsnorm({test_dtype}) executed!')
            try:
                torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
            except AssertionError as e:
                logging.error(f'{e}')
                logging.error(f'torch_npu: {standard}')
                logging.error(f'xlite: {y}')