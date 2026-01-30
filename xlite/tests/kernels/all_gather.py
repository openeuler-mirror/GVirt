#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os
import torch
import torch.distributed as dist
from xlite._C import Runtime, all_gather


world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("hccl")

rt = Runtime(local_rank, 500, rank, world_size)
torch.npu.set_device(local_rank)

with torch.device("npu"):
    x = torch.randn(8, 7168, dtype=torch.float)
    y = [torch.empty_like(x) for _ in range(world_size)]
    z = torch.empty(8 * world_size, 7168, dtype=torch.float)

dist.all_gather(y, x)
standard = torch.cat(y, dim=0)

torch.npu.synchronize()
all_gather(rt, z, x)
torch.npu.synchronize()
if rank == 1:
    print('all gather executed!')
    try:
        torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        print(f'{e}')
        print(f'torch_npu: {standard}')
        print(f'xlite: {z}')