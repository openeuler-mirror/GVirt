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
from xlite._C import runtime, reduce_scatter


world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("hccl")

rt = runtime(local_rank, 500, rank, world_size)
torch.npu.set_device(local_rank)

with torch.device("npu"):
    x = [torch.randn(8, 7168, dtype=torch.float16) for _ in range(world_size)]
    y = torch.cat(x, dim=0)
    standard = torch.empty(8, 7168, dtype=torch.float16)
    z = torch.empty_like(standard)

dist.reduce_scatter(standard, x, op=dist.ReduceOp.SUM)

torch.npu.synchronize()
reduce_scatter(rt, z, y)
torch.npu.synchronize()
if rank == 0:
    print('reduce scatter executed!')

try:
    torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
except AssertionError as e:
    print(f'{e}')
    print(f'torch_npu: {standard}')
    print(f'xlite: {z}')