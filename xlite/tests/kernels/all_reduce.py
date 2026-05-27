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
#   2 cards: torchrun --nproc_per_node=2 --master_port=29500 all_reduce.py
#   4 cards: torchrun --nproc_per_node=4 --master_port=29500 all_reduce.py
# ===============================================================================
import os
import torch
import torch.distributed as dist
from xlite._C import Runtime, all_reduce


world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("hccl")

rt = Runtime(local_rank, 500, rank, world_size)
torch.npu.set_device(local_rank)

passed = True
for dim1, dim2 in zip([1, 1, 1, 1, 32, 7168], [1, 4, 16, 37, 32, 7168]):
    with torch.device("npu"):
        x = torch.rand(dim1, dim2, dtype=torch.float)
        standard = x.clone()
        z = torch.empty_like(x)

        dist.all_reduce(standard, op=dist.ReduceOp.SUM)

        torch.npu.synchronize()
        all_reduce(rt, z, x)
        torch.npu.synchronize()

        try:
            torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(
                f"{'=' * 50}\n"
                f"AllReduce result mismatch for dim ({dim1}, {dim2}) at rank {rank}. {e}\n"
                f"Expected: {standard}\n"
                f"Got: {z}\n",
                end="",
                flush=True,
            )
            passed = False

print(f"***** AllReduce test {'passed' if passed else 'failed'} on rank {rank}! *****\n", end="", flush=True)
