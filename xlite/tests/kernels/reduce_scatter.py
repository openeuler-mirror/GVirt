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
#   TP (default): torchrun --nproc_per_node=2 --master_port=29500 reduce_scatter.py
#   4 cards TP:   torchrun --nproc_per_node=4 --master_port=29500 reduce_scatter.py
#   DP:           COMM_TYPE=1 torchrun --nproc_per_node=2 --master_port=29500 reduce_scatter.py
#   4 cards DP:   COMM_TYPE=1 torchrun --nproc_per_node=4 --master_port=29500 reduce_scatter.py
# ===============================================================================
import os
import torch
import torch.distributed as dist
from xlite._C import Runtime, reduce_scatter


world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
comm_type = int(os.getenv("COMM_TYPE", "0"))  # 0=TP, 1=DP

dist.init_process_group("hccl")

# Configure communicator sizes and reference group per domain.
if comm_type == 1:  # DP: all ranks in one DP group
    tp_size = 1
    dp_size = world_size
    ref_group = dist.new_group(list(range(world_size)))
else:  # TP (default): all ranks in one TP group
    tp_size = world_size
    dp_size = 1
    ref_group = None  # default group is the TP group

rt = Runtime(local_rank, 500, rank, tp_size, dp_size)
torch.npu.set_device(local_rank)

domain = "DP" if comm_type == 1 else "TP"
passed = True
for dim1, dim2 in zip([1, 1, 2, 4, 8, 32, 512], [1, 37, 4, 16, 16, 32, 7168]):
    total_dim1 = dim1 * world_size
    with torch.device("npu"):
        x_list = [torch.randn(dim1, dim2, dtype=torch.float) for _ in range(world_size)]
        y = torch.cat(x_list, dim=0)
        standard = torch.empty(dim1, dim2, dtype=torch.float)
        z = torch.empty_like(standard)

        dist.reduce_scatter(standard, x_list, op=dist.ReduceOp.SUM, group=ref_group)

        torch.npu.synchronize()
        reduce_scatter(rt, z, y, comm_type)
        torch.npu.synchronize()

        try:
            torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(
                f"{'=' * 50}\n"
                f"ReduceScatter ({domain}) result mismatch for dim ({total_dim1}, {dim2}) "
                f"(per-rank {dim1}, {dim2}) at rank {rank}. {e}\n"
                f"Expected: {standard}\n"
                f"Got: {z}\n",
                end="",
                flush=True,
            )
            passed = False

print(
    f"***** ReduceScatter ({domain}) test {'passed' if passed else 'failed'} on rank {rank}! *****\n",
    end="",
    flush=True,
)
