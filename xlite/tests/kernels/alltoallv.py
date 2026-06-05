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
#   TP (default): torchrun --nproc_per_node=4 --master_port=29500 alltoallv.py
#   DP:           COMM_TYPE=1 torchrun --nproc_per_node=4 --master_port=29500 alltoallv.py
#   EP:           COMM_TYPE=2 torchrun --nproc_per_node=4 --master_port=29500 alltoallv.py
# ===============================================================================

import os
import torch
import torch.distributed as dist
from xlite._C import Runtime, alltoallv

world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
comm_type = int(os.getenv("COMM_TYPE", "0"))  # 0=TP, 1=DP, 2=EP

dist.init_process_group("hccl")

# Configure communicator sizes and reference group per domain.
if comm_type == 2:  # EP: all ranks in one EP group
    tp_size = 1
    dp_size = 1
    moe_tp_size = 1
    moe_ep_size = world_size
    ref_group = None  # default (world) group matches EP when moe_ep_size == world_size
elif comm_type == 1:  # DP: all ranks in one DP group
    tp_size = 1
    dp_size = world_size
    moe_tp_size = 1
    moe_ep_size = 1
    ref_group = dist.new_group(list(range(world_size)))
else:  # TP (default): all ranks in one TP group
    tp_size = world_size
    dp_size = 1
    moe_tp_size = 1
    moe_ep_size = 1
    ref_group = None  # default group is the TP group

rt = Runtime(local_rank, 500, rank, tp_size, dp_size, moe_tp_size, moe_ep_size)
torch.npu.set_device(local_rank)

send_counts = torch.tensor([rank + j + 1 for j in range(world_size)], dtype=torch.int64)
recv_counts = torch.tensor(
    [sum(j + i + 1 for j in range(world_size)) for i in range(world_size)], dtype=torch.int64
)

sdispls = torch.zeros_like(send_counts)
rdispls = torch.zeros_like(recv_counts)
for i in range(1, world_size):
    sdispls[i] = sdispls[i - 1] + send_counts[i - 1]
    rdispls[i] = rdispls[i - 1] + recv_counts[i - 1]

total_send = send_counts.sum().item()
total_recv = recv_counts.sum().item()

with torch.device("npu"):
    x = torch.randn(total_send, dtype=torch.float) * (rank + 1)
    standard = torch.zeros(total_recv, dtype=torch.float)
    xlite = torch.zeros(total_recv, dtype=torch.float)

dist.all_to_all_single(
    standard, x, recv_counts.tolist(), send_counts.tolist(), group=ref_group
)
torch.npu.synchronize()

alltoallv(rt, xlite, x, send_counts, recv_counts, sdispls, rdispls, comm_type)
torch.npu.synchronize()

domain = ["TP", "DP", "EP"][comm_type]
print(
    f"[rank {rank}] domain={domain} | "
    f"send_counts: {send_counts.tolist()} | recv_counts: {recv_counts.tolist()} | "
    f"total_send: {total_send} | total_recv: {total_recv}\n"
)

if rank == 1:
    print(f'alltoallv ({domain}) executed!')
    try:
        torch.testing.assert_close(standard, xlite, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        print(f'FAIL: {e}')
        print(f'torch_npu: {standard.shape}, {standard}')
        print(f'xlite: {xlite.shape}, {xlite}')
