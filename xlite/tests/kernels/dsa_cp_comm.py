#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Check the collective boundaries removed from DSA-CP prefill.

torchrun --nproc_per_node=2 tests/kernels/dsa_cp_comm.py

Uses real xlite collectives, full W^O projection and residual/Norm kernels.
This tests the boundary equivalence, not the complete model forward.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from xlite._C import Runtime, add_and_rmsnorm, all_gather, matmul, reduce_scatter, rmsnorm


def run_case(rt, rank, tp, tokens, dtype):
    hidden, head_width = 128, 256
    local_token_capacity = (tokens + tp - 1) // tp
    rank_offset = rank * local_token_capacity
    shard_begin = min(rank_offset, tokens)
    shard_end = min(rank_offset + local_token_capacity, tokens)
    shard_len = shard_end - shard_begin
    torch.manual_seed(42)
    residual = torch.randn(local_token_capacity * tp, hidden, dtype=dtype).npu()
    residual[tokens:] = 0
    norm_weight = torch.ones(hidden, dtype=dtype).npu()
    local_residual = residual[rank_offset : rank_offset + local_token_capacity].clone()
    local_norm = torch.empty_like(local_residual)
    full_norm = torch.zeros_like(residual)

    # First layer: local Norm must equal full Norm followed by token selection.
    torch.npu.synchronize()
    rmsnorm(rt, residual[:tokens], norm_weight, full_norm[:tokens], 1e-5)
    rmsnorm(rt, local_residual, norm_weight, local_norm, 1e-5)
    torch.testing.assert_close(local_norm, full_norm[rank_offset : rank_offset + local_token_capacity], rtol=0, atol=0)

    # Later layers: AllGather followed by taking the original shard is identity.
    gathered = torch.empty_like(residual)
    torch.npu.synchronize()
    all_gather(rt, gathered, local_norm)
    torch.testing.assert_close(local_norm, gathered[rank_offset : rank_offset + local_token_capacity], rtol=0, atol=0)

    attention = (torch.randn(local_token_capacity, head_width, dtype=dtype) * 0.1).npu()
    weight = (torch.randn(hidden, head_width, dtype=dtype) * 0.1).npu()
    old_global = torch.zeros_like(residual)
    old_local = old_global[rank_offset : rank_offset + local_token_capacity]
    # A poisoned buffer catches skipped writes and uncleared padding.
    direct = torch.full_like(local_residual, float("nan"))
    direct[shard_len:] = 0
    torch.npu.synchronize()
    if shard_len:
        matmul(rt, attention[:shard_len], weight, old_local[:shard_len])
        matmul(rt, attention[:shard_len], weight, direct[:shard_len])
    reduce_scatter(rt, old_local, old_global)
    torch.testing.assert_close(direct, old_local, rtol=0, atol=0)
    if shard_len:
        expected = torch.nn.functional.linear(attention[:shard_len].float(), weight.float())
        torch.testing.assert_close(direct[:shard_len].float(), expected, rtol=0.02, atol=0.002)

    # The next residual/Norm must see the same token ordering and padding.
    old_residual = local_residual.clone()
    new_residual = local_residual.clone()
    torch.npu.synchronize()
    add_and_rmsnorm(rt, old_local, old_residual, norm_weight, old_local, 1e-5)
    add_and_rmsnorm(rt, direct, new_residual, norm_weight, direct, 1e-5)
    torch.testing.assert_close(new_residual, old_residual, rtol=0, atol=0)
    torch.testing.assert_close(direct, old_local, rtol=0, atol=0)
    if rank == 0:
        print(f"PASS tokens={tokens} TP={tp} dtype={dtype}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    tp = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    try:
        rt = Runtime(local_rank, size=512, rank=rank, tp_size=tp, dp_size=1)
        for dtype in (torch.bfloat16, torch.float16):
            for tokens in sorted({1, tp - 1, tp, tp + 1, 2 * tp + 3, 65}):
                if tokens > 0:
                    run_case(rt, rank, tp, tokens, dtype)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
