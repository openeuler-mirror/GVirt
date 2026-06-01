#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
# Test for the reorder_moe kernel, which permutes token rows between two layouts:
#   - source-grouped: tokens are grouped by their source EP rank.
#   - expert-grouped: tokens are grouped by their target expert index.
#
# The kernel supports two directions:
#   forward  (True):  source-grouped -> expert-grouped (permute into expert layout)
#   reverse  (False): expert-grouped -> source-grouped (restore original order)
# ===============================================================================

import logging
import torch
import numpy as np
from xlite._C import Runtime, reorder_moe

logging.getLogger().setLevel(logging.INFO)

# Single-card test: device 0, 500 MB tensor pool.
rt = Runtime(0, 500)
torch.npu.set_device(0)

HIDDEN_SIZE = 3072
DTYPE_LIST = [torch.float, torch.float16, torch.bfloat16]


def reorder_forward_ref(inp_host, counts_host, local_start, local_end):
    """CPU reference implementation for forward reorder.

    Permutes token rows from source-grouped layout (grouped by EP rank)
    into expert-grouped layout (grouped by expert index).

    Args:
        inp_host:  Input tokens in source-grouped order  [total_tokens, hidden_size].
        counts_host: Per-source per-expert token counts  [moe_ep_size, n_experts].
        local_start: First local expert index (inclusive).
        local_end:   Last local expert index (exclusive).

    Returns:
        Output tokens in expert-grouped order [total_tokens, hidden_size].
    """
    moe_ep_size = counts_host.shape[0]
    n_experts = counts_host.shape[1]
    total_tokens = int(counts_host[:, local_start:local_end].sum())
    out = np.zeros((total_tokens, inp_host.shape[1]), dtype=np.float32)

    # Compute per-expert start offsets in the output buffer.
    # expert_offsets[e] = total tokens destined for experts [local_start, e).
    expert_offsets = np.zeros(n_experts, dtype=np.int32)
    running = 0
    for e in range(local_start, local_end):
        expert_offsets[e] = running
        for i in range(moe_ep_size):
            running += counts_host[i, e]

    # Compute per-source start offsets in the input buffer.
    # chunk_offsets[i] = total tokens from sources [0, i) for local experts.
    chunk_offsets = np.zeros(moe_ep_size, dtype=np.int32)
    chunk_acc = 0
    for i in range(moe_ep_size):
        chunk_offsets[i] = chunk_acc
        for e in range(local_start, local_end):
            chunk_acc += counts_host[i, e]

    # Scatter: for each source chunk, copy its tokens to the correct expert region.
    for i in range(moe_ep_size):
        chunk_run = chunk_offsets[i]
        for e in range(local_start, local_end):
            cnt = counts_host[i, e]
            if cnt == 0:
                continue
            src_start = chunk_run
            dst_start = expert_offsets[e]
            out[dst_start:dst_start + cnt] = inp_host[src_start:src_start + cnt]
            expert_offsets[e] += cnt
            chunk_run += cnt
    return out


def reorder_reverse_ref(inp_host, counts_host, local_start, local_end):
    """CPU reference implementation for reverse reorder.

    Permutes token rows from expert-grouped layout back into source-grouped
    layout (the inverse of forward).

    Args:
        inp_host:  Input tokens in expert-grouped order  [total_tokens, hidden_size].
        counts_host: Per-source per-expert token counts  [moe_ep_size, n_experts].
        local_start: First local expert index (inclusive).
        local_end:   Last local expert index (exclusive).

    Returns:
        Output tokens in source-grouped order [total_tokens, hidden_size].
    """
    moe_ep_size = counts_host.shape[0]
    n_experts = counts_host.shape[1]
    total_tokens = int(counts_host[:, local_start:local_end].sum())
    out = np.zeros((total_tokens, inp_host.shape[1]), dtype=np.float32)

    # Compute per-expert start offsets in the input buffer (same as forward).
    expert_offsets = np.zeros(n_experts, dtype=np.int32)
    running = 0
    for e in range(local_start, local_end):
        expert_offsets[e] = running
        for i in range(moe_ep_size):
            running += counts_host[i, e]

    # Compute per-source start offsets in the output buffer (same as forward).
    chunk_offsets = np.zeros(moe_ep_size, dtype=np.int32)
    chunk_acc = 0
    for i in range(moe_ep_size):
        chunk_offsets[i] = chunk_acc
        for e in range(local_start, local_end):
            chunk_acc += counts_host[i, e]

    # Gather: for each source chunk, read tokens from the correct expert region.
    for i in range(moe_ep_size):
        chunk_run = chunk_offsets[i]
        for e in range(local_start, local_end):
            cnt = counts_host[i, e]
            if cnt == 0:
                continue
            src_start = expert_offsets[e]
            dst_start = chunk_run
            out[dst_start:dst_start + cnt] = inp_host[src_start:src_start + cnt]
            expert_offsets[e] += cnt
            chunk_run += cnt
    return out


def generate_random_counts(rows, cols, total):
    """Generate a random non-negative integer matrix of shape (rows, cols)
    whose entries sum exactly to ``total``.

    Uses the "random split points" method on a line segment of length ``total``:
    generate ``rows*cols - 1`` random split points in [0, total], sort them,
    then the gaps between consecutive splits (including 0 and total) give the
    per-cell counts.  This guarantees uniform distribution over all possible
    count matrices with the given sum.

    Args:
        rows:  Number of rows (e.g. moe_ep_size).
        cols:  Number of columns (e.g. n_routed_experts).
        total: Target sum of all entries.

    Returns:
        A numpy int32 array of shape (rows, cols).
    """
    num_elements = rows * cols

    # Draw random split points along [0, total] and sort them.
    splits = np.random.randint(0, total + 1, size=num_elements - 1)
    splits = np.sort(splits)

    # Gap lengths between consecutive splits are the random counts.
    flat_counts = np.diff(np.concatenate(([0], splits, [total])))

    # Reshape to the target matrix shape.
    counts = flat_counts.reshape(rows, cols).astype(np.int32)
    return counts


# Loop over MoE configurations to test.
# moe_ep_size: number of expert-parallel ranks (simulated on a single card).
# n_routed_experts: total number of routed experts.

for moe_ep_size in [8, 16]:
    for n_routed_experts in [160, 256]:
        n_local_experts = n_routed_experts // moe_ep_size

        # Simulate a small batch: m tokens, each selecting n_act experts.
        m = 8
        n_act = 8
        total_assign = m * n_act  # Total expert assignments across all sources.

        # Generate a random dispatch count matrix for all experts.
        # Shape: [moe_ep_size, n_routed_experts].
        counts_host = generate_random_counts(moe_ep_size, n_routed_experts, total_assign)
        counts = torch.from_numpy(counts_host).to(torch.int32).to("npu:0")
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        # Iterate over local expert shards. Each shard has n_local_experts
        # consecutive experts, simulating one EP rank's responsibility.
        for local_start in range(0, n_routed_experts, n_local_experts):
            local_end = local_start + n_local_experts
            total_tokens = int(counts_host[:, local_start:local_end].sum())
            if total_tokens == 0:
                continue

            for test_dtype in DTYPE_LIST:
                # Create random input tokens on device.
                inp = torch.randn(total_tokens, HIDDEN_SIZE, dtype=test_dtype, device="npu:0")
                inp_host = inp.cpu().float().numpy()

                # Allocate output buffers for forward and reverse passes.
                out_fwd = torch.zeros(total_tokens, HIDDEN_SIZE, dtype=test_dtype, device="npu:0")
                out_rev = torch.zeros(total_tokens, HIDDEN_SIZE, dtype=test_dtype, device="npu:0")

                # ---- Forward: source-grouped -> expert-grouped ----
                torch.npu.synchronize()
                reorder_moe(rt, inp, out_fwd, counts, HIDDEN_SIZE,
                            local_start, local_end, True)
                torch.npu.synchronize()
                fwd_got = out_fwd.cpu().float().numpy()
                fwd_ref = reorder_forward_ref(inp_host, counts_host, local_start, local_end)

                # ---- Reverse: expert-grouped -> source-grouped ----
                # Use forward output as input to verify round-trip consistency.
                torch.npu.synchronize()
                reorder_moe(rt, out_fwd, out_rev, counts, HIDDEN_SIZE,
                            local_start, local_end, False)
                torch.npu.synchronize()
                rev_got = out_rev.cpu().float().numpy()
                rev_ref = reorder_reverse_ref(fwd_ref, counts_host, local_start, local_end)

                # Validate forward pass.
                fwd_got_t = torch.from_numpy(fwd_got)
                fwd_ref_t = torch.from_numpy(fwd_ref)
                try:
                    torch.testing.assert_close(fwd_got_t, fwd_ref_t, atol=0, rtol=0)
                except AssertionError as e:
                    logging.error(f"\n[FORWARD] ep={moe_ep_size} dtype={test_dtype} "
                          f"local=[{local_start},{local_end}) FAILED:")
                    logging.error(e)
                    logging.error(f"  [FORWARD] inp ({inp.shape}):\n{inp}")
                    logging.error(f"  [FORWARD] ref ({fwd_ref_t.shape}):\n{fwd_ref_t}")
                    logging.error(f"  [FORWARD] got ({fwd_got_t.shape}):\n{fwd_got_t}")
                    continue

                # Validate reverse pass.
                rev_got_t = torch.from_numpy(rev_got)
                rev_ref_t = torch.from_numpy(rev_ref)
                try:
                    torch.testing.assert_close(rev_got_t, rev_ref_t, atol=0, rtol=0)
                except AssertionError as e:
                    logging.error(f"\n[REVERSE] ep={moe_ep_size} dtype={test_dtype} "
                          f"local=[{local_start},{local_end}) FAILED:")
                    logging.error(e)
                    logging.error(f"  [REVERSE] inp ({out_fwd.shape}):\n{out_fwd}")
                    logging.error(f"  [REVERSE] ref ({rev_ref_t.shape}):\n{rev_ref_t}")
                    logging.error(f"  [REVERSE] got ({rev_got_t.shape}):\n{rev_got_t}")
                    continue

        logging.info(f"ep={moe_ep_size} experts={n_routed_experts} "
                     f"dtypes={[str(d) for d in DTYPE_LIST]}  executed!")
