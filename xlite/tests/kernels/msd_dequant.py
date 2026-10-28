#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
#
# Single-op test for the MSD W4A8 post-stage kernel `msd_merge_dequant`.
#
# In the end-to-end MSD pipeline, the Mid-stage INT4×INT4 matmul produces a
# row-merged result of shape [2*m, n] interleaved by token: rows (2r, 2r+1) hold
# (Y_low, Y_high) for token r. The post-stage op takes this single merged matrix
# (no separate y_high/y_low pointers) and computes:
#
#   Y[r] = (Y_high[2r+1] * 16 + Y_low[2r] + scale_bias[col]) * perTokenScale[row]
#
# where
#   y_merged      : fp16 [2*m, n]  — interleaved (2r = low, 2r+1 = high)
#   scale_bias    : fp32 [n]       — per-column low-nibble -8 offset compensation
#   perTokenScale : fp32 [m]       — per-row activation dequant scale
#   out           : bf16 [m, n]
#
# pnum_tokens (DP-pad real token count) is passed as nullptr on the C++ side, so
# the kernel uses m as the real output row count.

import torch
import torch_npu
from xlite._C import Runtime, msd_merge_dequant

npu_devid = 0
rt = Runtime(npu_devid, 500)
torch.npu.set_device(npu_devid)
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

def msd_merge_dequant_ref(y_merged, scale_bias, per_token_scale):
    """Pure-torch reference for the MSD merge + bias + per-token dequant.

    Args:
        y_merged:        fp16 [2*m, n]  (interleaved: rows 2r = low, 2r+1 = high)
        scale_bias:      fp32 [n]
        per_token_scale: fp32 [m]
    Returns:
        bf16 [m, n]
    """
    m = y_merged.shape[0] // 2
    y_low = y_merged[0::2].float()          # rows 0, 2, 4, ... = low
    y_high = y_merged[1::2].float()        # rows 1, 3, 5, ... = high
    merged = y_high * 16.0 + y_low                       # [m, n]
    merged = merged + scale_bias.float().view(1, -1)     # per-column bias
    merged = merged * per_token_scale.float().view(-1, 1) # per-row scale
    return merged.to(torch.bfloat16)


# (m, n) test cases — m output rows, n columns. Small N (n < ~12288) to stay
# within the kernel's UB layout (single-row UB buffer, no N-tiling).
test_cases = [
    [8, 96],
    [40, 512],
    [200, 2048],
    [1, 2048],
    [8192, 6144],
]

def msd_merge_dequant_multi_ref(y_merged, scale_biases, counts, per_token_scale):
    """Multi-expert reference: row r's scale_bias is picked by its expert id,
    located via the per-expert token counts (rows are grouped by expert on the
    row axis, in counts order)."""
    m = y_merged.shape[0] // 2
    num_experts = len(counts)
    y_low = y_merged[0::2].float()
    y_high = y_merged[1::2].float()
    merged = y_high * 16.0 + y_low                                          # [m, n]
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=y_merged.device), counts.to(y_merged.device))
    bias_matrix = torch.cat([b.view(1, -1) for b in scale_biases], dim=0)   # [E, n]
    merged = merged + bias_matrix[expert_ids]                               # [m, n]
    merged = merged * per_token_scale.float().view(-1, 1)
    return merged.to(torch.bfloat16)


for m, n in test_cases:
    # Random inputs. Keep magnitudes modest so fp16->fp32->bf16 round-trips
    # cleanly; per_token_scale must be nonzero.
    y_low = torch.randn(m, n, dtype=torch.float16, device=f"npu:{npu_devid}")
    y_high = torch.randn(m, n, dtype=torch.float16, device=f"npu:{npu_devid}")
    # Row-merged [2*m, n] interleaved: rows (2r, 2r+1) = (low, high) for token r.
    y_merged = torch.stack([y_low, y_high], dim=1).reshape(2 * m, n)
    scale_bias = torch.randn(1, n, dtype=torch.float32, device=f"npu:{npu_devid}")
    per_token_scale = torch.randn(m, dtype=torch.float32, device=f"npu:{npu_devid}")
    per_token_scale = torch.where(per_token_scale == 0, torch.ones_like(per_token_scale),
                                 per_token_scale)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{npu_devid}")

    # torch reference (compute on NPU in fp32, same as kernel's fp32 chain)
    expected = msd_merge_dequant_ref(y_merged, scale_bias, per_token_scale)

    # New multi-expert signature: scale_biases (per-expert [n] list) + counts. Single-expert
    # case = one expert with counts=[m].
    counts = torch.tensor([m], dtype=torch.int32, device=f"npu:{npu_devid}")
    torch.npu.synchronize()
    msd_merge_dequant(rt, y_merged, [scale_bias], counts, per_token_scale, out)
    torch.npu.synchronize()

    # bf16 has ~3 decimal digits of precision; atol=1, rtol=1/64 are generous
    # enough for the merge+bias+scale chain while still catching real bugs.
    try:
        torch.testing.assert_close(expected, out, atol=1, rtol=1 / 64)
        print(f"msd_merge_dequant [{m}, {n}] output check passed")
    except AssertionError as e:
        print(f"msd_merge_dequant [{m}, {n}] output check failed")
        print(f"{e}")
        print(f"expected: {expected[:4, :8]}")
        print(f"xlite:    {out[:4, :8]}")

# Multi-expert cases: per-expert scale_bias must be selected by the row's expert
# id (counts prefix sums), including skewed splits that force frequent expert
# switches inside each block's strided row sequence.
multi_cases = [
    (200, 4096, 4, [50, 50, 50, 50]),
    (200, 4096, 8, [80, 40, 20, 18, 15, 12, 10, 5]),
    (8192, 4096, 3, [4096, 3072, 1024]),
]

for m, n, num_experts, token_counts in multi_cases:
    assert sum(token_counts) == m
    y_low = torch.randn(m, n, dtype=torch.float16, device=f"npu:{npu_devid}")
    y_high = torch.randn(m, n, dtype=torch.float16, device=f"npu:{npu_devid}")
    y_merged = torch.stack([y_low, y_high], dim=1).reshape(2 * m, n)
    scale_biases = [
        torch.randn(1, n, dtype=torch.float32, device=f"npu:{npu_devid}")
        for _ in range(num_experts)
    ]
    per_token_scale = torch.randn(m, dtype=torch.float32, device=f"npu:{npu_devid}")
    per_token_scale = torch.where(per_token_scale == 0, torch.ones_like(per_token_scale),
                                 per_token_scale)
    counts = torch.tensor(token_counts, dtype=torch.int32, device=f"npu:{npu_devid}")
    out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{npu_devid}")

    expected = msd_merge_dequant_multi_ref(y_merged, scale_biases, counts, per_token_scale)

    torch.npu.synchronize()
    msd_merge_dequant(rt, y_merged, scale_biases, counts, per_token_scale, out)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(expected, out, atol=1, rtol=1 / 64)
        print(f"msd_merge_dequant multi-expert [{m}, {n}, E={num_experts}] output check passed")
    except AssertionError as e:
        print(f"msd_merge_dequant multi-expert [{m}, {n}, E={num_experts}] output check failed")
        print(f"{e}")
        print(f"expected: {expected[:4, :8]}")
        print(f"xlite:    {out[:4, :8]}")


# Determinism cases: bias_buf is reused across an expert's rows and reloaded on
# expert switches, so multi-expert row layouts stress the MTE2-overwrite vs
# V-read interleaving (single expert never reloads after its first row and
# cannot expose it). RUNS identical launches must produce bit-identical output.
DETERMINISM_RUNS = 10

determinism_cases = [
    # (m, n, num_experts, token_counts) — None splits m evenly across experts.
    (8, 96, 1, None),
    (40, 512, 1, None),
    (200, 2048, 1, None),
    (200, 4096, 1, None),
    (200, 6144, 1, None),
    (200, 4096, 4, None),
    (200, 4096, 8, [80, 40, 20, 18, 15, 12, 10, 5]),
    # Real prefill scale: expertsSorted rows x inter*2, skewed routing counts.
    (33760, 4096, 8, [8640, 6800, 5200, 4400, 3200, 2800, 2000, 720]),
]


def check_determinism(m, n, num_experts, token_counts, runs=DETERMINISM_RUNS):
    """Run the kernel `runs` times on identical inputs.

    Returns (all_same, max_diff, fail_msgs): all_same is True only when every
    run is bit-identical to the first; fail_msgs carries per-run diff details.
    """
    torch.manual_seed(12345)
    y_merged = torch.randn(2 * m, n, dtype=torch.float16, device=f"npu:{npu_devid}")
    per_token_scale = torch.randn(m, dtype=torch.float32, device=f"npu:{npu_devid}")
    per_token_scale = torch.where(per_token_scale == 0, torch.ones_like(per_token_scale),
                                 per_token_scale)
    scale_biases = [
        torch.randn(1, n, dtype=torch.float32, device=f"npu:{npu_devid}")
        for _ in range(num_experts)
    ]
    if token_counts is not None:
        assert sum(token_counts) == m, f"sum(token_counts)={sum(token_counts)} != m={m}"
        counts = torch.tensor(token_counts, dtype=torch.int32, device=f"npu:{npu_devid}")
    else:
        base = m // num_experts
        rem = m - base * num_experts
        counts_list = [base + (1 if i < rem else 0) for i in range(num_experts)]
        counts = torch.tensor(counts_list, dtype=torch.int32, device=f"npu:{npu_devid}")

    out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{npu_devid}")
    outs = []
    for _ in range(runs):
        # NaN sentinel: any row the kernel fails to write stays detectable.
        out.fill_(float("nan"))
        torch.npu.synchronize()
        msd_merge_dequant(rt, y_merged, scale_biases, counts, per_token_scale, out)
        torch.npu.synchronize()
        outs.append(out.clone())

    all_same = True
    max_diff = 0.0
    fail_msgs = []
    for i in range(1, runs):
        diff = (outs[0].float() - outs[i].float()).abs()
        md = diff.max().item()
        max_diff = max(max_diff, md)
        neq = (outs[0] != outs[i]).sum().item()
        if neq > 0:
            all_same = False
            idx = torch.where(outs[0] != outs[i])
            r0, c0 = idx[0][0].item(), idx[1][0].item()
            fail_msgs.append(f"run{i} vs run0: {neq} elems differ, max_diff={md:.6f}, "
                             f"first at [{r0},{c0}]: run0={outs[0][r0, c0].item()} "
                             f"run{i}={outs[i][r0, c0].item()}")
    return all_same, max_diff, fail_msgs


for m, n, num_experts, token_counts in determinism_cases:
    all_same, max_diff, fail_msgs = check_determinism(m, n, num_experts, token_counts)
    try:
        assert all_same, "\n".join(fail_msgs)
        print(f"msd_merge_dequant determinism [{m}, {n}, E={num_experts}] check passed "
              f"({DETERMINISM_RUNS} runs, max_diff={max_diff:.6f})")
    except AssertionError as e:
        print(f"msd_merge_dequant determinism [{m}, {n}, E={num_experts}] check failed")
        print(f"{e}")
