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
# row-merged result of shape [2*m, n]: rows [0, m) hold Y_low, rows [m, 2*m) hold
# Y_high (low-first / high-after, NOT interleaved). The post-stage op takes this
# single merged matrix (no separate y_high/y_low pointers) and computes:
#
#   Y[r] = (Y_high[r+m] * 16 + Y_low[r] + scale_bias[col]) * perTokenScale[row]
#
# where
#   y_merged      : fp16 [2*m, n]  — low rows [0,m), high rows [m,2*m)
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
        y_merged:        fp16 [2*m, n]  (low rows [0,m), high rows [m,2*m))
        scale_bias:      fp32 [n]
        per_token_scale: fp32 [m]
    Returns:
        bf16 [m, n]
    """
    m = y_merged.shape[0] // 2
    y_low = y_merged[:m].float()          # rows [0, m)
    y_high = y_merged[m:].float()         # rows [m, 2*m)
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

for m, n in test_cases:
    # Random inputs. Keep magnitudes modest so fp16->fp32->bf16 round-trips
    # cleanly; per_token_scale must be nonzero.
    y_low = torch.randn(m, n, dtype=torch.float16, device=f"npu:{npu_devid}")
    y_high = torch.randn(m, n, dtype=torch.float16, device=f"npu:{npu_devid}")
    # Row-merged [2*m, n]: low-first, high-after (matches the Mid-stage layout).
    y_merged = torch.cat([y_low, y_high], dim=0)
    scale_bias = torch.randn(n, dtype=torch.float32, device=f"npu:{npu_devid}")
    per_token_scale = torch.randn(m, dtype=torch.float32, device=f"npu:{npu_devid}")
    per_token_scale = torch.where(per_token_scale == 0, torch.ones_like(per_token_scale),
                                 per_token_scale)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{npu_devid}")

    # torch reference (compute on NPU in fp32, same as kernel's fp32 chain)
    expected = msd_merge_dequant_ref(y_merged, scale_bias, per_token_scale)

    torch.npu.synchronize()
    msd_merge_dequant(rt, y_merged, scale_bias, per_token_scale, out)
    torch.npu.synchronize()

    # bf16 has ~3 decimal digits of precision; atol=1, rtol=1/64 are generous
    # enough for the merge+bias+scale chain while still catching real bugs.
    try:
        torch.testing.assert_close(expected, out, atol=1, rtol=1 / 64)
        print(f"msd_merge_dequant [{m}, {n}] output check passed")
    except AssertionError as e:
        print(f"[{m}, {n}] output check FAILED:")
        print(f"{e}")
        print(f"y_merged: {y_merged[:4, :8]} ... {y_merged[m:m+4, :8]}, shape: {y_merged.shape}")
        print(f"scale_bias: {scale_bias[:8]}, shape: {scale_bias.shape}")
        print(f"per_token_scale: {per_token_scale[:8]}, shape: {per_token_scale.shape}")
        print(f"expected: {expected[:4, :8]}")
        print(f"xlite:    {out[:4, :8]}")
        # element-wise diff for debugging
        diff = (expected.float() - out.float()).abs()
        print(f"max abs diff: {diff.max().item():.6f} at index {diff.argmax().item()}")
        print(f"mean abs diff: {diff.mean().item():.6f}")
