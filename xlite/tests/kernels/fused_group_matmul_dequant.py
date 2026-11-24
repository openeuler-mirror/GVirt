#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os

import torch
import torch_npu

from xlite._C import (
    Runtime,
    dequant,
    fused_group_matmul_dequant,
    group_matmul,
    quant_dynamic,
)

# allow weight_nz
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

dev_id = int(os.environ.get("XLITE_TEST_DEVICE", "0"))
rt = Runtime(dev_id, size=500)
torch.npu.set_device(dev_id)
npu_dev = f"npu:{dev_id}"

# (m, out_dim, in_dim, group_num, has_scale, weight_nz) — GLM MoE-ish shapes
# plus m0/128 boundaries (kM < m0, kM crossing m0, multi-round tile dispatch,
# zero-count experts) and matmul k-tiling paths (k above the single-tile k
# budget forces kLoop >= 2 inside the Matmul sub-op). The quant stage runs
# OUTSIDE the fused kernel: both arms consume the same quant_dynamic output
# (int8 x + per-token scale); has_scale=False passes an empty scale to both
# arms, exercising the dequant mirror's no-scale path. Ragged k (k % 32 != 0)
# is rejected by the quant producer's host entry, so it is not covered here.
test_sizes = [
    [20, 256, 512, 4, True, False],
    [20, 256, 512, 4, False, False],
    [8, 4096, 6144, 16, True, False],  # sparse decode: 15 zero-count experts, kM < m0/2
    [129, 2048, 7168, 8, True, False],  # kM crosses m0=128
    [1030, 2048, 7168, 6, True, False],  # multi-round tiles per core
    [4096, 2048, 7168, 16, True, False],
    [4096, 6144, 2048, 16, False, False],
    [4096, 6144, 2048, 16, False, True],  # NZ weights
    # matmul k-tiling paths: k > k0 forces kLoop >= 2 in the Matmul sub-op
    [64, 256, 16384, 2, True, False],
    [64, 256, 4096, 2, False, False],
    # AIC-ahead-of-AIV race: mLoop=1 with a large nLoop lets the single AIC
    # finish all tiles and return while the paired AIVs are still in their
    # last dequant tiles (MIX retirement contract: the kernel completes only
    # when every core finishes)
    [128, 6144, 768, 1, True, False],
    # tiny routed batches (m < g, most or all experts zero-count) in the
    # deployment layout — decode tail below the dispatch floor
    [1, 2048, 7168, 4, True, False],
    [2, 7168, 2048, 16, True, True],  # NZ w2-like, m < g
    [5, 4096, 7168, 16, True, True],  # NZ w13-like, m < g
    [8, 7168, 2048, 16, True, True],  # NZ w2-like, m = g/2
]

# Long-sequence prefill shapes exercise mLoop >> aicNum (multiple tile rounds
# per AIC/AIV pair, cross-core flag pairing runs many rounds).
long_seq_sizes = [
    [2048, 2048, 7168, 16, True, False],
    [8192, 2048, 7168, 16, True, False],
    [8192, 6144, 2048, 16, False, False],
]

for transpose in [False, True]:
    for m, n, k, group_num, has_scale, weight_nz in test_sizes:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=npu_dev) * 2.0
        counts = torch.zeros(group_num, dtype=torch.int32, device=npu_dev)
        base = m // group_num
        counts[:] = base
        counts[0] += m - base * group_num
        # force a zero-count expert on a middle slot whenever room allows
        if group_num >= 4 and counts[group_num // 2] > 1:
            counts[group_num // 2 + 1] += counts[group_num // 2]
            counts[group_num // 2] = 0

        weights = []
        deq_scales_fixpipe = []
        for i in range(group_num):
            weight_ref = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=npu_dev)
            if transpose:
                y_in = weight_ref.t().contiguous()
            else:
                y_in = weight_ref
            if weight_nz:
                ACL_FORMAT_FRACTAL_NZ = 29
                y_in = torch_npu.npu_format_cast(y_in, ACL_FORMAT_FRACTAL_NZ)
            weights.append(y_in)

            # fixpipe: fp32 via bf16 so the conversion is lossless
            deq_scale = torch.rand(n, dtype=torch.float32, device=npu_dev) * 0.015 + 0.001
            deq_scale = deq_scale.to(torch.bfloat16).to(torch.float32)
            # fixpipe hw requirement: uint64 storing fp32, high 32b zero
            scale = torch.zeros(n * 2, dtype=torch.float32, device=npu_dev)
            scale[0::2] = deq_scale
            deq_scales_fixpipe.append(scale)

        # quant runs outside both arms: one shared int8 input + per-token scale
        # (quant_dynamic always writes a scale; has_scale=False discards it and
        # both arms take the no-scale dequant path)
        token_scale = torch.empty(m, dtype=torch.float32, device=npu_dev)
        out_scale = token_scale if has_scale else torch.empty(0, dtype=torch.float32, device=npu_dev)

        # reference: serial two-kernel chain group_matmul -> dequant
        xq = torch.empty(m, k, dtype=torch.int8, device=npu_dev)
        serial_tmp = torch.zeros(m, n, dtype=torch.float16, device=npu_dev)
        standard = torch.zeros(m, n, dtype=torch.bfloat16, device=npu_dev)
        fused_out = torch.zeros(m, n, dtype=torch.bfloat16, device=npu_dev)

        torch.npu.synchronize()
        quant_dynamic(rt, x, token_scale, xq)
        torch.npu.synchronize()
        group_matmul(rt, xq, weights, deq_scales_fixpipe, counts, 0, group_num, n, k, serial_tmp, weight_nz, transpose)
        dequant(rt, serial_tmp, out_scale, standard, has_scale)
        torch.npu.synchronize()

        fused_group_matmul_dequant(
            rt, xq, weights, deq_scales_fixpipe, counts, 0, group_num, n, k, fused_out, out_scale, weight_nz, transpose
        )
        torch.npu.synchronize()
        print(
            f"[m={m}, n={n}, k={k}, g={group_num}] transpose:{transpose} has_scale:{has_scale} "
            f"weight_nz:{weight_nz} fused group matmul dequant executed!"
        )

        try:
            torch.testing.assert_close(standard.to("cpu"), fused_out.to("cpu"), atol=1e-2, rtol=1e-2)
        except AssertionError as e:
            print(f"{e}")
            print(f"serial: {standard}")
            print(f"fused: {fused_out}")

for m, n, k, group_num, has_scale, weight_nz in long_seq_sizes:
    for transpose in [False, True]:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=npu_dev) * 2.0
        counts = torch.zeros(group_num, dtype=torch.int32, device=npu_dev)
        base = m // group_num
        counts[:] = base
        counts[0] += m - base * group_num
        if group_num >= 4 and counts[group_num // 2] > 1:
            counts[group_num // 2 + 1] += counts[group_num // 2]
            counts[group_num // 2] = 0

        weights = []
        deq_scales_fixpipe = []
        for i in range(group_num):
            weight_ref = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=npu_dev)
            y_in = weight_ref.t().contiguous() if transpose else weight_ref
            weights.append(y_in)

            deq_scale = torch.rand(n, dtype=torch.float32, device=npu_dev) * 0.015 + 0.001
            deq_scale = deq_scale.to(torch.bfloat16).to(torch.float32)
            scale = torch.zeros(n * 2, dtype=torch.float32, device=npu_dev)
            scale[0::2] = deq_scale
            deq_scales_fixpipe.append(scale)

        token_scale = torch.empty(m, dtype=torch.float32, device=npu_dev)
        out_scale = token_scale if has_scale else torch.empty(0, dtype=torch.float32, device=npu_dev)

        xq = torch.empty(m, k, dtype=torch.int8, device=npu_dev)
        serial_tmp = torch.zeros(m, n, dtype=torch.float16, device=npu_dev)
        standard = torch.zeros(m, n, dtype=torch.bfloat16, device=npu_dev)
        fused_out = torch.zeros(m, n, dtype=torch.bfloat16, device=npu_dev)

        torch.npu.synchronize()
        quant_dynamic(rt, x, token_scale, xq)
        torch.npu.synchronize()
        group_matmul(rt, xq, weights, deq_scales_fixpipe, counts, 0, group_num, n, k, serial_tmp, weight_nz, transpose)
        dequant(rt, serial_tmp, out_scale, standard, has_scale)
        torch.npu.synchronize()

        fused_group_matmul_dequant(
            rt, xq, weights, deq_scales_fixpipe, counts, 0, group_num, n, k, fused_out, out_scale, weight_nz, transpose
        )
        torch.npu.synchronize()
        print(
            f"[m={m}, n={n}, k={k}, g={group_num}] transpose:{transpose} has_scale:{has_scale} "
            f"weight_nz:{weight_nz} fused group matmul dequant executed!"
        )

        try:
            torch.testing.assert_close(standard.to("cpu"), fused_out.to("cpu"), atol=1e-2, rtol=1e-2)
        except AssertionError as e:
            print(f"{e}")
            print(f"serial: {standard}")
            print(f"fused: {fused_out}")

# Negative case: empty weights (counts of length 0) must be rejected at the
# binding entry, before any weights[0] dereference.
empty_counts = torch.zeros(0, dtype=torch.int32, device=npu_dev)
xq_guard = torch.zeros(1, 512, dtype=torch.int8, device=npu_dev)
out_guard = torch.zeros(1, 256, dtype=torch.bfloat16, device=npu_dev)
try:
    fused_group_matmul_dequant(rt, xq_guard, [], [], empty_counts, 0, 0, 256, 512, out_guard, None, False, False)
    raise AssertionError("empty-weights call must raise; it returned silently")
except (ValueError, RuntimeError) as e:
    assert "weights must not be empty" in str(e), f"unexpected error: {e}"
    print(f"empty weights rejected at binding entry: {e}")
