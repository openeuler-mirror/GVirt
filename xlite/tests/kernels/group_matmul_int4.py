#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import torch_npu
from xlite._C import Runtime, group_matmul, unpack_activation, msd_merge_dequant

# allow weight_nz
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

npu_id = 0
npu_dev = f"npu:{npu_id}"
rt = Runtime(npu_id, 500)
torch.npu.set_device(npu_id)

m = 20
in_dim = 2048
out_dim = 512
group_num = 8

# Generate random counts that sum to m
counts = torch.ones(group_num, dtype=torch.int32, device=npu_dev)
remaining = m - group_num
if remaining > 0:
    random_vals = torch.rand(group_num, device=npu_dev)
    random_vals = (random_vals / random_vals.sum() * remaining).round().int()
    sum_random = random_vals.sum().item()
    if sum_random != remaining:
        diff = remaining - sum_random
        random_vals[0] += diff
    counts += random_vals
counts_cpu = counts.cpu().tolist()


def w4a8_msd_ref(x, weights, counts_list, deq_scales, scale_biases, per_token_scales,
                 transpose=False):
    """CPU reference for the W4A8 MSD group_matmul (Unpack + INT4 GroupMatmul + Merge).

    Mirrors the three-stage pipeline (XModel::ForwardMoEMSD):
      1. unpack   : INT8 [m, k] -> (low4, high4), each INT4 in [-8, 7] (floor decomposition)
      2. group_matmul INT4 : per-expert (low4|high4) [c, k] x W_int4 -> INT32 [c, n]
                             then * deq_scale -> FP16 mid-stage, rows interleaved low/high
      3. merge    : (y_high*16 + y_low + scale_bias) * per_token_scale -> BF16

    x:               [m, k] int8 (QuantDyn output, range [-128, 127])
    weights[i]:      int4 values in [-8, 7] in an int8 container, [n, k](transpose=False)
                    or [k, n](transpose=True)
    counts_list:     per-expert token counts (un-doubled)
    deq_scales[i]:   [n] float32 per-channel antiquant scale
    scale_biases[i]: [n] float32 = 8 * sum_k(W[k, col] * deq_scale[col])
    per_token_scales[i]: [c] float32 (QuantDyn absmax scale per token, per expert)
    Returns BF16 [m, n].
    """
    results = []
    start = 0
    for i in range(group_num):
        c = counts_list[i]
        end = start + c
        x_slice = x[start:end].to("cpu")  # [c, k] int8
        # Stage 1: floor decomposition (arithmetic shift, NOT trunc)
        high4 = (x_slice.to(torch.int32) >> 4).to(torch.int8)          # [-8, 7]
        low4 = ((x_slice.to(torch.int32) & 0x0F) - 8).to(torch.int8)    # [-8, 7]
        # Stage 2: INT4 x INT4 -> INT32, then * per-channel deq_scale -> FP16
        w = weights[i].to("cpu").to(torch.int32)
        w_t = w if transpose else w.t()  # [k, n]
        y_low_i32 = torch.matmul(low4.to(torch.int32), w_t)      # [c, n]
        y_high_i32 = torch.matmul(high4.to(torch.int32), w_t)    # [c, n]
        scale = deq_scales[i].to("cpu").view(1, -1)
        y_low_fp16 = (y_low_i32.float() * scale).half()
        y_high_fp16 = (y_high_i32.float() * scale).half()
        # Stage 3: merge + scale_bias + per_token_scale
        merged = y_high_fp16.to(torch.float32) * 16.0 + y_low_fp16.to(torch.float32)
        merged = merged + scale_biases[i].to("cpu").view(1, -1)
        out = merged * per_token_scales[i].to("cpu").view(-1, 1)
        results.append(out.to(torch.bfloat16))
        start = end
    return torch.cat(results, dim=0)  # [m, n] bf16


for weight_nz in [True, False]:
    for transpose in [False, True]:
        n = out_dim if not transpose else in_dim
        k = in_dim if not transpose else out_dim
        # W4A8 activation: INT8 [m, k] (simulates QuantDyn output)
        x = torch.randint(low=-128, high=128, size=(m, k), dtype=torch.int8, device=npu_dev)
        # Per-token activation dequant scale (QuantDyn absmax scale, positive), per expert
        per_token_scales = []
        weights_int8 = []        # int4 values in [-8, 7] as int8 (for torch ref + W8A8)
        weights_packed = []     # npu int4pack (for xlite group_matmul)
        deq_scales = []          # per-channel fp32 antiquant scale
        deq_scales_fixpipe = []  # fixpipe TF32 layout [2n] fp32
        scale_biases = []        # per-expert [n] fp32 = 8 * sum(W * deq_scale)
        for i in range(group_num):
            if transpose:
                # weight stored as [k, n] (matches group_matmul_dequant.py convention);
                # ref functions transpose to [k,n] for matmul, xlite gets [k,n] int4pack.
                weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32, device=npu_dev)
                weight_int8_i = weight.to(torch.int8).clone()  # [k, n]
            else:
                weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32, device=npu_dev)
                weight_int8_i = weight.to(torch.int8).clone()  # [n, k]
            weights_int8.append(weight_int8_i)

            deq_scale = torch.randn(n, dtype=torch.bfloat16, device=npu_dev)
            deq_scale = deq_scale.to(torch.float32).contiguous().clone()
            # fixpipe硬件要求：以uint64_t存储fp32，高位为0，低位为fp32格式的二进制值
            scale = torch.zeros(n * 2, dtype=torch.float32, device=npu_dev)
            scale[0::2] = deq_scale[0::1]
            scale[1::2] = 0
            deq_scales.append(deq_scale)
            deq_scales_fixpipe.append(scale)

            # scale_bias = 8 * sum_k(W[k, col] * deq_scale[col])  -> [n] fp32
            # (low4's -8 offset compensation, deq_scale folded in). W is logically [n, k]:
            # transpose path stores [k,n], so view as [n,k] first.
            w_nk = weight_int8_i.t().contiguous() if transpose else weight_int8_i  # [n, k]
            sb = 8.0 * (w_nk.to(torch.float32) * deq_scale.view(-1, 1)).sum(dim=1)
            scale_biases.append(sb.contiguous())

            # per-token scale (per expert); QuantDyn absmax scale must be positive
            pts = torch.randn(counts_cpu[i], dtype=torch.float32, device=npu_dev).abs() + 0.1
            per_token_scales.append(pts)

            w_for_pack = weight.contiguous()
            if weight_nz:
                ACL_FORMAT_FRACTAL_NZ = 29
                w_for_pack = torch_npu.npu_format_cast(w_for_pack, ACL_FORMAT_FRACTAL_NZ)
            weights_packed.append(torch_npu.npu_convert_weight_to_int4pack(w_for_pack))

        # ---------- torch W4A8 MSD reference (CPU) ----------
        # Concatenate per-expert per-token scales into [m] following counts order
        pts_cat = torch.cat(per_token_scales, dim=0).to(npu_dev)
        result = w4a8_msd_ref(x, weights_int8, counts_cpu, deq_scales, scale_biases,
                              per_token_scales, transpose)

        # ---------- xlite W4A8 pipeline ----------
        # Stage 1: unpack_activation  INT8 [m, k] -> INT4-packed [2m, k/2] (rows interleaved)
        x_unpacked = torch.empty(2 * m, k // 2, dtype=torch.int8, device=npu_dev)
        torch.npu.synchronize()
        unpack_activation(rt, x, x_unpacked)
        torch.npu.synchronize()

        # Stage 2: group_matmul INT4  [2m, k] x [k, n] -> FP16 [2m, n] (rows interleaved)
        z = torch.zeros(2 * m, n, dtype=torch.float16, device=npu_dev)
        torch.npu.synchronize()
        group_matmul(rt, x_unpacked, weights_packed, deq_scales_fixpipe, counts,
                     0, group_num, n, k, z, weight_nz, transpose)
        torch.npu.synchronize()

        # Stage 3: msd_merge_dequant  [2m, n] -> BF16 [m, n]
        # Multi-expert in one launch: pass scale_biases (per-expert [n] list) + counts; the
        # kernel scans counts prefix sums per row to locate the owning expert. No host tiling.
        out = torch.zeros(m, n, dtype=torch.bfloat16, device=npu_dev)
        torch.npu.synchronize()
        msd_merge_dequant(rt, z, scale_biases, counts, pts_cat, out)
        torch.npu.synchronize()

        try:
            torch.testing.assert_close(result, out.to("cpu"), atol=1, rtol=1/128)
            print(f'group_matmul_dequant_int4 (int4, transpose={transpose}, nz={weight_nz}) passed!')
        except AssertionError as e:
            print(f'group_matmul_dequant_int4 (int4, transpose={transpose}, nz={weight_nz}) failed!')
            print(f'{e}')
            print(f'torch_npu: {result}, shape: {result.shape}')
            print(f'xlite: {out}, shape: {out.shape}')

