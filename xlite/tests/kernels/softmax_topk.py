#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import sys
import torch
from xlite._C import Runtime, softmax_topk
import numpy as np

rt = Runtime(0, 500)
torch.npu.set_device(0)

topK = 8
n_tokens = 8

# SelectTopK pads the candidate count up to 4 * SORT_BLOCK_SIZE (128) and always runs
# vmrgsort4's fixed 4-way merge, so n_routed_experts below 128 exercises the padded path
# (InitSortTail + PadCalcTail). 64 is the critical case: CalcSoftmax() runs only
# DIV_ROUND_UP(64, VECTOR_MAX_NUM_OF_FP32=64) = 1 repeat, i.e. it writes lanes [0,64) and
# never touches the [64,128) padding, so those lanes MUST be re-pinned to FLOAT_MIN per
# token or vmrgsort4 merges uninitialised garbage into the top-K. (96 would cover its own
# padding -- DIV_ROUND_UP(96,64) = 2 repeats writes all 128 lanes -- but it is not
# expressible here: the routing bitmap is carried as int64, while the kernel strides GM by
# n_routed_experts/32 uint32 = 12 bytes per token. See sqrtsoftplus_hash_topk.py, which
# uses an int32 carrier and therefore does cover 96.)
expert_counts = [64, 128]

failed = []
for n_routed_experts in expert_counts:
    # BIT1 routing bitmap, carried on the torch side as int64: one bit per expert,
    # 64 experts per word. The kernel strides GM by n_routed_experts/32 uint32 and DMAs
    # exactly n_routed_experts/8 bytes per token, which is a whole number of int64 words
    # precisely when n_routed_experts is a multiple of 64 -- hence expert_counts above.
    n_indices_map = int(n_routed_experts / 64)

    for dtype in [torch.bfloat16, torch.float32]:
        scores = torch.randn(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
        indices = torch.arange(n_routed_experts, dtype=torch.int32, device="npu:0")
        # Zero-initialised on purpose: the bitmap DMA covers only n_routed_experts/8 bytes,
        # so trailing bytes of the last word are never written by the kernel.
        xlite_indices_out = torch.zeros(n_tokens, n_indices_map, dtype=torch.int64, device="npu:0")
        xlite_weights_out = torch.zeros(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")

        standard_indices_out = torch.zeros(n_tokens, n_indices_map, dtype=torch.int64, device="npu:0")
        standard_weights_out = torch.zeros(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
        standard_softmax = scores.softmax(dim=-1, dtype=dtype)
        standard_weights, standard_indices = torch.topk(standard_softmax, topK, dim=-1)
        standard_weights = standard_weights / standard_weights.sum(dim=-1, keepdim=True)

        for i in range(n_tokens):
            for j in range(topK):
                elem = np.int64(standard_indices[i][j].cpu())
                idx = int(elem / 64)
                value = np.uint64(standard_indices_out[i][idx].cpu()) | np.uint64(1 << (elem % 64))
                standard_indices_out[i][idx] = value.view(np.int64)
                standard_weights_out[i][elem] = standard_weights[i][j]

        torch.npu.synchronize()
        softmax_topk(rt, scores, indices, xlite_weights_out, xlite_indices_out, topK, True)
        torch.npu.synchronize()
        tag = f'n_routed_experts={n_routed_experts}, dtype={dtype}'
        print(f'softmax topK ({tag}) executed!')

        # The kernel computes softmax entirely in fp32 and only rounds at the output DMA,
        # whereas the reference above rounds through `dtype` at every step. For bf16 that
        # leaves ~2 ulp of legitimate divergence on normalised weights (~0.125 each), so a
        # single 1e-5 tolerance would flag pre-existing rounding rather than a real defect.
        w_atol = 1e-5 if dtype == torch.float32 else 4e-3
        try:
            torch.testing.assert_close(standard_indices_out, xlite_indices_out, atol=0, rtol=0)
            torch.testing.assert_close(standard_weights_out, xlite_weights_out, atol=w_atol, rtol=1e-3)
        except AssertionError as e:
            print(f'FAIL ({tag})')
            print(f'{e}')
            print(f'standard_indices_out: {standard_indices_out}')
            print(f'xlite_indices_out: {xlite_indices_out}')
            print(f'standard_weights_out: {standard_weights_out}')
            print(f'xlite_weights_out: {xlite_weights_out}')
            failed.append(tag)

if failed:
    print(f'FAIL: {len(failed)} softmax_topk config(s) mismatched:')
    for tag in failed:
        print(f'  - {tag}')
    sys.exit(1)
print(f'PASS: all {len(expert_counts) * 2} softmax_topk configs matched')
