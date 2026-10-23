#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import random
import sys
from xlite._C import Runtime, sigmoid_topk
import numpy as np
import math

rt = Runtime(0, 500)
torch.npu.set_device(0)

n_tokens = 9
n_total = 0
n_fail = 0

models = [
    # (n_routed_experts, n_group, topk_group, topK, has_bias)
    (160, 1, 1, 8, True),
    (256, 8, 4, 8, True),
    (256, 1, 1, 8, True),
    # GLM-4.7-Flash: 64 routed experts, no expert groups, top-4, sigmoid scoring and no
    # e_score_correction_bias. Here nScores == 64 < 4 * SORT_BLOCK_SIZE, which used to underflow
    # the merge-sort tail length and make vmrgsort4 read/write past its UB scratch buffers, and
    # the missing bias made the kernel DMA from a null GM address.
    (64, 1, 1, 4, True),
    (64, 1, 1, 4, False),
    # 96 routed experts, nGroup=1: a non-64-multiple below 128. CalcGroupScores' vadd spans
    # DIV_ROUND_UP(96, 64) = 2 repeats = 128 lanes and rewrites the [96,128) sort padding every
    # token, so SelectTopK's PadScoresTail() must re-pin those lanes to FLOAT_MIN after the
    # V-pipe writes (fenced with PIPE_ALL) or vmrgsort4 can merge garbage into the top-K. The
    # int32 bitmap carrier supports non-64-multiple counts (unlike softmax_topk's int64 carrier).
    (96, 1, 1, 8, True),
]

for dtype in [torch.bfloat16, torch.float32]:
    for n_routed_experts, n_group, topk_group, topK, has_bias in models:
        n_indices_map = math.ceil(n_routed_experts / 32)
        scale = 2.5
        scores = torch.randn(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
        if has_bias:
            bias = torch.randn(n_routed_experts, dtype=torch.float32, device="npu:0")
        else:
            # Empty tensor -> XTensor.ptr == nullptr -> device skips the bias DMA and the add.
            bias = torch.empty(0, dtype=torch.float32, device="npu:0")
        indices = torch.arange(n_routed_experts, dtype=torch.int32, device="npu:0")
        xlite_indices_out = torch.empty(n_tokens, n_indices_map, dtype=torch.int32, device="npu:0")
        xlite_weights_out = torch.empty(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")

        # The kernel converts bf16 scores to fp32 (vconv_bf162f32) and computes the whole
        # sigmoid / bias-add / select / normalize chain in fp32, only rounding to bf16 on the
        # way out (vconv_f322bf16r). Computing the reference sigmoid in bf16 instead introduces
        # a ~1 ULP (0.004 near 0.5) systematic offset, which is enough to flip the ranking of
        # two experts sitting on the topK boundary -- a spurious mismatch that shows up only for
        # bf16 and only on some random draws. Match the kernel: fp32 throughout, cast at the end.
        scores_sigmoid = scores.float().sigmoid()
        standard_sigmoid = (scores_sigmoid + bias) if has_bias else scores_sigmoid
        group_scores = (
            standard_sigmoid.view(-1, n_group, n_routed_experts // n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, n_routed_experts // n_group)
            .reshape(-1, n_routed_experts)
        )
        standard_sigmoid = standard_sigmoid.masked_fill(~score_mask.bool(), 0.0)
        standard_indices = torch.topk(standard_sigmoid, k=topK, dim=-1, sorted=False)[1]
        standard_indices, _ = torch.sort(standard_indices, descending=False)
        standard_weights = scores_sigmoid.gather(1, standard_indices)
        standard_weights = standard_weights / standard_weights.sum(dim=-1, keepdim=True)
        standard_weights = standard_weights * scale
        standard_weights = standard_weights.to(dtype)

        torch.npu.synchronize()
        sigmoid_topk(rt, scores, indices, bias, scale, xlite_weights_out, xlite_indices_out, n_group, topk_group, topK, True)
        torch.npu.synchronize()
        cfg = (f'n_routed_experts={n_routed_experts}, n_group={n_group}, '
               f'topk_group={topk_group}, topK={topK}, has_bias={has_bias}, dtype={dtype}')
        print(f'sigmoid topK ({cfg}) executed!')

        # Convert Xlite outputs from bitmaps to lists of indices
        xlite_indices = torch.zeros(*standard_indices.shape, dtype=torch.int32, device="npu:0")

        for i in range(n_tokens):
            bitmap = 0
            for j in reversed(range(xlite_indices_out.shape[1])):
                bitmap = (bitmap << 32) | int(np.uint32(xlite_indices_out[i][j].cpu()))

            for pos in range(n_routed_experts):
                if bitmap & (1 << pos):
                    xlite_indices[i][j] = pos
                    j += 1

        # Convert Xlite weights from sparse format to a sorted list
        mask = torch.ne(torch.zeros(xlite_weights_out.shape, device="npu:0"), xlite_weights_out)
        xlite_weights = torch.masked_select(xlite_weights_out, mask).reshape(standard_weights.shape)

        n_total += 1
        try:
            # Compare scores and weights. Score values may become very close after applying sigmoid
            # function. As long as both algorithms select elements with close sigmoid values it does
            # not matter which ones are choosen.
            torch.testing.assert_close(standard_indices.to(dtype=torch.int32), xlite_indices, atol=1e-4, rtol=1e-2)
            torch.testing.assert_close(standard_weights, xlite_weights, atol=1e-4, rtol=1e-2)
        except AssertionError as e:
            n_fail += 1
            print(f'[FAIL] {cfg}')
            print(f'{e}')
            print(f'standard_indices:\n{standard_indices}')
            print(f'xlite_indices:\n{xlite_indices}')
            print(f'standard_weights:\n{standard_weights}')
            print(f'xlite_weights:\n{xlite_weights}')
        else:
            print(f'[PASS] {cfg}')

if n_fail:
    print(f'FAIL: {n_fail} sigmoid_topK config(s) mismatched')
    sys.exit(1)
print(f'PASS: all {n_total} sigmoid_topK configs matched')
