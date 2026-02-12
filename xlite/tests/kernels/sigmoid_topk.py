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
from xlite._C import Runtime, sigmoid_topk
import numpy as np
import math

rt = Runtime(0, 500)
torch.npu.set_device(0)

topK = 8
n_routed_experts = 160
n_tokens = 9
n_indices_map = math.ceil(n_routed_experts / 32)

for dtype in [torch.bfloat16, torch.float32]:
    scale = 2.5
    scores = torch.randn(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
    bias = torch.randn(n_routed_experts, dtype=dtype, device="npu:0")
    indices = torch.arange(n_routed_experts, dtype=torch.int32, device="npu:0")
    xlite_indices_out = torch.empty(n_tokens, n_indices_map, dtype=torch.int32, device="npu:0")
    xlite_weights_out = torch.empty(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")

    standard_sigmoid = scores.sigmoid() + bias
    standard_weights, standard_indices = torch.topk(standard_sigmoid, topK, dim=-1)
    standard_weights = standard_weights / standard_weights.sum(dim=-1, keepdim=True)
    standard_weights = standard_weights * scale

    torch.npu.synchronize()
    sigmoid_topk(rt, scores, indices, bias, scale, xlite_weights_out, xlite_indices_out, topK, True)
    torch.npu.synchronize()
    print(f'sigmoid topK (dtype={dtype}) executed!')

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

    # Get sigmoid values, selected by returned indices.
    standard_scores = torch.gather(standard_sigmoid, dim=1, index=standard_indices)
    xlite_scores = torch.gather(standard_sigmoid, dim=1, index=xlite_indices)
    xlite_scores, _ = torch.sort(xlite_scores, descending=True)

    # Convert Xlite weights from sparse format to a sorted list
    mask = torch.ne(torch.zeros(xlite_weights_out.shape, device="npu:0"), xlite_weights_out)
    xlite_weights = torch.masked_select(xlite_weights_out, mask).reshape(standard_weights.shape)
    xlite_weights, _ =torch.sort(xlite_weights,descending=True)

    try:
        # Compare scores and weights. Score values may become very close after applying sigmoid
        # function. As long as both algorithms select elements with close sigmoid values it does
        # not matter which ones are choosen.
        torch.testing.assert_close(standard_scores, xlite_scores, atol=1e-4, rtol=1e-2)
        torch.testing.assert_close(standard_weights, xlite_weights, atol=1e-4, rtol=1e-2)
    except AssertionError as e:
        print(f'{e}')
        print(f'standard_values:\n{standard_scores}')
        print(f'xlite_values:\n{xlite_scores}')
        print(f'standard_indices:\n{standard_indices}')
        print(f'xlite_indices:\n{xlite_indices}')
        print(f'standard_weights:\n{standard_weights}')
        print(f'xlite_weights:\n{xlite_weights}')
