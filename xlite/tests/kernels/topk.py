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
from xlite._C import Runtime, topk as xlite_topk
import numpy as np
import math

rt = Runtime(0, 500)
torch.npu.set_device(0)
torch.set_default_device("npu:0")
torch.set_default_dtype(torch.bfloat16)

n_tokens = 3000
K = 2048
n_routed_experts = 6144

scores = torch.randn(n_tokens, n_routed_experts)

values, torch_indices = scores.topk(K)

indices = torch.arange(n_routed_experts, dtype=torch.int32)
xlite_indices = torch.empty(n_tokens, K, dtype=torch.int32)

torch.npu.synchronize()
xlite_topk(rt, scores, indices, xlite_indices, K)
torch.npu.synchronize()

torch_indices = torch_indices.to(dtype=torch.int32)


# torch.set_printoptions(threshold=1000000)
for i in range(n_tokens):
    try:
        torch.testing.assert_close(torch_indices[i], xlite_indices[i])
    except Exception as e:
        print(f'row {i}: {e}')
print(torch_indices)
print(xlite_indices)

torch.testing.assert_close(torch_indices, xlite_indices)
