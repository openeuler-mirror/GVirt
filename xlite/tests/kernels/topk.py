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

n_tokens = 1
K = 2048
n_routed_experts = 6144

scores = torch.randn(n_tokens, n_routed_experts)

values, torch_indices = scores.topk(K)

indices = torch.arange(n_routed_experts, dtype=torch.int32)
xlite_indices = torch.empty(1, K, dtype=torch.int32)

print(scores.shape)
torch.npu.synchronize()
xlite_topk(rt, scores, indices, xlite_indices, K)
torch.npu.synchronize()

torch_indices = torch_indices.to(dtype=torch.int32)

print(xlite_indices)
print(torch_indices)
torch.testing.assert_close(torch_indices, xlite_indices)
