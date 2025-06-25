#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import print as xlite_print


torch.npu.set_device(0)

dims = [1, 3, 6, 7, 256, [1, 1], [3, 3], [6, 6], [7, 7], [256, 256], [1, 1, 1], [3, 3, 3], [6, 6, 6], [7, 7, 7], [10, 10, 10]]
for dim in dims:
    x = torch.randn(dim, dtype=torch.float, device="npu:0")
    print(x)
    xlite_print(x)