#!/usr/bin/env/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import Runtime, split_col


if __name__ == "__main__":
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")
    torch.set_printoptions(threshold=torch.inf)

    for dtype in [torch.bfloat16, torch.float16]:
        torch.set_default_dtype(dtype)
        x = torch.randn(8, 4096, 6144)
        q = torch.empty(8, 4096, 2048)
        k = torch.empty(8, 4096, 2048)
        v = torch.empty(8, 4096, 2048)
        torch.npu.synchronize()
        split_col(rt, x, [q, k, v])
        torch.npu.synchronize()
        [trch_q, trch_k, trch_v] = torch.split(x, 2048, dim=2)
        torch.testing.assert_close(q, trch_q)
        torch.testing.assert_close(k, trch_k)
        torch.testing.assert_close(v, trch_v)
