#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import
import logging
import torch
from xlite._C import Runtime, l2norm

logging.getLogger().setLevel(logging.INFO)

npu_id = 0
rt = Runtime(npu_id, 500)
torch.npu.set_device(npu_id)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("npu")


def standard_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)
    return x * inv_norm


def do_test(batch_size, dim):
    x = torch.randn(batch_size, dim)
    y = torch.empty(batch_size, dim)

    std_y = standard_l2norm(x)

    torch.npu.synchronize()
    l2norm(rt, x, y, 1e-6)
    torch.npu.synchronize()

    torch.testing.assert_close(y, std_y, atol=1e-4, rtol=1e-2)


def main():
    # do_test(batch_size=4, dim=5)
    for dtype in [torch.float16, torch.bfloat16]:
        torch.set_default_dtype(dtype)
        do_test(batch_size=64, dim=8192)
        do_test(batch_size=64, dim=128)
        print(f"l2norm({dtype}) executed")


if __name__ == "__main__":
    main()
