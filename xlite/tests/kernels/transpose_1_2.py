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
from xlite._C import Runtime, transpose_1_2

channels = 6144


def run_test(rt, batch, seq_len, msg):
    input = torch.randn(batch, seq_len, channels)
    output = torch.zeros(batch, channels, seq_len)
    standard = torch.zeros(batch, channels, seq_len)
    torch.npu.synchronize()
    standard = input.transpose(1, 2)
    torch.npu.synchronize()
    transpose_1_2(rt, input, output)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(standard, output)
        print(f"{msg}: PASS")
    except Exception as e:
        print(f"{msg}: FAILED")
        raise e


if __name__ == "__main__":
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")
    torch.set_printoptions(threshold=torch.inf)

    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        torch.set_default_dtype(dtype)
        for batch in range(1, 9):
            for i in range(13):
                seq_len = 2**i
                msg = f'[{dtype}/{batch}/{seq_len}]'
                run_test(rt, batch, seq_len, msg)
