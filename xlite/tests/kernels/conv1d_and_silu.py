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
import torch.nn.functional as F
from xlite._C import Runtime, transpose_1_2, linear_att_conv_and_silu

channels = 6144
kernel_dim = 4


def my_impl(rt, input, weight, conv_state, output, batch, seq_len):
    mix_qkv = torch.empty(batch, channels, seq_len)
    torch.npu.synchronize()
    transpose_1_2(rt, input, mix_qkv)
    torch.npu.synchronize()
    out = torch.empty(batch, channels, seq_len)
    linear_att_conv_and_silu(rt, mix_qkv, conv_state, weight, out)
    torch.npu.synchronize()
    transpose_1_2(rt, out, output)
    torch.npu.synchronize()


def torch_impl(input, weight, conv_state, standard):
    input_with_state = torch.cat([conv_state, input.transpose(1, 2)], dim=-1)
    out = F.conv1d(input_with_state, weight, padding=0, groups=channels)
    torch.npu.synchronize()
    out = F.silu(out[:, :, -seq_len:])
    out = out.transpose(1, 2)
    torch.npu.synchronize()
    standard.copy_(out)
    torch.npu.synchronize()


def run_test(rt, batch, seq_len, msg):
    input = torch.randn(batch, seq_len, channels)
    weight = torch.randn(channels, 1, kernel_dim)
    conv_state = torch.randn(batch, channels, kernel_dim)
    output = torch.zeros(batch, seq_len, channels)
    standard = torch.zeros(batch, seq_len, channels)
    torch.npu.synchronize()
    torch_impl(input, weight, conv_state, standard)
    my_impl(rt, input, weight, conv_state, output, batch, seq_len)
    try:
        torch.testing.assert_close(standard, output, rtol=1e-2, atol=1e-3)
        print(f"{msg}: PASS")
    except Exception as e:
        print(f"{msg}: FAILED")
        raise e


if __name__ == "__main__":
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")
    torch.set_printoptions(threshold=torch.inf)

    for dtype in [torch.bfloat16, torch.float16]:
        torch.set_default_dtype(dtype)
        for batch in range(1, 9):
            for i in range(13):
                seq_len = 2**i
                msg = f'[{dtype}/{batch}/{seq_len}]'
                run_test(rt, batch, seq_len, msg)
