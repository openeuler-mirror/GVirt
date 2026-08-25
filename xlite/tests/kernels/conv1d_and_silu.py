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
import time
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
    expected_state = torch.cat([conv_state, input.transpose(1,2)], dim=-1)[..., -kernel_dim:]
    torch.npu.synchronize()
    my_impl(rt, input, weight, conv_state, output, batch, seq_len)
    torch.npu.synchronize()
    try:
        torch.testing.assert_close(standard, output, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(expected_state, conv_state, rtol=1e-2, atol=1e-3)
        print(f"{msg}: PASS")
    except Exception as e:
        print(f"{msg}: FAILED")
        raise e


def run_packed_mixed(rt, lens, msg):
    batch = len(lens)
    total = sum(lens)
    inp = torch.randn(total, channels)
    weight = torch.randn(channels, 1, kernel_dim)
    conv_state = torch.randn(batch, channels, kernel_dim)
    output = torch.zeros(total, channels)
    starts = [0]
    for seq_len in lens[:-1]:
        starts.append(starts[-1] + seq_len)
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device="npu:0")
    query_lens = torch.tensor(lens, dtype=torch.int32, device="npu:0")
    # 32B GM pad so kernel block copies of int32 meta stay in-bounds.
    if query_start_loc.numel() < 8:
        pad = 8 - query_start_loc.numel()
        query_start_loc = torch.cat(
            [query_start_loc, torch.zeros(pad, dtype=torch.int32, device="npu:0")]
        )
        query_lens = torch.cat(
            [query_lens, torch.zeros(pad, dtype=torch.int32, device="npu:0")]
        )

    ref_parts = []
    expected_state = conv_state.clone()
    for i, seq_len in enumerate(lens):
        start = starts[i]
        x = inp[start : start + seq_len]  # [S, C]
        cat = torch.cat([expected_state[i : i + 1], x.transpose(0, 1).unsqueeze(0)], dim=-1)
        out = F.silu(F.conv1d(cat, weight, padding=0, groups=channels)[:, :, -seq_len:])
        ref_parts.append(out.squeeze(0).transpose(0, 1))
        expected_state[i] = cat[0, :, -kernel_dim:]
    standard = torch.cat(ref_parts, dim=0)

    torch.npu.synchronize()
    linear_att_conv_and_silu(
        rt, inp, conv_state, weight, output, query_start_loc, query_lens
    )
    torch.npu.synchronize()
    torch.testing.assert_close(standard, output, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(expected_state, conv_state, rtol=1e-2, atol=1e-3)
    print(f"{msg}: PASS")


if __name__ == "__main__":
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")
    torch.set_printoptions(threshold=torch.inf)

    t = time.time()
    for dtype in [torch.bfloat16, torch.float16]:
        torch.set_default_dtype(dtype)
        for batch in range(1, 9):
            for i in range(13):
                seq_len = 2**i
                msg = f'[{dtype}/{batch}/{seq_len}]'
                run_test(rt, batch, seq_len, msg)
        run_packed_mixed(rt, [1, 4, 2], f"[{dtype}/packed-mixed=[1,4,2]]")
        run_packed_mixed(rt, [8, 1], f"[{dtype}/packed-mixed=[8,1]]")
    total = time.time() - t
    print(f'completed in {total} s.')
