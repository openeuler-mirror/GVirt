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
from xlite._C import Runtime, concat_col


def run_case(rt, dtype, shape_rows, widths):
    """inputs: list of [*(rows), w_i]; out: [*(rows), sum(widths)]."""
    torch.set_default_dtype(dtype)
    inputs = [torch.randn(*shape_rows, w) for w in widths]
    out = torch.empty(*shape_rows, sum(widths))
    torch.npu.synchronize()
    concat_col(rt, inputs, out)
    torch.npu.synchronize()
    ref = torch.cat(inputs, dim=-1)
    # pure byte copy -> must be exactly equal
    assert torch.equal(out, ref), f"concat_col MISMATCH dtype={dtype} rows={shape_rows} widths={widths}"
    print(f"OK dtype={dtype} rows={shape_rows} widths={widths}")


if __name__ == "__main__":
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")

    for dtype in [torch.bfloat16, torch.float16]:
        # 3D, 均匀列宽
        run_case(rt, dtype, (8, 4096), [2048, 2048, 2048])
        # 3D, 非均匀且非 32B 对齐列宽 (bf16/f16=2B, 奇数宽度产生任意字节偏移)
        run_case(rt, dtype, (4, 128), [100, 7, 333, 1, 42])
        # 2D
        run_case(rt, dtype, (1024,), [512, 256, 768])
        # numPackets=1 (单行)
        run_case(rt, dtype, (1,), [64, 128, 32])
        # 单输入
        run_case(rt, dtype, (256,), [1024])
        # 超宽单列 (> UB 半缓冲, 触发分块路径): bf16 96KB=49152 elem
        run_case(rt, dtype, (2,), [60000, 1000])

    print("concat_col ALL PASS")
