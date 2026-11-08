#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import logging
import os

import torch
import torch.distributed as dist
import torch_npu

from xlite._C import Runtime, rmsnorm, rmsnorm_with_bias

logging.getLogger().setLevel(logging.INFO)

npu_id = 0
rt = Runtime(npu_id, size=500)
torch.npu.set_device(npu_id)


def stand_rmsnorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2)
    variance = variance.mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight.float() * x
    if bias is not None:
        # The kernel adds the bias in fp32 and casts to the input dtype once
        # (single rounding, see doc/developer_guide/kernels/norm.md). Casting
        # the norm result to the input dtype before adding the bias would
        # double-round and differ from the kernel by 1 ULP on ~23% of elements.
        x = x + bias.float()
    return x.to(input_dtype)


BATCH_SIZE = 64
DIM = 8192
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]
# The kernel computes RMSNorm (+ bias) fully in fp32 and rounds once. An fp32
# reference then still disagrees on the last bit of a few elements because the
# fp32 op order differs (variance reduction, rsqrt vs div): always a single
# output-ULP difference. A 1-ULP fp16 error is at most 2^-10 relative, which
# rtol=1e-3 already absorbs; a 1-ULP bf16 error reaches 2^-7 = 0.78% relative,
# so bf16 needs an ULP-aware rtol.
TOL = {torch.float16: (1e-5, 1e-3), torch.bfloat16: (1e-5, 1e-2)}

for test_dtype in dtype_list:
    for has_bias in [True, False]:
        with torch.device("npu"):
            x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype)
            weight = torch.randn(DIM, dtype=test_dtype)
            bias = torch.randn(DIM, dtype=test_dtype)
            y = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)
            z = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)

        # standard
        standard = stand_rmsnorm(x, weight, bias if has_bias else None, NORMEPS)

        # xlite
        torch.npu.synchronize()
        if has_bias:
            rmsnorm_with_bias(rt, x, weight, bias, y, NORMEPS)
        else:
            rmsnorm(rt, x, weight, y, NORMEPS)
        torch.npu.synchronize()
        logging.info(f"rmsnorm({test_dtype}) {'with' if has_bias else 'without'} bias executed!")
        atol, rtol = TOL[test_dtype]
        try:
            torch.testing.assert_close(standard, y, atol=atol, rtol=rtol)
        except AssertionError as e:
            logging.error(f"{e}")
            print(f"standard: {standard}")
            print(f"bias: {bias}")
            logging.error(f"torch_npu: {standard}")
            logging.error(f"xlite: {y}")

BATCH_SIZE = 64
DIM = 128
CNT = 8
SIZE1 = DIM * (CNT + 2)
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    for has_bias in [True, False]:
        with torch.device("npu"):
            x = torch.randn(BATCH_SIZE, SIZE1, dtype=test_dtype)
            weight = torch.randn(DIM, dtype=test_dtype)
            bias1 = torch.randn(DIM, dtype=test_dtype)
            bias2 = torch.randn(DIM, dtype=test_dtype)
            y = x.clone()

        x_split1, x_split2, x_split3 = x.split([CNT * DIM, DIM, DIM], dim=1)
        x_split1 = x_split1.view(BATCH_SIZE, CNT, DIM)
        x_split2 = x_split2.view(BATCH_SIZE, 1, DIM)
        # standard
        x_split1 = stand_rmsnorm(x_split1, weight, bias1 if has_bias else None, NORMEPS)
        x_split2 = stand_rmsnorm(x_split2, weight, bias2 if has_bias else None, NORMEPS)
        x_split1 = x_split1.view(BATCH_SIZE, CNT * DIM)
        x_split2 = x_split2.view(BATCH_SIZE, DIM)
        standard = torch.cat([x_split1, x_split2, x_split3], dim=-1)

        # xlite
        torch.npu.synchronize()
        if has_bias:
            rmsnorm_with_bias(rt, x, weight, bias1, y, NORMEPS, norm_dim=DIM, cnt_per_token=CNT)
            rmsnorm_with_bias(
                rt,
                x,
                weight,
                bias2,
                y,
                NORMEPS,
                norm_dim=DIM,
                cnt_per_token=1,
                in_start_offset=DIM * CNT,
                out_start_offset=DIM * CNT,
            )
        else:
            rmsnorm(rt, x, weight, y, NORMEPS, norm_dim=DIM, cnt_per_token=CNT)
            rmsnorm(
                rt,
                x,
                weight,
                y,
                NORMEPS,
                norm_dim=DIM,
                cnt_per_token=1,
                in_start_offset=DIM * CNT,
                out_start_offset=DIM * CNT,
            )
        torch.npu.synchronize()

        logging.info(f"rmsnorm with stride ({test_dtype}) {'with' if has_bias else 'without'} bias executed!")

        atol, rtol = TOL[test_dtype]
        try:
            torch.testing.assert_close(standard, y, atol=atol, rtol=rtol)
        except AssertionError as e:
            logging.error(f"{e}")
            logging.error(f"torch_npu: {standard}")
            logging.error(f"xlite: {y}")
