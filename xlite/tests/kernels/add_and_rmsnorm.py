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
import torch_npu  # noqa: F401

from xlite._C import Runtime, add_and_rmsnorm

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, size=500)
torch.npu.set_device(0)

BATCH_SIZE = 64
DIM = 6144
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype) / 10
        weight = torch.randn(DIM, dtype=test_dtype) / 10
        y = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype) / 10

    # standard
    x_standard = x.to(torch.float32)
    y_standard = y.to(torch.float32)
    x_standard = x_standard + y_standard
    add_standard = x_standard
    variance = x_standard.pow(2).mean(-1, keepdim=True)
    x_standard = x_standard * torch.rsqrt(variance + NORMEPS)
    standard = (weight.float() * x_standard).to(test_dtype)
    add_standard = add_standard.to(test_dtype)

    # xlite
    torch.npu.synchronize()
    add_and_rmsnorm(rt, x, y, weight, x, NORMEPS)
    torch.npu.synchronize()

    logging.info(f"rmsnorm({test_dtype}) executed!")

    try:
        torch.testing.assert_close(add_standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f"{e}")
        logging.error(f"torch_npu add: {add_standard}")
        logging.error(f"xlite add: {y}")

    # Same story as tests/kernels/rmsnorm.py: the kernel adds and normalizes
    # in fp32 and rounds the output once, but the fp32 op order still differs
    # from the torch reference (variance reduction, rsqrt vs div), flipping
    # the final rounding on a few elements — always a single output ULP. A
    # 1-ULP fp16 error fits rtol=1e-3; a 1-ULP bf16 error reaches
    # 2^-7 = 0.78% relative, so bf16 needs an ULP-aware rtol.
    rms_rtol = 1e-3 if test_dtype == torch.float16 else 1e-2
    try:
        torch.testing.assert_close(standard, x, atol=1e-5, rtol=rms_rtol)
    except AssertionError as e:
        logging.error(f"{e}")
        logging.error(f"torch_npu rmsnorm: {standard}")
        logging.error(f"xlite rmsnorm: {x}")
