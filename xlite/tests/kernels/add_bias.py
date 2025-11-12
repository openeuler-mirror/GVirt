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
from xlite._C import Runtime, add_bias

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

REAL_M = 1024
HIDDEN_SIZE = 7168
dtype_list = [torch.float32, torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        z = torch.zeros(REAL_M, HIDDEN_SIZE, dtype=test_dtype)
        bias = torch.randn(HIDDEN_SIZE, dtype=test_dtype)

    # standard
    standard = z + bias

    # xlite
    torch.npu.synchronize()
    add_bias(rt, z, bias, z)
    torch.npu.synchronize()

    logging.info(f'add bias ({test_dtype}) executed!')

    try:
        torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {standard}')
        logging.error(f'xlite: {z}')
