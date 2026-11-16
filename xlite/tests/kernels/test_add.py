#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""xlite add vs torch_npu elementwise add.

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_add.py``.
"""

import sys

import pytest
import torch
import torch_npu  # noqa: F401

from xlite._C import add

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("seed", range(20), ids=lambda x: f"seed{x}")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda x: str(x).removeprefix("torch."))
def test_add(runtime, dtype, seed):
    torch.manual_seed(seed)
    x = torch.randn(8, 2048, dtype=dtype)
    y = torch.randn(8, 2048, dtype=dtype)
    z = torch.empty(8, 2048, dtype=dtype)

    standard = x + y

    torch.npu.synchronize()
    add(runtime, x, y, z)
    torch.npu.synchronize()

    # Identical inputs into the same elementwise op: only a real kernel bug
    # (wrong operand, wrong addressing) can miss this tolerance.
    torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
