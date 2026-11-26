#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""xlite muls (tensor * scalar, optionally over a leading row region) vs torch_npu.

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_muls.py``.
"""

import sys

import pytest
import torch
import torch_npu  # noqa: F401

from xlite._C import muls

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
def test_muls(runtime, dtype):
    x = torch.randn(8, 2048, dtype=dtype)
    y = torch.empty(8, 2048, dtype=dtype)
    scale = 2.5

    standard = x * scale

    torch.npu.synchronize()
    muls(runtime, x, scale, y)
    torch.npu.synchronize()

    # Identical inputs into the same elementwise op: only a real kernel bug
    # can miss this tolerance.
    torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda x: str(x).removeprefix("torch."))
@pytest.mark.parametrize(
    "n,d,calc_num",
    [
        pytest.param(5120, 128, 64, id="n5120-d128-cn64"),
        pytest.param(5120, 192, 128, id="n5120-d192-cn128"),
        pytest.param(5120, 576, 512, id="n5120-d576-cn512"),
    ],
)
@pytest.mark.parametrize("seed", range(20), ids=lambda x: f"seed{x}")
def test_muls_multi_iteration(runtime, dtype, n, d, calc_num, seed):
    torch.manual_seed(seed)
    x = torch.randn(n, d, dtype=dtype)
    scale = (d**-0.5) * 2.0
    y = x.clone()

    torch.npu.synchronize()
    muls(runtime, y, scale, y, calc_offset=0, calc_num=calc_num)
    torch.npu.synchronize()

    standard = (x.float() * scale).to(dtype)
    torch.testing.assert_close(standard[:, :calc_num], y[:, :calc_num], atol=1e-5, rtol=1e-3)
    # Columns at/after calc_num are outside the calc region: the in-place
    # kernel must leave them bit-exact.
    torch.testing.assert_close(x[:, calc_num:], y[:, calc_num:], atol=0, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
