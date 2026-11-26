#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""xlite matmul vs torch_npu F.linear over layout, dtype and shape matrices.

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_matmul.py`` or
``pytest tests/kernels/test_matmul.py``.
"""

import itertools
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from tests.models.weight_utils import nd2nz_or_skip
from xlite._C import matmul

# Per-dtype comparison tolerances. For k up to ~7.6k an element's exact result
# can land on an output-dtype rounding tie, which xlite and the torch_npu
# reference round in opposite directions (~1 output ulp apart), so allow ~2
# ulp relative error. The absolute floor covers fp32 accumulation-order noise
# on near-zero (cancelled) outputs, which is ~1e-3 for the largest k here.
ATOL_RTOL = {
    torch.float16: (1e-2, 2e-3),
    torch.bfloat16: (1e-2, 1.6e-2),
    torch.float: (4e-3, 1e-3),
}

MS = [1, 8]
NS = [32, 64, 16128]
KS = [32, 128, 7168, 7184, 7232, 7296, 7424, 7488, 7616]


def _cases():
    dtypes = [torch.float16, torch.bfloat16, torch.float]
    inputs = itertools.product([False, True], [False, True], dtypes)
    for weight_nz, transpose, dtype in inputs:
        # The fp32 path takes no NZ weights and no transpose: the
        # kernel would be called with weight_nz=False and no format
        # cast, so weight_nz=True would just duplicate that case.
        if dtype == torch.float and (weight_nz or transpose):
            continue
        dtype_s = str(dtype).removeprefix("torch.")
        test_id = f"nz{int(weight_nz)}-T{int(transpose)}-{dtype_s}"
        yield pytest.param(weight_nz, transpose, dtype, id=test_id)


@pytest.mark.parametrize("k", KS, ids=lambda x: f"k{x}")
@pytest.mark.parametrize("n", NS, ids=lambda x: f"n{x}")
@pytest.mark.parametrize("m", MS, ids=lambda x: f"m{x}")
@pytest.mark.parametrize("weight_nz,transpose,dtype", list(_cases()))
def test_matmul(runtime, weight_nz, transpose, dtype, m, n, k):
    x = torch.randn(m, k, dtype=dtype)
    if transpose:
        y = torch.randn(k, n, dtype=dtype)
        y_standard = y.transpose(0, 1).contiguous().reshape(n, k)
    else:
        y = torch.randn(n, k, dtype=dtype)
        y_standard = y
    z = torch.zeros(m, n, dtype=dtype)

    standard = F.linear(x, y_standard, None)

    # NZ weight layout: physical format 29, see conftest's allow_internal_format.
    if weight_nz:
        y = nd2nz_or_skip(y, f"[{m}, {k}] x [{n}, {k}] {dtype} weight_nz True transpose:{transpose}")
        if y is None:
            pytest.skip("FRACTAL_NZ weight cast unavailable")

    torch.npu.synchronize()
    matmul(runtime, x, y, z, weight_nz=weight_nz, transpose=transpose)
    torch.npu.synchronize()

    atol, rtol = ATOL_RTOL[dtype]
    torch.testing.assert_close(standard, z, atol=atol, rtol=rtol)


# Large-vocab NZ regression: the GM->L1 batched copy's srcGap is uint16 (32B
# blocks), so an NZ weight row pitch above 65535 blocks (lm_head vocab sizes
# 98304/151936, and 65552 which overflows only on the 16-wide tail tile) used
# to wrap around and silently load wrong rows.
@pytest.mark.parametrize("n", [65552, 98304, 151936], ids=lambda x: f"n{x}")
def test_matmul_large(runtime, n):
    m, k = 1, 4096
    x = torch.randn(m, k, dtype=torch.bfloat16)
    y = torch.randn(n, k, dtype=torch.bfloat16)
    z = torch.zeros(m, n, dtype=torch.bfloat16)

    standard = F.linear(x, y, None)

    y = nd2nz_or_skip(y, f"[{m}, {k}] x [{n}, {k}] large-n NZ")
    if y is None:
        pytest.skip("FRACTAL_NZ weight cast unavailable")
    torch.npu.synchronize()
    matmul(runtime, x, y, z, weight_nz=True, transpose=False)
    torch.npu.synchronize()

    # Own tolerances, not ATOL_RTOL: the error of a k=4096 dot product scales
    # with the accumulation magnitude (~sqrt(k)), not the output value, so
    # cancellation-prone outputs exceed the table's relative budget; ~1 abs is
    # normal bf16 noise here, while the overflow this guards against reads
    # entirely wrong rows and produces diffs of hundreds.
    torch.testing.assert_close(standard, z, atol=1.0, rtol=0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
