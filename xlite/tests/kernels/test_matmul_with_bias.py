#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""xlite matmul_with_bias vs torch_npu F.linear with an fp32 bias.

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_matmul_with_bias.py``.
"""

import itertools
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from tests.models.weight_utils import nd2nz_or_skip
from xlite._C import matmul_with_bias

# Per-dtype comparison tolerances. torch_npu and xlite both accumulate in
# fp32 but split K differently (K up to 7616 here), so results differ by
# 1 output ULP (a bf16 ULP is up to 2^-8 relative) and by up to ~1e-3
# absolute at cancellation-heavy near-zero outputs — the same
# accumulation-order noise as test_matmul.py, whose fp64 check showed both
# sides equidistant from the true value. A real operand/layout bug produces
# O(1)-O(100) garbage and still fails hard.
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
    inputs = itertools.product([False, True], dtypes)
    for weight_nz, dtype in inputs:
        if dtype == torch.float and weight_nz:
            continue
        dtype_s = str(dtype).removeprefix("torch.")
        test_id = f"nz{int(weight_nz)}-{dtype_s}"
        yield pytest.param(weight_nz, dtype, id=test_id)


@pytest.mark.parametrize("k", KS, ids=lambda x: f"k{x}")
@pytest.mark.parametrize("n", NS, ids=lambda x: f"n{x}")
@pytest.mark.parametrize("m", MS, ids=lambda x: f"m{x}")
@pytest.mark.parametrize("weight_nz,dtype", list(_cases()))
def test_matmul_with_bias(runtime, weight_nz, dtype, m, n, k):
    x = torch.randn(m, k, dtype=dtype)
    y = torch.randn(n, k, dtype=dtype)
    z = torch.zeros(m, n, dtype=dtype)
    bias = torch.randn(n, dtype=torch.float)

    standard = F.linear(x, y, bias).to(dtype)

    # NZ weight layout: physical format 29, see conftest's allow_internal_format.
    if weight_nz:
        y = nd2nz_or_skip(y, f"[{m}, {k}] x [{n}, {k}] {dtype} weight_nz True")
        if y is None:
            pytest.skip("FRACTAL_NZ weight cast unavailable")

    torch.npu.synchronize()
    matmul_with_bias(runtime, x, y, z, bias, weight_nz=weight_nz)
    torch.npu.synchronize()

    atol, rtol = ATOL_RTOL[dtype]
    torch.testing.assert_close(standard, z, atol=atol, rtol=rtol)


# L1-residency regression guard (hard assert on purpose, unlike the loop
# above). Two failure modes need m > m0 and n > n0 so a core processes
# several output tiles and the L1 residency path actually engages — the
# loop above (m <= 8) never leaves single-tile-per-dim territory:
#   1. bias/deqScale residency must track the single l0BiasBuf / fixpipeBuf,
#      not the ping-pong l1bBuf state (distance-2 nOffset revisits);
#   2. l1BiasBuf / l1DeqScaleBuf alias the head of l1bBuf[0]: staging bias
#      through L1 overwrites buf0's resident B tile, so a later B residency
#      "hit" on buf0 reads the bias fp32 bytes as B fragments (~1e32 garbage
#      in a scatter of rows/cols, not an O(1) bias error).
# k <= 4 * k0 keeps one B load per tile so a core's tile subsequence forms
# the revisit pattern. Shapes picked by sweeping m/n on a kernel with the
# bugs reproduced (agent_log_scripts/sweep_bias_residency.py).
BIAS_RESIDENCY_SHAPES = [
    (640, 384, 256),
    (640, 512, 256),
    (1024, 256, 256),
    (2048, 1024, 256),
]


@pytest.mark.parametrize("m,n,k", BIAS_RESIDENCY_SHAPES, ids=lambda x: f"d{x}")
def test_matmul_with_bias_residency(runtime, m, n, k):
    x = torch.randn(m, k, dtype=torch.bfloat16)
    y = torch.randn(n, k, dtype=torch.bfloat16)
    z = torch.zeros(m, n, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.float)
    standard = F.linear(x, y, bias).to(torch.bfloat16)
    torch.npu.synchronize()
    matmul_with_bias(runtime, x, y, z, bias, weight_nz=False)
    torch.npu.synchronize()
    torch.testing.assert_close(standard, z, atol=1e-2, rtol=1.6e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
