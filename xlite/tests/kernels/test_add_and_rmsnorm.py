#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""xlite fused residual-add + rmsnorm vs an fp32 torch_npu reference.

The residual sum is accumulated in place into the second input, and the
normalized result is written over the first input (out may alias in_).

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_add_and_rmsnorm.py``.
"""

import sys

import pytest
import torch
import torch_npu  # noqa: F401

from xlite._C import add_and_rmsnorm

DTYPES = [torch.float16, torch.bfloat16]
BATCH = 64
DIM = 6144
NORM_EPS = 1e-6

# The residual add is computed in fp32 and rounded once, so it matches the
# fp32 reference within rtol=1e-3 for both dtypes.
ADD_ATOL = 1e-5
ADD_RTOL = 1e-3

# Per-dtype rmsnorm tolerances. Same single-ULP story as
# tests/kernels/test_rmsnorm.py: the kernel adds and normalizes in fp32 and
# rounds the output once, but the fp32 op order still differs from the torch
# reference (variance reduction, rsqrt vs div), flipping the final rounding
# on a few elements. A 1-ULP fp16 error fits rtol=1e-3; a 1-ULP bf16 error
# reaches 2^-7 = 0.78% relative, so bf16 needs an ULP-aware rtol.
RMS_ATOL_RTOL = {
    torch.float16: (1e-5, 1e-3),
    torch.bfloat16: (1e-5, 1e-2),
}


def stand_add_and_rmsnorm(x, y, weight, eps=NORM_EPS):
    """fp32 reference: residual add, then RMSNorm of the accumulated sum."""
    input_dtype = x.dtype
    add = x.to(torch.float32) + y.to(torch.float32)
    variance = add.pow(2).mean(-1, keepdim=True)
    normed = add * torch.rsqrt(variance + eps)
    return add.to(input_dtype), (weight.float() * normed).to(input_dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
def test_add_and_rmsnorm(runtime, dtype):
    x = torch.randn(BATCH, DIM, dtype=dtype) / 10
    y = torch.randn(BATCH, DIM, dtype=dtype) / 10
    weight = torch.randn(DIM, dtype=dtype) / 10

    add_standard, standard = stand_add_and_rmsnorm(x, y, weight)

    torch.npu.synchronize()
    add_and_rmsnorm(runtime, x, y, weight, x, NORM_EPS)
    torch.npu.synchronize()

    torch.testing.assert_close(add_standard, y, atol=ADD_ATOL, rtol=ADD_RTOL)
    atol, rtol = RMS_ATOL_RTOL[dtype]
    torch.testing.assert_close(standard, x, atol=atol, rtol=rtol)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
