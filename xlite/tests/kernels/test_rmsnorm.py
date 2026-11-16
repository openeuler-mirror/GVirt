#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""xlite rmsnorm (with optional bias) vs an fp32 torch_npu reference.

Covers whole-row normalization ([batch, hidden]) and the segmented layout
(norm_dim x cnt_per_token slices at in/out column offsets, applied in place).

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_rmsnorm.py``.
"""

import sys

import pytest
import torch
import torch_npu  # noqa: F401

from xlite._C import rmsnorm, rmsnorm_with_bias

DTYPES = [torch.float16, torch.bfloat16]
BATCH = 64
NORM_EPS = 1e-6

# Per-dtype comparison tolerances. The kernel computes RMSNorm (+ bias) fully
# in fp32 and rounds once, but the fp32 op order still differs from the
# reference (variance reduction, rsqrt vs div), flipping the final rounding
# on a few elements — always a single output ULP. A 1-ULP fp16 error is at
# most 2^-10 relative, which rtol=1e-3 already absorbs; a 1-ULP bf16 error
# reaches 2^-7 = 0.78% relative, so bf16 needs an ULP-aware rtol.
ATOL_RTOL = {
    torch.float16: (1e-5, 1e-3),
    torch.bfloat16: (1e-5, 1e-2),
}


def stand_rmsnorm(x, weight, bias=None, eps=NORM_EPS):
    """fp32 reference honoring the kernel's single-rounding bias contract."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight.float() * x
    if bias is not None:
        # The kernel adds the bias in fp32 and casts to the input dtype once
        # (single rounding, see doc/developer_guide/kernels/norm.md). Casting
        # the norm result to the input dtype before adding the bias would
        # double-round and differ from the kernel by 1 ULP on ~23% of elements.
        x = x + bias.float()
    return x.to(input_dtype)


def run_xlite_rmsnorm(runtime, x, weight, bias, y, **tail):
    """Run the xlite RMSNorm binding, dispatching on the optional bias."""
    if bias is None:
        rmsnorm(runtime, x, weight, y, NORM_EPS, **tail)
    else:
        rmsnorm_with_bias(runtime, x, weight, bias, y, NORM_EPS, **tail)


@pytest.mark.parametrize("has_bias", [False, True], ids=lambda b: "bias" if b else "no_bias")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
def test_rmsnorm(runtime, dtype, has_bias):
    hidden = 8192
    x = torch.randn(BATCH, hidden, dtype=dtype)
    weight = torch.randn(hidden, dtype=dtype)
    bias = torch.randn(hidden, dtype=dtype) if has_bias else None
    y = torch.empty(BATCH, hidden, dtype=dtype)

    standard = stand_rmsnorm(x, weight, bias)

    torch.npu.synchronize()
    run_xlite_rmsnorm(runtime, x, weight, bias, y)
    torch.npu.synchronize()

    atol, rtol = ATOL_RTOL[dtype]
    torch.testing.assert_close(standard, y, atol=atol, rtol=rtol)


# Strided-case geometry for test_rmsnorm_strided; module scope because the
# parametrize below evaluates at collection time.
SEG_DIM = 128
SEG_CNT = 8


@pytest.mark.parametrize(
    "cnt,offset",
    [
        pytest.param(SEG_CNT, 0, id=f"cnt{SEG_CNT}-at0"),
        pytest.param(1, SEG_CNT * SEG_DIM, id=f"cnt1-at{SEG_CNT * SEG_DIM}"),
    ],
)
@pytest.mark.parametrize("has_bias", [False, True], ids=lambda b: "bias" if b else "no_bias")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
def test_rmsnorm_strided(runtime, dtype, has_bias, cnt, offset):
    # Row of SEG_CNT + 2 segments: cnt of them (SEG_DIM wide) are normalized
    # in place at the column offset; every other column — including the
    # trailing segment — must stay bit-exact.
    x = torch.randn(BATCH, SEG_DIM * (SEG_CNT + 2), dtype=dtype)
    weight = torch.randn(SEG_DIM, dtype=dtype)
    bias = torch.randn(SEG_DIM, dtype=dtype) if has_bias else None
    y = x.clone()

    span = cnt * SEG_DIM
    chunk = x[:, offset : offset + span].reshape(-1, SEG_DIM)
    standard = stand_rmsnorm(chunk, weight, bias).reshape(-1, span)

    torch.npu.synchronize()
    run_xlite_rmsnorm(
        runtime,
        x,
        weight,
        bias,
        y,
        norm_dim=SEG_DIM,
        cnt_per_token=cnt,
        in_start_offset=offset,
        out_start_offset=offset,
    )
    torch.npu.synchronize()

    atol, rtol = ATOL_RTOL[dtype]
    torch.testing.assert_close(standard, y[:, offset : offset + span], atol=atol, rtol=rtol)
    torch.testing.assert_close(x[:, :offset], y[:, :offset], atol=0, rtol=0)
    torch.testing.assert_close(x[:, offset + span :], y[:, offset + span :], atol=0, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
