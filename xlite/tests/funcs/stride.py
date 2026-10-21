#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os
import tempfile

import torch
from xlite._C import Runtime, all_reduce, matmul
from xlite._C import print as xlite_print


torch.npu.set_device(0)


def printed_values(x):
    """Print x via xlite and parse the values back in logical (row-major) order.

    row/col are enlarged so Print does not elide the middle of any dimension.
    xlite writes from C++, so stdout is captured at the file-descriptor level.
    """
    rows = max(1, x.numel() // x.shape[-1])
    cols = x.shape[-1]
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved = os.dup(1)
        os.dup2(tmp.fileno(), 1)
        try:
            xlite_print(x, "", 2 * rows, 2 * cols)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        tmp.seek(0)
        out = tmp.read()
    body = out[out.index("XTensor(") + len("XTensor("):out.index(", shape=")]
    return [float(t) for t in body.replace("[", " ").replace("]", " ").split(",") if t.strip()]


def check_print(x, expected, name, atol=0.0, rtol=0.0):
    got = printed_values(x)
    assert len(got) == x.numel(), f"{name}: printed {len(got)} values, expected {x.numel()}"
    expected_vals = torch.tensor(expected.cpu().tolist(), dtype=torch.float64)
    torch.testing.assert_close(
        torch.tensor(got, dtype=torch.float64), expected_vals, atol=atol, rtol=rtol
    )
    print(f"PASS: {name} ({tuple(x.shape)}, stride={tuple(x.stride())})")


def check_rejected(x, name):
    try:
        xlite_print(x)
    except ValueError as e:
        assert "stride" in str(e), f"{name}: unexpected error message: {e}"
        print(f"PASS: {name} rejected at wrap time ({e})")
        return
    raise AssertionError(f"{name}: expected a stride ValueError but printing succeeded")


def check_empty_accepted(x, name):
    """Zero-element tensors wrap as contiguous regardless of stride: they carry no data,
    and torch's canonical empty strides (size-0 dims clamped) never match the row-major
    layout, so rejecting them would make the affected bindings unusable."""
    assert x.numel() == 0
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved = os.dup(1)
        os.dup2(tmp.fileno(), 1)
        try:
            xlite_print(x)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        tmp.seek(0)
        out = tmp.read()
    assert "empty tensor" in out, f"{name}: unexpected print output: {out!r}"
    print(f"PASS: {name} accepted at wrap time (stride={tuple(x.stride())})")


def check_contiguous_guard(x, name):
    """Non-cache operations must refuse non-contiguous torch tensors at wrap time."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    try:
        all_reduce(rt, out, x)
    except ValueError as e:
        assert "contiguous" in str(e), f"{name}: unexpected error message: {e}"
        print(f"PASS: {name} rejected by the contiguous guard ({e})")
        return
    raise AssertionError(f"{name}: expected a contiguity ValueError but the call succeeded")


# Contiguous tensors must keep printing their logical values unchanged.
x = torch.arange(24, dtype=torch.float32, device="npu:0").reshape(3, 8)
check_print(x, x.flatten(), "contiguous 2D")

# torch calls this contiguous (size-1 dim); a size-1 dim's stride never affects element
# addressing, so the wrap derives the contiguous stride and prints the logical values.
x = torch.arange(12, dtype=torch.float32, device="npu:0").reshape(1, 4, 3).transpose(0, 1)
assert x.is_contiguous()
check_print(x, x.flatten(), "torch-contiguous size-1-dim 3D")

# Dim-0 strided (row pitch) layouts print their logical values (debug output accepts
# pitches; ops other than the kv-cache paths do not, see the guard tests below).
x = torch.arange(12, dtype=torch.float32, device="npu:0")[::3]
check_print(x, x.flatten(), "1D dim-0 strided")

x = torch.arange(48, dtype=torch.float32, device="npu:0").reshape(6, 8)[::2]
check_print(x, x.flatten(), "2D dim-0 strided")

x = torch.arange(60, dtype=torch.float32, device="npu:0").reshape(4, 3, 5)[::2]
check_print(x, x.flatten(), "3D dim-0 strided")

x = torch.randn(6, 8, dtype=torch.float32, device="npu:0")[::2]
check_print(x, x.flatten(), "2D dim-0 strided fp32", atol=1e-3, rtol=1e-3)

# 2D last-dim split of a fused buffer is a dim-0 pitch layout: rows stay contiguous.
fused = torch.randn(4, 16, dtype=torch.float32, device="npu:0")
check_print(fused[:, :8], fused[:, :8].flatten(), "2D fused-buffer split", atol=1e-3, rtol=1e-3)

# Overlapping dim-0 layouts (torch expand, stride 0) are rejected: rows must not overlap.
x = torch.randn(1, 8, dtype=torch.float32, device="npu:0").expand(4, 8)
check_rejected(x, "2D broadcast expand (stride 0)")

# Strides mismatching the contiguous layout in dims >= 1 are rejected at wrap time.
check_rejected(torch.randn(4, 8, dtype=torch.float32, device="npu:0").t(), "transposed 2D")
check_rejected(
    torch.randn(4, 16, dtype=torch.float32, device="npu:0")[:, ::2], "inner-dim sliced 2D"
)

# Zero-element tensors carry no data, so any stride is accepted at wrap time (torch's
# canonical strides for size-0 dims, e.g. (5, 0) -> (1, 1), never match the row-major
# layout, and .contiguous() cannot produce one that does).
for empty in (
    torch.empty(0, 5, dtype=torch.float32, device="npu:0"),
    torch.empty(5, 0, dtype=torch.float32, device="npu:0"),
    torch.empty(2, 3, 0, 4, dtype=torch.float32, device="npu:0"),
    torch.empty(0, 5, dtype=torch.float32, device="npu:0").t(),
):
    check_empty_accepted(empty, f"empty {tuple(empty.shape)}")

# The guard tests below are rejected at wrap time, before any kernel launch, so the
# runtime needs no tensor pool.
rt = Runtime(0)

# Non-cache operations enforce contiguity at the bridge: a pitched tensor must be
# .contiguous()'d by the caller instead of being misread as packed rows.
src = torch.arange(48, dtype=torch.float32, device="npu:0").reshape(6, 8)
check_contiguous_guard(src[::2], "pitched all_reduce input")
check_contiguous_guard(
    torch.randn(4, 8, dtype=torch.float32, device="npu:0").t(), "transposed all_reduce input"
)

# The matmul binding rejects a pitched input at the bridge as well; it previously kept
# the pitched stride at wrap time and only rejected the tensor later, inside the op, at
# the layout-restricted View guard.
x = torch.arange(48, device="npu:0").to(torch.int32).reshape(6, 8)[::2]
y = torch.arange(64, device="npu:0").to(torch.int32).reshape(8, 8)
z = torch.empty(3, 8, dtype=torch.int32, device="npu:0")
try:
    matmul(rt, x, y, z)
except ValueError as e:
    assert "contiguous" in str(e), f"unexpected error: {e}"
    print(f"PASS: pitched int32 rejected by the matmul bridge guard ({e})")
else:
    raise AssertionError("matmul: expected ValueError for pitched int32 input")

print("All XTensor stride tests passed!")
