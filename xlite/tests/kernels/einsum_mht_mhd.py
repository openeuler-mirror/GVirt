#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, einsum_mht_hdt_mhd, einsum_mht_htd_mhd
from tests.models.weight_utils import matrix_nd2nz


rt = Runtime(0, size=500)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True

# (m, h, t, d)
test_cases = [
    (3, 4, 192, 512),
    (3, 4, 512, 192),
    (3, 4, 512, 256),
    (3, 4, 256, 512),
    (3, 4, 128, 512),
    (3, 4, 512, 128),
    (5230, 4, 192, 512),
    (5230, 4, 512, 192),
    (5230, 4, 512, 256),
    (5230, 4, 256, 512),
    (5230, 4, 128, 512),
    (5230, 4, 512, 128),
    (963, 8, 512, 128),
]


def run_case(einsum_op, weight_layout, dtype, m, h, t, d, weight_nz):
    mht = torch.randn(m, h, t, dtype=dtype, device="npu:0")
    htd = torch.randn(h, t, d, dtype=dtype, device="npu:0")
    mhd = torch.zeros(m, h, d, dtype=dtype, device="npu:0")

    standard = torch.einsum("mht,htd->mhd", mht, htd)

    if weight_layout == "htd":
        # kernel consumes right operand as [h, t, d], transpose path
        y = htd.contiguous()
    else:
        # weight_layout == "hdt": kernel consumes [h, d, t], non-transpose path
        y = htd.transpose(-1, -2).contiguous()

    if weight_nz:
        y = matrix_nd2nz(y)

    torch.npu.synchronize()
    einsum_op(rt, mht, y, mhd, m, h, t, d, weight_nz=weight_nz)

    # torch.einsum and the kernel both accumulate in fp32 but in different
    # orders (the kernel splits K into 64-wide Mmad chunks), so a small
    # fraction of elements straddles a bf16 rounding midpoint and the two
    # bf16 outputs land on adjacent values: every observed mismatch is
    # exactly 1 bf16 ULP (abs diff a power of two, rel diff in
    # [2^-8, 2^-7]). atol=1e-5/rtol=1e-3 sits below the bf16 output
    # resolution and cannot hold for two independently rounded fp32->bf16
    # GEMMs. (1e-3, 1e-2) accepts any 1-ULP flip (ulp/|y| <= 2^-7 < 1e-2)
    # while a real kernel bug (wrong tile / garbage) still fails loudly.
    # fp16 keeps the tight bound: its 1-ULP relative error 2^-10 < 1e-3.
    if dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    else:
        atol, rtol = 1e-5, 1e-3
    try:
        torch.testing.assert_close(standard, mhd, atol=atol, rtol=rtol)
    except AssertionError as e:
        print(f'[m={m}, h={h}, t={t}, d={d}, {dtype}, layout={weight_layout}, '
              f'weight_nz={weight_nz}] {e}')
        print(f'torch npu: {standard}')
        print(f'xlite: {mhd}')
        return False
    print(f'[m={m}, h={h}, t={t}, d={d}, {dtype}, layout={weight_layout}, '
          f'weight_nz={weight_nz}] einsum executed!')
    return True


if __name__ == "__main__":
    failed = False
    for dtype in [torch.float16, torch.bfloat16]:
        for weight_nz in [False, True]:
            for m, h, t, d in test_cases:
                ok = run_case(einsum_mht_htd_mhd, "htd", dtype, m, h, t, d, weight_nz)
                failed = failed or not ok
                ok = run_case(einsum_mht_hdt_mhd, "hdt", dtype, m, h, t, d, weight_nz)
                failed = failed or not ok
    if failed:
        raise SystemExit(1)
