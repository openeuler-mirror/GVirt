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
from xlite._C import Runtime, einsum_mht_hdt_mhd, einsum_mht_htd_mhd
from tests.models.weight_utils import matrix_nd2nz


rt = Runtime(0, 500)
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
    einsum_op(rt, mht, y, mhd, m, h, t, d, weight_nz)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(standard, mhd, atol=1e-5, rtol=1e-3)
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
