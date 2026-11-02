#!/usr/bin/python3
# coding=utf-8 -*-
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""quant_q_c8 query quantization and fused scaled-weights tests."""
import logging
import math

import torch


QUANT_MAX = 127
Q8_FILL = -101
SW_FILL = -17

npu_id = 0

# Row dims exercise calcRepeat/quantRepeat combinations (64: 1x1, 128: 2x1,
# 192: 3x2); row counts cross the 32-block launch stride.
dims = [64, 128, 192]
test_cases = [0, 1, 17, 129, 257]
# 1/sqrt(128) is the V4 rotate_activation factor; 1.0 is pure quantization.
rotate_scales = [1.0, 1.0 / math.sqrt(128.0)]


def check_values(q8, scaled_weights, rows, head_weight, rotate_scale, name):
    # Quantization uses the FP32 absmax of the un-rotated row; the rotate
    # factor is folded into scaled_weights only. Match the kernel's fp32
    # op order (c/127 rounded once, then two fp32 multiplies) so the fp16
    # fused weights compare exactly.
    rows_f = rows.float()
    amax = rows_f.abs().amax(-1)
    c = torch.tensor(rotate_scale, dtype=torch.float32)
    scale = amax * (c / torch.tensor(float(QUANT_MAX), dtype=torch.float32))
    expected_sw = (head_weight.float() * scale).half()
    torch.testing.assert_close(scaled_weights, expected_sw, rtol=0, atol=0)

    quant_scale = amax * (1.0 / QUANT_MAX)
    normalized = rows_f * torch.where(quant_scale == 0, 0, quant_scale.reciprocal())[:, None]
    expected_q8 = normalized.round().to(torch.int8)
    torch.testing.assert_close(q8, expected_q8, atol=1, rtol=1 / 128)
    assert bool(((q8 >= -QUANT_MAX) & (q8 <= QUANT_MAX)).all()), name


def run_test(rt, device, name, rows, head_weight, rotate_scale, out_offset=0):
    from xlite._C import quant_q_c8

    tokens, dim = rows.shape
    q8_base = torch.full((tokens * dim + 2 * out_offset,), Q8_FILL, dtype=torch.int8,
                         device=device)
    sw_base = torch.full((tokens + 2 * out_offset,), SW_FILL, dtype=torch.float16, device=device)
    q8 = q8_base[out_offset:out_offset + tokens * dim].view(tokens, dim)
    scaled_weights = sw_base[out_offset:out_offset + tokens]
    device_rows = rows.to(device)
    device_hw = head_weight.to(device)
    torch.npu.synchronize(device)
    quant_q_c8(rt, device_rows, device_hw, q8, scaled_weights, rotate_scale)
    torch.npu.synchronize(device)
    assert torch.equal(device_rows.cpu(), rows), (name, "input mutated")
    assert torch.equal(device_hw.cpu(), head_weight), (name, "head_weight mutated")

    q8_base, sw_base = q8_base.cpu(), sw_base.cpu()
    q8 = q8_base[out_offset:out_offset + tokens * dim].view(tokens, dim)
    scaled_weights = sw_base[out_offset:out_offset + tokens]
    if tokens:
        check_values(q8, scaled_weights, rows, head_weight, rotate_scale, name)
    if out_offset:
        assert bool((q8_base[[0, -1]] == Q8_FILL).all()), (name, "q8 guard")
        assert bool((sw_base[[0, -1]] == SW_FILL).all()), (name, "sw guard")
    logging.info("quant_q_c8 %s (%d rows, dim %d, rotate %s) passed", name, tokens, dim,
                 rotate_scale)


def main():
    import torch_npu  # noqa: F401
    from xlite._C import Runtime

    logging.getLogger().setLevel(logging.INFO)
    torch.npu.set_device(npu_id)
    device = torch.device(f"npu:{npu_id}")
    rt = Runtime(device.index, 32)
    generator = torch.Generator().manual_seed(9128)
    for dim in dims:
        for tokens in test_cases:
            rows = torch.randn(tokens, dim, generator=generator).bfloat16()
            head_weight = torch.randn(tokens, generator=generator).bfloat16()
            for rotate_scale in rotate_scales:
                run_test(rt, device, f"random-dim{dim}-{tokens}", rows, head_weight,
                         rotate_scale)

    # Per-row edge cases: zero row (zeros + zero fused weight), constant row,
    # and a single large outlier dominating one row's absmax. head_weight edge
    # cases: zero, negative and large values (fused weight may be negative).
    dim = 128
    rows = torch.randn(17, dim, generator=generator).bfloat16()
    rows[3] = 0
    rows[5] = 4
    rows[7, 0] = 1024
    head_weight = torch.randn(17, generator=generator).bfloat16()
    head_weight[2] = 0
    head_weight[4] = -8.5
    head_weight[9] = 100
    for rotate_scale in rotate_scales:
        run_test(rt, device, "edge-rows", rows, head_weight, rotate_scale)

    # Guard both ends of the output views (the 2-byte fused-weight store takes
    # the b16 path; q8 rows stay 32B-aligned because dim is a multiple of 64).
    run_test(rt, device, "offset-views", rows, head_weight, 1.0 / math.sqrt(128.0),
             out_offset=1)

    logging.info("quant_q_c8 tests passed")


if __name__ == "__main__":
    main()
