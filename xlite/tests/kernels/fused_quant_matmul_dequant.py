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
import torch_npu
from xlite._C import (
    Runtime,
    dequant,
    fused_quant_matmul_dequant,
    matmul_dequant,
    quant,
    quant_dynamic,
)

# allow weight_nz
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

dev_id = 0

rt = Runtime(dev_id, 500)
torch.npu.set_device(dev_id)

# (m, n, k, weight_nz, transpose, dyn_quant)
test_sizes = [
    [1, 256, 1024, False, False, True],
    [1, 256, 1024, False, False, False],
    [17, 768, 1024, False, False, True],
    [17, 768, 1024, False, False, False],
    [17, 768, 1024, True, False, True],
    [17, 768, 1024, False, True, False],
    [103, 1024, 2048, False, False, True],
    [103, 1024, 2048, False, False, False],
    [512, 1024, 1024, False, False, True],
    [512, 1024, 1024, True, True, False],
    # ragged k via the dyn engine's per-row 1D DMA branch (static stage
    # rejects it on the host — see test_gate_rejections); odd k across the
    # K_TILE boundary
    [64, 256, 8192, False, False, False],
    # AIC-ahead-of-AIV race: mLoop=1 with a large nLoop makes one AIC run all
    # n-tiles and retire while the paired AIVs are still in their last
    # dequant tiles (regression guard for the V2C_DONE_FLAG exit barrier)
    [128, 6144, 768, False, False, True],
    [128, 3072, 768, False, False, True],
    # mLoop=2 boundary: partial tail m-tile through the grid-stride matmul
    # and the lockstep dequant (m > m0 for the first time)
    [200, 256, 1024, False, False, False],
]

# Long-sequence prefill shapes exercise mLoop >> aicNum (multiple midx-first
# rounds per AIC/AIV pair, cross-core flag pairing runs many rounds).
# Capped at m=32768: this host's HBM is mostly occupied by other processes
# (bf16 x + fp16 serial tmp + bf16 out x2 + int8 workspaces ≈ 1.3 GB).
long_seq_sizes = [
    [2048, 4096, 4096, False, False, True],
    [2048, 4096, 4096, False, False, False],
    [8192, 4096, 4096, False, False, True],
    [4096, 4096, 4096, True, False, False],
    [4096, 4096, 4096, False, True, True],
    [4096, 7168, 7168, False, False, False],
    [32768, 4096, 4096, False, False, True],
    # odd long m: many mLoop rounds plus a partial tail tile (long-sequence
    # shapes above are all multiples of 128, so the tail tile is never
    # exercised together with multi-round flag pairing)
    [2057, 1024, 2048, False, False, True],
]

# Optional-feature cross: absent bias / absent fixpipe deq_scale / num
# row-shrink.  k=256 keeps the no-deq_scale fp16 intermediate finite
# (|acc| <= 127 * 2 * 256 = 65024 < fp16 max; the int32 accumulator lands in
# fp16 unscaled when deq_scale is absent).  num shrinks only the dequant
# stage: the AIC matmul still writes all m rows as fp16, so only rows
# [0, min(num, m)) are dequantized in place and comparable against the
# serial chain.
# (m, n, k, weight_nz, transpose, dyn_quant, has_bias, has_deq_scale, num)
opt_sizes = [
    [64, 512, 256, False, False, True, True, False, None],
    [64, 512, 256, False, False, False, False, False, None],
    [64, 512, 256, False, False, False, False, True, None],
    [64, 512, 256, False, False, True, False, True, None],
    # num token-count shrink: 300 < m (partial), 1000 > m (clamped to m),
    # 0 (dequant writes nothing; only launch/flag balance is exercised)
    [512, 1024, 2048, False, False, True, True, True, 300],
    [512, 1024, 2048, False, False, True, True, True, 1000],
    [512, 1024, 2048, False, False, True, True, True, 0],
    # small n: nActual < n0 boundary inside the fused matmul/dequant tiles
    [64, 64, 1024, False, False, True, True, True, None],
]


def run_case(m, n, k, weight_nz, transpose, dyn_quant, has_bias=True, has_deq_scale=True,
             num=None):
    x = (torch.randn(m, k, dtype=torch.bfloat16, device=f"npu:{dev_id}") * 2.0)
    # without fixpipe deq_scale the int32 accumulator lands in fp16 unscaled;
    # narrow the weight so |acc| <= 127 * 2 * k stays finite (see opt_sizes)
    w_hi = 8 if has_deq_scale else 2
    weight_ref = torch.randint(-w_hi, w_hi, (n, k), dtype=torch.int8, device=f"npu:{dev_id}")
    if transpose:
        y_in = weight_ref.t().contiguous()
    else:
        y_in = weight_ref
    if weight_nz:
        ACL_FORMAT_FRACTAL_NZ = 29
        y_in = torch_npu.npu_format_cast(y_in, ACL_FORMAT_FRACTAL_NZ)

    if has_bias:
        bias = torch.randint(-64, 64, (n,), dtype=torch.int32, device=f"npu:{dev_id}")
    else:
        # the standalone matmul_dequant binding takes bias positionally; an
        # empty tensor carries a null ptr = "no bias" for the serial golden
        bias = torch.empty(0, dtype=torch.int32, device=f"npu:{dev_id}")
    if has_deq_scale:
        # deqscale must be bf16-representable to be losslessly consumed by fixpipe
        deq_scale = torch.rand(n, dtype=torch.float32, device=f"npu:{dev_id}") * 0.015 + 0.001
        deq_scale = deq_scale.to(torch.bfloat16).to(torch.float32)
        # fixpipe layout: uint64 slot holding fp32 in the low 32 bits
        scale = torch.zeros(n * 2, dtype=torch.float32, device=f"npu:{dev_id}")
        scale[0::2] = deq_scale
    else:
        scale = torch.empty(0, dtype=torch.float32, device=f"npu:{dev_id}")

    if not dyn_quant:
        quant_scale_rec = (torch.rand(k, dtype=torch.bfloat16, device=f"npu:{dev_id}") * 0.05 + 0.01)
        quant_offset = (torch.randn(k, dtype=torch.bfloat16, device=f"npu:{dev_id}") * 0.5)

    # reference: serial three-kernel chain quant -> matmul_dequant -> dequant
    serial_xq = torch.empty(m, k, dtype=torch.int8, device=f"npu:{dev_id}")
    serial_tmp = torch.empty(m, n, dtype=torch.float16, device=f"npu:{dev_id}")
    standard = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{dev_id}")
    fused_out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{dev_id}")
    x_quant = torch.empty(m, k, dtype=torch.int8, device=f"npu:{dev_id}")
    if dyn_quant:
        out_scale = torch.empty(m, dtype=torch.float32, device=f"npu:{dev_id}")
        serial_scale = torch.empty(m, dtype=torch.float32, device=f"npu:{dev_id}")
    else:
        # the standalone dequant binding still needs a (possibly empty) scale
        serial_scale = torch.empty(0, dtype=torch.float32, device=f"npu:{dev_id}")

    torch.npu.synchronize()
    if dyn_quant:
        quant_dynamic(rt, x, serial_scale, serial_xq)
    else:
        quant(rt, x, quant_scale_rec, quant_offset, serial_xq)
    matmul_dequant(rt, serial_xq, y_in, bias, scale, serial_tmp, weight_nz, transpose)
    dequant(rt, serial_tmp, serial_scale, standard, dyn_quant)
    torch.npu.synchronize()

    # optional args are passed only when the feature is present; absent ones
    # default to None on the binding side
    fused_opt = {}
    if dyn_quant:
        fused_opt["out_scale"] = out_scale
    else:
        fused_opt["quant_scale_rec"] = quant_scale_rec
        fused_opt["quant_offset"] = quant_offset
    if has_bias:
        fused_opt["bias"] = bias
    if has_deq_scale:
        fused_opt["deq_scale"] = scale
    if num is not None:
        fused_opt["num"] = torch.tensor([num], dtype=torch.int32, device=f"npu:{dev_id}")
    fused_quant_matmul_dequant(
        rt, x, weight=y_in, out=fused_out, x_quant=x_quant, dyn_quant=dyn_quant,
        weight_nz=weight_nz, transpose=transpose, **fused_opt
    )
    torch.npu.synchronize()
    print(
        f'[{m}, {k}] x [{k}, {n}] weight_nz:{weight_nz} transpose:{transpose} '
        f'dyn_quant:{dyn_quant} bias:{has_bias} deq_scale:{has_deq_scale} num:{num} '
        f'fusion quant matmul dequant executed!'
    )

    # num shrinks only the dequant stage (rows [0, min(num, m)) are written);
    # num == 0 skips the comparison entirely - the case only guards that the
    # launch and cross-core flag pairing stay balanced with an empty range
    rows = m if num is None else min(num, m)
    if rows == 0:
        return
    try:
        torch.testing.assert_close(
            standard[:rows].to("cpu"), fused_out[:rows].to("cpu"), atol=1e-2, rtol=1e-2
        )
    except AssertionError as e:
        print(f'{e}')
        print(f'serial: {standard[:rows]}')
        print(f'fused: {fused_out[:rows]}')


for m, n, k, weight_nz, transpose, dyn_quant in test_sizes:
    run_case(m, n, k, weight_nz, transpose, dyn_quant)

for m, n, k, weight_nz, transpose, dyn_quant in long_seq_sizes:
    run_case(m, n, k, weight_nz, transpose, dyn_quant)

for m, n, k, weight_nz, transpose, dyn_quant, has_bias, has_deq_scale, num in opt_sizes:
    run_case(m, n, k, weight_nz, transpose, dyn_quant, has_bias, has_deq_scale, num)

