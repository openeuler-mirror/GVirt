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
    fusion_operator_matmul_dequant_pipeline,
    matmul_dequant,
)

# allow weight_nz
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

dev_id = 0

rt = Runtime(dev_id, 500)
torch.npu.set_device(dev_id)

test_sizes = [
    [1, 256, 1024],
    [17, 768, 1024],
    [103, 1024, 2048],
]

for weight_nz in [True, False]:
    for transpose in [True, False]:
        for has_out_scale in [True, False]:
            for m, n, k in test_sizes:
                x = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=f"npu:{dev_id}")
                weight_ref = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=f"npu:{dev_id}")
                if transpose:
                    y_in = weight_ref.t().contiguous()
                else:
                    y_in = weight_ref
                if weight_nz:
                    ACL_FORMAT_FRACTAL_NZ = 29
                    y_in = torch_npu.npu_format_cast(y_in, ACL_FORMAT_FRACTAL_NZ)

                bias = torch.randint(-64, 64, (n,), dtype=torch.int32, device=f"npu:{dev_id}")
                # deqscale 从 bf16 转换来的 fp32 可被 fixpipe 无损消费
                deq_scale = torch.rand(n, dtype=torch.float32, device=f"npu:{dev_id}") * 0.015 + 0.001
                deq_scale = deq_scale.to(torch.bfloat16).to(torch.float32)
                # fixpipe硬件要求：以uint64_t存储fp32，高位为0，低位为fp32格式的二进制值
                scale = torch.zeros(n * 2, dtype=torch.float32, device=f"npu:{dev_id}")
                scale[0::2] = deq_scale

                if has_out_scale:
                    out_scale = torch.rand(m, dtype=torch.float32, device=f"npu:{dev_id}") * 0.25 + 0.75
                else:
                    out_scale = torch.empty(0, dtype=torch.float32, device=f"npu:{dev_id}")
                num = torch.empty(0, dtype=torch.int32, device=f"npu:{dev_id}")

                # reference: 串行路径 matmul_dequant(FP16) + dequant(BF16)
                serial_tmp = torch.empty(m, n, dtype=torch.float16, device=f"npu:{dev_id}")
                standard = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{dev_id}")
                fused_out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{dev_id}")

                torch.npu.synchronize()
                matmul_dequant(rt, x, y_in, bias, scale, serial_tmp, weight_nz, transpose)
                dequant(rt, serial_tmp, out_scale, standard, has_out_scale)
                torch.npu.synchronize()

                fusion_operator_matmul_dequant_pipeline(
                    rt, x, y_in, fused_out, bias, scale, weight_nz, transpose, out_scale, num
                )
                torch.npu.synchronize()
                print(
                    f'[{m}, {k}] x [{k}, {n}] weight_nz:{weight_nz} transpose:{transpose} '
                    f'has_out_scale:{has_out_scale} fusion matmul dequant executed!'
                )

                try:
                    torch.testing.assert_close(
                        standard.to("cpu"), fused_out.to("cpu"), atol=1e-2, rtol=1e-2
                    )
                except AssertionError as e:
                    print(f'{e}')
                    print(f'serial: {standard}')
                    print(f'fused: {fused_out}')
