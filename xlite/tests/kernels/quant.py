#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import Runtime, quant

npu_devid = 0
rt = Runtime(npu_devid, 500)
torch.npu.set_device(npu_devid)

supported_dtype_list = [torch.bfloat16]

def quantize_cpu(
    x: torch.Tensor,
    input_scale_reciprocal: torch.Tensor,
    aclnn_input_offset: torch.Tensor,
) -> torch.Tensor:
    """
    公式: q = clamp(round(x * (1/scale) + offset), -128, 127)
    """
    x_fp32 = x.float()
    scale = input_scale_reciprocal.float()

    if scale.dim() == 1 and x.dim() > 1:
        # x: [batch, seq, hidden] 或 [batch, hidden]
        # scale: [hidden] -> [1, ..., 1, hidden]
        target_shape = [1] * (x.dim() - 1) + [-1]
        scale = scale.view(target_shape)
    
    if aclnn_input_offset is not None:
        offset = aclnn_input_offset.float()
        if offset.dim() == 1 and x.dim() > 1:
            offset = offset.view(target_shape)
    
    # round(x / scale + offset)
    x_quant = torch.round(x_fp32 * scale + offset)
    x_quant = torch.clamp(x_quant, -128, 127)
    return x_quant.to(torch.int8)

m = 8192
n = 2048
in_type = torch.bfloat16
out_type = torch.int8
x = torch.rand(m, n, dtype=in_type, device=f"npu:{npu_devid}")
z = torch.empty(m, n, dtype=out_type, device=f"npu:{npu_devid}")
input_scale_reciprocal = torch.randn(n, dtype=torch.bfloat16, device=f"npu:{npu_devid}")
input_offset = torch.randn(n, dtype=in_type, device=f"npu:{npu_devid}")

scale = input_scale_reciprocal.to(torch.float)
offset = input_offset.to(torch.float)

expected_z = quantize_cpu(x.to("cpu"), scale.to("cpu"), offset.to("cpu"))

torch.npu.synchronize()
quant(rt, x, scale, offset, z)
torch.npu.synchronize()

# torch.set_printoptions(threshold=1000000)

try:
    torch.testing.assert_close(expected_z, z.to("cpu"), atol=1, rtol=1/128)
except AssertionError as e:
    print(f'{e}')
    print(f'x: {x}, shape: {x.shape}')
    print(f'scale: {scale}')
    print(f'offset: {offset}')
    print(f'expected: {expected_z}, shape:{expected_z.shape}')
    print(f'xlite: {z}, shape: {z.shape}')
