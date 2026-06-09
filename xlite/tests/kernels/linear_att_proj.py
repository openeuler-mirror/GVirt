#!/usr/bin/env/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import Runtime, linear_att_proj


def run_test(rt, M, N, V, H, K, rtol=1e-3, atol=1e-5):
    x = torch.rand(M, K)
    W_qkv = torch.rand(K, N)
    W_z = torch.rand(K, V)
    W_b = torch.rand(K, H)
    W_a = torch.rand(K, H)

    torch_qkv = torch.matmul(x, W_qkv)
    torch_z = torch.matmul(x, W_z)
    torch_b = torch.matmul(x, W_b)
    torch_a = torch.matmul(x, W_a)
    torch.npu.synchronize()

    mix_qkv = torch.empty(M, N)
    z = torch.empty(M, V)
    b = torch.empty(M, H)
    a = torch.empty(M, H)

    linear_att_proj(rt, x, W_qkv, W_z, W_b, W_a, mix_qkv, z, b, a,
                    M, N, V, H, K)

    torch.testing.assert_close(torch_qkv, mix_qkv, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch_z, z, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch_b, b, rtol=rtol, atol=atol)
    torch.testing.assert_close(torch_a, a, rtol=rtol, atol=atol)


def main():
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")

    print("=" * 80)
    print("Running tests for linear_att_projection")
    print("Tolerance: rtol=1e-3, atol=1e-5")
    print("=" * 80)

    for dtype in [torch.bfloat16, torch.float16]:
        torch.set_default_dtype(dtype)
        for M in [4096]:
            for N in [6144]:
                for V in [2048]:
                    for H in [16]:
                        for K in [1024]:
                            try:
                                run_test(rt, M, N, V, H, K)
                            except Exception as e:
                                print(f"FAILED for dtype={dtype}, M={M}, ",
                                      f"N={N}, V={V}, H={H}, K={K}: {e}")
                                raise e
                            print("-" * 60)

    print("=" * 80)
    print("All tests completed.")


if __name__ == "__main__":
    main()
