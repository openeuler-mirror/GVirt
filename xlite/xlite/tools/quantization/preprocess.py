#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from xlite.tools.quantization.algorithms.rotation import (
    get_orthogonal_matrix,
)
import torch
import transformers
import math


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def main(model_path, num_tp, save_path):
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    Q = get_orthogonal_matrix(
        config.hidden_size,
        mode="hadamard",
        device="cpu").to(torch.float32)
    dim_left, dim_right = get_decompose_dim(config.moe_intermediate_size // num_tp)
    kron_left = get_orthogonal_matrix(
        dim_left,
        mode="hadamard",
        device="cpu").to(torch.float32)
    kron_right = get_orthogonal_matrix(
        dim_right,
        mode="hadamard",
        device="cpu").to(torch.float32)

    rotate_dict = {}
    rotate_dict["Q"] = Q
    rotate_dict["Q_kron"] = [kron_left, kron_right]
    torch.save(rotate_dict, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_tp", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./rotate_dict.bin")
    args = parser.parse_args()
    main(model_path=args.model_path, num_tp=args.num_tp, save_path=args.save_path)
