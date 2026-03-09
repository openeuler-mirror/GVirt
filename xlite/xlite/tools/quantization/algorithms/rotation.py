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
import os
import utils

# The implementation of the Rotation Quantization was based on Quarot: https://github.com/spcl/QuaRot

curr_file = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_file)
hadamard_matrices_dir = os.path.join(curr_dir, 'hadamard_matrices')


def read_hadamard_matrix(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        matrix = [[1 if char == '+' else -1 for char in row] for row in content.split('\n') if row]
    return torch.tensor(matrix, dtype=torch.float)


# From http://www.neilsloane.com/hadamard/index.html
hadamard_172 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.172.will.txt'))
hadamard_156 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.156.will.txt'))
hadamard_140 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.140.pal.txt'))
hadamard_108 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.108.pal.txt'))
hadamard_60 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.60.pal.txt'))
hadamard_52 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.52.will.txt'))
hadamard_40 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.40.tpal.txt'))
hadamard_36 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.36.pal2.txt'))
hadamard_28 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.28.pal2.txt'))
hadamard_20 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.20.pal.txt'))
hadamard_12 = read_hadamard_matrix(os.path.join(hadamard_matrices_dir, 'had.12.txt'))


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def generate_hadamard(n):
    hadamard, K = None, None
    if n % 172 == 0:
        K = 172
        hadamard = hadamard_172
    elif n % 156 == 0 and (is_pow2(n // 156)):
        K = 156
        hadamard = hadamard_156
    elif n % 140 == 0 and (is_pow2(n // 140)):
        K = 140
        hadamard = hadamard_140
    elif n % 108 == 0 and (is_pow2(n // 108)):
        K = 108
        hadamard = hadamard_108
    elif n % 60 == 0 and (is_pow2(n // 60)):
        K = 60
        hadamard = hadamard_60
    elif n % 52 == 0 and (is_pow2(n // 52)):
        K = 52
        hadamard = hadamard_52
    elif n % 36 == 0 and (is_pow2(n // 36)):
        K = 36
        hadamard = hadamard_36
    elif n % 28 == 0 and (is_pow2(n // 28)):
        K = 28
        hadamard = hadamard_28
    elif n % 40 == 0 and (is_pow2(n // 40)):
        K = 40
        hadamard = hadamard_40
    elif n % 20 == 0 and (is_pow2(n // 20)):
        K = 20
        hadamard = hadamard_20
    elif n % 12 == 0 and (is_pow2(n // 12)):
        K = 12
        hadamard = hadamard_12
    else:
        assert (is_pow2(n)), "Dimension not compatible for generating Hadamard matrix yet!"
        K = 1

    return hadamard, K


def matmul_hadU(X):
    n = X.shape[-1]
    hadamard, K = generate_hadamard(n)
    input_ = X.clone().view(-1, n, 1)
    output_ = input_.clone()
    while input_.shape[1] > K:
        input_ = input_.view(input_.shape[0], input_.shape[1] // 2, 2, input_.shape[2])
        output_ = output_.view(input_.shape)
        output_[:, :, 0, :] = input_[:, :, 0, :] + input_[:, :, 1, :]
        output_[:, :, 1, :] = input_[:, :, 0, :] - input_[:, :, 1, :]
        output_ = output_.view(input_.shape[0], input_.shape[1], -1)
        (input_, output_) = (output_, input_)
    del output_

    if K > 1:
        input_ = hadamard.view(1, K, K).to(input_) @ input_

    return input_.view(X.shape) / torch.tensor(n).sqrt()


def random_hadamard_matrix(size, device):
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def random_orthogonal_matrix(size, device, dtype=None):
    utils.cleanup_memory(device)
    random_matrix = torch.randn(size, size, dtype=dtype if dtype is not None else torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, device, mode="hadamard"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        try:
            return random_hadamard_matrix(size, device)
        except Exception:
            print(f"Hadamard rotation matrix doesn't exist for size {size}. Use random rotation instead")
            return random_orthogonal_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")
