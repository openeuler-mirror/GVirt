#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import itertools
import multiprocessing as mp
import json
import random
import time
import torch
import argparse
import tqdm
from xlite._C import Runtime, matmul
from tests.models.weight_utils import matrix_nd2nz


weight_nz = True
transpose = False
dtype = torch.bfloat16
torch.npu.config.allow_internal_format = True

repeat = 100_000
US_IN_S = 100_000


def validate_globals():
    if transpose and dtype == torch.float:
        raise ValueError('Transposed float inputs are not supported')


def process_args():
    device_count = torch.npu.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_devices', type=int, default=device_count,
                        help='number of NPU devices to use (all by default)')

    args = parser.parse_args()
    if args.num_devices <= 0 or args.num_devices > device_count:
        msg = (f'num_devices must be between 1 and {device_count},'
               'got {args.num_devices}')
        raise ValueError(msg)

    return args


def make_inputs(m, n, k):
    x = torch.randn(m, k, dtype=dtype)
    if transpose:
        y = torch.randn(k, n, dtype=dtype)
    else:
        y = torch.randn(n, k, dtype=dtype)
    z = torch.zeros(m, n, dtype=dtype)

    if weight_nz and dtype != torch.float:
        y = matrix_nd2nz(y)

    return (x, y, z)


def run_matmul(rt, x, y, z):
    start = time.time()
    for _ in range(repeat):
        torch.npu.synchronize()
        matmul(rt, x, y, z, weight_nz and dtype != torch.float, transpose)
        torch.npu.synchronize()
    end = time.time()

    avg = (end - start) / repeat * US_IN_S
    return avg


def run_test(m, n, k, swizzle, device):
    torch.set_default_device(f'npu:{device}')
    rt = Runtime(device, 500)
    rt.configure_swizzle(swizzle, False)
    torch.npu.set_device(device)
    x, y, z = make_inputs(m, n, k)
    time = run_matmul(rt, x, y, z)

    result = {
        'swizzle': hex(swizzle),
        'm': m,
        'n': n,
        'k': k,
        'time': time
    }

    return result


def process_job(x):
    swizzle, (m, n, k) = x
    name = mp.current_process().name
    device = int(''.join(filter(str.isnumeric, name))) - 1

    t = run_test(m, n, k, swizzle, device)
    return t


def worker_init():
    # Warm-up run
    x = (0x100, (32, 2048, 2048))
    process_job(x)


def main():
    args = process_args()
    validate_globals()

    swizzles = [0x700, 0x500, 0x600, 0x400, 0x900, 0x800, 0xE00, 0xB00, 0x301,
                0x1400, 0x1200, 0x101, 0xF00, 0xD01, 0xA00, 0x601, 0x401,
                0x201, 0x1F00, 0x1C00, 0x1B00, 0x1100, 0x1000, 0xD00, 0xC00,
                0x1900, 0x1700, 0x1600, 0x1500, 0x1300]
    shapes = [(32, 6144, 2048), (32, 2048, 6144),
              (16, 6144, 2048), (16, 2048, 6144),
              (8, 6144, 2048), (8, 2048, 6144),
              (4, 6144, 2048), (4, 2048, 6144),
              (1, 6144, 2048), (1, 2048, 6144)]
    random.shuffle(swizzles)
    random.shuffle(shapes)

    total = len(swizzles) * len(shapes)

    jobs = itertools.product(swizzles, shapes)

    with mp.Pool(processes=args.num_devices, initializer=worker_init) as pool:
        res = list(tqdm.tqdm(pool.imap(process_job, jobs), total=total))
        with open('swizzle_perf.json', 'w') as f:
            json.dump(res, f)

    return 0


if __name__ == "__main__":
    main()
