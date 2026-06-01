#!/usr/bin/env python3
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
import torch
import argparse
import tqdm
from xlite._C import Runtime, matmul_bench
from tests.models.weight_utils import matrix_nd2nz


weight_nz = True
transpose = False
dtype = torch.bfloat16
torch.npu.config.allow_internal_format = True

WARMUP_RUNS = 2
BENCH_RUNS = 1
RUNS_PER_INPUT = 8


def validate_globals():
    if transpose and dtype == torch.float:
        raise ValueError('Transposed float inputs are not supported')


def process_args():
    device_count = torch.npu.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_devices', type=int, default=device_count,
                        help='number of NPU devices to use (all by default)')
    parser.add_argument('--swizzle', type=lambda x: int(x, 0), default=0x100,
                        help='swizzle')

    args = parser.parse_args()
    if args.num_devices <= 0 or args.num_devices > device_count:
        msg = (f'num_devices must be between 1 and {device_count},'
               f'got {args.num_devices}')
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


def run_matmul(rt, real_args, warmup_args, device):
    torch.npu.synchronize()
    nz = weight_nz and dtype != torch.float
    runs_args = (BENCH_RUNS, WARMUP_RUNS)
    t = matmul_bench(rt, *real_args, *warmup_args, *runs_args, nz, transpose)
    torch.npu.synchronize()

    return t


def run_test(m, n, k, swizzle, device):
    torch.set_default_device(f'npu:{device}')
    rt = Runtime(device, 500)
    torch.npu.set_device(device)

    rt.configure_swizzle(swizzle, False)

    real_args = make_inputs(m, n, k)
    warmup_args = make_inputs(32768, 32768, 32768)

    run_time = run_matmul(rt, real_args, warmup_args, device)
    res = {
        'swizzle': hex(swizzle),
        'm': m,
        'n': n,
        'k': k,
        'time': run_time,
    }

    return res


def process_job(x):
    (swizzle, (m, n, k)) = x
    name = mp.current_process().name
    device = int(''.join(filter(str.isnumeric, name))) - 1
    # TODO: Process all swizzle values on a single NPU may decrease
    # variance caused by per-device differences in clocks and voltages.
    # From my tests this may decrease error by ~2%, but we can use
    # --num_devices=1 on candidate swizzles.

    return run_test(m, n, k, swizzle, device)


def worker_init():
    # Warm-up run
    # x = (32, 2048, 2048)
    # process_job(x)
    pass


def main():
    args = process_args()
    validate_globals()

    m = [32, 2048, 4096, 5000]
    nk = [(6144, 2048), (2048, 6144)]

    shapes = [(x[0], *x[1]) for x in itertools.product(m, nk)]
    r = range(0x100, 0x1000, 0x100)
    swizzles = itertools.chain(*[(x, x+1) for x in r])

    jobs = itertools.product(swizzles, shapes)
    jobs = list(jobs) * RUNS_PER_INPUT
    random.shuffle(jobs)

    print(f"Running {len(jobs)} jobs on {args.num_devices} devices")

    with mp.Pool(processes=args.num_devices, initializer=worker_init) as pool:
        res = tqdm.tqdm(pool.imap(process_job, jobs), total=len(jobs))

        res = list(res)
        with open('swizzle_perf_raw.json', 'w') as f:
            json.dump(res, f)

    return 0


if __name__ == "__main__":
    main()
