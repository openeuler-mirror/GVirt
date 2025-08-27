#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import
import logging
import os
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from xlite._C import Runtime, embed, all_reduce

logging.getLogger().setLevel(logging.INFO)

world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("hccl")

rt = Runtime(local_rank, 500, rank, world_size)
torch.npu.set_device(local_rank)

BATCH_SIZE = 64
VOCAB_SIZE = 32000
DIM = 4096
rank_id = local_rank % world_size
emb_start_idx = VOCAB_SIZE // world_size * rank_id
emb_end_idx = emb_start_idx + VOCAB_SIZE // world_size
x_base = [random.randint(0, VOCAB_SIZE - 1) for _ in range(BATCH_SIZE)]
dtype_list = [torch.float16, torch.bfloat16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        x = torch.tensor(x_base, dtype=torch.int32)
        x_standard = x.clone()
        weight = torch.randn(VOCAB_SIZE, DIM, dtype=test_dtype)
        y = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)

    # standard
    if world_size > 1:
        mask = (x_standard < emb_start_idx) | (x_standard >= emb_end_idx)
        x_standard = x_standard - emb_start_idx
        x_standard[mask] = 0
    standard = F.embedding(x_standard, weight)
    if world_size > 1:
        standard[mask] = 0
        dist.all_reduce(standard)

    # xlite
    torch.npu.synchronize()
    embed(rt, weight, x, y, emb_start_idx, emb_end_idx)
    if world_size > 1:
        all_reduce(rt, y, y)
    torch.npu.synchronize()

    if rank == 0:
        logging.info(f'embed({test_dtype}) executed!')

    try:
        torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {standard}')
        logging.error(f'xlite: {y}')