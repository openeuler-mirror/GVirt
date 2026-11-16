#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import logging
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401

from xlite._C import Runtime, all_reduce, embed

logging.getLogger().setLevel(logging.INFO)

world_size = int(os.getenv("WORLD_SIZE", "1"))
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("hccl")

rt = Runtime(local_rank, size=500, rank=rank, tp_size=world_size)
torch.npu.set_device(local_rank)

VOCAB_SIZE = 32000
DIM = 4096
rank_id = local_rank % world_size
emb_start_idx = VOCAB_SIZE // world_size * rank_id
emb_end_idx = emb_start_idx + VOCAB_SIZE // world_size
dtype_list = [torch.float16, torch.bfloat16]
# 64 rows exercise the barrier-per-row small-batch path; 512 rows clears the
# rows-per-core threshold that routes embed_kernel to the ping-pong pipelined
# loop on 910B.
BATCH_SIZES = (64, 512)

for test_dtype in dtype_list:
    for batch_size in BATCH_SIZES:
        with torch.device("npu"):
            x_base = [random.randint(0, VOCAB_SIZE - 1) for _ in range(batch_size)]
            x = torch.tensor(x_base, dtype=torch.int32)
            x_standard = x.clone()
            weight = torch.randn(VOCAB_SIZE, DIM, dtype=test_dtype)
            y = torch.empty(batch_size, DIM, dtype=test_dtype)

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
            logging.info(f"embed({test_dtype}, batch={batch_size}) executed!")

        try:
            torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            logging.error(f"{e}")
            logging.error(f"torch_npu: {standard}")
            logging.error(f"xlite: {y}")
            raise
