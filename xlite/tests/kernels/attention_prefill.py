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
import torch
import math
import numpy as np
from xlite._C import Runtime, attention_prefill

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

HEAD_DIM = 128
N_HEADS = 32
N_KV_HEADS = 32

MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 8
BATCH_SIZE = 8
SEQ_LEN = 20

START_POS = 0
BLOCK_SIZE = 128
TILESIZE_OF_QUERY = 128
AICNUM = 20
max_m = MAX_SEQ_LEN if MAX_SEQ_LEN > MAX_BATCH_SIZE else MAX_BATCH_SIZE
out_features = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
dtype_list = [torch.float16]

for test_dtype in dtype_list:
    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        # standard
        qkv_standard = torch.randn(BATCH_SIZE, SEQ_LEN, out_features) / 1000
        k_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM)
        v_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM)
        mask = torch.full((SEQ_LEN, SEQ_LEN), float("-inf")).triu_(1)

        # xlite
        qkv_xlite = qkv_standard.clone().view(BATCH_SIZE * SEQ_LEN, out_features)
        qk = torch.zeros(AICNUM * TILESIZE_OF_QUERY * 2 * max_m)
        output_xlite = torch.zeros(BATCH_SIZE * SEQ_LEN, N_HEADS * HEAD_DIM)
        max_num_block = math.ceil((SEQ_LEN + START_POS) / BLOCK_SIZE)

        kvcache_block_num = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE * MAX_BATCH_SIZE
        k_cache_xlite = torch.zeros(kvcache_block_num, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
        v_cache_xlite = torch.zeros(kvcache_block_num, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)

        lens_list = [SEQ_LEN] * BATCH_SIZE
        lens = torch.tensor(lens_list, dtype=torch.int32).flatten()

        lens_array = np.array(lens_list)
        prefix_sums_array = np.cumsum(lens_array) - lens_array
        cum_prompt_lens = torch.tensor(prefix_sums_array.tolist(), dtype=torch.int32).flatten()

        cached_lens_list = [START_POS] * BATCH_SIZE
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()

        padding_list = [math.ceil((SEQ_LEN + START_POS) / BLOCK_SIZE) * BLOCK_SIZE] * BATCH_SIZE
        padding = torch.tensor(padding_list, dtype=torch.int32).flatten()

        step = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_num = (SEQ_LEN + START_POS + BLOCK_SIZE - 1) // BLOCK_SIZE
        batch_indices = np.arange(BATCH_SIZE, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(block_num, dtype=np.uint32)
        block_tables_array = batch_indices * step + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

    # standard
    q, k, v = qkv_standard.split([N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM], dim=2)

    q = q.view(BATCH_SIZE, SEQ_LEN, N_HEADS, HEAD_DIM)
    k = k.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
    v = v.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, HEAD_DIM)

    k_cache[:BATCH_SIZE, START_POS:START_POS + SEQ_LEN] = k
    v_cache[:BATCH_SIZE, START_POS:START_POS + SEQ_LEN] = v

    keys = k_cache[:BATCH_SIZE, :START_POS + SEQ_LEN]
    values = v_cache[:BATCH_SIZE, :START_POS + SEQ_LEN]

    keys = keys.repeat_interleave(N_HEADS // N_KV_HEADS, dim=2)
    values = values.repeat_interleave(N_HEADS // N_KV_HEADS, dim=2)

    scores = torch.matmul(
        q.transpose(1, 2),  # [BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM]
        keys.permute(0, 2, 3, 1)  # [BATCH_SIZE, N_HEADS, HEAD_DIM, SEQ_LEN+START_POS]
    ) # [BATCH_SIZE, N_HEADS, SEQ_LEN, SEQ_LEN+START_POS]

    if mask is not None:
        scores = scores.float() + mask.float()

    scores = torch.softmax(scores, dim=-1, dtype=torch.float).to(test_dtype)

    output = torch.matmul(
        scores,  # [BATCH_SIZE, N_HEADS, SEQ_LEN, SEQ_LEN+START_POS]
        values.transpose(1, 2)  # [BATCH_SIZE, N_HEADS, SEQ_LEN+START_POS, HEAD_DIM]
    ).transpose(1, 2).contiguous()  # [BATCH_SIZE, SEQ_LEN, N_HEADS, HEAD_DIM]

    output_standard = output.view(BATCH_SIZE * SEQ_LEN, N_HEADS * HEAD_DIM)

    # xlite
    q_xlite, k_xlite, v_xlite = qkv_xlite.split([N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM], dim=1)
    batch_block_num = kvcache_block_num // MAX_BATCH_SIZE
    k_xlite = k_xlite.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
    v_xlite = v_xlite.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
    k_xlite = k_xlite.transpose(1, 2)
    v_xlite = v_xlite.transpose(1, 2)
    for i in range(BATCH_SIZE):
        k_cache_xlite[i * batch_block_num : ((i + 1) * batch_block_num), :, :SEQ_LEN] = k_xlite[i : i + 1]
        v_cache_xlite[i * batch_block_num : ((i + 1) * batch_block_num), :, :SEQ_LEN] = v_xlite[i : i + 1]

    torch.npu.synchronize()
    attention_prefill(rt, qkv_xlite, k_cache_xlite, qk, block_tables, padding, cached_lens,
                      v_cache_xlite, output_xlite, lens, cum_prompt_lens,
                      HEAD_DIM, N_HEADS, N_KV_HEADS, BLOCK_SIZE, BATCH_SIZE, max_num_block)
    torch.npu.synchronize()

    logging.info(f'attention_preifll ({test_dtype}) executed!')

    try:
        torch.testing.assert_close(output_standard, output_xlite, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {output_standard}')
        logging.error(f'xlite: {output_xlite}')
