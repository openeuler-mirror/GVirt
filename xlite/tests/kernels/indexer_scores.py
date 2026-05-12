#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import

import logging
import torch
import numpy as np
from xlite._C import Runtime, indexer_scores

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

BLOCK_SIZE = 128

# model configurations: name, n_heads, head_dim, dtype
models = [
    ("indexer_64", 64, 128, torch.bfloat16),
]

# work configurations: batch_size, cached_lens, query_lens
work = [
    (1, [0], [1]),
    (1, [0], [4]),
    (1, [0], [30]),
    (1, [0], [128]),
    (1, [10], [4]),
    (1, [100], [30]),
    (2, [0] * 2, [4, 8]),
    (4, [50, 127, 89,500], [4, 2, 6, 8]),
]


def max_blocks(query_lens: list[int], cached_lens: list[int], BLOCK_SIZE: int) -> int:
    """
    计算 (query_lens[i] + cached_lens[i]) 列表中最大元素，向上整除 BLOCK_SIZE 后的块数。
    """
    max_sum = max(a + b for a, b in zip(query_lens, cached_lens))
    return (max_sum + BLOCK_SIZE - 1) // BLOCK_SIZE


for name, n_heads, head_dim, test_dtype in models:
    scale = head_dim ** -0.5
    for batch, cached_lens_list, query_len_list in work:
        max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE)
        max_seq_len = max_num_blocks * BLOCK_SIZE
        total_query_len = sum(query_len_list)

        torch.set_default_dtype(test_dtype)
        with torch.device("npu"):
            # standard
            q_standard = torch.randn(total_query_len, n_heads * head_dim)
            k_cache = torch.randn(batch, max_seq_len, head_dim)
            weight_standard = torch.randn(total_query_len, head_dim + n_heads)

            # xlite
            q_xlite = q_standard.clone()
            weight_xlite = weight_standard.clone()

            kvcache_block_num = max_num_blocks * batch
            k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, head_dim)

            scores_xlite = torch.zeros(total_query_len, max_num_blocks * BLOCK_SIZE)

            query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
            cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
            query_lens_np = np.array(query_len_list)
            query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
            query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

            batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
            block_indices = np.arange(max_num_blocks, dtype=np.uint32)
            block_tables_array = batch_indices * max_num_blocks + block_indices
            block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

        # standard indexer_scores forward: process each sample with its own query_len and cached_len
        index_scores_standard_list = []
        offset = 0
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]

            # slice this sample's flat q from concatenated q_standard
            q_chunk = q_standard[offset: offset + qlen]
            weight_chunk = weight_standard[offset: offset + qlen, head_dim: head_dim + n_heads]
            offset += qlen

            # reshape to [query_len, n_heads, head_dim]
            q_chunk = q_chunk.view(qlen, n_heads, head_dim)

            # scale q
            #q_chunk = q_chunk * scale

            # scores = q * k_cache^T, result: [query_len, n_heads, cached_len + query_len]
            k_cache_slice = k_cache[i:i+1, :clen + qlen]  # shape: [1, t, head_dim]
            scores = torch.einsum("bshd,btd->bsht", q_chunk.unsqueeze(0), k_cache_slice)  # [1, query_len, n_heads, t]
            # index_score = scores * weight, result: [query_len, cached_len + query_len]
            scores = scores.squeeze(0)  # [query_len, n_heads, t]
            index_score = torch.einsum("sht,sh->st", scores, weight_chunk)
            index_scores_standard_list.append(index_score)

        # xlite: write per-sample KV into block cache
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]

            # extract this sample's k
            current_k = k_cache[i:i+1, :qlen + clen]

            sample_cache_start = i * max_num_blocks
            total_len = clen + qlen
            num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            for block_idx in range(num_blocks_needed):
                seq_start = block_idx * BLOCK_SIZE
                seq_end = min((block_idx + 1) * BLOCK_SIZE, total_len)
                current_seq_len = seq_end - seq_start

                cache_block_idx = sample_cache_start + block_idx
                if cache_block_idx >= (i + 1) * max_num_blocks:
                    break

                k_cache_xlite[cache_block_idx, :current_seq_len] = current_k[:, seq_start:seq_end]

        torch.npu.synchronize()
        indexer_scores(rt, q_xlite, k_cache_xlite, weight_xlite, scores_xlite,
                      query_start_loc, query_lens, cached_lens, block_tables,
                      n_heads, head_dim, BLOCK_SIZE, batch, max_num_blocks)

        logging.info(
            "indexer_scores %s (%d heads, %d head_dim, %s) work (%d batch, cached_lens=%s, query_lens=%s) executed!",
            name,
            n_heads,
            head_dim,
            test_dtype,
            batch,
            cached_lens_list,
            query_len_list,
        )

        # Compare results (only compare valid positions)
        # Compare per batch since samples may have different output shapes
        all_match = True
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]
            total_len = clen + qlen

            # Get standard result for this sample
            sample_standard = index_scores_standard_list[i]

            # Get xlite result for this sample
            # scores_xlite is [total_query_len, max_num_blocks * BLOCK_SIZE]
            # Find the offset for this sample in the flat scores_xlite
            sample_offset = query_start_loc[i].item()
            sample_xlite = scores_xlite[sample_offset: sample_offset + qlen, :total_len]

            try:
                torch.testing.assert_close(sample_standard, sample_xlite.to("npu"),
                                        atol=1e-5, rtol=5e-02)
            except AssertionError as e:
                logging.error(f'{e}')
                logging.error(f'Sample {i} mismatch:')
                logging.error(f'torch: {sample_standard}')
                logging.error(f'xlite: {sample_xlite}')
                all_match = False

        if all_match:
            logging.info("All samples passed!")
