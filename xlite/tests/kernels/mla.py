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
import math
import numpy as np
import warnings
from typing import Iterable
from xlite._C import Runtime, mla

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 3000)
torch.npu.set_device(0)

BLOCK_SIZE = 128

# model configurations: name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, dtype
models = [
    ("deepseek_v3", 16, 64, 128, 128, 512, torch.bfloat16),
]

# work configurations: batch_size, cached_lens, query_lens
work = [
    (1, [0], [1]),
    (1, [0], [30]),
    (1, [0], [77]),
    (1, [0], [128]),
    (1, [0], [256]),
    (1, [0], [1600]),
    (1, [100], [30]),
    (8, [0] * 8, [789, 65, 13, 6545, 24, 190, 2432, 124]),
    (2, [4000] * 2, [1] * 2),
    (2, [6000] * 2, [1] * 2),
    (2, [131071] * 2, [1] * 2),
    (5, [8, 13, 65, 11, 5], [1] * 5),
    (1, [128], [128]),
]

def max_blocks(query_lens: Iterable[int], cached_lens: Iterable[int], BLOCK_SIZE: int) -> int:
    """
    计算 (query_lens[i] + cached_lens[i]) 列表中最大元素，向上整除 BLOCK_SIZE 后的块数。
    """
    query_lens = list(query_lens)
    cached_lens = list(cached_lens)
    max_sum = max(a + b for a, b in zip(query_lens, cached_lens))
    return (max_sum + BLOCK_SIZE - 1) // BLOCK_SIZE

def rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # normalize over the last dimension by root-mean-square
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms

for name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, test_dtype in models:
    scale = (nope_head_dim + rope_head_dim) ** -0.5
    for batch, cached_lens_list, query_len_list in work:
        max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE)
        max_seq_len = max_num_blocks * BLOCK_SIZE
        total_query_len = sum(query_len_list)

        torch.set_default_dtype(test_dtype)
        with torch.device("npu"):
            # standard
            qWithQr_standard = torch.randn(total_query_len, n_heads, nope_head_dim + rope_head_dim)
            qWithQr_standard = rms_norm_last_dim(qWithQr_standard)
            k_cache = torch.randn(batch, max_seq_len, kv_lora_rank).clamp(0.2, 1)
            k_cache = rms_norm_last_dim(k_cache)
            v_cache = torch.randn(batch, max_seq_len, rope_head_dim)
            v_cache = rms_norm_last_dim(v_cache)
            wkvb = torch.randn(n_heads, nope_head_dim + v_head_dim, kv_lora_rank).clamp(0.2, 1)

            masks = []
            for i in range(batch):
                query_len = query_len_list[i]
                cached_len = cached_lens_list[i]
                # 生成目标掩码：当列索引 >= cached_len + 行索引 时为 -inf，其余为 0
                cols = torch.arange(query_len + cached_len).unsqueeze(0)
                rows = torch.arange(query_len).unsqueeze(1)
                mask = torch.where(cols > cached_len + rows, float("-inf"), 0.0)
                masks.append(mask)

            # xlite
            qWithQr_xlite = qWithQr_standard.clone()
            output_xlite = torch.zeros(total_query_len, n_heads, v_head_dim)

            kvcache_block_num = max_num_blocks * batch
            k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, kv_lora_rank)
            v_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, rope_head_dim)

            query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
            cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
            query_lens_np = np.array(query_len_list)
            query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
            query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

            batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
            block_indices = np.arange(max_num_blocks, dtype=np.uint32)
            block_tables_array = batch_indices * max_num_blocks + block_indices
            block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

        # standard MLA forward: process each sample with its own query_len and cached_len
        outputs = []
        offset = 0
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]

            # slice this sample's flat qWithQr from concatenated qWithQr_standard
            qWithQr_chunk = qWithQr_standard[offset: offset + qlen].unsqueeze(0)
            offset += qlen

            # split into nope and rope parts
            q_nope, q_rope = qWithQr_chunk.split([nope_head_dim, rope_head_dim], dim=-1)

            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkvb[:, :nope_head_dim])
            qkc = torch.einsum("bshc,btc->bsht", q_nope, k_cache[i:i+1, :clen + qlen])

            # compute QR * KR
            qkr = torch.einsum("bshr,btr->bsht", q_rope, v_cache[i:i+1, :clen + qlen])

            # combine scores
            scores = qkc + qkr
            scores = scores * scale

            # add per-sample mask if present
            if masks is not None:
                scores = scores + masks[i].unsqueeze(0).unsqueeze(1).permute(0, 2, 1, 3)

            scores = torch.softmax(scores, dim=-1)

            x = torch.einsum("bsht,btc->bshc", scores, k_cache[i:i+1, :clen + qlen])
            x = torch.einsum("bshc,hdc->bshd", x, wkvb[:, -v_head_dim:])

            outputs.append(x.squeeze(0))

        output_standard = torch.cat(outputs, dim=0)

        # xlite: write per-sample KV into block cache
        for i in range(batch):
            start = int(query_start_loc[i].item())
            qlen = int(query_lens[i].item())
            clen = int(cached_lens[i].item())

            # extract this sample's k/v
            current_k = k_cache[i:i+1, :qlen + clen]
            current_v = v_cache[i:i+1, :qlen + clen]

            sample_cache_start = i * max_num_blocks
            total_len = clen + qlen
            num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            for block_idx in range(num_blocks_needed):
                seq_start = block_idx * BLOCK_SIZE
                seq_end = min((block_idx + 1) * BLOCK_SIZE, total_len)
                current_seq_len = seq_end - seq_start

                cache_block_idx = sample_cache_start + block_idx
                if cache_block_idx >= (i + 1) * max_num_blocks:
                    warnings.warn(f"第{i}个样本的缓存块容量不足，无法容纳超长序列（需要{num_blocks_needed}块，仅分配{max_num_blocks}块）")
                    break

                k_cache_xlite[cache_block_idx, :current_seq_len] = current_k[:, seq_start:seq_end]
                v_cache_xlite[cache_block_idx, :current_seq_len] = current_v[:, seq_start:seq_end]

        torch.npu.synchronize()
        mla(rt, qWithQr_xlite, k_cache_xlite, v_cache_xlite, wkvb,
                   output_xlite, query_start_loc, query_lens, cached_lens,
                   block_tables, n_heads, rope_head_dim, nope_head_dim,
                   v_head_dim, kv_lora_rank, BLOCK_SIZE, batch, max_num_blocks, scale)

        logging.info(
            "mla %s (%d heads, %d rope_head_dim, %d nope_head_dim, %d v_head_dim, %d kv_lora_rank, %s) work (%d batch, cached_lens=%s, query_lens=%s) executed!",
            name,
            n_heads,
            rope_head_dim,
            nope_head_dim,
            v_head_dim,
            kv_lora_rank,
            test_dtype,
            batch,
            cached_lens_list,
            query_len_list,
        )

        try:
            torch.testing.assert_close(output_xlite, output_standard, atol=1e-5, rtol=5e-02)
        except AssertionError as e:
            logging.error(f'{e}')
            logging.error(f'torch_npu: {output_standard}')
            logging.error(f'xlite: {output_xlite}')
