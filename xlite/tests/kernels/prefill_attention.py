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
from xlite._C import Runtime, prefill_attention

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

N_HEADS = 32
N_KV_HEADS = 32

MAX_BATCH_SIZE = 8
BATCH_SIZE = 8

START_POS = 0
BLOCK_SIZE = 128
TILESIZE_OF_QUERY = 128
AICNUM = 25
dtype_list = [torch.float16, torch.bfloat16]
head_dim_list = [64, 128]
seq_len_list = [30, 77, 1800, 24322, 32769]

for test_dtype in dtype_list:
    for head_dim in head_dim_list:
        for seq_len in seq_len_list:
            if seq_len >= 24322:
                N_HEADS = 1
                N_KV_HEADS = 1
                MAX_BATCH_SIZE = 1
                BATCH_SIZE = 1
            else:
                N_HEADS = 32
                N_KV_HEADS = 32
                MAX_BATCH_SIZE = 8
                BATCH_SIZE = 8
            MAX_SEQ_LEN = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
            torch.set_default_dtype(test_dtype)
            out_features = (N_HEADS + 2 * N_KV_HEADS) * head_dim
            with torch.device("npu"):
                # standard
                qkv_standard = torch.randn(BATCH_SIZE, seq_len, out_features) / 1000
                k_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, head_dim)
                v_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, head_dim)
                mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

                # xlite
                qkv_xlite = qkv_standard.clone().view(BATCH_SIZE * seq_len, out_features)
                qk = torch.zeros(AICNUM * TILESIZE_OF_QUERY * 2 * MAX_SEQ_LEN)
                output_xlite = torch.zeros(BATCH_SIZE * seq_len, N_HEADS * head_dim)
                max_num_block = math.ceil((seq_len + START_POS) / BLOCK_SIZE)

                kvcache_block_num = MAX_SEQ_LEN // BLOCK_SIZE * MAX_BATCH_SIZE
                k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, N_KV_HEADS, head_dim)
                v_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, N_KV_HEADS, head_dim)

                prefill_index = torch.arange(BATCH_SIZE, dtype=torch.int32)

                lens_list = [seq_len] * BATCH_SIZE
                lens = torch.tensor(lens_list, dtype=torch.int32).flatten()

                lens_array = np.array(lens_list)
                prefix_sums_array = np.cumsum(lens_array) - lens_array
                cum_prompt_lens = torch.tensor(prefix_sums_array.tolist(), dtype=torch.int32).flatten()

                cached_lens_list = [START_POS] * BATCH_SIZE
                cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()

                step = MAX_SEQ_LEN // BLOCK_SIZE
                block_num = (seq_len + START_POS + BLOCK_SIZE - 1) // BLOCK_SIZE
                batch_indices = np.arange(BATCH_SIZE, dtype=np.uint32).reshape(-1, 1)
                block_indices = np.arange(block_num, dtype=np.uint32)
                block_tables_array = batch_indices * step + block_indices
                block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

            # standard
            q, k, v = qkv_standard.split([N_HEADS * head_dim, N_KV_HEADS * head_dim, N_KV_HEADS * head_dim], dim=2)

            q = q.view(BATCH_SIZE, seq_len, N_HEADS, head_dim)
            k = k.view(BATCH_SIZE, seq_len, N_KV_HEADS, head_dim)
            v = v.view(BATCH_SIZE, seq_len, N_KV_HEADS, head_dim)

            k_cache[:BATCH_SIZE, START_POS:START_POS + seq_len] = k
            v_cache[:BATCH_SIZE, START_POS:START_POS + seq_len] = v

            keys = k_cache[:BATCH_SIZE, :START_POS + seq_len]
            values = v_cache[:BATCH_SIZE, :START_POS + seq_len]

            keys = keys.repeat_interleave(N_HEADS // N_KV_HEADS, dim=2)
            values = values.repeat_interleave(N_HEADS // N_KV_HEADS, dim=2)

            scores = torch.matmul(
                q.transpose(1, 2),  # [BATCH_SIZE, N_HEADS, seq_len, head_dim]
                keys.permute(0, 2, 3, 1)  # [BATCH_SIZE, N_HEADS, head_dim, seq_len+START_POS]
            ) # [BATCH_SIZE, N_HEADS, seq_len, seq_len+START_POS]

            if mask is not None:
                scores = scores + mask

            scores = torch.softmax(scores, dim=-1)

            output = torch.matmul(
                scores,  # [BATCH_SIZE, N_HEADS, seq_len, seq_len+START_POS]
                values.transpose(1, 2)  # [BATCH_SIZE, N_HEADS, seq_len+START_POS, head_dim]
            ).transpose(1, 2).contiguous()  # [BATCH_SIZE, seq_len, N_HEADS, head_dim]

            output_standard = output.view(BATCH_SIZE * seq_len, N_HEADS * head_dim)

            # xlite
            q_xlite, k_xlite, v_xlite = qkv_xlite.split([N_HEADS * head_dim, N_KV_HEADS * head_dim, N_KV_HEADS * head_dim], dim=1)
            batch_block_num = kvcache_block_num // MAX_BATCH_SIZE
            k_xlite = k_xlite.view(BATCH_SIZE, seq_len, N_KV_HEADS, head_dim)
            v_xlite = v_xlite.view(BATCH_SIZE, seq_len, N_KV_HEADS, head_dim)

            for i in range(BATCH_SIZE):
                # 1. 获取第i个样本的专属缓存块起始索引和当前样本的KV数据
                sample_cache_start = i * batch_block_num  # 第i个样本的缓存块起始位置
                current_k = k_xlite[i:i+1]  # 形状：[1, seq_len, N_KV_HEADS, head_dim]
                current_v = v_xlite[i:i+1]  # 形状：[1, seq_len, N_KV_HEADS, head_dim]
                
                # 2. 计算需要拆分的子块数量（向上取整，确保覆盖完整SEQ_LEN）
                num_blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE  # 等价于ceil(seq_len / BLOCK_SIZE)
                
                # 3. 循环分块填充，避免超出BLOCK_SIZE限制
                for block_idx in range(num_blocks_needed):
                    # 3.1 计算当前子块的序列索引范围
                    seq_start = block_idx * BLOCK_SIZE
                    seq_end = min((block_idx + 1) * BLOCK_SIZE, seq_len)  # 最后一个子块避免超出SEQ_LEN
                    current_seq_len = seq_end - seq_start  # 当前子块的有效序列长度
                    
                    # 3.2 计算当前子块对应的缓存块索引（确保不超出该样本的缓存块总数）
                    cache_block_idx = sample_cache_start + block_idx
                    if cache_block_idx >= (i + 1) * batch_block_num:
                        # 超出该样本分配的缓存块容量，抛出警告或终止（根据业务需求选择）
                        warnings.warn(f"第{i}个样本的缓存块容量不足，无法容纳超长序列（需要{num_blocks_needed}块，仅分配{batch_block_num}块）")
                        break
                    
                    # 3.3 分块赋值填充，确保维度匹配
                    k_cache_xlite[cache_block_idx, :current_seq_len] = current_k[:, seq_start:seq_end]
                    v_cache_xlite[cache_block_idx, :current_seq_len] = current_v[:, seq_start:seq_end]

            torch.npu.synchronize()
            prefill_attention(rt, qkv_xlite, k_cache_xlite, qk, block_tables, cached_lens,
                            v_cache_xlite, output_xlite, lens, prefill_index, cum_prompt_lens,
                            head_dim, N_HEADS, N_KV_HEADS, BLOCK_SIZE, BATCH_SIZE, max_num_block)

            logging.info(f'attention_prefill (head_dim={head_dim}, seq_len={seq_len}, {test_dtype}) executed!')

            try:
                torch.testing.assert_close(output_standard, output_xlite, atol=1e-5, rtol=1e-3)
            except AssertionError as e:
                logging.error(f'{e}')
                logging.error(f'torch_npu: {output_standard}')
                logging.error(f'xlite: {output_xlite}')
