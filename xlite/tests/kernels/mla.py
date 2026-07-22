#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""
mla kernel test (unified).

Each (model, work) case runs the `mla` and `mla_v2` interfaces against the
*same* random inputs, so the two kernel paths share one standard reference
implementation. The standard loop computes:

    q_absorb  = q_nope @ WUK[:, :nope_head_dim]            # host-side absorb
    scores    = (q_absorb @ k_cache + q_rope @ pe_cache) * scale + mask
    o_absorb  = softmax(scores) @ k_cache                  # shape (t, h, kv_lora_rank)
    output    = o_absorb @ WUV                             # shape (t, h, v_head_dim)

- `mla` / `mla_with_indices`: one fused kernel takes qWithQr + wkvb, does
  absorb + attention + WUV internally; output is the projected tensor (v_head_dim).
- `mla_v2`: three-kernel path — wuk einsum + mla_v2 attention kernel + wuv
  einsum — takes qWithQr + qr + wuk_t + wuv, output is still the projected
  tensor (v_head_dim). Both paths are compared against `output`.

`pe_cache` here is the RoPE'd key slice (shape (batch, max_seq, rope_head_dim))
— named `v_cache` historically in mla.py; semantically identical to pe_cache in
mla_v2.py.
"""
from __future__ import absolute_import

import logging
import torch
import math
import numpy as np
import warnings
from typing import Iterable
from xlite._C import Runtime, mla, mla_with_indices, mla_v2
from xlite._C import print as xlite_print
from tests.models.weight_utils import matrix_nd2nz

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 3000)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
weight_nz = True
enable_flash = True
tile_size = 8192

BLOCK_SIZE = 128
MAX_SOFTMAX_PINGPONG_LEN = 11776

# model configurations: name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, dtype
models = [
    ("base", 1, 64, 128, 128, 16, torch.bfloat16),
    ("deepseek_v3", 16, 64, 128, 128, 512, torch.bfloat16),
    ("glm5", 8, 64, 192, 256, 512, torch.bfloat16),
]

# work configurations: batch_size, cached_lens, query_lens
work = [
    (1, [0], [1]),
    (1, [0], [30]),
    (1, [0], [77]),
    (1, [0], [128]),
    (1, [0], [256]),
    (1, [0], [1600]),
    (1, [0], [2528]),
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
            pe_cache = torch.randn(batch, max_seq_len, rope_head_dim)
            pe_cache = rms_norm_last_dim(pe_cache)
            wkvb = torch.randn(n_heads * (nope_head_dim + v_head_dim), kv_lora_rank).clamp(0.2, 1)
            wkvb = wkvb.view(n_heads, nope_head_dim + v_head_dim, kv_lora_rank)

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
            pe_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, rope_head_dim)

            query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
            cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
            query_lens_np = np.array(query_len_list)
            query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
            query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

            batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
            block_indices = np.arange(max_num_blocks, dtype=np.uint32)
            block_tables_array = batch_indices * max_num_blocks + block_indices
            block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

            # mla_v2 inputs: pre-absorbed q_absorb + split qr (built after standard loop)
            q_absorb_xlite = torch.empty(total_query_len, n_heads, kv_lora_rank)
            qr_xlite = torch.empty(total_query_len, n_heads, rope_head_dim)
            # empty topk_indices placeholder for the top_k=0 path
            topk_indices_empty = torch.zeros(0, dtype=torch.int32)

            # mla_v2 now runs wuk einsum + mla_v2 + wuv einsum in one call;
            # prepare split wukT / wuv weights + separate v2 output buffers.
            # wukT shape: (n_heads, nope_head_dim, kv_lora_rank)
            # wuv  shape: (n_heads, kv_lora_rank, v_head_dim)  — htd layout for einsum
            wukT_xlite = wkvb[:, :nope_head_dim].reshape(n_heads, nope_head_dim, kv_lora_rank).contiguous()
            wuv_xlite = wkvb[:, -v_head_dim:].transpose(1, 2).contiguous()
            if weight_nz:
                wukT_xlite = matrix_nd2nz(wukT_xlite.reshape(n_heads * nope_head_dim, kv_lora_rank))
                wuv_xlite = matrix_nd2nz(wuv_xlite.reshape(n_heads * kv_lora_rank, v_head_dim))
            output_xlite_v2 = torch.zeros(total_query_len, n_heads, v_head_dim)

        # standard MLA forward: process each sample with its own query_len and cached_len
        # produces o_absorb_standard (stop at absorb) and output_standard (with WUV projection).
        # q_absorb_list is reused as mla_v2 kernel input to avoid recomputing the absorb einsum.
        o_absorbs = []
        outputs = []
        q_absorb_list = []
        offset = 0
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]

            # slice this sample's flat qWithQr from concatenated qWithQr_standard
            qWithQr_chunk = qWithQr_standard[offset: offset + qlen].unsqueeze(0)
            offset += qlen

            # split into nope and rope parts
            q_nope, q_rope = qWithQr_chunk.split([nope_head_dim, rope_head_dim], dim=-1)

            # q_absorb = q_nope @ WUK[:, :nope_head_dim]
            q_absorb = torch.einsum("bshd,hdc->bshc", q_nope, wkvb[:, :nope_head_dim])
            q_absorb_list.append(q_absorb.squeeze(0))

            qkc = torch.einsum("bshc,btc->bsht", q_absorb, k_cache[i:i+1, :clen + qlen])

            # compute QR * KR (KR is pe_cache, the RoPE'd key slice)
            qkr = torch.einsum("bshr,btr->bsht", q_rope, pe_cache[i:i+1, :clen + qlen])

            # combine scores
            scores = qkc + qkr
            scores = scores * scale

            # add per-sample mask if present
            if masks is not None:
                scores = scores + masks[i].unsqueeze(0).unsqueeze(1).permute(0, 2, 1, 3)

            scores = torch.softmax(scores, dim=-1)

            # o_absorb = softmax(QK) @ k_cache — stop here for mla_v2 comparison
            o_absorb = torch.einsum("bsht,btc->bshc", scores, k_cache[i:i+1, :clen + qlen])
            o_absorbs.append(o_absorb.squeeze(0))

            # output = o_absorb @ WUV — for mla comparison
            x = torch.einsum("bshc,hdc->bshd", o_absorb, wkvb[:, -v_head_dim:])
            outputs.append(x.squeeze(0))

        o_absorb_standard = torch.cat(o_absorbs, dim=0)
        output_standard = torch.cat(outputs, dim=0)

        # build mla_v2 inputs from the shared q_absorb_list (no extra einsum)
        offset = 0
        for i in range(batch):
            qlen = query_len_list[i]
            qWithQr_chunk = qWithQr_standard[offset: offset + qlen].unsqueeze(0)
            offset += qlen
            _, q_rope = qWithQr_chunk.split([nope_head_dim, rope_head_dim], dim=-1)
            q_absorb_xlite[offset - qlen: offset] = q_absorb_list[i]
            qr_xlite[offset - qlen: offset] = q_rope.squeeze(0)

        # xlite: write per-sample KV into block cache
        for i in range(batch):
            start = int(query_start_loc[i].item())
            qlen = int(query_lens[i].item())
            clen = int(cached_lens[i].item())

            # extract this sample's k/pe
            current_k = k_cache[i:i+1, :qlen + clen]
            current_pe = pe_cache[i:i+1, :qlen + clen]

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
                pe_cache_xlite[cache_block_idx, :current_seq_len] = current_pe[:, seq_start:seq_end]

        xlite_wkvb = wkvb.view(n_heads * (nope_head_dim + v_head_dim), kv_lora_rank)
        if weight_nz:
            xlite_wkvb = matrix_nd2nz(xlite_wkvb)
        torch.npu.synchronize()

        # ----- mla kernel (absorb + WUV inside) -----
        mla(rt, qWithQr_xlite, k_cache_xlite, pe_cache_xlite, xlite_wkvb,
                   output_xlite, query_start_loc, query_lens, cached_lens,
                   block_tables, n_heads, rope_head_dim, nope_head_dim,
                   v_head_dim, kv_lora_rank, BLOCK_SIZE, batch, max_num_blocks, scale, weight_nz, enable_flash, tile_size)

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

        # ----- mla_v2 kernel (wuk einsum + mla_v2 + wuv einsum) -----
        torch.npu.synchronize()
        mla_v2(rt, qWithQr_xlite, qr_xlite, k_cache_xlite, pe_cache_xlite, wukT_xlite, wuv_xlite,
               output_xlite_v2, query_start_loc, query_lens, cached_lens, block_tables, n_heads,
               rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, BLOCK_SIZE, batch,
               max_num_blocks, scale, topk_indices_empty, 0, weight_nz, enable_flash, tile_size)

        logging.info(
            "mla_v2 %s (%d heads, %d rope_head_dim, %d nope_head_dim, %d v_head_dim, "
            "%d kv_lora_rank, %s) work (%d batch, cached_lens=%s, query_lens=%s) executed!",
            name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, test_dtype,
            batch, cached_lens_list, query_len_list,
        )

        try:
            torch.testing.assert_close(output_xlite_v2, output_standard, atol=1e-5, rtol=5e-02)
        except AssertionError as e:
            logging.error(f'{e}')
            logging.error(f'torch_npu: {output_standard}')
            logging.error(f'xlite: {output_xlite_v2}')

        if not enable_flash and max_seq_len > MAX_SOFTMAX_PINGPONG_LEN:
            continue
        # Test MLA with topkIndices: verify that MLA with topkIndices produces the same output
        # as standard MLA with -inf mask applied before softmax
        # This simulates the DSA (Dual Sparse Attention) behavior used in deepseek_v32/glm5
        top_k_test_cases = [
            # (min(128, max_seq_len),),  # small topk
            # (min(1024, max_seq_len),),  # medium topk
            (min(2048, max_seq_len),),  # large topk
        ]

        for topk_value in top_k_test_cases:
            topk = topk_value[0]
            if topk <= 0:
                continue

            # Generate random topk_indices for each sample in the batch
            # This simulates the indexer output that selects top-k positions
            # topk_indices is a flattened tensor of shape (total_query_len, topk)
            topk_indices_list = []
            for i in range(batch):
                qlen = query_len_list[i]
                clen = cached_lens_list[i]
                total_len = qlen + clen
                # For each query position, generate topk indices
                # Each query position q_idx can only attend to positions in [0, clen + q_idx]
                # due to causal mask constraint
                for q_idx in range(qlen):
                    valid_len = clen + q_idx + 1  # +1 because q_idx can attend to itself
                    perm = torch.randperm(valid_len, device="npu")
                    if valid_len < topk:
                        # Pad with the last valid index to reach topk size
                        perm = torch.cat([perm, perm[-1:].expand(topk - valid_len)])
                    else:
                        perm = perm[:topk]
                    topk_indices_list.append(perm)

            if len(topk_indices_list) == 0:
                continue

            # Concatenate topk_indices into a flattened tensor
            # Shape: (total_query_len, topk)
            topk_indices_tensor = torch.stack(topk_indices_list)  # (total_query_len, topk)
            # mla_with_indices / mla_v2 假设每行的 topk indices 已按升序排列
            topk_indices_tensor, _ = torch.sort(topk_indices_tensor, dim=-1)

            # Standard MLA with topk mask: apply -inf mask before softmax
            # Produces o_absorb_standard_with_topk (stop at absorb) and
            # output_standard_with_topk (with WUV projection).
            o_absorbs_with_topk = []
            outputs_with_topk = []
            q_offset = 0
            indices_offset = 0
            for i in range(batch):
                qlen = query_len_list[i]
                clen = cached_lens_list[i]

                qWithQr_chunk = qWithQr_standard[q_offset: q_offset + qlen].unsqueeze(0)
                q_offset += qlen

                q_nope, q_rope = qWithQr_chunk.split([nope_head_dim, rope_head_dim], dim=-1)
                # 复用第一个循环里已算好的 q_absorb（按 batch 索引）
                q_absorb = q_absorb_list[i].unsqueeze(0)
                qkc = torch.einsum("bshc,btc->bsht", q_absorb, k_cache[i:i+1, :clen + qlen])
                qkr = torch.einsum("bshr,btr->bsht", q_rope, pe_cache[i:i+1, :clen + qlen])
                scores = qkc + qkr
                scores = scores * scale

                # Apply topk mask: -inf for positions not in topk_indices, 0 for topk positions
                total_len = clen + qlen
                # Get indices for this sample: stack indices for all query positions in this sample
                sample_indices = torch.stack(topk_indices_list[indices_offset:indices_offset + qlen])  # (qlen, topk)
                indices_offset += qlen

                # scores shape (1, qlen, heads, total_len), topk_indices shape (qlen, topk)
                index_mask = torch.full((1, qlen, total_len), float("-inf"), device="npu")
                topk_idx = sample_indices.unsqueeze(0)  # (1, qlen, topk)
                index_mask.scatter_(-1, topk_idx, 0)

                # add per-sample mask if present
                if masks is not None:
                    scores = scores + masks[i].unsqueeze(0).unsqueeze(1).permute(0, 2, 1, 3)

                scores = scores + index_mask.unsqueeze(2)  # Add mask to scores

                scores = torch.softmax(scores, dim=-1)
                # o_absorb = softmax(QK) @ k_cache — stop here for mla_v2 comparison
                o_absorb = torch.einsum("bsht,btc->bshc", scores, k_cache[i:i+1, :clen + qlen])
                o_absorbs_with_topk.append(o_absorb.squeeze(0))
                # output = o_absorb @ WUV — for mla comparison
                x = torch.einsum("bshc,hdc->bshd", o_absorb, wkvb[:, -v_head_dim:])
                outputs_with_topk.append(x.squeeze(0))

            o_absorb_standard_with_topk = torch.cat(o_absorbs_with_topk, dim=0)
            output_standard_with_topk = torch.cat(outputs_with_topk, dim=0)

            # xlite MLA with topkIndices
            output_xlite_with_topk = torch.zeros(total_query_len, n_heads, v_head_dim, device="npu", dtype=test_dtype)
            output_xlite_with_topk_v2 = torch.zeros(total_query_len, n_heads, v_head_dim, device="npu", dtype=test_dtype)

            topk_indices_tensor = topk_indices_tensor.to(dtype=torch.int32)
            torch.npu.synchronize()

            mla_with_indices(rt, qWithQr_xlite, k_cache_xlite, pe_cache_xlite, xlite_wkvb,
                output_xlite_with_topk, query_start_loc, query_lens, cached_lens,
                block_tables, n_heads, rope_head_dim, nope_head_dim,
                v_head_dim, kv_lora_rank, BLOCK_SIZE, batch, max_num_blocks, scale,
                topk, topk_indices_tensor, weight_nz, enable_flash, tile_size)

            logging.info(
                "mla with topkIndices %s (%d heads, %d rope_head_dim, %d nope_head_dim, %d v_head_dim, %d kv_lora_rank, %s) work (%d batch, cached_lens=%s, query_lens=%s, topk=%d) executed!",
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
                topk,
            )

            try:
                torch.testing.assert_close(output_xlite_with_topk, output_standard_with_topk, atol=1e-5, rtol=5e-02)
            except AssertionError as e:
                logging.error(f'MLA with topkIndices test failed: {e}')
                logging.error(f'Standard MLA with topk mask: {output_standard_with_topk}')
                logging.error(f'xlite MLA with topkIndices: {output_xlite_with_topk}')

            torch.npu.synchronize()
            mla_v2(rt, qWithQr_xlite, qr_xlite, k_cache_xlite, pe_cache_xlite, wukT_xlite, wuv_xlite,
                   output_xlite_with_topk_v2, query_start_loc, query_lens, cached_lens, block_tables,
                   n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, BLOCK_SIZE, batch,
                   max_num_blocks, scale, topk_indices_tensor, topk, weight_nz, enable_flash,
                   tile_size)

            logging.info(
                "mla_v2 with topkIndices %s (%d heads, %d rope_head_dim, %d nope_head_dim, "
                "%d v_head_dim, %d kv_lora_rank, %s) work (%d batch, cached_lens=%s, "
                "query_lens=%s, topk=%d) executed!",
                name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, test_dtype,
                batch, cached_lens_list, query_len_list, topk,
            )

            try:
                torch.testing.assert_close(
                    output_xlite_with_topk_v2, output_standard_with_topk, atol=1e-5, rtol=5e-02)
            except AssertionError as e:
                logging.error(f'mla_v2 with topkIndices test failed: {e}')
                logging.error(f'Standard mla_v2 with topk mask: {output_standard_with_topk}')
                logging.error(f'xlite mla_v2 with topkIndices: {output_xlite_with_topk_v2}')
