#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""
mla kernel test (unified).

Each (model, work) case runs the `mla_v2` interface against a standard
reference implementation. The standard loop computes:

    q_absorb  = q_nope @ WUK[:, :nope_head_dim]            # host-side absorb
    scores    = (q_absorb @ k_cache + q_rope @ pe_cache) * scale + mask
    o_absorb  = softmax(scores) @ k_cache                  # shape (t, h, kv_lora_rank)
    output    = o_absorb @ WUV                             # shape (t, h, v_head_dim)

- `mla_v2`: three-kernel path — wuk einsum + mla_v2 attention kernel + wuv
  einsum — takes qWithQr + qr + wuk_t + wuv, output is the projected tensor
  (v_head_dim), compared against `output`.

`pe_cache` here is the RoPE'd key slice (shape (batch, max_seq, rope_head_dim))
— named `v_cache` historically in mla.py; semantically identical to pe_cache in
mla_v2.py.
"""

import logging
import math
import sys
from collections.abc import Iterable

import torch
import torch_npu  # noqa: F401

from tests.models.weight_utils import matrix_nd2nz
from xlite._C import Runtime, mla_v2
from xlite._C import print as xlite_print

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, size=3000)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
weight_nz = True
enable_flash = True
tile_size = 8192

BLOCK_SIZE = 128
MAX_SOFTMAX_PINGPONG_LEN = 11776
# dense (gather) path supports decode and small prefill; mirrors the model's
# XLITE_SMALL_PREFILL_THRESHOLD gating (csrc/model.cpp).
MAX_DENSE_QUERY_LEN = 6

# model configurations: name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, dtype
models = [
    ("base", 1, 64, 128, 128, 16, torch.bfloat16),
    ("deepseek_v3", 16, 64, 128, 128, 512, torch.bfloat16),
    ("glm5", 8, 64, 192, 256, 512, torch.bfloat16),
]

# work configurations: batch_size, cached_lens, query_lens, [pitched]
# pitched=True allocates the paged caches as buf[::2] views so each physical
# block's dim-0 stride is double the contiguous pitch — exercises the
# stride-aware block addressing in mla_v2 / flash_mla_v2 / gather.
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
    (1, [16384], [1]),
    (3, [32751] * 3, [1] * 3),
    (3, [2047, 8191, 45841], [1] * 3),
    (4, [2047] * 4, [1] * 4),
    (1, [131071], [1]),
    (1, [127], [1]),
    (1, [4096], [1]),
    (1, [0], [30], True),
    (1, [127], [1], True),
    (2, [4000] * 2, [1] * 2, True),
    # small-prefill (MTP) cases: long cached + query_len <= MAX_DENSE_QUERY_LEN,
    # exercising the per-query-token dense gather path. Mixed dense/sparse rows
    # (short cached stays dense, long cached goes sparse) within one batch.
    (1, [8191], [5]),
    (2, [8191, 127], [3, 3]),
    (3, [99, 8191, 45841], [4, 3, 2]),
    (2, [4096, 4097], [2, 6]),
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


def check_dsa_cp_gather_mla():
    from xlite._C import dsa_cp_gather_mla

    torch.manual_seed(42)
    # cached_lens already include the prefix preceding this rank's local queries.
    cases = [
        (64, [0], [19], 2048),
        (64, [2030], [39], 2048),
        (64, [32768], [64], 2048),
        (64, [4096, 0, 131000, 16], [13, 0, 33, 3], 2048),
        (128, [4096, 1], [1, 2], 128),
        (2, [8192], [3], 128),
        (1, [127], [1], 128),
        (64, [0, 0], [0, 0], 2048),
        (64, [2176, 17, 4096], [11, 29, 27], 2048),
        # Multiple two-query-per-core groups on A3, including short-history
        # rows, a request boundary and an incomplete final group.
        (64, [4096, 17, 8192], [31, 17, 81], 2048),
        # Gather chunk/padding boundaries and different original page tables.
        (64, [0, 30, 31, 32, 126, 127, 128, 2048], [1] * 8, 128),
        # Persistent blockTables retain extra rows and the original row pitch
        # after the runtime narrows columns to this step's context length.
        (64, [4096, 17, 8192], [31, 17, 81], 2048, True),
    ]
    latent_dim, rope_dim = 512, 64
    scale = 256**-0.5
    for case in cases:
        heads, histories, lengths, top_k = case[:4]
        pitched_blocks = len(case) > 4 and case[4]
        total_queries = sum(lengths)
        contexts = [history + length for history, length in zip(histories, lengths)]
        pages = [(length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in contexts]
        num_pages = max(1, sum(pages))
        max_pages = max(1, max(pages))
        with torch.device("npu"):
            q = torch.randn(total_queries, heads, latent_dim, dtype=torch.bfloat16)
            qr = torch.randn(total_queries, heads, rope_dim, dtype=torch.bfloat16)
            out = torch.full_like(q, float("nan"))
            expected = torch.zeros_like(q)
            # Different block pitches for K and RoPE; untouched rows are NaNs,
            # so reading unselected KV in SV is observable even at zero probability.
            k_cache = torch.full((num_pages * 2, BLOCK_SIZE, latent_dim), float("nan"), dtype=torch.bfloat16)[::2]
            pe_cache = torch.full((num_pages * 3, BLOCK_SIZE, rope_dim), float("nan"), dtype=torch.bfloat16)[::3]
            if pitched_blocks:
                blocks = torch.full((len(lengths) + 2, max_pages + 8), -1, dtype=torch.int32)[:, :max_pages]
            else:
                blocks = torch.full((len(lengths), max_pages), -1, dtype=torch.int32)
            physical_pages = torch.randperm(num_pages)
            topk = torch.full((total_queries, top_k), -1, dtype=torch.int32)
            lens = torch.tensor(lengths, dtype=torch.int32)
            cached = torch.tensor(histories, dtype=torch.int32)
            starts = torch.cumsum(lens, 0, dtype=torch.int32) - lens
            query_offset, page_offset = 0, 0
            for request, (history, length, context, page_count) in enumerate(zip(histories, lengths, contexts, pages)):
                blocks[request, :page_count] = physical_pages[page_offset : page_offset + page_count]
                page_offset += page_count
                logical_k = torch.randn(context, latent_dim, dtype=torch.bfloat16)
                logical_pe = torch.randn(context, rope_dim, dtype=torch.bfloat16)
                needed = []
                for local_query in range(length):
                    row = query_offset + local_query
                    causal = history + local_query + 1
                    if causal <= top_k:
                        selected = torch.arange(causal)
                    else:
                        indices = torch.randperm(causal)[:top_k]
                        if row % 2 == 0:
                            indices = indices.sort().values
                        # Long rows follow the existing indexer/gather contract:
                        # top_k distinct, valid causal positions in arbitrary order.
                        if row % 3 == 0:
                            indices = indices.sort(descending=True).values
                        topk[row] = indices.to(torch.int32)
                        selected = indices
                    needed.append(selected)
                    if selected.numel():
                        keys, positions = logical_k[selected], logical_pe[selected]
                        # Match the kernel's BF16 QK and probability round trips.
                        scores = (q[row].float() @ keys.float().T + qr[row].float() @ positions.float().T).to(
                            torch.bfloat16
                        )
                        probability = (scores.float() * scale).softmax(-1).to(torch.bfloat16)
                        expected[row] = (probability.float() @ keys.float()).to(torch.bfloat16)
                if needed:
                    selected = torch.unique(torch.cat(needed))
                    physical = blocks[request, selected // BLOCK_SIZE].long()
                    within_page = selected % BLOCK_SIZE
                    k_cache[physical, within_page] = logical_k[selected]
                    pe_cache[physical, within_page] = logical_pe[selected]
                query_offset += length
            torch.npu.synchronize()
            dsa_cp_gather_mla(rt, q, qr, k_cache, pe_cache, out, starts, lens, cached, blocks, topk, scale)
            torch.testing.assert_close(out, expected, rtol=5e-2, atol=5e-3)
        logging.info(
            "DSA-CP gather + dense MLA passed: heads=%s cached=%s queries=%s top_k=%s pitched_blocks=%s",
            heads,
            histories,
            lengths,
            top_k,
            pitched_blocks,
        )


if "--dsa-cp" in sys.argv:
    check_dsa_cp_gather_mla()
    raise SystemExit(0)


for name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, test_dtype in models:
    scale = (nope_head_dim + rope_head_dim) ** -0.5
    for work_cfg in work:
        batch, cached_lens_list, query_len_list = work_cfg[:3]
        pitched = len(work_cfg) > 3 and bool(work_cfg[3])
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
            # WUK must stay zero-mean (fan-in scaled); WUV stays positive so
            # outputs stay away from zero and rtol remains meaningful. With
            # WUK also clamped positive, every key points into the positive
            # orthant and the qk logits share a huge constant offset (~4e3):
            # the bf16 ulp there (~2.3 after scale, larger than the whole
            # top-score spread) lets the reference's separate bf16 roundings
            # of qkc/qkr flip softmax ties against the kernel's single fused
            # dot, breaking the rtol check on near-tie rows. Zero-mean WUK
            # keeps the scaled logits ~N(0, 1.7).
            wuk = torch.randn(n_heads, nope_head_dim, kv_lora_rank) / math.sqrt(nope_head_dim)
            wuv = torch.randn(n_heads, v_head_dim, kv_lora_rank).clamp(0.2, 1)
            wkvb = torch.cat([wuk, wuv], dim=1).reshape(n_heads * (nope_head_dim + v_head_dim), kv_lora_rank)
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

            kvcache_block_num = max_num_blocks * batch
            if pitched:
                k_cache_xlite = torch.zeros(kvcache_block_num * 2, BLOCK_SIZE, kv_lora_rank)[::2]
                pe_cache_xlite = torch.zeros(kvcache_block_num * 2, BLOCK_SIZE, rope_head_dim)[::2]
            else:
                k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, kv_lora_rank)
                pe_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, rope_head_dim)

            query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
            cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
            query_start_loc = (query_lens.cumsum(0) - query_lens).int()

            block_tables = (torch.arange(batch)[:, None] * max_num_blocks + torch.arange(max_num_blocks)).to(
                dtype=torch.int32
            )

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
        qkc_list = []
        qkr_list = []
        offset = 0
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]

            # slice this sample's flat qWithQr from concatenated qWithQr_standard
            qWithQr_chunk = qWithQr_standard[offset : offset + qlen].unsqueeze(0)
            offset += qlen

            # split into nope and rope parts
            q_nope, q_rope = qWithQr_chunk.split([nope_head_dim, rope_head_dim], dim=-1)

            # q_absorb = q_nope @ WUK[:, :nope_head_dim]
            q_absorb = torch.einsum("bshd,hdc->bshc", q_nope, wkvb[:, :nope_head_dim])
            q_absorb_list.append(q_absorb.squeeze(0))

            qkc = torch.einsum("bshc,btc->bsht", q_absorb, k_cache[i : i + 1, : clen + qlen])
            qkc_list.append(qkc)

            # compute QR * KR (KR is pe_cache, the RoPE'd key slice)
            qkr = torch.einsum("bshr,btr->bsht", q_rope, pe_cache[i : i + 1, : clen + qlen])
            qkr_list.append(qkr)

            # combine scores
            scores = qkc + qkr
            scores = scores * scale

            # add per-sample mask if present
            if masks is not None:
                scores = scores + masks[i].unsqueeze(0).unsqueeze(1).permute(0, 2, 1, 3)

            scores = torch.softmax(scores, dim=-1)

            # o_absorb = softmax(QK) @ k_cache — stop here for mla_v2 comparison
            o_absorb = torch.einsum("bsht,btc->bshc", scores, k_cache[i : i + 1, : clen + qlen])
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
            qWithQr_chunk = qWithQr_standard[offset : offset + qlen].unsqueeze(0)
            offset += qlen
            _, q_rope = qWithQr_chunk.split([nope_head_dim, rope_head_dim], dim=-1)
            q_absorb_xlite[offset - qlen : offset] = q_absorb_list[i]
            qr_xlite[offset - qlen : offset] = q_rope.squeeze(0)

        # xlite: write per-sample KV into block cache
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]

            # extract this sample's k/pe
            current_k = k_cache[i : i + 1, : qlen + clen]
            current_pe = pe_cache[i : i + 1, : qlen + clen]

            total_len = clen + qlen
            num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            sample_cache_start = i * max_num_blocks
            if pitched:
                for block_idx in range(num_blocks_needed):
                    seq_start = block_idx * BLOCK_SIZE
                    seq_end = min((block_idx + 1) * BLOCK_SIZE, total_len)
                    k_cache_xlite[sample_cache_start + block_idx, : seq_end - seq_start] = current_k[
                        :, seq_start:seq_end
                    ]
                    pe_cache_xlite[sample_cache_start + block_idx, : seq_end - seq_start] = current_pe[
                        :, seq_start:seq_end
                    ]
            else:
                dst_k = k_cache_xlite[sample_cache_start : sample_cache_start + num_blocks_needed].view(
                    -1, kv_lora_rank
                )
                dst_pe = pe_cache_xlite[sample_cache_start : sample_cache_start + num_blocks_needed].view(
                    -1, rope_head_dim
                )
                dst_k[:total_len] = current_k.flatten(0, 1)
                dst_pe[:total_len] = current_pe.flatten(0, 1)

        # ----- mla_v2 kernel (wuk einsum + mla_v2 + wuv einsum) -----
        torch.npu.synchronize()
        mla_v2(
            rt,
            qWithQr_xlite,
            qr_xlite,
            k_cache_xlite,
            pe_cache_xlite,
            wukT_xlite,
            wuv_xlite,
            output_xlite_v2,
            query_start_loc,
            query_lens,
            cached_lens,
            block_tables,
            n_heads,
            rope_head_dim,
            nope_head_dim,
            v_head_dim,
            kv_lora_rank,
            BLOCK_SIZE,
            batch,
            scale,
            topk_indices_empty,
            top_k=0,
            nz=weight_nz,
            enable_flash_attention=enable_flash,
            tile_size_of_cached_kv=tile_size,
        )

        logging.info(
            "mla_v2 %s (%d heads, %d rope_head_dim, %d nope_head_dim, %d v_head_dim, "
            "%d kv_lora_rank, %s) work (%d batch, cached_lens=%s, query_lens=%s, "
            "pitched=%s) executed!",
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
            pitched,
        )

        try:
            torch.testing.assert_close(output_xlite_v2, output_standard, atol=1e-5, rtol=5e-02)
        except AssertionError as e:
            logging.error(f"{e}")
            logging.error(f"torch_npu: {output_standard}")
            logging.error(f"xlite: {output_xlite_v2}")

        # Test MLA with topkIndices: verify that MLA with topkIndices produces the same output
        # as standard MLA with -inf mask applied before softmax
        topk_values = sorted({min(t, max_seq_len) for t in (128, 1024, 2048)})

        # The sparse topkIndices path needs flash attention for long sequences
        # (the non-flash mla_v2 kernel rejects topK > 0 when max_seq_len >
        # MAX_SOFTMAX_PINGPONG_LEN); the dense path is independent of flash and
        # has no sequence-length limit, so it always runs for decode cases.
        run_sparse_topk = enable_flash or max_seq_len <= MAX_SOFTMAX_PINGPONG_LEN

        for topk in topk_values:
            # Generate random topk_indices for each sample in the batch
            # This simulates the indexer output that selects top-k positions
            # topk_indices is a flattened tensor of shape (total_query_len, topk)
            topk_indices_list = []
            for i in range(batch):
                qlen = query_len_list[i]
                clen = cached_lens_list[i]
                # For each query position, generate topk indices
                # Each query position q_idx can only attend to positions in [0, clen + q_idx]
                # due to causal mask constraint
                for q_idx in range(qlen):
                    valid_len = clen + q_idx + 1  # +1 because q_idx can attend to itself
                    perm = torch.randperm(valid_len, device="npu")
                    if valid_len < topk:
                        pad = perm.new_full((topk - valid_len,), valid_len)
                        perm = torch.cat([perm, pad])
                    else:
                        perm = perm[:topk]
                    topk_indices_list.append(perm)

            # Concatenate topk_indices into a flattened tensor
            # Shape: (total_query_len, topk)
            topk_indices_tensor = torch.stack(topk_indices_list)  # (total_query_len, topk)
            topk_indices_tensor, _ = torch.sort(topk_indices_tensor, dim=-1)
            topk_indices_tensor = topk_indices_tensor.to(dtype=torch.int32)

            # Standard MLA with topk mask: apply -inf mask before softmax
            # Produces o_absorb_standard_with_topk (stop at absorb) and
            # output_standard_with_topk (with WUV projection).
            o_absorbs_with_topk = []
            outputs_with_topk = []
            indices_offset = 0
            for i in range(batch):
                qlen = query_len_list[i]
                clen = cached_lens_list[i]

                # 复用第一个循环里已算好的 qkc/qkr（按 batch 索引）
                qkc = qkc_list[i]
                qkr = qkr_list[i]
                scores = qkc + qkr
                scores = scores * scale

                # Apply topk mask: -inf for positions not in topk_indices, 0 for topk positions
                total_len = clen + qlen
                # Get indices for this sample: stack indices for all query positions in this sample
                sample_indices = torch.stack(topk_indices_list[indices_offset : indices_offset + qlen])  # (qlen, topk)
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
                o_absorb = torch.einsum("bsht,btc->bshc", scores, k_cache[i : i + 1, : clen + qlen])
                o_absorbs_with_topk.append(o_absorb.squeeze(0))
                # output = o_absorb @ WUV — for mla comparison
                x = torch.einsum("bshc,hdc->bshd", o_absorb, wkvb[:, -v_head_dim:])
                outputs_with_topk.append(x.squeeze(0))

            o_absorb_standard_with_topk = torch.cat(o_absorbs_with_topk, dim=0)
            output_standard_with_topk = torch.cat(outputs_with_topk, dim=0)

            # ----- xlite sparse path: mla_v2 with topkIndices (paged KV cache,
            # -inf mask applied inside the kernel's softmax) -----
            if run_sparse_topk:
                output_xlite_with_topk_v2 = torch.zeros(
                    total_query_len, n_heads, v_head_dim, device="npu", dtype=test_dtype
                )

                torch.npu.synchronize()
                mla_v2(
                    rt,
                    qWithQr_xlite,
                    qr_xlite,
                    k_cache_xlite,
                    pe_cache_xlite,
                    wukT_xlite,
                    wuv_xlite,
                    output_xlite_with_topk_v2,
                    query_start_loc,
                    query_lens,
                    cached_lens,
                    block_tables,
                    n_heads,
                    rope_head_dim,
                    nope_head_dim,
                    v_head_dim,
                    kv_lora_rank,
                    BLOCK_SIZE,
                    batch,
                    scale,
                    topk_indices_tensor,
                    top_k=topk,
                    nz=weight_nz,
                    enable_flash_attention=enable_flash,
                    tile_size_of_cached_kv=tile_size,
                )

                logging.info(
                    "mla_v2 with topkIndices %s (%d heads, %d rope_head_dim, %d nope_head_dim, "
                    "%d v_head_dim, %d kv_lora_rank, %s) work (%d batch, cached_lens=%s, "
                    "query_lens=%s, topk=%d) executed!",
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
                    torch.testing.assert_close(
                        output_xlite_with_topk_v2, output_standard_with_topk, atol=1e-5, rtol=5e-02
                    )
                except AssertionError as e:
                    logging.error(f"mla_v2 with topkIndices test failed: {e}")
                    logging.error(f"Standard mla_v2 with topk mask: {output_standard_with_topk}")
                    logging.error(f"xlite mla_v2 with topkIndices: {output_xlite_with_topk_v2}")
            else:
                logging.info(
                    "mla_v2 with topkIndices %s work (%d batch, cached_lens=%s, "
                    "query_lens=%s, topk=%d) skipped: non-flash kernel rejects topk > 0 "
                    "when max_seq_len > %d",
                    name,
                    batch,
                    cached_lens_list,
                    query_len_list,
                    topk,
                    MAX_SOFTMAX_PINGPONG_LEN,
                )

            # ----- xlite dense mode: gather the same topk_indices into a
            # contiguous per-query-token dense cache, then run mla_v2 in dense
            # mode on it. Supports decode (query_len == 1) and small prefill
            # (query_len > 1); each query token owns its own dense cache row and
            # its own topk_indices row (already generated above with the causal
            # offset valid_len = clen + q_idx + 1).
            max_qlen = max(query_len_list)
            if max_qlen > MAX_DENSE_QUERY_LEN:
                logging.info(
                    "mla_v2 dense %s work (%d batch, cached_lens=%s, query_lens=%s, topk=%d) "
                    "skipped: dense mode requires max query_len <= %d",
                    name,
                    batch,
                    cached_lens_list,
                    query_len_list,
                    topk,
                    MAX_DENSE_QUERY_LEN,
                )
            else:
                output_xlite_dense = torch.zeros(total_query_len, n_heads, v_head_dim, device="npu", dtype=test_dtype)

                torch.npu.synchronize()
                mla_v2(
                    rt,
                    qWithQr_xlite,
                    qr_xlite,
                    k_cache_xlite,
                    pe_cache_xlite,
                    wukT_xlite,
                    wuv_xlite,
                    output_xlite_dense,
                    query_start_loc,
                    query_lens,
                    cached_lens,
                    block_tables,
                    n_heads,
                    rope_head_dim,
                    nope_head_dim,
                    v_head_dim,
                    kv_lora_rank,
                    BLOCK_SIZE,
                    batch,
                    scale,
                    topk_indices_tensor,
                    top_k=topk,
                    nz=weight_nz,
                    enable_flash_attention=enable_flash,
                    tile_size_of_cached_kv=tile_size,
                    dense=True,
                )

                logging.info(
                    "mla_v2 dense %s (%d heads, %d rope_head_dim, %d nope_head_dim, %d "
                    "v_head_dim, %d kv_lora_rank, %s) work (%d batch, cached_lens=%s, "
                    "query_lens=%s, topk=%d) executed!",
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
                    torch.testing.assert_close(output_xlite_dense, output_standard_with_topk, atol=1e-5, rtol=5e-02)
                except AssertionError as e:
                    logging.error(f"mla_v2 dense test failed: {e}")
                    logging.error(f"Standard mla_v2 with topk mask: {output_standard_with_topk}")
                    logging.error(f"xlite mla_v2 dense: {output_xlite_dense}")
