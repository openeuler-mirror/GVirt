#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""cxa kernel test (unified).

Each (model, work) case runs the `cxa` interface against the reference
implementation `sparse_attn` in tests/models/deepseek_v4_kernel.py.

CXA (C4A and C128A) fuses a sliding-window (SWA) segment and a
compressed sparse segment into one softmax. The reference `sparse_attn` models
the same thing as a single gather+softmax over a concatenated
``[window_size + compressed_kv_len]`` KV matrix:

    kv          : [b, window_size + compressed_kv_len, d]   (SWA | compress)
    topk_idxs   : [b, s, topk] int32   (-1 = masked-out position)
    attn_sink   : [h] fp32, learnable sink bias folded into softmax denominator
    o           : [b, s, h, d] bf16

The xlite `cxa` op splits that KV matrix across two paged caches
(`swa_k_cache` + `compress_k_cache`), each with its own block table, and feeds
the per-query top-k indices (``-1`` masked) into `topk_indices`.

Equivalence is established by making **both sides read the same token data**:
the SWA tokens laid out in `swa_k_cache` (expanded via `swa_block_tables`)
equal the leading `window_size` columns of `sparse_attn`'s `kv`, and the
compressed tokens in `compress_k_cache` (expanded via
`compress_block_tables`) equal the trailing columns. That way the kernel's
internal layout / wrap-around convention for the SWA cache does not matter --
both sides attend over the same per-query KV, so outputs are directly
comparable.

"""
from __future__ import absolute_import

import logging
import math
import numpy as np
import torch
from xlite._C import Runtime, cxa
from xlite._C import print as xlite_print
from tests.models.deepseek_v4_kernel import sparse_attn

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 3000)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
MAX_SOFTMAX_PINGPONG_LEN = 11776

# block sizes (per-cache, per the CXA 5-tuple; here we pick)
swa_block_size = 64
compress_block_size = 128

# model configurations:
#   name, n_heads, head_dim, window_size, compress_ratio, index_topk, dtype
models = [
    ("base-swa", 1, 512, 128, 0, 0, torch.bfloat16),
    ("swa", 16, 512, 128, 0, 0, torch.bfloat16),
    ("c128a", 16, 512, 128, 128, 0, torch.bfloat16),
    ("c4a-ntok", 16, 512, 128, 4, 0, torch.bfloat16),
    ("c4a-topk", 16, 512, 128, 4, 512, torch.bfloat16),
]

# work configurations: batch, cached_lens, query_lens
work = [
    (1, [0], [1]),
    (1, [0], [30]),
    (1, [0], [77]),
    (1, [0], [128]),
    (1, [0], [129]),
    (1, [0], [256]),
    (1, [0], [1600]),
    (8, [0] * 8, [789, 65, 13, 6545, 24, 190, 2432, 124]),
    (2, [4000] * 2, [1] * 2),
    (2, [6000] * 2, [1] * 2),
    (2, [131071] * 2, [1] * 2),
    (5, [8, 13, 65, 11, 5], [1] * 5),
    (1, [8192], [1024]),
    (1, [128], [128]),
    (1, [129], [129]),
    (1, [143], [1]),
]


def max_blocks(lengths, block_size):
    """Number of blocks needed to hold max(lengths) tokens."""
    max_len = max(lengths)
    return (max_len + block_size - 1) // block_size


def build_swa_topk(window_size, total_len, q_len, device):
    """Window-segment top-k indices for one sample.

    Mirrors get_window_topk_idxs(start_pos=total_len-q_len) semantics: for a
    query at row r (0..q_len-1), the absolute position is (total_len - q_len +
    r). The visible window is [max(0, pos - window_size + 1), pos]. Positions
    that have not been generated yet (beyond total_len) are masked with -1.

    Returns: (q_len, window_size) int32, indices into the SWA segment
    [0, window_size) of the concatenated KV.
    """
    base_pos = total_len - q_len  # absolute position of query row 0
    rows = torch.arange(q_len, device=device).unsqueeze(1)          # (q_len, 1)
    pos = base_pos + rows                                         # (q_len, 1) abs pos
    # visible window start for each row
    win_start = (pos - window_size + 1).clamp(min=0)               # (q_len, 1)
    # columns: offsets 0..window_size-1 relative to win_start
    cols = torch.arange(window_size, device=device).unsqueeze(0)  # (1, window_size)
    abs_idx = win_start + cols                                    # (q_len, window_size)
    invalid = (abs_idx >= total_len) | (abs_idx < 0) | (abs_idx > pos)
    abs_idx = abs_idx.clamp(min=0)
    return torch.where(invalid, torch.full_like(abs_idx, -1), abs_idx).to(torch.int32)


def build_compress_topk(compress_ratio, compressed_kv_len, total_len, q_len, window_size, device):
    """Compress-segment top-k indices for one sample.

    Mirrors get_compress_topk_idxs(start_pos=total_len-q_len) semantics: a
    query at row r can attend to compressed tokens with global index <
    (abs_pos + 1) // compress_ratio, where abs_pos = total_len - q_len + r.
    Indices are offset by total_len so they land in the compress segment of
    the concatenated KV ([SWA(total_len rows) | compress]) that sparse_attn
    gathers from. (deepseek_v4 uses offset = kv.size(1) == seqlen on prefill.)

    The returned count is min(index_topk, compressed_kv_len); padding to
    index_topk (with -1) happens at the caller.
    """
    base_pos = total_len - q_len
    rows = torch.arange(q_len, device=device).unsqueeze(1)          # (q_len, 1)
    abs_pos = base_pos + rows                                      # (q_len, 1)
    # number of compressed tokens visible to row r (causal)
    n_visible = (abs_pos + 1) // compress_ratio                   # (q_len, 1)
    n_visible = n_visible.clamp(min=0)
    max_k = compressed_kv_len
    # build a (q_len, max_k) matrix of compressed-token indices
    cols = torch.arange(max_k, device=device).unsqueeze(0)         # (1, max_k)
    comp_idx = cols                                               # global compressed idx
    invalid = cols >= n_visible                                   # (q_len, max_k)
    comp_idx = comp_idx.clamp(min=0)
    # offset into the concatenated KV: compress segment starts at total_len
    comp_idx = comp_idx + total_len
    return torch.where(invalid, torch.full_like(comp_idx, -1), comp_idx).to(torch.int32)


for name, n_heads, head_dim, window_size, compress_ratio, index_topk, test_dtype in models:
    scale = head_dim ** -0.5
    for batch, cached_lens_list, query_len_list in work:
        total_lens = [cl + ql for cl, ql in zip(cached_lens_list, query_len_list)]
        max_total_len = max(total_lens)

        if index_topk > 0 and compress_ratio > 0:
            swa_seg_width = 0 if window_size == 0 else window_size + 128 + 16
            max_row = max(
                swa_seg_width + -(-(tl // compress_ratio) // 256) * 256
                for tl in total_lens
            )
            max_row = max(max_row, index_topk + swa_seg_width)
            if max_row > MAX_SOFTMAX_PINGPONG_LEN:
                logging.info(
                    "cxa %s SKIP work (batch %d, cached_lens=%s, query_lens=%s): "
                    "topk path row width %d exceeds PingPong UB budget MAX_SOFTMAX_PINGPONG_LEN",
                    name, batch, cached_lens_list, query_len_list, max_row)
                continue

        # compressed KV length per sample (== total_len // compress_ratio).
        # kv_size (compress-segment capacity) is derived INSIDE the op from the compress
        if compress_ratio > 0:
            compressed_kv_lens = [tl // compress_ratio for tl in total_lens]
            kv_size = max_total_len // compress_ratio
        else:
            compressed_kv_lens = [0] * batch
            kv_size = 0
        total_query_len = sum(query_len_list)

        swa_num_blocks_per_sample = (max_total_len + swa_block_size - 1) // swa_block_size
        swa_max_num_blocks = swa_num_blocks_per_sample
        compress_num_blocks_per_sample = (kv_size + compress_block_size - 1) // compress_block_size if kv_size > 0 else 0
        compress_max_num_blocks = compress_num_blocks_per_sample

        torch.set_default_dtype(test_dtype)
        with torch.device("npu"):
            # ---- reference: build sparse_attn inputs per sample ----
            # q_standard: [total_query_len, n_heads, head_dim]
            q_standard = torch.randn(total_query_len, n_heads, head_dim)
            # attn_sink: [n_heads] fp32 (learnable per-head sink bias)
            attn_sink = torch.randn(n_heads, dtype=torch.float32) * 0.1

            # per-sample KV: [b, window_size + compressed_kv_len_b, head_dim]
            # We build the full token data once and share between reference and xlite.
            kv_list = []
            for i in range(batch):
                n_compress = compressed_kv_lens[i]
                swa_full = torch.randn(total_lens[i], head_dim)
                # compress segment: n_compress compressed tokens
                if n_compress > 0:
                    comp_part = torch.randn(n_compress, head_dim)
                    kv_i = torch.cat([swa_full, comp_part], dim=0)
                else:
                    kv_i = swa_full
                kv_list.append(kv_i)

            # ---- topk indices per sample (two segments) ----
            topk_indices_list = []
            for i in range(batch):
                tl = total_lens[i]
                ql = query_len_list[i]
                swa_topk = build_swa_topk(window_size, tl, ql, "npu")  # (ql, window_size)
                if compress_ratio > 0 and kv_size > 0:
                    # number of compress indices this sample contributes
                    n_comp = compressed_kv_lens[i]
                    comp_topk = build_compress_topk(
                        compress_ratio, n_comp, tl, ql, window_size, "npu")  # (ql, n_comp)
                    # Sparse top-k path: when a row sees more than index_topk visible
                    # compress tokens, select a random subset so gather/scatter offsets
                    # span the whole compress segment, and hand the SAME subset to both
                    # the reference and the kernel. Rows with calcLen <= index_topk take
                    # the kernel's full-causal path and keep all visible entries.
                    if index_topk > 0:
                        base_pos = tl - ql
                        n_visible_np = (np.arange(ql) + base_pos + 1) // compress_ratio
                        n_visible_np = np.clip(n_visible_np, 0, n_comp)
                        calc_len_np = base_pos + np.arange(ql) + 1  # causal kv len per row
                        sel_np = np.full((ql, index_topk), -1, dtype=np.int64)
                        for r in range(ql):
                            nv = int(n_visible_np[r])
                            if nv > index_topk and int(calc_len_np[r]) > index_topk:
                                chosen = np.sort(np.random.choice(
                                    nv, index_topk, replace=False))
                            else:
                                chosen = np.arange(min(nv, index_topk))
                            sel_np[r, :len(chosen)] = chosen
                        # offset into the concatenated KV [SWA(total_len) | compress]
                        sel_np[sel_np >= 0] += tl
                        comp_topk = torch.from_numpy(sel_np).to(torch.int32).to("npu")
                    else:
                        comp_topk_len = n_comp  # full causal, no padding needed
                    sample_topk = torch.cat([swa_topk, comp_topk], dim=1)  # (ql, window_size + comp_topk_len)
                else:
                    # no compress segment; pad compress part with -1 so total width matches
                    comp_pad = torch.full((ql, index_topk), -1,
                                          dtype=torch.int32, device="npu")
                    sample_topk = torch.cat([swa_topk, comp_pad], dim=1)
                topk_indices_list.append(sample_topk)

            # ---- run reference sparse_attn per sample ----
            ref_outputs = []
            q_offset = 0
            for i in range(batch):
                ql = query_len_list[i]
                q_chunk = q_standard[q_offset:q_offset + ql].unsqueeze(0)  # (1, ql, h, d)
                q_offset += ql
                # sparse_attn expects [b, s, h, d]; kv [b, n, d]; topk [b, s, topk]
                o = sparse_attn(q_chunk.float(), kv_list[i].unsqueeze(0).float(),
                                attn_sink, topk_indices_list[i].unsqueeze(0), scale).to(test_dtype)
                ref_outputs.append(o.squeeze(0))  # (ql, h, d)
            output_standard = torch.cat(ref_outputs, dim=0)  # (total_query_len, h, d)

            # ---- xlite side: build paged caches reading the SAME token data ----
            # SWA paged cache: swa_num_blocks_per_sample blocks, each swa_block_size.
            swa_cache_block_num = swa_max_num_blocks * batch
            swa_k_cache = torch.zeros(swa_cache_block_num, swa_block_size, head_dim)
            # compress paged cache
            if kv_size > 0:
                compress_cache_block_num = compress_max_num_blocks * batch
                compress_k_cache = torch.zeros(compress_cache_block_num, compress_block_size, head_dim)
            else:
                compress_k_cache = torch.zeros(1, compress_block_size, head_dim)  # dummy, unused

            swa_abs_blocks = (max_total_len + swa_block_size - 1) // swa_block_size
            batch_indices_swa = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
            abs_indices_swa = np.arange(swa_abs_blocks, dtype=np.uint32).reshape(1, -1)
            swa_block_tables_array = (batch_indices_swa * swa_max_num_blocks + abs_indices_swa)
            swa_block_tables = torch.tensor(swa_block_tables_array.tolist(),
                                            dtype=torch.int32).flatten()

            if kv_size > 0:
                batch_indices_comp = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
                block_indices_comp = np.arange(compress_max_num_blocks, dtype=np.uint32)
                compress_block_tables_array = (batch_indices_comp * compress_max_num_blocks
                                               + block_indices_comp)
                compress_block_tables = torch.tensor(compress_block_tables_array.tolist(),
                                                     dtype=torch.int32).flatten()
            else:
                compress_block_tables = torch.zeros(batch, dtype=torch.int32)

            for i in range(batch):
                tl = total_lens[i]
                swa_full = kv_list[i][:tl]  # absolute positions [0, tl)
                sample_block_start = i * swa_max_num_blocks
                for p in range(0, tl):
                    block_id = p // swa_block_size
                    offset = p % swa_block_size
                    phys = sample_block_start + block_id
                    swa_k_cache[phys, offset] = swa_full[p]

            for i in range(batch):
                n_compress = compressed_kv_lens[i]
                if n_compress == 0:
                    continue
                # compress content is kv_list[i][total_len:total_len+n_compress]
                tl = total_lens[i]
                comp_content = kv_list[i][tl:tl + n_compress]
                sample_block_start = i * compress_max_num_blocks
                for j in range(n_compress):
                    block_id = j // compress_block_size
                    offset = j % compress_block_size
                    phys = sample_block_start + block_id
                    compress_k_cache[phys, offset] = comp_content[j]

            # ---- query/cached length tensors ----
            query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
            cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
            query_lens_np = np.array(query_len_list)
            query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
            query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

            # ---- build topk_indices tensor for the op ----
            op_topk_width = index_topk
            op_topk_list = []
            if op_topk_width == 0:
                # full-causal path never reads topk_indices; empty placeholder
                topk_indices_op = torch.zeros(total_query_len, 0, dtype=torch.int32)
            for i in range(batch):
                ql = query_len_list[i]
                tl = total_lens[i]
                # Convert coordinates for the op: the reference's compress indices are
                # absolute (total_len + comp_idx) into [SWA(total_len) | compress], while
                # the op expects compress-relative coordinates (compressed token j ->
                # index j; the kernel places it at score column swaSegWidth + j, and the
                # dense-mode gather resolves it through compress_block_tables). -1 stays -1.
                comp_part = topk_indices_list[i][:, window_size:].clone()  # (ql, comp_topk_len)
                mask_neg = comp_part < 0
                comp_part = comp_part - tl
                comp_part = torch.where(mask_neg, torch.full_like(comp_part, -1), comp_part)
                if comp_part.shape[1] < op_topk_width:
                    pad = torch.full((ql, op_topk_width - comp_part.shape[1]), -1,
                                     dtype=torch.int32, device="npu")
                    comp_part = torch.cat([comp_part, pad], dim=1)
                op_topk_list.append(comp_part)
            if op_topk_width > 0:
                topk_indices_op = torch.cat(op_topk_list, dim=0).to(torch.int32)  # (total_q, index_topk)
                assert topk_indices_op.shape[1] == op_topk_width

            # output buffer
            output_xlite = torch.zeros(total_query_len, n_heads, head_dim)

        # ----- run cxa kernel -----
        case_desc = (
            f"cxa {name} ({n_heads} heads, {head_dim} head_dim, {test_dtype}) "
            f"work ({batch} batch, cached_lens={cached_lens_list}, "
            f"query_lens={query_len_list}, window={window_size}, ratio={compress_ratio}, "
            f"kv_size={kv_size}, index_topk={index_topk})"
        )
        try:
            torch.npu.synchronize()
            cxa(rt, q_standard, swa_k_cache, compress_k_cache, swa_block_tables,
                compress_block_tables, swa_block_size, compress_block_size, attn_sink,
                output_xlite, batch, query_start_loc, query_lens, cached_lens, n_heads,
                head_dim, scale, window_size, compress_ratio, index_topk,
                topk_indices_op)
            torch.npu.synchronize()
        except Exception as e:
            logging.error(f'{case_desc} KERNEL FAILED: {e}')
            continue

        logging.info("%s executed!", case_desc)

        try:
            torch.testing.assert_close(output_xlite, output_standard, atol=5e-5, rtol=5e-02)
        except AssertionError as e:
            logging.error(f'cxa mismatch: {e}')
            logging.error(f'sparse_attn ref: {output_standard}')
            logging.error(f'xlite cxa:        {output_xlite}')

        # ----- xlite dense mode: the selected compressed tokens are gathered into a
        # per-batch contiguous cache (batch b holds index_topk compressed tokens at
        # [b * index_topk, (b + 1) * index_topk)), no block table inside the kernel,
        # no top-k masking -- the compress segment is attended causally, clamped to
        # index_topk. The reference gathers the SAME sampled selection, so both sides
        # attend over the identical token set. SWA stays paged (unchanged caches).
        if compress_ratio == 0 or index_topk == 0:
            continue

        # ----- xlite dense mode: gather the same topk_indices into a contiguous dense
        # cache, then run cxa in dense mode on it. The gather kernel resolves each
        # index in topk_indices_op through compress_block_tables, so the dense cache
        # holds exactly the sampled top-k tokens (a random subset of [0, n_visible)
        # when n_visible > index_topk). The reference therefore attends over the SAME
        # subset: reuse topk_indices_list[i] verbatim. SWA stays paged (unchanged).
        if any(qlen != 1 for qlen in query_len_list):
            logging.info(
                "cxa dense %s work (%d batch, cached_lens=%s, query_lens=%s, topk=%d) "
                "skipped: dense mode requires query_len == 1 per batch",
                name, batch, cached_lens_list, query_len_list, index_topk,
            )
            continue

        # reference with the same sampled compress selection the kernel gathers
        dense_ref_outputs = []
        q_offset = 0
        for i in range(batch):
            ql = query_len_list[i]
            q_chunk = q_standard[q_offset:q_offset + ql].unsqueeze(0)
            q_offset += ql
            o = sparse_attn(q_chunk.float(), kv_list[i].unsqueeze(0).float(),
                            attn_sink, topk_indices_list[i].unsqueeze(0), scale).to(test_dtype)
            dense_ref_outputs.append(o.squeeze(0))
        output_dense_standard = torch.cat(dense_ref_outputs, dim=0)

        output_xlite_dense = torch.zeros(total_query_len, n_heads, head_dim, device="npu")

        try:
            torch.npu.synchronize()
            cxa(rt, q_standard, swa_k_cache, compress_k_cache, swa_block_tables,
                compress_block_tables, swa_block_size, compress_block_size, attn_sink,
                output_xlite_dense, batch, query_start_loc, query_lens, cached_lens, n_heads,
                head_dim, scale, window_size, compress_ratio, index_topk,
                topk_indices_op, dense=True)
            torch.npu.synchronize()
        except Exception as e:
            logging.error(f'{case_desc} DENSE KERNEL FAILED: {e}')
            continue

        logging.info("%s dense executed!", case_desc)

        try:
            torch.testing.assert_close(output_xlite_dense, output_dense_standard, atol=5e-05,
                                       rtol=5e-02)
        except AssertionError as e:
            logging.error(f'cxa dense mismatch: {e}')
            logging.error(f'sparse_attn dense ref: {output_dense_standard}')
            logging.error(f'xlite cxa dense:        {output_xlite_dense}')

