#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""SFA-C8 attention compared with the same dequantized BF16 cache."""

import logging

import torch
import torch_npu
from gather_sparse_kv_cache_c8 import dequantize, quantize

from xlite._C import Runtime, mla_v2

npu_id = 0
tile_size = 8192

MAX_SOFTMAX_PINGPONG_LEN = 11776
MAX_DENSE_QUERY_LEN = 6


def run_pair(
    rt, q, qr, caches, table, indices, query_lens, cached_lens, wuk, wuv, topk, flash, dense=False, kv_tile=tile_size
):
    heads, nope_dim, latent_dim = wuk.shape
    rope_dim = qr.shape[-1]
    block_size = caches[0].shape[1]
    batch = len(query_lens)
    k, pe = dequantize(caches)
    lens = torch.tensor(query_lens, dtype=torch.int32, device="npu")
    cached = torch.tensor(cached_lens, dtype=torch.int32, device="npu")
    starts = (lens.cumsum(0) - lens).int()
    out = torch.empty(len(q), heads, wuv.shape[-1], dtype=torch.bfloat16, device="npu")
    expected = torch.empty_like(out)
    for cache, rope, scale_cache, output in ((k, pe, None, expected), (*caches, out)):
        torch.npu.synchronize()
        mla_v2(
            rt,
            q,
            qr,
            cache,
            rope,
            wuk,
            wuv,
            output,
            starts,
            lens,
            cached,
            table,
            heads,
            rope_dim,
            nope_dim,
            wuv.shape[-1],
            latent_dim,
            block_size,
            batch,
            (nope_dim + rope_dim) ** -0.5,
            indices,
            top_k=topk,
            nz=False,
            enable_flash_attention=flash,
            tile_size_of_cached_kv=kv_tile,
            dense=dense,
            k_scale_cache=scale_cache,
        )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert bool(torch.isfinite(out).all())


def test_attention(
    rt,
    query_lens,
    cached_lens,
    heads=8,
    block_size=128,
    topk=2048,
    latent_dim=512,
    rope_dim=64,
    kv_tile=tile_size,
    late_indices=False,
):
    batch, tokens = len(query_lens), sum(query_lens)
    total_lens = [q + c for q, c in zip(query_lens, cached_lens)]
    width = (max(total_lens) + block_size - 1) // block_size
    capacity = batch * width * block_size
    latent = torch.randn(capacity, latent_dim, dtype=torch.bfloat16, device="npu")
    pe = torch.randn(capacity, rope_dim, dtype=torch.bfloat16, device="npu")
    caches = [t.view(-1, block_size, 1, t.shape[-1]) for t in quantize(latent, pe)]
    table = torch.randperm(batch * width, device="npu").view(batch, width).int()
    q = torch.randn(tokens, heads, 192 + rope_dim, dtype=torch.bfloat16, device="npu")
    qr = q[..., 192:].contiguous()
    wuk = (torch.randn(heads, 192, latent_dim, device="npu") / 192**0.5).bfloat16()
    wuv = (torch.randn(heads, latent_dim, 256, device="npu") / latent_dim**0.5).bfloat16()
    indices = torch.full((tokens, topk), -1, dtype=torch.int32, device="npu")
    offset = 0
    for qlen, clen in zip(query_lens, cached_lens):
        if topk:
            for q_idx in range(qlen):
                visible = clen + q_idx + 1
                if visible > topk:
                    selected = (
                        torch.arange(visible - topk, visible, device="npu")
                        if late_indices
                        else torch.randperm(visible, device="npu")[:topk]
                    )
                    selected, _ = torch.sort(selected)
                else:
                    selected = torch.arange(topk, device="npu")
                indices[offset + q_idx] = selected.int()
        offset += qlen
    modes = [(False, False)] if topk == 0 or max(total_lens) <= MAX_SOFTMAX_PINGPONG_LEN else []
    modes.append((True, False))
    if topk and max(query_lens) <= MAX_DENSE_QUERY_LEN:
        modes.append((False, True))
    for flash, dense in modes:
        run_pair(rt, q, qr, caches, table, indices, query_lens, cached_lens, wuk, wuv, topk, flash, dense, kv_tile)
        logging.info(
            "SFA-C8 attention B=%d Q=%s KV=%s H=%d block=%d dims=%d/%d flash=%s dense=%s tile=%d: passed",
            batch,
            query_lens[:8],
            total_lens[:8],
            heads,
            block_size,
            latent_dim,
            rope_dim,
            flash,
            dense,
            kv_tile,
        )


def main():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(20261003)
    torch.npu.set_device(npu_id)
    rt = Runtime(npu_id, size=3000)
    cases = [
        ([64], [0], 8, 128, 0),
        ([128], [0], 4, 16, 2048),
        ([4], [2045], 8, 128, 2048),
        ([3, 1], [257, 4096], 8, 128, 2048),
        ([128], [8192], 8, 128, 2048),
        ([128], [8192], 8, 128, 0),
        ([4], [16384], 4, 128, 0),
        ([1], [4096], 64, 128, 2048),
        ([1] * 32, [256] * 32, 4, 128, 2048),
        ([1] * 64, [4096] * 64, 4, 128, 2048),
        ([1] * 128, [128] * 128, 4, 128, 2048),
        ([128] * 4, [4096] * 4, 4, 128, 2048),
        ([1024] * 4, [3072] * 4, 4, 128, 2048),
        ([512] * 4, [8192] * 4, 8, 128, 2048),
        ([512] * 4, [16384] * 4, 8, 128, 2048),
        ([33, 129, 7, 64, 17], [2041, 8192, 4096, 0, 257], 4, 128, 2048),
    ]
    for case in cases:
        test_attention(rt, *case)
    for latent_dim, rope_dim in ((256, 64), (512, 128)):
        for query_lens in ([17, 3], [1, 1]):
            test_attention(rt, query_lens, [257, 129], heads=4, topk=128, latent_dim=latent_dim, rope_dim=rope_dim)
    for block_size, kv_tile in ((128, 8192), (16, 4112), (64, 4160)):
        for topk in (0, 2048):
            test_attention(rt, [3, 2], [8192, 16384], block_size=block_size, topk=topk, kv_tile=kv_tile)
    for heads in (1, 3, 8, 64):
        test_attention(rt, [17, 7], [8192, 2041], heads=heads, topk=128, late_indices=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
