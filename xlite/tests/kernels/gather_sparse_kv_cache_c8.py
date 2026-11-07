#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Test SFA-C8 sparse KV gather against a dequantized reference."""
import logging

import torch
import torch_npu
from xlite._C import Runtime, gather_sparse_kv_cache

npu_id = 0


def quantize(latent, pe):
    latent_dim = latent.shape[-1]
    q, scale = torch_npu.npu_dynamic_block_quant(
        latent.view(-1, 1, latent_dim), dst_type=torch.int8, row_block_size=1, col_block_size=128)
    return q.reshape(-1, latent_dim), pe, scale.reshape(-1, latent_dim // 128)


def dequantize(cache):
    q, pe, scale = cache
    return (q.float() * scale.repeat_interleave(128, -1)).bfloat16(), pe


def test_gather(rt, block_size, lengths, topk, latent_dim=512, rope_dim=64, query_lens=None,
                burst_copy_size=16 * 1024, sequential=False):
    batch = len(lengths)
    if query_lens is None:
        query_lens = [1] * batch
    tokens = sum(query_lens)
    width = (max(lengths) + block_size - 1) // block_size
    count = batch * width * block_size
    latent = torch.randn(count, latent_dim, dtype=torch.bfloat16, device="npu")
    pe = torch.randn(count, rope_dim, dtype=torch.bfloat16, device="npu")
    cache_bytes = latent_dim + rope_dim * 2 + (latent_dim // 128) * 4
    cache = quantize(latent, pe)
    decoded, pe = dequantize(cache)
    cache = [t.view(-1, block_size, 1, t.shape[-1]) for t in cache]
    table = torch.randperm(batch * width, device="npu").view(batch, width).int()
    if sequential:
        indices = torch.empty(0, dtype=torch.int32, device="npu")
    else:
        indices = torch.full((tokens, topk), -1, dtype=torch.int32, device="npu")
    rows = batch if sequential else tokens
    k_out = torch.ones(rows, topk, latent_dim, dtype=torch.bfloat16, device="npu")
    pe_out = torch.ones(rows, topk, rope_dim, dtype=torch.bfloat16, device="npu")
    expected_k = k_out.clone()
    expected_pe = pe_out.clone()
    sparse_rows = []
    offset = 0
    for b, (length, qlen) in enumerate(zip(lengths, query_lens)):
        for q in range(1 if sequential else qlen):
            row = b if sequential else offset + q
            visible = length if sequential else length - qlen + q + 1
            sparse = not sequential and visible > topk
            sparse_rows.append(sparse)
            if sparse:
                ids = torch.arange(visible - 1, visible - topk - 1, -1, device="npu")
                indices[row] = ids.int()
            else:
                ids = torch.arange(min(visible, topk), device="npu")
            slots = table[b, ids // block_size].long() * block_size + ids % block_size
            expected_k[row, :len(ids)] = decoded[slots]
            expected_pe[row, :len(ids)] = pe[slots]
        offset += qlen
    lens = torch.tensor(query_lens, dtype=torch.int32, device="npu")
    starts = (lens.cumsum(0) - lens).int()
    cached_lens = torch.tensor(lengths, dtype=torch.int32, device="npu") - lens
    torch.npu.synchronize()
    gather_sparse_kv_cache(rt, cache[0], cache[1], table, indices,
                           starts, lens, cached_lens, k_out, pe_out, batch, topk, block_size,
                           latent_dim, rope_dim, 1, burst_copy_size=burst_copy_size,
                           k_scale_cache=cache[2])
    actual = torch.cat((k_out, pe_out), dim=-1).cpu()
    expected = torch.cat((expected_k, expected_pe), dim=-1).cpu()
    for row, sparse in enumerate(sparse_rows):
        if sparse and burst_copy_size > cache_bytes:
            a = sorted(r.view(torch.uint8).numpy().tobytes() for r in actual[row])
            b = sorted(r.view(torch.uint8).numpy().tobytes() for r in expected[row])
            assert a == b, ("gathered KV rows", row)
        else:
            torch.testing.assert_close(actual[row], expected[row], rtol=0, atol=0)
    logging.info("SFA-C8 gather block=%d lengths=%s dims=%d/%d burst=%d: passed",
                 block_size, lengths, latent_dim, rope_dim, burst_copy_size)


def main():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(20261003)
    torch.npu.set_device(npu_id)
    rt = Runtime(npu_id, 512)
    for block_size in (16, 128):
        test_gather(rt, block_size, [47, 129, 2049], 2048)
    test_gather(rt, 128, [47, 129, 2049], 2048, burst_copy_size=512)
    test_gather(rt, 128, [47, 129, 2051], 2048, query_lens=[3, 1, 4])
    for latent_dim, rope_dim in ((128, 32), (256, 64), (384, 64), (512, 128)):
        test_gather(rt, 16, [47, 130], 129, latent_dim, rope_dim)
    test_gather(rt, 128, [47, 129, 2051], 2048,
                query_lens=[3, 1, 4], sequential=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
