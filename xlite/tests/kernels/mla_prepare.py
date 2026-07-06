#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import
import logging
import torch
from xlite._C import Runtime, mla_prepare

logging.getLogger().setLevel(logging.INFO)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """[cos(t,0), cos(t,2), ... | sin(t,0), sin(t,2), ...] layout (TTTWWW)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: [bsz, seqlen, rope_dim]; freqs_cis: [seqlen, rope_dim/2]; returns [bsz, seqlen, rope_dim] TTTWWW."""
    dtype = x.dtype
    bsz, seqlen, rope_dim = x.shape
    x = torch.view_as_complex(x.float().view(bsz, seqlen, rope_dim // 2, 2))  # [bsz, seqlen, rope_dim/2]
    freqs_cis = freqs_cis.view(1, seqlen, rope_dim // 2)  # [1, seqlen, rope_dim/2]
    y = torch.view_as_real(x * freqs_cis)  # [bsz, seqlen, rope_dim/2, 2]
    y = torch.cat([y[..., 0], y[..., 1]], dim=-1)  # [bsz, seqlen, rope_dim]
    return y.to(dtype)


def stand_rmsnorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight.float() * x
    x = x + bias.float()
    return x.to(input_dtype)


rt = Runtime(0, 500)
torch.npu.set_device(0)

ROPE_THETA = 10000.0
MAX_SEQ_LEN = 1024
BATCH_SIZE = 8
SEQ_LEN = 10
BLOCK_SIZE = 128
BLOCK_NUM = 1
NORM_EPS = 1e-6

test_cases = {
    (torch.float16, 2048, 512, 64),
    (torch.bfloat16, 2048, 512, 64),
}

for test_dtype, q_lora_rank, kv_lora_rank, rope_head_dim in test_cases:
    num_tokens = BATCH_SIZE * SEQ_LEN
    total_dim = q_lora_rank + kv_lora_rank + rope_head_dim

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        attn_qkvc = torch.randn(num_tokens, total_dim) / 10
        attn_qkvc_ref = attn_qkvc.clone()

        q_norm = torch.randn(q_lora_rank) / 10
        q_norm_bias = torch.randn(q_lora_rank) / 10
        kv_norm = torch.randn(kv_lora_rank) / 10
        kv_norm_bias = torch.randn(kv_lora_rank) / 10

        attn_norm_qc = torch.zeros(num_tokens, q_lora_rank, dtype=test_dtype)
        attn_norm_kvc = torch.zeros(num_tokens, kv_lora_rank, dtype=test_dtype)

        freqs_cis = precompute_freqs_cis(rope_head_dim, MAX_SEQ_LEN, ROPE_THETA)[0:SEQ_LEN]
        position = torch.arange(SEQ_LEN, dtype=torch.int64).repeat(BATCH_SIZE)

        k_cache = torch.zeros(BLOCK_NUM, BLOCK_SIZE, kv_lora_rank, dtype=test_dtype)
        pe_cache = torch.zeros(BLOCK_NUM, BLOCK_SIZE, rope_head_dim, dtype=test_dtype)

        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

    # ---- reference ----
    q_slice = attn_qkvc_ref[:, :q_lora_rank]
    kv_slice = attn_qkvc_ref[:, q_lora_rank:q_lora_rank + kv_lora_rank]
    pe_slice = attn_qkvc_ref[:, q_lora_rank + kv_lora_rank:]

    attn_norm_qc_ref = stand_rmsnorm(q_slice, q_norm, q_norm_bias, NORM_EPS)
    attn_norm_kvc_ref = stand_rmsnorm(kv_slice, kv_norm, kv_norm_bias, NORM_EPS)

    pe_reshaped = pe_slice.view(BATCH_SIZE, SEQ_LEN, rope_head_dim)
    pe_rot = apply_rotary_emb(pe_reshaped, freqs_cis)
    pe_rot = pe_rot.view(num_tokens, rope_head_dim)
    attn_qkvc_ref[:, q_lora_rank + kv_lora_rank:] = pe_rot

    k_cache_ref = torch.zeros(BLOCK_NUM, BLOCK_SIZE, kv_lora_rank, dtype=test_dtype, device="npu")
    pe_cache_ref = torch.zeros(BLOCK_NUM, BLOCK_SIZE, rope_head_dim, dtype=test_dtype, device="npu")
    for i in range(num_tokens):
        slot = int(slot_mapping[i].item())
        b = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        k_cache_ref[b, off] = attn_norm_kvc_ref[i]
        pe_cache_ref[b, off] = pe_rot[i]

    # ---- xlite ----
    torch.npu.synchronize()
    mla_prepare(rt, attn_qkvc, q_norm, q_norm_bias, attn_norm_qc, kv_norm, kv_norm_bias,
                attn_norm_kvc, freqs_cis, position, q_lora_rank, kv_lora_rank, rope_head_dim,
                BLOCK_SIZE, k_cache, pe_cache, slot_mapping, NORM_EPS)
    torch.npu.synchronize()

    logging.info(f'mla_prepare (q={q_lora_rank}, kv={kv_lora_rank}, rope={rope_head_dim}, '
                 f'{test_dtype}) executed!')

    try:
        torch.testing.assert_close(attn_norm_qc_ref, attn_norm_qc, atol=2e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'attn_norm_qc mismatch ({test_dtype}): {e}')

    try:
        torch.testing.assert_close(attn_norm_kvc_ref, attn_norm_kvc, atol=2e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'attn_norm_kvc mismatch ({test_dtype}): {e}')

    try:
        torch.testing.assert_close(attn_qkvc_ref, attn_qkvc, atol=2e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'attn_qkvc (pe slice) mismatch ({test_dtype}): {e}')

    try:
        torch.testing.assert_close(k_cache_ref, k_cache, atol=2e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'k_cache mismatch ({test_dtype}): {e}')

    try:
        torch.testing.assert_close(pe_cache_ref, pe_cache, atol=2e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'pe_cache mismatch ({test_dtype}): {e}')
