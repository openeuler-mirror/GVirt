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
import math
import torch
from xlite._C import Runtime, rope_complex

logging.getLogger().setLevel(logging.INFO)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.
    Based on tests/models/deepseek_v3.py precompute_freqs_cis function.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    Based on tests/models/deepseek_v3.py apply_rotary_emb function.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


rt = Runtime(0, 500)
torch.npu.set_device(0)

ROPE_THETA = 10000.0
MAX_SEQ_LEN = 1024
BATCH_SIZE = 8
SEQ_LEN = 10

test_cases = {
    (torch.float16, 64, 128),
    (torch.float16, 64, 192),
    (torch.float16, 64, 576),
    (torch.bfloat16, 64, 128),
    (torch.bfloat16, 64, 192),
    (torch.bfloat16, 64, 576),
}

for test_dtype, rope_dim, q_dim in test_cases:
    n_local_heads = 16
    num_tokens = BATCH_SIZE * SEQ_LEN
    
    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        attnQWithQr = torch.randn(num_tokens, n_local_heads, q_dim)
        freqs_cis = precompute_freqs_cis(rope_dim, MAX_SEQ_LEN, ROPE_THETA)[0:SEQ_LEN]
        
        attnQWithQr_xlite = attnQWithQr.clone()
        freqs_cis_xlite = freqs_cis.clone()
        
        output_pe = torch.zeros(num_tokens, n_local_heads, rope_dim)
        
        position = torch.arange(SEQ_LEN, dtype=torch.int64).repeat(BATCH_SIZE)
        
        vGatherIndices = torch.arange(rope_dim, dtype=torch.int32)
        for i in range(rope_dim // 2):
            vGatherIndices[i * 2] = i * 4
            vGatherIndices[i * 2 + 1] = i * 4 + 1024

    # standard
    # Reshape to (bsz, seqlen, n_local_heads, q_dim) for apply_rotary_emb
    attnQWithQr_reshaped = attnQWithQr.view(BATCH_SIZE, SEQ_LEN, n_local_heads, q_dim)
    # Extract only the rope part (last rope_dim elements)
    q_pe_input = attnQWithQr_reshaped[..., -rope_dim:]
    # Reshape to (bsz, seqlen, n_local_heads, rope_dim) for apply_rotary_emb
    q_pe_input = q_pe_input.contiguous()
    q_pe_standard = apply_rotary_emb(q_pe_input, freqs_cis)
    # Reshape back to (num_tokens, n_local_heads, rope_dim)
    q_pe_standard = q_pe_standard.view(num_tokens, n_local_heads, rope_dim)

    # xlite
    torch.npu.synchronize()
    rope_complex(rt, num_tokens, n_local_heads, q_dim, rope_dim, attnQWithQr_xlite,
                freqs_cis_xlite, position, vGatherIndices, output_pe, 0)
    torch.npu.synchronize()

    logging.info(f'rope_complex (rope_dim={rope_dim}, q_dim={q_dim}, {test_dtype}) executed!')

    try:
        torch.testing.assert_close(q_pe_standard, output_pe, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        logging.error(f'{e}')
        logging.error(f'torch_npu: {q_pe_standard}')
        logging.error(f'xlite: {output_pe}')
