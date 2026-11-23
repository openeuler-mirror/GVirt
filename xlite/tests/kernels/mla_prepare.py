#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""
mla_prepare (rmsnorm + rope + paged cache scatter) regression test.

Geometries cover both the original contiguous prefill layout and the GLM5
decode layout: per-token absolute positions reused as slot ids, so tokens
scatter across blocks with non-aligned block-interior offsets.
"""

import logging
import sys

import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, mla_prepare

logging.getLogger().setLevel(logging.INFO)

# Norm refs can't be bit-matched: FP32 reduction order differs from torch,
# so 1-ULP flips are inherent.
ATOL_RTOL = {
    torch.float16: (2e-5, 1e-3),
    torch.bfloat16: (1e-4, 1e-2),
}

ROPE_THETA = 10000.0
BLOCK_SIZE = 128
NORM_EPS = 1e-6

rt = Runtime(0, size=500)
torch.npu.set_device(0)

# (name, batch, seq_len, positions). positions=None: prefill (arange per
# batch, contiguous slots); a list: one absolute position per token, reused
# as its slot id (decode, cached ~21k).
GEOMS = [
    ("prefill-b8-s10", 8, 10, None),
    ("decode-b4-c21k", 4, 1, [21000 + i * 137 for i in range(4)]),
    ("decode-b20-c21k", 20, 1, [21000 + i * 137 for i in range(20)]),
]

# (dtype, q_lora_rank, kv_lora_rank, rope_head_dim)
TEST_CASES = [
    (torch.float16, 2048, 512, 64),
    (torch.bfloat16, 2048, 512, 64),
]


def check_close(ref: torch.Tensor, actual: torch.Tensor, dtype: torch.dtype) -> None:
    atol, rtol = ATOL_RTOL[dtype]
    torch.testing.assert_close(ref, actual, atol=atol, rtol=rtol)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Complex freqs_cis [end, dim/2]; interleaved real view
    [cos0, sin0, cos1, sin1, ...] - the layout the kernel expects.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: [bsz, seqlen, rope_dim]; freqs_cis: [bsz, seqlen, rope_dim/2] complex per token; returns [bsz, seqlen, rope_dim] TTTWWW."""
    dtype = x.dtype
    bsz, seqlen, rope_dim = x.shape
    x = torch.view_as_complex(x.float().view(bsz, seqlen, rope_dim // 2, 2))
    y = torch.view_as_real(x * freqs_cis)
    y = torch.cat([y[..., 0], y[..., 1]], dim=-1)
    return y.to(dtype)


def stand_rmsnorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight.float() * x
    x = x + bias.float()
    return x.to(input_dtype)


def run_case(geom_name, batch_size, seq_len, pos_spec, test_dtype, q_lora_rank, kv_lora_rank, rope_head_dim):
    """Runs one (geometry, dtype) combination; returns a list of failure names."""
    tag = f"{geom_name}/{test_dtype}"
    failures = []
    num_tokens = batch_size * seq_len
    total_dim = q_lora_rank + kv_lora_rank + rope_head_dim

    if pos_spec is None:
        position = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
    else:
        position = torch.tensor(pos_spec, dtype=torch.int64)
        slot_mapping = position.to(torch.int32)
    max_pos = int(max(position.max().item(), slot_mapping.max().item())) + 1
    block_num = max_pos // BLOCK_SIZE + 1

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        # Bind to the NPU, otherwise the kernel receives host pointers.
        position = position.npu()
        slot_mapping = slot_mapping.npu()
        attn_qkvc = torch.randn(num_tokens, total_dim) / 10
        attn_qkvc_ref = attn_qkvc.clone()

        q_norm = torch.randn(q_lora_rank) / 10
        q_norm_bias = torch.randn(q_lora_rank) / 10
        kv_norm = torch.randn(kv_lora_rank) / 10
        kv_norm_bias = torch.randn(kv_lora_rank) / 10

        attn_norm_qc = torch.zeros(num_tokens, q_lora_rank, dtype=test_dtype)
        attn_norm_kvc = torch.zeros(num_tokens, kv_lora_rank, dtype=test_dtype)

        # Freqs cache must cover every position row the kernel reads.
        freqs_cis = precompute_freqs_cis(rope_head_dim, max_pos, ROPE_THETA)

        k_cache = torch.zeros(block_num, BLOCK_SIZE, kv_lora_rank, dtype=test_dtype)
        pe_cache = torch.zeros(block_num, BLOCK_SIZE, rope_head_dim, dtype=test_dtype)

    # ---- reference ----
    q_slice = attn_qkvc_ref[:, :q_lora_rank]
    kv_slice = attn_qkvc_ref[:, q_lora_rank : q_lora_rank + kv_lora_rank]
    pe_slice = attn_qkvc_ref[:, q_lora_rank + kv_lora_rank :]

    attn_norm_qc_ref = stand_rmsnorm(q_slice, q_norm, q_norm_bias, NORM_EPS)
    attn_norm_kvc_ref = stand_rmsnorm(kv_slice, kv_norm, kv_norm_bias, NORM_EPS)

    # aclnnIndex has no complex64 kernel: gather rows in the float view,
    # then rebuild the complex phases per token.
    fc = torch.view_as_complex(torch.view_as_real(freqs_cis)[position].view(batch_size, seq_len, -1, 2))
    pe_rot = apply_rotary_emb(pe_slice.view(batch_size, seq_len, rope_head_dim), fc)
    pe_rot = pe_rot.view(num_tokens, rope_head_dim)
    k_cache_ref = torch.zeros(block_num, BLOCK_SIZE, kv_lora_rank, dtype=test_dtype, device="npu")
    pe_cache_ref = torch.zeros(block_num, BLOCK_SIZE, rope_head_dim, dtype=test_dtype, device="npu")
    for i in range(num_tokens):
        slot = int(slot_mapping[i].item())
        b = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        k_cache_ref[b, off] = attn_norm_kvc_ref[i]
        pe_cache_ref[b, off] = pe_rot[i]

    # ---- xlite ----
    torch.npu.synchronize()
    mla_prepare(
        rt,
        attn_qkvc,
        q_norm,
        q_norm_bias,
        attn_norm_qc,
        kv_norm,
        kv_norm_bias,
        attn_norm_kvc,
        freqs_cis,
        position,
        q_lora_rank,
        kv_lora_rank,
        rope_head_dim,
        BLOCK_SIZE,
        k_cache,
        pe_cache,
        slot_mapping,
        NORM_EPS,
    )
    torch.npu.synchronize()
    logging.info(f"mla_prepare {tag} (q={q_lora_rank}, kv={kv_lora_rank}, rope={rope_head_dim}) executed!")

    # attn_qkvc is read-only (rotated pe goes to pe_cache only): must stay
    # bit-identical.
    checks = [
        ("attn_norm_qc", lambda: check_close(attn_norm_qc_ref, attn_norm_qc, test_dtype)),
        ("attn_norm_kvc", lambda: check_close(attn_norm_kvc_ref, attn_norm_kvc, test_dtype)),
        ("k_cache", lambda: check_close(k_cache_ref, k_cache, test_dtype)),
        ("pe_cache", lambda: check_close(pe_cache_ref, pe_cache, test_dtype)),
        (
            "attn_qkvc read-only",
            lambda: (
                torch.equal(attn_qkvc, attn_qkvc_ref)
                or (_ for _ in ()).throw(AssertionError("attn_qkvc modified by kernel"))
            ),
        ),
    ]
    for name, check in checks:
        try:
            check()
        except AssertionError as e:
            logging.error(f"{tag}: {name} FAILED: {e}")
            failures.append(f"{tag}:{name}")
    return failures


def main():
    failures = []
    for geom_name, batch_size, seq_len, pos_spec in GEOMS:
        for test_dtype, q_lora_rank, kv_lora_rank, rope_head_dim in TEST_CASES:
            failures += run_case(
                geom_name, batch_size, seq_len, pos_spec, test_dtype, q_lora_rank, kv_lora_rank, rope_head_dim
            )
    logging.info("==== SUMMARY ====")
    if failures:
        for f in failures:
            logging.error("FAILED: %s", f)
        sys.exit(1)
    logging.info("ALL PASS")


if __name__ == "__main__":
    main()
