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
import os
import torch
from xlite._C import Runtime, rope_and_cache

logging.getLogger().setLevel(logging.INFO)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """fp32 [end, dim] table, row = [cos(dim/2) | sin(dim/2)] (xlite contract)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_cache = freqs.cos()
    sin_cache = freqs.sin()
    freq_cis = torch.cat((cos_cache, sin_cache), dim=-1)
    return freq_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor) -> torch.Tensor:
    seqlen = x.size(2)  # [bsz, n_local_heads, seqlen, head_dim]
    cos, sin = freqs_cis[start_pos : start_pos + seqlen, :].chunk(2, dim=-1)
    cos = cos.repeat(1, 2)  # [seqlen, head_dim]
    sin = sin.repeat(1, 2)

    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x1 = x_rot[..., : x_rot.shape[-1] // 2]
    x2 = x_rot[..., x_rot.shape[-1] // 2 :]
    x_rot_half = torch.cat((-x2, x1), dim=-1)

    x_rot_embedded = (x_rot * cos) + (x_rot_half * sin)
    return torch.cat([x_rot_embedded, x_pass], dim=-1)


def apply_mrope_interleaved(x, freqs_cis, mrope_section, positions):
    """Reference for the interleaved mrope path (Qwen3-VL). Freq channel c of
    rot_dim/2 picks position T/H/W: c%3==1 and c < section[1]*3 -> H,
    c%3==2 and c < section[2]*3 -> W, else T. Neox half-split rotation.
    x: [1, heads, num_tokens, rot_dim]; positions: [3, num_tokens] int64."""
    t_pos, h_pos, w_pos = positions[0], positions[1], positions[2]
    cos, sin = freqs_cis.chunk(2, dim=-1)  # [max_pos, rot_dim/2]

    lanes = torch.arange(cos.shape[-1], device=freqs_cis.device)
    is_h = (lanes % 3 == 1) & (lanes < mrope_section[1] * 3)
    is_w = (lanes % 3 == 2) & (lanes < mrope_section[2] * 3)
    pos_sel = torch.where(is_h, h_pos.unsqueeze(1), torch.where(is_w, w_pos.unsqueeze(1), t_pos.unsqueeze(1)))  # [s, c]

    cos = cos[pos_sel, lanes]  # [s, rot_dim/2]
    sin = sin[pos_sel, lanes]
    cos = torch.cat([cos, cos], dim=-1)  # [s, rot_dim]
    sin = torch.cat([sin, sin], dim=-1)

    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1 = x_rot[..., : x_rot.shape[-1] // 2]
    x2 = x_rot[..., x_rot.shape[-1] // 2 :]
    x_rot_half = torch.cat((-x2, x1), dim=-1)
    x_rot_embedded = (x_rot * cos) + (x_rot_half * sin)
    return torch.cat([x_rot_embedded, x_pass], dim=-1)


# Pick a device: honor XLITE_TEST_DEVICE env var, default 0. The op uses no pool
# memory (all tensors are caller-allocated), so the pool only needs to cover
# runtime init; lower XLITE_TEST_POOL_MB when sharing a busy device.
DEVICE = int(os.environ.get("XLITE_TEST_DEVICE", "0"))
POOL_MB = int(os.environ.get("XLITE_TEST_POOL_MB", "500"))
rt = Runtime(DEVICE, POOL_MB)
torch.npu.set_device(DEVICE)

ROPE_THETA = 10000.0
N_HEADS = 32
N_KV_HEADS = 32

MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 8
BATCH_SIZE = 8
SEQ_LEN = 10

START_POS = 0
BLOCK_SIZE = 128
BLOCK_NUM = 1

passed = 0
failed = 0

test_cases = {
    (torch.float16, 64, 64),
    (torch.float16, 128, 128),
    (torch.float16, 128, 64),
    (torch.bfloat16, 64, 64),
    (torch.bfloat16, 128, 128),
    (torch.bfloat16, 128, 64),
}

# use_fp32=True (default): fp32 cos/sin table, fp32 rotation, single final
# rounding to the model dtype. use_fp32=False: deprecated model-dtype table
# path (bf16 round-trip emulation / fp16 direct math); kept covered until the
# legacy path is removed.
for use_fp32 in (True, False):
    for test_dtype, head_dim, rot_dim in test_cases:
        out_features = (N_HEADS + 2 * N_KV_HEADS) * head_dim
        torch.set_default_dtype(test_dtype)
        with torch.device("npu"):
            qkv_standard = torch.randn(BATCH_SIZE, SEQ_LEN, out_features)
            freqs_cis_standard = precompute_freqs_cis(rot_dim, MAX_SEQ_LEN, ROPE_THETA)
            if not use_fp32:
                freqs_cis_standard = freqs_cis_standard.to(test_dtype)

            k_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, head_dim)
            v_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, head_dim)

            qkv_xlite = qkv_standard.clone().view(BATCH_SIZE * SEQ_LEN, out_features)
            freqs_cis_xlite = freqs_cis_standard.clone()

            k_cache_xlite = torch.zeros(BLOCK_NUM, BLOCK_SIZE, N_KV_HEADS, head_dim)
            v_cache_xlite = torch.zeros(BLOCK_NUM, BLOCK_SIZE, N_KV_HEADS, head_dim)

            len = torch.arange(SEQ_LEN, dtype=torch.int64)
            position = len.unsqueeze(0).repeat(BATCH_SIZE, 1)
            len = torch.arange(SEQ_LEN, dtype=torch.int32)
            slot_mapping = len.unsqueeze(0).repeat(BATCH_SIZE, 1)

        # standard
        q, k, v = qkv_standard.split([N_HEADS * head_dim, N_KV_HEADS * head_dim, N_KV_HEADS * head_dim], dim=2)

        q = q.view(BATCH_SIZE, SEQ_LEN, N_HEADS, head_dim)
        k = k.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, head_dim)
        v = v.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q = apply_rotary_emb(q, START_POS, freqs_cis=freqs_cis_standard)
        k = apply_rotary_emb(k, START_POS, freqs_cis_standard)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

        q = q * head_dim**-0.5
        # single final rounding to the model dtype; no-op on the legacy path,
        # where apply_rotary_emb already returned test_dtype
        q = q.to(test_dtype)
        k = k.to(test_dtype)

        q = q.view(BATCH_SIZE * SEQ_LEN, N_HEADS * head_dim)
        k = k.view(BATCH_SIZE * SEQ_LEN, N_KV_HEADS * head_dim)
        v = v.view(BATCH_SIZE * SEQ_LEN, N_KV_HEADS * head_dim)
        qkv_standard_out = torch.cat([q, k, v], dim=1)

        # xlite
        torch.npu.synchronize()
        rope_and_cache(
            rt=rt,
            inout=qkv_xlite,
            k_cache=k_cache_xlite,
            v_cache=v_cache_xlite,
            position=position,
            cosin=freqs_cis_xlite,
            slot_mapping=slot_mapping,
            n_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS,
            head_dim=head_dim,
            rot_dim=rot_dim,
            block_size=BLOCK_SIZE,
            is_neox=True,
        )
        torch.npu.synchronize()

        logging.info(
            f"rope and cache (head_dim={head_dim}, rot_dim={rot_dim}, {test_dtype}, use_fp32={use_fp32}) executed!"
        )

        try:
            torch.testing.assert_close(qkv_standard_out, qkv_xlite, atol=1e-5, rtol=1e-3)
            passed += 1
        except AssertionError as e:
            failed += 1
            logging.error(f"{e}")
            logging.error(f"torch_npu: {qkv_standard_out}")
            logging.error(f"xlite: {qkv_xlite}")

# ---------------------------------------------------------------------------
# Interleaved-mrope large-batch prefill regression (Qwen3-VL-8B crash shape).
#
# rope_and_cache stages positions/slotMapping in UB in slices of
# iter_posslot_num = (UB_SIZE - params_start) / (8*pos_dim + 4) tokens. For
# bf16 + mrope (pos_dim=3, localHeads=8/localKvHeads=2 @ headDim 128) the
# staged region overflowed UB_SIZE by 32B and the scalar read of
# slot[poslot_idx] on block 40 trapped ("scalar to access the internal buffer
# of AICore is out of bounds"). Needs num_tokens >= 5849 to stage that many
# tokens in one iteration; 8192 = the MNBt of the crashing service. The
# legacy (use_fp32=False) mode keeps the exact crash-shape layout; the fp32
# mode stages a smaller region but is exercised here as well.
# ---------------------------------------------------------------------------
MROPE_SECTION = [24, 20, 20]
mrope_mask_h = sum(1 << i for i in range(1, MROPE_SECTION[1] * 3, 3))
mrope_mask_w = sum(1 << i for i in range(2, MROPE_SECTION[2] * 3, 3))
rot_dim_test = 128

for use_fp32 in (True, False):
    for test_dtype in (torch.bfloat16, torch.float16):
        torch.set_default_dtype(test_dtype)
        n_heads, n_kv_heads = 8, 2
        head_dim = rot_dim_test
        out_features = (n_heads + 2 * n_kv_heads) * head_dim
        num_tokens = 8192
        block_size = 128
        block_num = 64  # 8192 slots

        with torch.device("npu"):
            qkv = torch.randn(num_tokens, out_features)
            freqs_cis = precompute_freqs_cis(rot_dim_test, 1024, ROPE_THETA)
            if not use_fp32:
                freqs_cis = freqs_cis.to(test_dtype)
            k_cache = torch.zeros(block_num, block_size, n_kv_heads, head_dim)
            v_cache = torch.zeros(block_num, block_size, n_kv_heads, head_dim)
            # T/H/W streams: temporal advances 1:1, spatial coords stay bounded
            position = torch.stack(
                [
                    torch.arange(num_tokens, dtype=torch.int64) % 1000,
                    torch.arange(num_tokens, dtype=torch.int64) % 30,
                    torch.arange(num_tokens, dtype=torch.int64) % 40,
                ]
            )
            slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

        q, k, v = qkv.split([n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim], dim=1)

        q4 = q.view(1, num_tokens, n_heads, head_dim).transpose(1, 2)  # [1, h, s, d]
        k4 = k.view(1, num_tokens, n_kv_heads, head_dim).transpose(1, 2)
        q_std = apply_mrope_interleaved(q4, freqs_cis, MROPE_SECTION, position)
        k_std = apply_mrope_interleaved(k4, freqs_cis, MROPE_SECTION, position)
        q_std = q_std.transpose(1, 2).reshape(num_tokens, n_heads * head_dim) * head_dim**-0.5
        q_std = q_std.to(test_dtype)  # single final rounding (no-op on legacy)
        k_std = k_std.transpose(1, 2).reshape(num_tokens, n_kv_heads * head_dim).to(test_dtype)
        qkv_standard_out = torch.cat([q_std, k_std, v], dim=1)

        torch.npu.synchronize()
        rope_and_cache(
            rt=rt,
            inout=qkv,
            k_cache=k_cache,
            v_cache=v_cache,
            position=position,
            cosin=freqs_cis,
            slot_mapping=slot_mapping,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rot_dim=rot_dim_test,
            block_size=block_size,
            is_neox=True,
            mrope_mask_h=mrope_mask_h,
            mrope_mask_w=mrope_mask_w,
        )
        torch.npu.synchronize()

        logging.info(
            f"rope and cache (mrope interleaved, num_tokens={num_tokens}, {test_dtype}, use_fp32={use_fp32}) executed!"
        )

        try:
            torch.testing.assert_close(qkv_standard_out, qkv, atol=1e-5, rtol=1e-3)
            passed += 1
        except AssertionError as e:
            failed += 1
            logging.error(f"{e}")
            logging.error(f"torch_npu: {qkv_standard_out}")
            logging.error(f"xlite: {qkv}")

        # cache side effect: rotated K must land at [slot] in k_cache
        try:
            k_cache_flat = k_cache.view(num_tokens, n_kv_heads * head_dim)[slot_mapping.long()]
            torch.testing.assert_close(k_cache_flat, k_std, atol=1e-5, rtol=1e-3)
            passed += 1
        except AssertionError as e:
            failed += 1
            logging.error(f"{e}")

logging.info(f"rope_and_cache: {passed} passed, {failed} failed")
assert failed == 0, f"{failed} test(s) failed"
