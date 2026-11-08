#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""SFA-C8 quantization and cache writes."""
import logging

import torch
import torch_npu
from xlite._C import Runtime, mla_cache_c8, mla_prepare, rmsnorm_with_bias

npu_id = 0


def quantize(latent, pe):
    latent_dim = latent.shape[-1]
    q, scale = torch_npu.npu_dynamic_block_quant(
        latent.view(-1, 1, latent_dim), dst_type=torch.int8, row_block_size=1, col_block_size=128)
    return q.reshape(-1, latent_dim), pe, scale.reshape(-1, latent_dim // 128)


def test_write(rt, latent, pe, name, check_reuse=False):
    count = len(latent)
    dims = (latent.shape[-1], pe.shape[-1], latent.shape[-1] // 128)
    capacity = ((count + 31) // 16) * 16
    backing = [torch.full((capacity * dim + 64,), -101, dtype=dtype, device="npu")
               for dim, dtype in zip(dims, (torch.int8, torch.bfloat16, torch.float32))]
    cache = [t[32:-32].view(-1, 16, 1, dim) for t, dim in zip(backing, dims)]
    slots = torch.randperm(capacity, device="npu")[:count].int()
    if count > 1:
        slots[1] = -1
    before = latent.clone()
    expected = [t.clone().view(-1, dim) for t, dim in zip(cache, dims)]
    if count:
        rows = quantize(latent, pe)
        valid = slots >= 0
        for out, row in zip(expected, rows):
            out[slots[valid].long()] = row[valid]
    torch.npu.synchronize()
    mla_cache_c8(rt, latent, pe, slots, *cache)
    for actual, ref, dim in zip(cache, expected, dims):
        torch.testing.assert_close(actual.view(-1, dim).view(torch.uint8),
                                   ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(latent, before, rtol=0, atol=0)
    for t in backing:
        assert bool((t[:32] == -101).all() & (t[-32:] == -101).all())
    if check_reuse:
        for t in cache:
            t.fill_(-101)
        split = count // 2
        torch.npu.synchronize()
        mla_cache_c8(rt, latent[:split], pe[:split], slots[:split], *cache)
        mla_cache_c8(rt, latent[split:], pe[split:], slots[split:], *cache)
        for actual, ref, dim in zip(cache, expected, dims):
            torch.testing.assert_close(actual.view(-1, dim).view(torch.uint8),
                                       ref.view(torch.uint8), rtol=0, atol=0)
        if count:
            replacement = latent.neg()
            rows = quantize(replacement, pe)
            for out, row in zip(expected, rows):
                out[slots[valid].long()] = row[valid]
            torch.npu.synchronize()
            mla_cache_c8(rt, replacement, pe, slots, *cache)
            for actual, ref, dim in zip(cache, expected, dims):
                torch.testing.assert_close(actual.view(-1, dim).view(torch.uint8),
                                           ref.view(torch.uint8), rtol=0, atol=0)
    logging.info("mla_cache_c8 %s: passed", name)


def test_prepare(rt, count, latent_dim=512, rope_dim=64, norm_eps=1e-5, use_bias=True):
    block_size, q_rank = 128, 2048
    x = torch.randn(count, q_rank + latent_dim + rope_dim, dtype=torch.bfloat16, device="npu")
    if count == 129:
        x[:, q_rank:q_rank + latent_dim] *= 1e-3
    before = x.clone()
    qw = torch.randn(q_rank, dtype=torch.bfloat16, device="npu")
    qb = torch.randn_like(qw) / 10
    kw = torch.randn(latent_dim, dtype=torch.bfloat16, device="npu")
    kb = torch.randn_like(kw) / 10
    q0, q1 = [torch.empty(count, q_rank, dtype=torch.bfloat16, device="npu") for _ in range(2)]
    k0, k1 = [torch.empty(count, latent_dim, dtype=torch.bfloat16, device="npu") for _ in range(2)]
    blocks = (count + block_size - 1) // block_size
    kc = torch.zeros(blocks, block_size, 1, latent_dim, dtype=torch.bfloat16, device="npu")
    pc = torch.zeros(blocks, block_size, 1, rope_dim, dtype=torch.bfloat16, device="npu")
    dims = (latent_dim, rope_dim, latent_dim // 128)
    cache = [torch.full((blocks, block_size, 1, dim), -101, dtype=dtype, device="npu")
             for dim, dtype in zip(dims, (torch.int8, torch.bfloat16, torch.float32))]
    pos = torch.arange(count, dtype=torch.int64, device="npu")
    slots = torch.randperm(blocks * block_size, device="npu")[:count].int()
    angles = torch.outer(torch.arange(count).float(),
                         1e6 ** (-torch.arange(0, rope_dim, 2).float() / rope_dim))
    freqs = torch.polar(torch.ones_like(angles), angles).npu()
    empty = torch.empty(0, dtype=torch.bfloat16, device="npu")
    if not use_bias:
        qb, kb = empty, empty
    torch.npu.synchronize()
    if rope_dim == 64:
        mla_prepare(rt, x, qw, qb, q0, kw, kb, k0, freqs, pos, q_rank, latent_dim, rope_dim,
                    block_size, kc, pc, slots, norm_eps)
        expected = quantize(k0, pc.view(-1, rope_dim)[slots.long()])
    else:
        rmsnorm_with_bias(rt, x, qw, qb, q0, norm_eps, q_rank)
        rmsnorm_with_bias(rt, x, kw, kb, k0, norm_eps, latent_dim, 1, q_rank)
        pe = x[:, q_rank + latent_dim:].float().cpu().reshape(count, rope_dim // 2, 2)
        trig = torch.view_as_real(freqs.cpu())
        # Match the kernel's separate FP32 multiply and add/subtract steps.
        real = pe[..., 0] * trig[..., 0] - pe[..., 1] * trig[..., 1]
        imag = pe[..., 0] * trig[..., 1] + pe[..., 1] * trig[..., 0]
        pe = torch.cat((real, imag), dim=-1).to(x)
        expected = quantize(k0, pe)
    if count > 1:
        slots[1] = -1
    valid = slots >= 0
    expected_cache = [t.clone().view(-1, dim) for t, dim in zip(cache, dims)]
    for out, row in zip(expected_cache, expected):
        out[slots[valid].long()] = row[valid]
    torch.npu.synchronize()
    mla_prepare(rt, x, qw, qb, q1, kw, kb, k1, freqs, pos, q_rank, latent_dim, rope_dim,
                block_size, cache[0], cache[1], slots, norm_eps, k_scale_cache=cache[2])
    torch.testing.assert_close(q1, q0, rtol=0, atol=0)
    torch.testing.assert_close(k1, k0, rtol=0, atol=0)
    for actual, ref, dim in zip(cache, expected_cache, dims):
        torch.testing.assert_close(actual.view(-1, dim).view(torch.uint8),
                                   ref.view(torch.uint8), rtol=0, atol=0)
    cache = [torch.zeros_like(t) for t in cache]
    for out, row in zip(expected_cache, expected):
        out.zero_()
        out[slots[valid].long()] = row[valid]
    torch.npu.synchronize()
    mla_prepare(rt, x, qw, qb, q1, kw, kb, empty, freqs, pos, q_rank, latent_dim, rope_dim,
                block_size, cache[0], cache[1], slots, norm_eps, k_scale_cache=cache[2])
    for actual, ref, dim in zip(cache, expected_cache, dims):
        torch.testing.assert_close(actual.view(-1, dim).view(torch.uint8),
                                   ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    logging.info("SFA-C8 mla_prepare tokens=%d dims=%d/%d eps=%g bias=%s: passed",
                 count, latent_dim, rope_dim, norm_eps, use_bias)


def main():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(20261003)
    torch.npu.set_device(npu_id)
    rt = Runtime(npu_id, size=512)
    for count in (0, 1, 17, 129, 257):
        latent = torch.randn(count, 4, 128, dtype=torch.bfloat16, device="npu")
        latent *= torch.tensor([0.001, 0.1, 1, 10], device="npu")[None, :, None]
        pe = torch.randn(count, 64, dtype=torch.bfloat16, device="npu")
        test_write(rt, latent.flatten(1), pe, f"random-{count}",
                   check_reuse=(count == 129))
    for value in (0.0, 2.0 ** -125, 2.0 ** 100):
        latent = torch.full((3, 512), value, dtype=torch.bfloat16, device="npu")
        latent[:, 1::2].neg_()
        test_write(rt, latent, torch.zeros(3, 64, dtype=torch.bfloat16, device="npu"), str(value))
    # Half-integers and their neighboring BF16 values.
    halfway = torch.arange(-63.5, 64, dtype=torch.bfloat16)
    latent = torch.stack((torch.nextafter(halfway, torch.full_like(halfway, -float("inf"))),
                          halfway,
                          torch.nextafter(halfway, torch.full_like(halfway, float("inf")))))
    latent = latent.repeat_interleave(2, 0).repeat(1, 4).npu()
    latent[:, 127::128] = 127
    test_write(rt, latent, torch.zeros(6, 64, dtype=torch.bfloat16, device="npu"), "rounding")
    for count in (1, 17, 129):
        test_prepare(rt, count)
    test_prepare(rt, 129, norm_eps=1e-6, use_bias=False)
    for latent_dim, rope_dim in ((128, 32), (256, 64), (384, 64), (512, 128)):
        latent = torch.randn(129, latent_dim, dtype=torch.bfloat16, device="npu")
        pe = torch.randn(129, rope_dim, dtype=torch.bfloat16, device="npu")
        test_write(rt, latent, pe, f"dims-{latent_dim}-{rope_dim}")
        test_prepare(rt, 17, latent_dim, rope_dim)
    test_prepare(rt, 17, latent_dim=128, rope_dim=144)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
