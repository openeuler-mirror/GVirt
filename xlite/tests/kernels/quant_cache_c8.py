#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""quant_cache_c8 row quantization and paged-write tests."""
import logging

import torch


QUANT_MAX = 127
K_CACHE_FILL = -101
SCALE_CACHE_FILL = -17

npu_id = 0

# Row dims exercise calcRepeat/quantRepeat combinations (64: 1x1, 128: 2x1,
# 192: 3x2); token counts cross the 32-block launch stride.
dims = [64, 128, 192]
test_cases = [0, 1, 17, 129, 257]


def check_values(k8, scales, rows, name):
    # Quantization uses the FP32 scale, not the stored FP16 scale.
    amax = rows.float().abs().amax(-1)
    scale = amax * (1.0 / QUANT_MAX)
    expected_scale = scale.half()
    torch.testing.assert_close(scales, expected_scale, rtol=0, atol=0)
    normalized = rows.float() * torch.where(scale == 0, 0, scale.reciprocal())[:, None]
    expected_k8 = normalized.round().to(torch.int8)
    torch.testing.assert_close(k8, expected_k8, atol=1, rtol=1 / 128)
    assert bool(((k8 >= -QUANT_MAX) & (k8 <= QUANT_MAX)).all()), name


def run_test(rt, device, name, rows, slots, capacity, cache_offset=0):
    from xlite._C import quant_cache_c8

    tokens, dim = rows.shape
    k_base = torch.full((capacity * dim + 2 * cache_offset,), K_CACHE_FILL,
                        dtype=torch.int8, device=device)
    scale_base = torch.full((capacity + 2 * cache_offset,), SCALE_CACHE_FILL,
                            dtype=torch.float16, device=device)
    k_cache = k_base[cache_offset:cache_offset + capacity * dim].view(capacity, dim)
    scale_cache = scale_base[cache_offset:cache_offset + capacity]
    device_rows = rows.to(device)
    device_slots = slots.to(device)
    torch.npu.synchronize(device)
    # One block covering the whole pool keeps the dense flat addressing.
    quant_cache_c8(rt, device_rows, device_slots, k_cache, scale_cache, capacity)
    torch.npu.synchronize(device)
    assert torch.equal(device_rows.cpu(), rows), (name, "input mutated")

    k_base, scale_base = k_base.cpu(), scale_base.cpu()
    k8 = k_base[cache_offset:cache_offset + capacity * dim].view(capacity, dim)
    scales = scale_base[cache_offset:cache_offset + capacity]
    written = slots[(slots >= 0) & (slots < capacity)].long()
    untouched = torch.ones(capacity, dtype=torch.bool)
    untouched[written] = False

    # Only mapped in-range slots may change.
    assert bool((k8[untouched] == K_CACHE_FILL).all()), (name, "K canary")
    assert bool((scales[untouched] == SCALE_CACHE_FILL).all()), (name, "scale canary")
    if cache_offset:
        assert bool((k_base[[0, -1]] == K_CACHE_FILL).all()), (name, "K guard")
        assert bool((scale_base[[0, -1]] == SCALE_CACHE_FILL).all()), (name, "scale guard")
    valid = (slots >= 0) & (slots < capacity)
    check_values(k8[slots[valid]], scales[slots[valid]], rows[valid], name)
    logging.info("quant_cache_c8 %s (%d rows, dim %d) passed", name, tokens, dim)


def run_packed_test(rt, device, name, rows, slots, blocks, bs):
    """DSV4 vllm pool layout: each page packs the int8 K region (bs*dim bytes)
    followed by the fp16 scale region (bs*2 bytes), so the k and scale views are
    block-strided over one allocation."""
    from xlite._C import quant_cache_c8

    tokens, dim = rows.shape
    capacity = blocks * bs
    assert slots.min() >= 0 and slots.max() < capacity
    page = bs * (dim + 2)  # bytes per block
    raw = torch.full((blocks * page,), K_CACHE_FILL & 0xFF, dtype=torch.uint8, device=device)
    k_cache = torch.as_strided(raw.view(torch.int8), (blocks, bs, dim), (page, dim, 1))
    scale_cache = torch.as_strided(raw.view(torch.float16), (blocks, bs), (page // 2, 1),
                                   storage_offset=bs * dim // 2)
    device_rows = rows.to(device)
    device_slots = slots.to(device)
    torch.npu.synchronize(device)
    quant_cache_c8(rt, device_rows, device_slots, k_cache, scale_cache, bs)
    torch.npu.synchronize(device)
    assert torch.equal(device_rows.cpu(), rows), (name, "input mutated")

    k_cache, scale_cache = k_cache.cpu(), scale_cache.cpu()
    block_ids = (slots // bs).long()
    offsets = (slots % bs).long()
    k8 = k_cache[block_ids, offsets]
    scales = scale_cache[block_ids, offsets]
    # 0x9b9b (the repeated K_CACHE_FILL byte pattern) is negative, and written
    # scales are absmax/127 >= 0, so untouched scales stay bit-identical.
    untouched = torch.ones(capacity, dtype=torch.bool)
    untouched[slots.long()] = False
    untouched_k = k_cache.view(capacity, dim)[untouched]
    untouched_s = scale_cache.view(capacity)[untouched]
    assert bool((untouched_k == K_CACHE_FILL).all()), (name, "K canary")
    assert bool((untouched_s < 0).all()), (name, "scale canary")
    check_values(k8, scales, rows, name)
    logging.info("quant_cache_c8 %s (%d rows, dim %d, %dx%d packed) passed", name, tokens, dim,
                 blocks, bs)


def main():
    import torch_npu  # noqa: F401
    from xlite._C import Runtime

    logging.getLogger().setLevel(logging.INFO)
    torch.npu.set_device(npu_id)
    device = torch.device(f"npu:{npu_id}")
    rt = Runtime(device.index, size=32)
    generator = torch.Generator().manual_seed(9128)
    for dim in dims:
        for tokens in test_cases:
            rows = torch.randn(tokens, dim, generator=generator).bfloat16()
            slots = torch.randperm(384, generator=generator)[:tokens].int()
            if tokens > 1:
                slots[::7] = -1
            run_test(rt, device, f"random-dim{dim}-{tokens}", rows, slots, 384)

    # Per-row scale edge cases: zero row (zeros + zero scale), constant row,
    # and a single large outlier dominating one row's absmax.
    dim = 128
    rows = torch.randn(17, dim, generator=generator).bfloat16()
    rows[3] = 0
    rows[5] = 4
    rows[7, 0] = 1024
    run_test(rt, device, "edge-rows", rows, torch.arange(17).int(), 32)

    # Slots at/above capacity are skipped entirely (both caches keep canaries).
    slots = torch.arange(17).int()
    slots[2] = 32
    slots[4] = 1000
    run_test(rt, device, "slot-out-of-range", rows, slots, 32)

    # Guard both ends of cache views with unaligned addresses.
    run_test(rt, device, "offset-cache-views", rows, torch.arange(17).int(), 32, cache_offset=1)

    # DSV4's vllm pool: K and scales packed page-strided into one allocation.
    dim = 128
    rows = torch.randn(17, dim, generator=generator).bfloat16()
    slots = torch.randperm(24, generator=generator)[:17].int()
    run_packed_test(rt, device, "packed-vllm-pool", rows, slots, blocks=3, bs=8)

    logging.info("quant_cache_c8 tests passed")


if __name__ == "__main__":
    main()
