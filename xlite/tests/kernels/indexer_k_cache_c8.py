#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""K-side C8 prepare and cache-write tests."""
import logging
import math

import torch
import torch.nn.functional as F


HEAD_DIM = 128
ROPE_DIM = 64
NORM_EPS = 1e-6
ROPE_THETA = 1e6
QUANT_MAX = 127
K_CACHE_FILL = -101
SCALE_CACHE_FILL = -17

npu_id = 0

# Token counts for random inputs.
test_cases = [0, 1, 17, 129, 257]


def frequencies(length):
    inv = ROPE_THETA ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
    angles = torch.arange(length).float()[:, None] * inv
    return torch.stack((angles.cos(), angles.sin()), -1).flatten(1).bfloat16()


def rotation():
    h = torch.ones(1, 1)
    while len(h) < HEAD_DIM:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(HEAD_DIM)).bfloat16()


def reference(k, k_norm, k_norm_bias, freqs, positions, norm_eps=NORM_EPS):
    """LayerNorm, interleaved RoPE and Hadamard with BF16 output at each stage."""
    norm = F.layer_norm(k[:, :HEAD_DIM].float(), (HEAD_DIM,),
                        k_norm.float(), k_norm_bias.float(), norm_eps).bfloat16()
    pairs = norm[:, :ROPE_DIM].float().reshape(-1, ROPE_DIM // 2, 2)
    trig = freqs[positions].float().reshape(-1, ROPE_DIM // 2, 2)
    real = pairs[..., 0] * trig[..., 0] - pairs[..., 1] * trig[..., 1]
    imag = pairs[..., 1] * trig[..., 0] + pairs[..., 0] * trig[..., 1]
    rope = torch.cat((torch.stack((real, imag), -1).flatten(1).bfloat16(), norm[:, ROPE_DIM:]), 1)
    return (rope.float() @ rotation().float()).bfloat16()


def check_values(k8, scales, rotated, name):
    # Quantization uses the FP32 scale, not the stored FP16 scale.
    amax = rotated.float().abs().amax(-1)
    scale = amax * (1.0 / QUANT_MAX)
    expected_scale = scale.half()
    torch.testing.assert_close(scales, expected_scale, rtol=0, atol=0)
    normalized = rotated.float() * torch.where(scale == 0, 0, scale.reciprocal())[:, None]
    expected_k8 = normalized.round().to(torch.int8)
    torch.testing.assert_close(k8, expected_k8, atol=1, rtol=1 / 128)
    assert bool(((k8 >= -QUANT_MAX) & (k8 <= QUANT_MAX)).all()), name


def run_test(rt, device, name, k, k_norm, k_norm_bias, freqs, positions, slots,
             num_blocks, block_size, norm_eps=NORM_EPS, cache_offset=0):
    from xlite._C import indexer_k_cache_c8

    capacity = num_blocks * block_size
    k_base = torch.full((capacity * HEAD_DIM + 2 * cache_offset,), K_CACHE_FILL,
                        dtype=torch.int8, device=device)
    scale_base = torch.full((capacity + 2 * cache_offset,), SCALE_CACHE_FILL,
                            dtype=torch.float16, device=device)
    k_cache = k_base[cache_offset:cache_offset + capacity * HEAD_DIM].view(
        num_blocks, block_size, 1, HEAD_DIM)
    scale_cache = scale_base[cache_offset:cache_offset + capacity].view(num_blocks, block_size, 1, 1)
    inputs = (k, k_norm, k_norm_bias, freqs, positions, slots)
    device_inputs = [tensor.to(device) for tensor in inputs]
    torch.npu.synchronize(device)
    indexer_k_cache_c8(rt, *device_inputs, k_cache, scale_cache, norm_eps=norm_eps)
    torch.npu.synchronize(device)
    assert torch.equal(device_inputs[0].cpu(), k), (name, "K input mutated")

    k_base, scale_base = k_base.cpu(), scale_base.cpu()
    k8 = k_base[cache_offset:cache_offset + capacity * HEAD_DIM].view(-1, HEAD_DIM)
    scales = scale_base[cache_offset:cache_offset + capacity]
    active = slots[:len(k)] >= 0
    written = slots[:len(k)][active].long()
    untouched = torch.ones(capacity, dtype=torch.bool)
    untouched[written] = False

    # Only mapped slots may change.
    assert bool((k8[untouched] == K_CACHE_FILL).all()), (name, "K canary")
    assert bool((scales[untouched] == SCALE_CACHE_FILL).all()), (name, "scale canary")
    if cache_offset:
        assert bool((k_base[[0, -1]] == K_CACHE_FILL).all()), (name, "K guard")
        assert bool((scale_base[[0, -1]] == SCALE_CACHE_FILL).all()), (name, "scale guard")
    ref_freqs = torch.view_as_real(freqs) if freqs.is_complex() else freqs
    rotated = reference(k[active], k_norm, k_norm_bias, ref_freqs, positions[:len(k)][active], norm_eps)
    check_values(k8[written], scales[written], rotated, name)
    logging.info("indexer_k_cache_c8 %s (%d tokens) passed", name, len(k))
    return k8, scales


def main():
    import torch_npu  # noqa: F401
    from xlite._C import Runtime

    logging.getLogger().setLevel(logging.INFO)
    torch.npu.set_device(npu_id)
    device = torch.device(f"npu:{npu_id}")
    rt = Runtime(device.index, 32)
    generator = torch.Generator().manual_seed(9128)
    k_norm, k_norm_bias = torch.ones(HEAD_DIM), torch.zeros(HEAD_DIM)
    freqs = frequencies(256)
    freqs_fp32 = freqs.float().view(len(freqs), ROPE_DIM // 2, 2)
    freqs_complex = torch.view_as_complex(freqs_fp32)
    for tokens in test_cases:
        k = torch.randn(tokens, HEAD_DIM + 32, generator=generator).bfloat16()
        positions = torch.arange(tokens, dtype=torch.int64) % 256
        slots = torch.randperm(384, generator=generator)[:tokens].int()
        if tokens > 1:
            slots[::7] = -1
            positions[::7] = -100  # Padding must not read the RoPE table.
        expected = None
        for table in (freqs, freqs_fp32, freqs_complex):
            actual = run_test(rt, device, f"random-{tokens}-{table.dtype}", k, k_norm, k_norm_bias,
                              table, positions, slots, 12, 32)
            if expected is None:
                expected = actual
            else:
                for value, reference_value in zip(actual, expected):
                    torch.testing.assert_close(value, reference_value, rtol=0, atol=0)

    edge_cases = [
        ("zero", torch.zeros(17, HEAD_DIM).bfloat16(), k_norm, k_norm_bias),
        ("constant", torch.full((17, HEAD_DIM), 4, dtype=torch.bfloat16), k_norm, k_norm_bias),
        ("tiny-scale", torch.zeros(17, HEAD_DIM).bfloat16(), k_norm,
         torch.arange(HEAD_DIM).float() * 1e-10),
        ("affine-only", torch.randn(17, HEAD_DIM, generator=generator).bfloat16(), k_norm * 0,
         torch.randn(HEAD_DIM, generator=generator)),
    ]
    for name, k, weight, bias in edge_cases:
        run_test(rt, device, name, k, weight, bias, freqs, torch.arange(17),
                 torch.arange(17).int() + 31, 3, 32)

    # Low-variance rows distinguish the two epsilon values.
    k = (torch.randn(17, HEAD_DIM, generator=generator) * 1e-4).bfloat16()
    epsilon_scales = []
    for norm_eps in (1e-6, 1e-5):
        _, scales = run_test(
            rt, device, f"epsilon-{norm_eps:g}", k, k_norm, k_norm_bias, freqs,
            torch.arange(17), torch.arange(17).int() + 31, 3, 32, norm_eps=norm_eps)
        epsilon_scales.append(scales)
    assert not torch.equal(*epsilon_scales), "norm_eps must affect K scales"

    # Exercise an outlier and a row stride not aligned to 32 bytes.
    k_outlier = torch.randn(17, HEAD_DIM + 1, generator=generator).bfloat16()
    k_outlier[:, 0] = 1024
    run_test(rt, device, "odd-stride-outlier", k_outlier, k_norm, k_norm_bias,
             freqs, torch.arange(17), torch.arange(17).int() + 31, 3, 32)

    # Guard both ends of cache views with unaligned addresses.
    expected = run_test(rt, device, "offset-cache-views", k_outlier, k_norm, k_norm_bias, freqs,
                        torch.arange(17), torch.arange(17).int(), 1, 32, cache_offset=1)

    # Unused metadata capacity is ignored.
    positions = torch.cat((torch.arange(17), torch.full((15,), -100)))
    slots = torch.cat((torch.arange(17).int(), torch.full((15,), 32, dtype=torch.int32)))
    actual = run_test(rt, device, "metadata-capacity", k_outlier, k_norm, k_norm_bias,
                      freqs, positions, slots, 1, 32)
    for value, reference_value in zip(actual, expected):
        torch.testing.assert_close(value, reference_value, rtol=0, atol=0)
    logging.info("indexer_k_cache_c8 tests passed")


if __name__ == "__main__":
    main()
