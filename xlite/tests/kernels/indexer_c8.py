#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""C8 Indexer prepare tests.

Compare with PyTorch and torch_npu.
"""
import argparse
import json
import logging
from pathlib import Path

import torch

from indexer_k_cache_c8 import frequencies, rotation


HEAD_DIM = 128
N_HEADS = 32
ROPE_DIM = 64
TOP_K = 2048
K_CACHE_FILL = -101
SCALE_CACHE_FILL = -17

# Token counts for Q/K phase tests.
prepare_tokens = [1, 17, 128, 129]


def allocate_cache(device, slots, block_size):
    num_blocks = (int(slots.max()) + block_size) // block_size
    return (torch.full((num_blocks, block_size, 1, HEAD_DIM), K_CACHE_FILL,
                       dtype=torch.int8, device=device),
            torch.full((num_blocks, block_size, 1, 1), SCALE_CACHE_FILL,
                       dtype=torch.float16, device=device))


def call_prepare(rt, device, kw, k_norm, k_norm_bias, freqs, positions, slots, caches, q=None):
    from xlite._C import indexer_prepare

    is_long = q is not None
    if q is None:
        q = torch.empty(0, dtype=torch.bfloat16)
    inputs = (kw, k_norm, k_norm_bias, freqs, positions, slots.int(), q)
    device_inputs = [tensor.to(device) for tensor in inputs]
    kw_npu, norm_npu, bias_npu, freqs_npu, positions_npu, slots_npu, q_npu = device_inputs
    outputs = {}
    if is_long:
        outputs = {
            "q8": torch.full(q.shape, K_CACHE_FILL, dtype=torch.int8, device=device),
            "q_scale": torch.full((len(q), N_HEADS), SCALE_CACHE_FILL,
                                  dtype=torch.float16, device=device),
            "scaled_weights": torch.full((len(q), N_HEADS), SCALE_CACHE_FILL,
                                         dtype=torch.float16, device=device),
        }
    torch.npu.synchronize(device)
    indexer_prepare(rt, kw_npu, norm_npu, bias_npu, freqs_npu, positions_npu,
                    HEAD_DIM, N_HEADS, ROPE_DIM, caches[0].shape[1], caches[0],
                    slots_npu, 1e-5, q_npu, 1 / 64, TOP_K, is_long,
                    k_scale_cache=caches[1], **outputs)
    for actual, original in zip(device_inputs, inputs):
        assert torch.equal(actual.cpu(), original), "C8 prepare must not mutate projected inputs"
    return {name: tensor.cpu() for name, tensor in outputs.items()}


def run_prepare_tests(rt, device):
    """Check the K/Q phase boundary."""
    generator = torch.Generator().manual_seed(912403)
    freqs = frequencies(256)
    k_norm = torch.randn(HEAD_DIM, generator=generator)
    k_norm_bias = torch.randn(HEAD_DIM, generator=generator)
    results = []
    for tokens in prepare_tokens:
        kw = torch.randn(tokens, HEAD_DIM + N_HEADS, generator=generator).bfloat16()
        q = torch.randn(tokens, N_HEADS * HEAD_DIM, generator=generator).bfloat16()
        positions = torch.arange(tokens).long() % len(freqs)
        allocation_slots = torch.arange(tokens).int() + 3
        slots = allocation_slots.clone()
        if tokens > 1:
            slots[::7] = -1
        for freqs_fp32 in (False, True):
            table = freqs.float() if freqs_fp32 else freqs
            combined, k_only = (allocate_cache(device, allocation_slots, 32) for _ in range(2))
            outputs = call_prepare(rt, device, kw, k_norm, k_norm_bias, table,
                                   positions, slots, combined, q)
            call_prepare(rt, device, kw, k_norm, k_norm_bias, table, positions, slots, k_only)
            for actual, expected in zip(combined, k_only):
                assert torch.equal(actual.cpu(), expected.cpu()), "combined/K-only cache mismatch"
            assert bool((outputs["q_scale"] > 0).all()), "some Q rows were not quantized"

            # Skipping K writes must not affect Q.
            inactive = allocate_cache(device, allocation_slots, 32)
            repeated = call_prepare(rt, device, kw, k_norm * 0, k_norm_bias * 0, table,
                                    positions, torch.full_like(slots, -1), inactive, q)
            for name in outputs:
                assert torch.equal(outputs[name], repeated[name]), ("Q depends on K phase", name)
            assert bool((inactive[0].cpu() == K_CACHE_FILL).all()), "padded K cache was changed"
            assert bool((inactive[1].cpu() == SCALE_CACHE_FILL).all()), "padded K scale cache was changed"
            results.append({"tokens": tokens, "freqs_fp32": freqs_fp32})
            logging.info("indexer_c8 prepare (%d tokens, freqs_fp32=%s) passed", tokens, freqs_fp32)
    return results


def run_quantization_test(rt, device):
    """Check rounding ties, zero rows and tiny scales."""
    import torch_npu

    q = torch.zeros(4, N_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    gains = torch.arange(1, N_HEADS + 1).bfloat16()
    q[0, :, 0], q[0, :, 1] = 3 * gains, gains
    q[1] = -q[0]
    q[3] = q[0] * (2.0 ** -30)
    # Position zero gives identity RoPE.
    rotated = (q.float() @ rotation().float()).bfloat16()
    active = rotated.float().abs().amax(-1) != 0
    expected_q = torch.zeros_like(q, dtype=torch.int8)
    expected_scale = torch.zeros(4, N_HEADS, dtype=torch.float16)
    quantized, scale = torch_npu.npu_dynamic_quant(rotated[active].to(device), dst_type=torch.int8)
    expected_q[active], expected_scale[active] = quantized.cpu(), scale.cpu().half()

    kw = torch.zeros(4, HEAD_DIM + N_HEADS, dtype=torch.bfloat16)
    kw[:, HEAD_DIM:] = 1
    slots = torch.arange(4).int()
    caches = allocate_cache(device, slots, 32)
    actual = call_prepare(rt, device, kw, torch.ones(HEAD_DIM), torch.zeros(HEAD_DIM),
                          frequencies(1), torch.zeros(4, dtype=torch.int64), slots,
                          caches, q.reshape(4, N_HEADS * HEAD_DIM))
    torch.testing.assert_close(actual["q8"].reshape_as(q), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual["q_scale"], expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual["scaled_weights"], expected_scale, rtol=0, atol=0)
    assert bool((caches[0].cpu().view(-1, HEAD_DIM)[:4] == 0).all())
    assert bool((caches[1].cpu().view(-1)[:4] == 0).all())
    logging.info("indexer_c8 quantization boundaries passed")
    return {"tokens": 4, "heads": N_HEADS, "native_byte_mismatches": 0}


def run_invalid_tests(rt, device):
    from xlite._C import indexer_prepare

    caches = allocate_cache(device, torch.arange(2), 32)
    kw = torch.zeros(2, HEAD_DIM + N_HEADS, dtype=torch.bfloat16).to(device)
    k_norm, k_norm_bias = torch.ones(HEAD_DIM).to(device), torch.zeros(HEAD_DIM).to(device)
    freqs = frequencies(8).to(device)
    positions, slots = torch.arange(2).to(device), torch.arange(2).int().to(device)
    q = torch.zeros(2, N_HEADS * HEAD_DIM, dtype=torch.bfloat16).to(device)
    outputs = {
        "k_scale_cache": caches[1],
        "q8": torch.zeros(2, N_HEADS * HEAD_DIM, dtype=torch.int8).to(device),
        "q_scale": torch.zeros(2, N_HEADS).half().to(device),
        "scaled_weights": torch.zeros(2, N_HEADS).half().to(device),
    }
    base_args = [rt, kw, k_norm, k_norm_bias, freqs, positions, HEAD_DIM, N_HEADS,
                 ROPE_DIM, 32, caches[0], slots, 1e-5, q, 1., TOP_K, True]
    prepare_changes = [
        ({"k_scale_cache": None}, {}),
        ({"q8": None}, {}),
        ({"q_scale": None}, {}),
        ({"scaled_weights": None}, {}),
        ({}, {2: k_norm.half()}),
        ({}, {7: 16}),
    ]
    rejected = 0
    for output_changes, arg_changes in prepare_changes:
        args, extra = list(base_args), dict(outputs)
        extra.update(output_changes)
        for index, value in arg_changes.items():
            args[index] = value
        torch.npu.synchronize(device)
        try:
            indexer_prepare(*args, **extra)
        except (ValueError, RuntimeError):
            assert bool((caches[0].cpu() == K_CACHE_FILL).all())
            assert bool((caches[1].cpu() == SCALE_CACHE_FILL).all())
            rejected += 1
        else:
            raise AssertionError(("Invalid C8 prepare input was accepted",
                                  output_changes.keys(), arg_changes.keys()))

    # Padding slots must leave both caches unchanged.
    call_prepare(rt, device, torch.zeros(2, HEAD_DIM + N_HEADS).bfloat16(),
                 torch.ones(HEAD_DIM), torch.zeros(HEAD_DIM), frequencies(8),
                 torch.tensor([0, -100]), torch.tensor([0, -1]), caches)
    assert bool((caches[0].cpu().view(-1, HEAD_DIM)[1:] == K_CACHE_FILL).all())
    assert bool((caches[1].cpu().view(-1)[1:] == SCALE_CACHE_FILL).all())

    logging.info("indexer_c8 rejected %d invalid inputs", rejected)
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch_npu  # noqa: F401
    from xlite._C import Runtime

    logging.getLogger().setLevel(logging.INFO)
    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    device = torch.device(f"npu:{args.device}")
    rt = Runtime(device.index, 160)

    result = {"rejected_inputs": run_invalid_tests(rt, device),
              "prepare_phases": run_prepare_tests(rt, device),
              "quantization_boundary": run_quantization_test(rt, device)}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    logging.info("indexer_c8 tests passed")


if __name__ == "__main__":
    main()
