#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""K-side C8 prepare and cache-write tests.

Compare with PyTorch and optional native fixtures.
"""
import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F


HEAD_DIM = 128
ROPE_DIM = 64
NORM_EPS = 1e-6
ROPE_THETA = 1e6
QUANT_MAX = 127
QUANT_ATOL = 0.5001
NATIVE_MAX_BYTE_DELTA = 1
NATIVE_MAX_MISMATCH_FRACTION = 0.001
K_CACHE_FILL = -101
SCALE_CACHE_FILL = -17

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


def reference(k, k_norm, k_norm_bias, freqs, positions):
    """LayerNorm, interleaved RoPE and Hadamard with BF16 output at each stage."""
    norm = F.layer_norm(k[:, :HEAD_DIM].float(), (HEAD_DIM,),
                        k_norm.float(), k_norm_bias.float(), NORM_EPS).bfloat16()
    pairs = norm[:, :ROPE_DIM].float().reshape(-1, ROPE_DIM // 2, 2)
    trig = freqs[positions].float().reshape(-1, ROPE_DIM // 2, 2)
    real = pairs[..., 0] * trig[..., 0] - pairs[..., 1] * trig[..., 1]
    imag = pairs[..., 1] * trig[..., 0] + pairs[..., 0] * trig[..., 1]
    rope = torch.cat((torch.stack((real, imag), -1).flatten(1).bfloat16(), norm[:, ROPE_DIM:]), 1)
    return (rope.float() @ rotation().float()).bfloat16()


def check_values(k8, scales, rotated, name, allow_bf16_ulp=False):
    # Quantization uses the FP32 scale, not the stored FP16 scale.
    amax = rotated.float().abs().amax(-1)
    scale = amax * (1.0 / QUANT_MAX)
    expected_scale = scale.half()
    torch.testing.assert_close(scales, expected_scale, rtol=0, atol=0)
    normalized = rotated.float() * torch.where(scale == 0, 0, scale.reciprocal())[:, None]
    error = (k8.float() - normalized).abs()
    max_error = float(error.max()) if error.numel() else 0.0
    rounding_values = int((error > QUANT_ATOL).sum())
    if allow_bf16_ulp:
        # Allow one BF16 step for matrix/butterfly rounding differences.
        bf = rotated.bfloat16()
        up = torch.nextafter(bf, torch.full_like(bf, float("inf"))).float()
        down = torch.nextafter(bf, torch.full_like(bf, -float("inf"))).float()
        ulp = torch.maximum(up - bf.float(), bf.float() - down)
        budget = QUANT_ATOL + ulp * torch.where(scale == 0, 0, scale.reciprocal())[:, None]
        assert bool((error <= budget).all()), (name, "BF16-step plus INT8 error", max_error)
    else:
        assert max_error <= QUANT_ATOL, (name, "nearest-integer error", max_error)
    assert bool(((k8 >= -QUANT_MAX) & (k8 <= QUANT_MAX)).all()), name
    return {"case": name, "tokens": len(k8), "max_quant_error": max_error,
            "preprocessing_rounding_values": rounding_values}


def allocate_cache(device, num_blocks, block_size):
    return (torch.full((num_blocks, block_size, 1, HEAD_DIM), K_CACHE_FILL,
                       dtype=torch.int8, device=device),
            torch.full((num_blocks, block_size, 1, 1), SCALE_CACHE_FILL,
                       dtype=torch.float16, device=device))


def call_op(rt, device, k, k_norm, k_norm_bias, freqs, positions, slots, caches):
    from xlite._C import indexer_k_cache_c8

    inputs = (k, k_norm, k_norm_bias, freqs, positions, slots)
    device_inputs = [tensor.to(device) for tensor in inputs]
    torch.npu.synchronize(device)
    indexer_k_cache_c8(rt, *device_inputs, *caches)
    for actual, original in zip(device_inputs, inputs):
        assert torch.equal(actual.cpu(), original), "device input mutated"
    return caches[0].cpu().reshape(-1, HEAD_DIM), caches[1].cpu().flatten()


def run_test(rt, device, name, k, k_norm, k_norm_bias, freqs, positions, slots,
             num_blocks, block_size, rotated=None, allow_bf16_ulp=False):
    caches = allocate_cache(device, num_blocks, block_size)
    inputs = (k, k_norm, k_norm_bias, freqs, positions, slots)
    before = [tensor.clone() for tensor in inputs]
    k8, scales = call_op(rt, device, *inputs, caches)
    active = slots >= 0
    written = slots[active].long()
    untouched = torch.ones(len(scales), dtype=torch.bool)
    untouched[written] = False

    # Only mapped slots may change.
    assert bool((k8[untouched] == K_CACHE_FILL).all()), (name, "K canary")
    assert bool((scales[untouched] == SCALE_CACHE_FILL).all()), (name, "scale canary")
    for actual, original in zip(inputs, before):
        assert torch.equal(actual, original), (name, "input mutated")

    if rotated is None:
        rotated = reference(k[active], k_norm, k_norm_bias, freqs, positions[active])
    else:
        rotated = rotated[active]
    result = check_values(k8[written], scales[written], rotated, name, allow_bf16_ulp)

    # Chunked writes must match a single call.
    split_cache = allocate_cache(device, num_blocks, block_size)
    chunk_size = max(1, len(k) // 3)
    for start in range(0, len(k), chunk_size):
        end = start + chunk_size
        call_op(rt, device, k[start:end], k_norm, k_norm_bias, freqs,
                positions[start:end], slots[start:end], split_cache)
    assert torch.equal(split_cache[0].cpu().reshape(-1, HEAD_DIM), k8), (name, "chunk K")
    assert torch.equal(split_cache[1].cpu().flatten(), scales), (name, "chunk scale")

    # Repeated writes must preserve the full cache.
    k_again, scales_again = call_op(rt, device, *inputs, caches)
    assert torch.equal(k_again, k8) and torch.equal(scales_again, scales), (name, "repeat")
    logging.info("indexer_k_cache_c8 %s (%d tokens) passed", name, len(k))
    return result, k8[written], scales[written]


def run_synthetic_tests(rt, device):
    generator = torch.Generator().manual_seed(9128)
    results = []
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
        result, _, _ = run_test(rt, device, f"random-{tokens}", k, k_norm, k_norm_bias,
                                freqs, positions, slots, 12, 32)
        results.append(result)

        # Complex and FP32 frequencies must produce identical caches.
        expected = call_op(rt, device, k, k_norm, k_norm_bias, freqs_fp32, positions, slots,
                           allocate_cache(device, 12, 32))
        actual = call_op(rt, device, k, k_norm, k_norm_bias, freqs_complex, positions, slots,
                         allocate_cache(device, 12, 32))
        for name, value, reference_value in zip(("K8", "K scale"), actual, expected):
            assert torch.equal(value, reference_value), ("complex frequencies", name)
        results.append({"case": "complex-frequencies", "tokens": tokens})
        logging.info("indexer_k_cache_c8 complex frequencies (%d tokens) passed", tokens)

    edge_cases = [
        ("zero", torch.zeros(17, HEAD_DIM).bfloat16(), k_norm, k_norm_bias),
        ("constant", torch.full((17, HEAD_DIM), 4, dtype=torch.bfloat16), k_norm, k_norm_bias),
        ("tiny-scale", torch.zeros(17, HEAD_DIM).bfloat16(), k_norm,
         torch.arange(HEAD_DIM).float() * 1e-10),
        ("affine-only", torch.randn(17, HEAD_DIM, generator=generator).bfloat16(), k_norm * 0,
         torch.randn(HEAD_DIM, generator=generator)),
        ("epsilon", (torch.randn(17, HEAD_DIM, generator=generator) * 1e-4).bfloat16(),
         k_norm, k_norm_bias),
    ]
    for name, k, weight, bias in edge_cases:
        result, _, _ = run_test(rt, device, name, k, weight, bias, freqs, torch.arange(17),
                                torch.arange(17).int() + 31, 3, 32)
        results.append(result)

    # Exercise an outlier and a row stride not aligned to 32 bytes.
    k_outlier = torch.randn(17, HEAD_DIM + 1, generator=generator).bfloat16()
    k_outlier[:, 0] = 1024
    result, _, _ = run_test(rt, device, "odd-stride-outlier", k_outlier, k_norm, k_norm_bias,
                            freqs, torch.arange(17), torch.arange(17).int() + 31, 3, 32)
    results.append(result)

    # Guard both ends of cache views with unaligned addresses.
    k_base = torch.full((32 * HEAD_DIM + 2,), K_CACHE_FILL, dtype=torch.int8, device=device)
    scale_base = torch.full((32 + 2,), SCALE_CACHE_FILL, dtype=torch.float16, device=device)
    caches = (k_base[1:-1].view(1, 32, 1, HEAD_DIM), scale_base[1:-1].view(1, 32, 1, 1))
    k8, scales = call_op(rt, device, k_outlier, k_norm, k_norm_bias, freqs,
                         torch.arange(17), torch.arange(17).int(), caches)
    assert bool((k_base.cpu()[[0, -1]] == K_CACHE_FILL).all())
    assert bool((scale_base.cpu()[[0, -1]] == SCALE_CACHE_FILL).all())
    assert bool((k8[17:] == K_CACHE_FILL).all()) and bool((scales[17:] == SCALE_CACHE_FILL).all())
    results.append(check_values(
        k8[:17], scales[:17], reference(k_outlier, k_norm, k_norm_bias, freqs, torch.arange(17)),
        "offset-cache-views"))
    logging.info("indexer_k_cache_c8 offset-cache-views passed")

    # Unused metadata capacity is ignored.
    positions = torch.cat((torch.arange(17), torch.full((15,), -100)))
    slots = torch.cat((torch.arange(17).int(), torch.full((15,), 32, dtype=torch.int32)))
    padded_k8, padded_scales = call_op(rt, device, k_outlier, k_norm, k_norm_bias, freqs,
                                       positions, slots, allocate_cache(device, 1, 32))
    assert torch.equal(padded_k8, k8) and torch.equal(padded_scales, scales)
    logging.info("indexer_k_cache_c8 metadata-capacity passed")
    return results


def run_invalid_tests(rt, device):
    k = torch.zeros(3, HEAD_DIM, dtype=torch.bfloat16)
    args = [k, torch.ones(HEAD_DIM), torch.zeros(HEAD_DIM), frequencies(8),
            torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2], dtype=torch.int32)]
    mutations = [
        (0, k.float()), (1, torch.ones(HEAD_DIM).half()),
        (3, frequencies(8).half()),
    ]
    count = 0
    for index, replacement in mutations:
        changed = list(args)
        changed[index] = replacement
        caches = allocate_cache(device, 1, 32)
        try:
            call_op(rt, device, *changed, caches)
        except (RuntimeError, ValueError):
            assert bool((caches[0].cpu() == K_CACHE_FILL).all())
            assert bool((caches[1].cpu() == SCALE_CACHE_FILL).all())
            count += 1
        else:
            raise AssertionError(("invalid input accepted", index, replacement.shape))

    bad_cache = torch.empty(1, 32, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    try:
        call_op(rt, device, *args, (bad_cache, allocate_cache(device, 1, 32)[1]))
    except (RuntimeError, ValueError):
        count += 1
    else:
        raise AssertionError("invalid K cache accepted")

    logging.info("indexer_k_cache_c8 rejected %d invalid inputs", count)
    return {"rejected_invalid_inputs": count}


def run_fixture_tests(rt, device, folder):
    results = []
    paths = sorted(Path(folder).glob("native-fixture-*.pt"))
    assert len(paths) == 4, "Expected all four frozen native fixtures"
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        slots = data["slot_mapping"].flatten().int()
        positions = data["k_positions"].long()
        block_size = 128
        num_blocks = (int(slots.max()) + block_size) // block_size + 1
        result, k8, scales = run_test(
            rt, device, path.stem, data["k_projected"], data["k_norm_weight"], data["k_norm_bias"],
            frequencies(int(positions.max()) + 1), positions, slots, num_blocks, block_size,
            data["k_rot"], allow_bf16_ulp=True)
        native_k8 = data["k8"].reshape(-1, HEAD_DIM)
        result["native_byte_mismatches"] = int((k8 != native_k8).sum())
        result["native_max_byte_delta"] = int((k8.int() - native_k8.int()).abs().max())
        result["native_byte_mismatch_fraction"] = result["native_byte_mismatches"] / k8.numel()
        # Bound native INT8 differences; scales must match exactly.
        assert result["native_max_byte_delta"] <= NATIVE_MAX_BYTE_DELTA
        assert result["native_byte_mismatch_fraction"] <= NATIVE_MAX_MISMATCH_FRACTION
        torch.testing.assert_close(scales, data["k_scale"].flatten().half(), rtol=0, atol=0)
        logging.info("%s: %d native byte differences (max delta %d)", path.stem,
                     result["native_byte_mismatches"], result["native_max_byte_delta"])
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch_npu  # noqa: F401
    from xlite._C import Runtime

    logging.getLogger().setLevel(logging.INFO)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    rt = Runtime(device.index, 32)

    results = run_synthetic_tests(rt, device)
    results.append(run_invalid_tests(rt, device))
    if args.fixtures:
        results.extend(run_fixture_tests(rt, device, args.fixtures))
    if args.output:
        args.output.write_text(json.dumps({
            "passed": True,
            "acceptance": "bounded numerical error, not bitwise native equivalence",
            "full_model_or_topk_tested": False,
            "results": results,
        }, indent=2) + "\n")
    logging.info("PASS: standalone LI-C8 writer; model/Top-K integration is not tested")


if __name__ == "__main__":
    main()
