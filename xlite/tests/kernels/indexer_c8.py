#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""C8 Indexer prepare and Top-K tests.

Compare with PyTorch and optional native fixtures.
"""
import argparse
import json
import logging
from pathlib import Path

import torch

from indexer_k_cache_c8 import check_values, frequencies, rotation


HEAD_DIM = 128
N_HEADS = 32
ROPE_DIM = 64
TOP_K = 2048
MAX_KV_TILE_LEN = 4096
SCORE_ATOL = 1e-7
SCORE_RTOL = 3e-6
NATIVE_MAX_BYTE_DELTA = 1
NATIVE_MAX_MISMATCH_FRACTION = 0.001
K_CACHE_FILL = -101
SCALE_CACHE_FILL = -17
TOPK_FILL = -777

# Token counts for Q/K phase tests.
prepare_tokens = [1, 17, 128, 129]

# Key lengths, query lengths, block size, topK.
test_cases = [
    ([33], [4], 16, 32),
    ([63, 4197], [5, 7], 64, 2048),
    ([65, 8257], [3, 9], 128, 512),
    ([47, 4103], [5, 9], 32, 2048),
    ([2049], [4], 128, 2048),
    ([4097], [4], 128, 2048),
]


def score_reference(q8, k8, scaled_weights, k_scale):
    dot = q8.reshape(-1, HEAD_DIM).int() @ k8.int().T
    dot = dot.reshape(len(q8), N_HEADS, len(k8)).clamp_min(0)
    return ((dot.float() / 1024).half().float() *
            scaled_weights.float()[..., None]).sum(1) * k_scale.float()[None, :]


def check_topk(actual, q8, k8, scaled_weights, k_scale, key_lengths, query_lengths,
               topk):
    query_offset = key_offset = 0
    rows = []
    for key_len, query_len in zip(key_lengths, query_lengths):
        scores = score_reference(
            q8[query_offset:query_offset + query_len], k8[key_offset:key_offset + key_len],
            scaled_weights[query_offset:query_offset + query_len],
            k_scale[key_offset:key_offset + key_len])
        for i in range(query_len):
            visible = key_len - query_len + i + 1
            if visible <= topk:
                continue  # Dense rows do not use Top-K.
            indices = actual[query_offset + i, :topk].long()
            valid = indices[indices >= 0]
            expected_count = topk
            assert len(valid) == expected_count and len(valid.unique()) == expected_count, (
                "Top-K count/uniqueness", query_offset + i, visible, expected_count, len(valid),
                len(valid.unique()), indices[:80].tolist())
            assert bool((valid < visible).all()), ("visibility violation", i, visible, valid.max())
            reference = scores[i, :visible].topk(expected_count).indices
            overlap = len(set(valid.tolist()) & set(reference.tolist()))
            excluded = torch.ones(visible, dtype=torch.bool)
            excluded[valid] = False
            gap = 0.0
            if bool(excluded.any()):
                gap = float(scores[i, :visible][excluded].max() - scores[i, valid].min())
                # Allow FP32 reduction-order differences at the Top-K boundary.
                tolerance = max(SCORE_ATOL, float(scores[i, :visible].abs().max()) * SCORE_RTOL)
                assert gap <= tolerance, ("Top-K score boundary", gap, tolerance, overlap)
            rows.append({"visible": visible, "selected": expected_count,
                         "oracle_set_overlap": overlap / expected_count, "boundary_gap": gap})
        query_offset += query_len
        key_offset += key_len
    return rows


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


def call_topk(rt, device, q8, scaled_weights, caches, block_table, key_lengths, query_lengths,
              topk=TOP_K):
    from xlite._C import indexer_topk

    q = q8.reshape(-1, N_HEADS * HEAD_DIM).to(device)
    weight = scaled_weights.to(device)
    lens = torch.tensor(query_lengths, dtype=torch.int32).to(device)
    cached_lens = (torch.tensor(key_lengths, dtype=torch.int32) - lens.cpu()).to(device)
    query_start_loc = torch.tensor([0] + query_lengths, dtype=torch.int32).cumsum(0)[:-1].int()
    query_start_loc = query_start_loc.to(device)
    block_tables = block_table.to(device)
    capacity = max(MAX_KV_TILE_LEN, block_table.shape[1] * caches[0].shape[1])
    indices = torch.arange(capacity, dtype=torch.int32).to(device)
    output = torch.full((len(q), topk), TOPK_FILL, dtype=torch.int32, device=device)
    torch.npu.synchronize(device)

    for repeat in range(4):
        indexer_topk(rt, q, caches[0], weight, indices, output, query_start_loc,
                     lens, cached_lens, block_tables, N_HEADS, HEAD_DIM, caches[0].shape[1],
                     len(query_lengths), topk, k_scale_cache=caches[1])
        actual = output.cpu()
        if repeat == 0:
            first = actual
            continue
        query_offset = 0
        for key_len, query_len in zip(key_lengths, query_lengths):
            start = min(query_len, max(0, topk - (key_len - query_len)))
            assert torch.equal(actual[query_offset + start:query_offset + query_len],
                               first[query_offset + start:query_offset + query_len]), (
                "Top-K repeat", query_offset, start)
            query_offset += query_len
    return first


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


def run_sign_test(rt, device):
    """Exercise RoPE signs and all Hadamard stages with basis vectors."""
    q = torch.eye(HEAD_DIM).bfloat16().reshape(-1, N_HEADS, HEAD_DIM)
    tokens = len(q)
    freqs = torch.zeros(1, ROPE_DIM, dtype=torch.bfloat16)
    freqs[:, 1::2] = 1
    rope = q.clone()
    rope[..., :ROPE_DIM:2] = -q[..., 1:ROPE_DIM:2]
    rope[..., 1:ROPE_DIM:2] = q[..., :ROPE_DIM:2]
    rotated = (rope.float() @ rotation().float()).bfloat16()
    expected_q = rotated.sign().to(torch.int8) * 127
    expected_scale = (rotated.float().abs().amax(-1) / 127).half()

    kw = torch.zeros(tokens, HEAD_DIM + N_HEADS, dtype=torch.bfloat16)
    kw[:, HEAD_DIM:] = 1
    slots = torch.arange(tokens).int()
    actual = call_prepare(rt, device, kw, torch.ones(HEAD_DIM), torch.zeros(HEAD_DIM),
                          freqs, torch.zeros(tokens, dtype=torch.int64), slots,
                          allocate_cache(device, slots, 32), q.reshape(tokens, -1))
    torch.testing.assert_close(actual["q8"].reshape_as(q), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual["q_scale"], expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual["scaled_weights"], expected_scale, rtol=0, atol=0)
    logging.info("indexer_c8 sign patterns passed")
    return {"tokens": tokens, "heads": N_HEADS}


def run_fixture_tests(rt, device, directory):
    paths = sorted(directory.glob("native-fixture-*.pt"))
    assert len(paths) == 4, "All four frozen native fixtures are required"
    results = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        key_lengths = data["key_lengths"].tolist()
        query_lengths = data["query_lengths"].tolist()
        out_nums = [TOP_K if key_len - query_len + i >= TOP_K else 0
                    for key_len, query_len in zip(key_lengths, query_lengths)
                    for i in range(query_len)]
        slots, block_table = data["slot_mapping"].long(), data["block_table"]
        caches = allocate_cache(device, slots, 128)
        caches[0].view(-1, HEAD_DIM)[slots.to(device)] = data["k8"].to(device)
        caches[1].view(-1)[slots.to(device)] = data["k_scale"].to(device)
        scaled_weights = (data["weights"].float() * data["q_scale"].float()).half()
        selected = call_topk(rt, device, data["q8"], scaled_weights, caches, block_table,
                             key_lengths, query_lengths)
        fixed_rows = check_topk(selected, data["q8"], data["k8"], scaled_weights,
                                data["k_scale"], key_lengths, query_lengths, TOP_K)
        native_overlap = []
        for actual, expected, out_num in zip(selected, data["native_topk"], out_nums):
            if out_num == 0:
                continue
            actual, expected = actual[:out_num], expected[:out_num]
            actual_set = set(actual[actual >= 0].tolist())
            expected_set = set(expected[expected >= 0].tolist())
            native_overlap.append(len(actual_set & expected_set) / len(expected_set))

        # Write K history, then prepare the current Q/K.
        caches = allocate_cache(device, slots, 128)
        kw = torch.zeros(len(slots), HEAD_DIM + N_HEADS, dtype=torch.bfloat16)
        kw[:, :HEAD_DIM] = data["k_projected"]
        freqs = frequencies(max(key_lengths))
        call_prepare(rt, device, kw, data["k_norm_weight"], data["k_norm_bias"], freqs,
                     data["k_positions"], slots, caches)
        query_rows, key_offset = [], 0
        for key_len, query_len in zip(key_lengths, query_lengths):
            query_rows.extend(range(key_offset + key_len - query_len, key_offset + key_len))
            key_offset += key_len
        query_kw = kw[query_rows].clone()
        query_kw[:, HEAD_DIM:] = data["weights"].bfloat16()
        prepared = call_prepare(rt, device, query_kw, data["k_norm_weight"], data["k_norm_bias"],
                                freqs, data["q_positions"], slots[query_rows], caches,
                                data["q_projected"].reshape(-1, N_HEADS * HEAD_DIM))
        q8 = prepared["q8"].reshape(-1, N_HEADS, HEAD_DIM)
        q_scale, scaled_weights = prepared["q_scale"], prepared["scaled_weights"]
        q_stats = check_values(q8.reshape(-1, HEAD_DIM), q_scale.flatten(),
                               data["q_rot"].reshape(-1, HEAD_DIM), path.name, True)
        q_diff = (q8.int() - data["q8"].int()).abs()
        assert int(q_diff.max()) <= NATIVE_MAX_BYTE_DELTA
        assert float((q_diff != 0).float().mean()) <= NATIVE_MAX_MISMATCH_FRACTION
        q_stats.update(native_byte_mismatches=int((q_diff != 0).sum()),
                       native_max_byte_delta=int(q_diff.max()))
        torch.testing.assert_close(scaled_weights, (data["weights"].float() * q_scale.float()).half(),
                                   rtol=0, atol=0)
        k8 = caches[0].cpu().view(-1, HEAD_DIM)[slots]
        k_scale = caches[1].cpu().view(-1)[slots]
        k_stats = check_values(k8, k_scale, data["k_rot"], path.name, True)
        k_diff = (k8.int() - data["k8"].int()).abs()
        assert int(k_diff.max()) <= NATIVE_MAX_BYTE_DELTA
        assert float((k_diff != 0).float().mean()) <= NATIVE_MAX_MISMATCH_FRACTION
        k_stats.update(native_byte_mismatches=int((k_diff != 0).sum()),
                       native_max_byte_delta=int(k_diff.max()))
        selected = call_topk(rt, device, q8, scaled_weights, caches, block_table,
                             key_lengths, query_lengths)
        integrated_rows = check_topk(selected, q8, k8, scaled_weights, k_scale,
                                     key_lengths, query_lengths, TOP_K)
        final_overlap, final_differences = [], []
        for row, (actual, expected, out_num) in enumerate(zip(selected, data["native_topk"], out_nums)):
            if out_num == 0:
                continue
            actual, expected = actual[:out_num], expected[:out_num]
            actual_set = set(actual[actual >= 0].tolist())
            expected_set = set(expected[expected >= 0].tolist())
            final_overlap.append(len(actual_set & expected_set) / len(expected_set))
            if actual_set != expected_set:
                final_differences.append({"query": row, "native_only": sorted(expected_set - actual_set),
                                          "xlite_only": sorted(actual_set - expected_set)})

        # FP32 frequencies must produce the same cache.
        alternate = allocate_cache(device, slots, 128)
        freqs_fp32 = freqs.float().view(len(freqs), ROPE_DIM // 2, 2)
        call_prepare(rt, device, kw, data["k_norm_weight"], data["k_norm_bias"], freqs_fp32,
                     data["k_positions"], slots, alternate)
        assert torch.equal(alternate[0].cpu(), caches[0].cpu())
        assert torch.equal(alternate[1].cpu(), caches[1].cpu())
        results.append({"fixture": path.name, "fixed_bytes_oracle": fixed_rows,
                        "fixed_bytes_native_overlap": native_overlap,
                        "q_quantization": q_stats, "k_quantization": k_stats,
                        "prepared_oracle": integrated_rows,
                        "prepared_native_overlap": final_overlap,
                        "prepared_native_differences": final_differences})
        logging.info("indexer_c8 %s passed (%d sparse rows)", path.stem, len(integrated_rows))
        if native_overlap:
            logging.info("native overlap: fixed=%.6f, prepared=%.6f",
                         min(native_overlap), min(final_overlap))
    return results


def run_synthetic_tests(rt, device):
    generator = torch.Generator().manual_seed(90128)
    results = []
    for case, (key_lengths, query_lengths, block_size, topk) in enumerate(test_cases):
        max_num_blocks = (max(key_lengths) + block_size - 1) // block_size
        block_table = torch.randperm(max_num_blocks * len(key_lengths), generator=generator)
        block_table = block_table.reshape(len(key_lengths), max_num_blocks).int()
        slots = torch.cat([
            block_table[i, torch.arange(key_len) // block_size].long() * block_size +
            torch.arange(key_len) % block_size for i, key_len in enumerate(key_lengths)])
        caches = allocate_cache(device, slots, block_size)
        q8 = torch.randint(-127, 128, (sum(query_lengths), N_HEADS, HEAD_DIM), generator=generator).char()
        k8 = torch.randint(-127, 128, (sum(key_lengths), HEAD_DIM), generator=generator).char()
        k_scale = (torch.rand(sum(key_lengths), generator=generator) * .1).half()
        scaled_weights = (torch.randn(sum(query_lengths), N_HEADS, generator=generator) * .1).half()
        if case == 3:
            q8.zero_()  # Ties across KV tiles and padding.
        caches[0].view(-1, HEAD_DIM)[slots.to(device)] = k8.to(device)
        caches[1].view(-1)[slots.to(device)] = k_scale.to(device)
        actual = call_topk(rt, device, q8, scaled_weights, caches, block_table,
                           key_lengths, query_lengths, topk)
        results.append(check_topk(actual, q8, k8, scaled_weights, k_scale,
                                  key_lengths, query_lengths, topk))
        logging.info("indexer_c8 synthetic case %d (block_size=%d, topK=%d) passed",
                     case, block_size, topk)
    return results


def run_invalid_tests(rt, device):
    from xlite._C import Runtime, indexer_prepare, indexer_topk

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

    q8 = torch.zeros(2, N_HEADS * HEAD_DIM, dtype=torch.int8).to(device)
    scaled_weights = torch.zeros(2, N_HEADS).half().to(device)
    indices = torch.arange(32).int().to(device)
    output = torch.full((2, 32), TOPK_FILL, dtype=torch.int32).to(device)
    query_start_loc = torch.tensor([0], dtype=torch.int32).to(device)
    lens = torch.tensor([2], dtype=torch.int32).to(device)
    cached_lens = torch.tensor([0], dtype=torch.int32).to(device)
    block_table = torch.tensor([[0]], dtype=torch.int32).to(device)
    args = [rt, q8, caches[0], scaled_weights, indices, output, query_start_loc,
            lens, cached_lens, block_table, N_HEADS, HEAD_DIM, 32, 1, 32]
    topk_changes = [
        ({}, None, "unsupported LI-C8 inputs"),
        ({}, caches[1].float(), "unsupported LI-C8 inputs"),
        ({1: q8.bfloat16()}, caches[1], "unsupported LI-C8 inputs"),
        ({3: scaled_weights.float()}, caches[1], "unsupported LI-C8 inputs"),
        ({14: 31}, caches[1], "unsupported LI-C8 configuration"),
        ({14: TOP_K + 32}, caches[1], "topK should be less than or equal"),
    ]
    for arg_changes, scale, message in topk_changes:
        altered = list(args)
        for index, value in arg_changes.items():
            altered[index] = value
        # Failed calls do not return scratch tensors to the pool.
        altered[0] = Runtime(device.index, 160)
        torch.npu.synchronize(device)
        try:
            indexer_topk(*altered, k_scale_cache=scale)
        except (ValueError, RuntimeError) as error:
            assert message in str(error), str(error)
            rejected += 1
        else:
            raise AssertionError("Invalid C8 Top-K input was accepted")
        finally:
            torch.npu.synchronize(device)
            altered[0] = rt

    assert bool((output.cpu() == TOPK_FILL).all())
    logging.info("indexer_c8 rejected %d invalid inputs", rejected)
    return rejected


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
    torch.set_num_threads(4)
    device = torch.device(f"npu:{args.device}")
    rt = Runtime(device.index, 160)

    result = {"rejected_inputs": run_invalid_tests(rt, device),
              "prepare_phases": run_prepare_tests(rt, device),
              "quantization_boundary": run_quantization_test(rt, device),
              "sign_patterns": run_sign_test(rt, device),
              "synthetic": run_synthetic_tests(rt, device),
              "fixtures": run_fixture_tests(rt, device, args.fixtures) if args.fixtures else [],
              "causal": True,
              "full_model_accuracy_tested": False, "performance_tested": False}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    logging.info("indexer_c8 tests passed")


if __name__ == "__main__":
    main()
