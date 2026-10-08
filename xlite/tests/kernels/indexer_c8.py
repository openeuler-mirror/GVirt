#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""C8 Indexer prepare and Top-K tests."""
import logging

import torch

from indexer_k_cache_c8 import NORM_EPS, check_values, frequencies, reference, rotation


HEAD_DIM = 128
N_HEADS = 32
ROPE_DIM = 64
TOP_K = 2048
SCORE_ATOL = 1e-7
SCORE_RTOL = 3e-6
K_CACHE_FILL = -101
SCALE_CACHE_FILL = -17
TOPK_FILL = -777

npu_id = 0

# Input pattern and token count.
prepare_cases = [
    ("random", 1),
    ("random", 17),
    ("random", 128),
    ("random", 129),
    ("quantization", 4),
    ("sign", 4),
]

# Key lengths, query lengths, block size, topK.
test_cases = [
    ([33], [4], 16, 32),
    ([63, 4197], [5, 7], 64, 2048),
    ([65, 8257], [3, 9], 128, 512),
    ([47, 4103], [5, 9], 32, 2048),
    ([2049], [4], 128, 2048),
    ([4097], [4], 128, 2048),
]


def reference_q(q, weight, freqs, positions, device):
    import torch_npu

    pairs = q[..., :ROPE_DIM].float().reshape(-1, N_HEADS, ROPE_DIM // 2, 2)
    trig = freqs[positions].float().reshape(-1, 1, ROPE_DIM // 2, 2)
    real = pairs[..., 0] * trig[..., 0] - pairs[..., 1] * trig[..., 1]
    imag = pairs[..., 1] * trig[..., 0] + pairs[..., 0] * trig[..., 1]
    rope = q.clone()
    rope[..., :ROPE_DIM] = torch.stack((real, imag), -1).flatten(-2).bfloat16()
    rotated = (rope.float() @ rotation().float()).bfloat16()

    active = rotated.float().abs().amax(-1) != 0
    q8 = torch.zeros_like(q, dtype=torch.int8)
    q_scale = torch.zeros(q.shape[:2], dtype=torch.float16)
    if active.any():
        quantized, scale = torch_npu.npu_dynamic_quant(rotated[active].to(device), dst_type=torch.int8)
        q8[active], q_scale[active] = quantized.cpu(), scale.cpu().half()
    scaled_weights = (weight.float() * q_scale.float()).half()
    return rotated, q8.reshape(len(q), -1), q_scale, scaled_weights


def run_prepare_test(rt, device, case, tokens):
    from xlite._C import indexer_prepare

    generator = torch.Generator().manual_seed(912403)
    k_norm = torch.randn(HEAD_DIM, generator=generator)
    k_norm_bias = torch.randn(HEAD_DIM, generator=generator)
    kw = torch.randn(tokens, HEAD_DIM + N_HEADS, generator=generator).bfloat16()
    q = torch.randn(tokens, N_HEADS, HEAD_DIM, generator=generator).bfloat16()
    freqs = frequencies(256)
    positions = torch.arange(tokens).long() % len(freqs)
    slots = torch.arange(tokens).int() + 3
    norm_eps = NORM_EPS
    if case == "random":
        if tokens > 1:
            slots[::7] = -1
        if tokens == 17:
            # Exercise a non-default epsilon in the Prepare entry.
            kw[:, :HEAD_DIM] *= 1e-4
            k_norm.fill_(1)
            k_norm_bias.zero_()
            norm_eps = 1e-5
    else:
        kw.zero_()
        kw[:, HEAD_DIM:] = 1
        k_norm.fill_(1)
        k_norm_bias.zero_()
        positions.zero_()
        freqs = frequencies(1)
        if case == "quantization":
            q.zero_()
            gains = torch.arange(1, N_HEADS + 1).bfloat16()
            q[0, :, 0], q[0, :, 1] = 3 * gains, gains
            q[1] = -q[0]
            q[3] = q[0] * (2.0 ** -30)
        else:
            q = torch.eye(HEAD_DIM).bfloat16().reshape(tokens, N_HEADS, HEAD_DIM)
            freqs.zero_()
            freqs[:, 1::2] = 1

    active = slots >= 0
    written = slots[active].long()
    rotated_k = reference(kw[active], k_norm, k_norm_bias, freqs, positions[active], norm_eps)
    rotated_q, q8_ref, q_scale_ref, weights_ref = reference_q(
        q, kw[:, HEAD_DIM:], freqs, positions, device)
    block_size = 32
    num_blocks = (tokens + 3 + block_size - 1) // block_size
    untouched = torch.ones(num_blocks * block_size, dtype=torch.bool)
    untouched[written] = False
    cache_baseline = q_baseline = None
    modes = (False, True) if case == "random" else (True,)
    freq_types = (torch.bfloat16, torch.float32, torch.complex64) if case == "random" else (torch.bfloat16,)

    for is_long in modes:
        for freq_type in freq_types:
            if freq_type == torch.complex64:
                table = torch.view_as_complex(freqs.float().view(len(freqs), ROPE_DIM // 2, 2))
            else:
                table = freqs.to(freq_type)
            kw_npu = kw.to(device)
            q_npu = q.reshape(tokens, -1).to(device) if is_long else torch.empty(
                0, dtype=torch.bfloat16, device=device)
            k_cache = torch.full((num_blocks, block_size, 1, HEAD_DIM), K_CACHE_FILL,
                                 dtype=torch.int8, device=device)
            k_scale_cache = torch.full((num_blocks, block_size, 1, 1), SCALE_CACHE_FILL,
                                       dtype=torch.float16, device=device)
            q8 = torch.empty_like(q_npu, dtype=torch.int8) if is_long else None
            q_scale = torch.empty((tokens, N_HEADS), dtype=torch.float16, device=device) if is_long else None
            scaled_weights = torch.empty_like(q_scale) if is_long else None

            torch.npu.synchronize(device)
            indexer_prepare(rt, kw_npu, k_norm.to(device), k_norm_bias.to(device), table.to(device),
                            positions.to(device), HEAD_DIM, N_HEADS, ROPE_DIM, block_size, k_cache,
                            slots.to(device), norm_eps, q_npu, 1 / 64, TOP_K, is_long,
                            k_scale_cache=k_scale_cache, q8=q8, q_scale=q_scale,
                            scaled_weights=scaled_weights)
            torch.npu.synchronize(device)
            assert torch.equal(kw_npu.cpu(), kw), "C8 prepare changed kw"

            cache_result = (k_cache.cpu().view(-1, HEAD_DIM), k_scale_cache.cpu().flatten())
            if cache_baseline is None:
                check_values(cache_result[0][written], cache_result[1][written], rotated_k, case)
                assert bool((cache_result[0][untouched] == K_CACHE_FILL).all()), "K padding changed"
                assert bool((cache_result[1][untouched] == SCALE_CACHE_FILL).all()), "K scale padding changed"
                cache_baseline = cache_result
            else:
                for value, expected in zip(cache_result, cache_baseline):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)

            if is_long:
                assert torch.equal(q_npu.cpu(), q.reshape(tokens, -1)), "C8 prepare changed Q"
                q_result = (q8.cpu(), q_scale.cpu(), scaled_weights.cpu())
                if q_baseline is None:
                    if case == "random":
                        check_values(q_result[0].reshape(-1, HEAD_DIM), q_result[1].flatten(),
                                     rotated_q.reshape(-1, HEAD_DIM), "Q")
                    else:
                        torch.testing.assert_close(q_result[0], q8_ref, rtol=0, atol=0)
                    torch.testing.assert_close(q_result[1], q_scale_ref, rtol=0, atol=0)
                    torch.testing.assert_close(q_result[2], weights_ref, rtol=0, atol=0)
                    q_baseline = q_result
                else:
                    for value, expected in zip(q_result, q_baseline):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
            logging.info("indexer_c8 prepare %s (%d tokens, %s, is_long=%s) passed",
                         case, tokens, freq_type, is_long)


def score_reference(q8, k8, scaled_weights, k_scale):
    dot = q8.reshape(-1, HEAD_DIM).int() @ k8.int().T
    dot = dot.reshape(len(q8), N_HEADS, len(k8)).clamp_min(0)
    return ((dot.float() / 1024).half().float()
            * scaled_weights.float()[..., None]).sum(1) * k_scale.float()[None, :]


def run_topk_test(rt, device, case, key_lengths, query_lengths, block_size, topk, generator):
    from xlite._C import indexer_topk

    max_num_blocks = (max(key_lengths) + block_size - 1) // block_size
    block_table = torch.randperm(max_num_blocks * len(key_lengths), generator=generator)
    block_table = block_table.reshape(len(key_lengths), max_num_blocks).int()
    slots = torch.cat([
        block_table[i, torch.arange(key_len) // block_size].long() * block_size
        + torch.arange(key_len) % block_size for i, key_len in enumerate(key_lengths)])
    num_blocks = (int(slots.max()) + block_size) // block_size
    k_cache = torch.full((num_blocks, block_size, 1, HEAD_DIM), K_CACHE_FILL,
                         dtype=torch.int8, device=device)
    k_scale_cache = torch.full((num_blocks, block_size, 1, 1), SCALE_CACHE_FILL,
                               dtype=torch.float16, device=device)
    q8 = torch.randint(-127, 128, (sum(query_lengths), N_HEADS, HEAD_DIM), generator=generator).char()
    k8 = torch.randint(-127, 128, (sum(key_lengths), HEAD_DIM), generator=generator).char()
    k_scale = (torch.rand(sum(key_lengths), generator=generator) * .1).half()
    scaled_weights = (torch.randn(sum(query_lengths), N_HEADS, generator=generator) * .1).half()
    if case == 3:
        q8.zero_()  # Ties across KV tiles and padding.
    k_cache.view(-1, HEAD_DIM)[slots.to(device)] = k8.to(device)
    k_scale_cache.view(-1)[slots.to(device)] = k_scale.to(device)
    lens = torch.tensor(query_lengths, dtype=torch.int32)
    cached_lens = torch.tensor(key_lengths, dtype=torch.int32) - lens
    query_start_loc = torch.tensor([0] + query_lengths, dtype=torch.int32).cumsum(0)[:-1].int()
    output = torch.full((len(q8), topk), TOPK_FILL, dtype=torch.int32, device=device)

    torch.npu.synchronize(device)
    indexer_topk(rt, q8.reshape(-1, N_HEADS * HEAD_DIM).to(device), k_cache,
                 scaled_weights.to(device), output, query_start_loc.to(device), lens.to(device),
                 cached_lens.to(device), block_table.to(device), N_HEADS, HEAD_DIM, block_size,
                 len(query_lengths), topk, k_scale_cache=k_scale_cache)
    torch.npu.synchronize(device)
    actual = output.cpu()

    query_offset = key_offset = 0
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
            assert len(valid) == topk and len(valid.unique()) == topk, (
                "Top-K count/uniqueness", query_offset + i, visible)
            assert bool((valid < visible).all()), ("visibility violation", i, visible, valid.max())
            excluded = torch.ones(visible, dtype=torch.bool)
            excluded[valid] = False
            if bool(excluded.any()):
                gap = float(scores[i, :visible][excluded].max() - scores[i, valid].min())
                # Allow FP32 reduction-order differences at the Top-K boundary.
                tolerance = max(SCORE_ATOL, float(scores[i, :visible].abs().max()) * SCORE_RTOL)
                assert gap <= tolerance, ("Top-K score boundary", gap, tolerance)
        query_offset += query_len
        key_offset += key_len
    logging.info("indexer_c8 Top-K case %d (block_size=%d, topK=%d) passed", case, block_size, topk)


def main():
    import torch_npu  # noqa: F401
    from xlite._C import Runtime

    logging.getLogger().setLevel(logging.INFO)
    torch.npu.set_device(npu_id)
    torch.set_num_threads(4)
    device = torch.device(f"npu:{npu_id}")
    rt = Runtime(device.index, 160)

    for case, tokens in prepare_cases:
        run_prepare_test(rt, device, case, tokens)
    generator = torch.Generator().manual_seed(90128)
    for case, (key_lengths, query_lengths, block_size, topk) in enumerate(test_cases):
        run_topk_test(rt, device, case, key_lengths, query_lengths, block_size, topk, generator)
    logging.info("indexer_c8 tests passed")


if __name__ == "__main__":
    main()
