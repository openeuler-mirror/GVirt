#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""DSA-CP metadata, replicated cache writes and all-head MLA kernel regression.

Run on one NPU: pytest -q tests/kernels/dsa_cp.py
Logical TP shards are exercised sequentially; this does not test collectives or
establish GLM end-to-end accuracy.
"""

import math

import pytest
import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, dsa_cp_metadata, dsa_cp_write_kv_cache_by_slot, mla_v2


@pytest.fixture(scope="module")
def runtime():
    torch.npu.set_device(0)
    return Runtime(0, size=512)


def tensor(values):
    return torch.tensor(values, dtype=torch.int32, device="npu")


@pytest.mark.parametrize(
    "query_lens,cached_lens",
    [
        ([1], [17]),
        ([1, 1, 1], [9, 33, 128]),
        ([17], [129]),
        ([3, 17, 1], [0, 129, 7]),
    ],
)
def test_metadata(runtime, query_lens, cached_lens):
    query_start_loc = [sum(query_lens[:i]) for i in range(len(query_lens))]
    total = sum(query_lens)
    local_token_capacity = math.ceil(total / 16)
    inputs = [tensor(values) for values in (query_start_loc, query_lens, cached_lens)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(3)]
    local_slot_mapping = torch.empty(local_token_capacity, dtype=torch.int32, device="npu")
    reconstructed = []
    for rank in range(16):
        shard_begin = min(rank * local_token_capacity, total)
        shard_end = min((rank + 1) * local_token_capacity, total)
        torch.npu.synchronize()
        dsa_cp_metadata(runtime, *inputs, *outputs, local_slot_mapping, shard_begin, shard_end)
        local_query_start_loc, local_query_lens, local_cached_lens = [x.cpu().tolist() for x in outputs]
        assert local_slot_mapping.cpu().tolist() == list(range(local_token_capacity))
        seen = []
        for request, (request_start, length) in enumerate(zip(query_start_loc, query_lens)):
            expected = [q for q in range(request_start, request_start + length) if shard_begin <= q < shard_end]
            assert local_query_lens[request] == len(expected)
            if expected:
                assert local_query_start_loc[request] == expected[0] - shard_begin
                assert local_cached_lens[request] == cached_lens[request] + expected[0] - request_start
                seen.extend(expected)
        assert seen == list(range(shard_begin, shard_end))
        reconstructed.extend(seen)
    assert reconstructed == list(range(total))


@pytest.mark.parametrize(
    "dtype,width", [(torch.bfloat16, 512), (torch.bfloat16, 64), (torch.int8, 128), (torch.float16, 1)]
)
@pytest.mark.parametrize("pitched_input", [False, True])
def test_dsa_cp_write_kv_cache_by_slot(runtime, dtype, width, pitched_input):
    block_size = 16
    # Gaps between physical blocks must remain untouched.
    storage = torch.full((8, block_size, width), -3, dtype=dtype, device="npu")
    cache = storage[::2]
    input_width = width * 2 if pitched_input else width
    packed = torch.arange(5 * input_width, device="npu").reshape(5, input_width).to(dtype)
    new = packed[:, width:] if pitched_input else packed
    slots = tensor([0, 17, -1, 63, 9])
    expected = storage.clone()
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            expected[2 * (slot // block_size), slot % block_size] = new[row]
    torch.npu.synchronize()
    dsa_cp_write_kv_cache_by_slot(runtime, new, cache, slots, len(slots), block_size)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("lengths,cached", [([17], [129]), ([1, 3, 1], [129, 0, 32])])
def test_local_all_head_attention(runtime, heads, flash, sparse, lengths, cached):
    torch.manual_seed(42)
    block_size, nope, rope, latent, value = 128, 192, 64, 512, 256
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    total, batch = sum(lengths), len(lengths)
    blocks = math.ceil(max(q + c for q, c in zip(lengths, cached)) / block_size)
    max_seq = blocks * block_size

    def random(shape):
        return (torch.randn(shape) * 0.05).bfloat16()

    q = random((total, heads, nope + rope))
    k = random((batch, max_seq, latent))
    pe = random((batch, max_seq, rope))
    uk = random((heads, nope, latent))
    uv = random((heads, latent, value))
    q_absorb = torch.einsum("thd,hdl->thl", q[..., :nope].float(), uk.float()).bfloat16()
    scale = (nope + rope) ** -0.5
    expected = torch.empty(total, heads, value)
    top_k = 128 if sparse else 0
    indices = torch.empty(total, top_k, dtype=torch.int32)
    for request, (start, length) in enumerate(zip(starts, lengths)):
        for t in range(length):
            end = cached[request] + t + 1
            first = max(0, end - top_k) if sparse else 0
            if sparse:
                indices[start + t].fill_(end)
                indices[start + t, : end - first] = torch.arange(first, end)
            scores = q_absorb[start + t].float() @ k[request, first:end].float().T
            scores += q[start + t, :, nope:].float() @ pe[request, first:end].float().T
            out = (torch.softmax(scores * scale, dim=-1) @ k[request, first:end].float()).bfloat16()
            expected[start + t] = torch.einsum("hl,hlv->hv", out.float(), uv.float())
    q, uk, uv = q.npu(), uk.npu(), uv.npu()
    k = k.reshape(batch * blocks, block_size, latent).npu()
    pe = pe.reshape(batch * blocks, block_size, rope).npu()
    tables = torch.arange(batch * blocks, dtype=torch.int32, device="npu").reshape(batch, blocks)
    inputs = [tensor(values) for values in (starts, lengths, cached)]
    metadata = [torch.empty_like(inputs[0]) for _ in range(3)]
    local_token_capacity = math.ceil(total / 4)
    local_slot_mapping = torch.empty(local_token_capacity, dtype=torch.int32, device="npu")
    for rank in range(4):
        shard_begin = min(rank * local_token_capacity, total)
        shard_end = min((rank + 1) * local_token_capacity, total)
        shard_len = shard_end - shard_begin
        torch.npu.synchronize()
        dsa_cp_metadata(runtime, *inputs, *metadata, local_slot_mapping, shard_begin, shard_end)
        if shard_len == 0:
            continue
        local_q = q[shard_begin:shard_end].contiguous()
        qr = local_q[..., nope:].contiguous()
        output = torch.empty(shard_len, heads, value, dtype=torch.bfloat16, device="npu")
        local_indices = indices[shard_begin:shard_end].contiguous().npu()
        torch.npu.synchronize()
        mla_v2(
            runtime,
            local_q,
            qr,
            k,
            pe,
            uk,
            uv,
            output,
            *metadata,
            tables,
            heads,
            rope,
            nope,
            value,
            latent,
            block_size,
            batch,
            scale,
            local_indices,
            top_k=top_k,
            nz=False,
            enable_flash_attention=flash,
            tile_size_of_cached_kv=128,
        )
        torch.testing.assert_close(output.float().cpu(), expected[shard_begin:shard_end], atol=2e-3, rtol=0.05)
