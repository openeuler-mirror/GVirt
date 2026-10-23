# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Paged causal attention: FP32 QK scores, native BF16/FP16 KV pages."""
import math
import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, attention

torch.npu.set_device(0)
torch.manual_seed(42)
rt = Runtime(0, 512)
passed = 0
for dtype in (torch.bfloat16, torch.float16):
    for queries, cached, heads, kv_heads, dim, scale in (
        ([1], [0], 4, 1, 64, 1),
        ([17, 3, 1], [0, 129, 255], 8, 2, 128, 1),
        ([128], [0], 24, 4, 256, 1),
        ([28], [128], 24, 4, 256, 3),
        ([65, 1], [4031, 4095], 4, 1, 256, 3),
        ([1], [8191], 4, 1, 256, 1),
    ):
        block = 128
        cols = math.ceil(max(q + c for q, c in zip(queries, cached)) / block)
        batch = len(queries)
        # Non-sequential physical page numbers, including unused sentinel pages.
        table = torch.randperm(batch * cols).reshape(batch, cols) + 1
        k = torch.randn(batch * cols + 2, block, kv_heads, dim).to(dtype)
        v = torch.randn_like(k)
        q = (torch.randn(sum(queries), heads, dim) * scale / math.sqrt(dim)).to(dtype)
        qkv = torch.cat((q.flatten(1), torch.zeros(sum(queries), 2 * kv_heads * dim, dtype=dtype)), 1)
        expected = []
        off = 0
        for row, (query, cache) in enumerate(zip(queries, cached)):
            key = k[table[row]].reshape(-1, kv_heads, dim)[:query + cache]
            val = v[table[row]].reshape(-1, kv_heads, dim)[:query + cache]
            key = key.repeat_interleave(heads // kv_heads, 1).transpose(0, 1).float()
            val = val.repeat_interleave(heads // kv_heads, 1).transpose(0, 1).float()
            score = q[off:off + query].transpose(0, 1).float() @ key.transpose(1, 2)
            mask = torch.arange(query + cache)[None, :] > cache + torch.arange(query)[:, None]
            probability = score.masked_fill(mask, -float('inf')).softmax(-1).to(dtype)
            expected.append((probability.float() @ val).transpose(0, 1).flatten(1).to(dtype))
            off += query
        expected = torch.cat(expected)
        qkv, k, v, table = [t.npu() for t in (qkv, k, v, table.to(torch.int32))]
        # Exercise upstream K/V page pitches and offsets in the new score path.
        protected = []
        if scale == 3:
            for tensor in (k, v):
                inner = math.prod(tensor.shape[1:])
                raw = torch.full((32 + tensor.shape[0] * (inner + 128) + 32,),
                                 float('nan'), dtype=dtype, device='npu')
                view = raw.as_strided(tensor.shape, (inner + 128, *tensor.stride()[1:]), 32)
                view.copy_(tensor)
                protected.append((raw, raw.clone(), view))
            k, v = [item[2] for item in protected]
        starts = torch.tensor([sum(queries[:i]) for i in range(batch)], dtype=torch.int32, device='npu')
        lens = torch.tensor(queries, dtype=torch.int32, device='npu')
        cache_lens = torch.tensor(cached, dtype=torch.int32, device='npu')
        out = torch.empty(sum(queries), heads * dim, dtype=dtype, device='npu')
        torch.npu.synchronize()
        attention(rt, qkv, k, v, out, starts, lens, cache_lens, table,
                  heads, kv_heads, dim, block, batch, False, 8192, True)
        actual = out.cpu()
        for raw, before, _ in protected:
            torch.testing.assert_close(raw, before, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(actual, expected, rtol=.015, atol=.005)
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error < .005, error
        passed += 1
        print(f'PASS {dtype} queries={queries} cached={cached} relative_l2={error:.6g}', flush=True)
    before = out.clone()
    try:
        attention(rt, qkv, k, v, out, starts, lens, cache_lens, table,
                  heads, kv_heads, dim, block, batch, True, 8192, True)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError('FP32 scores accepted together with tiled Flash')
    assert torch.equal(out, before), 'rejected attention call changed output'
print(f'PASS {passed} FP32-score paged attention cases', flush=True)
