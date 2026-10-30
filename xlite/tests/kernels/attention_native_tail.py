#!/usr/bin/env python3
"""Native cache tail can contain NaNs: attention must read only valid tokens."""
import math

import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, attention


def run():
    torch.npu.set_device(0)
    torch.manual_seed(17)
    rt = Runtime(0, 64)
    heads, kv_heads, dim, block = 24, 4, 256, 128
    for dtype in (torch.bfloat16, torch.float16):
        for flash in (False, True):
            for lengths in ([24], [19, 24, 22], [128, 129, 145]):
                batch = len(lengths)
                blocks = math.ceil(max(lengths) / block)
                qkv = torch.randn(batch, (heads + 2 * kv_heads) * dim, device="npu", dtype=dtype)
                # Model RoPE scales Q before this standalone attention entry.
                qkv[:, :heads * dim].div_(math.sqrt(dim))
                k = torch.full((batch * blocks, block, kv_heads, dim), float("nan"), device="npu", dtype=dtype)
                v = torch.full_like(k, float("nan"))
                refs = []
                for b, length in enumerate(lengths):
                    keys = torch.randn(length, kv_heads, dim, device="npu", dtype=dtype)
                    values = torch.randn_like(keys)
                    for t in range(length):
                        k[b * blocks + t // block, t % block] = keys[t]
                        v[b * blocks + t // block, t % block] = values[t]
                    q = qkv[b, :heads * dim].view(heads, dim).float()
                    keys = keys.repeat_interleave(heads // kv_heads, dim=1).float()
                    values = values.repeat_interleave(heads // kv_heads, dim=1).float()
                    probs = torch.softmax(torch.einsum("hd,thd->ht", q, keys), dim=-1)
                    refs.append(torch.einsum("ht,thd->hd", probs, values).flatten())
                k_before, v_before = k.clone(), v.clone()
                output = torch.empty(batch, heads * dim, device="npu", dtype=dtype)
                meta = lambda values: torch.tensor(values, dtype=torch.int32, device="npu")
                torch.npu.synchronize()
                attention(rt, qkv, k, v, output, meta(list(range(batch))), meta([1] * batch),
                          meta([n - 1 for n in lengths]), meta(list(range(batch * blocks))),
                          heads, kv_heads, dim, block, batch, flash, 128)
                torch.npu.synchronize()
                reference = torch.stack(refs)
                # Softmax probabilities and cube outputs are rounded to Dtype.
                atol = 0.01 if dtype == torch.bfloat16 else 0.005
                torch.testing.assert_close(output.float(), reference, rtol=0.025, atol=atol)
                torch.testing.assert_close(k, k_before, rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(v, v_before, rtol=0, atol=0, equal_nan=True)
                error = (output.float() - reference).abs().max().item()
                print(f"PASS {dtype} flash={flash} lengths={lengths} max_abs_error={error:.6g}", flush=True)


if __name__ == "__main__":
    run()
