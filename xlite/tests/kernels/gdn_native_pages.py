#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Compare native padded pages with PyTorch; untouched pages/guards must match exactly."""

import argparse
import math

import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, linear_att_conv_and_silu_paged, recurrent_gated_delta_rule_paged


def run(device, metadata_mode):
    torch.npu.set_device(device)
    torch.manual_seed(7)
    rt = Runtime(device, 64)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        # A storage offset and padding expose addressing bugs even when pages are reordered.
        channels, width, pages, batch = 64, 4, 8, 4
        stride = 256
        raw = torch.randn(32 + pages * stride, dtype=dtype, device="npu")
        state = raw.as_strided((pages, width - 1, channels), (stride, channels, 1), 32)
        x = torch.randn(batch, channels, dtype=dtype, device="npu")
        weight = torch.randn(channels, width, dtype=dtype, device="npu")
        orders = ([6, 2, 3, 0], [2, 6, 4, -1])
        indices = torch.tensor(orders[0], dtype=torch.int32, device="npu")
        out = torch.empty_like(x)
        for order in orders:
            indices.copy_(torch.tensor(order, dtype=torch.int32, device="npu"))
            metadata = {}
            selected = indices
            if metadata_mode != "legacy":
                cached = torch.tensor([63, 64, 128, 0], dtype=torch.int32)
                columns = (
                    cached // 64
                    if metadata_mode == "align"
                    else torch.zeros_like(cached)
                )
                table = torch.full((batch, 3), 7, dtype=torch.int32)
                table[torch.arange(batch), columns.long()] = torch.tensor(
                    order, dtype=torch.int32
                )
                table[-1, 0] = 7
                storage = torch.full(
                    (32 + batch * 7 + 32,), -1, dtype=torch.int32, device="npu"
                )
                selected = storage.as_strided((batch, 3), (7, 1), 32)
                selected.copy_(table)
                metadata = {
                    "cached_lens": cached.to("npu"),
                    "state_block_size": 64,
                    "aligned": metadata_mode == "align",
                }
            before = raw.cpu().clone()
            ref = before.clone()
            ref_state = ref.as_strided(state.shape, state.stride(), 32)
            expected = torch.zeros_like(x.cpu())
            for b, page in enumerate(order):
                if page <= 0:
                    continue
                history = torch.cat(
                    (ref_state[page].float(), x[b : b + 1].cpu().float()), dim=0
                )
                expected[b] = torch.nn.functional.silu(
                    (history.T * weight.cpu().float()).sum(-1)
                ).to(dtype)
                ref_state[page].copy_(history[1:].to(dtype))
            torch.npu.synchronize()
            linear_att_conv_and_silu_paged(
                rt, x, state, weight, out, selected, **metadata
            )
            torch.testing.assert_close(out.cpu(), expected, rtol=0.025, atol=0.025)
            torch.testing.assert_close(raw.cpu(), ref, rtol=0, atol=0)
        for state_dtype in dict.fromkeys((dtype, torch.float32)):
            heads, key_dim, value_dim = 2, 128, 128
            logical = heads * key_dim * value_dim
            stride = logical + 128
            raw = (
                torch.randn(32 + pages * stride, dtype=state_dtype, device="npu") * 0.05
            )
            ssm = raw.as_strided(
                (pages, heads, value_dim, key_dim),
                (stride, key_dim * value_dim, key_dim, 1),
                32,
            )
            q = torch.randn(batch, heads * key_dim, dtype=dtype, device="npu") * 0.1
            k = torch.randn_like(q) * 0.1
            v = torch.randn(batch, heads * value_dim, dtype=dtype, device="npu") * 0.1
            beta = torch.rand(batch, heads, dtype=dtype, device="npu")
            g = -torch.rand_like(beta)
            out = torch.empty_like(v)
            for order in orders:
                indices.copy_(torch.tensor(order, dtype=torch.int32, device="npu"))
                ref = raw.cpu().clone()
                rs = ref.as_strided(ssm.shape, ssm.stride(), 32)
                expected = torch.zeros_like(v.cpu())
                for b, page in enumerate(order):
                    if page <= 0:
                        continue
                    old = rs[page].float() * g[b].cpu().float().exp()[:, None, None]
                    kk = k[b].cpu().float().reshape(heads, key_dim)
                    qq = q[b].cpu().float().reshape(heads, key_dim) / math.sqrt(key_dim)
                    delta = (
                        v[b].cpu().float().reshape(heads, value_dim)
                        - (old * kk[:, None, :]).sum(-1)
                    ) * beta[b].cpu().float()[:, None]
                    new = old + delta[:, :, None] * kk[:, None, :]
                    expected[b] = (new * qq[:, None, :]).sum(-1).flatten().to(dtype)
                    rs[page].copy_(new.to(state_dtype))
                before = raw.cpu().clone()
                torch.npu.synchronize()
                recurrent_gated_delta_rule_paged(
                    rt,
                    q,
                    k,
                    v,
                    beta,
                    g,
                    ssm,
                    out,
                    indices,
                    heads,
                    heads,
                    key_dim,
                    value_dim,
                )
                actual = raw.cpu()
                torch.testing.assert_close(out.cpu(), expected, rtol=0.025, atol=0.005)
                torch.testing.assert_close(actual, ref, rtol=0.025, atol=0.005)
                modified = torch.zeros_like(before, dtype=torch.bool)
                for page in order:
                    if page > 0:
                        modified[32 + page * stride : 32 + page * stride + logical] = (
                            True
                        )
                assert torch.equal(actual[~modified], before[~modified]), (
                    "unselected page/guard changed"
                )
        print(f"PASS {metadata_mode} native padded conv/SSM pages: {dtype}", flush=True)

    table = torch.ones(batch, 3, dtype=torch.int32, device="npu")
    cached = torch.full((batch,), 63, dtype=torch.int32, device="npu")
    conv = torch.zeros(pages, width - 1, channels, dtype=x.dtype, device="npu")
    for source, lengths in ((table.cpu(), cached), (table, cached.cpu())):
        before = conv.clone()
        try:
            linear_att_conv_and_silu_paged(
                rt,
                x,
                conv,
                weight,
                torch.empty_like(x),
                source,
                cached_lens=lengths,
                state_block_size=64,
            )
        except RuntimeError as error:
            assert "runtime NPU" in str(error), str(error)
        else:
            raise AssertionError("CPU conv metadata accepted")
        assert torch.equal(before, conv), "rejected call changed conv state"
    print("PASS conv metadata device rejections", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--metadata-mode", choices=("all", "legacy", "none", "align"), default="all"
    )
    args = parser.parse_args()
    modes = (
        ("legacy", "none", "align")
        if args.metadata_mode == "all"
        else (args.metadata_mode,)
    )
    for mode in modes:
        run(args.device, mode)
