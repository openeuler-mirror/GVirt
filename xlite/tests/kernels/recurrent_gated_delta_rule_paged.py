#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Native VK decode: grouped heads, recurrent updates, nulls, offsets and guards."""

import argparse
import math

import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, recurrent_gated_delta_rule_paged


def check_case(rt, dtype, state_dtype, batch, kh, vh, kd, vd, mode="legacy"):
    pages, offset = batch + 4, 32
    logical = vh * vd * kd
    stride = logical + 128
    raw = (torch.randn(offset + pages * stride + 32) * 0.05).to("npu", state_dtype)
    state = raw.as_strided((pages, vh, vd, kd), (stride, vd * kd, kd, 1), offset)
    reference = raw.cpu().clone()
    ref_state = reference.as_strided(state.shape, state.stride(), offset)
    tolerance = {
        torch.bfloat16: (0.025, 6e-4),
        torch.float16: (0.004, 1e-4),
        torch.float32: (1e-4, 2e-6),
    }[dtype]
    state_tolerance = {
        torch.bfloat16: (0.025, 1e-3),
        torch.float16: (0.004, 2e-4),
        torch.float32: (1e-4, 2e-6),
    }[state_dtype]
    for step in range(3):
        order = list(range(2, batch + 2))
        if step % 2:
            order.reverse()
        if batch >= 3:
            order[-1] = 0 if step % 2 == 0 else -1
        indices = torch.tensor(order, dtype=torch.int32, device="npu")
        metadata = {}
        if mode != "legacy":
            cached = torch.tensor(([63, 64, 128] * batch)[:batch], dtype=torch.int32)
            if batch >= 3:
                cached[-1] = 0
            columns = cached // 64 if mode == "align" else torch.zeros_like(cached)
            table = torch.full((batch, 3), pages - 1, dtype=torch.int32)
            table[torch.arange(batch), columns.long()] = torch.tensor(
                order, dtype=torch.int32
            )
            if batch >= 3:
                # Padding must ignore a stale positive page in the raw table.
                table[-1, 0] = pages - 1
            storage = torch.full(
                (32 + batch * 7 + 32,), -1, dtype=torch.int32, device="npu"
            )
            indices = storage.as_strided((batch, 3), (7, 1), 32)
            indices.copy_(table)
            metadata = {
                "cached_lens": cached.to("npu"),
                "state_block_size": 64,
                "aligned": mode == "align",
            }
        q_cpu = torch.nn.functional.normalize(torch.randn(batch, kh, kd), dim=-1).to(
            dtype
        )
        k_cpu = torch.nn.functional.normalize(torch.randn(batch, kh, kd), dim=-1).to(
            dtype
        )
        v_cpu = (torch.randn(batch, vh, vd) * 0.1).to(dtype)
        beta_cpu = torch.rand(batch, vh).to(dtype)
        g_cpu = (-torch.rand(batch, vh)).to(dtype)
        q = q_cpu.flatten(1).to("npu")
        k = k_cpu.flatten(1).to("npu")
        v = v_cpu.flatten(1).to("npu")
        beta, g = beta_cpu.to("npu"), g_cpu.to("npu")
        out = torch.full_like(v, float("nan"))
        expected = torch.zeros_like(v_cpu)
        for b, page in enumerate(order):
            if page <= 0:
                continue
            old = ref_state[page].float() * g_cpu[b].float().exp()[:, None, None]
            kk = k_cpu[b].float().repeat_interleave(vh // kh, dim=0)
            qq = q_cpu[b].float().repeat_interleave(vh // kh, dim=0) / math.sqrt(kd)
            delta = (v_cpu[b].float() - (old * kk[:, None, :]).sum(-1)) * beta_cpu[
                b
            ].float()[:, None]
            updated = old + delta[:, :, None] * kk[:, None, :]
            expected[b] = (updated * qq[:, None, :]).sum(-1).to(dtype)
            ref_state[page].copy_(updated.to(state_dtype))
        before = raw.cpu().clone()
        torch.npu.synchronize()
        recurrent_gated_delta_rule_paged(
            rt, q, k, v, beta, g, state, out, indices, kh, vh, kd, vd, **metadata
        )
        actual = raw.cpu()
        torch.testing.assert_close(
            out.cpu(), expected.flatten(1), rtol=tolerance[0], atol=tolerance[1]
        )
        torch.testing.assert_close(
            actual, reference, rtol=state_tolerance[0], atol=state_tolerance[1]
        )
        modified = torch.zeros_like(before, dtype=torch.bool)
        for page in order:
            if page > 0:
                modified[offset + page * stride : offset + page * stride + logical] = (
                    True
                )
        assert torch.equal(actual[~modified], before[~modified]), (
            "unselected page/padding/guard changed"
        )
        for b, page in enumerate(order):
            if page <= 0:
                assert torch.count_nonzero(out[b]).item() == 0, (
                    "null output must be exactly zero"
                )
    print(
        f"PASS {mode} {dtype}/{state_dtype}: B={batch} Hk/Hv={kh}/{vh} K/V={kd}/{vd}",
        flush=True,
    )


def check_rejections(rt):
    q = torch.zeros(2, 2 * 128, device="npu", dtype=torch.bfloat16)
    v = torch.zeros(2, 6 * 128, device="npu", dtype=torch.bfloat16)
    beta = torch.zeros(2, 6, device="npu", dtype=torch.bfloat16)
    state = torch.zeros(4, 6, 128, 128, device="npu", dtype=torch.bfloat16)
    indices = torch.tensor([1, 2], device="npu", dtype=torch.int32)
    args = [rt, q, q, v, beta, beta, state, v.clone(), indices, 2, 6, 128, 128]
    mutations = [
        (9, 4),
        (11, 120),
        (12, 32),
        (8, indices.long()),
        (6, state.transpose(-1, -2)),
        (6, state.float().to(torch.float16)),
        (1, q[:, ::2]),
        (7, v[:, ::2]),
    ]
    for index, value in mutations:
        bad = args.copy()
        bad[index] = value
        try:
            recurrent_gated_delta_rule_paged(*bad)
        except (RuntimeError, ValueError):
            continue
        raise AssertionError(f"invalid descriptor accepted: argument {index}")
    print(f"PASS {len(mutations)} descriptor rejections", flush=True)

    table = indices[:, None]
    cached = torch.tensor([63, 64], device="npu", dtype=torch.int32)
    raw_args = args.copy()
    raw_args[8] = table
    for name, source in (("cached_lens", cached), ("state_indices", table)):
        bad = raw_args.copy()
        metadata = {"cached_lens": cached, "state_block_size": 64, "aligned": True}
        if name == "cached_lens":
            metadata[name] = source.cpu()
        else:
            bad[8] = source.cpu()
        before = state.clone()
        try:
            recurrent_gated_delta_rule_paged(*bad, **metadata)
        except RuntimeError as error:
            assert "runtime NPU" in str(error), str(error)
        else:
            raise AssertionError(f"CPU {name} accepted")
        assert torch.equal(state, before), "rejected call changed state"
    print("PASS raw metadata device rejections", flush=True)


def run(device, metadata_mode):
    torch.npu.set_device(device)
    torch.manual_seed(71)
    rt = Runtime(device, 64)
    cases = [(3, 2, 6, kd, 64 if kd % 32 else 128) for kd in range(16, 129, 16)]
    cases += [(1, 16, 48, 128, 128), (4, 16, 48, 128, 128), (16, 16, 48, 128, 128)]
    modes = ("legacy", "none", "align") if metadata_mode == "all" else (metadata_mode,)
    for mode in modes:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for state_dtype in dict.fromkeys((dtype, torch.float32)):
                for case in cases:
                    check_case(rt, dtype, state_dtype, *case, mode=mode)
    check_rejections(rt)
    print(
        f"PASS native VK paged recurrent: {55 * len(modes)} cases, "
        f"{165 * len(modes)} recurrent steps",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--metadata-mode", choices=("all", "legacy", "none", "align"), default="all"
    )
    args = parser.parse_args()
    run(args.device, args.metadata_mode)
