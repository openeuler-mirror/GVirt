#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""GLM-5.3-Flash KDA paged recurrent: in-kernel sigmoid gate/beta, per-k decay,
q/k L2-norm, native [P, Hv, V, K] pages, nulls and guards."""

import argparse
import math

import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, recurrent_gated_delta_rule_kda_paged, recurrent_gated_delta_rule_kda_prefill_paged

NORM_EPS = 1e-6


def l2norm(x):
    # fla l2norm: x * rsqrt(sum(x^2) + eps) along the last dim, fp32.
    return x.float() * torch.rsqrt((x.float() ** 2).sum(-1, keepdim=True) + NORM_EPS)


def torch_ref_refactored(
    b, page, ref_state, kh, vh, kd, vd, k_b, q_b, v_b, beta_b, g1_b, a_log_cpu, dt_bias_cpu, lower_bound, scale, dtype
):
    # In-kernel fp32 math: beta = sigmoid(b); per-k decay (per head h a [kd] row):
    #   g = lowerBound * sigmoid(exp(aLog[h]) * (g1[h] + dtBias[h])); state[h,k,:] *= exp(g)
    # beta_b: [vh] raw; g1_b: [kh, kd] raw gate row group.
    # Kernel reads bf16/fp16-rounded activations from GM; mirror that rounding.
    old = ref_state[page].float()  # [vh, vd, kd]
    decay = lower_bound * torch.sigmoid(
        a_log_cpu.float().exp()[:, None] * (g1_b.to(dtype).float() + dt_bias_cpu.float())
    )  # [kh, kd]
    old = old * decay.exp()[:, None, :]
    beta = torch.sigmoid(beta_b.to(dtype).float())[:, None]  # [vh, 1]
    kk = l2norm(k_b.to(dtype).float()).repeat_interleave(vh // kh, dim=0)  # [vh, kd]
    qq = (l2norm(q_b.to(dtype).float()) * scale).repeat_interleave(vh // kh, dim=0)
    delta = (v_b.to(dtype).float() - (old * kk[:, None, :]).sum(-1)) * beta
    updated = old + delta[:, :, None] * kk[:, None, :]
    return (updated * qq[:, None, :]).sum(-1), updated


def check_case(rt, dtype, state_dtype, batch, kh, vh, kd, vd, lower_bound, scale):
    pages, offset = batch + 4, 32
    logical = vh * vd * kd
    stride = logical + 128
    raw = (torch.randn(offset + pages * stride + 32) * 0.05).to("npu", state_dtype)
    state = raw.as_strided((pages, vh, vd, kd), (stride, vd * kd, kd, 1), offset)
    reference = raw.cpu().clone()
    ref_state = reference.as_strided(state.shape, state.stride(), offset)
    tolerance = {
        torch.bfloat16: (0.03, 8e-4),
        torch.float16: (0.006, 2e-4),
        torch.float32: (1e-4, 2e-6),
    }[dtype]
    state_tolerance = {
        torch.bfloat16: (0.03, 2e-3),
        torch.float16: (0.006, 4e-4),
        torch.float32: (1e-4, 2e-6),
    }[state_dtype]
    for step in range(3):
        order = list(range(2, batch + 2))
        if step % 2:
            order.reverse()
        if batch >= 3:
            order[-1] = 0 if step % 2 == 0 else -1
        indices = torch.tensor(order, dtype=torch.int32, device="npu")
        q_cpu = torch.randn(batch, kh, kd)
        k_cpu = torch.randn(batch, kh, kd)
        v_cpu = torch.randn(batch, vh, vd) * 0.1
        beta_cpu = torch.randn(batch, vh) * 2  # raw, sigmoid in-kernel
        g1_cpu = torch.randn(batch, kh, kd) * 2  # raw gate, [T, Hk*K]
        a_log_cpu = torch.randn(kh) * 0.5  # fp32 parameter
        dt_bias_cpu = torch.randn(kh, kd) * 0.5  # fp32 parameter
        q = q_cpu.to(dtype).flatten(1).to("npu")  # raw; kernel L2-norms in fp32
        k = k_cpu.to(dtype).flatten(1).to("npu")  # raw; kernel L2-norms in fp32
        v = v_cpu.to(dtype).flatten(1).to("npu")
        beta = beta_cpu.to(dtype).to("npu")
        g1 = g1_cpu.to(dtype).flatten(1).to("npu")
        a_log = a_log_cpu.to("npu", torch.float32)
        dt_bias = dt_bias_cpu.reshape(-1).to("npu", torch.float32)  # [Hk*K] contiguous
        out = torch.full_like(v, float("nan"))
        expected = torch.zeros(v_cpu.shape, dtype=dtype)  # container matches kernel out dtype
        for b, page in enumerate(order):
            if page <= 0:
                continue
            # Kernel L2-norms q/k itself; feed unnormalized fp32 to the ref.
            out_ref, updated = torch_ref_refactored(
                b,
                page,
                ref_state,
                kh,
                vh,
                kd,
                vd,
                k_cpu[b],
                q_cpu[b],
                v_cpu[b],
                beta_cpu[b],
                g1_cpu[b],
                a_log_cpu,
                dt_bias_cpu,
                lower_bound,
                scale,
                dtype,
            )
            expected[b] = out_ref.to(dtype)
            ref_state[page].copy_(updated.to(state_dtype))
        before = raw.cpu().clone()
        torch.npu.synchronize()
        recurrent_gated_delta_rule_kda_paged(
            rt,
            q,
            k,
            v,
            beta,
            g1,
            a_log,
            dt_bias,
            state,
            out,
            indices,
            kh,
            vh,
            kd,
            vd,
            scale,
            lower_bound,
            NORM_EPS,
        )
        actual = raw.cpu()
        err_page = (actual.float() - reference.float()).abs()
        for b in range(batch):
            page = order[b]
            if page > 0:
                seg = err_page[offset + page * stride : offset + page * stride + logical]
                print(
                    f"  diag b={b} page={page} state err={seg.max().item():.3e} "
                    f"out err={(out.cpu().float()[b] - expected.float()[b].flatten()).abs().max().item():.3e}",
                    flush=True,
                )
        torch.testing.assert_close(out.cpu(), expected.flatten(1), rtol=tolerance[0], atol=tolerance[1])
        torch.testing.assert_close(actual, reference, rtol=state_tolerance[0], atol=state_tolerance[1])
        modified = torch.zeros_like(before, dtype=torch.bool)
        for page in order:
            if page > 0:
                modified[offset + page * stride : offset + page * stride + logical] = True
        assert torch.equal(actual[~modified], before[~modified]), "unselected page/padding/guard changed"
        for b, page in enumerate(order):
            if page <= 0:
                assert torch.count_nonzero(out[b]).item() == 0, "null output must be exactly zero"
    print(
        f"PASS {dtype}/{state_dtype}: B={batch} Hk/Hv={kh}/{vh} K/V={kd}/{vd} lb={lower_bound}",
        flush=True,
    )


def _kda_token_step(cur, k_cpu, q_cpu, v_cpu, beta_b, g1_b, a_log_cpu, dt_bias_cpu, lower_bound, scale, dtype):
    # Same fp32 in-kernel math as torch_ref_refactored, but stepping an explicit
    # running state (kh == vh in this KDA contract, so per-head arrays broadcast 1:1).
    decay = lower_bound * torch.sigmoid(
        a_log_cpu.float().exp()[:, None] * (g1_b.to(dtype).float() + dt_bias_cpu.float())
    )  # [kh, kd]
    cur = cur * decay.exp()[:, None, :]
    beta = torch.sigmoid(beta_b.to(dtype).float())[:, None]  # [vh, 1]
    kk = l2norm(k_cpu.to(dtype).float())  # [vh, kd]
    qq = l2norm(q_cpu.to(dtype).float()) * scale
    delta = (v_cpu.to(dtype).float() - (cur * kk[:, None, :]).sum(-1)) * beta
    cur = cur + delta[:, :, None] * kk[:, None, :]
    out = (cur * qq[:, None, :]).sum(-1)
    return out, cur


def check_prefill_case(rt, dtype, state_dtype, seg_lens, kh, vh, kd, vd, lower_bound, scale):
    tokens = sum(seg_lens)
    pages, offset = len(seg_lens) + 4, 32
    logical = vh * vd * kd
    stride = logical + 128
    raw = (torch.randn(offset + pages * stride + 32) * 0.05).to("npu", state_dtype)
    state = raw.as_strided((pages, vh, vd, kd), (stride, vd * kd, kd, 1), offset)
    reference = raw.cpu().clone()
    ref_state = reference.as_strided(state.shape, state.stride(), offset)
    tolerance = {
        torch.bfloat16: (0.03, 8e-4),
        torch.float16: (0.006, 2e-4),
        torch.float32: (1e-4, 2e-6),
    }[dtype]
    state_tolerance = {
        torch.bfloat16: (0.03, 2e-3),
        torch.float16: (0.006, 4e-4),
        torch.float32: (1e-4, 2e-6),
    }[state_dtype]
    order = list(range(2, len(seg_lens) + 2))
    indices = torch.tensor(order, dtype=torch.int32, device="npu")
    starts = torch.tensor([0] + list(torch.cumsum(torch.tensor(seg_lens), 0)), dtype=torch.int32, device="npu")
    # Alternate fresh (initial=0: zero start, page overwritten at commit) and
    # continue (initial=1: keep and update the page's existing content).
    initial = torch.tensor([i % 2 for i in range(len(seg_lens))], dtype=torch.int32, device="npu")
    q_cpu = torch.randn(tokens, kh, kd)
    k_cpu = torch.randn(tokens, kh, kd)
    v_cpu = torch.randn(tokens, vh, vd) * 0.1
    beta_cpu = torch.randn(tokens, vh) * 2  # raw, sigmoid in-kernel
    g1_cpu = torch.randn(tokens, kh, kd) * 2  # raw gate, [T, Hk*K]
    a_log_cpu = torch.randn(kh) * 0.5
    dt_bias_cpu = torch.randn(kh, kd) * 0.5
    q = q_cpu.to(dtype).flatten(1).to("npu")
    k = k_cpu.to(dtype).flatten(1).to("npu")
    v = v_cpu.to(dtype).flatten(1).to("npu")
    beta = beta_cpu.to(dtype).to("npu")
    g1 = g1_cpu.to(dtype).flatten(1).to("npu")
    a_log = a_log_cpu.to("npu", torch.float32)
    dt_bias = dt_bias_cpu.reshape(-1).to("npu", torch.float32)
    out = torch.full((tokens, vh * vd), float("nan"), dtype=dtype, device="npu")
    expected = torch.zeros(tokens, vh, vd, dtype=dtype)
    t = 0
    for r, seg in enumerate(seg_lens):
        page = order[r]
        if seg == 0:
            continue  # empty segment leaves the page untouched
        cur = torch.zeros(vh, vd, kd) if initial[r] == 0 else ref_state[page].float().clone()
        for _ in range(seg):
            out_ref, cur = _kda_token_step(
                cur,
                k_cpu[t],
                q_cpu[t],
                v_cpu[t],
                beta_cpu[t],
                g1_cpu[t],
                a_log_cpu,
                dt_bias_cpu,
                lower_bound,
                scale,
                dtype,
            )
            expected[t] = out_ref.to(dtype)
            t += 1
        ref_state[page].copy_(cur.to(state_dtype))
    torch.npu.synchronize()
    recurrent_gated_delta_rule_kda_prefill_paged(
        rt,
        q,
        k,
        v,
        beta,
        g1,
        a_log,
        dt_bias,
        state,
        out,
        indices,
        starts,
        initial,
        kh,
        vh,
        kd,
        vd,
        scale,
        lower_bound,
        NORM_EPS,
    )
    torch.testing.assert_close(out.cpu(), expected.flatten(1), rtol=tolerance[0], atol=tolerance[1])
    torch.testing.assert_close(raw.cpu(), reference, rtol=state_tolerance[0], atol=state_tolerance[1])
    print(
        f"PASS prefill {dtype}/{state_dtype}: segs={seg_lens} Hk/Hv={kh}/{vh} K/V={kd}/{vd}",
        flush=True,
    )


def run(device):
    torch.npu.set_device(device)
    torch.manual_seed(73)
    rt = Runtime(device, size=64)
    # GLM-5.3-Flash KDA: kh == vh (same head count), kd == vd == head_dim.
    cases = [(3, 2, 2, kd, 64 if kd % 32 else 128) for kd in range(16, 129, 16)]
    cases += [(1, 16, 16, 128, 128), (4, 16, 16, 128, 128)]
    bounds = [(-5.0, None), (0.5, 1.0 / math.sqrt(128))]
    for lower_bound, scale in bounds:
        eff_scale = scale if scale is not None else 1.0
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for state_dtype in dict.fromkeys((dtype, torch.float32)):
                for case in cases:
                    check_case(rt, dtype, state_dtype, *case, lower_bound, eff_scale)
    # Prefill: packed ragged segments (incl. an empty one) at one geometry,
    # across the dtype/state matrix; kh == vh == 2, kd == vd == 64.
    for lower_bound, scale in bounds:
        eff_scale = scale if scale is not None else 1.0 / math.sqrt(64)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for state_dtype in dict.fromkeys((dtype, torch.float32)):
                check_prefill_case(rt, dtype, state_dtype, [3, 1, 5, 2], 2, 2, 64, 64, lower_bound, eff_scale)
        check_prefill_case(rt, torch.bfloat16, torch.float32, [4, 0, 2], 2, 2, 64, 64, 0.5, eff_scale)
    print("PASS KDA safe-gate paged recurrent: all cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    run(args.device)
