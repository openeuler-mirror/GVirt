#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Packed native-page prefill: initial/reset state, nulls, padding, chunks and decode handoff."""

import argparse
import math

import torch
import torch_npu  # noqa: F401

from xlite._C import (
    Runtime,
    conv1d_and_silu_prefill_paged,
    linear_att_conv_and_silu_paged,
    recurrent_gated_delta_rule_paged,
    recurrent_gated_delta_rule_prefill_paged,
)


def page_tensor(shape, dtype):
    logical = math.prod(shape[1:])
    stride, offset = logical + 128, 32
    raw = (torch.randn(offset + shape[0] * stride + 32) * 0.05).to("npu", dtype)
    inner = torch.empty(shape[1:]).stride()
    state = raw.as_strided(shape, (stride, *inner), offset)
    reference = raw.cpu().clone()
    return raw, state, reference, reference.as_strided(shape, state.stride(), offset)


def check_case(rt, dtype, state_dtype, lengths, kh=2, vh=6, kd=128, vd=128, fp32_gates=False):
    batch, tokens = len(lengths), sum(lengths)
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    pages = [3, 1, 4, 0, -1][:batch]
    flags = [0, 1, 1, 0, 1][:batch]
    indices, starts, initial = [torch.tensor(x, dtype=torch.int32, device="npu") for x in (pages, bounds, flags)]
    raw, state, ref_raw, ref_state = page_tensor((batch + 4, vh, vd, kd), state_dtype)
    channels, width = 2 * kh * kd + vh * vd, 4
    cr, conv, ref_cr, ref_conv = page_tensor((batch + 4, width - 1, channels), dtype)
    for step in range(2):
        q = torch.nn.functional.normalize(torch.randn(tokens, kh, kd), dim=-1).to(dtype)
        k = torch.nn.functional.normalize(torch.randn(tokens, kh, kd), dim=-1).to(dtype)
        v = (torch.randn(tokens, vh, vd) * 0.1).to(dtype)
        beta = torch.rand(tokens, vh).to(dtype)
        g = (-torch.rand(tokens, vh) * 0.5).to(torch.float32 if fp32_gates else dtype)
        expected = torch.zeros_like(v)
        x, w = (torch.randn(tokens, channels) * 0.2).to(dtype), torch.randn(channels, width).to(dtype)
        cy = torch.zeros_like(x)
        k_rep = k.float().repeat_interleave(vh // kh, dim=1)
        q_rep = q.float().repeat_interleave(vh // kh, dim=1) / math.sqrt(kd)
        g_exp = g.float().exp()
        v_f = v.float()
        beta_f = beta.float()
        x_f = x.float()
        w_t = w.float().T
        for b, page in enumerate(pages):
            if page <= 0 or lengths[b] == 0:
                continue
            st = ref_state[page].float().clone() if flags[b] else torch.zeros_like(ref_state[page]).float()
            history = ref_conv[page].float().clone() if flags[b] else torch.zeros_like(ref_conv[page]).float()
            lo, hi = bounds[b], bounds[b + 1]
            seq = torch.cat((history, x_f[lo:hi]))
            for j, t in enumerate(range(lo, hi)):
                kk = k_rep[t]
                qq = q_rep[t]
                st *= g_exp[t][:, None, None]
                delta = (v_f[t] - (st * kk[:, None]).sum(-1)) * beta_f[t][:, None]
                st += delta[:, :, None] * kk[:, None]
                expected[t] = (st * qq[:, None]).sum(-1).to(dtype)
                window = seq[j : j + width]
                cy[t] = torch.nn.functional.silu((window * w_t).sum(0).to(dtype).float()).to(dtype)
            history = seq[hi - lo :]
            ref_state[page].copy_(st.to(state_dtype))
            ref_conv[page].copy_(history.to(dtype))
        inputs = [t.to("npu") for t in (q.flatten(1), k.flatten(1), v.flatten(1), beta, g, x, w)]
        qn, kn, vn, bn, gn, xn, wn = inputs
        out, co = torch.full_like(vn, float("nan")), torch.full_like(xn, float("nan"))
        torch.npu.synchronize()
        recurrent_gated_delta_rule_prefill_paged(
            rt=rt,
            query=qn,
            key=kn,
            value=vn,
            beta=bn,
            g=gn,
            state=state,
            out=out,
            indices=indices,
            starts=starts,
            initial=initial,
            num_key_heads=kh,
            num_value_heads=vh,
            k_dim=kd,
            v_dim=vd,
        )
        conv1d_and_silu_prefill_paged(
            rt=rt, input=xn, state=conv, weight=wn, output=co, indices=indices, starts=starts, initial=initial
        )
        raw_cpu, cr_cpu = raw.cpu(), cr.cpu()
        torch.testing.assert_close(out.cpu(), expected.flatten(1), rtol=0.025, atol=0.001)
        torch.testing.assert_close(raw_cpu, ref_raw, rtol=0.025, atol=0.001)
        torch.testing.assert_close(co.cpu(), cy, rtol=0.025, atol=0.004)
        torch.testing.assert_close(cr_cpu, ref_cr, rtol=0, atol=0)
        # Check all unowned pages, null page, padding and storage guards exactly.
        for actual, reference, tensor in ((raw_cpu, ref_raw, state), (cr_cpu, ref_cr, conv)):
            mask = torch.zeros_like(reference, dtype=torch.bool)
            for b, page in enumerate(pages):
                if page > 0 and lengths[b]:
                    offset = tensor.storage_offset() + page * tensor.stride(0)
                    mask[offset : offset + math.prod(tensor.shape[1:])] = True
            assert torch.equal(actual[~mask], reference[~mask])
        flags = [1] * batch
        initial.fill_(1)
    # Decode directly from the same pages after prefill, without a cache conversion.
    ids = torch.tensor([3, 1], dtype=torch.int32, device="npu")
    qd = torch.nn.functional.normalize(torch.randn(2, kh, kd), dim=-1).to("npu", dtype)
    kd_t = torch.nn.functional.normalize(torch.randn(2, kh, kd), dim=-1).to("npu", dtype)
    vd_t = (torch.randn(2, vh, vd) * 0.1).to("npu", dtype)
    bd = torch.rand(2, vh).to("npu", dtype)
    gd = (-torch.rand(2, vh)).to("npu", torch.float32 if fp32_gates else dtype)
    output = torch.empty_like(vd_t)
    expected = torch.empty_like(output, device="cpu")
    gd_f = gd.cpu().float()
    kd_f = kd_t.cpu().float()
    qd_f = qd.cpu().float()
    vd_f = vd_t.cpu().float()
    bd_f = bd.cpu().float()
    for b, page in enumerate([3, 1]):
        st = ref_state[page].float() * gd_f[b].exp()[:, None, None]
        kk = kd_f[b].repeat_interleave(vh // kh, dim=0)
        qq = qd_f[b].repeat_interleave(vh // kh, dim=0) / math.sqrt(kd)
        delta = (vd_f[b] - (st * kk[:, None]).sum(-1)) * bd_f[b][:, None]
        st += delta[:, :, None] * kk[:, None]
        expected[b] = (st * qq[:, None]).sum(-1).to(dtype)
        ref_state[page].copy_(st.to(state_dtype))
    recurrent_gated_delta_rule_paged(
        rt, qd.flatten(1), kd_t.flatten(1), vd_t.flatten(1), bd, gd, state, output.flatten(1), ids, kh, vh, kd, vd
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0.025, atol=0.001)
    torch.testing.assert_close(raw.cpu(), ref_raw, rtol=0.025, atol=0.001)
    # Conv Decode keeps its fused FP32 SiLU contract after Prefill's rounded
    # conv1d->SiLU path and consumes the very same padded history pages.
    xd = (torch.randn(2, channels) * 0.2).to(dtype)
    yd_ref = torch.empty_like(xd)
    for b, page in enumerate([3, 1]):
        window = torch.cat((ref_conv[page].float(), xd[b : b + 1].float()))
        yd_ref[b] = torch.nn.functional.silu((window * w.float().T).sum(0)).to(dtype)
        ref_conv[page].copy_(window[1:].to(dtype))
    xd_npu = xd.to("npu")
    yd = torch.empty_like(xd_npu)
    linear_att_conv_and_silu_paged(rt, xd_npu, conv, wn, yd, ids)
    torch.testing.assert_close(yd.cpu(), yd_ref, rtol=0.025, atol=0.004)
    torch.testing.assert_close(cr.cpu(), ref_cr, rtol=0, atol=0)
    print(f"PASS {dtype}/{state_dtype}: lengths={lengths}, Hk/Hv={kh}/{vh}, K/V={kd}/{vd}", flush=True)


def check_rejections(rt):
    x = torch.zeros(2, 192, dtype=torch.bfloat16, device="npu")
    conv = torch.ones(3, 3, 192, dtype=x.dtype, device="npu")
    weight = torch.ones(192, 4, dtype=x.dtype, device="npu")
    q = torch.zeros(2, 64, dtype=x.dtype, device="npu")
    gates = torch.zeros(2, 1, dtype=x.dtype, device="npu")
    state = torch.ones(3, 1, 64, 64, dtype=torch.float32, device="npu")
    ids = torch.ones(1, dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 2], dtype=torch.int32, device="npu")
    initial = torch.zeros(1, dtype=torch.int32, device="npu")
    # The bindings are keyword-only; the bad tensor is swapped by parameter name
    # so the invalid descriptor still reaches kernel validation (RuntimeError),
    # not a binding-layer TypeError.
    conv_args = dict(
        rt=rt,
        input=x,
        state=conv,
        weight=weight,
        output=torch.empty_like(x),
        indices=ids,
        starts=starts,
        initial=initial,
    )
    ssm_args = dict(
        rt=rt,
        query=q,
        key=q,
        value=q,
        beta=gates,
        g=gates,
        state=state,
        out=torch.empty_like(q),
        indices=ids,
        starts=starts,
        initial=initial,
        num_key_heads=1,
        num_value_heads=1,
        k_dim=64,
        v_dim=64,
    )
    cases = [
        (conv1d_and_silu_prefill_paged, conv_args, "starts", starts.long()),
        (conv1d_and_silu_prefill_paged, conv_args, "initial", initial.float()),
        (conv1d_and_silu_prefill_paged, conv_args, "indices", ids.cpu()),
        (conv1d_and_silu_prefill_paged, conv_args, "output", x.T),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, "starts", starts.long()),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, "initial", initial.float()),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, "indices", ids.cpu()),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, "starts", starts[:1]),
    ]
    for function, kwargs, name, bad_tensor in cases:
        kwargs = dict(kwargs)
        kwargs[name] = bad_tensor
        before_conv, before_state = conv.clone(), state.clone()
        try:
            function(**kwargs)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid descriptor accepted at argument {name}")
        assert torch.equal(conv, before_conv), "rejected call changed Conv state"
        assert torch.equal(state, before_state), "rejected call changed SSM state"
    print("PASS 8 Prefill descriptor/device rejections with unchanged state", flush=True)


def main(device):
    torch.npu.set_device(device)
    torch.manual_seed(143)
    rt = Runtime(device, size=64)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for state_dtype in dict.fromkeys((dtype, torch.float32)):
            for kd, vd in ((16, 64), (64, 128), (128, 128)):
                check_case(rt, dtype, state_dtype, [1, 0, 65, 3, 7], kd=kd, vd=vd)
    for length in (512, 1024, 4096):
        check_case(rt, torch.bfloat16, torch.float32, [length, 1], kh=1, vh=3)
    check_case(rt, torch.bfloat16, torch.float32, [1, 0, 65, 3, 7], fp32_gates=True)
    check_rejections(rt)
    print("PASS 19 packed prefill cases, 38 continuation updates and 19 SSM + 19 Conv decode handoffs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    main(parser.parse_args().device)
