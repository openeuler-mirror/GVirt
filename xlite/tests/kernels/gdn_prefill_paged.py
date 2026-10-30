#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Packed native-page prefill: initial/reset state, nulls, padding, chunks and decode handoff."""
import argparse
import math

import torch
import torch_npu  # noqa: F401
from xlite._C import (
    Runtime, conv1d_and_silu_prefill_paged, linear_att_conv_and_silu_paged,
    recurrent_gated_delta_rule_paged,
    recurrent_gated_delta_rule_prefill_paged,
)


def page_tensor(shape, dtype):
    logical = math.prod(shape[1:])
    stride, offset = logical + 128, 32
    raw = (torch.randn(offset + shape[0] * stride + 32) * .05).to('npu', dtype)
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
    indices, starts, initial = [torch.tensor(x, dtype=torch.int32, device='npu')
                               for x in (pages, bounds, flags)]
    raw, state, ref_raw, ref_state = page_tensor((batch + 4, vh, vd, kd), state_dtype)
    channels, width = 2 * kh * kd + vh * vd, 4
    cr, conv, ref_cr, ref_conv = page_tensor((batch + 4, width - 1, channels), dtype)
    for step in range(2):
        q = torch.nn.functional.normalize(torch.randn(tokens, kh, kd), dim=-1).to(dtype)
        k = torch.nn.functional.normalize(torch.randn(tokens, kh, kd), dim=-1).to(dtype)
        v = (torch.randn(tokens, vh, vd) * .1).to(dtype)
        beta = torch.rand(tokens, vh).to(dtype)
        g = (-torch.rand(tokens, vh) * .5).to(torch.float32 if fp32_gates else dtype)
        expected = torch.zeros_like(v)
        x, w = (torch.randn(tokens, channels) * .2).to(dtype), torch.randn(channels, width).to(dtype)
        cy = torch.zeros_like(x)
        for b, page in enumerate(pages):
            if page <= 0 or lengths[b] == 0:
                continue
            st = ref_state[page].float().clone() if flags[b] else torch.zeros_like(ref_state[page]).float()
            history = ref_conv[page].float().clone() if flags[b] else torch.zeros_like(ref_conv[page]).float()
            for t in range(bounds[b], bounds[b + 1]):
                kk = k[t].float().repeat_interleave(vh // kh, dim=0)
                qq = q[t].float().repeat_interleave(vh // kh, dim=0) / math.sqrt(kd)
                st *= g[t].float().exp()[:, None, None]
                delta = (v[t].float() - (st * kk[:, None]).sum(-1)) * beta[t].float()[:, None]
                st += delta[:, :, None] * kk[:, None]
                expected[t] = (st * qq[:, None]).sum(-1).to(dtype)
                window = torch.cat((history, x[t:t + 1].float()))
                cy[t] = torch.nn.functional.silu((window * w.float().T).sum(0).to(dtype).float()).to(dtype)
                history = window[1:]
            ref_state[page].copy_(st.to(state_dtype))
            ref_conv[page].copy_(history.to(dtype))
        inputs = [t.to('npu') for t in (q.flatten(1), k.flatten(1), v.flatten(1), beta, g, x, w)]
        qn, kn, vn, bn, gn, xn, wn = inputs
        out, co = torch.full_like(vn, float('nan')), torch.full_like(xn, float('nan'))
        torch.npu.synchronize()
        recurrent_gated_delta_rule_prefill_paged(rt, qn, kn, vn, bn, gn, state, out,
                                                indices, starts, initial, kh, vh, kd, vd)
        conv1d_and_silu_prefill_paged(rt, xn, conv, wn, co, indices, starts, initial)
        torch.testing.assert_close(out.cpu(), expected.flatten(1), rtol=.025, atol=.001)
        torch.testing.assert_close(raw.cpu(), ref_raw, rtol=.025, atol=.001)
        torch.testing.assert_close(co.cpu(), cy, rtol=.025, atol=.004)
        torch.testing.assert_close(cr.cpu(), ref_cr, rtol=0, atol=0)
        # Check all unowned pages, null page, padding and storage guards exactly.
        for actual, reference, tensor in ((raw, ref_raw, state), (cr, ref_cr, conv)):
            mask = torch.zeros_like(reference, dtype=torch.bool)
            for b, page in enumerate(pages):
                if page > 0 and lengths[b]:
                    offset = tensor.storage_offset() + page * tensor.stride(0)
                    mask[offset:offset + math.prod(tensor.shape[1:])] = True
            assert torch.equal(actual.cpu()[~mask], reference[~mask])
        flags = [1] * batch
        initial.fill_(1)
    # Decode directly from the same pages after prefill, without a cache conversion.
    ids = torch.tensor([3, 1], dtype=torch.int32, device='npu')
    qd = torch.nn.functional.normalize(torch.randn(2, kh, kd), dim=-1).to('npu', dtype)
    kd_t = torch.nn.functional.normalize(torch.randn(2, kh, kd), dim=-1).to('npu', dtype)
    vd_t = (torch.randn(2, vh, vd) * .1).to('npu', dtype)
    bd = torch.rand(2, vh).to('npu', dtype)
    gd = (-torch.rand(2, vh)).to('npu', torch.float32 if fp32_gates else dtype)
    output = torch.empty_like(vd_t)
    expected = torch.empty_like(output, device='cpu')
    for b, page in enumerate([3, 1]):
        st = ref_state[page].float() * gd[b].cpu().float().exp()[:, None, None]
        kk = kd_t[b].cpu().float().repeat_interleave(vh // kh, dim=0)
        qq = qd[b].cpu().float().repeat_interleave(vh // kh, dim=0) / math.sqrt(kd)
        delta = (vd_t[b].cpu().float() - (st * kk[:, None]).sum(-1)) * bd[b].cpu().float()[:, None]
        st += delta[:, :, None] * kk[:, None]
        expected[b] = (st * qq[:, None]).sum(-1).to(dtype)
        ref_state[page].copy_(st.to(state_dtype))
    torch.npu.synchronize()
    recurrent_gated_delta_rule_paged(rt, qd.flatten(1), kd_t.flatten(1), vd_t.flatten(1), bd, gd,
                                    state, output.flatten(1), ids, kh, vh, kd, vd)
    torch.testing.assert_close(output.cpu(), expected, rtol=.025, atol=.001)
    torch.testing.assert_close(raw.cpu(), ref_raw, rtol=.025, atol=.001)
    # Conv Decode keeps its fused FP32 SiLU contract after Prefill's rounded
    # conv1d->SiLU path and consumes the very same padded history pages.
    xd = (torch.randn(2, channels) * .2).to(dtype)
    yd_ref = torch.empty_like(xd)
    for b, page in enumerate([3, 1]):
        window = torch.cat((ref_conv[page].float(), xd[b:b + 1].float()))
        yd_ref[b] = torch.nn.functional.silu((window * w.float().T).sum(0)).to(dtype)
        ref_conv[page].copy_(window[1:].to(dtype))
    xd_npu = xd.to('npu')
    yd = torch.empty_like(xd_npu)
    torch.npu.synchronize()
    linear_att_conv_and_silu_paged(rt, xd_npu, conv, wn, yd, ids)
    torch.testing.assert_close(yd.cpu(), yd_ref, rtol=.025, atol=.004)
    torch.testing.assert_close(cr.cpu(), ref_cr, rtol=0, atol=0)
    print(f'PASS {dtype}/{state_dtype}: lengths={lengths}, Hk/Hv={kh}/{vh}, K/V={kd}/{vd}', flush=True)


def check_rejections(rt):
    x = torch.zeros(2, 192, dtype=torch.bfloat16, device='npu')
    conv = torch.ones(3, 3, 192, dtype=x.dtype, device='npu')
    weight = torch.ones(192, 4, dtype=x.dtype, device='npu')
    q = torch.zeros(2, 64, dtype=x.dtype, device='npu')
    gates = torch.zeros(2, 1, dtype=x.dtype, device='npu')
    state = torch.ones(3, 1, 64, 64, dtype=torch.float32, device='npu')
    ids = torch.ones(1, dtype=torch.int32, device='npu')
    starts = torch.tensor([0, 2], dtype=torch.int32, device='npu')
    initial = torch.zeros(1, dtype=torch.int32, device='npu')
    conv_args = [rt, x, conv, weight, torch.empty_like(x), ids, starts, initial]
    ssm_args = [rt, q, q, q, gates, gates, state, torch.empty_like(q),
                ids, starts, initial, 1, 1, 64, 64]
    cases = [
        (conv1d_and_silu_prefill_paged, conv_args, 6, starts.long()),
        (conv1d_and_silu_prefill_paged, conv_args, 7, initial.float()),
        (conv1d_and_silu_prefill_paged, conv_args, 5, ids.cpu()),
        (conv1d_and_silu_prefill_paged, conv_args, 4, x.T),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, 9, starts.long()),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, 10, initial.float()),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, 8, ids.cpu()),
        (recurrent_gated_delta_rule_prefill_paged, ssm_args, 9, starts[:1]),
    ]
    for function, args, index, bad_tensor in cases:
        args = args.copy()
        args[index] = bad_tensor
        before_conv, before_state = conv.clone(), state.clone()
        try:
            function(*args)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f'invalid descriptor accepted at argument {index}')
        assert torch.equal(conv, before_conv), 'rejected call changed Conv state'
        assert torch.equal(state, before_state), 'rejected call changed SSM state'
    print('PASS 8 Prefill descriptor/device rejections with unchanged state', flush=True)


def main(device):
    torch.npu.set_device(device)
    torch.manual_seed(143)
    rt = Runtime(device, 64)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for state_dtype in dict.fromkeys((dtype, torch.float32)):
            for kd, vd in ((16, 64), (64, 128), (128, 128)):
                check_case(rt, dtype, state_dtype, [1, 0, 65, 3, 7], kd=kd, vd=vd)
    for length in (512, 1024, 4096):
        check_case(rt, torch.bfloat16, torch.float32, [length, 1], kh=1, vh=3)
    check_case(rt, torch.bfloat16, torch.float32, [1, 0, 65, 3, 7], fp32_gates=True)
    check_rejections(rt)
    print('PASS 19 packed prefill cases, 38 continuation updates and 19 SSM + 19 Conv decode handoffs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    main(parser.parse_args().device)
