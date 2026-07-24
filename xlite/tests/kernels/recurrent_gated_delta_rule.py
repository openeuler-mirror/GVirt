#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
import math

import torch
from xlite._C import Runtime, recurrent_gated_delta_rule


def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def torch_recurrent_gdr(query, key, value, g, beta, state, use_qk_l2norm=True):
    """Match qwen3_5._torch_recurrent_gated_delta_rule (no transpose in/out layout here).

    Inputs already token-major:
      q/k: [B, S, H, K], v: [B, S, H, V], beta/g: [B, S, H], state: [B, H, K, V]
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query.float(), dim=-1)
        key = l2norm(key.float(), dim=-1)
    else:
        query = query.float()
        key = key.float()
    value = value.float()
    beta = beta.float()
    g = g.float()
    state = state.float().clone()

    bsz, seqlen, num_heads, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1.0 / math.sqrt(k_dim)
    query = query * scale

    out = torch.zeros(bsz, seqlen, num_heads, v_dim, dtype=torch.float32, device=query.device)
    for s in range(seqlen):
        q_t = query[:, s]  # [B, H, K]
        k_t = key[:, s]
        v_t = value[:, s]
        g_t = g[:, s].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        beta_t = beta[:, s].unsqueeze(-1)  # [B, H, 1]

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, V]
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, s] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return out.to(initial_dtype), state.to(initial_dtype)


def run_case(rt, batch, seqlen, num_heads, k_dim, v_dim, dtype, msg):
    q = torch.randn(batch, seqlen, num_heads, k_dim, dtype=dtype)
    k = torch.randn(batch, seqlen, num_heads, k_dim, dtype=dtype)
    v = torch.randn(batch, seqlen, num_heads, v_dim, dtype=dtype)
    beta = torch.sigmoid(torch.randn(batch, seqlen, num_heads, dtype=dtype))
    # log-space decay (negative)
    g = -torch.nn.functional.softplus(
        torch.randn(batch, seqlen, num_heads, dtype=torch.float32)
    ).to(dtype)
    state = torch.randn(batch, num_heads, k_dim, v_dim, dtype=dtype)

    # Upstream already L2-norms in model path; exercise kernel without extra L2.
    q_n = l2norm(q.float(), dim=-1).to(dtype)
    k_n = l2norm(k.float(), dim=-1).to(dtype)

    ref_out, ref_state = torch_recurrent_gdr(
        q_n, k_n, v, g, beta, state.clone(), use_qk_l2norm=False
    )

    q_flat = q_n.reshape(batch * seqlen, num_heads * k_dim).contiguous()
    k_flat = k_n.reshape(batch * seqlen, num_heads * k_dim).contiguous()
    v_flat = v.reshape(batch * seqlen, num_heads * v_dim).contiguous()
    beta_flat = beta.reshape(batch * seqlen, num_heads).contiguous()
    g_flat = g.reshape(batch * seqlen, num_heads).contiguous()
    out_flat = torch.empty_like(v_flat)
    state_xlite = state.clone()

    torch.npu.synchronize()
    recurrent_gated_delta_rule(
        rt, q_flat, k_flat, v_flat, beta_flat, g_flat, state_xlite, out_flat,
        batch, seqlen, num_heads, k_dim, v_dim,
    )
    torch.npu.synchronize()

    my_out = out_flat.reshape(batch, seqlen, num_heads, v_dim)
    torch.testing.assert_close(my_out, ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_xlite, ref_state, rtol=2e-2, atol=2e-2)
    print(f"{msg}: PASS")


if __name__ == "__main__":
    rt = Runtime(0, 64)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")

    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for batch in [1, 2]:
            for seqlen in [1, 4, 8]:
                msg = f"[{dtype}/B={batch}/S={seqlen}/H=4/K=64/V=64]"
                run_case(rt, batch, seqlen, 4, 64, 64, dtype, msg)
        # Qwen3.5-0.8B-like head dims
        msg = f"[{dtype}/B=1/S=1/H=16/K=128/V=128]"
        run_case(rt, 1, 1, 16, 128, 128, dtype, msg)
