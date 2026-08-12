#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""PyTorch implementations of V4-specific kernels (sparse_attn, hc_split_sinkhorn).

These are simplified ports of the tilelang kernels in dsv4/inference/kernel.py so
that the V4 reference path runs on NPU without requiring tilelang compilation.
Correctness is preserved; performance is not optimized.
"""
import torch
import torch.nn.functional as F


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention via index gathering + softmax.

    Args:
        q: [b, s, h, d] bf16
        kv: [b, n, d] bf16 (n = window_size + compressed_kv_len)
        attn_sink: [h] fp32, learnable sink bias added to softmax denominator
        topk_idxs: [b, s, topk] int32, -1 means masked-out position
        softmax_scale: scalar

    Returns:
        o: [b, s, h, d] bf16
    """
    b, s, h, d = q.size()
    n = kv.size(1)
    topk = topk_idxs.size(-1)

    # Pad heads to >=16 only matters for the tilelang kernel; pure-torch path is fine with any h.

    # Clamp invalid indices to 0 and remember the mask.
    invalid = (topk_idxs < 0).to(q.device)  # [b, s, topk]
    idxs = topk_idxs.clamp(min=0).long().to(q.device)  # [b, s, topk]

    # Gather kv: [b, s, topk, d]
    # kv: [b, n, d] -> expand to [b, s, n, d] would blow memory; use gather along dim=1.
    # kv_expanded[b, i, t, d] = kv[b, idxs[b, i, t], d]
    # Implement via advanced indexing per batch.
    kv_gather = torch.empty(b, s, topk, d, dtype=kv.dtype, device=kv.device)
    for bi in range(b):
        kv_gather[bi] = kv[bi][idxs[bi]]  # [s, topk, d]
    # Mask out invalid positions: set their contribution to zero (via -inf scores).
    scores = torch.einsum("bshd,bstd->bsht", q, kv_gather) * softmax_scale  # [b, s, h, topk]
    scores = scores.masked_fill(invalid.unsqueeze(2), float("-inf"))

    # Numerically stable softmax with attn_sink folded into denominator.
    scores_max = scores.amax(dim=-1, keepdim=True)  # [b, s, h, 1]
    exp_scores = torch.exp(scores - scores_max)  # [b, s, h, topk]
    exp_scores = exp_scores.masked_fill(invalid.unsqueeze(2), 0.0)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)  # [b, s, h, 1]

    # attn_sink: [h] -> add exp(attn_sink - scores_max) to denominator (sink logit is learnable per-head).
    sink_term = torch.exp(attn_sink.view(1, 1, -1) - scores_max.squeeze(-1))  # [b, s, h]
    sum_exp = sum_exp.squeeze(-1) + sink_term  # [b, s, h]
    sum_exp = sum_exp.unsqueeze(-1)  # [b, s, h, 1]

    # Weighted sum: o = sum_t(exp_scores * kv_gather) / sum_exp
    o = torch.einsum("bsht,bstd->bshd", exp_scores, kv_gather)  # [b, s, h, d]
    o = o / sum_exp  # broadcast over d
    return o.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split the (2+hc_mult)*hc_mult mixing weights into pre/post/comb via Sinkhorn.

    Pure-PyTorch port of hc_split_sinkhorn_kernel in dsv4/inference/kernel.py.

    Args:
        mixes: [b, s, mix_hc] fp32, where mix_hc = (2 + hc_mult) * hc_mult
        hc_scale: [3] fp32
        hc_base: [mix_hc] fp32
        hc_mult: int
        sinkhorn_iters: int
        eps: float

    Returns:
        pre: [b, s, hc_mult] fp32
        post: [b, s, hc_mult] fp32
        comb: [b, s, hc_mult, hc_mult] fp32
    """
    b, s, mix_hc = mixes.size()
    assert mix_hc == (2 + hc_mult) * hc_mult

    m = mixes.view(-1, mix_hc)  # [b*s, mix_hc]
    n = m.size(0)

    # pre: [n, hc_mult], sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult]) + eps
    pre = torch.sigmoid(m[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps

    # post: [n, hc_mult], 2 * sigmoid(mixes[:, hc_mult:2*hc_mult] * scale[1] + base[hc_mult:2*hc_mult])
    post = 2 * torch.sigmoid(m[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])

    # comb: [n, hc_mult, hc_mult]
    comb = m[:, 2 * hc_mult:].view(n, hc_mult, hc_mult)
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult)

    # comb = comb.softmax(-1) + eps
    comb = comb.softmax(dim=-1) + eps
    # comb = comb / (comb.sum(-2) + eps)
    col_sum = comb.sum(dim=-2, keepdim=True)  # [n, 1, hc_mult]
    comb = comb / (col_sum + eps)

    for _ in range(sinkhorn_iters - 1):
        # comb = comb / (comb.sum(-1) + eps)
        row_sum = comb.sum(dim=-1, keepdim=True)  # [n, hc_mult, 1]
        comb = comb / (row_sum + eps)
        # comb = comb / (comb.sum(-2) + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)  # [n, 1, hc_mult]
        comb = comb / (col_sum + eps)

    pre = pre.view(b, s, hc_mult)
    post = post.view(b, s, hc_mult)
    comb = comb.view(b, s, hc_mult, hc_mult)
    return pre, post, comb
