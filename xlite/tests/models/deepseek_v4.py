#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu

from tests.models.deepseek_kernel import weight_dequant
from tests.models.deepseek_v4_kernel import sparse_attn, hc_split_sinkhorn
from tests.models.weight_utils import (
    hf_model_weights_iterator,
    convert_pyslice_to_tensor,
    load_tensor_parallel_weights,
    logger,
)


world_size = 1
rank = 0
global_rank = 0
global_world_size = 1


@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    temperature: float = 1
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_mtp_layers: int = 0
    n_heads: int = 64
    norm_eps: float = 1e-6
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # mla / c4a / c128a
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # moe ep/tp
    moe_ep_size: int = 8
    moe_tp_size: int = 1
    quantization: Literal["none", "w8a8"] = "w8a8"


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to weight_dequant+F.linear for int8 weight, else F.linear.

    V4 w8a8 dynamic: weight is int8 with per-row weight_scale / weight_offset,
    dequantized to bf16 here before the GEMM. bf16 weights skip dequant.
    """
    assert bias is None
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    offset = getattr(weight, "weight_offset", None)
    weight = weight_dequant(weight, weight.scale, offset)
    return F.linear(x, weight, bias)


class Linear(nn.Module):
    """Linear layer supporting bf16 and int8 (w8a8 dynamic) weight formats."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
        if self.weight.element_size() == 1:
            self.weight.scale = self.scale = nn.Parameter(torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False)
            self.weight.weight_offset = self.weight_offset = nn.Parameter(torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("scale", None)
            self.register_parameter("weight_offset", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partials."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len and original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Randomized Hadamard rotation. Falls back to identity*scale when
    fast_hadamard_transform is unavailable (sufficient for correctness testing)."""
    assert x.dtype == torch.bfloat16
    try:
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=x.size(-1) ** -0.5)
    except ImportError:
        return x * (x.size(-1) ** -0.5)


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother boundaries."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None
        self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x.float()).float()
        score = self.wgate(x.float()).float()
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio: cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset: offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset: offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        # FP4/FP8 simulation skipped (w8a8 dynamic reference path keeps bf16).
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        # indexer.wq_b is W8A8_DYNAMIC -> int8 ColumnParallel
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim, dtype=torch.int8)
        # indexer.weights_proj is FLOAT -> bf16 Linear
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim, dtype=torch.bfloat16), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # FP4 simulation skipped.
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # QAT-quant simulation skipped: kv stays bf16.
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            dist.all_reduce(index_score)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, torch.full_like(topk_idxs, -1), topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs.int()


class Attention(nn.Module):
    """C4A / C128A: MLA + sliding window + optional KV compression + Indexer."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        # wq_a / wq_b / wkv: W8A8_DYNAMIC -> int8
        self.wq_a = Linear(self.dim, self.q_lora_rank, dtype=torch.int8)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim, dtype=torch.int8)
        self.wkv = Linear(self.dim, self.head_dim, dtype=torch.int8)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # wo_a / wo_b: FLOAT -> bf16
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim, dtype=torch.bfloat16), persistent=False)
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                         rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # FP8 simulation skipped (kv stays bf16).
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset).int()
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs.to(x.device), compress_topk_idxs.to(x.device)], dim=-1)
        else:
            topk_idxs = topk_idxs.to(x.device)

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE gating with hash-based routing for the first n_hash_layers."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        # gate.weight is FLOAT (bf16) in checkpoint; cast to fp32 for scoring stability (matches V3).
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.float32))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN. w1/w2/w3 are W8A8_DYNAMIC -> int8."""
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=torch.int8)
        self.w2 = Linear(inter_dim, dim, dtype=torch.int8)
        self.w3 = Linear(dim, inter_dim, dtype=torch.int8)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Mixture-of-Experts with EP across the full world."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % global_world_size == 0, f"Number of experts must be divisible by world size (world_size={global_world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // global_world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = global_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit) if self.experts_start_idx <= i < self.experts_end_idx else None
                                       for i in range(self.n_routed_experts)])
        assert args.n_shared_experts == 1
        # V4 shared expert is the same SwiGLU structure as routed experts.
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        if global_world_size > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Multi-stage Hyper-Connections (MHC)."""
    attention_cls = Attention

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class ParallelHead(nn.Module):
    """LM head sharded along vocab dim."""

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = vocab_size // world_size
        # head.weight is FLOAT (fp32) in checkpoint; keep fp32 for easier logits computation.
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        logits = F.linear(x.float(), self.weight)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


class Transformer(nn.Module):
    """DeepSeek-V4 model: embed -> HC-expand -> N blocks -> HC-head -> logits."""

    def __init__(self, args: ModelArgs):
        global world_size, rank, global_rank, global_world_size
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        global_rank = rank
        global_world_size = world_size
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.temperature = args.temperature
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        h = self.layers[-1].hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        logits = self.head(self.norm(h))
        return logits

    def load_weights(self, model_path: str):
        """Load DeepSeek-V4-Flash-w8a8-mtp checkpoint."""
        args = self.args
        assert args.dim % world_size == 0, f"dim must be divisible by world_size (world_size={world_size})"
        assert args.n_heads % world_size == 0, f"n_heads must be divisible by world_size (world_size={world_size})"
        assert args.vocab_size % world_size == 0, f"vocab_size must be divisible by world_size (world_size={world_size})"
        assert args.o_groups % world_size == 0, f"o_groups must be divisible by world_size (world_size={world_size})"
        assert args.index_n_heads % world_size == 0, f"index_n_heads must be divisible by world_size (world_size={world_size})"

        n_local_experts = args.n_routed_experts // args.moe_ep_size
        moe_tp_id = global_rank % args.moe_tp_size
        moe_ep_id = global_rank // args.moe_tp_size

        param_dict = {name: param for name, param in self.named_parameters()}

        for _, param in self.named_parameters():
            param.requires_grad = False

        for name, loaded_weight in hf_model_weights_iterator(model_path):
            if "rotary_emb.inv_freq" in name or "g_idx" in name:
                continue

            # Skip MTP layers (not implemented in this reference path).
            if name.startswith("mtp."):
                continue

            # Skip layer ids beyond n_layers.
            # V4 checkpoint naming: layers.<layer_id>.attn... / layers.<layer_id>.ffn...
            if name.startswith("layers."):
                layer_id = int(name.split(".")[1])
                if layer_id >= args.n_layers:
                    continue

            # EP shard of experts: skip experts not on this rank.
            # experts weight key: layers.<id>.ffn.experts.<eid>.w1.weight[_scale/_offset]
            # idx is the 3rd-from-last token regardless of whether name starts with "layers." or "model.layers."
            if "experts" in name and "shared_experts" not in name:
                idx = int(name.split(".")[-3])
                if idx < moe_ep_id * n_local_experts or idx >= (moe_ep_id + 1) * n_local_experts:
                    continue

            # Map V4 checkpoint naming convention -> model param naming.
            # V4 ckpt uses weight_scale / weight_offset (not scale / weight_offset param attrs).
            # The model params are named like layers.N.attn.wq_a.weight / .scale / .weight_offset.
            # Build target name by replacing weight_scale -> scale, weight_offset -> weight_offset.
            is_scale = name.endswith(".weight_scale")
            is_offset = name.endswith(".weight_offset")
            if is_scale:
                target_name = name[:-len("weight_scale")] + "scale"
            elif is_offset:
                target_name = name[:-len("weight_offset")] + "weight_offset"
            else:
                target_name = name

            if target_name not in param_dict:
                logger.warning('Loading model has no param named %s in checkpoints, bypass.',
                               target_name)
                continue

            param = param_dict[target_name]

            # ===== Embed / Head (vocab parallel) =====
            if "embed" in target_name and "weight" in target_name and "weight_offset" not in target_name and "weight_scale" not in target_name:
                load_tensor_parallel_weights(param, loaded_weight, args.vocab_size, args.dim,
                                             target_name, True, False, rank, world_size)
                continue
            if "head" in target_name and "weight" in target_name and "weight_offset" not in target_name and "weight_scale" not in target_name:
                # head.weight: param shape [part_vocab, dim], stored as fp32. Shard vocab dim.
                load_tensor_parallel_weights(param, loaded_weight, args.dim, args.vocab_size,
                                             target_name, False, True, rank, world_size)
                continue

            # ===== Per-row scale/offset for int8 Linear (shard along output dim) =====
            # For ColumnParallel int8 layers (wq_b, wo indexer.wq_b, indexer.weights_proj), shard rows.
            # For RowParallel int8 layers (wo_b), the weight is [N, part_in] -> scale/offset is [N, 1] (NOT sharded along N).
            # For plain int8 Linear (wq_a, wkv, experts.w1/w2/w3), scale/offset is full [N, 1] (NOT sharded).
            if is_scale or is_offset:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                # Determine if this layer is ColumnParallel (shard output dim).
                col_parallel_names = ("wq_b", "wo_a", "indexer.wq_b", "indexer.weights_proj")
                is_col_parallel = any(s in target_name for s in col_parallel_names) and "wo_b" not in target_name
                if is_col_parallel:
                    shard_size = param.shape[0]
                    loaded_weight = loaded_weight[rank * shard_size:(rank + 1) * shard_size]
                    loaded_weight = loaded_weight.contiguous()
                    if loaded_weight.shape[0] < shard_size:
                        # Pad in case of uneven sharding (shouldn't happen with proper asserts).
                        pad = torch.zeros(shard_size - loaded_weight.shape[0], param.shape[1],
                                           dtype=loaded_weight.dtype, device=loaded_weight.device)
                        loaded_weight = torch.cat([loaded_weight, pad], dim=0)
                param.data.copy_(loaded_weight)
                continue

            # ===== int8 weight tensor (or bf16 weight tensor) =====
            # experts w1/w2/w3 EP-shard on output dim (w1, w3) or input dim (w2).
            if "experts" in target_name and "shared_experts" not in target_name and ".weight" in target_name and target_name.rsplit(".", 1)[-1] == "weight":
                # Determine w1/w2/w3 by suffix.
                last_tok = target_name.split(".")[-2]
                if last_tok == "w1" or last_tok == "w3":
                    # ColumnParallel-like: output dim sharded by EP.
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data.copy_(loaded_weight)
                elif last_tok == "w2":
                    # RowParallel-like: input dim sharded by EP (but EP only, no TP for V4 MoE since moe_tp_size=1).
                    # When moe_ep_size==world_size, n_local_experts = n_routed/world; each rank loads its own experts' full w2.
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data.copy_(loaded_weight)
                else:
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data.copy_(loaded_weight)
                continue

            # ===== Dense layer TP sharding =====
            # wq_b: ColumnParallel (shard output dim).
            if target_name.endswith(".wq_b.weight") and "indexer" not in target_name:
                load_tensor_parallel_weights(param, loaded_weight, args.q_lora_rank,
                                             args.n_heads * args.head_dim,
                                             target_name, False, True, rank, world_size)
                continue
            # wo_a: ColumnParallel (shard output dim). Input dim is n_heads*head_dim/o_groups.
            if target_name.endswith(".wo_a.weight"):
                in_features = args.n_heads * args.head_dim // args.o_groups
                load_tensor_parallel_weights(param, loaded_weight, in_features,
                                             args.o_groups * args.o_lora_rank,
                                             target_name, False, True, rank, world_size)
                continue
            # wo_b: RowParallel (shard input dim).
            if target_name.endswith(".wo_b.weight"):
                load_tensor_parallel_weights(param, loaded_weight,
                                             args.o_groups * args.o_lora_rank, args.dim,
                                             target_name, True, True, rank, world_size)
                continue
            # indexer.wq_b: ColumnParallel (shard output dim).
            if target_name.endswith("indexer.wq_b.weight"):
                load_tensor_parallel_weights(param, loaded_weight, args.q_lora_rank,
                                             args.index_n_heads * args.index_head_dim,
                                             target_name, False, True, rank, world_size)
                continue
            # indexer.weights_proj: ColumnParallel (shard output dim).
            if target_name.endswith("indexer.weights_proj.weight"):
                load_tensor_parallel_weights(param, loaded_weight, args.dim,
                                             args.index_n_heads,
                                             target_name, False, True, rank, world_size)
                continue

            # attn_sink: per-head sink, shard along head dim (n_local_heads = n_heads // world_size).
            if target_name.endswith(".attn.attn_sink"):
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                shard_size = param.shape[0]
                loaded_weight = loaded_weight[rank * shard_size:(rank + 1) * shard_size]
                param.data.copy_(loaded_weight)
                continue

            # Fallback: direct copy.
            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            if param.shape != loaded_weight.shape:
                logger.warning('shape mismatch for %s: param=%s ckpt=%s',
                               target_name, tuple(param.shape), tuple(loaded_weight.shape))
            param.data.copy_(loaded_weight)

        torch.npu.empty_cache()
