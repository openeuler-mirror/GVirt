#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from dataclasses import dataclass
from typing import Optional, Literal

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tests.models.weight_utils import (hf_model_weights_iterator,
                                       convert_pyslice_to_tensor,
                                       load_tensor_parallel_weights,
                                       matrix_nd2nz, logger)

debug = False
world_size = 1
rank = 0

forward_backend = os.getenv("FORWARD_BACKEND", "torch_npu")
if forward_backend == "xlite":
    block_size = 128
    from xlite._C import Runtime, ModelConfig, ModelAttnMeta, AttnMHA, Model
    import numpy as np


@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096
    max_num_batched_tokens: int = 4096
    dim: int = 1024
    head_dim: int = 256
    inter_dim: int = 3584
    vocab_size: int = 248320
    n_layers: int = 24
    n_heads: int = 8
    n_kv_heads: int = 2
    norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    tie_word_embeddings: bool = True
    qkv_bias: bool = False
    qk_norm: bool = True
    full_attention_interval: int = 4
    # Linear attention (GatedDeltaNet) parameters
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    model_type: str = "qwen3_5"
    # RoPE parameters
    partial_rotary_factor: float = 0.25  # Only 25% of head_dim gets rotary embeddings

    def __post_init__(self):
        self.max_num_batched_tokens = self.max_seq_len * self.max_batch_size
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
        # Compute rotary dimension
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
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
    return F.linear(x, weight, bias)


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache = emb.cos().to(torch.get_default_dtype())
    sin_cache = emb.sin().to(torch.get_default_dtype())
    freq_cis = torch.cat((cos_cache, sin_cache), dim=-1)
    return freq_cis.to("npu")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    seqlen = x.size(2)
    cos, sin = freqs_cis[start_pos:start_pos + seqlen, :].chunk(2, dim=-1)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_embed, x_pass], dim=-1)


class MHA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.rotary_dim = args.rotary_dim  # rotary dimension (partial)
        self.qk_norm = args.qk_norm
        self.n_local_heads = args.n_heads // world_size
        # Use replication strategy when n_kv_heads < world_size (same as llama.py)
        self.n_local_kv_heads = max(1, args.n_kv_heads // world_size)
        # q_proj outputs both query and gate (2x head_dim)
        self.q_proj = ColumnParallelLinear(args.dim, args.n_heads * args.head_dim * 2, bias=args.qkv_bias)
        # k_proj/v_proj use ColumnParallelLinear with replication
        self.k_proj = ColumnParallelLinear(args.dim, self.n_local_kv_heads * world_size * args.head_dim, bias=args.qkv_bias)
        self.v_proj = ColumnParallelLinear(args.dim, self.n_local_kv_heads * world_size * args.head_dim, bias=args.qkv_bias)
        self.o_proj = RowParallelLinear(args.n_heads * args.head_dim, args.dim, bias=False)
        if args.qk_norm:
            self.q_norm = RMSNorm(args.head_dim, args.norm_eps)
            self.k_norm = RMSNorm(args.head_dim, args.norm_eps)
        if forward_backend != "xlite":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), persistent=False)

    def forward(self, x, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape

        # q_proj outputs both query and gate
        q_gate = self.q_proj(x)
        q_gate = q_gate.view(bsz, seqlen, self.n_local_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)

        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q = apply_rotary_emb(q, start_pos, freqs_cis=freqs_cis, rotary_dim=self.rotary_dim)
        k = apply_rotary_emb(k, start_pos, freqs_cis=freqs_cis, rotary_dim=self.rotary_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

        q = q * self.head_dim ** -0.5

        self.k_cache[:bsz, start_pos:start_pos + seqlen] = k
        self.v_cache[:bsz, start_pos:start_pos + seqlen] = v

        keys = self.k_cache[:bsz, :start_pos + seqlen]
        values = self.v_cache[:bsz, :start_pos + seqlen]

        keys = keys.repeat_interleave(self.n_local_heads // self.n_local_kv_heads, dim=2)
        values = values.repeat_interleave(self.n_local_heads // self.n_local_kv_heads, dim=2)

        scores = torch.matmul(
            q.transpose(1, 2),
            keys.permute(0, 2, 3, 1)
        )

        if mask is not None:
            scores = scores.float() + mask.float()

        scores = torch.softmax(scores, dim=-1, dtype=torch.float).to(x.dtype)

        output = torch.matmul(
            scores,
            values.transpose(1, 2)
        ).transpose(1, 2).contiguous()

        # Apply sigmoid gate
        gate = gate.reshape(bsz, seqlen, self.n_local_heads * self.head_dim)
        output = output.reshape(bsz, seqlen, self.n_local_heads * self.head_dim)
        output = output * torch.sigmoid(gate)

        return self.o_proj(output)


class LinearAttn(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.num_k_heads = args.linear_num_key_heads
        self.num_v_heads = args.linear_num_value_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.conv_kernel_dim = args.linear_conv_kernel_dim

        # Local heads for tensor parallel
        self.num_local_k_heads = args.linear_num_key_heads // world_size
        self.num_local_v_heads = args.linear_num_value_heads // world_size

        key_dim = args.linear_num_key_heads * args.linear_key_head_dim
        value_dim = args.linear_num_value_heads * args.linear_value_head_dim
        local_key_dim = self.num_local_k_heads * self.head_k_dim
        local_value_dim = self.num_local_v_heads * self.head_v_dim

        self.conv_dim = local_key_dim * 2 + local_value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim,
            padding=self.conv_kernel_dim - 1,
        )
        self.in_proj_qkv = ColumnParallelLinear(args.dim, key_dim * 2 + value_dim, bias=False)
        self.in_proj_z = ColumnParallelLinear(args.dim, value_dim, bias=False)
        self.in_proj_b = ColumnParallelLinear(args.dim, args.linear_num_value_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(args.dim, args.linear_num_value_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.num_local_v_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.num_local_v_heads))
        self.norm = RMSNorm(self.head_v_dim, args.norm_eps)
        self.out_proj = RowParallelLinear(value_dim, args.dim, bias=False)
        if forward_backend != "xlite":
            self.register_buffer("conv_state", torch.zeros(args.max_batch_size, self.conv_dim, self.conv_kernel_dim), persistent=False)
            self.register_buffer("ssm_state", torch.zeros(args.max_batch_size, self.num_local_v_heads, self.head_k_dim, self.head_v_dim), persistent=False)

    def _l2norm(self, x, dim=-1, eps=1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    def _torch_causal_conv1d_update(self, hidden_states, conv_state, weight, activation='silu'):
        _, hidden_size, seq_len = hidden_states.shape
        state_len = conv_state.shape[-1]

        hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(hidden_states_new[:, :, -state_len:])
        # weight shape: [conv_dim, 1, kernel_dim] for nn.Conv1d
        # F.conv1d expects [out_channels, in_channels/groups, kernel_size]
        # For depthwise conv with groups=conv_dim, weight shape is already correct
        out = F.conv1d(hidden_states_new, weight, padding=0, groups=hidden_size)
        out = F.silu(out[:, :, -seq_len:])
        return out.to(hidden_states.dtype)

    def _torch_chunk_gated_delta_rule(self, query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm=True):
        initial_dtype = query.dtype

        if use_qk_l2norm:
            query = self._l2norm(query.float(), dim=-1)
            key = self._l2norm(key.float(), dim=-1)

        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        chunk_size = 64

        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))

        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def _torch_recurrent_gated_delta_rule(self, query, key, value, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm=True):
        initial_dtype = query.dtype

        if use_qk_l2norm:
            query = self._l2norm(query.float(), dim=-1)
            key = self._l2norm(key.float(), dim=-1)

        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )

        for i in range(sequence_length):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def forward(self, x, start_pos: int):
        bsz, seqlen, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = z.view(bsz, seqlen, self.num_local_v_heads, self.head_v_dim)

        beta = torch.sigmoid(b.float())
        g = -self.A_log.float().exp().unsqueeze(0).unsqueeze(0) * F.softplus(a.float() + self.dt_bias.float().unsqueeze(0).unsqueeze(0))
        g = g.to(x.dtype)
        beta = beta.to(x.dtype)

        use_precomputed_states = seqlen == 1 and start_pos > 0

        if use_precomputed_states:
            mixed_qkv = self._torch_causal_conv1d_update(
                mixed_qkv,
                self.conv_state[:bsz],
                self.conv1d.weight,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seqlen])
            original_mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
            self.conv_state[:bsz] = F.pad(original_mixed_qkv, (self.conv_kernel_dim - seqlen, 0))

        mixed_qkv = mixed_qkv.transpose(1, 2)

        local_key_dim = self.num_local_k_heads * self.head_k_dim
        local_value_dim = self.num_local_v_heads * self.head_v_dim
        query, key, value = mixed_qkv.split([local_key_dim, local_key_dim, local_value_dim], dim=-1)

        query = query.view(bsz, seqlen, self.num_local_k_heads, self.head_k_dim)
        key = key.view(bsz, seqlen, self.num_local_k_heads, self.head_k_dim)
        value = value.view(bsz, seqlen, self.num_local_v_heads, self.head_v_dim)

        if self.num_local_v_heads > self.num_local_k_heads:
            expand_factor = self.num_local_v_heads // self.num_local_k_heads
            query = query.repeat_interleave(expand_factor, dim=2)
            key = key.repeat_interleave(expand_factor, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self._torch_chunk_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm=True,
            )
            if last_recurrent_state is not None:
                self.ssm_state[:bsz] = last_recurrent_state
        else:
            core_attn_out, last_recurrent_state = self._torch_recurrent_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=self.ssm_state[:bsz],
                output_final_state=True,
                use_qk_l2norm=True,
            )
            if last_recurrent_state is not None:
                self.ssm_state[:bsz] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(bsz * seqlen, self.num_local_v_heads, self.head_v_dim)
        z = z.reshape(bsz * seqlen, self.num_local_v_heads, self.head_v_dim)
        core_attn_out = core_attn_out.to(torch.float32)
        variance = core_attn_out.pow(2).mean(-1, keepdim=True)
        core_attn_out = core_attn_out * torch.rsqrt(variance + self.norm.eps)
        core_attn_out = self.norm.weight * core_attn_out.to(x.dtype)
        core_attn_out = core_attn_out * F.silu(z.to(torch.float32)).to(x.dtype)

        core_attn_out = core_attn_out.view(bsz, seqlen, self.num_local_v_heads * self.head_v_dim)

        return self.out_proj(core_attn_out)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(dim, inter_dim * 2)
        self.down_proj = RowParallelLinear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gate_up_proj(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.down_proj(F.silu(y1) * y3)


def is_layer_full_attention(args: ModelArgs, layer_id: int):
    return (layer_id + 1) % args.full_attention_interval == 0


class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.is_full_attention = is_layer_full_attention(args, layer_id)
        if self.is_full_attention:
            self.self_attn = MHA(args)
        else:
            self.linear_attn = LinearAttn(args)
        self.mlp = MLP(dim=args.dim, inter_dim=args.inter_dim)
        self.input_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_full_attention:
            x = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask)
        else:
            x = x + self.linear_attn(self.input_layernorm(x), start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3_5(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.embed_tokens = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            self.layers.append(Block(args, i))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = ColumnParallelLinear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.rotary_dim, args.max_seq_len, args.rope_theta)

    @torch.inference_mode()
    def forward_naive(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)

        for layer in self.layers:
            h = layer(h, start_pos, self.freqs_cis, mask)

        h = self.norm(h)[:, -1]

        logits = self.lm_head(h)

        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)

        return logits

    @torch.inference_mode()
    def forward_xlite(self, tokens: torch.Tensor, start_pos: int = 0):
        logits = torch.empty(world_size, tokens.size(0), self.args.vocab_size // world_size, device=tokens.device)
        tokens = tokens.contiguous().view(tokens.size(0), tokens.size(1))
        attn_meta = self.prepare_xlite_attnmeta(tokens, start_pos)
        stream = torch.npu.current_stream().npu_stream
        h = torch.empty(tokens.numel(), self.args.dim, device=tokens.device)
        self.xlite_model.forward(self.xlite_rt, tokens.flatten(), attn_meta, self.xlite_kv_cache, self.freqs_cis, h, stream)
        self.xlite_model.forward_get_logits(self.xlite_rt, h, logits)
        logits = logits.permute(1, 0, 2).reshape(tokens.size(0), self.args.vocab_size)
        return logits

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        if forward_backend == "xlite":
            return self.forward_xlite(tokens, start_pos)
        else:
            return self.forward_naive(tokens, start_pos)

    def load_weights(self, model_path: str) -> None:
        assert self.args.dim % world_size == 0, f"dim must be divisible by world_size (world_size={world_size})"
        assert self.args.n_heads % world_size == 0, f"n_heads must be divisible by world_size (world_size={world_size})"
        assert self.args.inter_dim % world_size == 0, f"inter_dim must be divisible by world_size (world_size={world_size})"
        assert self.args.vocab_size % world_size == 0, f"vocab_size must be divisible by world_size (world_size={world_size})"

        self.xlite_weight_nz = True if forward_backend == "xlite" else False

        n_kv_heads_replicas = max(1, world_size // self.args.n_kv_heads)
        n_local_kv_heads = max(1, self.args.n_kv_heads // world_size)
        q_proj_shard_size = (self.args.head_dim * 2 * self.args.n_heads // world_size)
        kv_proj_shard_size = self.args.head_dim * n_local_kv_heads

        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, 0),
            ("v_proj", kv_proj_shard_size, 0),
        ]

        param_dict = {name if "lm_head" in name else "model." + name: param for name, param in self.named_parameters()}
        for _, param in self.named_parameters():
            param.requires_grad = False

        for name, loaded_weight in hf_model_weights_iterator(model_path):
            if "rotary_emb.inv_freq" in name or "g_idx" in name:
                continue

            if "visual" in name:
                continue

            # Skip MTP weights
            if "mtp" in name:
                continue

            # Handle weight name prefix
            if name.startswith("model.language_model"):
                name = name.replace("model.language_model", "model")

            # Handle linear_attn weights
            if "linear_attn" in name:
                # Handle conv1d.weight - keep the .weight suffix for nn.Conv1d
                if "conv1d.weight" in name:
                    # name after prefix replacement already has .weight suffix
                    if name not in param_dict:
                        logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                        continue
                    param = param_dict[name]
                    # conv1d weight: [total_conv_dim, 1, kernel_dim]
                    # total_conv_dim = key_dim * 2 + value_dim (for depthwise conv)
                    # For TP, we need to shard each component separately by heads
                    if world_size == 1:
                        loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    else:
                        key_dim = self.args.linear_num_key_heads * self.args.linear_key_head_dim
                        value_dim = self.args.linear_num_value_heads * self.args.linear_value_head_dim
                        loaded_weight_tensor = convert_pyslice_to_tensor(loaded_weight)

                        # Extract shards from each component
                        local_key_dim = key_dim // world_size
                        local_value_dim = value_dim // world_size

                        q_shard = loaded_weight_tensor[local_key_dim * rank:local_key_dim * (rank + 1), :, :]
                        k_shard = loaded_weight_tensor[key_dim + local_key_dim * rank:key_dim + local_key_dim * (rank + 1), :, :]
                        v_shard = loaded_weight_tensor[key_dim * 2 + local_value_dim * rank:key_dim * 2 + local_value_dim * (rank + 1), :, :]

                        # Concatenate shards: [q_shard, k_shard, v_shard]
                        loaded_weight = torch.cat([q_shard, k_shard, v_shard], dim=0)

                    if param.shape != loaded_weight.shape:
                        logger.warning(f'{name} model shape({param.shape}) mismatch checkpoint slice({loaded_weight.shape})')
                        continue
                    param.data.copy_(loaded_weight)
                    continue

                if "in_proj_qkv" in name:
                    if name not in param_dict:
                        logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                        continue
                    param = param_dict[name]
                    key_dim = self.args.linear_num_key_heads * self.args.linear_key_head_dim
                    value_dim = self.args.linear_num_value_heads * self.args.linear_value_head_dim
                    # Weight layout: [q(16heads), k(16heads), v(16heads)] = [key_dim, key_dim, value_dim]
                    # For TP, we need to shard each component separately by heads
                    # Each rank gets: q_shard + k_shard + v_shard
                    local_key_dim = key_dim // world_size
                    local_value_dim = value_dim // world_size

                    # Extract shards from each component
                    loaded_weight_tensor = convert_pyslice_to_tensor(loaded_weight)
                    q_shard = loaded_weight_tensor[local_key_dim * rank:local_key_dim * (rank + 1), :]
                    k_shard = loaded_weight_tensor[key_dim + local_key_dim * rank:key_dim + local_key_dim * (rank + 1), :]
                    v_shard = loaded_weight_tensor[key_dim * 2 + local_value_dim * rank:key_dim * 2 + local_value_dim * (rank + 1), :]

                    # Concatenate shards: [q_shard, k_shard, v_shard]
                    loaded_weight_shard = torch.cat([q_shard, k_shard, v_shard], dim=0)
                    param.data.copy_(loaded_weight_shard)
                    continue

                if "in_proj_z" in name:
                    if name not in param_dict:
                        logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                        continue
                    param = param_dict[name]
                    value_dim = self.args.linear_num_value_heads * self.args.linear_value_head_dim
                    shard_size = value_dim // world_size
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight[shard_size * rank:shard_size * (rank + 1), :])
                    param.data.copy_(loaded_weight)
                    continue

                if "out_proj" in name:
                    if name not in param_dict:
                        logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                        continue
                    param = param_dict[name]
                    value_dim = self.args.linear_num_value_heads * self.args.linear_value_head_dim
                    shard_size = value_dim // world_size
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight[:, shard_size * rank:shard_size * (rank + 1)])
                    param.data.copy_(loaded_weight)
                    continue

                if "in_proj_a" in name or "in_proj_b" in name:
                    if name not in param_dict:
                        logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                        continue
                    param = param_dict[name]
                    num_v_heads = self.args.linear_num_value_heads
                    shard_size = num_v_heads // world_size
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight[shard_size * rank:shard_size * (rank + 1), :])
                    if param.shape != loaded_weight.shape:
                        logger.warning(f'{name} model shape({param.shape}) mismatch checkpoint slice({loaded_weight.shape})')
                        continue
                    param.data.copy_(loaded_weight)
                    continue

                if "A_log" in name or "dt_bias" in name:
                    if name not in param_dict:
                        logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                        continue
                    param = param_dict[name]
                    num_v_heads = self.args.linear_num_value_heads
                    shard_size = num_v_heads // world_size
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight[shard_size * rank:shard_size * (rank + 1)])
                    param.data.copy_(loaded_weight)
                    continue

                if "linear_attn.norm" in name:
                    if name not in param_dict:
                        logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                        continue
                    param = param_dict[name]
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data.copy_(loaded_weight)
                    continue

                continue

            # Handle attention weights
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue

                param_name = name
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.', param_name)
                    continue

                param = param_dict[param_name]
                # Use replication strategy for k_proj/v_proj (same as llama.py)
                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = rank // n_kv_heads_replicas
                else:
                    shard_id = rank

                loaded_weight_slice = loaded_weight[shard_size * shard_id: shard_size * (shard_id + 1)]
                loaded_weight_slice = convert_pyslice_to_tensor(loaded_weight_slice)

                if param.shape != loaded_weight_slice.shape:
                    raise ValueError(f"{param_name} model shape({param.shape}) mismatch"
                                     f" checkpoint shape({loaded_weight_slice.shape})")

                param.data.copy_(loaded_weight_slice)
                is_attention_weight = True
                break

            if is_attention_weight:
                continue

            # Handle gate_up_proj merged weights
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue

                param_name = name.replace(weight_name, "gate_up_proj")
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.', param_name)
                    continue
                param = param_dict[param_name]
                shard_size = param.shape[0] // 2

                loaded_weight = loaded_weight[shard_size * rank:shard_size * (rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size * (stride_id + 1)]
                param_slice.data[:loaded_weight.shape[0]].copy_(loaded_weight)

                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            if name not in param_dict:
                logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                continue
            param = param_dict[name]

            if "embed_tokens" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.vocab_size,
                                             self.args.dim,
                                             name, True, False, rank, world_size)
                continue

            if "lm_head" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.dim,
                                             self.args.vocab_size,
                                             name, False, True, rank, world_size)
                continue

            if "o_proj" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.head_dim * self.args.n_heads, self.args.dim,
                                             name, True, True, rank, world_size)
                continue

            if "down_proj" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.inter_dim, self.args.dim,
                                             name, True, True, rank, world_size)
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            param.copy_(loaded_weight)
            torch.npu.empty_cache()

        if self.args.tie_word_embeddings:
            self.lm_head.weight.data = self.embed_tokens.weight.data

        if self.xlite_weight_nz:
            self.lm_head.weight.data = matrix_nd2nz(self.lm_head.weight)
            for layer_id, layer in enumerate(self.layers):
                if layer.is_full_attention:
                    layer.self_attn.q_proj.weight.data = matrix_nd2nz(layer.self_attn.q_proj.weight)
                    layer.self_attn.k_proj.weight.data = matrix_nd2nz(layer.self_attn.k_proj.weight)
                    layer.self_attn.v_proj.weight.data = matrix_nd2nz(layer.self_attn.v_proj.weight)
                    layer.self_attn.o_proj.weight.data = matrix_nd2nz(layer.self_attn.o_proj.weight)
                else:
                    layer.linear_attn.in_proj_qkv.weight.data = matrix_nd2nz(layer.linear_attn.in_proj_qkv.weight)
                    layer.linear_attn.in_proj_z.weight.data = matrix_nd2nz(layer.linear_attn.in_proj_z.weight)
                    layer.linear_attn.in_proj_a.weight.data = matrix_nd2nz(layer.linear_attn.in_proj_a.weight)
                    layer.linear_attn.in_proj_b.weight.data = matrix_nd2nz(layer.linear_attn.in_proj_b.weight)
                    layer.linear_attn.out_proj.weight.data = matrix_nd2nz(layer.linear_attn.out_proj.weight)
                    layer.linear_attn.conv1d.weight.data = matrix_nd2nz(layer.linear_attn.conv1d.weight)
                layer.mlp.gate_up_proj.weight.data = matrix_nd2nz(layer.mlp.gate_up_proj.weight)
                layer.mlp.down_proj.weight.data = matrix_nd2nz(layer.mlp.down_proj.weight)
            torch.npu.empty_cache()

        if forward_backend == "xlite":
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.xlite_rt = Runtime(local_rank, 0, rank, world_size)
            self.init_xlite_model(self.args)
            kv_size = self.init_xlite_kvcache(self.args)
            pool_size = self.xlite_model.get_tensor_pool_size()
            self.xlite_rt.init_tensor_pool(pool_size)

            total_model_memory = 0
            for _, param in self.named_parameters():
                memory_usage = param.element_size() * param.numel()
                total_model_memory += memory_usage
            if rank == 0:
                print(f"Memory usage: Model: {total_model_memory // 1024 // 1024} MB" +
                      f" KV Cache: {kv_size // 1024 // 1024} MB" +
                      f" Tensor pool: {pool_size} MB")

    def init_xlite_model(self, args: ModelArgs):
        config = ModelConfig()
        config.vocab_size = args.vocab_size
        config.hidden_size = args.dim
        config.n_layers = args.n_layers
        config.n_heads = args.n_heads
        config.n_kv_heads = args.n_kv_heads
        config.head_dim = args.head_dim
        config.rope_head_dim = args.head_dim
        config.norm_eps = args.norm_eps
        config.rope_theta = args.rope_theta
        config.softmax_scale = args.head_dim ** -0.5
        config.n_dense_layers = args.n_layers
        config.intermediate_size = args.inter_dim
        config.def_tp_size = world_size
        config.def_dp_size = 1
        config.moe_ep_size = 1
        config.moe_tp_size = 1
        config.block_size = block_size
        config.max_seq_len = args.max_seq_len
        config.max_batch_size = args.max_batch_size
        config.max_num_batched_tokens = args.max_num_batched_tokens
        config.attn_type = AttnMHA
        config.weight_nz = self.xlite_weight_nz
        config.qkv_bias = args.qkv_bias
        config.qk_norm = args.qk_norm

        self.xlite_model = Model()
        self.xlite_model.embed = self.embed_tokens.weight
        self.xlite_model.norm = self.norm.weight
        self.xlite_model.head = self.lm_head.weight
        self.xlite_model.attn_norm = [layer.input_layernorm.weight for layer in self.layers]
        self.xlite_model.attn_out = [layer.self_attn.o_proj.weight for layer in self.layers if layer.is_full_attention]
        # Combine q, k, v projections for full attention layers
        self.xlite_model.mha_qkv = [
            torch.cat([
                layer.self_attn.q_proj.weight,
                layer.self_attn.k_proj.weight,
                layer.self_attn.v_proj.weight
            ], dim=0) for layer in self.layers if layer.is_full_attention
        ]
        self.xlite_model.mlp_norm = [layer.post_attention_layernorm.weight for layer in self.layers]
        self.xlite_model.mlp_up_gate = [layer.mlp.gate_up_proj.weight for layer in self.layers]
        self.xlite_model.mlp_down = [layer.mlp.down_proj.weight for layer in self.layers]
        if args.qk_norm:
            self.xlite_model.mha_q_norm = [layer.self_attn.q_norm.weight for layer in self.layers if layer.is_full_attention]
            self.xlite_model.mha_k_norm = [layer.self_attn.k_norm.weight for layer in self.layers if layer.is_full_attention]
        if args.qkv_bias:
            self.xlite_model.mha_qkv_bias = [
                torch.cat([
                    layer.self_attn.q_proj.bias,
                    layer.self_attn.k_proj.bias,
                    layer.self_attn.v_proj.bias
                ], dim=0) for layer in self.layers if layer.is_full_attention
            ]

        self.xlite_model.init(config, rank)

    def init_xlite_kvcache(self, args: ModelArgs):
        block_num = (args.max_seq_len + block_size - 1) // block_size * args.max_batch_size
        head_num = max(args.n_kv_heads // world_size, 1)
        # Only full attention layers need KV cache
        n_full_attn_layers = sum(1 for layer in self.layers if layer.is_full_attention)
        self.xlite_kv_cache = [(torch.zeros(block_num, block_size, head_num, args.head_dim, dtype=torch.get_default_dtype(), device='npu'),
                                torch.zeros(block_num, block_size, head_num, args.head_dim, dtype=torch.get_default_dtype(), device='npu'))
                               for _ in range(n_full_attn_layers)]
        kv_size = (block_num * head_num * block_size * (args.head_dim + args.head_dim) *
                   self.xlite_kv_cache[0][0].element_size() * n_full_attn_layers)
        return kv_size

    def prepare_xlite_attnmeta(self, tokens: torch.Tensor, start_pos: int):
        batch = tokens.size(0)
        seqlen = tokens.size(1)
        step = (self.args.max_seq_len + block_size - 1) // block_size
        block_num = (seqlen + start_pos + block_size - 1) // block_size
        attn_meta = ModelAttnMeta()
        attn_meta.lens = [seqlen] * batch
        attn_meta.cached_lens = [start_pos] * batch
        attn_meta.is_prefills = [True if seqlen != 1 else False] * batch
        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(block_num, dtype=np.uint32)
        attn_meta.block_tables = batch_indices * step + block_indices
        return attn_meta
