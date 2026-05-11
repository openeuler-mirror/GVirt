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
from typing import Tuple, Optional, Literal

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
    from xlite._C import Runtime, ModelConfig, ModelAttnMeta, AttnMHA, Model, ScoringFuncSigmoid
    import numpy as np

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096
    max_num_batched_tokens: int = 4096
    dim: int = 5120
    head_dim: int = 128
    inter_dim: int = 12288
    moe_inter_dim: int = 1536
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 3
    n_routed_experts: int = 160
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    vocab_size: int = 151552
    n_layers: int = 92
    n_heads: int = 96
    n_kv_heads: int = 8
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.5
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    qkv_bias: bool = True
    qk_norm: bool = True
    qk_norm_full: bool = False
    moe_ep_size: int = 1
    moe_tp_size: int = 1
    model_type: str = "glm4moe"

    def __post_init__(self):
        self.max_num_batched_tokens = self.max_seq_len * self.max_batch_size
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads

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
        return (self.weight.float() * x).to(input_dtype)


class MinimaxBatchedTokens2RMSNorm(nn.Module):
    def __init__(self, head: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head * dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        input_shape = x.shape
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(variance)
        variance = variance / world_size
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x.reshape(*x.shape[:-1], -1)).reshape(input_shape).to(input_dtype)


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
    cos_cache = freqs.cos().to(torch.get_default_dtype())
    sin_cache = freqs.sin().to(torch.get_default_dtype())
    freq_cis = torch.cat((cos_cache, sin_cache), dim=-1)
    return freq_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor) -> torch.Tensor:
    seqlen = x.size(2) # [bsz, n_local_heads, seqlen, head_dim]
    cos, sin = freqs_cis[start_pos:start_pos + seqlen, :].chunk(2, dim=-1)
    cos = cos.repeat(1, 2) # [seqlen, head_dim]
    sin = sin.repeat(1, 2)

    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x1 = x_rot[..., :x_rot.shape[-1] // 2]
    x2 = x_rot[..., x_rot.shape[-1] // 2:]
    x_rot_half = torch.cat((-x2, x1), dim=-1)

    x_rot_embedded = (x_rot * cos) + (x_rot_half * sin)
    return torch.cat([x_rot_embedded, x_pass], dim=-1)


class MHA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.qk_norm = args.qk_norm
        self.qk_norm_full = args.qk_norm_full
        self.n_local_heads = args.n_heads // world_size
        self.n_local_kv_heads = max(1, args.n_kv_heads // world_size)
        self.qkv_proj = ColumnParallelLinear(args.dim, (self.n_heads + 2 * self.n_local_kv_heads * world_size) * self.head_dim, bias=args.qkv_bias)
        self.o_proj = RowParallelLinear(self.n_heads * self.head_dim, args.dim, bias=False)
        if args.qk_norm:
            self.q_norm = RMSNorm(args.head_dim, args.norm_eps) if not args.qk_norm_full else MinimaxBatchedTokens2RMSNorm(self.n_local_heads, args.head_dim, args.norm_eps)
            self.k_norm = RMSNorm(args.head_dim, args.norm_eps) if not args.qk_norm_full else MinimaxBatchedTokens2RMSNorm(self.n_local_kv_heads, args.head_dim, args.norm_eps)
        if forward_backend != "xlite":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), persistent=False)

    def forward(self, x, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([
            self.n_local_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim
        ], dim=2)

        if self.qk_norm_full:
            q = q.view(bsz, seqlen, self.n_local_heads * self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_kv_heads * self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_kv_heads * self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.qk_norm_full:
            q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q = apply_rotary_emb(q, start_pos, freqs_cis=freqs_cis)
        k = apply_rotary_emb(k, start_pos, freqs_cis=freqs_cis)
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
            q.transpose(1, 2),  # [bsz, n_local_heads, seqlen, head_dim]
            keys.permute(0, 2, 3, 1)  # [bsz, n_local_heads, head_dim, seqlen+start_pos]
        ) # [bsz, n_local_heads, seqlen, seqlen+start_pos]

        if mask is not None:
            scores = scores.float() + mask.float()

        scores = torch.softmax(scores, dim=-1, dtype=torch.float).to(x.dtype)

        output = torch.matmul(
            scores,  # [bsz, n_local_heads, seqlen, seqlen+start_pos]
            values.transpose(1, 2)  # [bsz, n_local_heads, seqlen+start_pos, head_dim]
        ).transpose(1, 2).contiguous()  # [bsz, seqlen, n_local_heads, head_dim]

        output = output.reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(dim, inter_dim * 2)
        self.down_proj = RowParallelLinear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gate_up_proj(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.down_proj(F.silu(y1) * y3)


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        weight (torch.nn.Parameter): Learnable weights for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_routed_experts = args.n_routed_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.e_score_correction_bias = nn.Parameter(torch.empty(args.n_routed_experts))
        self.norm_topk_prob = args.norm_topk_prob

    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.to(scores.dtype).unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.topk, dim=-1, sorted=False)[1]
        return topk_indices
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x.type(torch.float32), self.weight.type(torch.float32))
        scores = scores.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.type_as(x), topk_indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    """
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.gate_up_proj = Linear(dim, inter_dim * 2 // args.moe_tp_size)
        self.down_proj = Linear(inter_dim // args.moe_tp_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        y = self.gate_up_proj(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.down_proj(F.silu(y1) * y3)


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        assert args.n_routed_experts % args.moe_ep_size == 0, f"Number of experts must be divisible by moe ep size (moe_ep_size={args.moe_ep_size})"
        moe_ep_id = rank // args.moe_tp_size
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // args.moe_ep_size
        self.n_activated_experts = args.n_activated_experts
        self.n_shared_experts = args.n_shared_experts
        self.experts_start_idx = moe_ep_id * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, args) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        if args.n_shared_experts != 0:
            self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        if self.n_shared_experts != 0:
            z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        if self.n_shared_experts != 0:
            return (y + z).view(shape)
        else:
            return y.view(shape)

def is_layer_moe(args: ModelArgs, layer_id: int):
    return (layer_id >= args.first_k_dense_replace)

class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.self_attn = MHA(args)
        if is_layer_moe(args, layer_id):
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(dim=args.dim, inter_dim=args.inter_dim)
        self.input_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.first_k_dense_replace = args.first_k_dense_replace
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_norm_out = self.input_layernorm(x)
        if debug and rank == 0 and (self.layer_id == 0 or self.layer_id == self.first_k_dense_replace):
            print(f"layer{self.layer_id} in: {attn_norm_out}")
        attn_out = self.self_attn(attn_norm_out, start_pos, freqs_cis, mask)
        if debug and rank == 0 and (self.layer_id == 0 or self.layer_id == self.first_k_dense_replace):
            print(f"layer{self.layer_id} after attn: {attn_out}")
        x = x + attn_out
        ffn_norm_out = self.post_attention_layernorm(x)
        ffn_out = self.mlp(ffn_norm_out)
        if debug and rank == 0 and (self.layer_id == 0 or self.layer_id == self.first_k_dense_replace):
            print(f"layer{self.layer_id} after ffn: {ffn_out}")
        x = x + ffn_out
        return x


class GLM4MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        assert args.moe_ep_size * args.moe_tp_size == world_size, (f"moe parallel size(moe_ep_size={args.moe_ep_size}, moe_tp_size={args.moe_tp_size}) "
                                                                   f"must be same with word size(world_size={world_size})")
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
        self.freqs_cis = precompute_freqs_cis(int(args.head_dim * args.partial_rotary_factor), args.max_seq_len, args.rope_theta)


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


    def load_weights(
            self,
            model_path : str,
    ) -> None:
        """ load glm4 moe weight """
        assert self.args.dim % world_size == 0, f"dim must be divisible by world_size (world_size={world_size})"
        assert self.args.n_heads % world_size == 0, f"n_heads must be divisible by world_size (world_size={world_size})"
        assert self.args.inter_dim % world_size == 0, f"inter_dim must be divisible by world_size (world_size={world_size})"
        assert self.args.vocab_size % world_size == 0, f"vocab_size must be divisible by world_size (world_size={world_size})"

        self.xlite_weight_nz = True if forward_backend == "xlite" else False

        q_proj_shard_size = (self.args.head_dim * self.args.n_heads // world_size)
        n_kv_heads_replicas = max(1, world_size // self.args.n_kv_heads)
        n_local_kv_heads = max(1, self.args.n_kv_heads // world_size)
        kv_proj_shard_size = (self.args.head_dim * n_local_kv_heads)
        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        n_local_experts = self.args.n_routed_experts // self.args.moe_ep_size
        moe_tp_id = rank % self.args.moe_tp_size
        moe_ep_id = rank // self.args.moe_tp_size
        param_dict = {name if "lm_head" in name else "model." + name: param for name, param in self.named_parameters()}
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, loaded_weight in hf_model_weights_iterator(model_path):
            if "rotary_emb.inv_freq" in name or "g_idx" in name:
                continue

            # skip mtp layer
            if name.startswith("model.layers."):
                layer_id = int(name.split(".")[2])
                if layer_id >= self.args.n_layers:
                    continue

            if "experts" in name and "shared_experts" not in name:
                idx = int(name.split(".")[-3])
                if idx < moe_ep_id * n_local_experts or idx >= (moe_ep_id + 1) * n_local_experts:
                    continue

            name = name.replace("block_sparse_moe", "mlp")
            name = name.replace("w2", "down_proj")
            name = name.replace("w1", "gate_proj")
            name = name.replace("w3", "up_proj")
            name = name.replace("mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias")

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue

                param_name = name.replace(weight_name, "qkv_proj")
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.', param_name)
                    continue

                param = param_dict[param_name]
                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = rank // n_kv_heads_replicas
                else:
                    shard_id = rank

                loaded_weight = loaded_weight[shard_size * shard_id: shard_size * (shard_id + 1)]
                param_slice = param.data[offset:offset + shard_size]

                if param_slice.shape != loaded_weight.shape:
                    raise ValueError(f"{param_name} model shape({param_slice.shape}) mismatch"
                                     f" checkpoint shape({loaded_weight.shape})")

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break

            if is_attention_weight:
                continue

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
                gate_up_idx = moe_tp_id if ("experts" in name and "shared_experts" not in name) else rank

                loaded_weight = loaded_weight[shard_size * gate_up_idx:shard_size * (gate_up_idx + 1)]
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

            if self.args.qk_norm_full and "q_norm" in name:
                shard_size = param.shape[0]
                shard_id = rank
                loaded_weight = loaded_weight[shard_id * shard_size : (shard_id + 1) * shard_size]
                param.data.copy_(loaded_weight)
                continue

            if self.args.qk_norm_full and "k_norm" in name:
                shard_size = param.shape[0]
                shard_id = rank // n_kv_heads_replicas
                loaded_weight = loaded_weight[shard_id * shard_size : (shard_id + 1) * shard_size]
                param.data.copy_(loaded_weight)
                continue

            if "o_proj" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.head_dim * self.args.n_heads, self.args.dim,
                                             name, True, True, rank, world_size)
                continue

            if "down_proj" in name and not is_layer_moe(self.args, int(name.split(".")[2])):
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.inter_dim, self.args.dim,
                                             name, True, True, rank, world_size)
                continue

            if "down_proj" in name and "shared_experts" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.n_shared_experts * self.args.moe_inter_dim, self.args.dim,
                                             name, True, True, rank, world_size)
                continue

            if "down_proj" in name and "experts" in name:
                shard_size = param.shape[1]
                loaded_weight = loaded_weight[:, shard_size * moe_tp_id:shard_size * (moe_tp_id + 1)]
                param.data[:,:loaded_weight.shape[1]].copy_(loaded_weight)
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            param.copy_(loaded_weight)
            torch.npu.empty_cache()

        # transpose
        if forward_backend == "xlite":
            for layer_id, layer in enumerate(self.layers):
                if is_layer_moe(self.args, layer_id):
                    for i in range(layer.mlp.experts_start_idx, layer.mlp.experts_end_idx):
                        layer.mlp.experts[i].gate_up_proj.weight.data = layer.mlp.experts[i].gate_up_proj.weight.data.transpose(0,1).contiguous()
                        layer.mlp.experts[i].down_proj.weight.data = layer.mlp.experts[i].down_proj.weight.data.transpose(0,1).contiguous()

        if self.xlite_weight_nz:
            def pad_and_nd2nz(weight, name=""):
                rows, cols = weight.shape
                padded_rows = (rows + 15) // 16 * 16
                padded_cols = (cols + 15) // 16 * 16
                if rows != padded_rows or cols != padded_cols:
                    padded_weight = torch.zeros(padded_rows, padded_cols, dtype=weight.dtype, device=weight.device)
                    padded_weight[:rows, :cols] = weight
                    if debug and rank == 0:
                        print(f"[DEBUG] {name} padded from {weight.shape} to {padded_weight.shape}")
                    weight = padded_weight
                return matrix_nd2nz(weight)

            self.lm_head.weight.data = pad_and_nd2nz(self.lm_head.weight.data, "lm_head.weight")
            for layer_id, layer in enumerate(self.layers):
                layer.self_attn.qkv_proj.weight.data = pad_and_nd2nz(layer.self_attn.qkv_proj.weight.data, f"layer{layer_id}.qkv_proj.weight")
                layer.self_attn.o_proj.weight.data = pad_and_nd2nz(layer.self_attn.o_proj.weight.data, f"layer{layer_id}.o_proj.weight")
                if not is_layer_moe(self.args, layer_id):
                    layer.mlp.gate_up_proj.weight.data = pad_and_nd2nz(layer.mlp.gate_up_proj.weight.data, f"layer{layer_id}.gate_up_proj.weight")
                    layer.mlp.down_proj.weight.data = pad_and_nd2nz(layer.mlp.down_proj.weight.data, f"layer{layer_id}.down_proj.weight")
                else:
                    layer.mlp.gate.weight.data = pad_and_nd2nz(layer.mlp.gate.weight.data, f"layer{layer_id}.gate.weight")
                    for i in range(layer.mlp.experts_start_idx, layer.mlp.experts_end_idx):
                        layer.mlp.experts[i].gate_up_proj.weight.data = pad_and_nd2nz(layer.mlp.experts[i].gate_up_proj.weight.data, f"layer{layer_id}.experts[{i}].gate_up_proj.weight")
                        layer.mlp.experts[i].down_proj.weight.data = pad_and_nd2nz(layer.mlp.experts[i].down_proj.weight.data, f"layer{layer_id}.experts[{i}].down_proj.weight")
                    if self.args.n_shared_experts != 0:
                        layer.mlp.shared_experts.gate_up_proj.weight.data = pad_and_nd2nz(layer.mlp.shared_experts.gate_up_proj.weight.data, f"layer{layer_id}.shared_experts.gate_up_proj.weight")
                        layer.mlp.shared_experts.down_proj.weight.data = pad_and_nd2nz(layer.mlp.shared_experts.down_proj.weight.data, f"layer{layer_id}.shared_experts.down_proj.weight")
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
        config.rope_head_dim = int(args.head_dim * args.partial_rotary_factor)
        config.norm_eps = args.norm_eps
        config.rope_theta = args.rope_theta
        config.softmax_scale = args.head_dim ** -0.5
        config.n_dense_layers = args.first_k_dense_replace
        config.n_routed_experts = args.n_routed_experts
        config.n_shared_experts = args.n_shared_experts
        config.n_act_experts = args.n_activated_experts
        config.n_expert_groups = args.n_group
        config.n_limited_groups = args.topk_group
        config.intermediate_size = args.inter_dim
        config.moe_intermediate_size = args.moe_inter_dim
        config.def_tp_size = world_size
        config.def_dp_size = 1
        config.moe_ep_size = args.moe_ep_size
        config.moe_tp_size = args.moe_tp_size
        config.block_size = 128
        config.max_seq_len = args.max_seq_len
        config.max_batch_size = args.max_batch_size
        config.max_num_batched_tokens = args.max_num_batched_tokens
        config.attn_type = AttnMHA
        config.weight_nz = self.xlite_weight_nz
        config.qkv_bias = args.qkv_bias
        config.qk_norm = args.qk_norm
        config.qk_norm_full = args.qk_norm_full
        config.norm_topk_prob = args.norm_topk_prob
        config.scoring_func = ScoringFuncSigmoid
        config.route_scale = args.routed_scaling_factor
        config.experts_weight_transpose = True

        self.xlite_model = Model()
        self.xlite_model.embed = self.embed_tokens.weight
        self.xlite_model.norm = self.norm.weight
        self.xlite_model.head = self.lm_head.weight
        self.xlite_model.attn_norm = [layer.input_layernorm.weight for layer in self.layers]
        self.xlite_model.attn_out = [layer.self_attn.o_proj.weight for layer in self.layers]
        self.xlite_model.mha_qkv = [layer.self_attn.qkv_proj.weight for layer in self.layers]
        self.xlite_model.mlp_norm = [layer.post_attention_layernorm.weight for layer in self.layers]
        self.xlite_model.mlp_up_gate = [
            self.layers[i].mlp.gate_up_proj.weight
            for i in range(args.n_layers)
            if not is_layer_moe(self.args, i)
        ]
        self.xlite_model.mlp_down = [
            self.layers[i].mlp.down_proj.weight
            for i in range(args.n_layers)
            if not is_layer_moe(self.args, i)
        ]
        if args.qk_norm:
            self.xlite_model.mha_q_norm = [self.layers[i].self_attn.q_norm.weight for i in range(args.n_layers)]
            self.xlite_model.mha_k_norm = [self.layers[i].self_attn.k_norm.weight for i in range(args.n_layers)]
        if args.qkv_bias:
            self.xlite_model.mha_qkv_bias = [layer.self_attn.qkv_proj.bias for layer in self.layers]

        self.xlite_model.gate = [
            self.layers[i].mlp.gate.weight
            for i in range(args.n_layers)
            if is_layer_moe(self.args, i)
        ]
        self.xlite_model.gate_bias = [
            self.layers[i].mlp.gate.e_score_correction_bias.to(torch.float32)
            for i in range(args.n_layers)
            if is_layer_moe(self.args, i)
        ]
        self.xlite_model.re_up_gate = [
            self.layers[i].mlp.experts[j].gate_up_proj.weight
            for i in range(args.n_layers)
            if is_layer_moe(self.args, i)
            for j in range(self.layers[i].mlp.experts_start_idx, self.layers[i].mlp.experts_end_idx)
        ]
        self.xlite_model.re_down = [
            self.layers[i].mlp.experts[j].down_proj.weight
            for i in range(args.n_layers)
            if is_layer_moe(self.args, i)
            for j in range(self.layers[i].mlp.experts_start_idx, self.layers[i].mlp.experts_end_idx)
        ]
        if args.n_shared_experts != 0:
            self.xlite_model.se_up_gate = [
                self.layers[i].mlp.shared_experts.gate_up_proj.weight
                for i in range(args.n_layers)
                if is_layer_moe(self.args, i)
            ]
            self.xlite_model.se_down = [
                self.layers[i].mlp.shared_experts.down_proj.weight
                for i in range(args.n_layers)
                if is_layer_moe(self.args, i)
            ]
        self.xlite_model.init(config, rank)

    def init_xlite_kvcache(self, args: ModelArgs):
        block_num = (args.max_seq_len + block_size - 1) // block_size * args.max_batch_size
        head_num = max(args.n_kv_heads // world_size, 1)
        self.xlite_kv_cache = [(torch.zeros(block_num, block_size, head_num, args.head_dim, dtype=torch.get_default_dtype(), device='npu'),
                                torch.zeros(block_num, block_size, head_num, args.head_dim, dtype=torch.get_default_dtype(), device='npu'))
                               for _ in range(args.n_layers)]
        kv_size = (block_num * head_num * block_size * (args.head_dim + args.head_dim) *
                   self.xlite_kv_cache[0][0].element_size() * args.n_layers)
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
