#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu

from tests.models.deepseek_kernel import weight_dequant
from tests.models.weight_utils import (hf_model_weights_iterator,
                                       convert_pyslice_to_tensor,
                                       load_tensor_parallel_weights, logger)


debug = False
world_size = 1
rank = 0
block_size = 128
gemm_impl = "bf16"

forward_backend = os.getenv("FORWARD_BACKEND", "torch_npu")
if forward_backend == "xlite":
    xlite_rt = None
    xlite_model = None
    block_size = 64
    from xlite._C import Runtime, ModelConfig, ModelAttnMeta, AttnMLA, AttnDSA, Model, ScoringFuncSigmoid
    import numpy as np

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        max_num_batched_tokens (int): Maximum number of tokens.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of DeepSeek_V3 layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
        quantization (str): Quantization policy.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096
    max_num_batched_tokens: int = 4096
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-6
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    quantization: Literal["none", "experts_int8", "w8a8"] = "none"
    moe_ep_size: int = 1
    moe_tp_size: int = 1
    model_type: Literal["deepseek_v3", "deepseek_v32", "glm5"] = "deepseek_v3"
    # index: only used for deepseek_v32 & glm5
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    indexer_rope_interleave: bool = False

    def __post_init__(self):
        self.max_num_batched_tokens = self.max_seq_len * self.max_batch_size


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


def quantize_npu(
    x: torch.Tensor,
    input_scale_reciprocal: torch.Tensor,
    aclnn_input_offset: torch.Tensor,
) -> torch.Tensor:
    """NPU INT8 activation quantization: q = clamp(round(x / scale + offset), -128, 127)."""
    x_fp32 = x.float()
    scale_fp32 = input_scale_reciprocal.float()
    offset_fp32 = aclnn_input_offset.float()
    x_quant_fp32 = torch.round(x_fp32 / scale_fp32 + offset_fp32)
    x_quant_fp32 = torch.clamp(x_quant_fp32, -128.0, 127.0)
    return x_quant_fp32.to(torch.int8)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, is_row_parallel: bool = False) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), dequantization and a `bf16` GEMM operation are applied.
        - If `gemm_impl == "bf16"`, a `bf16` GEMM operation are applied.
        - If `weight` has static quantization parameters (input_scale, input_offset, quant_bias, deq_scale),
          uses `quantize_npu` + `torch_npu.npu_quant_matmul` for static quantization.
    """
    assert gemm_impl == "bf16"

    if hasattr(weight, 'input_scale') and weight.input_scale is not None:
        x_int8 = quantize_npu(x, weight.input_scale, weight.input_offset)
        weight_T = weight.T.contiguous()
        deq_scale = weight.scale.squeeze(-1).float()
        quant_bias_to_add = weight.quant_bias
        if is_row_parallel and rank != 0:
            quant_bias_to_add = None

        out = torch_npu.npu_quant_matmul(
            x_int8, weight_T,
            deq_scale,
            bias=quant_bias_to_add,
            output_dtype=torch.bfloat16
        )

        if bias is not None and quant_bias_to_add is None:
            out = out + bias.view(1, -1)
        return out

    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    else:
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
        static_quant (bool): Whether to use static quantization. Defaults to False.
            Only applicable when weight is int8 (element_size() == 1).
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, static_quant: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype), requires_grad=False)
        if self.weight.element_size() == 1:
            self.weight.scale = self.scale = nn.Parameter(torch.empty(out_features, 1, dtype=torch.float32))
            if forward_backend == "xlite":
                self.weight.xlite_scale = torch.zeros(out_features * 2, 1, dtype=torch.float32)

            if static_quant:
                self.weight.input_scale = self.input_scale = nn.Parameter(torch.empty(in_features, dtype=torch.bfloat16), requires_grad=False)
                self.weight.input_offset = self.input_offset = nn.Parameter(torch.empty(in_features, dtype=torch.bfloat16), requires_grad=False)
                self.weight.quant_bias = self.quant_bias = nn.Parameter(torch.empty(out_features, dtype=torch.int32), requires_grad=False)
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
        static_quant (bool): Whether to use static quantization. Defaults to False.
            Only applicable when weight is int8 (element_size() == 1).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, static_quant: bool = False):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype, static_quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias, False)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
        static_quant (bool): Whether to use static quantization. Defaults to False.
            Only applicable when weight is int8 (element_size() == 1).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, static_quant: bool = False):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype, static_quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight, None, True)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
        bias (bool): Whether to include a bias term. Defaults to False.
    """
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        if self.bias is None:
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return x * self.weight + self.bias


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate activation using Hadamard transform.
    Simplified implementation for compatibility when fast_hadamard_transform is not available.
    """
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    scale = hidden_size ** -0.5

    # Simple identity transformation as fallback
    # In production, this should use fast_hadamard_transform for better performance
    return x * scale


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        if args.quantization == "w8a8":
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim,
                               dtype=torch.int8, static_quant=True)
        else:
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk_weights_proj = Linear(self.dim, self.head_dim + self.n_heads, dtype=torch.float32)
        self.k_norm = LayerNorm(self.head_dim)
        self.softmax_scale = self.head_dim ** -0.5
        self.indexer_rope_interleave = args.indexer_rope_interleave

        if forward_backend != "xlite":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.get_default_dtype()), persistent=False)


    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, self.indexer_rope_interleave)
        q = torch.cat([q_pe, q_nope], dim=-1)
        wk_weights = self.wk_weights_proj(x.float())
        k = wk_weights[..., :self.head_dim].type_as(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, self.indexer_rope_interleave).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        self.k_cache[:bsz, start_pos:end_pos] = k
        weights = wk_weights[..., self.head_dim:] * self.n_heads ** -0.5
        scores = torch.einsum("bshd, btd -> bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        index_score = torch.einsum("bsht,bsh->bst", scores, weights)
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        return topk_indices


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # 根据 args.quantization 决定是否使用静态量化
        if args.quantization == "w8a8":
            # wqkv_a 合并 q_a_proj 和 kv_a_proj_with_mqa，静态量化
            self.wqkv_a = Linear(self.dim, self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                                 dtype=torch.int8, static_quant=True)
            # wq_b 是 ColumnParallelLinear，静态量化
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim,
                                              dtype=torch.int8, static_quant=True)
            # wo 是 RowParallelLinear，静态量化
            self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim,
                                         dtype=torch.int8, static_quant=True)
        else:
            self.wqkv_a = Linear(self.dim, self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
            self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps, bias=True if args.quantization == "w8a8" else False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.norm_eps, bias=True if args.quantization == "w8a8" else False)
        # wkv_b 保持 FLOAT（根据 quant_model_description.json）
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = None if args.model_type == "deepseek_v3" else Indexer(args)

        if forward_backend != "xlite":
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qkv_lora = self.wqkv_a(x)
        qr, kv_lora = torch.split(qkv_lora, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
        qr = self.q_norm(qr)
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv, k_pe = torch.split(kv_lora, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if mask is not None:    # MHA prefill
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(kv)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

            if self.indexer is None:
                scores += mask.unsqueeze(1)
            else:
                # indexer
                topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
                index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
                index_mask += mask
                scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:                   # MQA decode
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

            if self.indexer is not None:
                # indexer
                topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
                index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
                scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs = None):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        if args is not None and args.quantization == "w8a8":
            self.w13 = ColumnParallelLinear(dim, inter_dim * 2, dtype=torch.int8)
            self.w2 = RowParallelLinear(inter_dim, dim, dtype=torch.int8)
        else:
            self.w13 = ColumnParallelLinear(dim, inter_dim * 2)
            self.w2 = RowParallelLinear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        y = self.w13(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.w2(F.silu(y1) * y3)


class SharedExpertMLP(nn.Module):
    """
    MLP for shared experts without tensor parallelism.
    Each rank loads complete weights instead of sharding.
    """
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs = None):
        super().__init__()
        if args is not None and args.quantization == "w8a8":
            # w8a8: shared_experts 使用动态量化（W8A8_DYNAMIC）
            self.w13 = Linear(dim, inter_dim * 2, dtype=torch.int8)
            self.w2 = Linear(inter_dim, dim, dtype=torch.int8)
        else:
            self.w13 = Linear(dim, inter_dim * 2)
            self.w2 = Linear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w13(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.w2(F.silu(y1) * y3)


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
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
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 or self.dim == 6144 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[...,0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        if args.quantization in ("experts_int8", "w8a8"):
            self.w13 = Linear(dim, inter_dim * 2, dtype=torch.int8)
            self.w2 = Linear(inter_dim, dim, dtype=torch.int8)
        else:
            self.w13 = Linear(dim, inter_dim * 2)
            self.w2 = Linear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        y = self.w13(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.w2(F.silu(y1) * y3)


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
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, args) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = SharedExpertMLP(args.dim, args.n_shared_experts * args.moe_inter_dim, args)

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
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    DeepSeek_V3 block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the DeepSeek_V3 block.

        Args:
            layer_id (int): Layer index in the DeepSeek_V3.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim, args) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps, bias=True if args.quantization == "w8a8" else False)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, bias=True if args.quantization == "w8a8" else False)
        self.n_dense_layers = args.n_dense_layers
        self.layer_id = layer_id

    def forward_naive(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the DeepSeek_V3 block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        attn_norm_out = self.attn_norm(x)
        if debug and rank == 0 and (self.layer_id == 0 or self.layer_id == self.n_dense_layers):
            print(f"layer{self.layer_id} in: {attn_norm_out}")
        attn_out = self.attn(attn_norm_out, start_pos, freqs_cis, mask)
        if debug and rank == 0 and (self.layer_id == 0 or self.layer_id == self.n_dense_layers):
            print(f"layer{self.layer_id} after attn: {attn_out}")
        x = x + attn_out
        ffn_norm_out = self.ffn_norm(x)
        ffn_out = self.ffn(ffn_norm_out)
        if debug and rank == 0 and (self.layer_id == 0 or self.layer_id == self.n_dense_layers):
            print(f"layer{self.layer_id} after ffn: {ffn_out}")
        x = x + ffn_out
        return x

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the DeepSeek_V3 block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        return self.forward_naive(x, start_pos, freqs_cis, mask)

class DeepSeek_V3(nn.Module):
    """
    DeepSeek_V3 model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the DeepSeek_V3.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of DeepSeek_V3 blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the DeepSeek_V3 model.

        Args:
            args (ModelArgs): Model arguments containing DeepSeek_V3 parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, args.norm_eps, bias=True if args.quantization == "w8a8" else False)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward_naive(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the DeepSeek_V3 model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    @torch.inference_mode()
    def forward_xlite(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the DeepSeek_V3 model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        logits = torch.empty(world_size, tokens.size(0), self.args.vocab_size // world_size,
                             device=tokens.device)
        tokens = tokens.contiguous().view(tokens.size(0), tokens.size(1))
        batch = tokens.size(0)
        seqlen = tokens.size(1)
        logits_indices = torch.arange(batch, dtype=torch.int32, device=tokens.device) * seqlen + (seqlen - 1)
        attn_meta = self.prepare_xlite_attnmeta(tokens, start_pos)
        stream = torch.npu.current_stream().npu_stream
        h = torch.empty(tokens.numel(), self.args.dim, device=tokens.device)
        self.xlite_model.forward(self.xlite_rt, tokens.flatten(), attn_meta, self.xlite_kv_cache,
                                 self.freqs_cis, h, stream)
        self.xlite_model.forward_get_logits(self.xlite_rt, h, logits_indices, logits)
        logits = logits.permute(1, 0, 2).reshape(tokens.size(0), self.args.vocab_size)
        return logits

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the DeepSeek_V3 model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        if forward_backend == "xlite":
            return self.forward_xlite(tokens, start_pos)
        else:
            return self.forward_naive(tokens, start_pos)

    def load_weights(self, model_path: str):
        """ load deepseek v3 or v3.2 or glm5 weight """
        args = self.args
        assert args.dim % world_size == 0, f"dim must be divisible by world_size (world_size={world_size})"
        assert args.n_heads % world_size == 0, f"n_heads must be divisible by world_size (world_size={world_size})"
        assert args.inter_dim % world_size == 0, f"inter_dim must be divisible by world_size (world_size={world_size})"
        assert args.vocab_size % world_size == 0, f"vocab_size must be divisible by world_size (world_size={world_size})"

        static_quant_weights = ("q_a_proj", "kv_a_proj_with_mqa", "wq_b", "wo")

        n_local_experts = args.n_routed_experts // args.moe_ep_size
        moe_tp_id = rank % args.moe_tp_size
        moe_ep_id = rank // args.moe_tp_size
        param_dict = {name if "lm_head" in name else "model." + name: param for name, param in self.named_parameters()}
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, loaded_weight in hf_model_weights_iterator(model_path):
            if "rotary_emb.inv_freq" in name or "g_idx" in name:
                continue

            # skip mtp layer
            if name.startswith("model.layers."):
                layer_id = int(name.split(".")[2])
                if layer_id >= args.n_layers:
                    continue

            if "experts" in name and "shared_experts" not in name:
                idx = int(name.split(".")[-3])
                if idx < moe_ep_id * n_local_experts or idx >= (moe_ep_id + 1) * n_local_experts:
                    continue

            name = name.replace("self_attn", "attn")
            name = name.replace("mlp", "ffn")
            name = name.replace("weight_scale_inv", "scale")
            name = name.replace("e_score_correction_bias", "bias")
            name = name.replace("qweight", "weight")
            name = name.replace("scales", "scale")
            name = name.replace("down_proj", "w2")
            name = name.replace("embed_tokens", "embed")
            name = name.replace("input_layernorm", "attn_norm")
            name = name.replace("post_attention_layernorm", "ffn_norm")
            name = name.replace("q_a_layernorm", "q_norm")
            name = name.replace("kv_a_layernorm", "kv_norm")
            name = name.replace("kv_b_proj", "wkv_b")
            name = name.replace("lm_head", "head")
            name = name.replace("q_b_proj", "wq_b")
            name = name.replace("o_proj", "wo")

            if ".weight_offset" in name:
                continue
            if not name.startswith("model."):
                name = "model." + name

            if args.quantization == "w8a8" and "weight_scale" in name or ".bias" in name:
                if any(s in name for s in static_quant_weights):
                    continue
            name = name.replace("weight_scale", "scale")

            if "deq_scale" in name:
                if any(s in name for s in static_quant_weights):
                    name = name.replace("deq_scale", "scale")

            is_wqkv_a = False
            for stride_id, weight_name in enumerate(["q_a_proj", "kv_a_proj_with_mqa"]):
                if weight_name not in name:
                    continue

                param_name = name.replace(weight_name, "wqkv_a")
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.',
                                   param_name)
                    continue
                param = param_dict[param_name]
                if ".scale" in name:
                    if stride_id == 0:
                        param.data[:args.q_lora_rank, 0].copy_(loaded_weight[:])
                    else:  # kv_a_proj_with_mqa
                        param.data[args.q_lora_rank:, 0].copy_(loaded_weight[:])
                else:
                    if stride_id == 0:
                        param.data[:args.q_lora_rank].copy_(loaded_weight[:])
                    else:
                        param.data[args.q_lora_rank:].copy_(loaded_weight[:])

                is_wqkv_a = True
                break
            if is_wqkv_a:
                continue

            # Handle indexer wk and weights_proj merge into wk_weights_proj
            is_indexer_wk_weights = False
            for stride_id, weight_name in enumerate(["indexer.wk.weight", "indexer.weights_proj.weight"]):
                if weight_name not in name:
                    continue

                param_name = name.replace(weight_name, "indexer.wk_weights_proj.weight")
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.',
                                   param_name)
                    continue
                param = param_dict[param_name]
                if stride_id == 0:  # wk.weight -> head_dim
                    param.data[:args.index_head_dim].copy_(loaded_weight[:])
                else:  # weights_proj.weight -> n_heads
                    param.data[args.index_head_dim:].copy_(loaded_weight[:])

                is_indexer_wk_weights = True
                break
            if is_indexer_wk_weights:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue

                param_name = name.replace(weight_name, "w13")
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.',
                                   param_name)
                    continue
                param = param_dict[param_name]
                shard_size = param.shape[0] // 2
                # shared_experts 不切分，每个 rank 加载完整权重
                if "shared_experts" in name:
                    param_slice = param.data[shard_size * stride_id:shard_size * (stride_id + 1)]
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param_slice.data.copy_(loaded_weight)
                else:
                    gate_up_idx = moe_tp_id if "experts" in name else rank
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

            if any(s in name for s in static_quant_weights):
                if ".input_scale" in name or ".input_offset" in name:
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data.copy_(loaded_weight.expand(param.shape))
                    continue

            if "embed" in name:
                load_tensor_parallel_weights(param, loaded_weight, args.vocab_size, args.dim, name,
                                             True, False, rank, world_size)
                continue

            if "head" in name:
                load_tensor_parallel_weights(param, loaded_weight, args.dim, args.vocab_size, name,
                                             False, True, rank, world_size)
                continue

            if "wq_b" in name:
                if ".quant_bias" in name or ".scale" in name:
                    shard_size = param.shape[0]
                    loaded_weight_slice = loaded_weight[rank * shard_size:(rank + 1) * shard_size]
                    loaded_weight_slice = convert_pyslice_to_tensor(loaded_weight_slice)
                    if ".scale" in name:
                        param.data[:loaded_weight_slice.shape[0], 0].copy_(loaded_weight_slice)
                    else:
                        param.data[:loaded_weight_slice.shape[0]].copy_(loaded_weight_slice)
                    continue

            if "wq_b" in name and "indexer" not in name:
                load_tensor_parallel_weights(param, loaded_weight, args.q_lora_rank,
                                             args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim), name,
                                             False, True, rank, world_size)
                continue

            if "wkv_b" in name:
                load_tensor_parallel_weights(param, loaded_weight, args.kv_lora_rank,
                                             args.n_heads * (args.qk_nope_head_dim + args.v_head_dim),
                                             name, False, True, rank, world_size)
                continue

            if "wo" in name:
                # quant_bias: shape [N]
                if ".quant_bias" in name:
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data.copy_(loaded_weight)
                    continue

                # scale: shape [N, 1]
                if ".scale" in name:
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.data[:, 0].copy_(loaded_weight)
                    continue

                load_tensor_parallel_weights(param, loaded_weight, args.v_head_dim * args.n_heads,
                                             args.dim, name, True, True, rank, world_size)
                continue

            if "w2" in name and (int(name.split(".")[2]) < args.n_dense_layers):
                if ".scale" in name:
                    loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                    param.copy_(loaded_weight)
                else:
                    load_tensor_parallel_weights(param, loaded_weight, args.inter_dim, args.dim, name,
                                                True, True, rank, world_size)
                continue

            if "w2" in name and "shared_experts" in name:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                param.copy_(loaded_weight)
                continue

            if "w2" in name and "experts" in name:
                shard_size = param.shape[1]
                loaded_weight = loaded_weight[:, shard_size * moe_tp_id:shard_size * (moe_tp_id + 1)]
                param.data[:,:loaded_weight.shape[1]].copy_(loaded_weight)
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            param.copy_(loaded_weight)

        torch.npu.empty_cache()

        if forward_backend == "xlite":
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.xlite_rt = Runtime(local_rank, 0, rank, world_size)
            self.init_xlite_model(args)
            kv_size = self.init_xlite_kvcache(args)
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
        config.n_kv_heads = 1
        config.nope_head_dim = args.qk_nope_head_dim
        config.rope_head_dim = args.qk_rope_head_dim
        config.v_head_dim = args.v_head_dim
        config.q_lora_rank = args.q_lora_rank
        config.kv_lora_rank = args.kv_lora_rank
        config.norm_eps = args.norm_eps
        config.rope_theta = args.rope_theta
        config.softmax_scale = self.layers[0].attn.softmax_scale
        config.n_dense_layers = args.n_dense_layers
        config.n_routed_experts = args.n_routed_experts
        config.n_shared_experts = args.n_shared_experts
        config.n_expert_groups = args.n_expert_groups
        config.n_limited_groups = args.n_limited_groups
        config.n_act_experts = args.n_activated_experts
        config.intermediate_size = args.inter_dim
        config.moe_intermediate_size = args.moe_inter_dim
        config.route_scale = args.route_scale
        config.def_tp_size = world_size
        config.def_dp_size = 1
        config.moe_ep_size = args.moe_ep_size
        config.moe_tp_size = args.moe_tp_size
        config.block_size = block_size
        config.max_seq_len = args.max_seq_len
        config.max_batch_size = args.max_batch_size
        config.max_num_batched_tokens = args.max_num_batched_tokens
        config.attn_type = AttnMLA

        if args.score_func == "sigmoid":
            config.scoring_func = ScoringFuncSigmoid
            config.norm_topk_prob = True

        if args.model_type != "deepseek_v3":
            config.index_head_dim = args.index_head_dim
            config.index_n_heads = args.index_n_heads
            config.index_topk = args.index_topk
            config.index_softmax_scale = self.layers[0].attn.indexer.softmax_scale
            config.index_rope_interleaved = args.indexer_rope_interleave
            config.attn_type = AttnDSA

        global xlite_model
        xlite_model = self.xlite_model = Model()
        self.xlite_model.embed = self.embed.weight
        self.xlite_model.norm = self.norm.weight
        self.xlite_model.head = self.head.weight
        self.xlite_model.attn_norm = [layer.attn_norm.weight for layer in self.layers]
        self.xlite_model.attn_out = [layer.attn.wo.weight for layer in self.layers]
        self.xlite_model.mla_qkv_a = [layer.attn.wqkv_a.weight for layer in self.layers]
        self.xlite_model.mla_q_b = [layer.attn.wq_b.weight for layer in self.layers]
        self.xlite_model.mla_q_norm = [layer.attn.q_norm.weight for layer in self.layers]
        self.xlite_model.mla_kv_b = [layer.attn.wkv_b.weight for layer in self.layers]
        self.xlite_model.mla_kv_norm = [layer.attn.kv_norm.weight for layer in self.layers]
        self.xlite_model.index_q_b = [layer.attn.indexer.wq_b.weight for layer in self.layers if layer.attn.indexer is not None]
        self.xlite_model.index_k_weights_proj = [layer.attn.indexer.wk_weights_proj.weight.to(dtype=torch.get_default_dtype()) for layer in self.layers if layer.attn.indexer is not None]
        self.xlite_model.index_k_norm = [layer.attn.indexer.k_norm.weight.to(dtype=torch.get_default_dtype()) for layer in self.layers if layer.attn.indexer is not None]
        self.xlite_model.index_k_norm_bias = [layer.attn.indexer.k_norm.bias.to(dtype=torch.get_default_dtype()) for layer in self.layers if layer.attn.indexer is not None]
        self.xlite_model.mlp_norm = [layer.ffn_norm.weight for layer in self.layers]
        self.xlite_model.mlp_up_gate = [self.layers[i].ffn.w13.weight for i in range(args.n_dense_layers)]
        self.xlite_model.mlp_down = [self.layers[i].ffn.w2.weight for i in range(args.n_dense_layers)]
        self.xlite_model.gate = [self.layers[i].ffn.gate.weight for i in range(args.n_dense_layers, args.n_layers)]
        self.xlite_model.gate_bias = [self.layers[i].ffn.gate.bias for i in range(args.n_dense_layers, args.n_layers)]
        self.xlite_model.se_up_gate = [self.layers[i].ffn.shared_experts.w13.weight
                                       for i in range(args.n_dense_layers, args.n_layers)]
        self.xlite_model.se_down = [self.layers[i].ffn.shared_experts.w2.weight
                                    for i in range(args.n_dense_layers, args.n_layers)]
        self.xlite_model.re_up_gate = [self.layers[i].ffn.experts[j].w13.weight
                                       for i in range(args.n_dense_layers, args.n_layers)
                                       for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx)]
        self.xlite_model.re_down = [self.layers[i].ffn.experts[j].w2.weight
                                    for i in range(args.n_dense_layers, args.n_layers)
                                    for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx)]
        if args.quantization == "experts_int8":
            for i in range(self.args.n_dense_layers, self.args.n_layers):
                for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx):
                    self.layers[i].ffn.experts[j].w13.weight.xlite_scale[0::2] = self.layers[i].ffn.experts[j].w13.weight.scale[0::1]
                    self.layers[i].ffn.experts[j].w2.weight.xlite_scale[0::2] = self.layers[i].ffn.experts[j].w2.weight.scale[0::1]
            self.xlite_model.re_up_gate_deq_scale = [self.layers[i].ffn.experts[j].w13.weight.xlite_scale
                                                     for i in range(args.n_dense_layers, args.n_layers)
                                                     for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx)]
            self.xlite_model.re_down_deq_scale = [self.layers[i].ffn.experts[j].w2.weight.xlite_scale
                                                  for i in range(args.n_dense_layers, args.n_layers)
                                                  for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx)]
        elif args.quantization == "w8a8":
            for i in range(self.args.n_layers):
                if self.layers[i].attn.indexer is not None:
                    self.layers[i].attn.indexer.wq_b.weight.xlite_scale[0::2] = self.layers[i].attn.indexer.wq_b.weight.scale[0::1]
                self.layers[i].attn.wqkv_a.weight.xlite_scale[0::2] = self.layers[i].attn.wqkv_a.weight.scale[0::1]
                self.layers[i].attn.wq_b.weight.xlite_scale[0::2] = self.layers[i].attn.wq_b.weight.scale[0::1]
                self.layers[i].attn.wo.weight.xlite_scale[0::2] = self.layers[i].attn.wo.weight.scale[0::1]
            self.xlite_model.index_q_b_input_scale = [layer.attn.indexer.wq_b.weight.input_scale.reciprocal() for layer in self.layers if layer.attn.indexer is not None]
            self.xlite_model.index_q_b_input_offset = [layer.attn.indexer.wq_b.weight.input_offset for layer in self.layers if layer.attn.indexer is not None]
            self.xlite_model.index_q_b_quant_bias = [layer.attn.indexer.wq_b.weight.quant_bias for layer in self.layers if layer.attn.indexer is not None]
            self.xlite_model.index_q_b_deq_scale = [layer.attn.indexer.wq_b.weight.xlite_scale for layer in self.layers if layer.attn.indexer is not None]
            self.xlite_model.mla_qkv_a_input_scale = [layer.attn.wqkv_a.weight.input_scale.reciprocal() for layer in self.layers]
            self.xlite_model.mla_qkv_a_input_offset = [layer.attn.wqkv_a.weight.input_offset for layer in self.layers]
            self.xlite_model.mla_qkv_a_quant_bias = [layer.attn.wqkv_a.weight.quant_bias for layer in self.layers]
            self.xlite_model.mla_qkv_a_deq_scale = [layer.attn.wqkv_a.weight.xlite_scale for layer in self.layers]
            self.xlite_model.mla_q_b_input_scale = [layer.attn.wq_b.weight.input_scale.reciprocal() for layer in self.layers]
            self.xlite_model.mla_q_b_input_offset = [layer.attn.wq_b.weight.input_offset for layer in self.layers]
            self.xlite_model.mla_q_b_quant_bias = [layer.attn.wq_b.weight.quant_bias for layer in self.layers]
            self.xlite_model.mla_q_b_deq_scale = [layer.attn.wq_b.weight.xlite_scale for layer in self.layers]
            self.xlite_model.mla_q_norm_bias = [layer.attn.q_norm.bias for layer in self.layers]
            self.xlite_model.mla_kv_norm_bias = [layer.attn.kv_norm.bias for layer in self.layers]
            self.xlite_model.attn_out_input_scale = [layer.attn.wo.weight.input_scale.reciprocal() for layer in self.layers]
            self.xlite_model.attn_out_input_offset = [layer.attn.wo.weight.input_offset for layer in self.layers]
            self.xlite_model.attn_out_quant_bias = [layer.attn.wo.weight.quant_bias for layer in self.layers]
            self.xlite_model.attn_out_deq_scale = [layer.attn.wo.weight.xlite_scale for layer in self.layers]
            self.xlite_model.norm_bias = self.norm.bias
            self.xlite_model.attn_norm_bias = [layer.attn_norm.bias for layer in self.layers]
            self.xlite_model.mlp_norm_bias = [layer.ffn_norm.bias for layer in self.layers]
            for i in range(self.args.n_dense_layers):
                self.layers[i].ffn.w13.weight.xlite_scale[0::2] = self.layers[i].ffn.w13.weight.scale[0::1]
                self.layers[i].ffn.w2.weight.xlite_scale[0::2] = self.layers[i].ffn.w2.weight.scale[0::1]
            self.xlite_model.mlp_up_gate_deq_scale = [self.layers[i].ffn.w13.weight.xlite_scale
                                                  for i in range(args.n_dense_layers)]
            self.xlite_model.mlp_down_deq_scale = [self.layers[i].ffn.w2.weight.xlite_scale for i in range(args.n_dense_layers)]

            for i in range(self.args.n_dense_layers, self.args.n_layers):
                self.layers[i].ffn.shared_experts.w13.weight.xlite_scale[0::2] = self.layers[i].ffn.shared_experts.w13.weight.scale[0::1]
                self.layers[i].ffn.shared_experts.w2.weight.xlite_scale[0::2] = self.layers[i].ffn.shared_experts.w2.weight.scale[0::1]
                for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx):
                    self.layers[i].ffn.experts[j].w13.weight.xlite_scale[0::2] = self.layers[i].ffn.experts[j].w13.weight.scale[0::1]
                    self.layers[i].ffn.experts[j].w2.weight.xlite_scale[0::2] = self.layers[i].ffn.experts[j].w2.weight.scale[0::1]
            self.xlite_model.se_up_gate_deq_scale = [self.layers[i].ffn.shared_experts.w13.weight.xlite_scale
                                                 for i in range(args.n_dense_layers, args.n_layers)]
            self.xlite_model.se_down_deq_scale = [self.layers[i].ffn.shared_experts.w2.weight.xlite_scale
                                                 for i in range(args.n_dense_layers, args.n_layers)]
            self.xlite_model.re_up_gate_deq_scale = [self.layers[i].ffn.experts[j].w13.weight.xlite_scale
                                                 for i in range(args.n_dense_layers, args.n_layers)
                                                 for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx)]
            self.xlite_model.re_down_deq_scale = [self.layers[i].ffn.experts[j].w2.weight.xlite_scale
                                              for i in range(args.n_dense_layers, args.n_layers)
                                              for j in range(self.layers[i].ffn.experts_start_idx, self.layers[i].ffn.experts_end_idx)]
        self.xlite_model.init(config, rank)

    def init_xlite_kvcache(self, args: ModelArgs):
        block_num = (args.max_seq_len + block_size - 1) // block_size * args.max_batch_size
        head_num = 1
        if args.model_type == "deepseek_v3":
            self.xlite_kv_cache = [(torch.zeros(block_num, block_size, head_num, args.kv_lora_rank, dtype=torch.get_default_dtype(), device='npu'),
                                    torch.zeros(block_num, block_size, head_num, args.qk_rope_head_dim, dtype=torch.get_default_dtype(), device='npu'))
                                for _ in range(args.n_layers)]
            kv_size = (block_num * head_num * block_size * (args.kv_lora_rank + args.qk_rope_head_dim) *
                    self.xlite_kv_cache[0][0].element_size() * args.n_layers)
        else:
            self.xlite_kv_cache = [(torch.zeros(block_num, block_size, head_num, args.kv_lora_rank, dtype=torch.get_default_dtype(), device='npu'),
                                    torch.zeros(block_num, block_size, head_num, args.qk_rope_head_dim, dtype=torch.get_default_dtype(), device='npu'),
                                    torch.zeros(block_num, block_size, head_num, args.index_head_dim, dtype=torch.get_default_dtype(), device='npu'))
                                for _ in range(args.n_layers)]
            kv_size = (block_num * head_num * block_size * (args.kv_lora_rank + args.qk_rope_head_dim + args.index_head_dim) *
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
        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(block_num, dtype=np.uint32)
        attn_meta.block_tables = batch_indices * step + block_indices
        return attn_meta


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = DeepSeek_V3(args)
    print(model(x).size())
