"""Type stubs for :mod:`xlite._C`.

This file mirrors the symbols exported by `csrc/_C.cpp`.
The typing and docstrings are designed for Python 3.9 to 3.12.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence, Union

import torch

class Runtime:
    """Ascend runtime handle for streams, communication, and tensor pools.

    Attributes:
        task_id (int): Current task slot used by multi-task parallel execution.
        notify (object): Runtime notify handle used for cross-stream synchronization.
        peer_notify (object): Peer notify handle used by the peer-stream sync path.
        multi_task_parallel (bool): Enables the dual-task scheduling path.
    """

    task_id: int = ...
    """Current task slot used by multi-task parallel execution."""
    notify: object = ...
    """Runtime notify handle used for cross-stream synchronization."""
    peer_notify: object = ...
    """Peer notify handle used by the peer-stream sync path."""
    multi_task_parallel: bool = ...
    """Enables the dual-task scheduling path."""

    def __init__(
        self,
        devid: int,
        size: int = 0,
        rank: int = 0,
        tp_size: int = 1,
        dp_size: int = 1,
        moe_tp_size: int = 1,
        moe_ep_size: int = 1,
    ) -> None:
        """Create a runtime for one Ascend device.

        Args:
            devid (int): Device ID passed to the ACL runtime.
            size (int): Optional tensor-pool size in MB.
            rank (int): Global rank in the distributed group.
            tp_size (int): Tensor-parallel group size.
            dp_size (int): Data-parallel group size.
            moe_tp_size (int): Tensor-parallel size for MoE layers.
            moe_ep_size (int): Expert-parallel size for MoE layers.
        """

    def update_core_num(self, util: float) -> None:
        """Scale active AI/vector core counts by utilization.

        Args:
            util (float): Utilization ratio, typically in `[0, 1]`.

        Returns:
            None: The runtime updates rounded AI/vector core counts.
        """

    def init_tensor_pool(self, size: int) -> int:
        """Initialize the tensor pool for this runtime.

        Args:
            size (int): Requested pool size in megabytes.

        Returns:
            int: `0` on success.

        Note:
            Passing `0` leaves the current pool unchanged.
        """

    def set_current_context(self) -> None:
        """Set the runtime ACL context as current.

        Returns:
            None: Context is updated on the calling thread.
        """

    def configure_swizzle(self, swizzle: int, use_swizzle_table: bool) -> None:
        """Configure swizzle parameters for matrix multiplication

        Args:
            swizzle(int): new default swizzle valuey.
            use_swizzle_table(bool): Whether to use precomputed swizzle values from xlite.

        Returns:
            None: The runtime updates swizzle configuration.
        """

class ModelConfig:
    """Model hyperparameters and layout used by :class:`Model`.

    Attributes:
        vocab_size (int): Vocabulary size for embedding and LM head.
        hidden_size (int): Transformer hidden width.
        n_layers (int): Total number of transformer layers.
        attn_type (AttnType): Attention family.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of key/value heads.
        head_dim (int): Per-head dimension for MHA.
        nope_head_dim (int): Non-RoPE query dimension for MLA/DSA.
        rope_head_dim (int): RoPE query/value dimension.
        v_head_dim (int): Value projection dimension.
        q_lora_rank (int): LoRA rank for query projection.
        kv_lora_rank (int): LoRA rank for key/value projection.
        quant_attn_weight_transpose (bool): Whether quanted attention weights are transposed.
        quant_attn_weight_nz (bool): Whether quanted attention weights are in NZ layout.
        norm_eps (float): RMSNorm/LayerNorm epsilon.
        rope_theta (float): Rotary base frequency.
        softmax_scale (float): Attention softmax scale.
        n_dense_layers (int): Number of dense FFN layers.
        n_routed_experts (int): Number of routed experts.
        n_shared_experts (int): Number of shared experts.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited routing groups.
        n_act_experts (int): Number of active experts.
        intermediate_size (int): Dense FFN intermediate width.
        moe_intermediate_size (int): MoE FFN intermediate width.
        route_scale (float): Routing score scale.
        def_tp_size (int): Default tensor-parallel size.
        def_dp_size (int): Default data-parallel size.
        moe_ep_size (int): Expert-parallel size.
        moe_tp_size (int): tensor-parallel size in MoE.
        max_seq_len (int): Maximum sequence length.
        max_batch_size (int): Maximum batch size.
        max_m (int): Maximum token count batched.
        max_num_batched_tokens (int): Maximum token count batched.
        block_size (int): KV block size.(deprecated)
        block_sizes (List[int]): Per-attention-type KV block sizes. Single-element for
            non-CXA; for CXA one entry per 5-tuple cache in order
            ``(indexer_state, indexer_k, compress_kv, state, swa_kv)``. Seeded from
            ``block_size`` when empty at construction.
        weight_nz (bool): Whether weights are in NZ layout.
        experts_weight_transpose (bool): Whether expert weights are transposed.
        experts_weight_nz (bool): Whether expert weights are in NZ layout.
        gate_captured(bool): Whether gate layer is captured by vllm-ascend.
        qkv_bias (bool): Whether MHA QKV has bias.
        qk_norm (bool): Whether MHA applies Q/K norm.
        qk_norm_full (bool): Whether MHA applies Q/K norm full.
        attn_output_gate (bool): Whether full MHA applies sigmoid output gate (Qwen3.5).
            When true, fused mha_qkv layout is [Q | K | V | Gate].
        scoring_func (ScoringFuncType): MoE scoring function.
        norm_topk_prob (bool): Whether top-k probabilities are normalized.
        mrope_section (List[int]): mRoPE section layout values.
        mrope_interleaved (bool): Whether mRoPE layout is interleaved.
        deepstack_num_level (int): Number of deepstack levels.
        index_head_dim (int): Indexer head dimension.
        index_n_heads (int): Indexer head count.
        index_topk (int): Indexer top-k size.
        index_softmax_scale (float): Indexer softmax scale.(deprecated)
        index_rope_interleaved (bool): Whether indexer RoPE is interleaved.
        o_groups (int): Output projection group count (DeepSeek-V4).
        o_lora_rank (int): Output projection LoRA rank (DeepSeek-V4).
        window_size (int): Sliding window attention size (DeepSeek-V4).
        compress_rope_theta (float): KV compressor RoPE base (DeepSeek-V4).
        original_seq_len (int): YaRN original sequence length (DeepSeek-V4).
        rope_factor (float): YaRN extension factor (DeepSeek-V4).
        beta_fast (int): YaRN beta_fast (DeepSeek-V4).
        beta_slow (int): YaRN beta_slow (DeepSeek-V4).
        hc_mult (int): Multi-stage Hyper-Connections multiplier (DeepSeek-V4).
        hc_sinkhorn_iters (int): MHC Sinkhorn iterations (DeepSeek-V4).
        hc_eps (float): MHC Sinkhorn epsilon (DeepSeek-V4).
        swiglu_limit (float): SwiGLU clamp limit; 0 disables (DeepSeek-V4).
        n_hash_layers (int): Number of hash-routed MoE layers (DeepSeek-V4).
        compress_ratios (List[int]): Per-layer KV compression ratios (DeepSeek-V4).
    """

    vocab_size: int = ...
    """Vocabulary size for embedding and LM head."""
    hidden_size: int = ...
    """Transformer hidden width."""
    n_layers: int = ...
    """Total number of transformer layers."""
    attn_type: AttnType = ...
    """Attention family."""
    n_heads: int = ...
    """Number of attention heads."""
    n_kv_heads: int = ...
    """Number of key/value heads."""
    head_dim: int = ...
    """Per-head dimension for MHA."""
    nope_head_dim: int = ...
    """Non-RoPE query dimension for MLA/DSA."""
    rope_head_dim: int = ...
    """RoPE query/value dimension."""
    v_head_dim: int = ...
    """Value projection dimension."""
    q_lora_rank: int = ...
    """LoRA rank for query projection."""
    kv_lora_rank: int = ...
    """LoRA rank for key/value projection."""
    quant_attn_weight_transpose: bool = ...
    """Whether quanted attention weights are transposed."""
    quant_attn_weight_nz: bool = ...
    """Whether quanted attention weights are in NZ layout."""
    norm_eps: float = ...
    """RMSNorm/LayerNorm epsilon."""
    rope_theta: float = ...
    """Rotary base frequency."""
    softmax_scale: float = ...
    """Attention softmax scale."""
    n_dense_layers: int = ...
    """Number of dense FFN layers."""
    n_routed_experts: int = ...
    """Number of routed experts."""
    n_shared_experts: int = ...
    """Number of shared experts."""
    n_expert_groups: int = ...
    """Number of expert groups."""
    n_limited_groups: int = ...
    """Number of limited routing groups."""
    n_act_experts: int = ...
    """Number of active experts."""
    intermediate_size: int = ...
    """Dense FFN intermediate width."""
    moe_intermediate_size: int = ...
    """MoE FFN intermediate width."""
    route_scale: float = ...
    """Routing score scale."""
    def_tp_size: int = ...
    """Default tensor-parallel size."""
    def_dp_size: int = ...
    """Default data-parallel size."""
    moe_ep_size: int = ...
    """Expert-parallel size."""
    moe_tp_size: int = ...
    """Tensor-parallel size in MoE."""
    max_seq_len: int = ...
    """Maximum sequence length."""
    max_batch_size: int = ...
    """Maximum batch size."""
    max_m: int = ...
    """Maximum token count batched."""
    max_num_batched_tokens: int = ...
    """Maximum token count batched."""
    block_size: int = ...
    """KV block size.(deprecated)"""
    block_sizes: List[int] = ...
    """Per-attention-type KV block sizes. Single-element for non-CXA; for CXA one entry
    per 5-tuple cache in order (indexer_state, indexer_k, compress_kv, state, swa_kv).
    Seeded from ``block_size`` when empty at construction."""
    weight_nz: bool = ...
    """Whether weights are in NZ layout."""
    experts_weight_transpose: bool = ...
    """Whether expert weights are transposed."""
    experts_weight_nz: bool = ...
    """Whether expert weights are in NZ layout."""
    gate_captured: bool = ...
    """Whether gate layer is captured by vllm-ascend."""
    qkv_bias: bool = ...
    """Whether MHA QKV has bias."""
    qk_norm: bool = ...
    """Whether MHA applies Q/K norm."""
    qk_norm_full: bool = ...
    """Whether MHA applies Q/K norm full."""
    attn_output_gate: bool = ...
    """Whether full MHA applies sigmoid output gate (Qwen3.5)."""
    linear_num_k_heads: int = ...
    """Number of key heads for linear attention layers."""
    linear_num_v_heads: int = ...
    """Number of value heads for linear attention layers."""
    linear_key_head_dim: int = ...
    """Key head dim for linear attention layers."""
    linear_value_head_dim: int = ...
    """Value head dim for linear attention layers."""
    linear_conv_kernel_dim: int = ...
    """Causal conv1d kernel size for linear attention layers."""
    full_attention_interval: int = ...
    """Interval between full-attention layers in hybrid models."""
    scoring_func: ScoringFuncType = ...
    """MoE scoring function."""
    norm_topk_prob: bool = ...
    """Whether top-k probabilities are normalized."""
    mrope_section: List[int] = ...
    """mRoPE section layout values."""
    mrope_interleaved: bool = ...
    """Whether mRoPE layout is interleaved."""
    deepstack_num_level: int = ...
    """Number of deepstack levels."""
    index_head_dim: int = ...
    """Indexer head dimension."""
    index_n_heads: int = ...
    """Indexer head count."""
    index_topk: int = ...
    """Indexer top-k size."""
    index_softmax_scale: float = ...
    """Indexer softmax scale.(deprecated)"""
    index_rope_interleaved: bool = ...
    """Whether indexer RoPE is interleaved."""
    index_full_mask: List[bool] = ...
    """Per-layer indexer mask. ``True``: full indexer layer; ``False``: shared indexer layer. Pass as an empty list to
    enable all full indexer layers; otherwise, the list length must match :data:`n_layers`."""
    o_groups: int = ...
    """Output projection group count (DeepSeek-V4)."""
    o_lora_rank: int = ...
    """Output projection LoRA rank (DeepSeek-V4)."""
    window_size: int = ...
    """Sliding window attention size (DeepSeek-V4)."""
    compress_rope_theta: float = ...
    """KV compressor RoPE base (DeepSeek-V4)."""
    original_seq_len: int = ...
    """YaRN original sequence length (DeepSeek-V4)."""
    rope_factor: float = ...
    """YaRN extension factor (DeepSeek-V4)."""
    beta_fast: int = ...
    """YaRN beta_fast (DeepSeek-V4)."""
    beta_slow: int = ...
    """YaRN beta_slow (DeepSeek-V4)."""
    hc_mult: int = ...
    """Multi-stage Hyper-Connections multiplier (DeepSeek-V4)."""
    hc_sinkhorn_iters: int = ...
    """MHC Sinkhorn iterations (DeepSeek-V4)."""
    hc_eps: float = ...
    """MHC Sinkhorn epsilon (DeepSeek-V4)."""
    swiglu_limit: float = ...
    """SwiGLU clamp limit; 0 disables (DeepSeek-V4)."""
    n_hash_layers: int = ...
    """Number of hash-routed MoE layers (DeepSeek-V4)."""
    compress_ratios: List[int] = ...
    """Per-layer KV compression ratios (DeepSeek-V4)."""

class AttnMeta:
    """Attention metadata for the native runtime forward path.

    This path reuses host block-table lists while taking `positions` directly
    from the provided tensor for attention position indexing.

    Attributes:
        lens (List[int]): Per-sample query lengths.
        cached_lens (List[int]): Per-sample cached lengths.
        block_tables_cpu (List[List[int]]): Per-sample block tables on host.
        positions (torch.Tensor): Position tensor for version-1 attention metadata,
            shape ``[batched_tokens]`` int64 device.
    """

    lens: List[int] = ...
    """Per-sample query lengths."""
    cached_lens: List[int] = ...
    """Per-sample cached lengths."""
    block_tables_cpu: List[List[int]] = ...
    """Per-sample block tables on host."""
    positions: torch.Tensor = ...
    """Position tensor for version-1 attention metadata, shape ``[batched_tokens]`` int64 device."""


class AttnMetaV2:
    """Device-tensor attention metadata for the V2 forward path.

    Unlike :class:`AttnMeta` (V1), this variant carries ``lens``,
    ``cached_lens``, ``query_start_loc``, ``slot_mapping`` and
    ``block_tables`` as pre-built device tensors. The C++ side skips host
    computation and H2D copies, only shape-checking and aliasing these tensors
    (zero-copy). Per-sample block tables are flattened into a 1D tensor padded
    to ``max_num_blocks``. The host ``lens_cpu``/``cached_lens_cpu`` lists are
    still required for C++-side tile-size selection (``GetTileSizeOfCachedKV``).

    Attributes:
        lens (torch.Tensor): Per-sample query lengths, shape ``[batch]`` int32 device.
        cached_lens (torch.Tensor): Per-sample cached lengths, shape ``[batch]`` int32 device.
        positions (torch.Tensor): Position tensor, shape ``[batched_tokens]`` int64.
        lens_cpu (List[int]): Per-sample query lengths (host, for tile-size selection).
        cached_lens_cpu (List[int]): Per-sample cached lengths (host, for tile-size selection).
        query_start_loc (torch.Tensor): Prefix-sum of lens, shape ``[batch]`` int32.
        slot_mapping (Sequence[torch.Tensor]): Slot indices, each shape
            ``[batched_tokens]`` int32 device, one per kv cache.
        block_tables (Sequence[torch.Tensor]): Padded block tables, each shape
            ``[batch, max_num_blocks]`` int32 device, one per kv cache.
    """

    lens: torch.Tensor = ...
    """Per-sample query lengths, shape ``[batch]`` int32 device."""
    cached_lens: torch.Tensor = ...
    """Per-sample cached lengths, shape ``[batch]`` int32 device."""
    positions: torch.Tensor = ...
    """Position tensor, shape ``[batched_tokens]`` int64."""
    lens_cpu: List[int] = ...
    """Per-sample query lengths (host, for tile-size selection)."""
    cached_lens_cpu: List[int] = ...
    """Per-sample cached lengths (host, for tile-size selection)."""
    query_start_loc: torch.Tensor = ...
    """Prefix-sum of lens, shape ``[batch]`` int32."""
    slot_mapping: Sequence[torch.Tensor] = ...
    """Slot indices, each shape ``[batched_tokens]`` int32 device, per kv cache."""
    block_tables: Sequence[torch.Tensor] = ...
    """Padded block tables, each shape ``[batch, max_num_blocks]`` int32 device, per kv cache."""

class AttnType(Enum):
    """Attention type enum exported by the native extension."""

    AttnMHA = ...
    """Standard multi-head attention."""
    AttnMLA = ...
    """Multi-head latent attention."""
    AttnDSA = ...
    """Dual sparse attention."""
    AttnHybrid = ...
    """Hybrid full + linear attention (Qwen3.5)."""
    AttnCxA = ...
    """C4A / C128A attention (DeepSeek-V4)."""

class ScoringFuncType(Enum):
    """MoE scoring function enum exported by the native extension."""

    ScoringFuncSoftmax = ...
    """Softmax-based expert routing."""
    ScoringFuncSigmoid = ...
    """Sigmoid-based expert routing."""
    ScoringFuncSqrtsoftplus = ...
    """Sqrtsoftplus-based expert routing (hash or topk)."""

AttnMHA: AttnType = ...
"""Alias for :attr:`AttnType.AttnMHA`."""
AttnHybrid: AttnType = ...
"""Alias for :attr:`AttnType.AttnHybrid`."""
AttnMLA: AttnType = ...
"""Alias for :attr:`AttnType.AttnMLA`."""
AttnDSA: AttnType = ...
"""Alias for :attr:`AttnType.AttnDSA`."""
AttnCxA: AttnType = ...
"""Alias for :attr:`AttnType.AttnCxA`."""

ScoringFuncSoftmax: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSoftmax`."""
ScoringFuncSigmoid: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSigmoid`."""
ScoringFuncSqrtsoftplus: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSqrtsoftplus`."""

class Model:
    """Model weights and forward methods.

    Attributes:
        embed (torch.Tensor): Embedding table, shape ``[vocab_size/def_tp_size, hidden_size]``,
            model dtype (fp16/bf16).
        norm (torch.Tensor): Final RMSNorm weight, shape ``[hidden_size]``, model dtype.
        norm_bias (torch.Tensor): Final RMSNorm bias, shape ``[hidden_size]``, model dtype. Optional.
        head (torch.Tensor): LM head weight, shape ``[vocab_size/def_tp_size, hidden_size]``,
            model dtype.
        attn_norm (List[torch.Tensor]): Attention norm weights per layer, each
            ``[hidden_size]``, model dtype.
        attn_norm_bias (List[torch.Tensor]): Attention norm bias per layer, each
            ``[hidden_size]``, model dtype. Optional.
        attn_out (List[torch.Tensor]): Attention output projection weights per layer, each
            ``[hidden_size, n_local_heads*head_dim]`` (MHA) / ``[hidden_size,
            n_local_heads*v_head_dim]`` (MLA). Model dtype / int8 / int4. Empty for CxA.
        attn_out_input_scale (List[torch.Tensor]): Attn output static-quant input scale
            (reciprocal) per layer, each ``[in_features]``, bf16.
        attn_out_input_offset (List[torch.Tensor]): Attn output static-quant input offset
            per layer, each ``[in_features]``, bf16.
        attn_out_quant_bias (List[torch.Tensor]): Attn output quantization bias per layer,
            each ``[hidden_size]``, int32 (tp_rank 0 only).
        attn_out_deq_scale (List[torch.Tensor]): Attn output dequantization scale per layer,
            each ``[2*hidden_size, 1]``, fp32.
        mha_qkv (List[torch.Tensor]): MHA QKV weights per layer, each
            ``[(n_local_heads+2*n_local_kv_heads)*head_dim (+q_dim if gate), hidden_size]``,
            model dtype / int8 / int4.
        mha_qkv_bias (List[torch.Tensor]): MHA QKV bias per layer, each
            ``[qkv_dim (+q_dim if gate)]``, model dtype. Present only when ``qkv_bias`` is set.
        mha_qkv_input_scale (List[torch.Tensor]): MHA QKV static-quant input scale per layer,
            each ``[hidden_size]``, bf16.
        mha_qkv_input_offset (List[torch.Tensor]): MHA QKV static-quant input offset per
            layer, each ``[hidden_size]``, bf16.
        mha_qkv_quant_bias (List[torch.Tensor]): MHA QKV quantization bias per layer, each
            ``[qkv_dim(+gate)]``, int32.
        mha_qkv_deq_scale (List[torch.Tensor]): MHA QKV dequantization scale per layer, each
            ``[2*qkv_dim(+gate), 1]``, fp32.
        mha_q_norm (List[torch.Tensor]): MHA Q norm weights per layer, each ``[head_dim]``
            (qk_norm) or ``[head_dim*n_local_heads]`` (qk_norm_full), model dtype.
        mha_q_norm_bias (List[torch.Tensor]): MHA Q norm bias per layer, same shape as
            :attr:`mha_q_norm`, model dtype. Optional.
        mha_k_norm (List[torch.Tensor]): MHA K norm weights per layer, each ``[head_dim]``
            (qk_norm) or ``[head_dim*n_local_kv_heads]`` (qk_norm_full), model dtype.
        mha_k_norm_bias (List[torch.Tensor]): MHA K norm bias per layer, same shape as
            :attr:`mha_k_norm`, model dtype. Optional.
        mla_qkv_a (List[torch.Tensor]): MLA QA KVA weights per layer, each
            ``[q_lora_rank + kv_lora_rank + rope_head_dim, hidden_size]``, model dtype /
            int8 / int4.
        mla_qkv_a_input_scale (List[torch.Tensor]): MLA QA KVA quantization input scale per
            layer, each ``[hidden_size]``, bf16.
        mla_qkv_a_input_offset (List[torch.Tensor]): MLA QA KVA quantization input offset
            per layer, each ``[hidden_size]``, bf16.
        mla_qkv_a_quant_bias (List[torch.Tensor]): MLA QA KVA quantization bias per layer,
            each ``[q_lora_rank + kv_lora_rank + rope_head_dim]``, int32.
        mla_qkv_a_deq_scale (List[torch.Tensor]): MLA QA KVA dequantization scale per layer,
            each ``[2*(q_lora_rank + kv_lora_rank + rope_head_dim), 1]``, fp32.
        mla_q_b (List[torch.Tensor]): MLA QB weights per layer, each
            ``[n_local_heads*(nope_head_dim+rope_head_dim), q_lora_rank]`` (MLA/DSA) or
            ``[n_local_heads*head_dim, q_lora_rank]`` (CxA), model dtype / int8 / int4.
        mla_q_b_input_scale (List[torch.Tensor]): MLA QB quantization input scale per layer,
            each ``[q_lora_rank]``, bf16.
        mla_q_b_input_offset (List[torch.Tensor]): MLA QB quantization input offset per
            layer, each ``[q_lora_rank]``, bf16.
        mla_q_b_quant_bias (List[torch.Tensor]): MLA QB quantization bias per layer, each
            ``[out_dim]``, int32.
        mla_q_b_deq_scale (List[torch.Tensor]): MLA QB dequantization scale per layer, each
            ``[2*out_dim, 1]``, fp32.
        mla_q_norm (List[torch.Tensor]): MLA Q norm weights per layer, each
            ``[q_lora_rank]``, model dtype.
        mla_q_norm_bias (List[torch.Tensor]): MLA Q norm bias per layer, each
            ``[q_lora_rank]``, model dtype. Optional.
        mla_wuv (List[torch.Tensor]): MLA W_UV weights per layer, shape (n_local_heads, kv_lora_rank, v_head_dim).
        mla_wuk_t (List[torch.Tensor]): MLA W_UK^T weights per layer, shape (n_local_heads, qk_nope_head_dim, kv_lora_rank).
        mla_kv_norm (List[torch.Tensor]): MLA KV norm weights per layer, each
            ``[kv_lora_rank]`` (MLA/DSA) or ``[head_dim]`` (CxA), model dtype.
        mla_kv_norm_bias (List[torch.Tensor]): MLA KV norm bias per layer, same shape as
            :attr:`mla_kv_norm`, model dtype. Optional.
        index_q_b (List[torch.Tensor]): DSA index QB weights per layer, each
            ``[index_n_heads*index_head_dim, q_lora_rank]``, model dtype / int8 / int4
            (expected to run with ``def_tp_size == 1``; not enforced natively).
        index_q_b_input_scale (List[torch.Tensor]): DSA index QB quantization input scale
            per layer, each ``[q_lora_rank]``, bf16.
        index_q_b_input_offset (List[torch.Tensor]): DSA index QB quantization input offset
            per layer, each ``[q_lora_rank]``, bf16.
        index_q_b_quant_bias (List[torch.Tensor]): DSA index QB quantization bias per layer,
            each ``[index_n_heads*index_head_dim]``, int32.
        index_q_b_deq_scale (List[torch.Tensor]): DSA index QB dequantization scale per
            layer, each ``[2*index_n_heads*index_head_dim, 1]``, fp32.
        index_k_weights_proj (List[torch.Tensor]): DSA index K and weights projection
            combined per layer, each ``[index_head_dim + index_n_heads, hidden_size]``,
            model dtype.
        index_k_norm (List[torch.Tensor]): DSA index K norm weights per layer, each
            ``[index_head_dim]``, model dtype or fp32.
        index_k_norm_bias (List[torch.Tensor]): DSA index K norm bias per layer, each
            ``[index_head_dim]``, model dtype or fp32.
        linear_in_proj_qkv (List[torch.Tensor]): Linear attention QKV projection weights per
            layer, each ``[2*n_local_k*linear_key_head_dim + n_local_v*linear_value_head_dim,
            hidden_size]``, model dtype / int8 / int4.
        linear_in_proj_z (List[torch.Tensor]): Linear attention Z projection weights per
            layer, each ``[n_local_v*linear_value_head_dim, hidden_size]``, model dtype /
            int8 / int4.
        linear_in_proj_b (List[torch.Tensor]): Linear attention B projection weights per
            layer, each ``[n_local_v, hidden_size]``, model dtype / int8 / int4.
        linear_in_proj_a (List[torch.Tensor]): Linear attention A projection weights per
            layer, each ``[n_local_v, hidden_size]``, model dtype / int8 / int4.
        linear_conv1d (List[torch.Tensor]): Linear attention conv1d weights per layer, each
            ``[conv_dim, 1, linear_conv_kernel_dim]``, model dtype.
        linear_a_log (List[torch.Tensor]): Linear attention A_log parameters per layer, each
            ``[n_local_v]``, model dtype.
        linear_dt_bias (List[torch.Tensor]): Linear attention dt_bias parameters per layer,
            each ``[n_local_v]``, model dtype.
        linear_norm (List[torch.Tensor]): Linear attention gated RMSNorm weights per layer,
            each ``[linear_value_head_dim]``, model dtype.
        linear_out_proj (List[torch.Tensor]): Linear attention output projection weights per
            layer, each ``[hidden_size, n_local_v*linear_value_head_dim]``, model dtype /
            int8 / int4.
        mlp_norm (List[torch.Tensor]): MLP norm weights per layer, each ``[hidden_size]``,
            model dtype.
        mlp_norm_bias (List[torch.Tensor]): MLP norm bias per layer, each ``[hidden_size]``,
            model dtype. Optional.
        mlp_up_gate (List[torch.Tensor]): Dense up-gate weights per layer, each
            ``[2*local_intermediate_size, hidden_size]``, model dtype / int8 / int4.
        mlp_up_gate_input_scale (List[torch.Tensor]): Dense up-gate quantization input scale
            per layer, each ``[hidden_size]``, bf16.
        mlp_up_gate_input_offset (List[torch.Tensor]): Dense up-gate quantization input
            offset per layer, each ``[hidden_size]``, bf16.
        mlp_up_gate_quant_bias (List[torch.Tensor]): Dense up-gate quantization bias per
            layer, each ``[2*local_intermediate_size]``, int32.
        mlp_up_gate_deq_scale (List[torch.Tensor]): Dense up-gate dequantization scale per
            layer, each ``[2*2*local_intermediate_size, 1]``, fp32.
        mlp_down (List[torch.Tensor]): Dense down weights per layer, each
            ``[hidden_size, local_intermediate_size]``, model dtype / int8 / int4.
        mlp_down_input_scale (List[torch.Tensor]): Dense down quantization input scale per
            layer, each ``[local_intermediate_size]``, bf16.
        mlp_down_input_offset (List[torch.Tensor]): Dense down quantization input offset per
            layer, each ``[local_intermediate_size]``, bf16.
        mlp_down_quant_bias (List[torch.Tensor]): Dense down quantization bias per layer,
            each ``[hidden_size]``, int32.
        mlp_down_deq_scale (List[torch.Tensor]): Dense down dequantization scale per layer,
            each ``[2*hidden_size, 1]``, fp32.
        gate (List[torch.Tensor]): MoE gate weights per layer, each
            ``[n_routed_experts, hidden_size]``, fp32 or bf16.
        gate_bias (List[torch.Tensor]): MoE gate bias per layer, each
            ``[n_routed_experts]``, fp32. Optional on sqrtsoftplus scoring layers
            (required for sigmoid); not consumed by hash-mode layers.
        se_up_gate (List[torch.Tensor]): Shared-expert up-gate weights per layer, each
            ``[2*moe_intermediate_size, hidden_size]`` (or TP-sharded), model dtype / int8.
        se_up_gate_deq_scale (List[torch.Tensor]): Shared-expert up-gate scales per layer,
            each ``[2*2*moe_intermediate_size, 1]``, fp32 (or
            ``[2*2*moe_intermediate_size/moe_tp_size, 1]`` when the shared-expert up-gate
            weight is TP-sharded).
        se_down (List[torch.Tensor]): Shared-expert down weights per layer, each
            ``[hidden_size, moe_intermediate_size]`` (or TP-sharded), model dtype / int8.
        se_down_deq_scale (List[torch.Tensor]): Shared-expert down scales per layer, each
            ``[2*hidden_size, 1]``, fp32.
        se_gate (List[torch.Tensor]): Optional shared-expert sigmoid gate, shape [1, hidden] per MoE layer.
        re_up_gate (List[torch.Tensor]): Routed-expert up-gate weights, one per local expert,
            each ``[2*(moe_intermediate_size/moe_tp_size), hidden_size]``, model dtype /
            int8 / int4.
        re_up_gate_scale (List[torch.Tensor]): Routed-expert up-gate scales(deprecated).
        re_up_gate_deq_scale (List[torch.Tensor]): Routed-expert up-gate scales, one per
            local expert, each ``[2*2*(moe_intermediate_size/moe_tp_size), 1]``, fp32.
        re_down (List[torch.Tensor]): Routed-expert down weights, one per local expert, each
            ``[hidden_size, moe_intermediate_size/moe_tp_size]``, model dtype / int8 / int4.
        re_down_scale (List[torch.Tensor]): Routed-expert down scales(deprecated).
        re_down_deq_scale (List[torch.Tensor]): Routed-expert down scales, one per local
            expert, each ``[2*hidden_size, 1]``, fp32.
        attn_sink (List[torch.Tensor]): Per-head attention sink (DeepSeek-V4), each
            ``[n_local_heads]``, fp32.
        attn_wq_a (List[torch.Tensor]): Per-layer attention wq_a (DeepSeek-V4), each
            ``[q_lora_rank, hidden_size]``, model dtype / int8 / int4.
        attn_wq_a_input_scale (List[torch.Tensor]): Attn wq_a quantization input scale per
            layer, each ``[hidden_size]``, bf16.
        attn_wq_a_input_offset (List[torch.Tensor]): Attn wq_a quantization input offset per
            layer, each ``[hidden_size]``, bf16.
        attn_wq_a_quant_bias (List[torch.Tensor]): Attn wq_a quantization bias per layer,
            each ``[q_lora_rank]``, int32.
        attn_wq_a_deq_scale (List[torch.Tensor]): Attn wq_a dequantization scale per layer,
            each ``[2*q_lora_rank, 1]``, fp32.
        attn_wo_a (List[torch.Tensor]): Per-layer output projection wo_a (DeepSeek-V4), each
            ``[n_local_groups*o_lora_rank, n_local_heads*head_dim/n_local_groups]``, model dtype.
        attn_wo_b (List[torch.Tensor]): Per-layer output projection wo_b (DeepSeek-V4), each
            ``[hidden_size, n_local_groups*o_lora_rank]``, model dtype.
        attn_wkv (List[torch.Tensor]): Per-layer attention wkv (DeepSeek-V4), each
            ``[head_dim, hidden_size]``, model dtype / int8 / int4.
        attn_wkv_input_scale (List[torch.Tensor]): Attn wkv quantization input scale per
            layer, each ``[hidden_size]``, bf16.
        attn_wkv_input_offset (List[torch.Tensor]): Attn wkv quantization input offset per
            layer, each ``[hidden_size]``, bf16.
        attn_wkv_quant_bias (List[torch.Tensor]): Attn wkv quantization bias per layer, each
            ``[head_dim]``, int32.
        attn_wkv_deq_scale (List[torch.Tensor]): Attn wkv dequantization scale per layer,
            each ``[2*head_dim, 1]``, fp32.
        comp_ape (List[torch.Tensor]): Compressor ape per layer (DeepSeek-V4), each
            ``[compress_ratios[layer], coff*head_dim]``, fp32.
        comp_w_kv (List[torch.Tensor]): Compressor wkv per layer (DeepSeek-V4, fp32), each
            ``[coff*head_dim, hidden_size]``.
        comp_w_gate (List[torch.Tensor]): Compressor wgate per layer (DeepSeek-V4, fp32),
            each ``[coff*head_dim, hidden_size]``.
        comp_norm (List[torch.Tensor]): Compressor RMSNorm weight per layer (DeepSeek-V4),
            each ``[head_dim]``, fp32.
        idx_wq_b (List[torch.Tensor]): Indexer wq_b per layer (DeepSeek-V4), each
            ``[index_n_heads*index_head_dim/def_tp_size, q_lora_rank]`` (ColumnParallel,
            output dim sharded), model dtype / int8 / int4.
        idx_wq_b_input_scale (List[torch.Tensor]): Indexer wq_b quantization input scale per
            layer, each ``[q_lora_rank]``, bf16.
        idx_wq_b_input_offset (List[torch.Tensor]): Indexer wq_b quantization input offset
            per layer, each ``[q_lora_rank]``, bf16.
        idx_wq_b_quant_bias (List[torch.Tensor]): Indexer wq_b quantization bias per layer,
            each ``[index_n_heads*index_head_dim]``, int32.
        idx_wq_b_deq_scale (List[torch.Tensor]): Indexer wq_b dequantization scale per
            layer, each ``[2*index_n_heads*index_head_dim, 1]``, fp32.
        idx_weights_proj (List[torch.Tensor]): Indexer weights_proj per layer (DeepSeek-V4),
            each ``[index_n_heads/def_tp_size, hidden_size]`` (ColumnParallel, output dim
            sharded), model dtype.
        idx_comp_ape (List[torch.Tensor]): Indexer compressor ape per layer (DeepSeek-V4),
            each ``[4, 2*index_head_dim]``, fp32.
        idx_comp_w_kv (List[torch.Tensor]): Indexer compressor wkv per layer (fp32), each
            ``[2*index_head_dim, hidden_size]``.
        idx_comp_w_gate (List[torch.Tensor]): Indexer compressor wgate per layer (fp32),
            each ``[2*index_head_dim, hidden_size]``.
        idx_comp_norm (List[torch.Tensor]): Indexer compressor norm per layer (DeepSeek-V4),
            each ``[index_head_dim]``, fp32.
        hc_attn_fn (List[torch.Tensor]): MHC attn fn per layer (DeepSeek-V4), each
            ``[(2+hc_mult)*hc_mult, hc_mult*hidden_size]``, fp32.
        hc_ffn_fn (List[torch.Tensor]): MHC ffn fn per layer (DeepSeek-V4), each
            ``[(2+hc_mult)*hc_mult, hc_mult*hidden_size]``, fp32.
        hc_attn_base (List[torch.Tensor]): MHC attn base per layer (DeepSeek-V4), each
            ``[(2+hc_mult)*hc_mult]``, fp32.
        hc_ffn_base (List[torch.Tensor]): MHC ffn base per layer (DeepSeek-V4), each
            ``[(2+hc_mult)*hc_mult]``, fp32.
        hc_attn_scale (List[torch.Tensor]): MHC attn scale per layer (DeepSeek-V4), each
            ``[3]``, fp32.
        hc_ffn_scale (List[torch.Tensor]): MHC ffn scale per layer (DeepSeek-V4), each
            ``[3]``, fp32.
        hc_head_fn (torch.Tensor): MHC head fn (Transformer-level, DeepSeek-V4), shape
            ``[hc_mult, hc_mult*hidden_size]``, fp32.
        hc_head_base (torch.Tensor): MHC head base (Transformer-level, DeepSeek-V4), shape
            ``[hc_mult]``, fp32.
        hc_head_scale (torch.Tensor): MHC head scale (Transformer-level, DeepSeek-V4), shape
            ``[1]``, fp32.
    """

    embed: torch.Tensor = ...
    """Embedding table, shape ``[vocab_size/def_tp_size, hidden_size]``, model dtype (fp16/bf16)."""
    norm: torch.Tensor = ...
    """Final RMSNorm weight, shape ``[hidden_size]``, model dtype."""
    norm_bias: torch.Tensor = ...
    """Final RMSNorm bias, shape ``[hidden_size]``, model dtype. Optional (empty when absent)."""
    head: torch.Tensor = ...
    """LM head weight, shape ``[vocab_size/def_tp_size, hidden_size]``, model dtype."""
    attn_norm: List[torch.Tensor] = ...
    """Attention norm weights per layer, each shape ``[hidden_size]``, model dtype."""
    attn_norm_bias: List[torch.Tensor] = ...
    """Attention norm bias per layer, each shape ``[hidden_size]``, model dtype. Optional."""
    attn_out: List[torch.Tensor] = ...
    """Attention output projection weight per layer, shape ``[hidden_size, n_local_heads*head_dim]``
    (MHA) or ``[hidden_size, n_local_heads*v_head_dim]`` (MLA/DSA). Model dtype, int8 (quantized),
    or int32-packed int4. Empty for CxA (replaced by attn_wo_a/attn_wo_b). Row-parallel."""
    attn_out_input_scale: List[torch.Tensor] = ...
    """Attn output static-quant input scale (reciprocal) per layer, shape ``[in_features]``, bf16.
    Present only when the weight is int8 with static quant."""
    attn_out_input_offset: List[torch.Tensor] = ...
    """Attn output static-quant input offset per layer, shape ``[in_features]``, bf16. Static quant only."""
    attn_out_quant_bias: List[torch.Tensor] = ...
    """Attn output quantization bias per layer, shape ``[hidden_size]``, int32. Bound only on
    tp_rank 0 (row-parallel). Used by both static and dynamic quantization."""
    attn_out_deq_scale: List[torch.Tensor] = ...
    """Attn output weight dequant scale per layer, shape ``[2*hidden_size, 1]``, fp32. Required
    when the weight is int8."""
    mha_qkv: List[torch.Tensor] = ...
    """MHA fused QKV weight per layer, shape ``[(n_local_heads+2*n_local_kv_heads)*head_dim
    (+ n_local_heads*head_dim if attn_output_gate), hidden_size]`` with fused layout [Q|K|V]
    (or [Q|K|V|Gate]). Model dtype, int8, or int32-packed int4. MHA/hybrid full-attention layers."""
    mha_qkv_bias: List[torch.Tensor] = ...
    """MHA QKV bias per layer, shape ``[qkv_dim (+q_dim if gate)]``, model dtype. Present only
    when config ``qkv_bias`` is true."""
    mha_qkv_input_scale: List[torch.Tensor] = ...
    """MHA QKV static-quant input scale (reciprocal) per layer, shape ``[hidden_size]``, bf16.
    In hybrid models these lists also supply the linear_in_proj_* quant params."""
    mha_qkv_input_offset: List[torch.Tensor] = ...
    """MHA QKV static-quant input offset per layer, shape ``[hidden_size]``, bf16."""
    mha_qkv_quant_bias: List[torch.Tensor] = ...
    """MHA QKV quantization bias per layer, shape ``[qkv_dim(+gate)]``, int32."""
    mha_qkv_deq_scale: List[torch.Tensor] = ...
    """MHA QKV weight dequant scale per layer, shape ``[2*qkv_dim(+gate), 1]``, fp32. Required
    when the weight is int8."""
    mha_q_norm: List[torch.Tensor] = ...
    """MHA Q norm weight per layer, shape ``[head_dim]`` (qk_norm) or
    ``[head_dim*n_local_heads]`` (qk_norm_full), model dtype."""
    mha_q_norm_bias: List[torch.Tensor] = ...
    """MHA Q norm bias per layer, same shape as :attr:`mha_q_norm`, model dtype. Optional."""
    mha_k_norm: List[torch.Tensor] = ...
    """MHA K norm weight per layer, shape ``[head_dim]`` (qk_norm) or
    ``[head_dim*n_local_kv_heads]`` (qk_norm_full), model dtype."""
    mha_k_norm_bias: List[torch.Tensor] = ...
    """MHA K norm bias per layer, same shape as :attr:`mha_k_norm`, model dtype. Optional."""
    mla_qkv_a: List[torch.Tensor] = ...
    """MLA fused Q-A / KV-A weight per layer, shape
    ``[q_lora_rank + kv_lora_rank + rope_head_dim, hidden_size]``, model dtype / int8 /
    int32-packed int4. Not sharded."""
    mla_qkv_a_input_scale: List[torch.Tensor] = ...
    """MLA QKVA static-quant input scale (reciprocal) per layer, shape ``[hidden_size]``, bf16."""
    mla_qkv_a_input_offset: List[torch.Tensor] = ...
    """MLA QKVA static-quant input offset per layer, shape ``[hidden_size]``, bf16."""
    mla_qkv_a_quant_bias: List[torch.Tensor] = ...
    """MLA QKVA quantization bias per layer, shape
    ``[q_lora_rank + kv_lora_rank + rope_head_dim]``, int32."""
    mla_qkv_a_deq_scale: List[torch.Tensor] = ...
    """MLA QKVA weight dequant scale per layer, shape
    ``[2*(q_lora_rank + kv_lora_rank + rope_head_dim), 1]``, fp32."""
    mla_q_b: List[torch.Tensor] = ...
    """MLA Q-B weight per layer, shape ``[n_local_heads*(nope_head_dim+rope_head_dim),
    q_lora_rank]`` (MLA/DSA) or ``[n_local_heads*head_dim, q_lora_rank]`` (CxA reuses the
    field as wq_b). Model dtype / int8 / int32-packed int4. Column-parallel."""
    mla_q_b_input_scale: List[torch.Tensor] = ...
    """MLA QB static-quant input scale (reciprocal) per layer, shape ``[q_lora_rank]``, bf16."""
    mla_q_b_input_offset: List[torch.Tensor] = ...
    """MLA QB static-quant input offset per layer, shape ``[q_lora_rank]``, bf16."""
    mla_q_b_quant_bias: List[torch.Tensor] = ...
    """MLA QB quantization bias per layer, shape ``[out_dim]``, int32."""
    mla_q_b_deq_scale: List[torch.Tensor] = ...
    """MLA QB weight dequant scale per layer, shape ``[2*out_dim, 1]``, fp32."""
    mla_q_norm: List[torch.Tensor] = ...
    """MLA Q norm weight per layer, shape ``[q_lora_rank]``, model dtype."""
    mla_q_norm_bias: List[torch.Tensor] = ...
    """MLA Q norm bias per layer, shape ``[q_lora_rank]``, model dtype. Optional."""
    mla_wuv: List[torch.Tensor] = ...
    """MLA W_UV weight per layer, shape (n_local_heads, kv_lora_rank, v_head_dim), model dtype."""
    mla_wuk_t: List[torch.Tensor] = ...
    """MLA W_UK^T weight per layer, shape (n_local_heads, qk_nope_head_dim, kv_lora_rank), model dtype."""
    mla_kv_norm: List[torch.Tensor] = ...
    """MLA KV norm weight per layer, shape ``[kv_lora_rank]`` (MLA/DSA) or ``[head_dim]`` (CxA),
    model dtype."""
    mla_kv_norm_bias: List[torch.Tensor] = ...
    """MLA KV norm bias per layer, same shape as :attr:`mla_kv_norm`, model dtype. Optional."""
    index_q_b: List[torch.Tensor] = ...
    """DSA index QB weight per layer, shape ``[index_n_heads*index_head_dim, q_lora_rank]``,
    model dtype / int8 / int32-packed int4. Bound only on full-indexer layers (expected to
    run with ``def_tp_size == 1``; not enforced natively); empty on shared-indexer layers."""
    index_q_b_input_scale: List[torch.Tensor] = ...
    """DSA index QB static-quant input scale (reciprocal) per layer, shape ``[q_lora_rank]``, bf16."""
    index_q_b_input_offset: List[torch.Tensor] = ...
    """DSA index QB static-quant input offset per layer, shape ``[q_lora_rank]``, bf16."""
    index_q_b_quant_bias: List[torch.Tensor] = ...
    """DSA index QB quantization bias per layer, shape ``[index_n_heads*index_head_dim]``, int32."""
    index_q_b_deq_scale: List[torch.Tensor] = ...
    """DSA index QB weight dequant scale per layer, shape ``[2*index_n_heads*index_head_dim, 1]``, fp32."""
    index_k_weights_proj: List[torch.Tensor] = ...
    """DSA index K/weights projection weight per layer, shape
    ``[index_head_dim + index_n_heads, hidden_size]``, model dtype."""
    index_k_norm: List[torch.Tensor] = ...
    """DSA index K LayerNorm weight per layer, shape ``[index_head_dim]``, model dtype or fp32."""
    index_k_norm_bias: List[torch.Tensor] = ...
    """DSA index K LayerNorm bias per layer, shape ``[index_head_dim]``, model dtype or fp32."""
    linear_in_proj_qkv: List[torch.Tensor] = ...
    """Linear attention QKV projection weight per layer, shape
    ``[2*n_local_k*linear_key_head_dim + n_local_v*linear_value_head_dim, hidden_size]``,
    model dtype / int8 / int32-packed int4 (quant params via :attr:`mha_qkv_*`)."""
    linear_in_proj_z: List[torch.Tensor] = ...
    """Linear attention Z projection weight per layer, shape
    ``[n_local_v*linear_value_head_dim, hidden_size]``, model dtype / int8 / int4."""
    linear_in_proj_b: List[torch.Tensor] = ...
    """Linear attention B projection weight per layer, shape ``[n_local_v, hidden_size]``,
    model dtype / int8 / int4."""
    linear_in_proj_a: List[torch.Tensor] = ...
    """Linear attention A projection weight per layer, shape ``[n_local_v, hidden_size]``,
    model dtype / int8 / int4."""
    linear_conv1d: List[torch.Tensor] = ...
    """Linear attention causal conv1d kernel per layer, shape
    ``[conv_dim, 1, linear_conv_kernel_dim]`` where
    ``conv_dim = 2*n_local_k*linear_key_head_dim + n_local_v*linear_value_head_dim``,
    model dtype (fp32/fp16/bf16). kernel_dim <= 16."""
    linear_a_log: List[torch.Tensor] = ...
    """Linear attention A_log parameter per layer, shape ``[n_local_v]``, model dtype."""
    linear_dt_bias: List[torch.Tensor] = ...
    """Linear attention dt_bias parameter per layer, shape ``[n_local_v]``, model dtype."""
    linear_norm: List[torch.Tensor] = ...
    """Linear attention gated RMSNorm weight per layer, shape ``[linear_value_head_dim]``
    (replicated per v-head), model dtype."""
    linear_out_proj: List[torch.Tensor] = ...
    """Linear attention output projection weight per layer, shape
    ``[hidden_size, n_local_v*linear_value_head_dim]``, model dtype / int8 / int4 (quant
    params via :attr:`attn_out_*`). Row-parallel."""
    mlp_norm: List[torch.Tensor] = ...
    """MLP norm weights per layer, each shape ``[hidden_size]``, model dtype."""
    mlp_norm_bias: List[torch.Tensor] = ...
    """MLP norm bias per layer, each shape ``[hidden_size]``, model dtype. Optional."""
    mlp_up_gate: List[torch.Tensor] = ...
    """Dense up-gate weight per layer, shape ``[2*local_intermediate_size, hidden_size]``,
    model dtype / int8 / int32-packed int4. Dense layers only."""
    mlp_up_gate_input_scale: List[torch.Tensor] = ...
    """Dense up-gate static-quant input scale (reciprocal) per layer, shape ``[hidden_size]``, bf16."""
    mlp_up_gate_input_offset: List[torch.Tensor] = ...
    """Dense up-gate static-quant input offset per layer, shape ``[hidden_size]``, bf16."""
    mlp_up_gate_quant_bias: List[torch.Tensor] = ...
    """Dense up-gate quantization bias per layer, shape ``[2*local_intermediate_size]``, int32."""
    mlp_up_gate_deq_scale: List[torch.Tensor] = ...
    """Dense up-gate weight dequant scale per layer, shape ``[2*2*local_intermediate_size, 1]``, fp32."""
    mlp_down: List[torch.Tensor] = ...
    """Dense down weight per layer, shape ``[hidden_size, local_intermediate_size]``, model
    dtype / int8 / int32-packed int4. Row-parallel."""
    mlp_down_input_scale: List[torch.Tensor] = ...
    """Dense down static-quant input scale (reciprocal) per layer, shape
    ``[local_intermediate_size]``, bf16."""
    mlp_down_input_offset: List[torch.Tensor] = ...
    """Dense down static-quant input offset per layer, shape
    ``[local_intermediate_size]``, bf16."""
    mlp_down_quant_bias: List[torch.Tensor] = ...
    """Dense down quantization bias per layer, shape ``[hidden_size]``, int32 (tp_rank 0 only)."""
    mlp_down_deq_scale: List[torch.Tensor] = ...
    """Dense down weight dequant scale per layer, shape ``[2*hidden_size, 1]``, fp32."""
    gate: List[torch.Tensor] = ...
    """MoE gate weight per MoE layer, shape ``[n_routed_experts, hidden_size]``, fp32 or bf16."""
    gate_bias: List[torch.Tensor] = ...
    """MoE gate bias per MoE layer, shape ``[n_routed_experts]``, fp32. Required for sigmoid
    scoring; unused for softmax; optional on sqrtsoftplus hash layers."""
    tid2eid: List[torch.Tensor] = ...
    """MoE hash layers: token-id->expert lookup table ``[vocab_size, n_act_experts]`` int32 per layer."""
    se_up_gate: List[torch.Tensor] = ...
    """Shared-expert up-gate weight per MoE layer, shape ``[2*moe_intermediate_size, hidden_size]``
    (replicated) or ``[2*moe_intermediate_size/tp, hidden_size]`` (TP-sharded; auto-detected).
    Model dtype or int8."""
    se_up_gate_deq_scale: List[torch.Tensor] = ...
    """Shared-expert up-gate weight dequant scale per MoE layer, shape
    ``[2*2*moe_intermediate_size, 1]``, fp32. Empty when unquantized."""
    se_down: List[torch.Tensor] = ...
    """Shared-expert down weight per MoE layer, shape ``[hidden_size, moe_intermediate_size]``
    (replicated) or ``[hidden_size, moe_intermediate_size/tp]`` (sharded). Model dtype or int8."""
    se_down_deq_scale: List[torch.Tensor] = ...
    """Shared-expert down weight dequant scale per MoE layer, shape ``[2*hidden_size, 1]``, fp32."""
    se_gate: List[torch.Tensor] = ...
    """Optional shared-expert sigmoid gate weight per MoE layer, shape [1, hidden]."""
    re_up_gate: List[torch.Tensor] = ...
    """Routed-expert up-gate weights, one per local expert, each shape
    ``[2*(moe_intermediate_size/moe_tp_size), hidden_size]``. Model dtype, int8, or
    int32-packed int4 (MSD W4A8). Flat list ordered layer-major then local expert."""
    re_up_gate_scale: List[torch.Tensor] = ...
    """Routed-expert up-gate scales.(deprecated)"""
    re_up_gate_deq_scale: List[torch.Tensor] = ...
    """Routed-expert up-gate weight dequant scales, one per local expert, each shape
    ``[2*2*(moe_intermediate_size/moe_tp_size), 1]``, fp32."""
    re_down: List[torch.Tensor] = ...
    """Routed-expert down weights, one per local expert, each shape
    ``[hidden_size, moe_intermediate_size/moe_tp_size]``. Model dtype, int8, or int32-packed
    int4. Same flat ordering as :attr:`re_up_gate`."""
    re_down_scale: List[torch.Tensor] = ...
    """Routed-expert down scales.(deprecated)"""
    re_down_deq_scale: List[torch.Tensor] = ...
    """Routed-expert down weight dequant scales, one per local expert, each shape
    ``[2*hidden_size, 1]``, fp32."""

    # DeepSeek-V4 (CxA)
    attn_sink: List[torch.Tensor] = ...
    """Per-head attention sink (DeepSeek-V4), shape ``[n_local_heads]``, fp32."""
    attn_wq_a: List[torch.Tensor] = ...
    """Per-layer attention wq_a (DeepSeek-V4), shape ``[q_lora_rank, hidden_size]``, model
    dtype / int8 / int4."""
    attn_wq_a_input_scale: List[torch.Tensor] = ...
    """Attn wq_a static-quant input scale (reciprocal) per layer, shape ``[hidden_size]``, bf16."""
    attn_wq_a_input_offset: List[torch.Tensor] = ...
    """Attn wq_a static-quant input offset per layer, shape ``[hidden_size]``, bf16."""
    attn_wq_a_quant_bias: List[torch.Tensor] = ...
    """Attn wq_a quantization bias per layer, shape ``[q_lora_rank]``, int32."""
    attn_wq_a_deq_scale: List[torch.Tensor] = ...
    """Attn wq_a weight dequant scale per layer, shape ``[2*q_lora_rank, 1]``, fp32."""
    attn_wo_a: List[torch.Tensor] = ...
    """Per-layer output projection wo_a (DeepSeek-V4), shape
    ``[n_local_groups*o_lora_rank, n_local_heads*head_dim/n_local_groups]`` where
    ``n_local_groups = o_groups/tp``, model dtype."""
    attn_wo_b: List[torch.Tensor] = ...
    """Per-layer output projection wo_b (DeepSeek-V4), shape
    ``[hidden_size, n_local_groups*o_lora_rank]``, model dtype."""
    attn_wkv: List[torch.Tensor] = ...
    """Per-layer attention wkv (DeepSeek-V4), shape ``[head_dim, hidden_size]``, model dtype /
    int8 / int4."""
    attn_wkv_input_scale: List[torch.Tensor] = ...
    """Attn wkv static-quant input scale (reciprocal) per layer, shape ``[hidden_size]``, bf16."""
    attn_wkv_input_offset: List[torch.Tensor] = ...
    """Attn wkv static-quant input offset per layer, shape ``[hidden_size]``, bf16."""
    attn_wkv_quant_bias: List[torch.Tensor] = ...
    """Attn wkv quantization bias per layer, shape ``[head_dim]``, int32."""
    attn_wkv_deq_scale: List[torch.Tensor] = ...
    """Attn wkv weight dequant scale per layer, shape ``[2*head_dim, 1]``, fp32."""
    comp_ape: List[torch.Tensor] = ...
    """Compressor ape per layer (DeepSeek-V4), shape ``[compress_ratios[layer],
    coff*head_dim]`` with ``coff = 2`` if the ratio is 4 else ``1``, fp32. Empty on
    ratio-0 layers."""
    comp_w_kv: List[torch.Tensor] = ...
    """Compressor wkv per layer (DeepSeek-V4, fp32), shape ``[coff*head_dim, hidden_size]``.
    Empty on ratio-0 layers."""
    comp_w_gate: List[torch.Tensor] = ...
    """Compressor wgate per layer (DeepSeek-V4, fp32), shape ``[coff*head_dim, hidden_size]``.
    Empty on ratio-0 layers."""
    comp_norm: List[torch.Tensor] = ...
    """Compressor RMSNorm weight per layer (DeepSeek-V4), shape ``[head_dim]``, fp32. Empty on
    ratio-0 layers."""
    # Indexer.wq_b is v4-specific (different shape from DSA's index_q_b).
    idx_wq_b: List[torch.Tensor] = ...
    """Indexer wq_b per layer (DeepSeek-V4), shape ``[index_n_heads*index_head_dim,
    q_lora_rank]``, model dtype / int8 / int4. Bound only on compress_ratios==4 layers."""
    idx_wq_b_input_scale: List[torch.Tensor] = ...
    """Indexer wq_b static-quant input scale (reciprocal) per layer, shape ``[q_lora_rank]``, bf16."""
    idx_wq_b_input_offset: List[torch.Tensor] = ...
    """Indexer wq_b static-quant input offset per layer, shape ``[q_lora_rank]``, bf16."""
    idx_wq_b_quant_bias: List[torch.Tensor] = ...
    """Indexer wq_b quantization bias per layer, shape ``[index_n_heads*index_head_dim]``, int32."""
    idx_wq_b_deq_scale: List[torch.Tensor] = ...
    """Indexer wq_b weight dequant scale per layer, shape
    ``[2*index_n_heads*index_head_dim, 1]``, fp32."""
    idx_weights_proj: List[torch.Tensor] = ...
    """Indexer weights_proj per layer (DeepSeek-V4), shape ``[index_n_heads, hidden_size]``,
    model dtype. Ratio-4 layers only."""
    idx_comp_ape: List[torch.Tensor] = ...
    """Indexer compressor ape per layer (DeepSeek-V4), shape ``[4, 2*index_head_dim]``, fp32.
    Ratio-4 layers only."""
    idx_comp_w_kv: List[torch.Tensor] = ...
    """Indexer compressor wkv per layer (fp32), shape ``[2*index_head_dim, hidden_size]``.
    Ratio-4 layers only."""
    idx_comp_w_gate: List[torch.Tensor] = ...
    """Indexer compressor wgate per layer (fp32), shape ``[2*index_head_dim, hidden_size]``.
    Ratio-4 layers only."""
    idx_comp_norm: List[torch.Tensor] = ...
    """Indexer compressor norm per layer (DeepSeek-V4), shape ``[index_head_dim]``, fp32.
    Ratio-4 layers only."""
    hc_attn_fn: List[torch.Tensor] = ...
    """MHC attn mix matrix per layer (DeepSeek-V4), shape
    ``[(2+hc_mult)*hc_mult, hc_mult*hidden_size]``, fp32."""
    hc_ffn_fn: List[torch.Tensor] = ...
    """MHC ffn mix matrix per layer (DeepSeek-V4), shape
    ``[(2+hc_mult)*hc_mult, hc_mult*hidden_size]``, fp32."""
    hc_attn_base: List[torch.Tensor] = ...
    """MHC attn bias per layer (DeepSeek-V4), shape ``[(2+hc_mult)*hc_mult]``, fp32."""
    hc_ffn_base: List[torch.Tensor] = ...
    """MHC ffn bias per layer (DeepSeek-V4), shape ``[(2+hc_mult)*hc_mult]``, fp32."""
    hc_attn_scale: List[torch.Tensor] = ...
    """MHC attn per-segment scale per layer (DeepSeek-V4), shape ``[3]`` (pre/post/comb), fp32."""
    hc_ffn_scale: List[torch.Tensor] = ...
    """MHC ffn per-segment scale per layer (DeepSeek-V4), shape ``[3]`` (pre/post/comb), fp32."""
    hc_head_fn: torch.Tensor = ...
    """MHC head mix matrix (Transformer-level, DeepSeek-V4), shape
    ``[hc_mult, hc_mult*hidden_size]``, fp32."""
    hc_head_base: torch.Tensor = ...
    """MHC head bias (Transformer-level, DeepSeek-V4), shape ``[hc_mult]``, fp32 (pre-bias only)."""
    hc_head_scale: torch.Tensor = ...
    """MHC head scale (Transformer-level, DeepSeek-V4), shape ``[1]``, fp32 (scale_pre only)."""

    def init(self, config: ModelConfig, rank: int = 0) -> None:
        """Initialize native model state from Python-provided weights.

        Args:
            config (ModelConfig): Native model configuration.
            rank (int): Model-parallel rank.

        Returns:
            None: Native model state is created in place.

        Raises:
            ValueError: On configuration or parameter count mismatch.
            RuntimeError: If native initialization fails.
        """

    def forward(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass with host/vLLM-compatible attention metadata (V1).

        ``freqs_cis`` is a single tensor shared across all layers. For per-layer
        freqs tensors or device-side attention metadata, use :meth:`forward_v2`.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token ids, shape ``[num_tokens]``, int32/int64 device.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache, model dtype,
                paged layout (outer dim = block count):
                MHA ``[blocks, block_size, n_local_kv_heads, head_dim]`` (k, v);
                MLA/DSA k-nope ``[blocks, block_size, n_local_kv_heads, kv_lora_rank]`` +
                pe ``[blocks, block_size, n_local_kv_heads, rope_head_dim]`` (DSA adds
                indexer k ``[blocks, block_size, 1, index_head_dim]``);
                HYBRID linear layers hold conv state ``[max_batch, conv_dim,
                linear_conv_kernel_dim]`` + ssm state ``[max_batch, n_local_v,
                linear_key_head_dim, linear_value_head_dim]``;
                CXA holds the 5-tuple (indexer_state, indexer_k, compress_kv, state, swa_kv).
            freqs_cis (torch.Tensor): Rotary frequency table shared by all layers, shape
                ``[max_position, rope_head_dim]``, model dtype (per row ``[cos | sin]``
                concatenated). Indexed by each token's ``position`` value, not by row order,
                so it must cover the full ``max_seq_len`` range.
            output (torch.Tensor): Output hidden-state buffer, shape
                ``[num_tokens, hidden_size]``, model dtype.
            curr_stream (int): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On invalid shapes or cache mismatch.
        """

    def forward_get_logits(
        self,
        rt: Runtime,
        input: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run logits-only path.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input hidden states, shape ``[num_tokens, hidden_size]``,
                model dtype.
            indices (torch.Tensor): Logits indices (last-row gather per sample), shape
                ``[batch]``, int32 device.
            output (torch.Tensor): Output logits tensor, shape
                ``[def_tp_size, num_tokens, vocab_size/def_tp_size]``, model dtype
                (all-gathered over TP along dim 0).
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: If the native logits path fails.
        """

    def forward_and_get_logits(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass and materialize logits (host metadata, V1).

        ``freqs_cis`` is a single tensor shared across all layers. For per-layer
        freqs tensors or device-side attention metadata, use
        :meth:`forward_and_get_logits_v2`.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token ids, shape ``[num_tokens]``, int32/int64 device.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache (see :meth:`forward`).
            freqs_cis (Union[torch.Tensor, Sequence[torch.Tensor]]): Rotary
                frequency table shared by all layers, or a per-layer sequence; each
                ``[max_position, rope_head_dim]``, model dtype (per row ``[cos | sin]``
                concatenated, indexed by token position value).
            indices (torch.Tensor): Logits indices, shape ``[batch]``, int32 device.
            output (torch.Tensor): Output logits buffer, shape
                ``[def_tp_size, max_tokens_dp, vocab_size/def_tp_size]``, model dtype.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On invalid KV-cache layout or other native execution failures.
        """

    def forward_with_inputs_embeds(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
        deepstack_input: Sequence[torch.Tensor] = ...,
        input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Run forward pass with deepstack input embeddings (host metadata).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input embeddings, shape ``[num_tokens, hidden_size]``,
                model dtype (no embedding lookup).
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache (see :meth:`forward`).
            freqs_cis (torch.Tensor): Rotary frequency table, shape
                ``[max_position, rope_head_dim]``, model dtype (per row ``[cos | sin]``
                concatenated, indexed by token position value).
            output (torch.Tensor): Output hidden-state buffer, shape
                ``[num_tokens, hidden_size]``, model dtype.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.
            deepstack_input (Sequence[torch.Tensor], default empty): Extra deepstack embeddings,
                each shape ``[num_tokens, hidden_size]``, model dtype; length must equal
                ``deepstack_num_level``.
            input_ids (Optional[torch.Tensor], default None): Token ids for the sqrtsoftplus MoE
                hash-gate path (tid2eid[input_ids]), shape ``[num_tokens]`` int32; required only
                when scoring_func is sqrtsoftplus, otherwise may be left unset (defaults to None).

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On KV-cache/deepstack shape mismatch or other native execution failures.
        """

    def forward_v2(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMetaV2,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: Sequence[torch.Tensor],
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass with device-tensor attention metadata (V2).

        Unlike :meth:`forward` (V1), this path takes ``query_start_loc``,
        ``slot_mapping`` and ``block_tables`` as pre-built device tensors on
        ``attn_meta``; the native side skips host computation and H2D copies.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token ids, shape ``[num_tokens]``, int32/int64 device.
            attn_meta (AttnMetaV2): Device-tensor attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache (see :meth:`forward`).
            freqs_cis (Sequence[torch.Tensor]): Per-layer rotary frequency tables, each
                ``[max_position, rope_head_dim]``, model dtype (per row ``[cos | sin]``
                concatenated, indexed by token position value; list length >= n_layers for CXA;
                non-CXA reads only freqs_cis[0]).
            output (torch.Tensor): Output hidden-state buffer, shape
                ``[num_tokens, hidden_size]``, model dtype.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On shape mismatch or other native execution failures.
        """

    def forward_and_get_logits_v2(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMetaV2,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: Sequence[torch.Tensor],
        indices: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass and materialize logits (device metadata, V2).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token ids, shape ``[num_tokens]``, int32/int64 device.
            attn_meta (AttnMetaV2): Device-tensor attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache (see :meth:`forward`).
            freqs_cis (Sequence[torch.Tensor]): Per-layer rotary frequency tables, each
                ``[max_position, rope_head_dim]``, model dtype (per row ``[cos | sin]``
                concatenated, indexed by token position value).
            indices (torch.Tensor): Logits indices, shape ``[batch]``, int32 device.
            output (torch.Tensor): Output logits buffer, shape
                ``[def_tp_size, max_tokens_dp, vocab_size/def_tp_size]``, model dtype.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On shape mismatch or other native execution failures.
        """

    def forward_with_inputs_embeds_v2(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMetaV2,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
        deepstack_input: Sequence[torch.Tensor] = ...,
        input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Run forward pass with deepstack input embeddings (device metadata, V2).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input embeddings, shape ``[num_tokens, hidden_size]``,
                model dtype.
            attn_meta (AttnMetaV2): Device-tensor attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache (see :meth:`forward`).
            freqs_cis (torch.Tensor): Rotary frequency table, shape
                ``[max_position, rope_head_dim]``, model dtype (per row ``[cos | sin]``
                concatenated, indexed by token position value; single tensor, not a list).
            output (torch.Tensor): Output hidden-state buffer, shape
                ``[num_tokens, hidden_size]``, model dtype.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.
            deepstack_input (Sequence[torch.Tensor], default empty): Extra deepstack embeddings,
                each shape ``[num_tokens, hidden_size]``, model dtype; length must equal
                ``deepstack_num_level``.
            input_ids (Optional[torch.Tensor], default None): Token ids for the sqrtsoftplus MoE
                hash-gate path (tid2eid[input_ids]), shape ``[num_tokens]`` int32; required only
                when scoring_func is sqrtsoftplus, otherwise may be left unset (defaults to None).

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On shape mismatch or other native execution failures.
        """

    def get_tensor_pool_size(self, dbg: int = 0) -> int:
        """Get tensor-pool usage information.

        Args:
            dbg (int, default=0): Debug level forwarded to the native model.

        Returns:
            int: Current tensor pool size metric returned by native code.
        """

class CoreAssigner:
    """Helper to split hardware cores between prefill and decode work."""

    def __init__(self, prefill_ratio: float) -> None:
        """Create a core assigner.

        Args:
            prefill_ratio (float): Prefill-to-decode core split ratio.
        """

    def assign_core(self, is_decode: bool) -> float:
        """Assign cores for one request phase.

        Args:
            is_decode (bool): `True` for decode phase, `False` for prefill.

        Returns:
            float: Assigned core ratio.
        """

    def release_core(self, is_decode: bool) -> None:
        """Release cores previously assigned for one phase.

        Args:
            is_decode (bool): `True` for decode phase, `False` for prefill.

        Returns:
            None: Internal assignment state is updated.
        """

"""Low-level collective and kernel operator bindings."""

def all_gather(rt: Runtime, out: torch.Tensor, in_: torch.Tensor, comm_type: int = 0) -> None:
    """Collectively gather tensors from all ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for gathered values, any shape with
            ``out.numel == in_.numel * world_size``, dtype same as ``in_``.
        in_ (torch.Tensor): Local input shard, any shape/dtype (fp16/bf16/int8/int32/fp32/int64).
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def reduce_scatter(rt: Runtime, out: torch.Tensor, in_: torch.Tensor, comm_type: int = 0) -> None:
    """Reduce then scatter tensors across ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for the reduced local shard, any shape,
            dtype same as ``in_``.
        in_ (torch.Tensor): Input tensor to reduce (SUM), with
            ``in_.numel == out.numel * world_size``.
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def all_reduce(rt: Runtime, out: torch.Tensor, in_: torch.Tensor, comm_type: int = 0) -> None:
    """Collectively reduce tensors across all ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output tensor for reduced results (SUM), same numel and
            dtype as ``in_``.
        in_ (torch.Tensor): Input tensor to reduce, any shape (fp16/bf16/int8/int32/fp32/int64).
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def alltoallv(
    rt: Runtime,
    out: torch.Tensor,
    in_: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    sdispls: torch.Tensor,
    rdispls: torch.Tensor,
    comm_type: int = 0,
) -> None:
    """All-to-all vectorized collective communication.

    Each rank sends `send_counts[i]` elements starting at `sdispls[i]` to
    rank *i* and receives `recv_counts[i]` elements starting at `rdispls[i]`
    from rank *i*.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for received data, dtype same as ``in_``.
        in_ (torch.Tensor): Input tensor with data to send.
        send_counts (torch.Tensor): Per-rank send element counts, int, shape ``[world_size]``.
        recv_counts (torch.Tensor): Per-rank receive element counts, int, shape ``[world_size]``.
        sdispls (torch.Tensor): Per-rank send displacement offsets, int, shape ``[world_size]``.
        rdispls (torch.Tensor): Per-rank receive displacement offsets, int, shape ``[world_size]``.
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP), ``2`` (EP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If ``in_.dtype != out.dtype`` or the HCCL call fails.
    """
    ...

def add(rt: Runtime, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> None:
    """Elementwise add two tensors into output.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left operand, shape ``[m, n]``, fp16 or bf16.
        y (torch.Tensor): Right operand, shape ``[m, n]``, same dtype as ``x``.
        z (torch.Tensor): Output tensor, shape ``[m, n]``, same dtype as ``x``.

    Returns:
        None: `z` is written in place.

    Raises:
        RuntimeError: If tensor shapes or dtypes are unsupported by the kernel.
    """
    ...

def matmul(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    weight_nz: bool = False,
    transpose: bool = False,
) -> None:
    """Matrix multiplication with optional transpose/layout flags.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix, shape ``[m, k]``. Dtype combos (x, y, z): all fp16;
            all bf16; all fp32 (transpose=False); (bf16, fp32, fp32)/(bf16, fp32, bf16)
            (transpose=False); (int8, int8, fp16); (int4, int4, fp16) — int32-packed int4
            tensors are auto-viewed as int4.
        y (torch.Tensor): Right matrix/weight, shape ``[n, k]`` (transpose=False) or
            ``[k, n]`` (transpose=True).
        z (torch.Tensor): Output matrix, shape ``[m, n]``.
        weight_nz (bool): Whether `y` uses NZ weight layout.
        transpose (bool): Whether to transpose the right matrix in compute.

    Returns:
        None: `z` is written in place.

    Raises:
        RuntimeError: If the native kernel launch fails.
    """
    ...

def matmul_bench(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    x_warmup: torch.Tensor,
    y_warmup: torch.Tensor,
    z_warmup: torch.Tensor,
    iterations: int,
    warmup_iterations: int,
    weight_nz: bool = False,
    transpose: bool = False,
) -> int:
    """Matmul benchmark. Measures average time of a matmul operation.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix, shape ``[m, k]`` (see :func:`matmul` for dtype combos).
        y (torch.Tensor): Right matrix/weight, shape ``[n, k]`` or ``[k, n]`` per transpose.
        z (torch.Tensor): Output matrix, shape ``[m, n]``.
        x_warmup (torch.Tensor): Left matrix for warmup, same contract as ``x``.
        y_warmup (torch.Tensor): Right matrix/weight for warmup, same contract as ``y``.
        z_warmup (torch.Tensor): Output matrix for warmup, same contract as ``z``.
        iterations: Number of matmul iterations to run
        weight_nz (bool): Whether `y` uses NZ weight layout.
        transpose (bool): Whether to transpose the right matrix in compute.

    Returns:
        int: Measured average time for matmul operation in nanoseconds.

    Raises:
        RuntimeError: If the native kernel launch fails.
    """
    ...

def matmul_with_bias(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    bias: torch.Tensor,
    weight_nz: bool = False,
) -> None:
    """Matrix multiplication followed by bias add.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix, shape ``[m, k]`` (see :func:`matmul` for dtype combos,
            transpose=False only).
        y (torch.Tensor): Right matrix/weight, shape ``[n, k]``.
        z (torch.Tensor): Output matrix, shape ``[m, n]``.
        bias (torch.Tensor): Per-column bias added to output, shape ``[n]``.
        weight_nz (bool): Whether `y` uses NZ weight layout.

    Returns:
        None: `z` is written in place.

    Raises:
        RuntimeError: If the native kernel launch fails.
    """
    ...

def embed(
    rt: Runtime,
    weight: torch.Tensor,
    in_: torch.Tensor,
    out: torch.Tensor,
    start: int,
    end: int,
) -> None:
    """Embedding lookup over the provided token range.

    Args:
        rt (Runtime): Native runtime handle.
        weight (torch.Tensor): Embedding table, shape ``[vocab, hidden]``, fp16 or bf16
            (``hidden`` must be divisible by 16).
        in_ (torch.Tensor): Token IDs, shape ``[num_tokens]`` (read as uint32 row indices).
        out (torch.Tensor): Output embedding tensor, shape ``[num_tokens, hidden]``, same
            dtype as ``weight``.
        start (int): Start token index (inclusive).
        end (int): End token index (exclusive).

    Returns:
        None: `out` is written in place.
    """
    ...

def rmsnorm_variance_only(
    rt: Runtime,
    in_: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
    cnt_per_token: int = 1,
    in_start_offset: int = 0,
    out_start_offset: int = 0,
) -> None:
    """Compute variance for RMSNorm.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[tokens, dim]``, fp16 or bf16.
        out (torch.Tensor): Variance-only output tensor, shape ``[tokens, 1]``,
            fp32. The kernel writes one float per token row (only the first
            segment when ``cnt_per_token > 1``).
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.
        cnt_per_token (int): Number of contiguous segments per token.
        in_start_offset (int): Input offset for segmented normalization.
        out_start_offset (int): Output offset for segmented normalization.

    Returns:
        None: `out` is written in place.
    """

def rmsnorm(
    rt: Runtime,
    in_: torch.Tensor,
    norm: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
    cnt_per_token: int = 1,
    in_start_offset: int = 0,
    out_start_offset: int = 0,
    variance: Optional[torch.Tensor] = None,
) -> None:
    """Apply RMSNorm with optional offsets.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[tokens, dim]``, fp16 or bf16.
        norm (torch.Tensor): RMSNorm weight tensor, shape ``[norm_dim]``, model dtype.
            Optional (empty skips the affine term).
        out (torch.Tensor): Output tensor, shape ``[tokens, dim]`` (or a stepped variant),
            fp16/bf16 (same as in) or fp32.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.
        cnt_per_token (int): Number of contiguous segments per token.
        in_start_offset (int): Input offset for segmented normalization.
        out_start_offset (int): Output offset for segmented normalization.
        variance (Optional[torch.Tensor]): Optional output tensor for variance values, shape
            ``[tokens, 1]``, fp32. Only the first segment's variance is written per row.

    Returns:
        None: `out` is written in place.
    """
    ...

def rmsnorm_with_bias(
    rt: Runtime,
    in_: torch.Tensor,
    norm: torch.Tensor,
    norm_bias: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
    cnt_per_token: int = 1,
    in_start_offset: int = 0,
    out_start_offset: int = 0,
) -> None:
    """Apply RMSNorm with optional offsets.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[tokens, dim]``, fp16 or bf16.
        norm (torch.Tensor): RMSNorm weight tensor, shape ``[norm_dim]``, model dtype.
        norm_bias (torch.Tensor): RMSNorm Bias tensor, shape ``[norm_dim]``, model dtype.
        out (torch.Tensor): Output tensor, shape ``[tokens, dim]``, fp16/bf16 or fp32.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.
        cnt_per_token (int): Number of contiguous segments per token.
        in_start_offset (int): Input offset for segmented normalization.
        out_start_offset (int): Output offset for segmented normalization.

    Returns:
        None: `out` is written in place.
    """
    ...

def layernorm(
    rt: Runtime,
    in_: torch.Tensor,
    norm: torch.Tensor,
    norm_bias: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int,
) -> None:
    """Apply LayerNorm with learned weight and bias.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[tokens, dim]``, fp16 or bf16.
        norm (torch.Tensor): LayerNorm weight tensor, shape ``[norm_dim]``, model dtype.
        norm_bias (torch.Tensor): LayerNorm bias tensor, shape ``[norm_dim]``, model dtype.
        out (torch.Tensor): Output tensor, shape ``[tokens, dim]``, same dtype as ``in_``.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width.

    Returns:
        None: `out` is written in place.
    """
    ...

def l2norm(
    rt: Runtime,
    in_: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
) -> None:
    """Apply L2 Norm.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[tokens, dim]``, fp16 or bf16.
        out (torch.Tensor): Output tensor, same shape, fp16/bf16 (same as in) or fp32.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.

    Returns:
        None: `out` is written in place.
    """
    ...

def add_bias(rt: Runtime, in_: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Add bias tensor to input tensor.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[m, n]``, fp32/fp16/bf16 (all three same).
        weight (torch.Tensor): Bias tensor, shape ``[n]`` (broadcast per column), same dtype.
        out (torch.Tensor): Output tensor, shape ``[m, n]``, same dtype.

    Returns:
        None: `out` is written in place.
    """
    ...

def silu_and_mul(
    rt: Runtime,
    in_: torch.Tensor,
    out: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> None:
    """Apply SiLU and gated multiply.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[m, 2n]`` (last dim = 2 x out last dim),
            fp16/bf16/fp32 (same as out).
        out (torch.Tensor): Output tensor, shape ``[m, n]``, same dtype as ``in_``.
        swiglu_limit (float, default=0.0): Clamp limit for the swiglu variant
            (``0`` disables clamping).

    Returns:
        None: `out` is written in place.
    """
    ...

def sigmoid_gate_mul(rt: Runtime, attn: torch.Tensor, gate: torch.Tensor, out: torch.Tensor) -> None:
    """Compute out = attn * sigmoid(gate).

    Args:
        rt (Runtime): Native runtime handle.
        attn (torch.Tensor): Attention output, shape [num_tokens, dim], fp16 or bf16.
        gate (torch.Tensor): Gate logits, shape [num_tokens, dim] (elementwise) or
            [num_tokens, 1] (broadcast per token), same dtype.
        out (torch.Tensor): Output tensor, shape [num_tokens, dim]. May alias `attn`.

    Returns:
        None: `out` is written in place.
    """
    ...

def rope_and_cache(
    rt: Runtime,
    inout: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: torch.Tensor,
    cosin: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rot_dim: int,
    block_size: int,
    is_neox: bool,
    mrope_mask_h: int = 0,
    mrope_mask_w: int = 0,
) -> None:
    """Apply RoPE transform and update KV cache.

    Args:
        rt (Runtime): Native runtime handle.
        inout (torch.Tensor): Input/output packed QKV tensor, shape
            ``[tokens, (n_heads + 2*n_kv_heads)*head_dim]``, layout [Q|K|V], fp16/bf16
            (must match caches/cossin). RoPE is applied in place.
        k_cache (torch.Tensor): Paged key cache, shape
            ``[num_blocks, block_size, n_kv_heads, head_dim]``, same dtype.
        v_cache (torch.Tensor): Paged value cache, shape
            ``[num_blocks, block_size, n_kv_heads, head_dim]``, same dtype.
        position (torch.Tensor): Per-token position indices, shape ``[tokens]``, int64
            (mRoPE: flattened ``[3, tokens]`` int64 — base, height, width planes).
        cosin (torch.Tensor): Rotary cosine/sine table, shape ``[max_position, rot_dim]``,
            same dtype as ``inout``; each row is ``[cos | sin]`` concatenated
            (``cos`` at ``position*rot_dim``, ``sin`` right after), indexed by each
            token's ``position`` value, so it must cover the full position range.
        slot_mapping (torch.Tensor): Per-token paged-cache slots, shape ``[tokens]``, int32
            (slot = block_id*block_size + offset).
        n_heads (int): Global number of query heads (the host divides by tp size to get
            this rank's local head count).
        n_kv_heads (int): Global number of KV heads (divided by tp size on the host).
        head_dim (int): Head dimension.
        rot_dim (int): Rotary dimension.
        block_size (int): KV cache block size.
        is_neox (bool): Whether to use NeoX rotary layout (gptj style not supported).
        mrope_mask_h (int): Optional mRoPE height mask.
        mrope_mask_w (int): Optional mRoPE width mask.

    Returns:
        None: Inputs/caches are updated in place.
    """
    ...

def attention(
    rt: Runtime,
    qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    batch: int,
    enable_flash_attention: bool = False,
    tile_size_of_cached_kv: int = 8192,
) -> None:
    """Run paged attention for cached KV tensors.

    Args:
        rt (Runtime): Native runtime handle.
        qkv (torch.Tensor): Packed QKV tensor, shape
            ``[tokens, (n_heads + 2*n_kv_heads)*head_dim]``, fp16 or bf16 (must match
            caches/output).
        k_cache (torch.Tensor): Paged key cache, shape
            ``[num_blocks, block_size, n_kv_heads, head_dim]``, same dtype.
        v_cache (torch.Tensor): Paged value cache, same layout/dtype as ``k_cache``.
        output (torch.Tensor): Attention output tensor, shape
            ``[tokens, n_heads*head_dim]``, same dtype.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths, shape ``[batch(+1)]``,
            int32 device.
        lens (torch.Tensor): Current token lengths, shape ``[batch]``, int32 device.
        cached_lens (torch.Tensor): Cached token lengths, shape ``[batch]``, int32 device.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of query heads.
        n_kv_heads (int): Number of KV heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size.
        batch (int): Batch size.
        enable_flash_attention (bool): Whether to use flash attention kernels.
        tile_size_of_cached_kv (int): Tile size for cached KV in flash attention.

    Returns:
        None: `output` is written in place.
    """
    ...

def add_and_rmsnorm(
    rt: Runtime,
    in_: torch.Tensor,
    add_in_out: torch.Tensor,
    norm: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
) -> None:
    """Residual add followed by RMSNorm.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Residual input tensor, shape ``[m, dim]``, fp16 or bf16.
        add_in_out (torch.Tensor): In/out tensor for residual accumulation, shape
            ``[m, dim]``, same dtype (residual is accumulated in place).
        norm (torch.Tensor): RMSNorm weight tensor, shape ``[dim]``, model dtype.
        out (torch.Tensor): Output tensor, shape ``[m, dim]``, same dtype.
        norm_eps (float): Numerical epsilon used in normalization.

    Returns:
        None: `add_in_out`/`out` are updated in place per kernel behavior.
    """
    ...

def softmax_topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    out_weights: torch.Tensor,
    out_routing: torch.Tensor,
    top_k: int,
    norm_top_k_prob: bool,
) -> None:
    """Compute top-k routing with softmax scores.

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Routing score tensor, shape ``[tokens, n_routed_experts]``,
            fp32 or bf16.
        indices (torch.Tensor): Identity helper, shape ``[n_routed_experts]``, int32
            (``0..N-1``, sort-key payload).
        out_weights (torch.Tensor): Output SPARSE top-k weight tensor, shape
            ``[tokens, n_routed_experts]``, same dtype as ``scores`` (only selected slots
            hold nonzero values).
        out_routing (torch.Tensor): Output routing bitmap, shape
            ``[tokens, ceil(n_routed_experts/64)]``, int64 (one bit per expert).
        top_k (int): Number of experts selected per token.
        norm_top_k_prob (bool): Whether to normalize selected probabilities.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def sigmoid_topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    out_weights: torch.Tensor,
    out_routing: torch.Tensor,
    n_group: int,
    n_topk_group: int,
    top_k: int,
    norm_top_k_prob: bool,
) -> None:
    """Compute top-k routing with sigmoid scores.

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Routing score tensor, shape ``[tokens, n_routed_experts]``,
            fp32 or bf16.
        indices (torch.Tensor): Identity helper, shape ``[n_routed_experts]``, int32
            (``0..N-1``, sort-key payload).
        bias (torch.Tensor): Per-expert e_score_correction_bias, shape
            ``[n_routed_experts]``, fp32.
        scale (float): Scale factor applied to scores.
        out_weights (torch.Tensor): Output top-k weight tensor, shape
            ``[tokens, n_routed_experts]``, same dtype as ``scores`` (sparse, selected
            slots only).
        out_routing (torch.Tensor): Output routing bitmap, shape
            ``[tokens, ceil(n_routed_experts/64)]``, int64 (one bit per expert).
        n_group (int): Number of routing groups.
        n_topk_group (int): Number of groups participating in top-k.
        top_k (int): Number of experts selected per token.
        norm_top_k_prob (bool): Whether to normalize selected probabilities.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def sqrtsoftplus_hash_topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    bias: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    out_weights: torch.Tensor,
    routing_map: torch.Tensor,
    scale: float,
    top_k: int,
    hash: bool,
) -> None:
    """Compute V4 MoE gate routing (the part below the matmul), sqrtsoftplus only.

    The caller feeds pre-activation ``scores`` (the ``F.linear`` output); this op does
    the sqrtsoftplus activation, top-k (or hash lookup), gather, normalize, scale.

    Two outputs: a sparse ``[M, n_routed_experts]`` weight row (only the top-k expert
    slots hold a value, rest 0) and a ``[M, n_routed_experts]`` BIT1 routing bitmap (one
    bit per selected expert). The dense top-k indices are internal and NOT a GM output.

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Pre-activation routing scores ``[M, n_routed_experts]``
            (bf16 or fp32).
        indices (torch.Tensor): Identity helper ``[n_routed_experts]`` int32 (``0..N-1``),
            used as the vbitsort sort-key payload in the non-hash top-k path.
        bias (torch.Tensor): Per-expert bias ``[n_routed_experts]`` fp32. Pass an empty
            tensor on hash layers (bias is unused there).
        input_ids (torch.Tensor): Token ids ``[M]`` int32. Only read when ``hash`` is True;
            pass an empty tensor on non-hash layers.
        tid2eid (torch.Tensor): Token-id->expert lookup table
            ``[vocab_size, n_activated_experts]`` int32. Only read when ``hash`` is True;
            pass an empty tensor on non-hash layers.
        scale (float): ``route_scale`` multiplier applied after normalization.
        out_weights (torch.Tensor): Output SPARSE weights ``[M, n_routed_experts]`` (same
            dtype as ``scores``) — only the top-k selected experts hold a nonzero weight,
            the rest are 0.
        routing_map (torch.Tensor): Output routing bitmap ``[M, n_routed_experts]`` uint32
            (BIT1, one bit per expert; set for the top-k selected experts).
        top_k (int): Number of experts selected per token (``n_activated_experts``).
        hash (bool): If True, route via ``tid2eid[input_ids]`` (first ``n_hash_layers``);
            else route via ``scores.topk``.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    outIndices: torch.Tensor,
    query_lens: torch.Tensor,
    cached_lens: torch.Tensor,
    k: int,
) -> None:
    """Select top-k elements by scores in batches

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Score tensor, shape ``[batch, seq_len]``, bf16 or fp32.
            No-op when ``scores.shape[1] <= k``.
        indices (torch.Tensor): Identity index table, shape ``[max_seq_len]`` (1-D,
            shared by all batch rows), int32 (``0..max_seq_len-1``).
        outIndices (torch.Tensor): Output top-k index tensor, shape
            ``[sum(query_lens), k]`` int32 — one row per query token across the
            whole batch, written via a global row counter (``[batch, k]`` only
            when every query length is 1).
        query_lens (torch.Tensor): Vector of query lengths for each batch, shape ``[batch]``,
            int32 device (batch is taken from this tensor's shape[0]).
        cached_lens (torch.Tensor): Vector of cached KV lengths for each batch, shape
            ``[batch]``, int32 device.
        k (int): Number of experts selected per token (must be exactly 2048).

    Returns:
        None: Output tensors are written in place.
    """
    ...

def cast_up(rt: Runtime, in_: torch.Tensor, out: torch.Tensor) -> None:
    """Cast tensor values to a higher-precision type.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, bf16, any shape.
        out (torch.Tensor): Output tensor, fp32, same numel as ``in_``.

    Returns:
        None: `out` is written in place.
    """
    ...

def permutation(
    rt: Runtime,
    in_: torch.Tensor,
    routing: torch.Tensor,
    start: int,
    end: int,
    out: torch.Tensor,
    unp_idx: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Permute token rows into expert-local layout.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input token tensor, shape ``[tokens, hidden]``, bf16 (the
            only dtype the kernel supports; rows are moved as raw 2-byte elements).
        routing (torch.Tensor): Routing bitmap (BIT1 packed, e.g. int64), shape
            ``[tokens, ceil(n_experts/64)]``.
        start (int): Start expert index.
        end (int): End expert index.
        out (torch.Tensor): Permuted output tensor, shape ``[permuted_tokens, hidden]``,
            same dtype as ``in_``.
        unp_idx (torch.Tensor): Unpermutation index grid output, shape
            ``[n_experts, tokens + 1]``, int32: per expert ``e``, column ``t`` holds the
            row of token ``t`` within expert ``e``'s permuted segment; the final column
            holds each expert's segment start offset (exclusive prefix sum of counts);
            ``[0, 0]`` is overwritten with the total permuted token count (consumed by
            :func:`unpermutation`).
        counts (torch.Tensor): Per-expert count output, shape ``[n_routed_experts]``,
            int32 — indexed by absolute expert id (must be full-width even when
            ``start > 0``; experts outside ``[start, end)`` are written 0).

    Returns:
        None: Output tensors are written in place.
    """
    ...

def unpermutation(
    rt: Runtime,
    in_: torch.Tensor,
    routing: torch.Tensor,
    weights: torch.Tensor,
    start: int,
    end: int,
    out: torch.Tensor,
    unp_idx: torch.Tensor,
) -> None:
    """Restore original row order after expert routing.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Permuted input tensor, shape
            ``[max_expert_sorted, hidden]`` (the permuted buffer's capacity; rows are
            gathered via ``unp_idx``, so this need not equal the original token count),
            bf16.
        routing (torch.Tensor): Routing bitmap (BIT1 packed, e.g. int64), shape
            ``[tokens, ceil(n_experts/64)]``.
        weights (torch.Tensor): Per-token per-expert routing weight map, shape
            ``[tokens, n_experts]`` (full expert width), bf16 (when in/out are bf16) or
            fp32; entries for non-selected experts are ignored.
        start (int): Start expert index.
        end (int): End expert index.
        out (torch.Tensor): Unpermuted output tensor, shape ``[orig_tokens, hidden]``, bf16.
        unp_idx (torch.Tensor): Unpermutation index grid (from :func:`permutation`), shape
            ``[n_experts, tokens + 1]``, int32.

    Returns:
        None: `out` is written in place.
    """
    ...

def group_matmul(
    rt: Runtime,
    in_: torch.Tensor,
    weights: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    counts: torch.Tensor,
    start: int,
    end: int,
    out_dim: int,
    in_dim: int,
    output: torch.Tensor,
    weight_nz: bool,
    transpose: bool,
) -> None:
    """Run grouped matmul for per-expert weights.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor, shape ``[sum_tokens, in_dim]``. Dtype combos
            (in, weight, out): bf16/bf16/bf16, fp16/fp16/fp16, fp32/fp32/fp32
            (transpose=False), int8/int8/fp16, int4/int4/fp16 (int32-packed int4
            activations/weights are auto-viewed as int4).
        weights (Sequence[torch.Tensor]): Per-group (per-expert) 2D weight tensors, each
            ``[out_dim, in_dim]`` (or transposed layout when ``transpose=True``); must
            share dtype.
        scales (Sequence[torch.Tensor]): Optional per-group deq-scale tensors (uint64
            TF32-packed fp32 pairs), one per group; length must equal ``counts.size(0)``
            when provided.
        counts (torch.Tensor): Per-group token count tensor, shape ``[num_experts]``, int32.
        start (int): Start group index.
        end (int): End group index.
        out_dim (int): Output dimension.
        in_dim (int): Input dimension.
        output (torch.Tensor): Output tensor, shape ``[sum_tokens, out_dim]``.
        weight_nz (bool): Whether weight tensors use NZ layout.
        transpose (bool): Whether grouped weights are transposed.

    Returns:
        None: `output` is written in place.
    """
    ...

def softmax(rt: Runtime, x: torch.Tensor, calc_len: int, is_long: bool) -> None:
    """Apply softmax over the configured dimension.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input/output tensor, shape ``[m, n]``, fp16 or bf16 (in-place).
        calc_len (int): Base softmax length; row ``i`` computes ``calc_len + i`` elements
            (causal/row-dependent), clamped to ``n``. Pass ``calc_len == n`` for full-width
            softmax on a single row (``m == 1``).
        is_long (bool): Whether to use the long-sequence kernel path.

    Returns:
        None: `x` is updated in place.
    """
    ...

def rope_complex(
    rt: Runtime,
    n_local_heads: int,
    step_dim: int,
    rope_dim: int,
    input_with_r: torch.Tensor,
    freqs: torch.Tensor,
    position: torch.Tensor,
    output: torch.Tensor,
    inverse: bool = False,
    out_interleaved: bool = False,
) -> None:
    """Apply complex-domain rotary embedding helper.

    Args:
        rt (Runtime): Native runtime handle.
        n_local_heads (int): Number of local heads.
        step_dim (int): Per-step hidden dimension.
        rope_dim (int): Rotary dimension.
        input_with_r (torch.Tensor): Input tensor with real/imag layout, shape
            ``[tokens, n_local_heads*step_dim]``, fp16 or bf16.
        freqs (torch.Tensor): Rotary frequency table, shape ``[max_position, rope_dim/2]``
            complex64 (or equivalent real view ``[max_position, rope_dim]`` fp32). The
            kernel always reads it as float32 regardless of model dtype. Indexed by
            each token's ``position`` value
            (``freqs_ptr + position[token] * rope_dim``), so it must cover the full
            position range.
        position (torch.Tensor): Per-token position ids, shape ``[tokens]``, int64.
        output (torch.Tensor): Output tensor, rope-only slice, shape
            ``[tokens, n_local_heads*rope_dim]`` (out step = rope_dim), model dtype.
        inverse (bool): If True, apply the conjugate (reverse) rotation.
        out_interleaved (bool): If True, write the rope result interleaved
            ``[r0,i0,r1,i1,...]`` (matches torch ``view_as_real().flatten``);
            otherwise write the deinterleaved half layout
            ``[r0..r(half-1) | i0..i(half-1)]`` (MLA/DSA kv-cache convention).

    Returns:
        None: Output is produced in place according to kernel contract.
    """
    ...

def mla_prepare(
    rt: Runtime,
    attn_qkvc: torch.Tensor,
    q_norm: torch.Tensor,
    q_norm_bias: torch.Tensor,
    attn_norm_qc: torch.Tensor,
    kv_norm: torch.Tensor,
    kv_norm_bias: torch.Tensor,
    attn_norm_kvc: torch.Tensor,
    freqs: torch.Tensor,
    position: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    rope_head_dim: int,
    block_size: int,
    k_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_eps: float,
) -> None:
    """Fused MLA prepare: two RMSNorm passes followed by rope_complex_and_cache.

    Args:
        rt (Runtime): Native runtime handle.
        attn_qkvc (torch.Tensor): Concatenated [q_lora_rank | kv_lora_rank | rope_head_dim]
            per token, shape ``[tokens, q_lora_rank + kv_lora_rank + rope_head_dim]``,
            fp16 or bf16 (rope slice written in place).
        q_norm (torch.Tensor): RMSNorm weight for the q-lora slice, shape
            ``[q_lora_rank]``, model dtype.
        q_norm_bias (torch.Tensor): RMSNorm bias for the q-lora slice, shape
            ``[q_lora_rank]``, model dtype.
        attn_norm_qc (torch.Tensor): Output RMSNormed q-lora slice, shape
            ``[tokens, q_lora_rank]``, model dtype.
        kv_norm (torch.Tensor): RMSNorm weight for the kv-lora slice, shape
            ``[kv_lora_rank]``, model dtype.
        kv_norm_bias (torch.Tensor): RMSNorm bias for the kv-lora slice, shape
            ``[kv_lora_rank]``, model dtype.
        attn_norm_kvc (torch.Tensor): Output RMSNormed kv-lora slice, shape
            ``[tokens, kv_lora_rank]``, model dtype; also used as the `key` written into
            k_cache.
        freqs (torch.Tensor): Precomputed rotary freqs_cis (TTTWWW layout), shape
            ``[max_position, rope_head_dim/2]`` complex64 (or equivalent real view
            ``[max_position, rope_head_dim]`` fp32). The kernel always reads it as
            float32 regardless of model dtype; indexed by each
            token's ``position`` value, so it must cover the full position range.
        position (torch.Tensor): Per-token position ids (int64), shape ``[tokens]``.
        q_lora_rank (int): q-lora rank dimension.
        kv_lora_rank (int): kv-lora rank dimension.
        rope_head_dim (int): Rotary head dimension.
        block_size (int): Paged kv-cache block size; 0 disables cache writes.
        k_cache (torch.Tensor): Output paged k-cache, shape
            ``[num_blocks, block_size, 1, kv_lora_rank]``, model dtype.
        pe_cache (torch.Tensor): Output paged pe-cache (RoPE'd rope slice), shape
            ``[num_blocks, block_size, 1, rope_head_dim]``, model dtype.
        slot_mapping (torch.Tensor): Per-token paged-cache slot mapping, shape
            ``[tokens]``, int32.
        norm_eps (float): RMSNorm epsilon.

    Returns:
        None: Outputs are written in place into attn_norm_qc, attn_norm_kvc, k_cache, pe_cache, and the rope slice of attn_qkvc.
    """
    ...

def indexer_prepare(
    rt: Runtime,
    kw: torch.Tensor,
    k_norm: torch.Tensor,
    k_norm_bias: torch.Tensor,
    freqs: torch.Tensor,
    position: torch.Tensor,
    index_head_dim: int,
    index_n_heads: int,
    rope_head_dim: int,
    block_size: int,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_eps: float,
    q: torch.Tensor,
    scale: float,
    top_k: int,
    is_long: bool,
) -> None:
    """Fused DSA indexer prepare: LayerNorm + rope_complex_and_cache, optional rope_complex(q) + muls(kw).

    Always runs:
      * LayerNorm over ``kw[:, :index_head_dim]`` (in place).
      * rope_complex_and_cache on ``kw[:, :index_head_dim+index_n_heads]`` writing the
        rotary slice of ``kw`` and scattering ``index_head_dim`` slice into ``index_k_cache``.

    When ``is_long`` is true, additionally runs:
      * rope_complex on ``q`` (``index_n_heads`` heads, each ``index_head_dim`` wide), in place.
      * muls on ``kw[:, index_head_dim:index_head_dim+index_n_heads]`` by ``scale`` (in place).

    Args:
        rt (Runtime): Native runtime handle.
        kw (torch.Tensor): ``[token_num, index_head_dim + index_n_heads]`` per token, fp16
            or bf16 (norm/rope/muls slices written in place).
        k_norm (torch.Tensor): LayerNorm weight ``[index_head_dim]``, model dtype or fp32.
        k_norm_bias (torch.Tensor): LayerNorm bias ``[index_head_dim]``, model dtype or fp32.
        freqs (torch.Tensor): Precomputed rotary freqs_cis (TTTWWW layout), shape
            ``[max_position, rope_head_dim/2]`` complex64 (or equivalent real view
            ``[max_position, rope_head_dim]`` fp32). The kernel always reads it as
            float32 regardless of model dtype; indexed by each
            token's ``position`` value, so it must cover the full position range.
        position (torch.Tensor): Per-token position ids (int64), shape ``[token_num]``.
        index_head_dim (int): Indexer head dimension.
        index_n_heads (int): Indexer head count (also the muls width).
        rope_head_dim (int): Rotary head dimension.
        block_size (int): Paged k-cache block size; 0 disables cache writes.
        index_k_cache (torch.Tensor): Output paged indexer k-cache, shape
            ``[num_blocks, block_size, 1, index_head_dim]``, model dtype.
        slot_mapping (torch.Tensor): Per-token paged-cache slot mapping (int32), shape
            ``[token_num]``.
        norm_eps (float): LayerNorm epsilon.
        q (torch.Tensor): ``[token_num, index_n_heads * index_head_dim]`` query tensor, model
            dtype; only touched when ``is_long`` is true. The rotary slice is written in place.
        scale (float): Scalar applied to ``kw[:, index_head_dim:]`` when ``is_long`` is true.
        top_k (int): Number of top-k tokens to select in the indexer/sparse attention.
        is_long (bool): Whether to run the rope_complex(q) + muls(kw) tail.

    Returns:
        None: ``kw`` (norm+rope slices, optional muls slice), ``index_k_cache``, and (when
        ``is_long``) the rotary slice of ``q`` are written in place.
    """
    ...

def quant(
    rt: Runtime,
    x: torch.Tensor,
    scale_reciprocal: torch.Tensor,
    offset: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Quantize tensor using explicit reciprocal scale and offset.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input tensor, shape ``[m, n]``, bf16.
        scale_reciprocal (torch.Tensor): Reciprocal scale tensor, shape ``[n]``
            (per-column), bf16 (read as bf16 by the kernel; fp32 is not supported).
        offset (torch.Tensor): Quantization offset tensor, shape ``[n]``, bf16
            (same layout and dtype as the scale).
        out (torch.Tensor): Quantized output tensor, shape ``[m, n]``, int8
            (``int8(x*scale + offset)``).

    Returns:
        None: `out` is written in place.
    """
    ...

def quant_dynamic(rt: Runtime, x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    """Dynamically quantize tensor values and emit scale.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input tensor, shape ``[m, n]``, bf16.
        scale (torch.Tensor): Output dynamic per-token scale, shape ``[m]``, fp32.
        out (torch.Tensor): Quantized output tensor, shape ``[m, n]``, int8.

    Returns:
        None: `scale` and `out` are written in place.
    """
    ...

def matmul_dequant(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    deq_scale: torch.Tensor,
    z: torch.Tensor,
    weight_nz: bool = False,
    transpose: bool = False,
) -> None:
    """Matmul on quantized weights with dequantization.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix, shape ``[m, k]``, int8.
        y (torch.Tensor): Quantized right matrix, shape ``[n, k]`` (transpose=False) or
            ``[k, n]`` (transpose=True), int8.
        bias (torch.Tensor): Quantization bias, shape ``[n]``, int32 (optional).
        deq_scale (torch.Tensor): Weight dequantization scale, flat ``[2*n]``, fp32
            (uint64 TF32-packed pairs).
        z (torch.Tensor): Output matrix, shape ``[m, n]``, fp16.
        weight_nz (bool): Whether `y` uses NZ layout.
        transpose (bool): Whether to transpose the right matrix.

    Returns:
        None: `z` is written in place.
    """
    ...

def msd_merge_dequant(
    rt: Runtime,
    y_merged: torch.Tensor,
    scale_biases: Sequence[torch.Tensor],
    counts: torch.Tensor,
    per_token_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Merge MSD (W4A8) row-merged int8 result and per-token dequantize.

    Used by the MSD W4A8 MoE post-stage to turn a mid-stage row-merged result
    into the final BF16 output. ``y_merged`` packs two halves of a 4-bit weight
    matmul in an interleaved row layout: token ``r``'s low-nibble row is
    ``2*r`` and its high-nibble row is ``2*r + 1`` (each in int8 form). The
    kernel reconstructs the full value ``Y_high * 16 + Y_low``, compensates
    the low-nibble ``-8`` bias, adds a per-column ``scale_bias``, and scales
    by a per-token ``per_token_scale``::

        Y = (Y_high * 16 + Y_low + scale_bias) * perTokenScale

    Args:
        rt (Runtime): Native runtime handle.
        y_merged (torch.Tensor): Row-merged int8 mid-stage result, shape
            ``[2*m, n]`` (float16). Row ``2*r`` is the low nibble and row
            ``2*r + 1`` the high nibble of token ``r``.
        counts (torch.Tensor): Per-expert merged row counts, shape
            ``[num_experts]``, int32 (read as uint32 by the kernel).
        scale_bias (torch.Tensor): Per-column bias added after merge, shape
            ``[n]`` (float32).
        per_token_scale (torch.Tensor): Per-token dequantization scale, shape
            ``[m]`` (float32).
        out (torch.Tensor): Output tensor, shape ``[m, n]`` (bfloat16).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If dtypes/shapes are unsupported
            (requires ``y_merged`` float16, ``scale_bias``/``per_token_scale``
            float32, ``out`` bfloat16, and ``y_merged.shape[0]`` even).
    """
    ...

def dequant(rt: Runtime, in_: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, has_scale: bool) -> None:
    """Dequantize tensor values into output precision.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Quantized input tensor, shape ``[m, n]``, fp16.
        scale (torch.Tensor): Scale tensor, shape ``[m]``, fp32, per-token (one scale
            per row; the kernel indexes it by row number; only read when ``has_scale``
            is true).
        out (torch.Tensor): Dequantized output tensor, shape ``[m, n]``, bf16.
        has_scale (bool): Whether scale should be applied.

    Returns:
        None: `out` is written in place.
    """
    ...

def mla_v2(
    rt: Runtime,
    q_with_qr: torch.Tensor,
    qr: torch.Tensor,
    k_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    wuk_t: torch.Tensor,
    wuv: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    rope_head_dim: int,
    nope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    block_size: int,
    batch: int,
    scale: float,
    topk_indices: torch.Tensor,
    top_k: int = 0,
    nz: bool = False,
    enable_flash_attention: bool = False,
    tile_size_of_cached_kv: int = 8192,
    dense: bool = False,
) -> None:
    """Run MLA v2 path as three kernels: wuk einsum + mla_v2 attention + wuv einsum.

    ``dense=False`` (default): the mla_v2 attention kernel walks the paged KV
    cache via ``block_tables`` (with optional top-k token selection when
    ``top_k > 0``); ``enable_flash_attention`` selects the flash variant for
    long sequences.

    ``dense=True``: gather the top-k tokens selected by ``topk_indices`` into
    a contiguous per-batch dense cache (via :func:`gather_sparse_kv_cache`,
    called internally), then run mla_v2 on it. Requires ``top_k > 0`` and
    query_len == 1 per batch (decode); ``enable_flash_attention`` and
    ``block_tables`` are unused in this mode.

    Args:
        rt (Runtime): Native runtime handle.
        q_with_qr (torch.Tensor): Query tensor with rotary components, shape
            (total_query_tokens, n_heads, nope_head_dim + rope_head_dim), bf16.
        qr (torch.Tensor): Pre-rotated q_rope slice, contiguous, shape
            (total_query_tokens, n_heads, rope_head_dim), bf16.
        k_cache (torch.Tensor): Paged KV cache (kv_lora_rank slice), shape
            ``[num_blocks, block_size, 1, kv_lora_rank]``, bf16; in dense
            mode the source paged cache the top-k tokens are gathered from.
        pe_cache (torch.Tensor): Paged RoPE key cache (rope_head_dim slice), shape
            ``[num_blocks, block_size, 1, rope_head_dim]``, bf16.
        wuk_t (torch.Tensor): MLA W_UK^T weight, shape
            (n_heads, nope_head_dim, kv_lora_rank), bf16.
        wuv (torch.Tensor): MLA W_UV weight, shape
            (n_heads, kv_lora_rank, v_head_dim), bf16.
        output (torch.Tensor): Output tensor (v_head_dim), shape
            ``[total_query_tokens, n_heads*v_head_dim]``, bf16; written in place.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths, shape ``[batch(+1)]``,
            int32 device.
        lens (torch.Tensor): Current token lengths, shape ``[batch]``, int32 device.
        cached_lens (torch.Tensor): Cached token lengths, shape ``[batch]``, int32 device.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of query heads.
        rope_head_dim (int): Rotary head dimension.
        nope_head_dim (int): Non-rotary head dimension.
        v_head_dim (int): Value head dimension.
        kv_lora_rank (int): KV LoRA rank.
        block_size (int): KV block size.
        batch (int): Batch size.
        scale (float): Attention scaling factor.
        topk_indices (torch.Tensor): Top-k indices tensor for sparse attention, shape
            ``[total_query_tokens, top_k]`` int32, one row per query token
            (``[batch, top_k]`` in decode); may be empty when ``top_k == 0``.
        top_k (int): Number of top-k indices; 0 disables sparse attention in
            paged mode. In dense mode this is the dense cache length
            (``index_topk``) and must be > 0.
        nz (bool): Whether to use nz weights.
        enable_flash_attention (bool): Whether to use the flash MLA v2 kernel
            (paged mode only; ignored when ``dense=True``).
        tile_size_of_cached_kv (int): Tile size for cached KV in flash MLA v2.
        dense (bool): Whether to gather a contiguous dense KV cache (via
            ``topk_indices``) instead of walking the paged block-table layout.

    Returns:
        None: `output` is written in place.
    """
    ...

def gather_sparse_kv_cache(
    rt: Runtime,
    k_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    topk_indices: torch.Tensor,
    query_lens: torch.Tensor,
    cached_lens: torch.Tensor,
    k_dense_cache: torch.Tensor,
    pe_dense_cache: torch.Tensor,
    batch: int,
    index_topk: int,
    block_size: int,
    kv_lora_rank: int,
    rope_head_dim: int,
    kv_heads: int = 1,
) -> None:
    """Gather sparse KV cache into a contiguous dense cache per batch.

    For each ``(b, i)`` pair, reads the token index ``tok = topk_indices[b, i]``
    (token-index semantics), maps it to the physical block via
    ``block_tables[b * max_num_blocks + tok // block_size]`` and offset
    ``tok % block_size``, and copies one row from paged ``k_cache`` /
    ``pe_cache`` into the contiguous ``k_dense_cache`` / ``pe_dense_cache``.

    Only the first ``min(query_lens[b] + cached_lens[b], index_topk)`` slots per
    batch are written; slots beyond that (topk_indices padding tail) are
    **skipped** (left as-is, typically zero-initialized by the caller).
    :func:`mla_v2` with ``dense=True`` reads only those valid slots and masks
    the rest in softmax, so the skipped slots do not affect output.

    Used by the decode + DSA long-sequence path to feed a contiguous dense cache
    to :func:`mla_v2` (``dense=True``) instead of the paged layout.

    Args:
        rt (Runtime): Native runtime handle.
        k_cache (torch.Tensor): Paged KV cache (kv_lora_rank slice), shape
            (kvcache_block_num, block_size, kv_heads, kv_lora_rank), bf16.
        pe_cache (torch.Tensor): Paged RoPE key cache (rope_head_dim slice),
            shape (kvcache_block_num, block_size, kv_heads, rope_head_dim), bf16.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        topk_indices (torch.Tensor): Top-k token indices from IndexerTopK, shape
            (batch, index_topk), dtype int32.
        query_lens (torch.Tensor): Per-batch current query lengths, shape
            (batch,), dtype int32. Decode = 1 per batch.
        cached_lens (torch.Tensor): Per-batch cached token lengths, shape
            (batch,), dtype int32. Used with query_lens to derive the valid
            slot count per batch.
        k_dense_cache (torch.Tensor): Output contiguous K cache, shape
            (batch, index_topk, kv_heads, kv_lora_rank), bf16; written in place.
        pe_dense_cache (torch.Tensor): Output contiguous PE cache, shape
            (batch, index_topk, kv_heads, rope_head_dim), bf16; written in place.
        batch (int): Batch size.
        index_topk (int): Number of top-k tokens per batch (dense length).
        block_size (int): KV block size.
        kv_lora_rank (int): KV LoRA rank.
        rope_head_dim (int): Rotary head dimension.
        kv_heads (int): Number of KV heads (must be 1; defaults to 1 for MLA).

    Returns:
        None: `k_dense_cache` and `pe_dense_cache` are written in place.
    """
    ...

def cxa(
    rt: Runtime,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    compress_k_cache: torch.Tensor,
    swa_block_tables: torch.Tensor,
    compress_block_tables: torch.Tensor,
    swa_block_size: int,
    compress_block_size: int,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    n_heads: int,
    head_dim: int,
    scale: float,
    window_size: int,
    compress_ratio: int,
    index_topk: int,
    topk_indices: torch.Tensor,
) -> None:
    """Run the CXA (C4A and C128A) kernel for DeepSeek-V4.

    Fused sliding-window + compressed sparse attention. The attention score
    layout has stride ``window_size + kv_size``: the leading ``window_size``
    columns are the sliding-window (SWA) KV and the trailing ``kv_size``
    columns are the compressed KV. Softmax is computed over both segments
    jointly, with the per-head ``attn_sink`` bias folded into the denominator.

    * **Window causal mask** -- a query at position ``q`` may only attend to
      SWA tokens in ``[max(0, q - window_size + 1), q]``; positions beyond the
      currently generated length are masked to ``-inf``.
    * **Compress top-k mask** -- of the ``kv_size`` compressed tokens, only
      those referenced by ``topk_indices`` are kept; entries set to ``-1`` are
      masked to ``-inf`` (and their exp contributions zeroed).
    * **attn_sink** -- ``exp(attn_sink - row_max)`` is added to the softmax
      denominator (one learnable term per head).

    Args:
        rt (Runtime): Native runtime handle.
        q (torch.Tensor): Query, shape (total_query_tokens, n_heads, head_dim),
            dtype bfloat16.
        swa_k_cache (torch.Tensor): Paged sliding-window KV cache, shape
            (swa_block_num, swa_block_size, head_dim), dtype bfloat16.
        compress_k_cache (torch.Tensor): Paged compressed KV cache, shape
            (compress_block_num, compress_block_size, head_dim), dtype
            bfloat16. Unused when ``compress_ratio == 0`` (may be empty).
        swa_block_tables (torch.Tensor): Block table for the SWA cache, 1-D
            ``[batch * swa_max_num_blocks]`` or 2-D
            ``[batch, swa_max_num_blocks]`` int32.
        compress_block_tables (torch.Tensor): Block table for the compressed
            cache, 1-D or 2-D int32 (same convention as ``swa_block_tables``).
        swa_block_size (int): Block size of the SWA cache.
        compress_block_size (int): Block size of the compressed cache.
        attn_sink (torch.Tensor): Per-head attention sink bias, shape
            (n_heads,), dtype **float32**.
        output (torch.Tensor): Output tensor, shape
            (total_query_tokens, n_heads, head_dim), dtype bfloat16; written
            in place.
        batch (int): Batch size.
        query_start_loc (torch.Tensor): Prefix-sum of query lengths, shape
            (batch,), dtype int32.
        lens (torch.Tensor): Per-batch current query lengths, shape (batch,),
            dtype int32.
        cached_lens (torch.Tensor): Per-batch cached token lengths, shape
            (batch,), dtype int32.
        n_heads (int): Number of local query heads.
        head_dim (int): Head dimension.
        scale (float): Softmax scaling factor (``1 / sqrt(head_dim)``).
        window_size (int): Sliding-window size (SWA segment width).
        compress_ratio (int): Compression ratio; ``0`` disables the compressed
            segment (pure sliding-window attention).
        index_topk (int): Number of top-k compressed indices per query row;
            ``0`` disables the compress sparse path.
        topk_indices (torch.Tensor): Top-k compressed-token indices, shape
            (total_query_tokens, index_topk), dtype int32. ``-1`` marks a
            masked-out position.

    Returns:
        None: `output` is written in place.
    """
    ...

def indexer_scores(
    rt: Runtime,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    head_dim: int,
    block_size: int,
    batch: int,
) -> None:
    """Compute DSA indexer scores over cached keys.

    Args:
        rt (Runtime): Native runtime handle.
        q (torch.Tensor): Query tensor, shape ``[tokens, n_heads*head_dim]``, fp16 or bf16
            (must match k_cache/weight/scores).
        k_cache (torch.Tensor): Paged index-key cache, shape
            ``[num_blocks, block_size, 1, head_dim]``, same dtype.
        weight (torch.Tensor): Indexer weight tensor, shape ``[tokens, head_dim + n_heads]``,
            same dtype — the kernel strides rows by ``head_dim + n_heads`` and consumes
            only the trailing ``n_heads`` columns of each row.
        scores (torch.Tensor): Output score tensor, shape ``[tokens, ...]``, same dtype.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths, shape ``[batch(+1)]``,
            int32 device.
        lens (torch.Tensor): Current token lengths, shape ``[batch]``, int32 device.
        cached_lens (torch.Tensor): Cached token lengths, shape ``[batch]``, int32 device.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size.
        batch (int): Batch size.

    Returns:
        None: `scores` is written in place.
    """
    ...

def indexer_topk(
    rt: Runtime,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    head_dim: int,
    block_size: int,
    batch: int,
    top_k: int,
) -> None:
    """Fused DSA indexer scores + top-k selection over cached keys.

    Combines :func:`indexer_scores` and :func:`topk` into a single kernel
    launch with pingpong buffers. Scratch buffers (scores, last_topk, sync)
    are allocated internally by the runtime.

    Args:
        rt (Runtime): Native runtime handle.
        q (torch.Tensor): Query tensor ``[total_query_len, n_heads, head_dim]``, fp16/bf16
            (must match k_cache/weight).
        k_cache (torch.Tensor): Key cache tensor ``[max_num_block*batch, block_size,
        head_dim]``, same dtype.
        weight (torch.Tensor): Indexer weight tensor
            ``[total_query_len, head_dim + n_heads]`` (last ``n_heads`` columns are
            the indexer weights), same dtype.
        indices (torch.Tensor): Input index tensor ``[max_seq_len]`` (int32),
            pre-filled with ``0..max_seq_len-1``.
        topk_indices (torch.Tensor): Output top-k indices tensor
            ``[total_query_len, top_k]`` (int32).
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths, shape ``[batch(+1)]``,
            int32 device.
        lens (torch.Tensor): Current token lengths, shape ``[batch]``, int32 device.
        cached_lens (torch.Tensor): Cached token lengths, shape ``[batch]``, int32 device.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size (must be <= 128).
        batch (int): Batch size.
        top_k (int): Number of top-k indices to select (must be <= 2048).

    Returns:
        None: ``topk_indices`` is written in place.
    """
    ...

def muls(rt: Runtime, input: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """Multiply tensor by scalar and write to output.

    Args:
        rt (Runtime): Native runtime handle.
        input (torch.Tensor): Input tensor, 1D or 2D, fp16 or bf16.
        scale (float): Scalar multiplier.
        output (torch.Tensor): Output tensor, same shape and dtype as ``input``.

    Returns:
        None: `output` is written in place.
    """
    ...

def experts_counts_sum(
    rt: Runtime,
    experts_counts_input: torch.Tensor,
    tokens_per_epgroup: torch.Tensor,
    experts_counts_output: torch.Tensor,
    n_routed_experts: int,
) -> None:
    """Compute two reductions over a per-DP-rank per-expert token count matrix.

    Given an input ``experts_counts_input`` of shape ``[ep_size, n_routed_experts]``
    where row *i* is the dispatch count vector from DP rank *i*, the kernel writes:

    * ``tokens_per_epgroup[dp_idx, ep_id]`` — total tokens from DP rank ``dp_idx``
      destined for experts in EP group ``ep_id``.  Shape ``[ep_size, ep_size]``.
    * ``experts_counts_output[expert]`` — total tokens for expert ``expert``,
      summed across all DP ranks.  Shape ``[n_routed_experts]``.

    Args:
        rt (Runtime): Native runtime handle.
        experts_counts_input (torch.Tensor): Per-DP-rank per-expert count matrix
            ``[ep_size, n_routed_experts]`` (int32).
        tokens_per_epgroup (torch.Tensor): Output buffer for per-DP-rank
            per-EP-group token sums ``[ep_size, ep_size]`` (int32).
        experts_counts_output (torch.Tensor): Output buffer for per-expert total
            counts ``[n_routed_experts]`` (int32).
        n_routed_experts (int): Total number of routed experts.

    Returns:
        None: ``tokens_per_epgroup`` and ``experts_counts_output`` are written
        in place.
    """
    ...

def reorder_moe(
    rt: Runtime,
    in_: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    hidden_size: int,
    local_start: int,
    local_end: int,
    forward: bool,
) -> None:
    """Permute token rows between source-grouped and expert-grouped layouts.

    The kernel handles two directions based on ``forward``:

    * **forward=True**: source-grouped → expert-grouped.  Input tokens are
      grouped by source EP rank; output tokens are grouped by target expert
      index, ready for per-expert computation.
    * **forward=False**: expert-grouped → source-grouped.  The inverse
      permutation that restores the original source-grouped order.

    The ``counts`` tensor of shape ``[moe_ep_size, n_routed_experts]``
    (int32) specifies how many tokens each source rank sends to each expert.
    ``local_start`` and ``local_end`` select a contiguous range of local
    experts (the shard owned by the current EP rank).

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input token tensor ``[total_tokens, hidden_size]``, any dtype
            (handled bytewise).
        out (torch.Tensor): Output token tensor ``[total_tokens, hidden_size]``, same
            dtype as ``in_``.
        counts (torch.Tensor): Per-source per-expert token count matrix
            ``[moe_ep_size, n_routed_experts]`` (int32).
        hidden_size (int): Hidden dimension per token.
        local_start (int): First local expert index (inclusive).
        local_end (int): Last local expert index (exclusive).
        forward (bool): ``True`` for forward permutation, ``False`` for reverse.

    Returns:
        None: ``out`` is written in place.
    """
    ...

def linear_att_proj(
    rt: Runtime,
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    W_z: torch.Tensor,
    W_b: torch.Tensor,
    W_a: torch.Tensor,
    mix_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    m: int,
    n: int,
    v: int,
    h: int,
    k: int,
) -> None:
    """Linear attention projection.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input tensor, shape ``[m, k]``, fp16 or bf16.
        W_qkv (torch.Tensor): QKV weight tensor, shape ``[n, k]``, same dtype.
        W_z (torch.Tensor): Z weight tensor, shape ``[v, k]``, same dtype.
        W_b (torch.Tensor): B weight tensor, shape ``[h, k]``, same dtype.
        W_a (torch.Tensor): A weight tensor, shape ``[h, k]``, same dtype.
        mix_qkv (torch.Tensor): Output mixed QKV tensor, shape ``[m, n]``, same dtype.
        z (torch.Tensor): Output z(gating parameters) tensor, shape ``[m, v]``, same dtype.
        b (torch.Tensor): Output b(beta input) tensor, shape ``[m, h]``, same dtype.
        a (torch.Tensor): Output a(decay input) tensor, shape ``[m, h]``, same dtype.
        m (int): Dimension of the input x(batch*seqlen).
        n (int): QKV weight dimension.
        v (int): Z weight dimension.
        h (int): B,A weight dimension.
        k (int): Hidden layer dimension.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def transpose_1_2(rt: Runtime, input: torch.Tensor, output: torch.Tensor) -> None:
    """Transpose input 3D tensor along dimensions 1 and 2.

    Args:
        rt (Runtime): Native runtime handle.
        input (torch.Tensor): Input tensor of shape (b, m, n), fp16 or bf16.
        output (torch.Tensor): Output tensor of shape (b, n, m), same dtype.

    Returns:
        None: `output` is written in place.
    """
    ...

def linear_att_conv_and_silu(
    rt: Runtime,
    mix_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    query_lens: Optional[torch.Tensor] = None,
) -> None:
    """Fused causal conv1d + SiLU for linear attention (no host concat).

    Two modes. Uniform: ``mix_qkv``/``output`` are ``[B, S, C]`` and both
    optional tensors are omitted. Packed: ``mix_qkv``/``output`` are
    token-major ``[T, C]`` (``T = sum of per-request lengths``, ``T <= 256``
    requests), and ``query_start_loc``/``query_lens`` (both ``[batch]``,
    int32) describe the per-request segments; ``S <= 4096`` per request.

    Args:
        rt (Runtime): Native runtime handle.
        mix_qkv (torch.Tensor): Input mixed QKV tensor, ``[B, S, C]`` (uniform)
            or ``[T, C]`` (packed), fp32/fp16/bf16 (all operands same dtype).
        conv_state (torch.Tensor): Convolution state tensor, shape ``[B, C, K]``,
            same dtype; updated in place (K = kernel size, <= 16).
        weight (torch.Tensor): Kernel weight tensor, shape ``[C, 1, K]`` or
            ``[C, K]``, same dtype.
        output (torch.Tensor): Output tensor, same shape as ``mix_qkv``, same dtype.
        query_start_loc (Optional[torch.Tensor]): Packed mode only: exclusive
            prefix-sum segment starts, shape ``[batch]``, int32.
        query_lens (Optional[torch.Tensor]): Packed mode only: per-request token
            counts, shape ``[batch]``, int32.

    Returns:
        None: `output` is written in place. State is always updated.
    """
    ...

def linear_att_conv_and_silu_token(
    rt: Runtime,
    mix_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
) -> None:
    """Fused causal conv1d + SiLU on token-major input (uniform seqlen).

    Args:
        rt (Runtime): Native runtime handle.
        mix_qkv (torch.Tensor): Input mixed QKV tensor, token-major
            ``[T, C]`` with ``T = batch * seq_len``, fp32/fp16/bf16
            (all operands same dtype).
        conv_state (torch.Tensor): Convolution state tensor, shape
            ``[B, C, K]``, same dtype; updated in place. ``K`` (kernel
            dim) must be in ``{1, 2, 4}``; ``seq_len >= K``.
        weight (torch.Tensor): Kernel weight tensor, shape ``[C, 1, K]`` or
            ``[C, K]``, same dtype.
        output (torch.Tensor): Output tensor, shape ``[T, C]``, same dtype.
        seq_len (int): Per-request sequence length; ``batch * seq_len``
            must equal ``mix_qkv.shape[0]``. ``C`` must be a multiple of
            1024.

    Returns:
        None: `output` is written in place. State is updated via a separate
        kernel launch.
    """
    ...

def split_col(rt: Runtime, in_: torch.Tensor, outputs: List[torch.Tensor]) -> None:
    """Split tensor along column dimension into multiple outputs.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor to split, any rank >= 1, any dtype; split along
            the last dim.
        outputs (List[torch.Tensor]): List of output tensors (1..8 for the fused kernel),
            each sharing ``in_``'s dtype and leading dims, with the last dims summing to
            ``in_``'s last dim.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def concat(rt: Runtime, inputs: List[torch.Tensor], out: torch.Tensor) -> None:
    """Concatenate input tensors into one contiguous byte buffer (1D pack).

    The inputs are laid out end-to-end by raw bytes into ``out``. ``out`` is
    treated as a flat byte buffer: its total byte size must equal the sum of the
    input byte sizes. Element dtype / shape of the inputs need not match; only
    bytes are packed. Used by the MoE packed-send path to stage one AllGather.

    Args:
        rt (Runtime): Native runtime handle.
        inputs (List[torch.Tensor]): Input tensors to pack (each viewed as bytes; dtypes
            and shapes may differ).
        out (torch.Tensor): Flat output buffer holding all inputs concatenated; its total
            byte size must equal the sum of the input byte sizes.

    Returns:
        None: ``out`` is written in place.
    """
    ...

def split(
    rt: Runtime,
    in_: torch.Tensor,
    outputs: List[torch.Tensor],
    sizes: List[int],
    num_packets: int,
) -> None:
    """Split a contiguous byte buffer into outputs, repeated across packets.

    ``in_`` holds ``num_packets`` interleaved packets, each ``sum(sizes)``
    bytes. For packet ``i`` and output ``j``, ``sizes[j]`` bytes are copied from
    ``in_ + i*sum(sizes) + offset_j`` into ``outputs[j] + i*sizes[j]``. Each
    output is treated as a flat byte buffer of size ``num_packets * sizes[j]``
    bytes. Used by the MoE packed-recv path to deinterleave one AllGather result.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Flat input buffer holding all packets; byte size must equal
            ``sum(sizes) * num_packets``.
        outputs (List[torch.Tensor]): Output buffers, one per segment (len == len(sizes));
            each must hold at least ``sizes[j] * num_packets`` bytes.
        sizes (List[int]): Byte size of each segment within a single packet.
        num_packets (int): Number of interleaved packets in ``in_``.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def beta_decay(
    rt: Runtime,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    bsz: int,
    seqlen: int,
    num_v_heads: int,
) -> None:
    """Calculate beta and decay for linear attention.

    Args:
        rt (Runtime): Native runtime handle.
        b (torch.Tensor): b input tensor, shape ``[bsz, seqlen, num_v_heads]``, fp32/fp16/bf16
            (same as ``a``).
        a (torch.Tensor): a input tensor, same shape/dtype as ``b``.
        A_log (torch.Tensor): Learnable decay parameters (log-space; the kernel applies
            exp to it), shape ``[num_v_heads]``.
        dt_bias (torch.Tensor): Time bias, shape ``[num_v_heads]``.
        beta (torch.Tensor): beta output tensor, shape ``[bsz, seqlen, num_v_heads]``.
        g (torch.Tensor): g(decay) output tensor ``g = -exp(A_log) * softplus(a + dt_bias)``,
            shape ``[bsz, seqlen, num_v_heads]``.
        bsz (int): Batch size.
        seqlen (int): Sequence length.
        num_v_heads (int): Number of value heads.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def recurrent_gated_delta_rule(
    rt: Runtime,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    batch: int,
    seqlen: int,
    num_heads: int,
    k_dim: int,
    v_dim: int,
    query_start_loc: Optional[torch.Tensor] = None,
    query_lens: Optional[torch.Tensor] = None,
) -> None:
    """Recurrent gated delta rule (GDN linear-attention core).

    Two modes. Uniform: rows are ``[B*seqlen, ...]`` in batch-major order and
    both optional tensors are omitted. Packed: ``query_start_loc``/``query_lens``
    (both ``[batch]``, int32) describe per-request row segments
    (``[start[b], start[b]+lens[b])``) within the ``[T, ...]`` rows, and
    ``seqlen`` is ignored.

    Args:
        rt (Runtime): Native runtime handle.
        query (torch.Tensor): [B*S, H*k_dim], L2-normalized, fp32/fp16/bf16 (all operands
            same dtype).
        key (torch.Tensor): [B*S, H*k_dim], L2-normalized, same dtype.
        value (torch.Tensor): [B*S, H*v_dim], same dtype.
        beta (torch.Tensor): [B*S, H], same dtype.
        g (torch.Tensor): [B*S, H], log-space decay (kernel applies exp), same dtype.
        state (torch.Tensor): [B, H, k_dim, v_dim], same dtype, updated in-place.
        out (torch.Tensor): [B*S, H*v_dim], same dtype.
        batch (int): Batch size.
        seqlen (int): Sequence length.
        num_heads (int): Number of heads.
        k_dim (int): Key head dim (<=128).
        v_dim (int): Value head dim (<=128).
        query_start_loc (Optional[torch.Tensor]): Packed mode only: exclusive prefix-sum
            segment starts, shape ``[batch]``, int32.
        query_lens (Optional[torch.Tensor]): Packed mode only: per-request token counts,
            shape ``[batch]``, int32.

    Returns:
        None: Output and state are written in place.
    """
    ...

def einsum_mht_hdt_mhd(
    rt: Runtime,
    mht: torch.Tensor,
    hdt: torch.Tensor,
    mhd: torch.Tensor,
    m: int,
    h: int,
    t: int,
    d: int,
    weight_nz: bool = False,
) -> None:
    """Batched matmul for ``mhd = einsum("mht,hdt->mhd", mht, hdt)``.

    The right operand ``hdt`` has a head-major layout (``[h, d, t]``), which the
    kernel consumes via the matmul non-transpose path (transpose=0, loading
    ND2NZ with ``t`` as the row stride). The output ``mhd`` is laid
    out as ``[m, h, d]`` with the head dimension ``h`` kept as an outer loop.

    Args:
        rt (Runtime): Native runtime handle.
        mht (torch.Tensor): Left operand of shape ``[m, h, t]``, fp16 or bf16 (all three
            same dtype).
        hdt (torch.Tensor): Right operand of shape ``[h, d, t]``, same dtype.
        mhd (torch.Tensor): Output tensor of shape ``[m, h, d]``, same dtype.
        m (int): Outer batch dimension (token count).
        h (int): Head dimension.
        t (int): Inner reduction dimension.
        d (int): Output feature dimension.
        weight_nz (bool): Whether ``hdt`` is in NZ weight layout.

    Returns:
        None: ``mhd`` is written in place.
    """
    ...

def einsum_mht_htd_mhd(
    rt: Runtime,
    mht: torch.Tensor,
    htd: torch.Tensor,
    mhd: torch.Tensor,
    m: int,
    h: int,
    t: int,
    d: int,
    weight_nz: bool = False,
) -> None:
    """Batched matmul for ``mhd = einsum("mht,htd->mhd", mht, htd)``.

    The right operand ``htd`` has a head-row layout (``[h, t, d]``), which the
    kernel consumes via the matmul transpose path (transpose=1). The output ``mhd``
    is laid out as ``[m, h, d]`` with the head dimension ``h`` kept as an outer
    loop.

    Args:
        rt (Runtime): Native runtime handle.
        mht (torch.Tensor): Left operand of shape ``[m, h, t]``, fp16 or bf16 (all three
            same dtype).
        htd (torch.Tensor): Right operand of shape ``[h, t, d]``, same dtype.
        mhd (torch.Tensor): Output tensor of shape ``[m, h, d]``, same dtype.
        m (int): Outer batch dimension (token count).
        h (int): Head dimension.
        t (int): Inner reduction dimension.
        d (int): Output feature dimension.
        weight_nz (bool): Whether ``htd`` is in NZ weight layout.

    Returns:
        None: ``mhd`` is written in place.
    """
    ...

def unpack_activation(
    rt: Runtime,
    input: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Split int8 tensor to low/high int4 tensor.

    Args:
        rt (Runtime): Native runtime handle.
        input (torch.Tensor): input int8 tensor, shape ``[m, n]`` with ``n`` even.
        output (torch.Tensor): output low/high int4 tensor, shape ``[2*m, n/2]`` int8,
            interleaved: row ``2*r`` holds the low nibbles and row ``2*r + 1`` the
            high nibbles of input token ``r``.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def print(x: torch.Tensor, name: str = "", row: int = 6, col: int = 6) -> None:
    """Print a tensor preview for debugging.

    Args:
        x (torch.Tensor): Tensor to print.
        name (str): Optional label shown in output.
        row (int): Number of rows to print.
        col (int): Number of columns to print.

    Returns:
        None: Output is emitted to native stdout.
    """
    ...

def get_tile_size_of_cached_kv(
    cached_lens: List[int],
    query_lens: List[int],
    head_num_in_group: int,
    n_kv_heads: int,
    block_size: int,
    aic_num: int,
) -> int:
    """Get optimal tile size for cached KV based on workload.

    This function computes the optimal tile size for flash attention
    based on the current workload characteristics including cached KV
    lengths and query lengths.

    Args:
        cached_lens (List[int]): Per-sample cached KV token lengths.
        query_lens (List[int]): Per-sample query token lengths.
        head_num_in_group (int): Number of heads in each attention group.
        n_kv_heads (int): Number of key/value heads.
        block_size (int): KV cache block size.
        aic_num (int): Number of AI cores available.

    Returns:
        int: Optimal tile size for cached KV in flash attention.
    """
    ...

def hc_act(
    rt: Runtime,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
    eps: float,
    sinkhorn_iters: int,
    x_resid: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Hyper-Connection gate activation.

    Computes pre/post/comb gates from `mixes`:
      pre  = sigmoid(mixes[:, :K]       * scale[0] + base[:K])           + eps
      post = 2 * sigmoid(mixes[:, K:2K] * scale[1] + base[K:2K])
      comb = sinkhorn(softmax(mixes[:, 2K:] * scale[2] + base[2K:]) + eps)
    where K = hc_mult. `mixes` [n, mix_hc] (mix_hc = (2+K)*K), `hc_scale` [3],
    `hc_base` [mix_hc]; all fp32. Head mode is auto-detected when
    `hc_base.numel() == hc_mult` (only pre runs, post/comb untouched). `pre` is
    consumed internally and never written to GM.

    Args:
        rt (Runtime): Native runtime handle.
        mixes (torch.Tensor): Gate pre-activation [n, mix_hc] fp32, where
            ``mix_hc = (2+hc_mult)*hc_mult``.
        hc_scale (torch.Tensor): Per-segment scale [3] fp32 (or [1] in head mode).
        hc_base (torch.Tensor): Per-segment bias [mix_hc] fp32 (or [hc_mult] head).
        post (torch.Tensor): Output post gate [n, hc_mult] fp32 (empty in head mode).
        comb (torch.Tensor): Output comb [n, hc_mult*hc_mult] fp32 (empty in head mode).
        hc_mult (int): Hyper-connection multiplier K.
        eps (float): Epsilon added to pre, to the softmax input, and to every Sinkhorn
            denominator.
        sinkhorn_iters (int): Sinkhorn normalization iterations.
        x_resid (torch.Tensor): Merge input [n, hc_mult, hidden] bf16.
        output (torch.Tensor): Merge output [n, hidden] bf16.

    Returns:
        None: `post`/`comb`/`output` written in place.
    """
    ...

def hc_post(
    rt: Runtime,
    x: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    residual: torch.Tensor,
    y: torch.Tensor,
    m: int,
    hc_mult: int,
    hidden: int,
) -> None:
    """Hyper-Connection post-activation merge (DeepSeek-V4).

    y[m,k,D] = post[m,k]*x[m,D] (broadcast) + sum_h comb[m,h*H+k]*residual[m,h,D]:
    `comb` contracts the source stream h and indexes the output stream k, i.e. the
    flat layout is source-major. `x` [m, hidden] bf16, `post` [m, hc_mult] fp32,
    `comb` [m, hc_mult*hc_mult] fp32, `residual` [m, hc_mult, hidden] bf16,
    `y` [m, hc_mult, hidden] bf16. `residual` may alias `y` (in-place): all sources
    are read before any output is written.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Submodule output [m, hidden] bf16.
        post (torch.Tensor): Post gate [m, hc_mult] fp32.
        comb (torch.Tensor): Comb matrix [m, hc_mult*hc_mult] fp32.
        residual (torch.Tensor): Residual stream [m, hc_mult, hidden] bf16 (may alias ``y``).
        y (torch.Tensor): Output [m, hc_mult, hidden] bf16.
        m (int): Token count.
        hc_mult (int): Hyper-connection multiplier H.
        hidden (int): Hidden dimension D.

    Returns:
        None: `y` written in place.
    """
    ...