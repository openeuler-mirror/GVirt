"""Type stubs for :mod:`xlite._C`.

This file mirrors the symbols exported by `csrc/_C.cpp`.
The typing and docstrings are designed for Python 3.9 to 3.11.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence, overload

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
    ) -> None:
        """Create a runtime for one Ascend device.

        Args:
            devid (int): Device ID passed to the ACL runtime.
            size (int): Optional tensor-pool size in MB.
            rank (int): Global rank in the distributed group.
            tp_size (int): Tensor-parallel group size.
            dp_size (int): Data-parallel group size.
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
        block_size (int): KV block size.
        weight_nz (bool): Whether weights are in NZ layout.
        experts_weight_transpose (bool): Whether expert weights are transposed.
        experts_weight_nz (bool): Whether expert weights are in NZ layout.
        qkv_bias (bool): Whether MHA QKV has bias.
        qk_norm (bool): Whether MHA applies Q/K norm.
        qk_norm_full (bool): Whether MHA applies Q/K norm full.
        scoring_func (ScoringFuncType): MoE scoring function.
        norm_topk_prob (bool): Whether top-k probabilities are normalized.
        mrope_section (List[int]): mRoPE section layout values.
        mrope_interleaved (bool): Whether mRoPE layout is interleaved.
        deepstack_num_level (int): Number of deepstack levels.
        index_head_dim (int): Indexer head dimension.
        index_n_heads (int): Indexer head count.
        index_topk (int): Indexer top-k size.
        index_softmax_scale (float): Indexer softmax scale.
        index_rope_interleaved (bool): Whether indexer RoPE is interleaved.
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
    """KV block size."""
    weight_nz: bool = ...
    """Whether weights are in NZ layout."""
    experts_weight_transpose: bool = ...
    """Whether expert weights are transposed."""
    experts_weight_nz: bool = ...
    """Whether expert weights are in NZ layout."""
    qkv_bias: bool = ...
    """Whether MHA QKV has bias."""
    qk_norm: bool = ...
    """Whether MHA applies Q/K norm."""
    qk_norm_full: bool = ...
    """Whether MHA applies Q/K norm full."""
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
    """Indexer softmax scale."""
    index_rope_interleaved: bool = ...
    """Whether indexer RoPE is interleaved."""

class ModelAttnMeta:
    """Device-side attention metadata for the legacy forward path.

    The native runtime derives slot mapping and block-table tensors from these
    host lists before launching attention kernels.

    Attributes:
        lens (List[int]): Per-sample query lengths.
        cached_lens (List[int]): Per-sample cached lengths.
        is_prefills (List[bool]): Prefill/decode flags per sample.
        block_tables (List[List[int]]): Per-sample block tables.
    """

    lens: List[int] = ...
    """Per-sample query lengths."""
    cached_lens: List[int] = ...
    """Per-sample cached lengths."""
    is_prefills: List[bool] = ...
    """Prefill/decode flags per sample."""
    block_tables: List[List[int]] = ...
    """Per-sample block tables."""

class AttnMeta:
    """Host-side attention metadata for the version-1 vLLM-compatible path.

    This path reuses host block-table lists while taking `positions` directly
    from the provided tensor for attention position indexing.

    Attributes:
        lens (List[int]): Per-sample query lengths.
        cached_lens (List[int]): Per-sample cached lengths.
        is_prefills (List[bool]): Prefill/decode flags per sample.
        block_tables_cpu (List[List[int]]): Per-sample block tables on host.
        positions (torch.Tensor): Position tensor for version-1 attention metadata.
    """

    lens: List[int] = ...
    """Per-sample query lengths."""
    cached_lens: List[int] = ...
    """Per-sample cached lengths."""
    is_prefills: List[bool] = ...
    """Prefill/decode flags per sample."""
    block_tables_cpu: List[List[int]] = ...
    """Per-sample block tables on host."""
    positions: torch.Tensor = ...
    """Position tensor for version-1 attention metadata."""

class AttnType(Enum):
    """Attention type enum exported by the native extension."""

    AttnMHA = ...
    """Standard multi-head attention."""
    AttnMLA = ...
    """Multi-head latent attention."""
    AttnDSA = ...
    """Dual sparse attention."""

class ScoringFuncType(Enum):
    """MoE scoring function enum exported by the native extension."""

    ScoringFuncSoftmax = ...
    """Softmax-based expert routing."""
    ScoringFuncSigmoid = ...
    """Sigmoid-based expert routing."""

AttnMHA: AttnType = ...
"""Alias for :attr:`AttnType.AttnMHA`."""
AttnMLA: AttnType = ...
"""Alias for :attr:`AttnType.AttnMLA`."""
AttnDSA: AttnType = ...
"""Alias for :attr:`AttnType.AttnDSA`."""

ScoringFuncSoftmax: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSoftmax`."""
ScoringFuncSigmoid: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSigmoid`."""

class Model:
    """Model weights and forward methods.

    Attributes:
        embed (torch.Tensor): Embedding weights.
        norm (torch.Tensor): Final normalization weights.
        norm_bias (torch.Tensor): Final normalization bias.
        head (torch.Tensor): LM head weights.
        attn_norm (List[torch.Tensor]): Attention norm weights per layer.
        attn_norm_bias (List[torch.Tensor]): Attention norm bias per layer.
        attn_out (List[torch.Tensor]): Attention output projection weights per layer.
        attn_out_input_scale (List[torch.Tensor]): Attn output quantization input scale per layer.
        attn_out_input_offset (List[torch.Tensor]): Attn output quantization input offset per layer.
        attn_out_quant_bias (List[torch.Tensor]): Attn output quantization bias per layer.
        attn_out_deq_scale (List[torch.Tensor]): Attn output dequantization scale per layer.
        mha_qkv (List[torch.Tensor]): MHA QKV weights per layer.
        mha_qkv_bias (List[torch.Tensor]): MHA QKV bias per layer.
        mha_qkv_input_scale (List[torch.Tensor]): MHA QKV quantization input scale per layer.
        mha_qkv_input_offset (List[torch.Tensor]): MHA QKV quantization input offset per layer.
        mha_qkv_quant_bias (List[torch.Tensor]): MHA QKV quantization bias per layer.
        mha_qkv_deq_scale (List[torch.Tensor]): MHA QKV dequantization scale per layer.
        mha_q_norm (List[torch.Tensor]): MHA Q norm weights per layer.
        mha_q_norm_bias (List[torch.Tensor]): MHA Q norm bias per layer.
        mha_k_norm (List[torch.Tensor]): MHA K norm weights per layer.
        mha_k_norm_bias (List[torch.Tensor]): MHA K norm bias per layer.
        mla_q_a (List[torch.Tensor]): MLA QA weights per layer.
        mla_q_b (List[torch.Tensor]): MLA QB weights per layer.
        mla_q_norm (List[torch.Tensor]): MLA Q norm weights per layer.
        mla_kv_a (List[torch.Tensor]): MLA KVA weights per layer.
        mla_kv_b (List[torch.Tensor]): MLA KVB weights per layer.
        mla_kv_norm (List[torch.Tensor]): MLA KV norm weights per layer.
        index_q_b (List[torch.Tensor]): DSA index QB weights per layer.
        index_k (List[torch.Tensor]): DSA index K weights per layer.
        index_k_norm (List[torch.Tensor]): DSA index K norm weights per layer.
        index_k_norm_bias (List[torch.Tensor]): DSA index K norm bias per layer.
        index_weight (List[torch.Tensor]): DSA index weighting tensors per layer.
        mlp_norm (List[torch.Tensor]): MLP norm weights per layer.
        mlp_norm_bias (List[torch.Tensor]): MLP norm bias per layer.
        mlp_up_gate (List[torch.Tensor]): Dense up-gate weights per layer.
        mlp_down (List[torch.Tensor]): Dense down weights per layer.
        gate (List[torch.Tensor]): MoE gate weights per layer.
        gate_bias (List[torch.Tensor]): MoE gate bias per layer.
        se_up_gate (List[torch.Tensor]): Shared-expert up-gate weights per layer.
        se_down (List[torch.Tensor]): Shared-expert down weights per layer.
        re_up_gate (List[torch.Tensor]): Routed-expert up-gate weights.
        re_up_gate_scale (List[torch.Tensor]): Routed-expert up-gate scales.
        re_down (List[torch.Tensor]): Routed-expert down weights.
        re_down_scale (List[torch.Tensor]): Routed-expert down scales.
    """

    embed: torch.Tensor = ...
    """Embedding weights."""
    norm: torch.Tensor = ...
    """Final normalization weights."""
    norm_bias: torch.Tensor = ...
    """Final normalization bias."""
    head: torch.Tensor = ...
    """LM head weights."""
    attn_norm: List[torch.Tensor] = ...
    """Attention norm weights per layer."""
    attn_norm_bias: List[torch.Tensor] = ...
    """Attention norm bias per layer."""
    attn_out: List[torch.Tensor] = ...
    """Attention output projection weights per layer."""
    attn_out_input_scale: List[torch.Tensor] = ...
    """Attn output quantization input scale per layer."""
    attn_out_input_offset: List[torch.Tensor] = ...
    """Attn output quantization input offset per layer."""
    attn_out_quant_bias: List[torch.Tensor] = ...
    """Attn output quantization bias per layer."""
    attn_out_deq_scale: List[torch.Tensor] = ...
    """Attn output dequantization scale per layer."""
    mha_qkv: List[torch.Tensor] = ...
    """MHA QKV weights per layer."""
    mha_qkv_bias: List[torch.Tensor] = ...
    """MHA QKV bias per layer."""
    mha_qkv_input_scale: List[torch.Tensor] = ...
    """MHA QKV quantization input scale per layer."""
    mha_qkv_input_offset: List[torch.Tensor] = ...
    """MHA QKV quantization input offset per layer."""
    mha_qkv_quant_bias: List[torch.Tensor] = ...
    """MHA QKV quantization bias per layer."""
    mha_qkv_deq_scale: List[torch.Tensor] = ...
    """MHA QKV dequantization scale per layer."""
    mha_q_norm: List[torch.Tensor] = ...
    """MHA Q norm weights per layer."""
    mha_q_norm_bias: List[torch.Tensor] = ...
    """MHA Q norm bias per layer."""
    mha_k_norm: List[torch.Tensor] = ...
    """MHA K norm weights per layer."""
    mha_k_norm_bias: List[torch.Tensor] = ...
    """MHA K norm bias per layer."""
    mla_q_a: List[torch.Tensor] = ...
    """MLA QA weights per layer."""
    mla_q_b: List[torch.Tensor] = ...
    """MLA QB weights per layer."""
    mla_q_norm: List[torch.Tensor] = ...
    """MLA Q norm weights per layer."""
    mla_kv_a: List[torch.Tensor] = ...
    """MLA KVA weights per layer."""
    mla_kv_b: List[torch.Tensor] = ...
    """MLA KVB weights per layer."""
    mla_kv_norm: List[torch.Tensor] = ...
    """MLA KV norm weights per layer."""
    index_q_b: List[torch.Tensor] = ...
    """DSA index QB weights per layer."""
    index_k: List[torch.Tensor] = ...
    """DSA index K weights per layer."""
    index_k_norm: List[torch.Tensor] = ...
    """DSA index K norm weights per layer."""
    index_k_norm_bias: List[torch.Tensor] = ...
    """DSA index K norm bias per layer."""
    index_weight: List[torch.Tensor] = ...
    """DSA index weighting tensors per layer."""
    mlp_norm: List[torch.Tensor] = ...
    """MLP norm weights per layer."""
    mlp_norm_bias: List[torch.Tensor] = ...
    """MLP norm bias per layer."""
    mlp_up_gate: List[torch.Tensor] = ...
    """Dense up-gate weights per layer."""
    mlp_down: List[torch.Tensor] = ...
    """Dense down weights per layer."""
    gate: List[torch.Tensor] = ...
    """MoE gate weights per layer."""
    gate_bias: List[torch.Tensor] = ...
    """MoE gate bias per layer."""
    se_up_gate: List[torch.Tensor] = ...
    """Shared-expert up-gate weights per layer."""
    se_down: List[torch.Tensor] = ...
    """Shared-expert down weights per layer."""
    re_up_gate: List[torch.Tensor] = ...
    """Routed-expert up-gate weights."""
    re_up_gate_scale: List[torch.Tensor] = ...
    """Routed-expert up-gate scales."""
    re_down: List[torch.Tensor] = ...
    """Routed-expert down weights."""
    re_down_scale: List[torch.Tensor] = ...
    """Routed-expert down scales."""

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

    @overload
    def forward(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: ModelAttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass with device-side attention metadata.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (ModelAttnMeta): Device-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output hidden-state buffer.
            curr_stream (int): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On invalid shapes or cache mismatch.
        """

    @overload
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
        """Run forward pass with host/vLLM-compatible attention metadata.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output hidden-state buffer.
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
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run logits-only path.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input hidden-state tensor.
            output (torch.Tensor): Output logits tensor.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: If the native logits path fails.
        """

    @overload
    def forward_and_get_logits(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: ModelAttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass and materialize logits (device metadata).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (ModelAttnMeta): Device-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output logits buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On invalid KV-cache layout or other native execution failures.
        """

    @overload
    def forward_and_get_logits(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass and materialize logits (host metadata).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output logits buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On invalid KV-cache layout or other native execution failures.
        """

    @overload
    def forward_with_inputs_embeds(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: ModelAttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
        deepstack_input: Sequence[torch.Tensor] = ...,
    ) -> None:
        """Run forward pass with deepstack input embeddings (device metadata).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (ModelAttnMeta): Device-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output hidden-state buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.
            deepstack_input (Sequence[torch.Tensor]): Extra deepstack embeddings.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On KV-cache/deepstack shape mismatch or other native execution failures.
        """

    @overload
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
    ) -> None:
        """Run forward pass with deepstack input embeddings (host metadata).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output hidden-state buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.
            deepstack_input (Sequence[torch.Tensor]): Extra deepstack embeddings.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On KV-cache/deepstack shape mismatch or other native execution failures.
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

def all_gather(rt: Runtime, out: torch.Tensor, in_: torch.Tensor) -> None:
    """Collectively gather tensors from all ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for gathered values.
        in_ (torch.Tensor): Local input shard.

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def reduce_scatter(rt: Runtime, out: torch.Tensor, in_: torch.Tensor) -> None:
    """Reduce then scatter tensors across ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for the reduced local shard.
        in_ (torch.Tensor): Input tensor to reduce across ranks.

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def all_reduce(rt: Runtime, out: torch.Tensor, in_: torch.Tensor) -> None:
    """Collectively reduce tensors across all ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output tensor for reduced results.
        in_ (torch.Tensor): Input tensor to reduce.

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def add(rt: Runtime, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> None:
    """Elementwise add two tensors into output.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left operand.
        y (torch.Tensor): Right operand.
        z (torch.Tensor): Output tensor.

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
        x (torch.Tensor): Left matrix.
        y (torch.Tensor): Right matrix/weight.
        z (torch.Tensor): Output matrix.
        weight_nz (bool): Whether `y` uses NZ weight layout.
        transpose (bool): Whether to transpose the right matrix in compute.

    Returns:
        None: `z` is written in place.

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
        x (torch.Tensor): Left matrix.
        y (torch.Tensor): Right matrix/weight.
        z (torch.Tensor): Output matrix.
        bias (torch.Tensor): Bias tensor added to output.
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
        weight (torch.Tensor): Embedding table.
        in_ (torch.Tensor): Token IDs.
        out (torch.Tensor): Output embedding tensor.
        start (int): Start token index (inclusive).
        end (int): End token index (exclusive).

    Returns:
        None: `out` is written in place.
    """
    ...

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
    use_norm: bool = True,
    variance: Optional[torch.Tensor] = None
) -> None:
    """Apply RMSNorm with optional offsets.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        norm (torch.Tensor): RMSNorm weight tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.
        cnt_per_token (int): Number of contiguous segments per token.
        in_start_offset (int): Input offset for segmented normalization.
        out_start_offset (int): Output offset for segmented normalization.
        use_norm (bool): Whether to apply normalization.
        variance (Optional[torch.Tensor]): Optional output tensor for variance values.

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
        in_ (torch.Tensor): Input tensor.
        norm (torch.Tensor): RMSNorm weight tensor.
        norm_bias (torch.Tensor): RMSNorm Bias tensor.
        out (torch.Tensor): Output tensor.
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
        in_ (torch.Tensor): Input tensor.
        norm (torch.Tensor): LayerNorm weight tensor.
        norm_bias (torch.Tensor): LayerNorm bias tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width.

    Returns:
        None: `out` is written in place.
    """
    ...

def add_bias(rt: Runtime, in_: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Add bias tensor to input tensor.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        weight (torch.Tensor): Bias tensor.
        out (torch.Tensor): Output tensor.

    Returns:
        None: `out` is written in place.
    """
    ...

def silu_and_mul(rt: Runtime, in_: torch.Tensor, out: torch.Tensor) -> None:
    """Apply SiLU and gated multiply.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        out (torch.Tensor): Output tensor.

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
        inout (torch.Tensor): Input/output QKV tensor.
        k_cache (torch.Tensor): Key cache tensor.
        v_cache (torch.Tensor): Value cache tensor.
        position (torch.Tensor): Position indices.
        cosin (torch.Tensor): Rotary cosine/sine tensor.
        slot_mapping (torch.Tensor): Slot mapping for paged cache writes.
        n_heads (int): Number of query heads.
        n_kv_heads (int): Number of KV heads.
        head_dim (int): Head dimension.
        rot_dim (int): Rotary dimension.
        block_size (int): KV cache block size.
        is_neox (bool): Whether to use NeoX rotary layout.
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
    max_num_block: int,
) -> None:
    """Run paged attention for cached KV tensors.

    Args:
        rt (Runtime): Native runtime handle.
        qkv (torch.Tensor): Query tensor.
        k_cache (torch.Tensor): Key cache tensor.
        v_cache (torch.Tensor): Value cache tensor.
        output (torch.Tensor): Attention output tensor.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        block_tables (torch.Tensor): Block table tensor.
        n_heads (int): Number of query heads.
        n_kv_heads (int): Number of KV heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size.
        batch (int): Batch size.
        max_num_block (int): Maximum number of blocks per request.

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
        in_ (torch.Tensor): Residual input tensor.
        add_in_out (torch.Tensor): In/out tensor for residual accumulation.
        norm (torch.Tensor): RMSNorm weight tensor.
        out (torch.Tensor): Output tensor.
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
        scores (torch.Tensor): Routing score tensor.
        indices (torch.Tensor): Output top-k index tensor.
        out_weights (torch.Tensor): Output top-k probability tensor.
        out_routing (torch.Tensor): Output routing mask tensor.
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
        scores (torch.Tensor): Routing score tensor.
        indices (torch.Tensor): Output top-k index tensor.
        bias (torch.Tensor): Bias tensor applied before top-k selection.
        scale (float): Scale factor applied to scores.
        out_weights (torch.Tensor): Output top-k weight tensor.
        out_routing (torch.Tensor): Output routing mask tensor.
        n_group (int): Number of routing groups.
        n_topk_group (int): Number of groups participating in top-k.
        top_k (int): Number of experts selected per token.
        norm_top_k_prob (bool): Whether to normalize selected probabilities.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def cast_up(rt: Runtime, in_: torch.Tensor, out: torch.Tensor) -> None:
    """Cast tensor values to a higher-precision type.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        out (torch.Tensor): Output tensor.

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
        in_ (torch.Tensor): Input token tensor.
        routing (torch.Tensor): Routing/expert assignment tensor.
        start (int): Start expert index.
        end (int): End expert index.
        out (torch.Tensor): Permuted output tensor.
        unp_idx (torch.Tensor): Unpermutation index tensor.
        counts (torch.Tensor): Per-expert count tensor.

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
        in_ (torch.Tensor): Permuted input tensor.
        routing (torch.Tensor): Routing/expert assignment tensor.
        weights (torch.Tensor): Routing weight tensor.
        start (int): Start expert index.
        end (int): End expert index.
        out (torch.Tensor): Unpermuted output tensor.
        unp_idx (torch.Tensor): Unpermutation index tensor.

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
        in_ (torch.Tensor): Input tensor.
        weights (Sequence[torch.Tensor]): Per-group weight tensors.
        scales (Sequence[torch.Tensor]): Optional per-group scale tensors.
        counts (torch.Tensor): Per-group token count tensor.
        start (int): Start group index.
        end (int): End group index.
        out_dim (int): Output dimension.
        in_dim (int): Input dimension.
        output (torch.Tensor): Output tensor.
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
        x (torch.Tensor): Input/output tensor.
        calc_len (int): Effective softmax length.
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
    v_gather: torch.Tensor,
) -> None:
    """Apply complex-domain rotary embedding helper.

    Args:
        rt (Runtime): Native runtime handle.
        n_local_heads (int): Number of local heads.
        step_dim (int): Per-step hidden dimension.
        rope_dim (int): Rotary dimension.
        input_with_r (torch.Tensor): Input tensor with real/imag layout.
        freqs (torch.Tensor): Rotary frequency tensor.
        position (torch.Tensor): Position tensor.
        v_gather (torch.Tensor): Gather index/helper tensor.

    Returns:
        None: Output is produced in place according to kernel contract.
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
        x (torch.Tensor): Input tensor.
        scale_reciprocal (torch.Tensor): Reciprocal scale tensor.
        offset (torch.Tensor): Quantization offset tensor.
        out (torch.Tensor): Quantized output tensor.

    Returns:
        None: `out` is written in place.
    """
    ...

def quant_dynamic(rt: Runtime, x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    """Dynamically quantize tensor values and emit scale.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input tensor.
        scale (torch.Tensor): Output scale tensor.
        out (torch.Tensor): Quantized output tensor.

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
        x (torch.Tensor): Left matrix.
        y (torch.Tensor): Quantized right matrix.
        bias (torch.Tensor): Bias tensor.
        deq_scale (torch.Tensor): Dequantization scale tensor.
        z (torch.Tensor): Output matrix.
        weight_nz (bool): Whether `y` uses NZ layout.
        transpose (bool): Whether to transpose the right matrix.

    Returns:
        None: `z` is written in place.
    """
    ...

def dequant(rt: Runtime, in_: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, has_scale: bool) -> None:
    """Dequantize tensor values into output precision.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Quantized input tensor.
        scale (torch.Tensor): Scale tensor.
        out (torch.Tensor): Dequantized output tensor.
        has_scale (bool): Whether scale should be applied.

    Returns:
        None: `out` is written in place.
    """
    ...

def mla(
    rt: Runtime,
    q_with_qr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    wkvb: torch.Tensor,
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
    max_num_block: int,
    scale: float,
) -> None:
    """Run MLA kernel using cached KV blocks.

    Args:
        rt (Runtime): Native runtime handle.
        q_with_qr (torch.Tensor): Query tensor with rotary components.
        k_cache (torch.Tensor): Key cache tensor.
        v_cache (torch.Tensor): Value cache tensor.
        wkvb (torch.Tensor): MLA projection tensor.
        output (torch.Tensor): Attention output tensor.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        block_tables (torch.Tensor): Block table tensor.
        n_heads (int): Number of query heads.
        rope_head_dim (int): Rotary head dimension.
        nope_head_dim (int): Non-rotary head dimension.
        v_head_dim (int): Value head dimension.
        kv_lora_rank (int): KV LoRA rank.
        block_size (int): KV block size.
        batch (int): Batch size.
        max_num_block (int): Maximum number of blocks per request.
        scale (float): Attention scaling factor.

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
    max_num_block: int,
) -> None:
    """Compute DSA indexer scores over cached keys.

    Args:
        rt (Runtime): Native runtime handle.
        q (torch.Tensor): Query tensor.
        k_cache (torch.Tensor): Key cache tensor.
        weight (torch.Tensor): Indexer weight tensor.
        scores (torch.Tensor): Output score tensor.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        block_tables (torch.Tensor): Block table tensor.
        n_heads (int): Number of heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size.
        batch (int): Batch size.
        max_num_block (int): Maximum number of blocks per request.

    Returns:
        None: `scores` is written in place.
    """
    ...

def muls(rt: Runtime, input: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """Multiply tensor by scalar and write to output.

    Args:
        rt (Runtime): Native runtime handle.
        input (torch.Tensor): Input tensor.
        scale (float): Scalar multiplier.
        output (torch.Tensor): Output tensor.

    Returns:
        None: `output` is written in place.
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
