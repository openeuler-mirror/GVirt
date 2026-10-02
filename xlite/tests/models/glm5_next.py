#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# GLM-5.3-Flash (model_type = "glm5_next", arch Glm5NextForConditionalGeneration)
# torch_npu reference implementation for the xlite test harness.
#
# Hybrid architecture (released 2026-09-30; not present in the installed
# vllm-ascend 0.26.0 / transformers 5.14.1, both predate it). Assembled from
# three lineages, following the vLLM torch reference maths:
#   * MHC (multi-head conv residual mixing)  -- DeepSeek-V4 (layers/mhc.py,
#       kernels/mhc/torch.py: mhc_pre_torch / mhc_post_torch)
#   * Linear attention (KDA / Gated DeltaNet) -- Kimi-Linear
#       (layers/mamba/gdn/kimi_gdn_linear_attn.py + flash_linear_attention/ops/kda.py)
#   * Sparse MLA + DSA Indexer (kpool-compressed, qk_rope_head_dim=0) -- GLM-5.2
#       / DeepSeek-V3.2 lineage
#   * MoE (sigmoid + noaux_tc bias + swiglu_limit clamp)
#
# Quantization map (verified from the checkpoint): attention / indexer / MHC /
# norms / gate are BF16; only the MLP linear weights (dense / experts / shared
# gate_proj|up_proj|down_proj) are W8A8_DYNAMIC (int8 weight + per-channel scale,
# dequantized inside deepseek_v3.linear()). Only the torch_npu forward path is
# implemented; the xlite C++ path raises NotImplementedError. MTP layer 45 is
# skipped (non-speculative generation).
import os
import math
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Tuple, Optional, Literal, List

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu

# Reuse the low-level primitives from the GLM-5.2 reference (deepseek_v3.py).
# The linear-attention / MHC / kpool pieces are GLM-5.3 only and live here.
from tests.models.deepseek_v3 import (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear,
    RMSNorm,
    LayerNorm,
    ParallelEmbedding,
    quantize_npu,
    linear as ds_linear,
    apply_rotary_emb,
    precompute_freqs_cis,
    unpack_int4_weight,
    w4a8_msd_linear,
    load_tensor_parallel_weights,
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    logger,
    forward_backend as ds_forward_backend,
)
from tests.models.deepseek_kernel import weight_dequant
# ColumnParallelLinear / RowParallelLinear / ParallelEmbedding and the shared linear()
# helper below are imported from deepseek_v3 and read *its* module-level world_size / rank
# globals (not ours). Keep a handle on that module so GLM5Next.__init__ can mirror our TP
# rebind into it; otherwise those layers allocate full-size weights and skip all_reduce.
from tests.models import deepseek_v3 as _ds3

# Module-level globals mirror deepseek_v3's convention: rebound inside the model
# __init__ to the TP-local view (world_size<-tp_size, rank<-tp_rank). MoE keeps
# the pre-rebind full-world values via self.global_*.
debug = False
world_size = 1
rank = 0
global_rank = 0
global_world_size = 1
forward_backend = ds_forward_backend


@dataclass
class ModelArgs:
    """GLM-5.3-Flash (glm5_next) model configuration."""
    # general
    max_batch_size: int = 1
    max_seq_len: int = 1024
    max_num_batched_tokens: int = 1024
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 154880
    dim: int = 4096
    inter_dim: int = 12288
    moe_inter_dim: int = 2048
    n_layers: int = 45
    n_dense_layers: int = 3
    n_heads: int = 64
    norm_eps: float = 1e-5
    # moe
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5
    swiglu_limit: float = 10.0
    # mla (sparse layers; qk_rope_head_dim=0 -> no rope)
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    # yarn (GLM-5.3 has no rope_scaling -> rope_type "default", no YaRN)
    original_seq_len: int = 4096
    rope_theta: float = 1000000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    rope_type: str = "default"
    # linear attention (KDA)
    la_num_heads: int = 64
    la_head_dim: int = 128
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    # mhc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    hc_post_alpha: float = 2.0  # DeepSeek-V4 default for hc_post_mult_value
    # indexer (sparse MLA layers)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    # vision tower (GLM-5.3-Flash multimodal; out_hidden_size == dim for direct merge)
    vision_depth: int = 24
    vision_hidden_size: int = 1024
    vision_num_heads: int = 16
    vision_in_channels: int = 3
    vision_patch_size: int = 14
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2
    vision_out_hidden_size: int = 4096
    vision_intermediate_size: int = 4096
    vision_projection_intermediate_size: int = 10240
    vision_swiglu_limit: float = 10.0
    vision_rms_norm_eps: float = 1e-5
    vision_rope_theta: float = 10000.0  # config.vision_config.rope_parameters["rope_theta"]
    image_token_id: int = 154854
    indexer_rope_interleave: bool = True
    index_full_mask: Optional[List[bool]] = None
    # layer dispatch
    layer_types: Optional[List[str]] = None
    kda_layers: Optional[List[int]] = None
    full_attn_layers: Optional[List[int]] = None
    # quant / parallelism
    quantization: Literal["none", "w8a8", "w4a8"] = "w8a8"
    moe_ep_size: int = 16
    moe_tp_size: int = 1
    model_type: str = "glm5_next"
    config_path: Optional[Path] = None

    def __post_init__(self):
        self.max_num_batched_tokens = self.max_seq_len * self.max_batch_size
        if self.config_path and self.config_path.exists():
            with open(self.config_path, "r") as f:
                cfg = json.load(f)
            tc = cfg.get("text_config", cfg)
            if not self.layer_types:
                self.layer_types = tc.get("layer_types", []) or []
            la = tc.get("linear_attn_config", {}) or {}
            if not self.kda_layers:
                self.kda_layers = la.get("kda_layers", []) or []
            if not self.full_attn_layers:
                self.full_attn_layers = la.get("full_attn_layers", []) or []
            if not self.index_full_mask:
                itypes = tc.get("indexer_types", []) or []
                self.index_full_mask = [str(t).lower().startswith("full") for t in itypes]
            rp = tc.get("rope_parameters", {}) or {}
            rs = tc.get("rope_scaling", {}) or {}
            self.rope_theta = rp.get("rope_theta", self.rope_theta)
            rtype = rp.get("rope_type") or rs.get("type") or "default"
            self.rope_type = str(rtype).lower()
            if "original_max_position_embeddings" in rs:
                self.original_seq_len = rs["original_max_position_embeddings"]
        # pad/truncate dispatch masks to n_layers
        if self.layer_types and len(self.layer_types) < self.n_layers:
            self.layer_types = self.layer_types + ["linear_attention"] * (self.n_layers - len(self.layer_types))
        if not self.index_full_mask:
            self.index_full_mask = [True] * self.n_layers
        self.index_full_mask = self.index_full_mask[: self.n_layers]
        assert len(self.index_full_mask) == self.n_layers, "index_full_mask length mismatch"


# --------------------------------------------------------------------------------------
# MHC (multi-head conv residual mixing). Faithful port of vllm
# kernels/mhc/torch.py:mhc_pre_torch / mhc_post_torch. hc_mult3 = 2h + h^2 (here 24).
# --------------------------------------------------------------------------------------
class MHC(nn.Module):
    """Per-layer MHC mixing params (attn_hc / ffn_hc). fn=[hc_mult3, hc_mult*dim]."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.dim = args.dim
        self.hc_mult3 = 2 * self.hc_mult + self.hc_mult * self.hc_mult
        self.hc_hidden = self.hc_mult * self.dim
        self.sinkhorn_iters = args.hc_sinkhorn_iters
        self.eps = args.hc_eps
        self.rms_eps = args.norm_eps
        self.post_alpha = args.hc_post_alpha
        # fn: [hc_mult3, hc_mult*dim]; base: [hc_mult3]; scale: [3] -- all bf16 in checkpoint
        self.fn = nn.Parameter(torch.empty(self.hc_mult3, self.hc_hidden))
        self.base = nn.Parameter(torch.empty(self.hc_mult3))
        self.scale = nn.Parameter(torch.empty(3))

    def pre(self, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """residual: [bsz, seq, hc_mult, dim] -> (post_mix, comb_mix, layer_input).

        Mirrors mhc_pre_torch: RMS-normalize the flattened mix logits by the input
        RMS, then sigmoid (pre/post) and softmax+sinkhorn (comb). layer_input is
        the pre_mix-weighted sum of residual streams (single stream, bf16). The
        layer's input_layernorm is applied separately by the Block (equivalent to
        vllm's fused mhc_pre_big_fuse_with_norm, which folds RMSNorm_w into pre).
        """
        bsz, seqlen, h, d = residual.shape
        flat = residual.reshape(-1, h, d)  # [N, h, d]
        N = flat.shape[0]
        x = flat.reshape(N, h * d).to(torch.float32)
        fn = self.fn.to(torch.float32)
        base = self.base.to(torch.float32)
        scale = self.scale.to(torch.float32)
        mixes = torch.matmul(x, fn.t())
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / (h * d) + self.rms_eps)

        pre_logits = mixes[:, :h] * scale[0] + base[:h]
        pre_mix = torch.sigmoid(pre_logits) + self.eps  # [N, h]

        post_logits = mixes[:, h:2 * h] * scale[1] + base[h:2 * h]
        post_mix = torch.sigmoid(post_logits) * self.post_alpha  # [N, h]

        comb_logits = (mixes[:, 2 * h:].view(N, h, h) * scale[2]
                       + base[2 * h:].view(1, h, h))
        comb_mix = torch.softmax(comb_logits, dim=-1) + self.eps
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + self.eps)
            comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + self.eps)

        layer_input = torch.sum(pre_mix.unsqueeze(-1) * flat.to(torch.float32), dim=1).to(residual.dtype)
        return (post_mix.view(bsz, seqlen, h, 1),
                comb_mix.view(bsz, seqlen, h, h),
                layer_input.view(bsz, seqlen, d))

    def post(self, x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor,
             comb_mix: torch.Tensor) -> torch.Tensor:
        """out_j = post_mix_j * x + sum_i comb_mix_ij * residual_i.

        Mirrors mhc_post_torch einsum "...ij,...ih->...jh".
        """
        mixed = torch.einsum("bsij,bsih->bsjh", comb_mix.to(torch.float32), residual.to(torch.float32))
        post_term = post_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
        return (mixed + post_term).to(residual.dtype)


# --------------------------------------------------------------------------------------
# Linear attention (KDA / Gated DeltaNet), torch reference per vllm
# layers/mamba/gdn/kimi_gdn_linear_attn.py + third_party/flash_linear_attention/ops/kda.py.
# Pure-torch, token-by-token recurrence (deterministic; not triton/aiter).
# --------------------------------------------------------------------------------------
def _causal_conv1d_prefill(x: torch.Tensor, weight: torch.Tensor, k: int) -> torch.Tensor:
    """Causal short conv1d with SiLU for prefill. x:[bsz,seq,proj]; weight:[proj,1,k]."""
    bsz, seq, proj = x.shape
    x_t = x.transpose(1, 2)  # [B, proj, L]
    x_padded = F.pad(x_t, (k - 1, 0))  # causal left-pad
    out = F.conv1d(x_padded, weight, groups=proj)  # [B, proj, L]
    return F.silu(out).transpose(1, 2)  # [B, L, proj]


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-norm with eps inside the sqrt (matches official l2norm; F.normalize puts
    it outside via max(||x||, eps)). x is already fp32."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class LinearAttention(nn.Module):
    """KDA (Kimi Delta Attention) linear attention layer.

    Weights (all BF16): q/k/v_proj (ColumnParallel, 4096->8192), f_a_proj
    (replicated, 4096->128), f_b_proj (ColumnParallel, 128->8192), b_proj
    (ColumnParallel, 4096->64), q/k/v_conv1d [8192,1,4], A_log [64], dt_bias
    [8192], g_a_proj (replicated, 4096->128), g_b_proj (ColumnParallel, 128->8192),
    o_norm (RMSNorm, 128), o_proj (RowParallel, 8192->4096).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.la_num_heads
        self.n_local_heads = args.la_num_heads // world_size
        self.head_dim = args.la_head_dim
        self.proj_size = self.n_heads * self.head_dim  # 8192
        self.local_proj_size = self.n_local_heads * self.head_dim
        self.conv_k = args.short_conv_kernel_size
        self.scale = self.head_dim ** -0.5   # applied to q after L2-norm (official KDA)
        # safe_gate lower bound (-5.0): the forget gate is bounded, not the unbounded
        # vllm/KDA softplus form (which caused the greedy loop). See _gate.
        self.linear_lower_bound = args.gate_lower_bound

        self.q_proj = ColumnParallelLinear(self.dim, self.proj_size)
        self.k_proj = ColumnParallelLinear(self.dim, self.proj_size)
        self.v_proj = ColumnParallelLinear(self.dim, self.proj_size)
        self.f_a_proj = Linear(self.dim, self.head_dim)            # replicated
        self.f_b_proj = ColumnParallelLinear(self.head_dim, self.proj_size)
        self.b_proj = ColumnParallelLinear(self.dim, self.n_heads)
        self.g_a_proj = Linear(self.dim, self.head_dim)            # replicated
        self.g_b_proj = ColumnParallelLinear(self.head_dim, self.proj_size)
        self.wo = RowParallelLinear(self.proj_size, self.dim)   # named "wo" to match the
        # global o_proj->wo rename (MLA also uses "wo"); loaded row-parallel in load_weights.
        self.o_norm = RMSNorm(self.head_dim, args.norm_eps, bias=False)
        # conv1d / A_log / dt_bias stored FULL (small); sliced to local heads in forward
        self.q_conv1d = nn.Parameter(torch.empty(self.proj_size, 1, self.conv_k))
        self.k_conv1d = nn.Parameter(torch.empty(self.proj_size, 1, self.conv_k))
        self.v_conv1d = nn.Parameter(torch.empty(self.proj_size, 1, self.conv_k))
        self.A_log = nn.Parameter(torch.empty(self.n_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.proj_size))

        if forward_backend != "xlite":
            # q/k/v each need their own conv state (sharing one buffer makes
            # _conv1d_decode overwrite q's state with k's then v's).
            for _n in ("conv_state_q", "conv_state_k", "conv_state_v"):
                self.register_buffer(_n,
                                     torch.zeros(args.max_batch_size, self.n_local_heads, self.head_dim, self.conv_k - 1),
                                     persistent=False)
            # Recurrent state must be fp32: vllm's MambaStateDtypeCalculator.kda_state_dtype
            # hardcodes the KDA recurrent-state dtype to float32; conv_state stays bf16.
            self.register_buffer("recurrent_state",
                                 torch.zeros(args.max_batch_size, self.n_local_heads, self.head_dim, self.head_dim, dtype=torch.float32),
                                 persistent=False)

    def _gate(self, g1: torch.Tensor) -> torch.Tensor:
        """KDA forget gate (safe_gate, official Glm5NextTextForgetGate.forward):

            g' = linear_lower_bound * sigmoid(exp(A_log) * (g1 + dt_bias))

        With linear_lower_bound=-5.0 the gate is bounded in (-5, 0) → decay
        exp(g') ∈ (0.0067, 1), retaining long-range memory. The unbounded
        vllm form (-exp(A_log)*softplus(...)) was the root cause of the
        greedy-decoding loop. Sliced to local heads; g1: [..., local_proj] -> [., H, D].
        """
        local_A = self.A_log[rank * self.n_local_heads:(rank + 1) * self.n_local_heads]
        local_dt = self.dt_bias[rank * self.local_proj_size:(rank + 1) * self.local_proj_size]
        g = (g1.float() + local_dt).view(*g1.shape[:-1], self.n_local_heads, self.head_dim)
        decay_rate = torch.exp(local_A.float()).view(*([1] * (g1.dim() - 1)), self.n_local_heads, 1)
        return self.linear_lower_bound * torch.sigmoid(decay_rate * g)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis, mask) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.b_proj(x).float())  # [bsz, seq, n_local_heads]
        g1 = self.f_b_proj(self.f_a_proj(x))           # [bsz, seq, local_proj]
        g2 = self.g_b_proj(self.g_a_proj(x))           # [bsz, seq, local_proj]
        gate = self._gate(g1)                           # [bsz, seq, n_local_heads, head_dim]

        # slice conv weights to local heads
        s = slice(rank * self.local_proj_size, (rank + 1) * self.local_proj_size)
        qcw, kcw, vcw = self.q_conv1d[s], self.k_conv1d[s], self.v_conv1d[s]

        if seqlen > 1:
            # Save the raw projected inputs (pre-conv) to seed decode conv_state; the conv
            # calls below return new tensors, so q_raw/k_raw/v_raw keep the pre-conv values.
            q_raw, k_raw, v_raw = q, k, v
            q = _causal_conv1d_prefill(q, qcw, self.conv_k)
            k = _causal_conv1d_prefill(k, kcw, self.conv_k)
            v = _causal_conv1d_prefill(v, vcw, self.conv_k)
            q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            # q/k L2-norm (use_qk_l2norm_in_kernel=True), then scale q by
            # head_dim ** -0.5 (kernel: b_q = b_q * scale). k is NOT scaled. Keep q/k/v
            # fp32 for the delta-rule recurrence (vllm keeps the recurrent state fp32).
            q = _l2norm(q.float()) * self.scale
            k = _l2norm(k.float())
            v = v.float()
            core = self._delta_rule_prefill(q, k, v, gate, beta).to(x.dtype)  # [bsz, seq, H, D]
            # Seed decode conv_state with the last (k-1) raw inputs, newest-first.
            self._init_conv_states(q_raw, k_raw, v_raw, bsz, seqlen)
        else:
            q = self._conv1d_decode(q, qcw, self.conv_state_q).view(bsz, self.n_local_heads, self.head_dim)
            k = self._conv1d_decode(k, kcw, self.conv_state_k).view(bsz, self.n_local_heads, self.head_dim)
            v = self._conv1d_decode(v, vcw, self.conv_state_v).view(bsz, self.n_local_heads, self.head_dim)
            q = _l2norm(q.float()) * self.scale
            k = _l2norm(k.float())
            v = v.float()
            core = self._delta_rule_step(q, k, v, gate, beta).to(x.dtype)     # [bsz, H, D]

        core = core.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        g2 = g2.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # FusedRMSNormGated: RMSNorm(core) * sigmoid(g2)
        core = self.o_norm(core) * torch.sigmoid(g2.float()).to(core.dtype)
        return self.wo(core.reshape(bsz, seqlen, self.local_proj_size))

    def _delta_rule_prefill(self, q, k, v, gate, beta):
        """Token-by-token delta-rule recurrence (prefill).

        S_t = exp(gate_t) * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) ⊗ k_t;  o_t = q_t @ S_t.
        gate_t is the raw KDA gate g' = (-exp(A_log)) * softplus(g1 + dt_bias) (< 0), so
        exp(gate_t) in (0, 1) is the per-(head, head_dim) decay (matches the vllm
        fused_recurrent_gated_delta_rule kernel: b_h *= exp(b_gk)).
        """
        bsz, seq, H, D = q.shape
        S = self.recurrent_state[:bsz].clone()  # [bsz, H, D, D]
        out = torch.empty(bsz, seq, H, D, dtype=q.dtype, device=q.device)
        for t in range(seq):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]        # [bsz, H, D]
            gt = gate[:, t].unsqueeze(-1)                  # [bsz, H, D, 1]
            bt = beta[:, t, :, None, None]                 # [bsz, H, 1, 1]
            # Match the vllm fused_recurrent_gated_delta_rule kernel (IS_KDA):
            #   S <- exp(gate) * S          (decay FIRST)
            #   delta <- v - S k            (against the *decayed* S, not S_old)
            #   S <- S + beta * (delta ⊗ k)
            #   o <- S q
            S = torch.exp(gt) * S                          # [bsz, H, D, 1] * [bsz,H,D,D]
            # State layout S[b,h,d,e] with d=k-index, e=v-index (so S is the transpose of
            # the vllm kernel's [v,k] state: my Sk/o einsums contract d with k). The decay
            # gt[...,d,1] therefore scales the k-index, matching b_h *= exp(b_gk[None,:]).
            Sk = torch.einsum("bhde,bhd->bhe", S, kt)      # [bsz, H, D]  (contracts k-index d)
            delta = vt - Sk                                 # [bsz, H, D]  (v-index)
            # outer product delta[v] ⊗ k[k] -> S[k,v]: delta on e(v), k on d(k)
            S = S + bt * torch.einsum("bhe,bhd->bhde", delta, kt)  # [bsz, H, D, D]
            out[:, t] = torch.einsum("bhde,bhd->bhe", S, qt)
        self.recurrent_state[:bsz] = S
        return out

    def _delta_rule_step(self, q, k, v, gate, beta):
        """Single-token decode step (seqlen==1). gate/beta are [bsz, 1, H, D] / [bsz, 1, H].
        Decay-first ordering (matches vllm fused_recurrent_gated_delta_rule kernel)."""
        bsz, H, D = q.shape
        S = self.recurrent_state[:bsz]
        gt = gate[:, 0].unsqueeze(-1)          # [bsz, H, D, 1]
        bt = beta[:, 0, :, None, None]         # [bsz, H, 1, 1]
        S = torch.exp(gt) * S                   # decay first
        Sk = torch.einsum("bhde,bhd->bhe", S, k)   # contracts k-index d -> [bsz,H,D]
        delta = v - Sk                               # [bsz, H, D] (v-index)
        S = S + bt * torch.einsum("bhe,bhd->bhde", delta, k)  # delta[v] ⊗ k[k] -> S[k,v]
        self.recurrent_state[:bsz] = S
        return torch.einsum("bhde,bhd->bhe", S, q)  # [bsz, H, D]

    def _init_conv_states(self, q_raw, k_raw, v_raw, bsz: int, seqlen: int) -> None:
        """Seed decode conv_state_{q,k,v} with the last (k-1) raw projected inputs.

        conv_state is ordered newest-first: cs[...,0] = most recent past token. Matches
        vllm causal_conv1d_fn(conv_states=...) writing the trailing (width-1) inputs.
        """
        k = self.conv_k
        take = min(k - 1, seqlen)
        H, D = self.n_local_heads, self.head_dim
        for raw, buf in ((q_raw, self.conv_state_q),
                         (k_raw, self.conv_state_k),
                         (v_raw, self.conv_state_v)):
            last = raw[:, -take:, :].reshape(bsz, take, H, D)          # oldest->newest
            slot = torch.zeros(bsz, k - 1, H, D, device=raw.device, dtype=raw.dtype)
            slot[:, :take] = last.flip(1)                              # newest at index 0
            buf[:bsz] = slot.permute(0, 2, 3, 1)                      # [bsz, H, D, k-1]

    def _conv1d_decode(self, x_tok: torch.Tensor, cw: torch.Tensor,
                       state: torch.Tensor) -> torch.Tensor:
        """Single-step causal conv with SiLU, updating `state` in place.

        x_tok: [bsz, local_proj]; cw: [local_proj, 1, k]; state: [bsz, H, D, k-1].
        """
        bsz = x_tok.shape[0]
        xh = x_tok.view(bsz, self.n_local_heads, self.head_dim)  # [bsz, H, D]
        cs = state[:bsz]                                          # [bsz, H, D, k-1], newest-first
        new = torch.cat([xh.unsqueeze(-1), cs], dim=-1)           # [bsz, H, D, k] = [x_t, x_{t-1}, ...]
        # Reference causal_conv1d: out = x[t]*w[k-1] + x[t-1]*w[k-2] + ... + x[t-k+1]*w[0]
        # (w[k-1] for the current token, w[0] for the oldest). `new` is newest-first, so
        # reverse the kernel before the elementwise product.
        w = cw.view(self.n_local_heads, self.head_dim, self.conv_k).flip(-1)  # [H, D, k]
        y = (new * w.unsqueeze(0)).sum(dim=-1)                    # [bsz, H, D]
        y = F.silu(y)
        state[:bsz] = new[..., :-1]
        return y.reshape(bsz, self.local_proj_size)


# --------------------------------------------------------------------------------------
# Vision tower (GLM-5.3-Flash). Pure PyTorch, weights replicated on every rank.
# Mirrors HF Glm5NextVisionModel (modeling_glm5_next.py:1515-1883). out_hidden_size
# (4096) == LM dim, so image features scatter directly into inputs_embeds at the
# image_token_id placeholder rows (no projection needed). Only the image (not video)
# path is implemented.
# --------------------------------------------------------------------------------------
class VisionRMSNorm(nn.Module):
    """RMSNorm = x * rsqrt(mean(x^2)+eps) * weight (matches Glm5NextRMSNorm)."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(orig)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class VisionPatchEmbed(nn.Module):
    """Conv3d patch embedding: [N,3,T,P,P] -> [N, hidden]. Matches HF PatchEmbed."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.patch_size = args.vision_patch_size
        self.temporal_patch_size = args.vision_temporal_patch_size
        self.in_channels = args.vision_in_channels
        self.embed_dim = args.vision_hidden_size
        ksize = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=ksize, stride=ksize)

    def forward(self, x):
        dt = self.proj.weight.dtype
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(dt)).view(-1, self.embed_dim)


class VisionRotaryEmbedding(nn.Module):
    """Axial 2D RoPE: inv_freq over half head_dim; cos/sin laid out as [N,2d] -> [N,2d] via
    [h,w; h,w] concatenation (recomposition_frequencies). head_dim=64 -> 32 freqs, [h,w] each 32.
    attention_scaling is 1.0 for axial rope (compute_axial_rope_parameters), so omitted."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        base = args.vision_rope_theta
        dim = args.vision_hidden_size // args.vision_num_heads  # 1024 // 16 = 64
        spatial_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _recompose(freq):
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (N, 2) — rows are (h_coord, w_coord) for each token
        pos = position_ids[..., None].float()
        freqs = pos * self.inv_freq.float()
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos, sin


def _apply_rotary_vision(q, k, cos, sin):
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    qe = q * cos + _rotate_half(q) * sin
    ke = k * cos + _rotate_half(k) * sin
    return qe.to(orig_q), ke.to(orig_k)


class VisionAttention(nn.Module):
    """Vision self-attention (MQA-style fused qkv, q/k RMSNorm, rotary, no causal mask).
    HF uses flash varlen via cu_seqlens; here we process each image's tokens as one
    flattened, non-causal block (image=t=1)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.vision_num_heads
        self.head_dim = args.vision_hidden_size // self.num_heads
        self.qkv = nn.Linear(args.vision_hidden_size, args.vision_hidden_size * 3, bias=True)
        self.proj = nn.Linear(args.vision_hidden_size, args.vision_hidden_size, bias=True)
        self.scale = self.head_dim ** -0.5
        self.q_norm = VisionRMSNorm(self.head_dim, eps=args.vision_rms_norm_eps)
        self.k_norm = VisionRMSNorm(self.head_dim, eps=args.vision_rms_norm_eps)

    def forward(self, x, cu_seqlens, position_embeddings):
        # x: [N, hidden]; process each image as one contiguous, non-causal segment.
        seqlen = x.shape[0]
        q, k, v = self.qkv(x).reshape(seqlen, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # q,k,v: [seqlen, num_heads, head_dim] (HF applies q/k RMSNorm + rotary in this layout).
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        q, k = _apply_rotary_vision(q, k, cos, sin)
        # -> [num_heads, seqlen, head_dim] for batched per-segment attention.
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        starts = cu_seqlens[:-1]
        ends = cu_seqlens[1:]
        outs = []
        for s, e in zip(starts, ends):
            qs, ks, vs = q[:, s:e], k[:, s:e], v[:, s:e]  # [num_heads, L, head_dim]
            a = torch.matmul(qs, ks.transpose(-1, -2)) * self.scale  # [num_heads, L, L]
            a = a.softmax(dim=-1)
            outs.append(torch.matmul(a, vs))  # [num_heads, L, head_dim]
        out = torch.cat(outs, dim=1)  # [num_heads, seqlen, head_dim]
        out = out.transpose(0, 1).reshape(seqlen, -1).contiguous()  # [seqlen, hidden]
        return self.proj(out)


class VisionMLP(nn.Module):
    """SwiGLU with clamp (matches Glm5NextVisionMLP)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        h, i = args.vision_hidden_size, args.vision_intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=True)
        self.up_proj = nn.Linear(h, i, bias=True)
        self.down_proj = nn.Linear(i, h, bias=True)
        self.lim = args.vision_swiglu_limit

    def forward(self, x):
        g = self.gate_proj(x).clamp(min=None, max=self.lim)
        u = self.up_proj(x).clamp(min=-self.lim, max=self.lim)
        return self.down_proj(F.silu(g) * u)


class VisionBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm1 = VisionRMSNorm(args.vision_hidden_size, eps=args.vision_rms_norm_eps)
        self.norm2 = VisionRMSNorm(args.vision_hidden_size, eps=args.vision_rms_norm_eps)
        self.attn = VisionAttention(args)
        self.mlp = VisionMLP(args)

    def forward(self, x, cu_seqlens, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nn.Module):
    """proj -> post_projection_norm (LayerNorm) -> GELU -> SwiGLU (matches Glm5NextVisionPatchMerger)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.vision_out_hidden_size
        ctx = args.vision_projection_intermediate_size
        self.proj = nn.Linear(dim, dim, bias=False)
        # NB: HF uses nn.LayerNorm here (mean-subtracting), not RMSNorm like the ViT blocks.
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, ctx, bias=False)
        self.up_proj = nn.Linear(dim, ctx, bias=False)
        self.down_proj = nn.Linear(ctx, dim, bias=False)
        self.act1 = nn.GELU()
        self.lim = args.vision_swiglu_limit

    def forward(self, x):
        x = self.proj(x)
        x = self.act1(self.post_projection_norm(x))
        g = self.gate_proj(x).clamp(min=None, max=self.lim)
        u = self.up_proj(x).clamp(min=-self.lim, max=self.lim)
        return self.down_proj(F.silu(g) * u)


class VisionModel(nn.Module):
    """GLM-5.3-Flash vision tower. Forward mirrors Glm5NextVisionModel.forward.
    Input: pixel_values [N, 3, T, P, P] (already patchified), grid_thw [num_images, 3]=(T,H,W).
    Output: image_features [sum(T_i*H_i*W_i / merge^2), out_hidden_size]."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.spatial_merge_size = args.vision_spatial_merge_size
        self.patch_size = args.vision_patch_size
        self.config_depth = args.vision_depth
        self.patch_embed = VisionPatchEmbed(args)
        self.rotary_pos_emb = VisionRotaryEmbedding(args)
        self.blocks = nn.ModuleList([VisionBlock(args) for _ in range(args.vision_depth)])
        self.merger = VisionPatchMerger(args)
        self.downsample = nn.Conv2d(args.vision_hidden_size, args.vision_out_hidden_size,
                                    kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size)
        self.post_layernorm = VisionRMSNorm(args.vision_hidden_size, eps=args.vision_rms_norm_eps)

    def _position_ids(self, grid_thw):
        """Per-image (h, w) position ids, block-major over merge blocks; (N, 2)."""
        device = grid_thw.device
        m = self.spatial_merge_size
        pos = []
        for t, h, w in grid_thw.tolist():
            hp, wp = torch.meshgrid(torch.arange(h, device=device),
                                    torch.arange(w, device=device), indexing="ij")
            bs = (h // m, m, w // m, m)
            hp = hp.reshape(bs).transpose(1, 2).flatten()
            wp = wp.reshape(bs).transpose(1, 2).flatten()
            pos.append(torch.stack([hp, wp], dim=-1).repeat(t, 1))
        return torch.cat(pos, dim=0)

    def _cu_seqlens(self, grid_thw):
        # one contiguous non-causal segment per image: lengths = T*H*W (pre-merge)
        device = grid_thw.device
        lens = (grid_thw.prod(-1)).tolist()
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
        for i, L in enumerate(lens):
            cu[i + 1] = cu[i] + L
        return cu

    def forward(self, pixel_values, grid_thw):
        pos_ids = self._position_ids(grid_thw)
        cu_seqlens = self._cu_seqlens(grid_thw)
        x = self.patch_embed(pixel_values)
        pos_emb = self.rotary_pos_emb(x, pos_ids)
        for blk in self.blocks:
            x = blk(x, cu_seqlens, pos_emb)
        x = self.post_layernorm(x)
        # merge: [N, hidden] -> [-1, m, m, hidden] -> [N/m^2, m, m, hidden] -> [C, m, m, ...]
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.merger.proj.in_features)
        return self.merger(x)


# --------------------------------------------------------------------------------------
# Indexer (DSA, kpool-compressed, qk_rope_head_dim=0). All weights BF16, replicated.
# --------------------------------------------------------------------------------------
class Indexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.kpool = args.index_kpool
        self.compress = args.index_kpool_compress
        self.always_select_tail = args.index_kpool_always_select_tail
        self.softmax_scale = self.head_dim ** -0.5
        # wq_b loaded FULL (not TP-sharded), BF16. wk / weights_proj are distinct in this
        # checkpoint (the GLM-5.2 Indexer fuses them into one fp32 wk_weights_proj); kept
        # fp32 so the index scores retain precision -- the bf16 checkpoint weights copy_
        # into the fp32 params. The kpool gate / ape are fp32 for the same reason.
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim, dtype=torch.float32)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
        self.k_norm = LayerNorm(self.head_dim)
        if self.compress and self.kpool > 1:
            self.index_kpool_compress_gate = nn.Parameter(torch.empty(self.head_dim, self.dim, dtype=torch.float32))
            self.index_kpool_compress_ape = nn.Parameter(torch.empty(self.kpool, self.head_dim, dtype=torch.float32))
        if forward_backend != "xlite":
            self.register_buffer("k_cache",
                                 torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, mask) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        wk_weights = self.wk(x.float())                              # [bsz, seq, head_dim], fp32
        k = self.k_norm(wk_weights[..., :self.head_dim].type_as(x))
        weights = self.weights_proj(x.float()) * (self.n_heads ** -0.5)  # [bsz, seq, n_heads], fp32
        if self.compress and self.kpool > 1:
            # kpool compression: F.linear on the [head_dim, dim] gate yields only head_dim
            # outputs per token, so broadcast against the [kpool, head_dim] ape with unsqueeze
            # (NOT a reshape to [bsz, seq, kpool, head_dim], which would need kpool*head_dim
            # elements). Gated mean over the kpool slots then compresses back to [bsz, seq, head_dim].
            gate = torch.sigmoid(F.linear(x.float(), self.index_kpool_compress_gate).unsqueeze(2)
                                 + self.index_kpool_compress_ape)   # [bsz, seq, kpool, head_dim]
            k = (k.unsqueeze(2) * gate).mean(dim=2)                  # [bsz, seq, head_dim]
        self.k_cache[:bsz, start_pos:end_pos] = k
        scores = torch.relu(torch.einsum("bshd,btd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale)
        index_score = torch.einsum("bsht,bsh->bst", scores, weights)
        if mask is not None:
            index_score = index_score + mask
        topk = min(self.index_topk, end_pos)
        topk_indices = index_score.topk(topk, dim=-1)[1]
        if self.always_select_tail and (start_pos + seqlen - 1) < topk_indices.size(-1):
            topk_indices[..., -1] = start_pos + seqlen - 1
        return topk_indices


# --------------------------------------------------------------------------------------
# Sparse MLA (qk_rope_head_dim=0 -> no rope). Attention is BF16 (not int8).
# --------------------------------------------------------------------------------------
class MLA(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.layer_id = layer_id
        self.has_indexer = bool(args.index_full_mask[layer_id]) if args.index_full_mask else True

        # BF16 attention (no int8) -- wqkv_a fused (q_lora + kv_lora), wkv_b float
        self.wqkv_a = Linear(self.dim, self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.norm_eps, bias=False)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.indexer = Indexer(args) if self.has_indexer else None
        if forward_backend != "xlite":
            self.register_buffer("kv_cache",
                                 torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
                                 persistent=False)
            if self.qk_rope_head_dim > 0:
                self.register_buffer("pe_cache",
                                     torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
                                     persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis, mask) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        qkv_lora = self.wqkv_a(x)
        qr, kv_lora = torch.split(qkv_lora, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
        qr = self.q_norm(qr)
        q = self.wq_b(qr).view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        if self.qk_rope_head_dim > 0:
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            q_pe = apply_rotary_emb(q_pe, freqs_cis)
            kv, k_pe = torch.split(kv_lora, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv = self.kv_norm(kv)
            k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        else:
            q_nope = q
            kv = self.kv_norm(kv_lora)
        self.kv_cache[:bsz, start_pos:end_pos] = kv

        if mask is not None:  # prefill (seqlen > 1)
            if self.qk_rope_head_dim > 0:
                q = torch.cat([q_nope, q_pe], dim=-1)
            kvb = self.wkv_b(kv).view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kvb, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            if self.qk_rope_head_dim > 0:
                k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
                scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)
            else:
                scores = torch.einsum("bshd,bthd->bsht", q_nope, k_nope).mul_(self.softmax_scale)
            if self.indexer is not None:
                topk_indices = self.indexer(x, qr, start_pos, mask)
                index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
                scores = scores + (index_mask + mask).unsqueeze(2)
            else:
                scores = scores + mask.unsqueeze(1)
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:  # decode (seqlen == 1, MQA)
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_eff = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = torch.einsum("bshc,btc->bsht", q_eff, self.kv_cache[:bsz, :end_pos]) * self.softmax_scale
            if self.qk_rope_head_dim > 0:
                scores = scores + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos]) * self.softmax_scale
            if self.indexer is not None:
                # recompute top-k for the new token against the full kv_cache
                topk_indices = self.indexer(x, qr, start_pos, mask)
                index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
                scores = scores + index_mask.unsqueeze(2)
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        return self.wo(x.flatten(2))


# --------------------------------------------------------------------------------------
# MoE / MLP. MLP linear weights are W8A8_DYNAMIC (int8 + per-channel scale); gate BF16.
# --------------------------------------------------------------------------------------
class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = ds_linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1) if self.bias is not None else scores.amax(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weights = weights * self.route_scale
        return weights.type_as(x), indices


def _swiglu_clamp(y1: torch.Tensor, y3: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU with input clamping, matching vllm SiluAndMulWithClamp.forward_native:
    gate = clamp(y1, max=limit); up = clamp(y3, min=-limit, max=limit); silu(gate)*up.
    Omitting the up-clamp lets unbounded y3 spikes corrupt the residual, which under
    greedy decoding settles into a self-reinforcing token loop (sampling masks it)."""
    if limit > 0:
        return F.silu(y1.clamp(max=limit)) * y3.clamp(min=-limit, max=limit)
    return F.silu(y1) * y3


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs):
        super().__init__()
        # W8A8_DYNAMIC: int8 weight + per-channel scale (dequant in deepseek_v3.linear)
        self.w13 = Linear(dim, inter_dim * 2, dtype=torch.int8)
        self.w2 = Linear(inter_dim, dim, dtype=torch.int8)
        self.swiglu_limit = args.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w13(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.w2(_swiglu_clamp(y1, y3, self.swiglu_limit))


class SharedMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs):
        super().__init__()
        self.w13 = Linear(dim, inter_dim * 2, dtype=torch.int8)
        self.w2 = Linear(inter_dim, dim, dtype=torch.int8)
        self.swiglu_limit = args.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w13(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.w2(_swiglu_clamp(y1, y3, self.swiglu_limit))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % global_world_size == 0, "experts must divide world size"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // global_world_size
        self.swiglu_limit = args.swiglu_limit
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, args)
                                      if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = SharedMLP(args.dim, args.n_shared_experts * args.moe_inter_dim, args)

    @property
    def experts_start_idx(self):
        return global_rank * (self.n_routed_experts // global_world_size)

    @property
    def experts_end_idx(self):
        return self.experts_start_idx + (self.n_routed_experts // global_world_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x[idx]) * weights[idx, top, None]
        if global_world_size > 1:
            dist.all_reduce(y)
        z = self.shared_experts(x)
        return (y + z).view(shape)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int, args: ModelArgs):
        super().__init__()
        self.w13 = ColumnParallelLinear(dim, inter_dim * 2, dtype=torch.int8)
        self.w2 = RowParallelLinear(inter_dim, dim, dtype=torch.int8)
        self.swiglu_limit = args.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w13(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.w2(_swiglu_clamp(y1, y3, self.swiglu_limit))


# --------------------------------------------------------------------------------------
# Block: MHC pre -> attn -> MHC post -> MHC pre -> ffn -> MHC post.
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        is_sparse = (args.layer_types[layer_id] == "deepseek_sparse_attention") if args.layer_types else (layer_id in (args.full_attn_layers or []))
        self.attn = MLA(layer_id, args) if is_sparse else LinearAttention(args)
        self.ffn = MLP(args.dim, args.inter_dim, args) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_hc = MHC(args)
        self.ffn_hc = MHC(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps, bias=False)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, bias=False)
        self.layer_id = layer_id

    def forward(self, residual: torch.Tensor, start_pos: int, freqs_cis, mask) -> torch.Tensor:
        # attn sub-layer
        post_mix, comb_mix, layer_input = self.attn_hc.pre(residual)
        attn_out = self.attn(self.attn_norm(layer_input), start_pos, freqs_cis, mask)
        residual = self.attn_hc.post(attn_out, residual, post_mix, comb_mix)
        # ffn sub-layer
        post_mix2, comb_mix2, layer_input2 = self.ffn_hc.pre(residual)
        ffn_out = self.ffn(self.ffn_norm(layer_input2))
        residual = self.ffn_hc.post(ffn_out, residual, post_mix2, comb_mix2)
        return residual


# --------------------------------------------------------------------------------------
# GLM5Next top-level model.
# --------------------------------------------------------------------------------------
class GLM5Next(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank, global_rank, global_world_size
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        global_rank = rank
        global_world_size = world_size
        self.global_rank = rank
        self.global_world_size = world_size
        self.dp_size = int(os.getenv("XLITE_DP_SIZE", "1"))
        assert world_size % self.dp_size == 0, "world_size must divide dp_size"
        self.tp_size = world_size // self.dp_size
        self.tp_rank = rank % self.tp_size
        world_size = self.tp_size
        rank = self.tp_rank
        # Mirror the TP rebind (and the full-world EP values) into deepseek_v3's module
        # globals. The imported ColumnParallelLinear / RowParallelLinear / ParallelEmbedding
        # and linear() read deepseek_v3.world_size / .rank, not ours; without this they build
        # full-size params (only 1/tp_size loaded -> zeros) and RowParallelLinear.forward
        # never all_reduces (world_size==1), corrupting every TP-sharded projection.
        _ds3.world_size = world_size
        _ds3.rank = rank
        _ds3.global_rank = global_rank
        _ds3.global_world_size = global_world_size
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.hc_mult
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        # No learnable hc_head in GLM-5.3-Flash checkpoint -> weight-free sum collapse
        self.norm = RMSNorm(args.dim, args.norm_eps, bias=False)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # Vision tower: weights replicated on every rank (no TP sharding, like the indexer).
        # Only instantiated so model.visual.* weights can load; unused on the pure-text path.
        self.visual = VisionModel(args)
        # freqs_cis unused when qk_rope_head_dim=0 (GLM-5.3); built only when rope is in use.
        if args.qk_rope_head_dim > 0:
            self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        else:
            self.register_buffer("freqs_cis", torch.empty(0, 0), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor = None, start_pos: int = 0,
                inputs_embeds: torch.Tensor = None,
                pixel_values: torch.Tensor = None,
                image_grid_thw: torch.Tensor = None):
        if forward_backend == "xlite":
            raise NotImplementedError("GLM-5.3-Flash xlite C++ path not implemented; use FORWARD_BACKEND=torch_npu")
        if tokens is None and inputs_embeds is None:
            raise ValueError("forward requires either tokens or inputs_embeds")
        # Build inputs_embeds: from tokens via embed, then scatter image features at
        # image_token_id placeholder rows (prefill only; decode reuses the cached KV).
        if inputs_embeds is None:
            inputs_embeds = self.embed(tokens)
        if pixel_values is not None:
            img_feats = self.visual(pixel_values, image_grid_thw)  # [N_img_tokens, out_hidden_size]
            img_mask = (tokens == self.args.image_token_id).unsqueeze(-1)
            n = int(img_mask.sum().item())
            assert n * inputs_embeds.shape[-1] == img_feats.numel(), (
                f"image tokens ({n}) * dim != image_features ({img_feats.shape[0]})")
            inputs_embeds = inputs_embeds.masked_scatter(img_mask, img_feats.to(inputs_embeds.dtype))
        # broadcast single stream to hc_mult residual streams
        seqlen = inputs_embeds.size(1)
        residual = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs_embeds.device).triu_(1)
        for li, layer in enumerate(self.layers):
            residual = layer(residual, start_pos, self.freqs_cis[start_pos:start_pos + seqlen], mask)
        # weight-free collapse: mean over hc_mult streams (official "unweighted mean")
        h = residual.mean(dim=2)  # [bsz, seq, dim]
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    # ---- weight loading ---------------------------------------------------
    def load_weights(self, model_path: str):
        args = self.args
        assert args.dim % world_size == 0, "dim must divide world_size"
        assert args.n_heads % world_size == 0, "n_heads must divide world_size"
        assert args.la_num_heads % world_size == 0, "la_num_heads must divide world_size"
        assert args.vocab_size % world_size == 0, "vocab_size must divide world_size"

        n_local_experts = args.n_routed_experts // args.moe_ep_size
        moe_tp_id = global_rank % args.moe_tp_size
        moe_ep_id = global_rank // args.moe_tp_size
        param_dict = {name if "lm_head" in name else "model." + name: name_param
                      for name, name_param in self.named_parameters()}
        for _, param in self.named_parameters():
            param.requires_grad = False

        proj_size = args.la_num_heads * args.la_head_dim
        # column-parallel linear-attention projections (output-sharded); f_b_proj /
        # g_b_proj are matched before the shorter b_proj suffix to avoid collisions.
        cp_attn_keys = (
            ("attn.q_proj.weight", args.dim, proj_size),
            ("attn.k_proj.weight", args.dim, proj_size),
            ("attn.v_proj.weight", args.dim, proj_size),
            ("attn.f_b_proj.weight", args.la_head_dim, proj_size),
            ("attn.g_b_proj.weight", args.la_head_dim, proj_size),
            ("attn.b_proj.weight", args.dim, args.la_num_heads),
        )

        for name, loaded_weight in hf_model_weights_iterator(model_path):
            if "rotary_emb.inv_freq" in name or "g_idx" in name:
                continue
            if name == "rot.weight":
                continue
            if name.startswith("model.visual."):
                # Vision tower: load weights 1:1 into self.visual.* (no rename, no TP shard).
                # param_dict keys are "model.<named_param>" (the "model." prefix is added to
                # every named_parameters() name except lm_head), so the checkpoint key
                # "model.visual.blocks.0.attn.qkv.weight" matches the param_dict key directly.
                param = param_dict.get(name)
                if param is None:
                    logger.warning("vision: no param for %s", name)
                    continue
                with torch.no_grad():
                    param.copy_(convert_pyslice_to_tensor(loaded_weight).to(param.dtype))
                continue
            # strip "language_model." so names align with param_dict ("model.layers....")
            name = name.replace("model.language_model.", "model.")
            # skip MTP layer 45 (non-speculative generation)
            if name.startswith("model.layers."):
                layer_id = int(name.split(".")[2])
                if layer_id >= args.n_layers:
                    continue
            # expert EP skip
            if "experts" in name and "shared_experts" not in name:
                idx = int(name.split(".")[-3])
                if idx < moe_ep_id * n_local_experts or idx >= (moe_ep_id + 1) * n_local_experts:
                    continue

            # generic renames (mirror GLM-5.2 / deepseek_v3)
            name = name.replace("self_attn", "attn")
            name = name.replace("mlp", "ffn")
            name = name.replace("weight_scale_inv", "scale")
            name = name.replace("e_score_correction_bias", "bias")
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
            # GLM-5.3 linear-attention: forget_gate.* -> attn.*; conv1d.weight -> conv1d
            name = name.replace("forget_gate.", "")
            name = name.replace("q_conv1d.weight", "q_conv1d")
            name = name.replace("k_conv1d.weight", "k_conv1d")
            name = name.replace("v_conv1d.weight", "v_conv1d")

            if ".weight_offset" in name:  # W8A8 offset unused by the dynamic dequant path
                continue
            if not name.startswith("model."):
                name = "model." + name
            # GLM-5.3 attention is BF16 -> the static-quant guard below never skips an
            # attention weight; kept for parity with GLM-5.2 static-quant models.
            if (args.quantization == "w8a8" and "weight_scale" in name) or ".bias" in name:
                if any(s in name for s in ("q_a_proj", "kv_a_proj_with_mqa", "wq_b", "wo")):
                    continue
            name = name.replace("weight_scale", "scale")  # global per-channel scale rename

            # ---- fused wqkv_a (q_a_proj + kv_a_proj_with_mqa) ----
            is_wqkv_a = False
            for stride_id, weight_name in enumerate(["q_a_proj", "kv_a_proj_with_mqa"]):
                if weight_name not in name:
                    continue
                param_name = name.replace(weight_name, "wqkv_a")
                if param_name not in param_dict:
                    logger.warning("Loading model has no param named %s in checkpoints, bypass.", param_name)
                    break
                param = param_dict[param_name]
                if ".scale" in name:
                    if stride_id == 0:
                        param.data[:args.q_lora_rank, 0].copy_(loaded_weight[:].reshape(-1))
                    else:
                        param.data[args.q_lora_rank:, 0].copy_(loaded_weight[:].reshape(-1))
                else:
                    if stride_id == 0:
                        param.data[:args.q_lora_rank].copy_(loaded_weight[:])
                    else:
                        param.data[args.q_lora_rank:args.q_lora_rank + args.kv_lora_rank].copy_(loaded_weight[:])
                is_wqkv_a = True
                break
            if is_wqkv_a:
                continue

            # ---- gate_proj / up_proj -> w13 fusion (uniform weight + scale) ----
            is_gate_up = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param_name = name.replace(weight_name, "w13")
                if param_name not in param_dict:
                    logger.warning("Loading model has no param named %s in checkpoints, bypass.", param_name)
                    break
                param = param_dict[param_name]
                shard_size = param.shape[0] // 2
                if "shared_experts" in name:
                    lw = convert_pyslice_to_tensor(loaded_weight)
                    param.data[shard_size * stride_id:shard_size * (stride_id + 1)].copy_(lw)
                else:
                    gate_up_idx = moe_tp_id if "experts" in name else rank
                    lw = convert_pyslice_to_tensor(
                        loaded_weight[shard_size * gate_up_idx:shard_size * (gate_up_idx + 1)])
                    slot = param.data[shard_size * stride_id:shard_size * (stride_id + 1)]
                    slot.data[:lw.shape[0]].copy_(lw)
                is_gate_up = True
                break
            if is_gate_up:
                continue

            if name not in param_dict:
                logger.warning("Loading model has no param named %s in checkpoints, bypass.", name)
                continue
            param = param_dict[name]

            if "embed" in name:
                load_tensor_parallel_weights(param, loaded_weight, args.vocab_size, args.dim, name,
                                             True, False, rank, world_size)
                continue
            if "head" in name:
                load_tensor_parallel_weights(param, loaded_weight, args.dim, args.vocab_size, name,
                                             False, True, rank, world_size)
                continue
            # MLA wq_b (non-indexer): column-parallel, shard output
            if "wq_b" in name and "indexer" not in name:
                load_tensor_parallel_weights(param, loaded_weight, args.q_lora_rank,
                                             args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim),
                                             name, False, True, rank, world_size)
                continue
            # indexer wq_b: replicated (full copy)
            if "wq_b" in name and "indexer" in name:
                param.data.copy_(convert_pyslice_to_tensor(loaded_weight))
                continue
            if "wkv_b" in name:
                load_tensor_parallel_weights(param, loaded_weight, args.kv_lora_rank,
                                             args.n_heads * (args.qk_nope_head_dim + args.v_head_dim),
                                             name, False, True, rank, world_size)
                continue
            if "wo" in name:
                # row-parallel; full input dim differs by layer type (proj_size for
                # linear-attention, v*n_heads for MLA) -> derive from the param shape.
                in_f = param.shape[1] * world_size
                load_tensor_parallel_weights(param, loaded_weight, in_f, args.dim, name,
                                             True, True, rank, world_size)
                continue

            # ---- column-parallel linear-attention projections (output-sharded) ----
            matched_cp = False
            for suffix, in_f, out_f in cp_attn_keys:
                if name.endswith(suffix):
                    load_tensor_parallel_weights(param, loaded_weight, in_f, out_f, name,
                                                 False, True, rank, world_size)
                    matched_cp = True
                    break
            if matched_cp:
                continue

            # ---- w2 (dense row-parallel / shared full / expert moe_tp-shard) ----
            if "w2" in name and (int(name.split(".")[2]) < args.n_dense_layers):
                if ".scale" in name:
                    param.data.copy_(convert_pyslice_to_tensor(loaded_weight))
                else:
                    load_tensor_parallel_weights(param, loaded_weight, args.inter_dim, args.dim, name,
                                                 True, True, rank, world_size)
                continue
            if "w2" in name and "shared_experts" in name:
                param.data.copy_(convert_pyslice_to_tensor(loaded_weight))
                continue
            if "w2" in name and "experts" in name:
                shard_size = param.shape[1]
                lw = convert_pyslice_to_tensor(
                    loaded_weight[:, shard_size * moe_tp_id:shard_size * (moe_tp_id + 1)])
                param.data[:, :lw.shape[1]].copy_(lw)
                continue

            # ---- default: full copy (MHC fn/base/scale, conv1d, A_log, dt_bias, o_norm,
            #      f_a_proj, g_a_proj, indexer wk/weights_proj/k_norm, kpool gate/ape,
            #      norms, gate.weight/bias) ----
            param.data.copy_(convert_pyslice_to_tensor(loaded_weight))

        torch.npu.empty_cache()
