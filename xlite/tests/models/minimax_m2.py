#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096
    max_num_batched_tokens: int = 4096
    dim: int = 3072
    head_dim: int = 128
    inter_dim: int = 1536
    moe_inter_dim: int = 1536
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 0
    n_routed_experts: int = 256
    n_shared_experts: int = 0
    n_activated_experts: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    vocab_size: int = 200064
    n_layers: int = 62
    n_heads: int = 48
    n_kv_heads: int = 8
    norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    partial_rotary_factor: float = 0.5
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    qkv_bias: bool = False
    qk_norm: bool = True
    qk_norm_full: bool = True
    moe_ep_size: int = 1
    moe_tp_size: int = 1
    model_type: str = "minimax_m2"

    def __post_init__(self):
        self.max_num_batched_tokens = self.max_seq_len * self.max_batch_size
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads