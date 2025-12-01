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
from typing import Literal


@dataclass
class Qwen2ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096
    max_m: int = 4096
    dim: int = 5120
    head_dim: int = None
    inter_dim: int = 27648
    vocab_size: int = 152064
    n_layers: int = 64
    n_heads: int = 40
    n_kv_heads: int = 8
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    qkv_bias: bool = True
    qk_norm: bool = False
    model_type: str = "qwen2"

    def __post_init__(self):
        self.max_m = self.max_seq_len * self.max_batch_size
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
