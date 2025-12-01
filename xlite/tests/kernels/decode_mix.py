#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import Runtime, decode_attention_mix

rt = Runtime(0, 500)
torch.npu.set_device(0)

num_heads = 32
num_kv_heads = 32
max_batch_size = 8
batch_size = 4
max_seq_len = 1024
head_size = 128
block_size = 128
seq_len = 90
max_block_num = 64
hidden_size = 4096

q_size = num_heads * head_size
kv_size = num_kv_heads * head_size

a2v = torch.empty(int(num_heads * max_batch_size * 512 / 4), dtype=torch.int32, device="npu:0")
v2a = torch.empty(int(num_heads * max_batch_size * 512 / 4), dtype=torch.int32, device="npu:0")
cum_prompt_lens = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int32, device="npu:0")
cached_lens = torch.tensor([seq_len - 1] * batch_size + [0] * (max_batch_size - batch_size),
                           dtype=torch.int32, device="npu:0")
block_tables = torch.tensor([i * max_batch_size for i in range(batch_size)] + [0] * (max_block_num - batch_size),
                            dtype=torch.int32, device="npu:0")

q_xlite = torch.randn(batch_size, q_size + 2 * kv_size, dtype=torch.float16, device="npu:0").clamp(-5.0, 5.0)
k_cache_xlite = torch.zeros(max_block_num, block_size, num_heads, head_size, dtype=torch.float16, device="npu:0")
v_cache_xlite = torch.zeros(max_block_num, block_size, num_heads, head_size, dtype=torch.float16, device="npu:0")
output = torch.empty(batch_size, hidden_size, dtype=torch.float16, device="npu:0")

q_standard = q_xlite[:, :q_size].clone().view(batch_size, 1, num_heads, head_size)
k_cache_standard = torch.randn(batch_size, seq_len, num_heads, head_size,
                               dtype=torch.float16, device="npu:0").clamp(-5.0, 5.0)
v_cache_standard = torch.randn(batch_size, seq_len, num_heads, head_size,
                               dtype=torch.float16, device="npu:0").clamp(-5.0, 5.0)
for i in range(batch_size):
    k_cache_xlite[i * max_batch_size, :seq_len] = k_cache_standard[i]
    v_cache_xlite[i * max_batch_size, :seq_len] = v_cache_standard[i]

scores = torch.matmul(
    q_standard.transpose(1, 2),  # [bsz, n_local_heads, seqlen, head_dim]
    k_cache_standard.permute(0, 2, 3, 1),  # [bsz, n_local_heads, head_dim, seqlen+start_pos]
)

attn_weights = torch.nn.functional.softmax(scores, dim=-1)
standard = torch.matmul(attn_weights, v_cache_standard.transpose(1, 2)).reshape(batch_size, hidden_size)

torch.npu.synchronize()
decode_attention_mix(rt, a2v, v2a, q_xlite, k_cache_xlite, v_cache_xlite, cached_lens, block_tables,
                     output, cum_prompt_lens,
                     batch_size, num_heads, head_size, block_size, 1, num_kv_heads, max_seq_len)
torch.npu.synchronize()
print(f'decode attention mix executed!')

try:
    torch.testing.assert_close(standard, output, atol=1e-3, rtol=1e-2)
except AssertionError as e:
    print(f'{e}')
    print(f'torch_npu: {standard}')
    print(f'xlite: {output}')
