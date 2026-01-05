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
from xlite._C import Runtime, decode_attention

rt = Runtime(0, 500)
torch.npu.set_device(0)

num_heads = 4
num_kv_heads = 4
max_batch_size = 2
batch_size = 2
max_seq_len = 150000
head_size = 128
block_size = 128
max_block_num = batch_size * ((max_seq_len + block_size - 1) // block_size)
hidden_size = 512

q_size = num_heads * head_size
kv_size = num_kv_heads * head_size

supported_dtype_list = [
    (torch.float16, 2e-4, 2e-3),
    (torch.bfloat16, 2e-3, 2e-2)
]
seq_len_list = [4000, 6000, 16000, 24000, 33000, 60000, 120000, 140000]

a2v = torch.zeros(int(num_heads * max_batch_size * 512 / 4), dtype=torch.int32, device="npu:0")
v2a = torch.zeros(int(num_heads * max_batch_size * 512 / 4), dtype=torch.int32, device="npu:0")
for dtype, atol, rtol in supported_dtype_list:
    for seq_len in seq_len_list:
        decode_idx = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int32, device="npu:0")
        cum_prompt_lens = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int32, device="npu:0")
        cached_lens = torch.tensor([seq_len - 1] * batch_size + [0] * (max_batch_size - batch_size),
                                dtype=torch.int32, device="npu:0")
        block_tables = torch.tensor([i for i in range(max_block_num)],
                                    dtype=torch.int32, device="npu:0")

        q_xlite = torch.randn(batch_size, q_size + 2 * kv_size, dtype=dtype, device="npu:0").clamp(-5.0, 5.0)
        k_cache_xlite = torch.randn(max_block_num, block_size, num_kv_heads, head_size, dtype=dtype, device="npu:0")
        v_cache_xlite = torch.randn(max_block_num, block_size, num_kv_heads, head_size, dtype=dtype, device="npu:0")
        output = torch.empty(batch_size, hidden_size, dtype=dtype, device="npu:0")

        q_standard = q_xlite[:, :q_size].clone().view(batch_size, 1, num_heads, head_size)
        k_cache_standard = torch.randn(batch_size, seq_len, num_kv_heads, head_size,
                                    dtype=dtype, device="npu:0").clamp(-5.0, 5.0)
        v_cache_standard = torch.randn(batch_size, seq_len, num_kv_heads, head_size,
                                    dtype=dtype, device="npu:0").clamp(-5.0, 5.0)
        iter_num = int((seq_len + 127) / 128)
        for i in range(batch_size):
            for j in range(iter_num):
                start_offset = j * block_size
                if j + 1 != iter_num:
                    end_offset = block_size
                else:
                    end_offset = seq_len - j * block_size
                k_cache_xlite[i * iter_num + j, :end_offset] = k_cache_standard[i, start_offset:start_offset + end_offset]
                v_cache_xlite[i * iter_num + j, :end_offset] = v_cache_standard[i, start_offset:start_offset + end_offset]

        scores = torch.matmul(
            q_standard.transpose(1, 2),  # [bsz, n_local_heads, seqlen, head_dim]
            k_cache_standard.permute(0, 2, 3, 1)  # [bsz, n_local_heads, head_dim, seqlen+start_pos]
        )

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        standard = torch.matmul(attn_weights, v_cache_standard.transpose(1, 2)).reshape(batch_size, hidden_size)

        torch.npu.synchronize()
        decode_attention(rt, a2v, v2a, q_xlite, k_cache_xlite, v_cache_xlite, cached_lens, block_tables,
                            output, decode_idx, cum_prompt_lens,
                            batch_size, num_heads, head_size, block_size, iter_num, num_kv_heads, max_seq_len)
        torch.npu.synchronize()
        print(f'decode attention mix (seq_len={seq_len}, {dtype}) executed!')

        try:
            torch.testing.assert_close(standard, output, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f'{e}')
            print(f'torch_npu({dtype}): {standard}')
            print(f'xlite({dtype}): {output}')
