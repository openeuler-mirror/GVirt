/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

__aicore__ inline void experts_counts_sum_kernel(GM_ADDR experts_counts_input,
                                                 GM_ADDR tokens_per_epgroup,
                                                 GM_ADDR experts_counts_output,
                                                 uint32_t n_routed_experts, uint32_t ep_size)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    uint32_t experts_ep_group = n_routed_experts / ep_size;

    // ----- work distribution -----
    uint32_t row_start, row_end;
    {
        uint32_t remain = ep_size % block_num;
        uint32_t avg = ep_size / block_num;
        if (block_idx < remain) {
            row_start = block_idx * avg + block_idx;
            row_end = row_start + avg + 1;
        } else {
            row_start = block_idx * avg + remain;
            row_end = row_start + avg;
        }
    }

    // pick Phase 2 block: first idle block, or block 0 if none idle
    uint32_t phase2Block = (ep_size < block_num) ? ep_size : 0;

    // ----- Phase 1: compute tokens_per_epgroup -----
    if (row_start < ep_size) {
        uint32_t row_bytes = n_routed_experts * sizeof(uint32_t);
        uint32_t ep_counts_row_bytes = ep_size * sizeof(uint32_t);

        uint64_t off = 0;
        __ubuf__ uint32_t *c_row = (__ubuf__ uint32_t *)((uintptr_t)off);
        off += ROUND_UP(row_bytes, VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *ep_counts_row = (__ubuf__ uint32_t *)((uintptr_t)off);

        for (uint32_t ep_idx = row_start; ep_idx < row_end; ep_idx++) {
            __gm__ uint32_t *input_row =
                (__gm__ uint32_t *)experts_counts_input + ep_idx * n_routed_experts;
            copy_gm_to_ubuf(c_row, input_row, 0, 1, DIV_ROUND_UP(row_bytes, BLOCK_SIZE), 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

            vector_dup(ep_counts_row, uint32_t(0),
                       DIV_ROUND_UP(ep_size * sizeof(uint32_t), VECTOR_MAX_BYTESIZE), 1, 0, 8, 0);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            for (uint32_t curr_expert = 0; curr_expert < n_routed_experts; curr_expert++) {
                uint32_t ep_id = curr_expert / experts_ep_group;
                ep_counts_row[ep_id] += c_row[curr_expert];
            }

            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            __gm__ uint32_t *epg_row = (__gm__ uint32_t *)tokens_per_epgroup + ep_idx * ep_size;
            copy_ubuf_to_gm_align_b32(epg_row, ep_counts_row, 0, 1, ep_size * sizeof(uint32_t), 0,
                                      0, 0, 0);
        }
    }

    // ----- Phase 2: one block computes experts_counts_output -----
    if (block_idx == phase2Block) {
        uint32_t input_bytes = ep_size * n_routed_experts * sizeof(uint32_t);

        uint64_t off = 0;
        __ubuf__ uint32_t *c = (__ubuf__ uint32_t *)((uintptr_t)off);
        off += ROUND_UP(input_bytes, VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *experts_counts = (__ubuf__ uint32_t *)((uintptr_t)off);

        copy_gm_to_ubuf(c, experts_counts_input, 0, 1, DIV_ROUND_UP(input_bytes, BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        vector_dup(experts_counts, uint32_t(0),
                   DIV_ROUND_UP(n_routed_experts * sizeof(uint32_t), VECTOR_MAX_BYTESIZE), 1, 0, 8,
                   0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        for (uint32_t ep_idx = 0; ep_idx < ep_size; ep_idx++) {
            __ubuf__ uint32_t *c_row = c + ep_idx * n_routed_experts;
            for (uint32_t curr_expert = 0; curr_expert < n_routed_experts; curr_expert++) {
                experts_counts[curr_expert] += c_row[curr_expert];
            }
        }

        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b32(experts_counts_output, experts_counts, 0, 1,
                                  n_routed_experts * sizeof(uint32_t), 0, 0, 0, 0);
        pipe_barrier(PIPE_ALL);
    }
}

extern "C" __global__ __aicore__ void experts_counts_sum(GM_ADDR experts_counts_input,
                                                         GM_ADDR tokens_per_epgroup,
                                                         GM_ADDR experts_counts_output,
                                                         uint32_t n_routed_experts,
                                                         uint32_t ep_size)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    experts_counts_sum_kernel(experts_counts_input, tokens_per_epgroup, experts_counts_output,
                              n_routed_experts, ep_size);
}
#endif