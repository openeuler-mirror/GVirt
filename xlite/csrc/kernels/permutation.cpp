/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
#define TOKEN_TILE 4096

__aicore__ inline void permutation1(GM_ADDR input, GM_ADDR routing_map, GM_ADDR output,
                                    GM_ADDR unp_idx, GM_ADDR counts, uint32_t n_tokens,
                                    uint32_t dim, uint32_t n_routed_experts,
                                    uint32_t experts_start_idx, uint32_t experts_end_idx)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __ubuf__ uint32_t *c = (__ubuf__ uint32_t *)get_imm(0);
    __ubuf__ uint64_t *routing = (__ubuf__ uint64_t *)get_imm(n_routed_experts * sizeof(uint32_t));
    __ubuf__ uint32_t *routing_count =
        (__ubuf__ uint32_t *)get_imm(n_routed_experts * sizeof(uint32_t) +
                                     ROUND_UP(TOKEN_TILE * n_routed_experts / 8, BLOCK_SIZE));

    uint32_t experts_local = (experts_end_idx - experts_start_idx) / block_num;
    uint32_t experts_remain = (experts_end_idx - experts_start_idx) % block_num;
    uint32_t local_start, local_end;
    if (experts_remain == 0) {
        local_start = experts_start_idx + experts_local * block_idx;
        local_end = local_start + experts_local;
    } else if (block_idx < experts_remain) {
        local_start = experts_start_idx + experts_local * block_idx + block_idx;
        local_end = local_start + experts_local + 1;
    } else {
        local_start = experts_start_idx + experts_local * block_idx + experts_remain;
        local_end = local_start + experts_local;
    }
    if (block_idx == 0) {
        local_start = 0;
    }
    if (block_idx == block_num - 1) {
        local_end = n_routed_experts;
    }

    vector_dup(c, 0, DIV_ROUND_UP(n_routed_experts * sizeof(uint32_t), VECTOR_MAX_BYTESIZE), 1, 1,
               8, 1);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    for (uint32_t token_start = 0; token_start < n_tokens; token_start += TOKEN_TILE) {
        GM_ADDR routing_map_tile = routing_map + token_start * n_routed_experts / 8;
        uint32_t token_tile =
            (token_start + TOKEN_TILE > n_tokens) ? n_tokens - token_start : TOKEN_TILE;
        copy_gm_to_ubuf(routing, routing_map_tile, 0, 1,
                        DIV_ROUND_UP(token_tile * n_routed_experts / 8, BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        for (uint32_t curr_expert = local_start; curr_expert < local_end; curr_expert++) {
            if (curr_expert < experts_start_idx || curr_expert >= experts_end_idx) {
                continue;
            }

            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            for (uint32_t i = 0; i < token_tile; i++) {
                if (bitmapTest(routing, i * n_routed_experts + curr_expert)) {
                    routing_count[i] = c[curr_expert - local_start];
                    c[curr_expert - local_start]++;
                }
            }
            GM_ADDR unp_idx_tile = unp_idx + curr_expert * n_tokens * sizeof(uint32_t) +
                                   token_start * sizeof(uint32_t);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm_align_b32(unp_idx_tile, routing_count, 0, 1,
                                      token_tile * sizeof(uint32_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        }
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }

    if (local_end > local_start) {
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        GM_ADDR counts_local = counts + local_start * sizeof(uint32_t);
        copy_ubuf_to_gm_align_b32(counts_local, c, 0, 1,
                                  (local_end - local_start) * sizeof(uint32_t), 0, 0, 0, 0);
    }

    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void permutation2(GM_ADDR input, GM_ADDR routing_map, GM_ADDR output,
                                    GM_ADDR unp_idx, GM_ADDR counts, uint32_t n_tokens,
                                    uint32_t dim, uint32_t n_routed_experts,
                                    uint32_t experts_start_idx, uint32_t experts_end_idx)
{
    uint32_t sum = 0;
    __ubuf__ uint32_t *c = (__ubuf__ uint32_t *)get_imm(0);
    __ubuf__ uint32_t *s = (__ubuf__ uint32_t *)get_imm(
        ROUND_UP(n_routed_experts * sizeof(uint32_t), VECTOR_MAX_BYTESIZE));
    __ubuf__ uint32_t *psum = (__ubuf__ uint32_t *)get_imm(
        2 * ROUND_UP(n_routed_experts * sizeof(uint32_t), VECTOR_MAX_BYTESIZE));

    copy_gm_to_ubuf(c, counts, 0, 1, DIV_ROUND_UP(n_routed_experts * sizeof(uint32_t), BLOCK_SIZE),
                    0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    for (uint32_t i = 0; i < n_routed_experts; i++) {
        uint32_t count = *(c + i);
        *(s + i) = sum;
        sum += count;
    }
    *psum = sum;

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm_align_b32(unp_idx + n_routed_experts * n_tokens * sizeof(uint32_t), s, 0, 1,
                              n_routed_experts * sizeof(uint32_t), 0, 0, 0, 0);
    copy_ubuf_to_gm_align_b32(unp_idx, psum, 0, 1, sizeof(uint32_t), 0, 0, 0, 0);

    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void permutation3(GM_ADDR input, GM_ADDR routing_map, GM_ADDR output,
                                    GM_ADDR unp_idx, GM_ADDR counts, uint32_t n_tokens,
                                    uint32_t dim, uint32_t n_routed_experts,
                                    uint32_t experts_start_idx, uint32_t experts_end_idx)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    GM_ADDR starts = unp_idx + n_routed_experts * n_tokens * sizeof(uint32_t);
    uint64_t routing_dmi_config =
        __set_dmi_config(0, 1, DIV_ROUND_UP(n_routed_experts / 8, BLOCK_SIZE), 0, 0);
    uint64_t dmi_config =
        __set_dmi_config(0, 1, DIV_ROUND_UP(dim * sizeof(bfloat16_t), BLOCK_SIZE), 0, 0);
    uint32_t curr_expert, curr_token;
    int first_iter;
    int curr = 0;

    __ubuf__ uint32_t *s = (__ubuf__ uint32_t *)get_imm(0);
    __ubuf__ uint64_t *routing1 = (__ubuf__ uint64_t *)get_imm(s + n_routed_experts);
    __ubuf__ uint64_t *routing2 = (__ubuf__ uint64_t *)get_imm(
        routing1 + ROUND_UP(n_routed_experts / 8, BLOCK_SIZE) / sizeof(uint64_t));
    __ubuf__ uint64_t *routing[2] = {routing1, routing2};
    __ubuf__ uint8_t *row1 = (__ubuf__ uint8_t *)get_imm(
        routing2 + ROUND_UP(n_routed_experts / 8, BLOCK_SIZE) / sizeof(uint64_t));
    __ubuf__ uint8_t *row2 =
        (__ubuf__ uint8_t *)get_imm(row1 + ROUND_UP(dim * sizeof(bfloat16_t), BLOCK_SIZE));
    __ubuf__ uint8_t *row[2] = {row1, row2};

    copy_gm_to_ubuf_align_b32(s, starts, 0, 1, n_routed_experts * sizeof(uint32_t), 0, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (curr_token = block_idx; curr_token < n_tokens; curr_token += block_num) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, curr);
        copy_gm_to_ubuf(routing[curr],
                        (__gm__ uint8_t *)routing_map + (curr_token * n_routed_experts / 8),
                        routing_dmi_config);
        set_flag(PIPE_MTE2, PIPE_S, curr);
        wait_flag(PIPE_MTE2, PIPE_S, curr);

        first_iter = 1;
        for (curr_expert = experts_start_idx; curr_expert < experts_end_idx; curr_expert++) {
            if (!bitmapTest(routing[curr], curr_expert)) {
                continue;
            }
            if (first_iter) {
                copy_gm_to_ubuf(row[curr], input + (curr_token)*dim * sizeof(bfloat16_t),
                                dmi_config);
                set_flag(PIPE_MTE2, PIPE_MTE3, curr);
                wait_flag(PIPE_MTE2, PIPE_MTE3, curr);
                first_iter = 0;
            }
            uint32_t off_T =
                curr_expert * n_tokens * sizeof(uint32_t) + curr_token * sizeof(uint32_t);
            uint32_t j = off_T == 0 ? 0 : *((__gm__ uint32_t *)(unp_idx + off_T));
            copy_ubuf_to_gm(output + (s[curr_expert] + j) * dim * sizeof(bfloat16_t), row[curr],
                            dmi_config);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, curr);
        curr = 1 - curr;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ void permutation(GM_ADDR input, GM_ADDR routing_map,
                                                  GM_ADDR output, GM_ADDR unp_idx, GM_ADDR counts,
                                                  uint32_t n_tokens, uint32_t dim,
                                                  uint32_t n_routed_experts,
                                                  uint32_t experts_start_idx,
                                                  uint32_t experts_end_idx)
{
    uint64_t flag_id = 0;
    uint64_t mode = 0;
    uint64_t config = 1 | (mode << 4) | (flag_id << 8);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    permutation1(input, routing_map, output, unp_idx, counts, n_tokens, dim, n_routed_experts,
                 experts_start_idx, experts_end_idx);
    ffts_cross_core_sync(PIPE_MTE3, config);
    wait_flag_dev(flag_id);

    if (block_idx == 0) {
        permutation2(input, routing_map, output, unp_idx, counts, n_tokens, dim, n_routed_experts,
                     experts_start_idx, experts_end_idx);
    }
    ffts_cross_core_sync(PIPE_MTE3, config);
    wait_flag_dev(flag_id);

    permutation3(input, routing_map, output, unp_idx, counts, n_tokens, dim, n_routed_experts,
                 experts_start_idx, experts_end_idx);
}
#endif