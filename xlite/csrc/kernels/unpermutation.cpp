/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void unpermutation(GM_ADDR input, GM_ADDR routing_map, GM_ADDR output,
                                                    GM_ADDR unp_idx, GM_ADDR weights_map,
                                                    uint32_t n_tokens, uint32_t dim, uint32_t n_routed_experts,
                                                    uint32_t experts_start_idx, uint32_t experts_end_idx)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    GM_ADDR starts = unp_idx + n_routed_experts * n_tokens * sizeof(uint32_t);
    uint32_t curr_token;

    uint64_t routing_dmi_config = __set_dmi_config(0, 1, DIV_ROUND_UP(n_routed_experts / 8, BLOCK_SIZE), 0, 0);
    uint64_t dmi_config = __set_dmi_config(0, 1, DIV_ROUND_UP(dim * sizeof(bfloat16_t), BLOCK_SIZE), 0, 0);
    uint32_t repeats = DIV_ROUND_UP(dim * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE);
    uint32_t repeats32 = DIV_ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE);

    __ubuf__ uint8_t *row = (__ubuf__ uint8_t *)get_imm(0);
    __ubuf__ uint8_t *rowf32 = (__ubuf__ uint8_t *)get_imm(ROUND_UP(dim * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE));
    __ubuf__ uint8_t *row1 = (__ubuf__ uint8_t *)get_imm(ROUND_UP(dim * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE) + ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE));
    __ubuf__ uint8_t *row2 = (__ubuf__ uint8_t *)get_imm(ROUND_UP(dim * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE) + 2 * ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE));
    __ubuf__ uint8_t *row3 = (__ubuf__ uint8_t *)get_imm(ROUND_UP(dim * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE) + 3 * ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE));
    __ubuf__ uint64_t *routing = (__ubuf__ uint64_t *)get_imm(ROUND_UP(dim * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE) + 3 * ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE) + ROUND_UP(dim * sizeof(bfloat16_t), VECTOR_MAX_BYTESIZE));

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (curr_token = block_idx; curr_token < n_tokens; curr_token += block_num) {
        copy_gm_to_ubuf(routing, (__gm__ uint8_t *)routing_map + (curr_token * n_routed_experts / 8), routing_dmi_config);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        vector_dup((__ubuf__ float *)(row2), (float)(0), repeats32, 1, 0, 8, 0);
        vector_dup((__ubuf__ bfloat16_t *)(row3), (bfloat16_t)(0), repeats, 1, 0, 8, 0);
        for (uint32_t i = experts_start_idx; i < experts_end_idx; i++) {
            if (bitmapTest(routing, i)) {
                uint32_t start = *((__gm__ uint32_t *)(starts + i * sizeof(uint32_t)));
                uint32_t off_T = i * n_tokens * sizeof(uint32_t) + curr_token * sizeof(uint32_t);
                uint32_t j = off_T == 0 ? 0 : *((__gm__ uint32_t *)(unp_idx + off_T));
                float w = *((__gm__ float *)(weights_map + (curr_token * n_routed_experts + i) * sizeof(float)));
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                copy_gm_to_ubuf(row, input + (start + j) * dim * sizeof(bfloat16_t), dmi_config);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                vconv_bf162f32((__ubuf__ float *)rowf32, (__ubuf__ bfloat16_t *)row, repeats32, 1, 1, 8, 4);
                vmuls((__ubuf__ float *)row1, (__ubuf__ float *)rowf32, w, repeats32, 1, 1, 8, 8);
                vadd((__ubuf__ float *)row2, (__ubuf__ float *)row1, (__ubuf__ float *)row2, repeats32, 1, 1, 1, 8, 8, 8);
                vconv_f322bf16r((__ubuf__ bfloat16_t *)row3, (__ubuf__ float *)row2, repeats32, 1, 1, 4, 8);
            }
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(output + curr_token * dim * sizeof(bfloat16_t), row3, dmi_config);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
