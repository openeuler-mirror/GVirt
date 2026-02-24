/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

extern "C" __global__ __aicore__ void silu_and_mul_bfloat16_t(GM_ADDR x, GM_ADDR y, GM_ADDR pm,
                                                              int n_tokens, int dim)
{
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_atomic_none();

    if (pm) {
        uint32_t pm_val = *((__gm__ uint32_t *)pm);
        n_tokens = pm_val < n_tokens ? pm_val : n_tokens;
    }
    int calc_pad = VECTOR_MAX_BYTESIZE / sizeof(float);
    int step = dim * 2;
    int dim_split = dim;
    int max_dim = DIV_ROUND_UP(UB_SIZE, 5 * sizeof(float) + 5 * sizeof(bfloat16_t));
    int split = 1;
    while (dim_split > max_dim) {
        dim_split = dim_split / 2;
        split = split * 2;
    }
    int padded_dim = ROUND_UP(dim_split, calc_pad);
    __ubuf__ float *x32_ub0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)0);
    __ubuf__ float *x32_ub1 = reinterpret_cast<__ubuf__ float *>(x32_ub0 + 2 * padded_dim);
    __ubuf__ float *tmp = reinterpret_cast<__ubuf__ float *>(x32_ub0 + 4 * padded_dim);
    __ubuf__ bfloat16_t *output_ub = reinterpret_cast<__ubuf__ bfloat16_t *>(tmp + padded_dim);
    __ubuf__ bfloat16_t *x32_ub0_bf16 =
        reinterpret_cast<__ubuf__ bfloat16_t *>(output_ub + padded_dim);
    __ubuf__ bfloat16_t *x32_ub1_bf16 =
        reinterpret_cast<__ubuf__ bfloat16_t *>(x32_ub0_bf16 + 2 * padded_dim);

    int burst_copy = DIV_ROUND_UP(dim_split * sizeof(bfloat16_t), BLOCK_SIZE);
    int repeat_cal = DIV_ROUND_UP(dim_split, calc_pad);

    int event_id = 0;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int index = block_idx; index < n_tokens * split; index += block_num) {
        int row = index / split;
        int line = index % split;
        __gm__ bfloat16_t *gm_x_first_half =
            (__gm__ bfloat16_t *)(x) + row * step + line * dim_split;
        __gm__ bfloat16_t *gm_x_second_half =
            (__gm__ bfloat16_t *)(x) + row * step + line * dim_split + dim;
        __gm__ bfloat16_t *gm_y = (__gm__ bfloat16_t *)(y) + row * dim + line * dim_split;
        auto x32_ub = event_id == 0 ? x32_ub0 : x32_ub1;
        auto x32_ub_bf16 = event_id == 0 ? x32_ub0_bf16 : x32_ub1_bf16;

        wait_flag(PIPE_V, PIPE_MTE2, event_id);
        copy_gm_to_ubuf(x32_ub_bf16, gm_x_first_half, 0, 1, burst_copy, 0, 0);
        copy_gm_to_ubuf(x32_ub_bf16 + padded_dim, gm_x_second_half, 0, 1, burst_copy, 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_bf162f32(x32_ub, x32_ub_bf16, 2 * repeat_cal, 1, 1, 8, 4);
        set_flag(PIPE_V, PIPE_MTE2, event_id);
        pipe_barrier(PIPE_V);

        if (dim_split % calc_pad != 0) {
            SetMaskFromHighBit(calc_pad, calc_pad - dim_split % calc_pad);
            vector_dup(x32_ub + ROUND_DOWN(dim_split, calc_pad), float(0), 1, 1, 1, 8, 0);
            vector_dup(x32_ub + padded_dim + ROUND_DOWN(dim_split, calc_pad), float(0), 1, 1, 1, 8,
                       0);
            pipe_barrier(PIPE_V);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }

        // -x
        vmuls(tmp, x32_ub, float(-1.0000000000000000e+00f), repeat_cal, 1, 1, 8,
              8);  // 处理前dim个数
        pipe_barrier(PIPE_V);

        // exp(-x)
        vexp(tmp, tmp, repeat_cal, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // 1 + exp(-x)
        vadds(tmp, tmp, float(1.0), repeat_cal, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // x * sigmoid(x) = x / (1 + exp(-x))
        vdiv(tmp, x32_ub, tmp, repeat_cal, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // mul
        vmul(tmp, tmp, x32_ub + padded_dim, repeat_cal, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        vconv_f322bf16r(output_ub, tmp, repeat_cal, 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(gm_y, output_ub, 0, 1, burst_copy, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        event_id = 1 - event_id;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_ALL);
}

#endif