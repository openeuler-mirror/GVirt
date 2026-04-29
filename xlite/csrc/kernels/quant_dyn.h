/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

__aicore__ inline void quant_bf16_to_i8(GM_ADDR x, GM_ADDR scales, GM_ADDR z, GM_ADDR pnum_tokens,
                                        uint32_t m, uint32_t k)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (pnum_tokens) {
        uint32_t pnum_tokens_val = *((__gm__ uint32_t *)pnum_tokens);
        m = pnum_tokens_val < m ? pnum_tokens_val : m;
    }

    uint32_t k_pad = ROUND_UP(k, (256 / sizeof(bfloat16_t)));

    auto *x1 = reinterpret_cast<__ubuf__ bfloat16_t *>((uintptr_t)0);
    auto *x2 = reinterpret_cast<__ubuf__ bfloat16_t *>(x1 + k_pad);
    auto *xf32 = reinterpret_cast<__ubuf__ float *>(x2 + k_pad);
    auto *xAbs = reinterpret_cast<__ubuf__ float *>(xf32 + k_pad);
    auto *zf16 = reinterpret_cast<__ubuf__ half *>(xAbs + k_pad);
    auto *z1 = reinterpret_cast<__ubuf__ int8_t *>(zf16 + k_pad);
    auto *z2 = reinterpret_cast<__ubuf__ int8_t *>(z1 + k_pad);
    auto *scaleUb = reinterpret_cast<__ubuf__ float *>(z2 + k_pad);
    auto *sum_addr = reinterpret_cast<__ubuf__ float *>(scaleUb + 1);
    assert((uint64_t)sum_addr <= UB_SIZE);

    __ubuf__ bfloat16_t *xBufs[2] = {x1, x2};
    __ubuf__ int8_t *zBufs[2] = {z1, z2};

    int eventId = 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    for (int row = block_idx; row < m; row += block_num) {
        int rowOffset = row * k;

        wait_flag(PIPE_V, PIPE_MTE2, eventId);
        copy_gm_to_ubuf_align_b16(xBufs[eventId],
                                  reinterpret_cast<__gm__ bfloat16_t *>(x) + rowOffset, 0, 1,
                                  k * sizeof(bfloat16_t), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, eventId);

        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        vconv_bf162f32(xf32, xBufs[eventId], k_pad / VECTOR_MAX_NUM_OF_FP32, 1, 1, 8, 4);
        set_flag(PIPE_V, PIPE_MTE2, eventId);

        pipe_barrier(PIPE_V);
        vabs(xAbs, xf32, k_pad / VECTOR_MAX_NUM_OF_FP32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        ReduceMax(xAbs, xAbs, k);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);

        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float scale = float(*xAbs) / float(127);
        float scaleRec = float(127) / float(*xAbs);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        *scaleUb = float(scale);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b32(reinterpret_cast<__gm__ float *>(scales) + row, scaleUb, 0, 1,
                                  sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

        vmuls(xf32, xf32, float(scaleRec), k_pad / VECTOR_MAX_NUM_OF_FP32, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(zf16, xf32, k_pad / VECTOR_MAX_NUM_OF_FP32, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, eventId);
        vconv_f162s8(zBufs[eventId], zf16, k_pad / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE3, eventId);

        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        copy_ubuf_to_gm_align_b8(z + rowOffset, zBufs[eventId], 0, 1, k, 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, eventId);

        eventId = 1 - eventId;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define QUANT_DYN_FUNC_DEFINE(dtype)                                                         \
    extern "C" __global__ __aicore__ void quant_bf16_to_i8_dynamic(                          \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t k) \
    {                                                                                        \
        quant_bf16_to_i8(in, scale, out, pnum_tokens, m, k);                                 \
    }
#else
#define QUANT_DYN_FUNC_DEFINE(dtype)                                                         \
    extern "C" __global__ __aicore__ void quant_bf16_to_i8_dynamic(                          \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t k) \
    {                                                                                        \
    }

#endif
