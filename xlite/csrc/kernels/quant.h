/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"


#ifdef __DAV_C220_VEC__

// out = int8(in * scale_reciprocal + offset)
// per-token quantion
// [m, k], activation
__aicore__ inline void quant_bf16_to_i8(
    GM_ADDR in, GM_ADDR scale_reciprocal, GM_ADDR offset, GM_ADDR out, uint32_t m, uint32_t k)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ bfloat16_t *in_buf = reinterpret_cast<__gm__ bfloat16_t *>(in);
    __gm__ float *scale_reciprocal_gm = reinterpret_cast<__gm__ float *>(scale_reciprocal);
    __gm__ float *offset_gm = reinterpret_cast<__gm__ float *>(offset);
    __gm__ int8_t *out_buf = reinterpret_cast<__gm__ int8_t *>(out);

    auto *x_ping = reinterpret_cast<__ubuf__ bfloat16_t *>((uintptr_t)0);
    auto *x_pong = reinterpret_cast<__ubuf__ bfloat16_t *>(x_ping + k);

    auto *xf32_ping = reinterpret_cast<__ubuf__ float *>(x_pong + k);
    auto *xf32_pong = reinterpret_cast<__ubuf__ float *>(xf32_ping + k);
    
    // need align with 256B
    auto *xf16_ping = reinterpret_cast<__ubuf__ half *>(xf32_pong + k);
    auto *xf16_pong = reinterpret_cast<__ubuf__ half *>(xf16_ping + k);

    auto *z_ping = reinterpret_cast<__ubuf__ int8_t *>(xf16_pong + k);
    auto *z_pong = reinterpret_cast<__ubuf__ int8_t *>(z_ping + k);

    auto *scale_buf = reinterpret_cast<__ubuf__ float *>(z_pong + k);
    auto *offset_buf = reinterpret_cast<__ubuf__ float *>(scale_buf + k);

    __ubuf__ bfloat16_t *x_bufs[2] = {x_ping, x_pong};
    __ubuf__ float32_t *xf32_bufs[2] = {xf32_ping, xf32_pong};
    __ubuf__ half *xf16_bufs[2] = {xf16_ping, xf16_pong};
    __ubuf__ int8_t *z_bufs[2] = {z_ping, z_pong};

    int eventId = 0;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    // scale_reciprocal GM -> UB
    copy_gm_to_ubuf(scale_buf, scale_reciprocal_gm, 0, 1, k / 8, 0, 0);
    // offset GM -> UB
    copy_gm_to_ubuf(offset_buf, offset_gm, 0, 1, k / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

    for (uint32_t row = block_idx; row < m; row += block_num) {
        uint32_t row_offset = row * k;

        // GM -> UB
        wait_flag(PIPE_V, PIPE_MTE2, eventId);
        // lenBurst = k / (32B / sizeof(bfloat16))
        copy_gm_to_ubuf(x_bufs[eventId], in_buf + row_offset, 0, 1, k / 16, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, eventId);

        // BF16 -> FP32
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        // vconv 接口的 repeat 以其中宽度较大的数据类型为准
        vconv_bf162f32(xf32_bufs[eventId], x_bufs[eventId], k / VECTOR_MAX_NUM_OF_FP32, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, eventId);

        // dst = src * scale_reciprocal + offset
        vmul(xf32_bufs[eventId], xf32_bufs[eventId], scale_buf, k / VECTOR_MAX_NUM_OF_FP32, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vadd(xf32_bufs[eventId], xf32_bufs[eventId], offset_buf, k / VECTOR_MAX_NUM_OF_FP32, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // FP32 -> FP16
        vconv_f322f16a(xf16_bufs[eventId], xf32_bufs[eventId], k / VECTOR_MAX_NUM_OF_FP32, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        // // FP16 -> Int8
        wait_flag(PIPE_MTE3, PIPE_V, eventId);
        vconv_f162s8a(z_bufs[eventId], xf16_bufs[eventId], k / VECTOR_MAX_NUM_OF_BF16, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, eventId);

        // UB -> GM
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        copy_ubuf_to_gm(out_buf + row_offset, z_bufs[eventId], 0, 1, k / 32, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, eventId);

        eventId = 1 - eventId;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    pipe_barrier(PIPE_ALL);
}

#define QUANT_FUNC_DEFINE(dtype)                                                            \
extern "C" __global__ __aicore__ void quant_bf16_to_i8_static(GM_ADDR in,                   \
    GM_ADDR scale_reciprocal, GM_ADDR offset, GM_ADDR out, uint32_t m, uint32_t k)          \
{                                                                                           \
    quant_bf16_to_i8(in, scale_reciprocal, offset, out, m, k);                              \
}
#else
#define QUANT_FUNC_DEFINE(dtype)                                                            \
extern "C" __global__ __aicore__ void quant_bf16_to_i8_static(GM_ADDR in,                   \
    GM_ADDR scale_reciprocal, GM_ADDR offset, GM_ADDR out, uint32_t m, uint32_t k)          \
{                                                                                           \
}

#endif