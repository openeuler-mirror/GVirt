/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

// out = int8(in * scale_reciprocal + offset)
// per-token quantion
// [m, k], activation; dtype = float16_t / bfloat16_t
template <typename dtype>
__aicore__ inline void quant_to_i8(GM_ADDR in, GM_ADDR scale_reciprocal, GM_ADDR offset,
                                   GM_ADDR out, uint32_t m, uint32_t k)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ dtype *in_buf = reinterpret_cast<__gm__ dtype *>(in);
    __gm__ dtype *scale_reciprocal_gm = reinterpret_cast<__gm__ dtype *>(scale_reciprocal);
    __gm__ dtype *offset_gm = reinterpret_cast<__gm__ dtype *>(offset);
    __gm__ int8_t *out_buf = reinterpret_cast<__gm__ int8_t *>(out);

    // calculate k_tile to avoid UB overflow
    uint32_t k_tile = 4096;
    uint32_t k_loop = DIV_ROUND_UP(k, k_tile);

    auto *x_ping = reinterpret_cast<__ubuf__ dtype *>((uintptr_t)0);
    auto *x_pong = reinterpret_cast<__ubuf__ dtype *>(x_ping + k_tile);

    auto *xf32_ping = reinterpret_cast<__ubuf__ float *>(x_pong + k_tile);
    auto *xf32_pong = reinterpret_cast<__ubuf__ float *>(xf32_ping + k_tile);

    auto *xf16_ping = reinterpret_cast<__ubuf__ half *>(xf32_pong + k_tile);
    auto *xf16_pong = reinterpret_cast<__ubuf__ half *>(xf16_ping + k_tile);

    auto *z_ping = reinterpret_cast<__ubuf__ int8_t *>(xf16_pong + k_tile);
    auto *z_pong = reinterpret_cast<__ubuf__ int8_t *>(z_ping + k_tile);

    auto *scale_ping = reinterpret_cast<__ubuf__ dtype *>(z_pong + k_tile);
    auto *scale_pong = reinterpret_cast<__ubuf__ dtype *>(scale_ping + k_tile);
    auto *offset_ping = reinterpret_cast<__ubuf__ dtype *>(scale_pong + k_tile);
    auto *offset_pong = reinterpret_cast<__ubuf__ dtype *>(offset_ping + k_tile);
    auto *scale_fp32_ping = reinterpret_cast<__ubuf__ float *>(offset_pong + k_tile);
    auto *scale_fp32_pong = reinterpret_cast<__ubuf__ float *>(scale_fp32_ping + k_tile);
    auto *offset_fp32_ping = reinterpret_cast<__ubuf__ float *>(scale_fp32_pong + k_tile);
    auto *offset_fp32_pong = reinterpret_cast<__ubuf__ float *>(offset_fp32_ping + k_tile);

    auto *end_addr = reinterpret_cast<__ubuf__ float *>(offset_fp32_pong + k_tile);
    assert((uint64_t)end_addr <= UB_SIZE);

    __ubuf__ dtype *x_bufs[2] = {x_ping, x_pong};
    __ubuf__ float32_t *xf32_bufs[2] = {xf32_ping, xf32_pong};
    __ubuf__ half *xf16_bufs[2] = {xf16_ping, xf16_pong};
    __ubuf__ int8_t *z_bufs[2] = {z_ping, z_pong};
    __ubuf__ dtype *scale_bufs[2] = {scale_ping, scale_pong};
    __ubuf__ dtype *offset_bufs[2] = {offset_ping, offset_pong};
    __ubuf__ float *scale_fp32_bufs[2] = {scale_fp32_ping, scale_fp32_pong};
    __ubuf__ float *offset_fp32_bufs[2] = {offset_fp32_ping, offset_fp32_pong};

    int event_id = 0;
    int scale_offset_event_id = 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);

    for (uint32_t loop = 0; loop < k_loop; loop++) {
        uint32_t k_offset = loop * k_tile;
        bool last_loop = (loop == k_loop - 1);
        uint32_t k_size = last_loop ? (k - k_offset) : k_tile;
        uint32_t k_size_pad = ROUND_UP(k_size, 256 / sizeof(dtype));
        uint32_t k_repeats = k_size_pad / VECTOR_MAX_NUM_OF_FP32;

        wait_flag(PIPE_V, PIPE_MTE2, scale_offset_event_id + EVENT_ID2);
        // scale_reciprocal GM -> UB
        copy_gm_to_ubuf_align_b16(scale_bufs[scale_offset_event_id], scale_reciprocal_gm + k_offset,
                                  0, 1, k_size * sizeof(dtype), 0, 0, 0, 0);
        // offset GM -> UB
        copy_gm_to_ubuf_align_b16(offset_bufs[scale_offset_event_id], offset_gm + k_offset, 0, 1,
                                  k_size * sizeof(dtype), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, scale_offset_event_id + EVENT_ID2);

        wait_flag(PIPE_MTE2, PIPE_V, scale_offset_event_id + EVENT_ID2);
        convert_input<dtype>(scale_fp32_bufs[scale_offset_event_id],
                             scale_bufs[scale_offset_event_id], k_repeats);
        convert_input<dtype>(offset_fp32_bufs[scale_offset_event_id],
                             offset_bufs[scale_offset_event_id], k_repeats);
        pipe_barrier(PIPE_V);

        for (uint32_t row = block_idx; row < m; row += block_num) {
            uint32_t row_offset = row * k + k_offset;
            // GM -> UB
            wait_flag(PIPE_V, PIPE_MTE2, event_id);
            copy_gm_to_ubuf_align_b16(x_bufs[event_id], in_buf + row_offset, 0, 1,
                                      k_size * sizeof(dtype), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, event_id);

            // FP16/BF16 -> FP32
            wait_flag(PIPE_MTE2, PIPE_V, event_id);
            // vconv 接口的 repeat 以其中宽度较大的数据类型为准
            convert_input<dtype>(xf32_bufs[event_id], x_bufs[event_id], k_repeats);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, event_id);

            // dst = src * scale_reciprocal + offset
            vmul(xf32_bufs[event_id], xf32_bufs[event_id], scale_fp32_bufs[scale_offset_event_id],
                 k_repeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            vadd(xf32_bufs[event_id], xf32_bufs[event_id], offset_fp32_bufs[scale_offset_event_id],
                 k_repeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            // FP32 -> FP16
            vconv_f322f16a(xf16_bufs[event_id], xf32_bufs[event_id], k_repeats, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);

            // FP16 -> Int8
            wait_flag(PIPE_MTE3, PIPE_V, event_id);
            vconv_f162s8a(z_bufs[event_id], xf16_bufs[event_id],
                          k_size_pad / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, event_id);

            // UB -> GM
            wait_flag(PIPE_V, PIPE_MTE3, event_id);
            copy_ubuf_to_gm_align_b8(out_buf + row_offset, z_bufs[event_id], 0, 1, k_size, 0, 0, 0,
                                     0);
            set_flag(PIPE_MTE3, PIPE_V, event_id);

            event_id = 1 - event_id;
        }

        set_flag(PIPE_V, PIPE_MTE2, scale_offset_event_id + EVENT_ID2);
        scale_offset_event_id = 1 - scale_offset_event_id;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    pipe_barrier(PIPE_ALL);
}

#define QUANT_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void quant_static_##dtype(                                    \
        GM_ADDR in, GM_ADDR scale_reciprocal, GM_ADDR offset, GM_ADDR out, uint32_t m, uint32_t k) \
    {                                                                                              \
        quant_to_i8<dtype>(in, scale_reciprocal, offset, out, m, k);                               \
    }
#else
#define QUANT_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void quant_static_##dtype(                                    \
        GM_ADDR in, GM_ADDR scale_reciprocal, GM_ADDR offset, GM_ADDR out, uint32_t m, uint32_t k) \
    {                                                                                              \
    }

#endif
