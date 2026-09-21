/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"

#ifdef __DAV_C220_VEC__

// Per-token dynamic quant: scale = absmax(row)/127, out = round(x*127/absmax).
// Whole-row absmax must precede scaling, so each row runs two phases:
//   k_loop == 1 (k <= K_TILE): x read once; phase1 (absmax via vabs+ReduceMax into xAbs_buf)
//     leaves xf32_buf free for phase2 (vmuls->f322f16->f162s8) with no second GM read.
//   k_loop >  1 (k >  K_TILE): phase1 reduces cross-tile absmax into rowAbsUb via vmax; phase2
//     re-reads each x tile (not kept resident).
#define QUANT_DYN_K_TILE 8192
static_assert(ROUND_UP(QUANT_DYN_K_TILE, 256 / sizeof(bfloat16_t)) *
                          (2 * sizeof(bfloat16_t) + 2 * sizeof(float) + sizeof(half) +
                           2 * sizeof(int8_t)) +
                      sizeof(float) <=
                  UB_SIZE,
              "quant_bf16_to_i8_dynamic UB layout overflows UB_SIZE");
// ReduceMax supports float dim up to VECTOR_MAX_REPEAT(255) * pad(64) = 16320; K_TILE must fit.
static_assert(QUANT_DYN_K_TILE <= VECTOR_MAX_REPEAT * (VECTOR_MAX_BYTESIZE / sizeof(float)),
              "QUANT_DYN_K_TILE exceeds ReduceMax float limit");

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

    constexpr uint32_t k_tile = QUANT_DYN_K_TILE;
    uint32_t k_loop = DIV_ROUND_UP(k, k_tile);

    uint32_t k_pad = ROUND_UP(k_tile, (256 / sizeof(bfloat16_t)));
    uint32_t k_pad_row = ROUND_UP(k, VECTOR_MAX_NUM_OF_BF16);
    uint32_t k_repeats_row = k_pad_row / VECTOR_MAX_NUM_OF_FP32;

    auto *x1 = reinterpret_cast<__ubuf__ bfloat16_t *>((uintptr_t)0);
    auto *x2 = reinterpret_cast<__ubuf__ bfloat16_t *>(x1 + k_pad);
    auto *xf32_buf = reinterpret_cast<__ubuf__ float *>(x2 + k_pad);
    auto *xAbs_buf = reinterpret_cast<__ubuf__ float *>(xf32_buf + k_pad);
    auto *zf16_buf = reinterpret_cast<__ubuf__ half *>(xAbs_buf + k_pad);
    auto *z1 = reinterpret_cast<__ubuf__ int8_t *>(zf16_buf + k_pad);
    auto *z2 = reinterpret_cast<__ubuf__ int8_t *>(z1 + k_pad);
    auto *rowAbsUb = reinterpret_cast<__ubuf__ float *>(z2 + k_pad);
    auto *sum_addr = reinterpret_cast<__ubuf__ float *>(rowAbsUb + 1);
    assert((uint64_t)sum_addr <= UB_SIZE);

    __ubuf__ bfloat16_t *xBufs[2] = {x1, x2};
    __ubuf__ int8_t *zBufs[2] = {z1, z2};

    __gm__ bfloat16_t *x_gm = reinterpret_cast<__gm__ bfloat16_t *>(x);
    __gm__ int8_t *z_gm = reinterpret_cast<__gm__ int8_t *>(z);
    __gm__ float *scales_gm = reinterpret_cast<__gm__ float *>(scales);

    // Pre-set per-row ping-pong flags (drained at function end). eId flips each row.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    int eId = 0;
    for (int row = block_idx; row < m; row += block_num) {
        int64_t rowOffset = (int64_t)row * k;

        // ============================================================
        // Fast path: k <= K_TILE (k_loop == 1). See header comment.
        // ============================================================
        if (k_loop == 1) {
            // x GM -> UB
            wait_flag(PIPE_V, PIPE_MTE2, eId);
            copy_gm_to_ubuf_align_b16(xBufs[eId], x_gm + rowOffset, 0, 1, k * sizeof(bfloat16_t), 0,
                                      0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, eId);

            // BF16 -> FP32 (shared by phase1 absmax and phase2 convert)
            wait_flag(PIPE_MTE2, PIPE_V, eId);
            vconv_bf162f32(xf32_buf, xBufs[eId], k_repeats_row, 1, 1, 8, 4);
            set_flag(PIPE_V, PIPE_MTE2, eId);  // xBufs[eId] free for next row
            pipe_barrier(PIPE_V);

            // phase1: |x| -> absmax (single tile: ReduceMax writes whole-row absmax to xAbs_buf[0])
            vabs(xAbs_buf, xf32_buf, k_repeats_row, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            ReduceMax(xAbs_buf, xAbs_buf, k);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);  // absmax visible to S
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            float absmax = float(*xAbs_buf);
            float scale = absmax / float(127);
            float scaleRec = float(127) / absmax;

            // scale -> GM (reuse xAbs_buf as 1-float scratch)
            __ubuf__ float *scaleUb = xAbs_buf;
            *scaleUb = scale;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm_align_b32(scales_gm + row, scaleUb, 0, 1, sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

            // phase2: reuse xf32_buf -> vmuls -> f16 -> s8 -> GM
            wait_flag(PIPE_MTE3, PIPE_V, eId);  // zBufs[eId] free
            vmuls(xf32_buf, xf32_buf, scaleRec, k_repeats_row, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_f322f16(zf16_buf, xf32_buf, k_repeats_row, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            vconv_f162s8(zBufs[eId], zf16_buf, k_pad_row / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
            set_flag(PIPE_V, PIPE_MTE3, eId);  // zBufs[eId] ready for MTE3
            pipe_barrier(PIPE_V);

            wait_flag(PIPE_V, PIPE_MTE3, eId);
            copy_ubuf_to_gm_align_b8(z_gm + rowOffset, zBufs[eId], 0, 1, k, 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, eId);  // zBufs[eId] free for next row

            eId = 1 - eId;
            continue;
        }

        // ============================================================
        // Tiled path: k > K_TILE (k_loop > 1). See header comment.
        // ============================================================
        // phase 1 : reduce - cross-tile absmax into rowAbsUb
        *rowAbsUb = float(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);  // rowAbsUb=0 visible to V
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

        for (uint32_t loop = 0; loop < k_loop; loop++) {
            uint32_t k_offset = loop * k_tile;
            bool last_loop = (loop == k_loop - 1);
            uint32_t k_size = last_loop ? (k - k_offset) : k_tile;
            uint32_t k_size_pad = ROUND_UP(k_size, VECTOR_MAX_NUM_OF_BF16);
            uint32_t k_repeats = k_size_pad / VECTOR_MAX_NUM_OF_FP32;

            // x tile GM -> UB
            wait_flag(PIPE_V, PIPE_MTE2, eId);
            copy_gm_to_ubuf_align_b16(xBufs[eId], x_gm + rowOffset + k_offset, 0, 1,
                                      k_size * sizeof(bfloat16_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, eId);

            // BF16 -> FP32
            wait_flag(PIPE_MTE2, PIPE_V, eId);
            vconv_bf162f32(xf32_buf, xBufs[eId], k_repeats, 1, 1, 8, 4);
            set_flag(PIPE_V, PIPE_MTE2, eId);  // xBufs[eId] free for next tile
            pipe_barrier(PIPE_V);

            // |x| -> per-tile absmax -> merge into rowAbsUb
            vabs(xAbs_buf, xf32_buf, k_repeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            ReduceMax(xAbs_buf, xAbs_buf, k_size);
            pipe_barrier(PIPE_V);
            SetMask(1);
            vmax(rowAbsUb, rowAbsUb, xAbs_buf, 1, 1, 1, 1, 8, 8, 8);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);  // rowAbsUb visible to S
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        // phase 2 : convert - re-read x tile, scale, f16->s8 -> GM
        float absmax = float(*rowAbsUb);
        float scale = absmax / float(127);
        float scaleRec = float(127) / absmax;

        // scale -> GM (reuse xAbs_buf as 1-float scratch)
        __ubuf__ float *scaleUb = xAbs_buf;
        *scaleUb = scale;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b32(scales_gm + row, scaleUb, 0, 1, sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

        for (uint32_t loop = 0; loop < k_loop; loop++) {
            uint32_t k_offset = loop * k_tile;
            bool last_loop = (loop == k_loop - 1);
            uint32_t k_size = last_loop ? (k - k_offset) : k_tile;
            uint32_t k_size_pad = ROUND_UP(k_size, VECTOR_MAX_NUM_OF_BF16);
            uint32_t k_repeats = k_size_pad / VECTOR_MAX_NUM_OF_FP32;

            // x tile GM -> UB
            wait_flag(PIPE_V, PIPE_MTE2, eId);
            copy_gm_to_ubuf_align_b16(xBufs[eId], x_gm + rowOffset + k_offset, 0, 1,
                                      k_size * sizeof(bfloat16_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, eId);

            wait_flag(PIPE_MTE2, PIPE_V, eId);
            vconv_bf162f32(xf32_buf, xBufs[eId], k_repeats, 1, 1, 8, 4);
            set_flag(PIPE_V, PIPE_MTE2, eId);  // xBufs[eId] free for next tile
            pipe_barrier(PIPE_V);

            vmuls(xf32_buf, xf32_buf, scaleRec, k_repeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_f322f16(zf16_buf, xf32_buf, k_repeats, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, eId);  // zBufs[eId] free from prev MTE3
            vconv_f162s8(zBufs[eId], zf16_buf, k_size_pad / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
            set_flag(PIPE_V, PIPE_MTE3, eId);  // zBufs[eId] ready for MTE3
            pipe_barrier(PIPE_V);

            // z tile UB -> GM
            wait_flag(PIPE_V, PIPE_MTE3, eId);
            copy_ubuf_to_gm_align_b8(z_gm + rowOffset + k_offset, zBufs[eId], 0, 1, k_size, 0, 0, 0,
                                     0);
            set_flag(PIPE_MTE3, PIPE_V, eId);  // zBufs[eId] free for next V
        }
        eId = 1 - eId;
    }
    // drain the pre-set ping-pong flags before the final barrier
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
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
