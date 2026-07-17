/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

// Unpack INT4 weights stored as packed INT8 (2 INT4 per INT8) into separate
// INT4 weights for MSD (Mixed-precision Split-activation Decomposition) computation.
//
// Input:  act_packed [m, k] INT8, where each byte holds two INT4:
//         - Low  4 bits (bits 0-3): even-index INT4 (unsigned nibble 0-15)
//         - High 4 bits (bits 4-7): odd-index  INT4 (signed, via int8/16 floor)
//
// Output: weight_low  [m, k/2] packed-INT4 — low nibbles, 2 per byte, signed [-8,7]
//         weight_high [m, k/2] packed-INT4 — high nibbles, 2 per byte, signed [-8,7]
//
// MSD principle: INT8 = high_nibble × 16 + low_nibble
//   high_nibble = floor(x / 16) (already signed from int8 arithmetic)
//   low_nibble  = float(int8 & 0x0F) - 8 (sign-extend, 8 will be compensate in scale_bias)

#define GMA(T) __gm__ T *
#define UBA(T) __ubuf__ T *
#define PINGPONG 2

__aicore__ inline void unpack_activation(GM_ADDR act_packed, GM_ADDR act_out, uint32_t m,
                                         uint32_t k)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    // n = original column count (number of packed INT8 bytes per row).
    // Each packed INT8 byte contains 2 INT4 values for 2 adjacent columns.
    uint32_t k_half = k / 2;

    GMA(int8_t) packed_gm = reinterpret_cast<GMA(int8_t)>(act_packed);
    GMA(int8_t) low_gm = reinterpret_cast<GMA(int8_t)>(act_out);

    uint32_t k_pad = ROUND_UP(k, VECTOR_MAX_NUM_OF_INT8);
    uint32_t kh_pad = ROUND_UP(k_half, VECTOR_MAX_NUM_OF_INT8);
    uint32_t fp16_rep = DIV_ROUND_UP(k_pad, VECTOR_MAX_NUM_OF_FP16);
    uint32_t int16_rep = DIV_ROUND_UP(kh_pad, VECTOR_MAX_NUM_OF_INT16);

    uintptr_t off = 0;
    UBA(int8_t) in_buf_0 = reinterpret_cast<UBA(int8_t)>(off);
    off += k_pad;
    UBA(int8_t) in_buf_1 = reinterpret_cast<UBA(int8_t)>(off);
    off += k_pad;
    UBA(int8_t) low_and_buf = reinterpret_cast<UBA(int8_t)>(off);
    off += k_pad;
    UBA(half) work_fp16 = reinterpret_cast<UBA(half)>(off);
    off += k_pad * sizeof(half);
    UBA(half) vand_mask_buf = reinterpret_cast<UBA(half)>(off);
    off += k_pad * sizeof(half);
    UBA(half) fp16_buf = reinterpret_cast<UBA(half)>(off);
    off += k_pad * sizeof(half);
    UBA(int8_t) out_low_0 = reinterpret_cast<UBA(int8_t)>(off);
    off += kh_pad;
    UBA(int8_t) out_low_1 = reinterpret_cast<UBA(int8_t)>(off);
    off += kh_pad;
    UBA(int8_t) out_high_0 = reinterpret_cast<UBA(int8_t)>(off);
    off += kh_pad;
    UBA(int8_t) out_high_1 = reinterpret_cast<UBA(int8_t)>(off);
    off += kh_pad;
    assert(off <= UB_SIZE);

    UBA(int8_t) in_bufs[PINGPONG] = {in_buf_0, in_buf_1};
    UBA(int8_t) out_low_bufs[PINGPONG] = {out_low_0, out_low_1};
    UBA(int8_t) out_high_bufs[PINGPONG] = {out_high_0, out_high_1};

    // vand mask: 0x0F0F per int16 word, int8 data is viewed as int16
    UBA(int16_t) vand_mask = reinterpret_cast<UBA(int16_t)>(vand_mask_buf);
    vector_dup(vand_mask, (int16_t)0x0f0f, int16_rep, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);

    int event_id = 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    for (uint32_t row = block_idx; row < m; row += block_num) {
        uint32_t row_offset = row * k;

        // ============================================================
        // HIGH NIBBLE:  signed_high = floor(int8 / 16)
        //   Result is already in [-8,7] — no sign-extend needed.
        //   Output: INT8 → FP16 → vconv_f162s4f → packed INT4
        // ============================================================
        // LOW NIBBLE:  vand(int8, 0x0F) → unsigned nibble [0,15]
        //   Sign-extend: [0,15] → [-8,7]
        //   Output: signed INT8 → FP16 → vconv_f162s4f → packed INT4
        // ============================================================

        // x GM -> UB
        wait_flag(PIPE_V, PIPE_MTE2, event_id);
        copy_gm_to_ubuf_align_b8(in_bufs[event_id], packed_gm + row_offset, 0, 1,
                                 k * sizeof(int8_t), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, event_id);

        wait_flag(PIPE_MTE2, PIPE_V, event_id);
        // x_low_int8 = x & 0x0f
        vand(reinterpret_cast<UBA(int16_t)>(low_and_buf),
             reinterpret_cast<UBA(int16_t)>(in_bufs[event_id]), vand_mask, int16_rep, 1, 1, 1, 8, 8,
             8);
        // vconv(x) -> x_fp16
        vconv_s82f16(fp16_buf, in_bufs[event_id], fp16_rep, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, event_id);

        // vconv(x_low_int8) → x_low_fp16 [0.0, 15.0]
        vconv_s82f16(work_fp16, low_and_buf, fp16_rep, 1, 1, 8, 4);
        // vmuls(x_fp16, 0.0625) → x_high_fp16
        vmuls(fp16_buf, fp16_buf, (half)0.0625, fp16_rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // x_low_fp16 = x_low_fp16 - 8
        vadds(work_fp16, work_fp16, (half)-8.0, fp16_rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, event_id);
        // vconv(x_low_fp16) -> x_low_int4
        vconv_f162s4(reinterpret_cast<UBA(void)>(out_low_bufs[event_id]), work_fp16, fp16_rep, 1, 1,
                     2, 8);
        // vconv(x_high_fp16) -> x_high_int4
        vconv_f162s4z(reinterpret_cast<UBA(void)>(out_high_bufs[event_id]), fp16_buf, fp16_rep, 1,
                      1, 2, 8);
        set_flag(PIPE_V, PIPE_MTE3, event_id);

        wait_flag(PIPE_V, PIPE_MTE3, event_id);
        // x_low_int4 UB -> GM
        copy_ubuf_to_gm_align_b8(low_gm + row * k_half, out_low_bufs[event_id], 0, 1,
                                 k_half * sizeof(int8_t), 0, 0, 0, 0);
        // x_high_int4 UB -> GM
        copy_ubuf_to_gm_align_b8(low_gm + (row + m) * k_half, out_high_bufs[event_id], 0, 1,
                                 k_half * sizeof(int8_t), 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, event_id);

        event_id = 1 - event_id;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define UNPACK_ACTIVATION_FUNC_DEFINE(dtype)                         \
    extern "C" __global__ __aicore__ void unpack_activation_##dtype( \
        GM_ADDR act_packed, GM_ADDR act_out, uint32_t m, uint32_t k) \
    {                                                                \
        unpack_activation(act_packed, act_out, m, k);                \
    }
#else
#define UNPACK_ACTIVATION_FUNC_DEFINE(dtype)                         \
    extern "C" __global__ __aicore__ void unpack_activation_##dtype( \
        GM_ADDR act_packed, GM_ADDR act_out, uint32_t m, uint32_t k) \
    {                                                                \
    }

#endif
