/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

// Unpack INT8 activations stored as packed INT8 into interleaved low/high INT4
// rows for MSD (Mixed-precision Split-activation Decomposition) computation.
//
// Input:  act_packed [m, k] INT8, where each byte holds two INT4:
//         - Low  4 bits (bits 0-3): even-index INT4 (unsigned nibble 0-15)
//         - High 4 bits (bits 4-7): odd-index  INT4 (signed, via int8/16 floor)
//
// Output: act_out [2m, k/2] packed-INT4 — interleaved low/high rows, 2 per byte, signed [-8,7]
//         For input token at row r, the two output rows are:
//           row 2r     = low  nibbles of token r
//           row 2r + 1 = high nibbles of token r
//
// MSD principle: INT8 = high_nibble × 16 + low_nibble
//   high_nibble = floor(x / 16) (already signed from int8 arithmetic)
//   low_nibble  = float(int8 & 0x0F) - 8 (sign-extend, 8 will be compensate in scale_bias)

#define GMA(T) __gm__ T *
#define UBA(T) __ubuf__ T *

// ============================================================
// HIGH NIBBLE:  signed_high = floor(int8 / 16)
//   Result is already in [-8,7] — no sign-extend needed.
//   Output: INT8 → FP16 → vconv_f162s4f → packed INT4
// ============================================================
// LOW NIBBLE:  vand(int8, 0x0F) → unsigned nibble [0,15]
//   Sign-extend: [0,15] → [-8,7]
//   Output: signed INT8 → FP16 → vconv_f162s4f → packed INT4
// ============================================================

// Load one row of packed INT8 from GM into a UB input buffer (MTE2 pipe).
// wait_flag/set_flag around the DMA provide ping-pong synchronization:
//   wait_flag(V→MTE2, eid)  — V must have finished reading this in_buf slot
//   set_flag(MTE2→V, eid)   — data ready in UB for V compute
__aicore__ inline void unpack_load(UBA(int8_t) * in_bufs, GMA(int8_t) packed_gm, uint32_t gm_row,
                                   uint32_t k, uint32_t eid)
{
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + eid);
    CopyGmToUbufAligned(in_bufs[eid], packed_gm + gm_row * k, k * sizeof(int8_t));
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eid);
}

// Compute one row: split INT8 into low/high INT4 nibbles and pack into output
// UB buffers (V pipe).
//   wait_flag(MTE2→V, eid)  — input data must be ready
//   set_flag(V→MTE2, eid)   — input buffer freed for future load
//   wait_flag(MTE3→V, eid)  — output buffer must be free from previous store
//   set_flag(V→MTE3, eid)   — output ready for MTE3 store
__aicore__ inline void unpack_compute(UBA(int8_t) * in_bufs, UBA(int8_t) * out_low_bufs,
                                      UBA(int8_t) * out_high_bufs, UBA(int8_t) low_and_buf,
                                      UBA(half) work_fp16, UBA(half) fp16_buf,
                                      UBA(int16_t) vand_mask, uint32_t fp16_rep, uint32_t int16_rep,
                                      uint32_t eid)
{
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eid);
    vand(reinterpret_cast<UBA(int16_t)>(low_and_buf), reinterpret_cast<UBA(int16_t)>(in_bufs[eid]),
         vand_mask, int16_rep, 1, 1, 1, 8, 8, 8);
    vconv_s82f16(fp16_buf, in_bufs[eid], fp16_rep, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + eid);

    vconv_s82f16(work_fp16, low_and_buf, fp16_rep, 1, 1, 8, 4);
    vmuls(fp16_buf, fp16_buf, (half)0.0625, fp16_rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    vadds(work_fp16, work_fp16, (half)-8.0, fp16_rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eid);
    vconv_f162s4(reinterpret_cast<UBA(void)>(out_low_bufs[eid]), work_fp16, fp16_rep, 1, 1, 2, 8);
    vconv_f162s4f(reinterpret_cast<UBA(void)>(out_high_bufs[eid]), fp16_buf, fp16_rep, 1, 1, 2, 8);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eid);
}

// Store one row's low/high INT4 output from UB to GM (MTE3 pipe).
//   wait_flag(V→MTE3, eid)   — V must have finished writing output
//   set_flag(MTE3→V, eid)    — output buffer freed for future compute
__aicore__ inline void unpack_store(GMA(int8_t) low_gm, UBA(int8_t) * out_low_bufs,
                                    UBA(int8_t) * out_high_bufs, uint32_t gm_row, uint32_t k_half,
                                    uint32_t eid)
{
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eid);
    CopyUbufToGmAligned(low_gm + (2 * gm_row) * k_half, out_low_bufs[eid], k_half * sizeof(int8_t));
    CopyUbufToGmAligned(low_gm + (2 * gm_row + 1) * k_half, out_high_bufs[eid],
                        k_half * sizeof(int8_t));
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eid);
}

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

    UBA(int8_t) in_bufs[PINGPONG_BUF_NUM] = {in_buf_0, in_buf_1};
    UBA(int8_t) out_low_bufs[PINGPONG_BUF_NUM] = {out_low_0, out_low_1};
    UBA(int8_t) out_high_bufs[PINGPONG_BUF_NUM] = {out_high_0, out_high_1};

    // vand mask: 0x0F0F per int16 word, int8 data is viewed as int16
    UBA(int16_t) vand_mask = reinterpret_cast<UBA(int16_t)>(vand_mask_buf);
    vector_dup(vand_mask, (int16_t)0x0f0f, int16_rep, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);

    // Number of rows this block processes.
    if (block_idx >= m) {
        return;
    }
    uint32_t n = (m - block_idx + block_num - 1) / block_num;

    // Three-stage decoupled pipeline: iteration i does
    //   store(i-1) ∥ load(i+1) ∥ compute(i)
    // MTE3/MTE2 DMA instructions are dispatched before V's wait_flag,
    // so they proceed even while V is blocked.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    // row tracks the GM row being computed in the current iteration.
    uint32_t row = block_idx;
    uint32_t eid = 0;
    uint32_t other = 1 - eid;

    // ---- Prologue: load row 0 (and row 1 if n >= 2), compute row 0 ----
    unpack_load(in_bufs, packed_gm, row, k, eid);
    if (n >= 2) {
        unpack_load(in_bufs, packed_gm, row + block_num, k, other);
    }
    unpack_compute(in_bufs, out_low_bufs, out_high_bufs, low_and_buf, work_fp16, fp16_buf,
                   vand_mask, fp16_rep, int16_rep, eid);

    // ---- Steady state: i = 1 .. n-1 ----
    for (uint32_t i = 1; i < n; ++i) {
        other = eid;
        eid = 1 - eid;
        row += block_num;

        // 1. Store row (i-1) — dispatch MTE3 before V's wait, so MTE3 works
        //    even while V is blocked.
        unpack_store(low_gm, out_low_bufs, out_high_bufs, row - block_num, k_half, other);

        // 2. Load row (i+1) — dispatch MTE2 before V's wait, so MTE2 prefetches
        //    the next input while V computes the current row.
        if (i + 1 < n) {
            unpack_load(in_bufs, packed_gm, row + block_num, k, other);
        }

        // 3. Compute row (i) — V may block on wait_flag, but MTE2/MTE3 already
        //    have their DMA instructions dispatched.
        unpack_compute(in_bufs, out_low_bufs, out_high_bufs, low_and_buf, work_fp16, fp16_buf,
                       vand_mask, fp16_rep, int16_rep, eid);
    }

    // ---- Epilogue: store row (n-1) ----
    unpack_store(low_gm, out_low_bufs, out_high_bufs, row, k_half, eid);

    // ---- Drain: consume residual flags ----
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
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
