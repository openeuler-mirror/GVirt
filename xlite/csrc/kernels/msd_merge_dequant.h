/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

#define GMA(T) __gm__ T *
#define UBA(T) __ubuf__ T *
#define PINGPONG 2
#define MSD_MERGE_SCALE 16.0f

// MSD result merge, low-nibble bias compensation, and per-token dequantization.
//
// This AIV kernel performs Step 3 of the MSD W4A8 pipeline:
//   1. Merge:    Y = Y_high × 16 + Y_low
//   2. Bias:     Y = Y + scale_bias  (low nibble -8 offset compensation)
//   3. Dequant:  Y = Y × perTokenScale  (per-token activation dequant)
//
// Mathematical derivation of scale_bias:
//   MSD low nibble extraction: X_low = (X & 0x0F) - 8
//   Each activation row subtracts 8 from its low nibble, introducing a systematic bias:
//     Y_low_raw = X_low × W = ((X & 0x0F) - 8) × W = (X & 0x0F) × W - 8 × Σ_k W[k]
//   scale_bias = 8 × Σ_k W[k]  (per column, precomputed offline)
//   Adding it back cancels the -8 offset.
//
// Input:
//   y_merged  [2*m, n] half   — Mid-stage INT4×INT4 matmul results, row-merged:
//                              rows [0,   m) = Y_low, rows [m, 2*m) = Y_high
//                              (low-first/high-after layout, NOT interleaved).
//                              Each output row r pairs low row r with high row r+m.
//   scale_bias  [1, n]   float  — per-column MSD bias correction
//   perTokenScale [m]    float  — per-token activation dequant scale
//   pnum_tokens               — optional dynamic token count (clamps m)
//
// Output:
//   y           [m, n]   bfloat16 — final dequantized result (bf16 for downstream use)

__aicore__ inline void msd_merge_dequant(GM_ADDR y_merged, GM_ADDR scale_bias,
                                         GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR pnum_tokens,
                                         uint32_t m, uint32_t n)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (pnum_tokens) {
        uint32_t pnum_tokens_val = *((__gm__ uint32_t *)pnum_tokens);
        m = pnum_tokens_val < m ? pnum_tokens_val : m;
    }

    GMA(half) merged_gm = reinterpret_cast<GMA(half)>(y_merged);  // [2*m, n]
    GMA(float) bias_gm = reinterpret_cast<GMA(float)>(scale_bias);
    GMA(float) scale_gm = reinterpret_cast<GMA(float)>(perTokenScale);
    GMA(bfloat16_t) out_gm = reinterpret_cast<GMA(bfloat16_t)>(y);

    uint32_t n_pad = ROUND_UP(n, VECTOR_MAX_NUM_OF_FP16);
    uint32_t fp16_rep = DIV_ROUND_UP(n_pad, VECTOR_MAX_NUM_OF_FP16);
    uint32_t fp32_rep = DIV_ROUND_UP(n_pad, VECTOR_MAX_NUM_OF_FP32);

    auto *low0 = reinterpret_cast<UBA(half)>((uintptr_t)(0));
    auto *high0 = reinterpret_cast<UBA(half)>((uintptr_t)(low0 + n_pad));
    auto *low1 = reinterpret_cast<UBA(half)>((uintptr_t)(high0 + n_pad));
    auto *high1 = reinterpret_cast<UBA(half)>((uintptr_t)(low1 + n_pad));
    auto *low_fp32_buf = reinterpret_cast<UBA(float)>((uintptr_t)(high1 + n_pad));
    auto *high_fp32_buf = reinterpret_cast<UBA(float)>((uintptr_t)(low_fp32_buf + n_pad));
    auto *merged_fp32 = reinterpret_cast<UBA(float)>((uintptr_t)(high_fp32_buf + n_pad));
    auto *bias_buf = reinterpret_cast<UBA(float)>((uintptr_t)(merged_fp32 + n_pad));
    auto *out0 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(bias_buf + n_pad));
    auto *out1 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(out0 + n_pad));
    auto *scale_buf = reinterpret_cast<UBA(float)>((uintptr_t)(out1 + n_pad));
    auto *end_addr = reinterpret_cast<UBA(int8_t)>((uintptr_t)(scale_buf + 1));

    assert((uint64_t)end_addr <= UB_SIZE);

    UBA(half) high_bufs[PINGPONG] = {high0, high1};
    UBA(half) low_bufs[PINGPONG] = {low0, low1};
    UBA(bfloat16_t) out_bufs[PINGPONG] = {out0, out1};

    int event_id = 0;

    // Load scale_bias [n] ONCE (all rows share it). MTE2 writes bias_buf, then
    // signals V; V waits before using it. After this load completes, bias_buf is
    // V-only for the rest of the kernel — no per-row reload, no in-loop flag.
    copy_gm_to_ubuf_align_b32(bias_buf, bias_gm, 0, 1, n * sizeof(float), 0, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Prime the pingpong load/store chains for the first 2 rows.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    for (uint32_t row = block_idx; row < m; row += block_num) {
        // y_merged is [2*m, n]: rows [0, m) = Y_low, rows [m, 2*m) = Y_high.
        // Output row r pairs low row r with high row r + m; output is written to row r.
        uint32_t low_row_offset = row * n;
        uint32_t high_row_offset = (row + m) * n;

        // ===== Load Y_high and Y_low from GM into UB =====
        // Protect in_bufs (high/low): MTE2 writes, V reads. V waits until the
        // previous row released the same in_buf (primed / prev set_flag(V,MTE2)).
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        copy_gm_to_ubuf_align_b16(low_bufs[event_id], merged_gm + low_row_offset, 0, 1,
                                  n * sizeof(half), 0, 0, 0, 0);
        copy_gm_to_ubuf_align_b16(high_bufs[event_id], merged_gm + high_row_offset, 0, 1,
                                  n * sizeof(half), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);

        // ============================================================
        // Step 1: MSD Merge — Y_high × 16 + Y_low
        // ============================================================
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);
        // One vconv converts both rows: low_bufs[eid] and high_bufs[eid] are
        // adjacent in UB, so repeat = fp32_rep*2 with stride=1 walks low then
        // high, landing low_fp32 + high_fp32 contiguously into low_fp32_buf.
        vconv_f162f32(low_fp32_buf, low_bufs[event_id], fp32_rep * 2, 1, 1, 8, 4);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        pipe_barrier(PIPE_V);

        // Y_high × 16 — in-place on the high half of the vconv landing buffer.
        vmuls(high_fp32_buf, high_fp32_buf, float(MSD_MERGE_SCALE), fp32_rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // merged = Y_high × 16 + Y_low
        vadd(merged_fp32, high_fp32_buf, low_fp32_buf, fp32_rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // merged = merged + scale_bias
        vadd(merged_fp32, merged_fp32, bias_buf, fp32_rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // scale GM -> UB
        copy_gm_to_ubuf_align_b16(scale_buf, scale_gm + row, 0, 1, sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        // merged = merged * scale
        float tokenScale = float(*reinterpret_cast<__ubuf__ float *>(scale_buf));
        vmuls(merged_fp32, merged_fp32, tokenScale, fp32_rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // vconv(merged) -> out_bufs
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);
        vconv_f322bf16r(out_bufs[event_id], merged_fp32, fp32_rep, 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        pipe_barrier(PIPE_V);

        // out_buf UB -> GM
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        copy_ubuf_to_gm_align_b16(out_gm + low_row_offset, out_bufs[event_id], 0, 1,
                                  n * sizeof(bfloat16_t), 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);

        event_id = 1 - event_id;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define MSD_MERGE_DEQUANT_FUNC_DEFINE(dtype)                                          \
    extern "C" __global__ __aicore__ void msd_merge_dequant_##dtype(                  \
        GM_ADDR y_merged, GM_ADDR scale_bias, GM_ADDR perTokenScale, GM_ADDR y,       \
        GM_ADDR pnum_tokens, uint32_t m, uint32_t n)                                  \
    {                                                                                 \
        msd_merge_dequant(y_merged, scale_bias, perTokenScale, y, pnum_tokens, m, n); \
    }
#else
#define MSD_MERGE_DEQUANT_FUNC_DEFINE(dtype)                                    \
    extern "C" __global__ __aicore__ void msd_merge_dequant_##dtype(            \
        GM_ADDR y_merged, GM_ADDR scale_bias, GM_ADDR perTokenScale, GM_ADDR y, \
        GM_ADDR pnum_tokens, uint32_t m, uint32_t n)                            \
    {                                                                           \
    }
#endif
