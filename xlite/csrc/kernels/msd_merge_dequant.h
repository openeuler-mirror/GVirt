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
//   2. Bias:     Y = Y + scale_bias  (low nibble -8 offset compensation, 8 × Σ_k (W_int4[k] ×
//   w_scale[k]))
//   3. Dequant:  Y = Y × perTokenScale  (per-token activation dequant)
//
// Input:
//   y_merged  [2*m, n] half   — Mid-stage INT4×INT4 matmul results, interleaved by token:
//                              token r occupies rows (2r = Y_low, 2r+1 = Y_high).
//                              Each output row r pairs low row 2r with high row 2r+1.
//   scaleBiasPtrs  INT64[nRoutedExperts] — full GM pointer array (same storage as group_matmul's
//                              weight/deqScale arrays). Entries outside [startIdx,endIdx) may be
//                              null; the kernel indexes the owning slot by GLOBAL expert id.
//   counts    uint32[nRoutedExperts] — per-expert token counts (un-doubled, same row axis as out).
//                              Indexed by global expert id; only [startIdx,endIdx) carry tokens.
//   startIdx, endIdx      — [startIdx, endIdx) is the range of experts this pass processes
//                              (mirrors group_matmul). The kernel prefetches this local slice of
//                              counts to UB (length endIdx-startIdx, ≤8), scans its prefix sums
//                              to find each row's LOCAL expert id, then re-adds startIdx to index
//                              the global pointer/counts arrays. numExperts=endIdx-startIdx.
//   perTokenScale [m]    float  — per-token activation dequant scale
//   pnum_tokens               — optional dynamic token count (clamps m)
//
// Output:
//   y           [m, n]   bfloat16 — final dequantized result (bf16 for downstream use)

__aicore__ inline void msd_merge_dequant(GM_ADDR y_merged, GM_ADDR scaleBiasPtrs,
                                         GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR pnum_tokens,
                                         uint32_t m, uint32_t n, GM_ADDR counts, uint32_t startIdx,
                                         uint32_t endIdx)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (pnum_tokens) {
        uint32_t pnum_tokens_val = *((__gm__ uint32_t *)pnum_tokens);
        m = pnum_tokens_val < m ? pnum_tokens_val : m;
    }

    GMA(half) merged_gm = reinterpret_cast<GMA(half)>(y_merged);  // [2*m, n]
    GMA(float) scale_gm = reinterpret_cast<GMA(float)>(perTokenScale);
    GMA(bfloat16_t) out_gm = reinterpret_cast<GMA(bfloat16_t)>(y);
    __gm__ uint64_t *bias_ptrs_gm = reinterpret_cast<__gm__ uint64_t *>(scaleBiasPtrs);
    __gm__ uint32_t *counts_gm = reinterpret_cast<__gm__ uint32_t *>(counts);

    uint32_t numExperts = endIdx - startIdx;

    uint32_t n_pad = ROUND_UP(n, VECTOR_MAX_NUM_OF_FP16);
    uint32_t fp32_rep = DIV_ROUND_UP(n_pad, VECTOR_MAX_NUM_OF_FP32);

    auto *low0 = reinterpret_cast<UBA(half)>((uintptr_t)(0));
    auto *high0 = reinterpret_cast<UBA(half)>((uintptr_t)(low0 + n_pad));
    auto *low1 = reinterpret_cast<UBA(half)>((uintptr_t)(high0 + n_pad));
    auto *high1 = reinterpret_cast<UBA(half)>((uintptr_t)(low1 + n_pad));
    auto *low_fp32_buf = reinterpret_cast<UBA(float)>((uintptr_t)(high1 + n_pad));
    auto *merged_fp32 = reinterpret_cast<UBA(float)>((uintptr_t)(low_fp32_buf + n_pad));
    auto *bias_buf = reinterpret_cast<UBA(float)>((uintptr_t)(merged_fp32 + n_pad));
    auto *out0 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(bias_buf + n_pad));
    auto *out1 = reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(out0 + n_pad));
    auto *counts_buf = reinterpret_cast<UBA(uint32_t)>(ROUND_UP((uintptr_t)(out1 + n_pad), 32));
    uint32_t counts_pad = ROUND_UP(numExperts * sizeof(uint32_t), 32);
    auto *end_addr = reinterpret_cast<UBA(int8_t)>(counts_buf + counts_pad / sizeof(uint32_t));

    UBA(half) high_bufs[PINGPONG] = {high0, high1};
    UBA(half) low_bufs[PINGPONG] = {low0, low1};
    UBA(bfloat16_t) out_bufs[PINGPONG] = {out0, out1};

    int event_id = 0;

    // Rolling expert locator: rows advance monotonically and experts partition
    // the row axis by counts prefix sums, so a block's expert id only ever
    // moves forward.
    uint32_t expert_id = 0;
    uint32_t curr_expert_row = 0;

    // counts, GM -> UB; fence to the S pipe, whose scalar expert scan reads
    // counts_buf every row (cross-pipe UB reads need an explicit event flag).
    CopyGmToUbufAligned(counts_buf, counts_gm + startIdx, numExperts * sizeof(uint32_t));
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    for (uint32_t row = block_idx; row < m; row += block_num) {
        uint32_t low_row_offset = (2 * row) * n;
        uint32_t high_row_offset = (2 * row + 1) * n;

        // per-token scale of this row
        float tokenScale = *(scale_gm + row);

        // Rolling expert locator: resume from where the previous row's scan
        // stopped and advance to this row's expert. Rows advance monotonically
        // and experts partition the row axis by counts prefix sums, so the
        // scan only ever moves forward — amortized O(1) per row.
        while (expert_id + 1 < numExperts && row >= curr_expert_row + counts_buf[expert_id]) {
            curr_expert_row += counts_buf[expert_id];
            expert_id++;
        }

        // low, high, from GM -> UB. Interleaved layout: token r's low is row
        // 2r, high is row 2r+1. GM rows 2r and 2r+1 are contiguous, and the UB
        // low/high buffers are adjacent (high = low + n_pad), so when n is a
        // multiple of the 128-element vector granularity (no pad between the
        // halves) one 2-row block copy moves both; otherwise the pad forces
        // two copies.
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        if (n % VECTOR_MAX_NUM_OF_FP16 == 0) {  // n_pad(for vconv) == n, only need one copy
            CopyGmToUbufAligned(low_bufs[event_id], merged_gm + low_row_offset,
                                2 * n * sizeof(half));
        } else {
            CopyGmToUbufAligned(low_bufs[event_id], merged_gm + low_row_offset, n * sizeof(half));
            CopyGmToUbufAligned(high_bufs[event_id], merged_gm + high_row_offset, n * sizeof(half));
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);

        // vconv(Y_low, Y_high), fp16 -> f32: one shot fills [low_fp32 | merged]
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + event_id);
        vconv_f162f32(low_fp32_buf, low_bufs[event_id], fp32_rep * 2, 1, 1, 8, 4);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + event_id);
        pipe_barrier(PIPE_V);

        // scale_bias, GM -> UB. Single buffer: the reload waits until the
        // previous expert cycle's last reader releases the buffer, then the
        // copy-done flag hands it to this row's vadd.
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        GMA(float) bias_gm_row = reinterpret_cast<GMA(float)>(bias_ptrs_gm[startIdx + expert_id]);
        CopyGmToUbufAligned(bias_buf, bias_gm_row, n * sizeof(float));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

        // merged = Y_high × 16 + Y_low
        vmuls(merged_fp32, merged_fp32, float(MSD_MERGE_SCALE), fp32_rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(merged_fp32, merged_fp32, low_fp32_buf, fp32_rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // merged = merged + scale_bias
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        vadd(merged_fp32, merged_fp32, bias_buf, fp32_rep, 1, 1, 1, 8, 8, 8);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        pipe_barrier(PIPE_V);

        // merged = merged * scale
        vmuls(merged_fp32, merged_fp32, tokenScale, fp32_rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // vconv(merged) -> out_bufs
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);
        vconv_f322bf16r(out_bufs[event_id], merged_fp32, fp32_rep, 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        pipe_barrier(PIPE_V);

        // out_buf UB -> GM (output is [m, n]: write the merged result to row r)
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + event_id);
        CopyUbufToGmAligned(out_gm + row * n, out_bufs[event_id], n * sizeof(bfloat16_t));
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + event_id);

        event_id = 1 - event_id;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

#define MSD_MERGE_DEQUANT_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void msd_merge_dequant_##dtype(                            \
        GM_ADDR y_merged, GM_ADDR scaleBiasPtrs, GM_ADDR perTokenScale, GM_ADDR y,              \
        GM_ADDR pnum_tokens, uint32_t m, uint32_t n, GM_ADDR counts, uint32_t startIdx,         \
        uint32_t endIdx)                                                                        \
    {                                                                                           \
        msd_merge_dequant(y_merged, scaleBiasPtrs, perTokenScale, y, pnum_tokens, m, n, counts, \
                          startIdx, endIdx);                                                    \
    }
#else
#define MSD_MERGE_DEQUANT_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void msd_merge_dequant_##dtype(                    \
        GM_ADDR y_merged, GM_ADDR scaleBiasPtrs, GM_ADDR perTokenScale, GM_ADDR y,      \
        GM_ADDR pnum_tokens, uint32_t m, uint32_t n, GM_ADDR counts, uint32_t startIdx, \
        uint32_t endIdx)                                                                \
    {                                                                                   \
    }
#endif
