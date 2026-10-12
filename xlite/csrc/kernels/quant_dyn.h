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

// out = int8(x * 127 / absmax(x_row)), scale = absmax(x_row) / 127
// dynamic per-token quantization, [m, k] activation; dtype = float16_t / bfloat16_t

// Batched multi-row fast path (k % 32 == 0, k <= QUANT_DYN_K_TILE, rows >= 2):
// groups of `rows` consecutive tokens share one 2D MTE2 burst in, one 2D MTE3
// burst out, and — the dominant saving — one V->S / S->MTE3 sync round per
// group instead of per row: phase 1 reduces every row's absmax into a 32B-slot
// table (ReduceMax needs a k-sized dst scratch, so its scalar result is moved
// with a masked vmax), S converts all scales at once (staged for one small
// MTE3), then phase 2 re-scales each row from the still-resident xf32 block.
// The row-independent elementwise stages (input convert, FP32->FP16->int8) run
// as one whole-group vector op each; only the per-row absmax reduction and the
// per-row vmuls stay row-serial. x uses a single buffer (dead right after the
// group convert); z keeps a pingpong. rows is capped so r * kPad fits in one
// vector instruction (VECTOR_MAX_REPEAT repeats of 64 fp32).
template <typename dtype>
__aicore__ inline void quant_dyn_to_i8_rows(__gm__ dtype *x_gm, __gm__ float *scales_gm,
                                            __gm__ int8_t *z_gm, uint32_t m, uint32_t k,
                                            uint32_t rows)
{
    uint32_t kPad = ROUND_UP(k, 256 / sizeof(dtype));
    uint32_t kRepeats = kPad / VECTOR_MAX_NUM_OF_FP32;
    uint32_t numGroups = DIV_ROUND_UP(m, rows);

    // UB layout: x: [rows,kPad] dtype | xf32: [rows,kPad] float | z1,z2: [rows,kPad] int8
    //            | xAbs: [kPad] float | xf16: [rows,kPad] half
    //            | rowAbs: [rows][8] float (32B slots) | scaleStage: [2*rows] float
    //            bytes = 10*R*kPad + 4*kPad + 40*R.
    auto *x = reinterpret_cast<__ubuf__ dtype *>((uintptr_t)0);
    auto *xf32 = reinterpret_cast<__ubuf__ float *>(x + rows * kPad);
    auto *z1 = reinterpret_cast<__ubuf__ int8_t *>(xf32 + rows * kPad);
    auto *z2 = reinterpret_cast<__ubuf__ int8_t *>(z1 + rows * kPad);
    auto *xAbs = reinterpret_cast<__ubuf__ float *>(z2 + rows * kPad);
    auto *xf16 = reinterpret_cast<__ubuf__ half *>(xAbs + kPad);
    auto *rowAbs = reinterpret_cast<__ubuf__ float *>(xf16 + rows * kPad);
    auto *scaleStage = reinterpret_cast<__ubuf__ float *>(rowAbs + rows * 8);
    assert((uint64_t)(scaleStage + 2 * rows) <= UB_SIZE);

    __ubuf__ int8_t *zBufs[2] = {z1, z2};

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    // 2D DMA configs: one burst per row (see quant_to_i8_rows).
    uint64_t inCfg =
        __set_dmi_config(0, rows, k * sizeof(dtype) / 32, 0, (kPad - k) * sizeof(dtype) / 32);
    uint64_t outCfg = __set_dmi_config(0, rows, k / 32, (kPad - k) / 32, 0);
    int eId = 0;
    for (uint32_t g = block_idx; g < numGroups; g += block_num) {
        uint32_t base = g * rows;
        uint32_t r = MIN(rows, m - base);
        uint32_t rRepeats = r * kRepeats;
        if (r != rows) {
            inCfg =
                __set_dmi_config(0, r, k * sizeof(dtype) / 32, 0, (kPad - k) * sizeof(dtype) / 32);
            outCfg = __set_dmi_config(0, r, k / 32, (kPad - k) / 32, 0);
        }

        // x [r, k] GM -> UB
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        copy_gm_to_ubuf(x, x_gm + (uint64_t)base * k, inCfg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // whole-group FP16/BF16 -> FP32; x is dead right after, so the next
        // group's MTE2 can issue while this group's vector work is still running
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        convert_input<dtype>(xf32, x, rRepeats);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // phase 1: per-row |x| -> absmax, extracted to a 32B slot table
        for (uint32_t i = 0; i < r; i++) {
            vabs(xAbs, xf32 + i * kPad, kRepeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            ReduceMax(xAbs, xAbs, k);
            // move the scalar result to rowAbs[i]: ReduceMax's dst needs a k-sized
            // scratch, so reduce in place then copy lane 0 with a masked vmax
            SetMask(1);
            vmax(rowAbs + i * 8, xAbs, xAbs, 1, 1, 1, 1, 8, 8, 8);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);  // absmax table visible to S
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        // S: all scales at once; stage [0,r) = scale (absmax/127), [rows, rows+r) = 127/absmax
        for (uint32_t i = 0; i < r; i++) {
            float absmax = float(*(rowAbs + i * 8));
            *(scaleStage + i) = absmax / float(127);
            *(scaleStage + rows + i) = float(127) / absmax;
        }
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b32(scales_gm + base, scaleStage, 0, 1, r * sizeof(float), 0, 0, 0,
                                  0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);  // also gates V issue: scaleStage consumed

        // phase 2: per-row re-scale from the resident xf32 block, then
        // whole-group FP32 -> FP16 -> int8; the z-buffer credit is only needed
        // before the int8 write, so the f32->f16 stage runs during the wait
        for (uint32_t i = 0; i < r; i++) {
            float scaleRec = float(*(scaleStage + rows + i));
            vmuls(xf32 + i * kPad, xf32 + i * kPad, scaleRec, kRepeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        vconv_f322f16(xf16, xf32, rRepeats, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eId);  // zBufs[eId] free from prev MTE3
        vconv_f162s8(zBufs[eId], xf16, r * kPad / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eId);  // zBufs[eId] ready for MTE3

        // z [r, k] UB -> GM
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eId);
        copy_ubuf_to_gm(z_gm + (uint64_t)base * k, zBufs[eId], outCfg);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eId);
        eId = 1 - eId;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

template <typename dtype>
__aicore__ inline void quant_dyn_to_i8(GM_ADDR x, GM_ADDR scales, GM_ADDR z, GM_ADDR pnum_tokens,
                                       uint32_t m, uint32_t k)
{
    // UB layout below: x1,x2: dtype | xf32_buf,xAbs_buf: float | zf16_buf: half | z1,z2: int8_t
    // | rowAbsUb: float(1). bytes = k_pad * (2*sizeof(dtype) + 2*sizeof(float) + sizeof(half)
    //            + 2*sizeof(int8_t)) + sizeof(float); must fit in UB_SIZE.
    static_assert(
        ROUND_UP(QUANT_DYN_K_TILE, 256 / sizeof(dtype)) *
                    (2 * sizeof(dtype) + 2 * sizeof(float) + sizeof(half) + 2 * sizeof(int8_t)) +
                sizeof(float) <=
            UB_SIZE,
        "quant_dynamic UB layout overflows UB_SIZE");
    // ReduceMax supports float dim up to VECTOR_MAX_REPEAT(255) * pad(64) = 16320; K_TILE must fit.
    static_assert(QUANT_DYN_K_TILE <= VECTOR_MAX_REPEAT * (VECTOR_MAX_BYTESIZE / sizeof(float)),
                  "QUANT_DYN_K_TILE exceeds ReduceMax float limit");

    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (pnum_tokens) {
        uint32_t pnum_tokens_val = *((__gm__ uint32_t *)pnum_tokens);
        m = pnum_tokens_val < m ? pnum_tokens_val : m;
    }

    // Batched multi-row fast path: single k tile and 32B-aligned rows (see
    // quant_dyn_to_i8_rows); rows capped by the UB budget (2048B margin covers
    // the rowAbs/scaleStage tables), by the one-vector-instruction limit
    // (r * kPad <= 255 repeats of 64 fp32) and by m/block_num.
    if (k >= 32 && k % 32 == 0 && k <= QUANT_DYN_K_TILE) {
        uint32_t kPad = ROUND_UP(k, 256 / sizeof(dtype));
        uint32_t rMax = (uint32_t)MAX(1, (int64_t)(UB_SIZE - 4 * kPad - 2048) / (10 * kPad));
        uint32_t repMax = VECTOR_MAX_REPEAT * VECTOR_MAX_NUM_OF_FP32 / kPad;
        uint32_t rows = MIN(MIN(MIN(rMax, repMax), 32), MAX(1, m / block_num));
        if (rows >= 2) {
            __gm__ dtype *x_gm = reinterpret_cast<__gm__ dtype *>(x);
            __gm__ int8_t *z_gm = reinterpret_cast<__gm__ int8_t *>(z);
            __gm__ float *scales_gm = reinterpret_cast<__gm__ float *>(scales);
            quant_dyn_to_i8_rows<dtype>(x_gm, scales_gm, z_gm, m, k, rows);
            return;
        }
    }

    constexpr uint32_t k_tile = QUANT_DYN_K_TILE;
    uint32_t k_loop = DIV_ROUND_UP(k, k_tile);

    uint32_t k_pad = ROUND_UP(k_tile, (256 / sizeof(dtype)));
    uint32_t k_pad_row = ROUND_UP(k, 256 / sizeof(dtype));
    uint32_t k_repeats_row = k_pad_row / VECTOR_MAX_NUM_OF_FP32;

    auto *x1 = reinterpret_cast<__ubuf__ dtype *>((uintptr_t)0);
    auto *x2 = reinterpret_cast<__ubuf__ dtype *>(x1 + k_pad);
    auto *xf32_buf = reinterpret_cast<__ubuf__ float *>(x2 + k_pad);
    auto *xAbs_buf = reinterpret_cast<__ubuf__ float *>(xf32_buf + k_pad);
    auto *zf16_buf = reinterpret_cast<__ubuf__ half *>(xAbs_buf + k_pad);
    auto *z1 = reinterpret_cast<__ubuf__ int8_t *>(zf16_buf + k_pad);
    auto *z2 = reinterpret_cast<__ubuf__ int8_t *>(z1 + k_pad);
    auto *rowAbsUb = reinterpret_cast<__ubuf__ float *>(z2 + k_pad);
    auto *sum_addr = reinterpret_cast<__ubuf__ float *>(rowAbsUb + 1);
    assert((uint64_t)sum_addr <= UB_SIZE);

    __ubuf__ dtype *xBufs[2] = {x1, x2};
    __ubuf__ int8_t *zBufs[2] = {z1, z2};

    __gm__ dtype *x_gm = reinterpret_cast<__gm__ dtype *>(x);
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
            copy_gm_to_ubuf_align_b16(xBufs[eId], x_gm + rowOffset, 0, 1, k * sizeof(dtype), 0, 0,
                                      0, 0);
            set_flag(PIPE_MTE2, PIPE_V, eId);

            // FP16/BF16 -> FP32 (shared by phase1 absmax and phase2 convert)
            wait_flag(PIPE_MTE2, PIPE_V, eId);
            convert_input<dtype>(xf32_buf, xBufs[eId], k_repeats_row);
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
            uint32_t k_size_pad = ROUND_UP(k_size, 256 / sizeof(dtype));
            uint32_t k_repeats = k_size_pad / VECTOR_MAX_NUM_OF_FP32;

            // x tile GM -> UB
            wait_flag(PIPE_V, PIPE_MTE2, eId);
            copy_gm_to_ubuf_align_b16(xBufs[eId], x_gm + rowOffset + k_offset, 0, 1,
                                      k_size * sizeof(dtype), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, eId);

            // FP16/BF16 -> FP32
            wait_flag(PIPE_MTE2, PIPE_V, eId);
            convert_input<dtype>(xf32_buf, xBufs[eId], k_repeats);
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
            uint32_t k_size_pad = ROUND_UP(k_size, 256 / sizeof(dtype));
            uint32_t k_repeats = k_size_pad / VECTOR_MAX_NUM_OF_FP32;

            // x tile GM -> UB
            wait_flag(PIPE_V, PIPE_MTE2, eId);
            copy_gm_to_ubuf_align_b16(xBufs[eId], x_gm + rowOffset + k_offset, 0, 1,
                                      k_size * sizeof(dtype), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, eId);

            wait_flag(PIPE_MTE2, PIPE_V, eId);
            convert_input<dtype>(xf32_buf, xBufs[eId], k_repeats);
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
    extern "C" __global__ __aicore__ void quant_dynamic_##dtype(                             \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t k) \
    {                                                                                        \
        quant_dyn_to_i8<dtype>(in, scale, out, pnum_tokens, m, k);                           \
    }
#else
#define QUANT_DYN_FUNC_DEFINE(dtype)                                                         \
    extern "C" __global__ __aicore__ void quant_dynamic_##dtype(                             \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t k) \
    {                                                                                        \
    }

#endif
