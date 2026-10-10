/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

#define QUANT_K_TILE 4096

// out = int8(in * scale_reciprocal + offset)
// per-token quantion
// [m, k], activation; dtype = float16_t / bfloat16_t

// Batched multi-row path (k % 32 == 0, k <= QUANT_K_TILE, rows >= 2): each block
// takes groups of `rows` consecutive tokens. One 2D MTE2 burst loads [rows, k]
// (GM rows are contiguous, dstGap skips the UB row padding) and one 2D MTE3
// burst stores [rows, k] back, so the per-row cost of the single-row path
// (4 flag pairs + 2 row-sized bursts) collapses to 1 x per group. The per-k
// scale/offset tile is loaded and converted once per block. The row-independent
// elementwise stages (input convert, FP32->FP16->int8) run as one whole-group
// vector op each — only the per-row vmul/vadd against the shared fp32 tiles
// stays row-serial. x uses a single buffer (it is dead right after the group
// convert, so the next MTE2 still overlaps this group's vector work); z keeps
// a pingpong so MTE3 overlaps the next group's compute. k % 32 == 0 keeps every
// burst length (k * 2B in, k * 1B out) and UB gap a multiple of 32B; rows is
// also capped so r * kPad fits in one vector instruction (VECTOR_MAX_REPEAT
// repeats of 64 fp32).
template <typename dtype>
__aicore__ inline void quant_to_i8_rows(__gm__ dtype *in_buf, __gm__ dtype *scale_rec_gm,
                                        __gm__ dtype *offset_gm, __gm__ int8_t *out_buf, uint32_t m,
                                        uint32_t k, uint32_t rows)
{
    uint32_t kPad = ROUND_UP(k, VECTOR_REPEAT_BYTESIZE / sizeof(dtype));
    uint32_t kRepeats = kPad / VECTOR_MAX_NUM_OF_FP32;
    uint32_t numGroups = DIV_ROUND_UP(m, rows);

    // UB layout: x: [rows,kPad] dtype | xf32: [rows,kPad] float | z1,z2: [rows,kPad] int8
    //            | scale,offset: [kPad] dtype | scale_f32,offset_f32: [kPad] float
    //            | xf16: [rows,kPad] half.  bytes = 10*R*kPad + 12*kPad.
    auto *x = reinterpret_cast<__ubuf__ dtype *>((uintptr_t)0);
    auto *xf32 = reinterpret_cast<__ubuf__ float *>(x + rows * kPad);
    auto *z1 = reinterpret_cast<__ubuf__ int8_t *>(xf32 + rows * kPad);
    auto *z2 = reinterpret_cast<__ubuf__ int8_t *>(z1 + rows * kPad);
    auto *scale = reinterpret_cast<__ubuf__ dtype *>(z2 + rows * kPad);
    auto *offset = reinterpret_cast<__ubuf__ dtype *>(scale + kPad);
    auto *scale_f32 = reinterpret_cast<__ubuf__ float *>(offset + kPad);
    auto *offset_f32 = reinterpret_cast<__ubuf__ float *>(scale_f32 + kPad);
    auto *xf16 = reinterpret_cast<__ubuf__ half *>(offset_f32 + kPad);
    assert((uint64_t)(xf16 + rows * kPad) <= UB_SIZE);

    __ubuf__ int8_t *zBufs[2] = {z1, z2};

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    // scale/offset tile: loaded once per block, converted once
    copy_gm_to_ubuf_align_b16(scale, scale_rec_gm, 0, 1, k * sizeof(dtype), 0, 0, 0, 0);
    copy_gm_to_ubuf_align_b16(offset, offset_gm, 0, 1, k * sizeof(dtype), 0, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    convert_input<dtype>(scale_f32, scale, kRepeats);
    convert_input<dtype>(offset_f32, offset, kRepeats);
    pipe_barrier(PIPE_V);

    // 2D DMA configs: one burst per row, k % 32 == 0 keeps lengths in 32B units.
    // x in: GM rows contiguous (srcGap 0), UB rows padded to kPad (dstGap).
    // z out: UB rows padded (srcGap), GM rows contiguous (dstGap 0).
    // Tail groups (r < rows) rebuild the config with the actual burst count.
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

        // x [r, k] GM -> UB: r bursts of k elements, contiguous in GM, padded rows in UB
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        copy_gm_to_ubuf(x, in_buf + (uint64_t)base * k, inCfg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // whole-group FP16/BF16 -> FP32; x is dead right after, so the next
        // group's MTE2 can issue while this group's vector work is still running
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        convert_input<dtype>(xf32, x, rRepeats);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // dst = src * scale_reciprocal + offset, per row against the shared tile
        for (uint32_t i = 0; i < r; i++) {
            __ubuf__ float *row = xf32 + i * kPad;
            vmul(row, row, scale_f32, kRepeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(row, row, offset_f32, kRepeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }

        // whole-group FP32 -> FP16 -> int8; the z-buffer credit is only needed
        // before the int8 write, so the f32->f16 stage runs during the wait
        vconv_f322f16a(xf16, xf32, rRepeats, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eId);  // zBufs[eId] free from prev MTE3
        vconv_f162s8a(zBufs[eId], xf16, r * kPad / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eId);  // zBufs[eId] ready for MTE3

        // z [r, k] UB -> GM: r bursts of k int8, padded rows in UB, contiguous in GM
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eId);
        copy_ubuf_to_gm(out_buf + (uint64_t)base * k, zBufs[eId], outCfg);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eId);
        eId = 1 - eId;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

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

    // Batched multi-row path: single k tile and 32B-aligned rows let one group
    // share the DMA bursts; rows are capped by the UB budget, by the one-vector-
    // instruction limit (r * kPad <= 255 repeats of 64 fp32) and by m/block_num
    // so every launched block still gets at least one group.
    if (k >= 32 && k % 32 == 0 && k <= QUANT_K_TILE) {
        uint32_t kPad = ROUND_UP(k, VECTOR_REPEAT_BYTESIZE / sizeof(dtype));
        uint32_t rMax = (uint32_t)MAX(1, (int64_t)(UB_SIZE - 12 * kPad - 64) / (10 * kPad));
        uint32_t repMax = VECTOR_MAX_REPEAT * VECTOR_MAX_NUM_OF_FP32 / kPad;
        uint32_t rows = MIN(MIN(MIN(rMax, repMax), 32), MAX(1, m / block_num));
        if (rows >= 2) {
            quant_to_i8_rows<dtype>(in_buf, scale_reciprocal_gm, offset_gm, out_buf, m, k, rows);
            return;
        }
    }

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
        uint32_t k_size_pad = ROUND_UP(k_size, VECTOR_REPEAT_BYTESIZE / sizeof(dtype));
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
