/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"

#ifdef __DAV_C220_VEC__

// C8 cache writer for the V4 compressor path: quantize the post-norm/post-rope
// BF16 weightedSum rows and scatter them into the paged int8 key cache plus
// the fp16 per-slot scale. Byte-identical to indexer_prepare_c8's K path, so
// caches from either op are interchangeable for indexer_topk's C8 reader:
//   scale = absmax(row) / 127 (fp16)   int8 = round_nearest(x * 127/absmax)
// Zero row -> zeros with zero scale; slot < 0 or >= capacity -> row skipped.
// Addressing is block-strided: slot = block * blockSize + offset, int8 row at
// block * kCacheStride + offset * dim, scale at block * kScaleCacheStride +
// offset. DSV4's vllm pool packs K and scales into one page; a dense pool
// passes flat strides and degenerates to flat addressing.

template <typename dtype>
__aicore__ inline void quant_cache_c8(GM_ADDR x, GM_ADDR slots, GM_ADDR kCache, GM_ADDR kScaleCache,
                                      uint32_t rows, uint32_t dim, uint32_t capacity,
                                      uint32_t blockSize, uint32_t kCacheStride,
                                      uint32_t kScaleCacheStride)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t calcRepeat = dim / VECTOR_MAX_NUM_OF_FP32;
    uint32_t quantRepeat = DIV_ROUND_UP(dim, VECTOR_MAX_NUM_OF_FP16);

    // Ping-pong UB layout (sections 256B-aligned): only the cross-pipe buffers
    // double (xRow/slotBuf are MTE2-written, quant/scaleF16 MTE3-read);
    // V-internal buffers stay single — the V pipe serializes rows.
    uintptr_t off = 0;
    __ubuf__ dtype *xRow[2];
    xRow[0] = reinterpret_cast<__ubuf__ dtype *>(off);
    off += ROUND_UP(dim * sizeof(dtype), VECTOR_MAX_BYTESIZE);
    xRow[1] = reinterpret_cast<__ubuf__ dtype *>(off);
    off += ROUND_UP(dim * sizeof(dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ int32_t *slotBuf[2];
    slotBuf[0] = reinterpret_cast<__ubuf__ int32_t *>(off);
    off += VECTOR_MAX_BYTESIZE;
    slotBuf[1] = reinterpret_cast<__ubuf__ int32_t *>(off);
    off += VECTOR_MAX_BYTESIZE;
    auto *xRowF32 = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE);
    auto *absValues = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(dim * sizeof(float), VECTOR_MAX_BYTESIZE);
    auto *rounded = reinterpret_cast<__ubuf__ int32_t *>(off);
    off += ROUND_UP(dim * sizeof(int32_t), VECTOR_MAX_BYTESIZE);
    auto *halfValues = reinterpret_cast<__ubuf__ half *>(off);
    off += ROUND_UP(dim * sizeof(half), VECTOR_MAX_BYTESIZE);
    __ubuf__ int8_t *quant[2];
    quant[0] = reinterpret_cast<__ubuf__ int8_t *>(off);
    off += ROUND_UP(dim, VECTOR_MAX_BYTESIZE);
    quant[1] = reinterpret_cast<__ubuf__ int8_t *>(off);
    off += ROUND_UP(dim, VECTOR_MAX_BYTESIZE);
    auto *scaleF32 = reinterpret_cast<__ubuf__ float *>(off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ half *scaleF16[2];
    scaleF16[0] = reinterpret_cast<__ubuf__ half *>(off);
    off += VECTOR_MAX_BYTESIZE;
    scaleF16[1] = reinterpret_cast<__ubuf__ half *>(off);
    off += VECTOR_MAX_BYTESIZE;
    assert(off <= UB_SIZE);

    __gm__ dtype *xGm = reinterpret_cast<__gm__ dtype *>(x);
    __gm__ int32_t *slotsGm = reinterpret_cast<__gm__ int32_t *>(slots);
    __gm__ int8_t *kCacheGm = reinterpret_cast<__gm__ int8_t *>(kCache);
    __gm__ half *kScaleGm = reinterpret_cast<__gm__ half *>(kScaleCache);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    uint32_t curr = 0;
    for (uint32_t row = block_idx; row < rows; row += block_num) {
        // row + slot GM -> UB (slot curr returned two rows back)
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(xRow[curr], xGm + static_cast<uint64_t>(row) * dim,
                            dim * sizeof(dtype));
        CopyGmToUbufAligned(slotBuf[curr], slotsGm + row, sizeof(int32_t));
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int32_t slot = slotBuf[curr][0];
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if (slot < 0 || static_cast<uint32_t>(slot) >= capacity) {
            // Relay only: no compute/stores, but both slots are returned to
            // keep every wait paired
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
        } else {
            uint32_t block = static_cast<uint32_t>(slot) / blockSize;
            uint32_t offset = static_cast<uint32_t>(slot) % blockSize;

            convert_input<dtype>(xRowF32, xRow[curr], calcRepeat);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // xRow[curr] free two rows ahead

            vabs(absValues, xRowF32, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            ReduceMax(absValues, absValues, dim);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            float absmax = *absValues;
            // zero row -> factor 0: int8 zeros, scale 0
            float factor = absmax != 0.0f ? float(127) / absmax : 0.0f;

            // round_nearest via the int32r round-trip (see indexer_prepare_c8)
            vmuls(scaleF32, absValues, 1.0f / float(127), 1, 1, 1, 8, 8);
            vmuls(xRowF32, xRowF32, factor, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_f322s32r(rounded, xRowF32, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_s322f32(xRowF32, rounded, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_f322f16(halfValues, xRowF32, calcRepeat, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
            vconv_f322f16(scaleF16[curr], scaleF32, 1, 1, 1, 4, 8);
            vconv_f162s8(quant[curr], halfValues, quantRepeat, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);

            // paged stores: int8 row (align_b8 tolerates unaligned cache
            // views), fp16 scale (b16 path)
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm_align_b8(kCacheGm + static_cast<uint64_t>(block) * kCacheStride +
                                         static_cast<uint64_t>(offset) * dim,
                                     quant[curr], 0, 1, dim, 0, 0, 0, 0);
            CopyUbufToGmAligned(
                kScaleGm + static_cast<uint64_t>(block) * kScaleCacheStride + offset,
                scaleF16[curr], sizeof(half));
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // quant/scaleF16 free two rows ahead
        }

        curr = 1 - curr;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define QUANT_CACHE_C8_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void quant_cache_c8_##dtype(                            \
        GM_ADDR x, GM_ADDR slots, GM_ADDR kCache, GM_ADDR kScaleCache, uint32_t rows,        \
        uint32_t dim, uint32_t capacity, uint32_t blockSize, uint32_t kCacheStride,          \
        uint32_t kScaleCacheStride)                                                          \
    {                                                                                        \
        quant_cache_c8<dtype>(x, slots, kCache, kScaleCache, rows, dim, capacity, blockSize, \
                              kCacheStride, kScaleCacheStride);                              \
    }
#else
#define QUANT_CACHE_C8_FUNC_DEFINE(dtype)                                             \
    extern "C" __global__ __aicore__ void quant_cache_c8_##dtype(                     \
        GM_ADDR x, GM_ADDR slots, GM_ADDR kCache, GM_ADDR kScaleCache, uint32_t rows, \
        uint32_t dim, uint32_t capacity, uint32_t blockSize, uint32_t kCacheStride,   \
        uint32_t kScaleCacheStride)                                                   \
    {                                                                                 \
    }
#endif
