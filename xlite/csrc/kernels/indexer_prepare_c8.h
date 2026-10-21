/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_param.h"
#include "matmul.h"
#include "norm.h"

#ifdef __DAV_C220_VEC__

class IndexerC8Buffers
{
public:
    static constexpr uint32_t dim = 128;
    static constexpr uint32_t rope_dim = 64;
    static constexpr uint32_t rows = INDEXER_C8_TILE_ROWS / 2;
    static constexpr float quant_max = 127.0f;
    __ubuf__ float *ub_weight;
    __ubuf__ float *ub_bias;
    __ubuf__ float *x;
    __ubuf__ float *tmp;
    __ubuf__ float *other;
    __ubuf__ float *trig;
    __ubuf__ float *cos;
    __ubuf__ float *sin;
    __ubuf__ bfloat16_t *bf;
    __ubuf__ bfloat16_t *trig_bf;
    __ubuf__ int32_t *rounded;
    __ubuf__ half *half_values;
    __ubuf__ half *scale_half;
    __ubuf__ half *scale_scatter;
    __ubuf__ half *scaled_weights;
    __ubuf__ int8_t *quant;
    __ubuf__ float *stats;
    __ubuf__ float *scale;
    __ubuf__ float *factor;
    __ubuf__ float *numerator;
    __ubuf__ int64_t *position[PINGPONG_BUF_NUM];
    __ubuf__ int32_t *slot[PINGPONG_BUF_NUM];
    __ubuf__ bfloat16_t *head_weight[PINGPONG_BUF_NUM];
    __ubuf__ uint32_t *cos_indices;
    __ubuf__ uint32_t *sin_indices;
    __ubuf__ uint32_t *swap_indices;
    __ubuf__ uint32_t *row_indices;
    __ubuf__ float *rope_sign;

    __aicore__ inline void Init(GM_ADDR normWeight, GM_ADDR normBias)
    {
        uint64_t off = 0;
        ub_weight = reinterpret_cast<__ubuf__ float *>(off);
        off += dim * sizeof(float);
        ub_bias = reinterpret_cast<__ubuf__ float *>(off);
        off += dim * sizeof(float);
        x = reinterpret_cast<__ubuf__ float *>(off);
        off += rows * dim * sizeof(float);
        tmp = reinterpret_cast<__ubuf__ float *>(off);
        off += rows * dim * sizeof(float);
        other = reinterpret_cast<__ubuf__ float *>(off);
        off += rows * rope_dim * sizeof(float);
        trig = reinterpret_cast<__ubuf__ float *>(off);
        off += rows * rope_dim * sizeof(float);
        cos = reinterpret_cast<__ubuf__ float *>(off);
        off += rows * rope_dim * sizeof(float);
        sin = reinterpret_cast<__ubuf__ float *>(off);
        off += rows * rope_dim * sizeof(float);
        bf = reinterpret_cast<__ubuf__ bfloat16_t *>(off);
        off += rows * dim * sizeof(bfloat16_t);
        trig_bf = reinterpret_cast<__ubuf__ bfloat16_t *>(off);
        off += rows * rope_dim * sizeof(bfloat16_t);
        rounded = reinterpret_cast<__ubuf__ int32_t *>(off);
        off += rows * dim * sizeof(int32_t);
        half_values = reinterpret_cast<__ubuf__ half *>(off);
        off += rows * dim * sizeof(half);
        quant = reinterpret_cast<__ubuf__ int8_t *>(off);
        off += rows * dim * sizeof(int8_t);
        stats = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_MAX_BYTESIZE;
        scale = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_MAX_BYTESIZE;
        factor = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_MAX_BYTESIZE;
        numerator = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_MAX_BYTESIZE;
        scale_half = reinterpret_cast<__ubuf__ half *>(off);
        off += VECTOR_MAX_BYTESIZE;
        scale_scatter = reinterpret_cast<__ubuf__ half *>(off);
        off += rows * UB_BUF_ALIGN_SIZE;
        scaled_weights = reinterpret_cast<__ubuf__ half *>(off);
        off += VECTOR_MAX_BYTESIZE;
        for (uint32_t i = 0; i < PINGPONG_BUF_NUM; ++i) {
            position[i] = reinterpret_cast<__ubuf__ int64_t *>(off);
            off += rows * sizeof(int64_t);
            slot[i] = reinterpret_cast<__ubuf__ int32_t *>(off);
            off += rows * sizeof(int32_t);
            head_weight[i] = reinterpret_cast<__ubuf__ bfloat16_t *>(off);
            off += rows * sizeof(bfloat16_t);
        }
        cos_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += rows * rope_dim * sizeof(uint32_t);
        sin_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += rows * rope_dim * sizeof(uint32_t);
        swap_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += rows * rope_dim * sizeof(uint32_t);
        row_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += VECTOR_MAX_BYTESIZE;
        rope_sign = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_MAX_BYTESIZE;
        assert(off <= UB_SIZE);

        constexpr uint64_t sign_masks[] = {0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC,
                                           0xF0F0F0F0F0F0F0F0, 0xFF00FF00FF00FF00,
                                           0xFFFF0000FFFF0000, 0xFFFFFFFF00000000};
        vector_dup(rounded, int32_t(0), 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        for (uint32_t bit = 0; bit < 6; ++bit) {
            set_vector_mask(0, sign_masks[bit]);
            vadds(rounded, rounded, int32_t((1u << bit) * sizeof(float)), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        vmuls(reinterpret_cast<__ubuf__ int32_t *>(row_indices), rounded, int32_t(dim), 1, 1, 1, 8,
              8);
        // Each row has a different gather base; fill its 64 offsets together.
        for (uint32_t r = 0; r < rows; ++r) {
            auto *cos_index = reinterpret_cast<__ubuf__ int32_t *>(cos_indices) + r * rope_dim;
            auto *sin_index = reinterpret_cast<__ubuf__ int32_t *>(sin_indices) + r * rope_dim;
            auto *swap_index = reinterpret_cast<__ubuf__ int32_t *>(swap_indices) + r * rope_dim;
            vadds(cos_index, rounded, int32_t(r * rope_dim * sizeof(float)), 1, 1, 1, 8, 8);
            vadds(sin_index, rounded, int32_t((r * rope_dim + 1) * sizeof(float)), 1, 1, 1, 8, 8);
            vadds(swap_index, rounded, int32_t((r * dim + 1) * sizeof(float)), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0, sign_masks[0]);
            vadds(cos_index, rounded, (int32_t(r * rope_dim) - 1) * int32_t(sizeof(float)), 1, 1, 1,
                  8, 8);
            vadds(sin_index, rounded, int32_t(r * rope_dim * sizeof(float)), 1, 1, 1, 8, 8);
            vadds(swap_index, rounded, (int32_t(r * dim) - 1) * int32_t(sizeof(float)), 1, 1, 1, 8,
                  8);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }
        vector_dup(rope_sign, 1.0f, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, 0x5555555555555555);
        vector_dup(rope_sign, -1.0f, 1, 1, 1, 8, 0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        copy_gm_to_ubuf_align_b32(ub_weight, normWeight, 0, 1, dim * sizeof(float), 0, 0, 0, 0);
        copy_gm_to_ubuf_align_b32(ub_bias, normBias, 0, 1, dim * sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void LayerNorm(uint32_t count, float norm_eps)
    {
        uint32_t norm_repeat = DIV_ROUND_UP(dim, VECTOR_MAX_NUM_OF_FP32);
        vmuls(tmp, x, 1.0f / dim, count * norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        reduce_sum(tmp, count, dim);
        pipe_barrier(PIPE_V);
        SetMask(count);
        vgather((__ubuf__ uint32_t *)stats, row_indices, (uint32_t)(uint64_t)tmp, 8, 1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        for (uint32_t r = 0; r < count; ++r) {
            vadds(x + r * dim, x + r * dim, -stats[r], norm_repeat, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        vmul(tmp, x, x, count * norm_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(tmp, tmp, 1.0f / dim, count * norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        reduce_sum(tmp, count, dim);
        pipe_barrier(PIPE_V);
        SetMask(count);
        vgather((__ubuf__ uint32_t *)stats, row_indices, (uint32_t)(uint64_t)tmp, 8, 1);
        pipe_barrier(PIPE_V);
        vadds(stats, stats, norm_eps, 1, 1, 1, 8, 8);
        vector_dup(numerator, 1.0f, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vsqrt(stats, stats, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(factor, numerator, stats, 1, 1, 1, 1, 8, 8, 8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        for (uint32_t r = 0; r < count; ++r) {
            float rowFactor = factor[r];
            vmuls(x + r * dim, x + r * dim, rowFactor, norm_repeat, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        for (uint32_t half = 0; half < 2; ++half) {
            uint32_t off = half * VECTOR_MAX_NUM_OF_FP32;
            vmul(x + off, x + off, ub_weight + off, count, 1, 1, 1, 16, 16, 0);
        }
        pipe_barrier(PIPE_V);
        for (uint32_t half = 0; half < 2; ++half) {
            uint32_t off = half * VECTOR_MAX_NUM_OF_FP32;
            vadd(x + off, x + off, ub_bias + off, count, 1, 1, 1, 16, 16, 0);
        }
        pipe_barrier(PIPE_V);
        convert_output(bf, x, count * norm_repeat);
        pipe_barrier(PIPE_V);
        convert_input(x, bf, count * norm_repeat);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void RoPE(uint32_t count, bool isKey)
    {
        uint32_t calc_repeat = DIV_ROUND_UP(dim, VECTOR_MAX_NUM_OF_FP32);
        uint32_t trig_base = static_cast<uint32_t>(reinterpret_cast<uint64_t>(trig));
        uint32_t x_base = static_cast<uint32_t>(reinterpret_cast<uint64_t>(x));
        uint32_t trigRows = isKey ? count : 1;
        vgather((__ubuf__ uint32_t *)cos, cos_indices, trig_base, 8, trigRows);
        vgather((__ubuf__ uint32_t *)sin, sin_indices, trig_base, 8, trigRows);
        vgather((__ubuf__ uint32_t *)other, swap_indices, x_base, 8, count);
        pipe_barrier(PIPE_V);
        vmul(tmp, x, cos, count, 1, 1, 1, 8, 16, isKey ? 8 : 0);
        vmul(other, other, sin, count, 1, 1, 1, 8, 8, isKey ? 8 : 0);
        pipe_barrier(PIPE_V);
        vmul(other, other, rope_sign, count, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        vadd(x, tmp, other, count, 1, 1, 1, 16, 8, 8);
        pipe_barrier(PIPE_V);
        convert_output(bf, x, count * calc_repeat);
    }

    __aicore__ inline void Quantize(uint32_t count)
    {
        uint32_t calc_repeat = DIV_ROUND_UP(dim, VECTOR_MAX_NUM_OF_FP32);
        uint32_t quant_repeat = DIV_ROUND_UP(dim, VECTOR_MAX_NUM_OF_FP16);
        vabs(tmp, x, count * calc_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        for (uint32_t r = 0; r < count; ++r) {
            ReduceMax(tmp + r * dim, tmp + r * dim, dim);
        }
        pipe_barrier(PIPE_V);
        SetMask(count);
        vgather((__ubuf__ uint32_t *)stats, row_indices, (uint32_t)(uint64_t)tmp, 8, 1);
        vector_dup(factor, 0.0f, 1, 1, 1, 8, 0);
        vector_dup(numerator, static_cast<float>(quant_max), 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vmuls(scale, stats, 1.0f / quant_max, 1, 1, 1, 8, 8);
        vcmpvs_ne(reinterpret_cast<__ubuf__ uint8_t *>(rounded), stats, 0.0f, 1, 0, 1, 0, 8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        uint64_t nonzero = *reinterpret_cast<__ubuf__ uint64_t *>(rounded);
        nonzero &= (1ull << count) - 1;
        if (nonzero) {
            set_vector_mask(0, nonzero);
            vdiv(factor, numerator, stats, 1, 1, 1, 1, 8, 8, 8);
        }
        SetMask(count);
        vconv_f322f16(scale_half, scale, 1, 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        for (uint32_t r = 0; r < count; ++r) {
            float rowFactor = factor[r];
            vmuls(x + r * dim, x + r * dim, rowFactor, calc_repeat, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        // Preserve direct FP32 -> nearest INT32 rounding before INT8 packing.
        vconv_f322s32r(rounded, x, count * calc_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_s322f32(x, rounded, count * calc_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(half_values, x, count * calc_repeat, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_f162s8(quant, half_values, count * quant_repeat, 1, 1, 4, 8);
    }
};

__aicore__ inline bool indexer_c8_valid(int64_t pos, int32_t slot, uint32_t maxPosition,
                                        uint32_t capacity)
{
    return pos >= 0 && static_cast<uint64_t>(pos) < maxPosition && slot >= 0 &&
           static_cast<uint32_t>(slot) < capacity;
}

__aicore__ inline void indexer_c8_prepare_tile(GM_ADDR kw, GM_ADDR freqs, GM_ADDR positions,
                                               GM_ADDR slots, GM_ADDR q,
                                               __gm__ bfloat16_t *prepared, uint32_t task,
                                               uint32_t keyTiles, uint32_t tokens,
                                               uint32_t rowStride, uint32_t maxPosition,
                                               uint32_t capacity, bool freqsFp32, float normEps,
                                               uint32_t curr, IndexerC8Buffers &ub)
{
    bool isKey = task < keyTiles;
    uint32_t rowOffset = get_subblockid() * ub.rows;
    uint32_t first = isKey ? task * INDEXER_C8_TILE_ROWS + rowOffset : task - keyTiles;
    uint32_t count = isKey ? (first < tokens ? MIN(ub.rows, tokens - first) : 0) : ub.rows;
    if (count == 0) {
        return;
    }
    // TODO: Use ping-pong UB input buffers to overlap loading and vector computation.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    CopyGmToUbufAligned(ub.position[curr], (__gm__ int64_t *)positions + first,
                        (isKey ? count : 1) * sizeof(int64_t));
    if (isKey) {
        CopyGmToUbufAligned(ub.slot[curr], (__gm__ int32_t *)slots + first,
                            count * sizeof(int32_t));
        copy_gm_to_ubuf_align_b16(ub.bf, (__gm__ bfloat16_t *)kw + (uint64_t)first * rowStride, 0,
                                  count, ub.dim * sizeof(bfloat16_t), 0, 0,
                                  (rowStride - ub.dim) * sizeof(bfloat16_t), 0);
    } else {
        CopyGmToUbufAligned(ub.bf,
                            (__gm__ bfloat16_t *)q + ((uint64_t)first * 32 + rowOffset) * ub.dim,
                            count * ub.dim * sizeof(bfloat16_t));
        copy_gm_to_ubuf_align_b16(
            ub.head_weight[curr],
            (__gm__ bfloat16_t *)kw + (uint64_t)first * rowStride + ub.dim + rowOffset, 0, 1,
            count * sizeof(bfloat16_t), 0, 0, 0, 0);
    }
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    uint32_t trigRows = isKey ? count : 1;
    vector_dup(ub.trig, 0.0f, trigRows, 1, 1, 8, 0);
    if (!freqsFp32) {
        pipe_barrier(PIPE_V);
        convert_output(ub.trig_bf, ub.trig, trigRows);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (uint32_t r = 0; r < trigRows; ++r) {
        int64_t pos = ub.position[curr][r];
        bool valid = isKey ? indexer_c8_valid(pos, ub.slot[curr][r], maxPosition, capacity)
                           : pos >= 0 && static_cast<uint64_t>(pos) < maxPosition;
        if (!valid) {
            continue;
        }
        if (freqsFp32) {
            CopyGmToUbufAligned(ub.trig + r * ub.rope_dim,
                                (__gm__ float *)freqs + pos * ub.rope_dim,
                                ub.rope_dim * sizeof(float));
        } else {
            CopyGmToUbufAligned(ub.trig_bf + r * ub.rope_dim,
                                (__gm__ bfloat16_t *)freqs + pos * ub.rope_dim,
                                ub.rope_dim * sizeof(bfloat16_t));
        }
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    convert_input(ub.x, ub.bf, count * 2);
    if (freqsFp32) {
        convert_output(ub.trig_bf, ub.trig, trigRows);
        pipe_barrier(PIPE_V);
    }
    convert_input(ub.trig, ub.trig_bf, trigRows);
    pipe_barrier(PIPE_V);
    if (isKey) {
        ub.LayerNorm(count, normEps);
    }
    // TODO: Evaluate removing invalid-row zeroing.
    for (uint32_t r = 0; r < count; ++r) {
        int64_t pos = ub.position[curr][isKey ? r : 0];
        bool valid = isKey ? indexer_c8_valid(pos, ub.slot[curr][r], maxPosition, capacity)
                           : pos >= 0 && static_cast<uint64_t>(pos) < maxPosition;
        if (!valid) {
            vector_dup(ub.x + r * ub.dim, 0.0f, 2, 1, 1, 8, 0);
        }
    }
    pipe_barrier(PIPE_V);
    ub.RoPE(count, isKey);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    CopyUbufToGmAligned(prepared + rowOffset * ub.dim, ub.bf, count * ub.dim * sizeof(bfloat16_t));
    // The next tile loads into bf through MTE2, not V.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

__aicore__ inline void indexer_c8_store_tile(__gm__ bfloat16_t *rotated, GM_ADDR kCache,
                                             GM_ADDR kScaleCache, GM_ADDR q8, GM_ADDR qScale,
                                             GM_ADDR scaledWeights, uint32_t task,
                                             uint32_t keyTiles, uint32_t tokens,
                                             uint32_t maxPosition, uint32_t capacity, uint32_t curr,
                                             IndexerC8Buffers &ub)
{
    bool isKey = task < keyTiles;
    uint32_t rowOffset = get_subblockid() * ub.rows;
    uint32_t first = isKey ? task * INDEXER_C8_TILE_ROWS + rowOffset : task - keyTiles;
    uint32_t count = isKey ? (first < tokens ? MIN(ub.rows, tokens - first) : 0) : ub.rows;
    if (count == 0 || (!isKey && (ub.position[curr][0] < 0 ||
                                  static_cast<uint64_t>(ub.position[curr][0]) >= maxPosition))) {
        return;
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    CopyGmToUbufAligned(ub.bf, rotated + rowOffset * ub.dim, count * ub.dim * sizeof(bfloat16_t));
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    convert_input(ub.x, ub.bf, count * 2);
    pipe_barrier(PIPE_V);
    ub.Quantize(count);
    if (!isKey) {
        SetMask(count);
        convert_input(ub.tmp, ub.head_weight[curr], 1);
        vconv_f162f32(ub.other, ub.scale_half, 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmul(ub.tmp, ub.tmp, ub.other, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(ub.scaled_weights, ub.tmp, 1, 1, 1, 4, 8);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    } else {
        // Each scattered 2-byte scale needs a 32-byte-aligned UB source.
        vbrcb(reinterpret_cast<__ubuf__ uint16_t *>(ub.scale_scatter),
              reinterpret_cast<__ubuf__ uint16_t *>(ub.scale_half), 1, 8, DIV_ROUND_UP(count, 8));
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if (isKey) {
        for (uint32_t r = 0; r < count; ++r) {
            int32_t dest = ub.slot[curr][r];
            if (!indexer_c8_valid(ub.position[curr][r], dest, maxPosition, capacity)) {
                continue;
            }
            copy_ubuf_to_gm_align_b8((__gm__ int8_t *)kCache + (uint64_t)dest * ub.dim,
                                     ub.quant + r * ub.dim, 0, 1, ub.dim, 0, 0, 0, 0);
            CopyUbufToGmAligned((__gm__ half *)kScaleCache + dest, ub.scale_scatter + r * 16,
                                sizeof(half));
        }
    } else {
        uint64_t row = (uint64_t)first * 32 + rowOffset;
        CopyUbufToGmAligned((__gm__ int8_t *)q8 + row * ub.dim, ub.quant, count * ub.dim);
        CopyUbufToGmAligned((__gm__ half *)qScale + row, ub.scale_half, count * sizeof(half));
        CopyUbufToGmAligned((__gm__ half *)scaledWeights + row, ub.scaled_weights,
                            count * sizeof(half));
    }
    // TODO: Wait for output DMA only when its UB buffers are reused.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}
#endif

__aicore__ inline void indexer_prepare_c8(GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs,
                                          GM_ADDR positions, GM_ADDR slots, GM_ADDR kCache,
                                          GM_ADDR kScaleCache, GM_ADDR q, GM_ADDR q8,
                                          GM_ADDR qScale, GM_ADDR scaledWeights, GM_ADDR hadamard,
                                          GM_ADDR workspace, uint32_t tokens, uint32_t rowStride,
                                          uint32_t heads, uint32_t maxPosition, uint32_t capacity,
                                          bool freqsFp32, bool isLong, float normEps)
{
    uint32_t keyTiles = DIV_ROUND_UP(tokens, INDEXER_C8_TILE_ROWS);
    uint32_t tasks = keyTiles + (isLong ? tokens : 0);
    uint32_t perCore = block_idx < tasks ? DIV_ROUND_UP(tasks - block_idx, block_num) : 0;
    if (perCore == 0) {
        return;
    }
    constexpr uint32_t tileElems = INDEXER_C8_TILE_ROWS * 128;
    __gm__ bfloat16_t *prepared[PINGPONG_BUF_NUM];
    __gm__ bfloat16_t *rotated[PINGPONG_BUF_NUM];
    for (uint32_t i = 0; i < PINGPONG_BUF_NUM; ++i) {
        prepared[i] = (__gm__ bfloat16_t *)workspace + (i * block_num + block_idx) * tileElems;
        rotated[i] = (__gm__ bfloat16_t *)workspace +
                     ((i + PINGPONG_BUF_NUM) * block_num + block_idx) * tileElems;
    }
#ifdef __DAV_C220_CUBE__
    Matmul<bfloat16_t, float, bfloat16_t> mm;
    mm.Init(INDEXER_C8_TILE_ROWS, 128, 128, false, false, false, false, 0);
    mm.SetFlags();
    for (uint32_t i = 0; i < perCore; ++i) {
        uint32_t curr = i % PINGPONG_BUF_NUM;
        uint32_t task = block_idx + i * block_num;
        uint32_t rows = task < keyTiles
                            ? MIN(INDEXER_C8_TILE_ROWS, tokens - task * INDEXER_C8_TILE_ROWS)
                            : heads;
        wait_flag_dev(curr + 2);
        mm.TaskTilesInit((GM_ADDR)prepared[curr], hadamard, (GM_ADDR)rotated[curr], nullptr,
                         nullptr, rows, 128, 128);
        mm.RunTileByIdx(0);
        ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (curr << 8));
    }
    mm.WaitFlags();
#elif defined(__DAV_C220_VEC__)
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    IndexerC8Buffers ub;
    ub.Init(weight, bias);
    // Prime both slots; reuse a slot only after its consumer finishes.
    for (uint32_t i = 0; i < MIN(PINGPONG_BUF_NUM, perCore); ++i) {
        indexer_c8_prepare_tile(kw, freqs, positions, slots, q, prepared[i],
                                block_idx + i * block_num, keyTiles, tokens, rowStride, maxPosition,
                                capacity, freqsFp32, normEps, i, ub);
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((i + 2) << 8));
    }
    for (uint32_t i = 0; i < perCore; ++i) {
        uint32_t curr = i % PINGPONG_BUF_NUM;
        wait_flag_dev(curr);
        indexer_c8_store_tile(rotated[curr], kCache, kScaleCache, q8, qScale, scaledWeights,
                              block_idx + i * block_num, keyTiles, tokens, maxPosition, capacity,
                              curr, ub);
        if (i + PINGPONG_BUF_NUM < perCore) {
            indexer_c8_prepare_tile(kw, freqs, positions, slots, q, prepared[curr],
                                    block_idx + (i + PINGPONG_BUF_NUM) * block_num, keyTiles,
                                    tokens, rowStride, maxPosition, capacity, freqsFp32, normEps,
                                    curr, ub);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((curr + 2) << 8));
        }
    }
    pipe_barrier(PIPE_ALL);
#endif
}

#define INDEXER_PREPARE_C8_FUNC_DEFINE(dtype)                                                      \
    extern "C" __global__ __aicore__ void indexer_prepare_c8_##dtype(                              \
        GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs, GM_ADDR positions, GM_ADDR slots, \
        GM_ADDR k_cache, GM_ADDR k_scale_cache, GM_ADDR q, GM_ADDR q8, GM_ADDR q_scale,            \
        GM_ADDR scaled_weights, GM_ADDR hadamard, GM_ADDR workspace, uint32_t tokens,              \
        uint32_t row_stride, uint32_t heads, uint32_t max_position, uint32_t capacity,             \
        bool freqs_fp32, bool is_long, float norm_eps)                                             \
    {                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                         \
        indexer_prepare_c8(kw, weight, bias, freqs, positions, slots, k_cache, k_scale_cache, q,   \
                           q8, q_scale, scaled_weights, hadamard, workspace, tokens, row_stride,   \
                           heads, max_position, capacity, freqs_fp32, is_long, norm_eps);          \
    }
