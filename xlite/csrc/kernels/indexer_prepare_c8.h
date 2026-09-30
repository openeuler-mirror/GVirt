/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "norm.h"

#ifdef __DAV_C220_VEC__

__aicore__ inline void IndexerC8Gather(__ubuf__ float *dst, __ubuf__ float *src,
                                       __ubuf__ uint32_t *indices, uint32_t repeat)
{
    vgather((__ubuf__ uint32_t *)dst, indices,
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(src)), 8, repeat);
}

// One per-core UB workspace, reused by the K and Q phases. No GM workspace is added.
struct IndexerC8Buffers {
    static constexpr uint32_t dim = 128;
    static constexpr uint32_t rope_dim = 64;
    static constexpr uint32_t hadamard_stages = 7;
    static constexpr float quant_max = 127.0f;
    static constexpr float norm_eps = 1e-6f;  // NOT the model's RMSNorm epsilon.
    // BF16(H / sqrt(128)), as exported by the target QuaRot recipe.
    static constexpr float hadamard_coefficient = 0.08837890625f;

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
    __ubuf__ int8_t *quant;
    __ubuf__ float *scale;
    __ubuf__ half *scale_half;
    __ubuf__ int64_t *position;
    __ubuf__ int32_t *slot;
    __ubuf__ uint32_t *cos_indices;
    __ubuf__ uint32_t *sin_indices;
    __ubuf__ uint32_t *swap_indices;
    __ubuf__ float *rope_sign;
    __ubuf__ uint32_t *h_indices;
    __ubuf__ float *h_sign;

    __aicore__ inline IndexerC8Buffers()
    {
    }

    __aicore__ inline void Init()
    {
        // UB layout: keep full-row padding and the existing byte offsets.
        // Separate BF16 buffers preserve the LN/RoPE/Hadamard rounding steps.
        constexpr uint32_t calc_size = ROUND_UP(dim * sizeof(float), UB_BUF_ALIGN_SIZE);
        constexpr uint32_t input_size = ROUND_UP(dim * sizeof(bfloat16_t), UB_BUF_ALIGN_SIZE);
        constexpr uint32_t rope_index_size =
            ROUND_UP(rope_dim * sizeof(uint32_t), UB_BUF_ALIGN_SIZE);
        uint64_t off = 0;
        ub_weight = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        ub_bias = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        x = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        tmp = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        other = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        trig = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        cos = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        sin = reinterpret_cast<__ubuf__ float *>(off);
        off += calc_size;
        bf = reinterpret_cast<__ubuf__ bfloat16_t *>(off);
        off += input_size;
        trig_bf = reinterpret_cast<__ubuf__ bfloat16_t *>(off);
        off += input_size;
        rounded = reinterpret_cast<__ubuf__ int32_t *>(off);
        off += ROUND_UP(dim * sizeof(int32_t), UB_BUF_ALIGN_SIZE);
        half_values = reinterpret_cast<__ubuf__ half *>(off);
        off += ROUND_UP(dim * sizeof(half), UB_BUF_ALIGN_SIZE);
        quant = reinterpret_cast<__ubuf__ int8_t *>(off);
        off += ROUND_UP(dim * sizeof(int8_t), VECTOR_REPEAT_BYTESIZE);
        scale = reinterpret_cast<__ubuf__ float *>(off);
        off += UB_BUF_ALIGN_SIZE;
        scale_half = reinterpret_cast<__ubuf__ half *>(off);
        off += UB_BUF_ALIGN_SIZE;
        position = reinterpret_cast<__ubuf__ int64_t *>(off);
        off += UB_BUF_ALIGN_SIZE;
        slot = reinterpret_cast<__ubuf__ int32_t *>(off);
        off += UB_BUF_ALIGN_SIZE;
        cos_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += rope_index_size;
        sin_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += rope_index_size;
        swap_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += rope_index_size;
        rope_sign = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(rope_dim * sizeof(float), UB_BUF_ALIGN_SIZE);
        h_indices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(hadamard_stages * dim * sizeof(uint32_t), UB_BUF_ALIGN_SIZE);
        h_sign = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(hadamard_stages * dim * sizeof(float), UB_BUF_ALIGN_SIZE);
        assert(off <= UB_SIZE);

        // Initialize read-only gather indices/signs once per core for both K and Q.
        for (uint32_t i = 0; i < rope_dim; ++i) {
            cos_indices[i] = (i / 2 * 2) * sizeof(float);
            sin_indices[i] = (i / 2 * 2 + 1) * sizeof(float);
            swap_indices[i] = (i ^ 1) * sizeof(float);
            rope_sign[i] = (i & 1) ? 1.0f : -1.0f;
        }
        for (uint32_t stage = 0; stage < hadamard_stages; ++stage) {
            for (uint32_t i = 0; i < dim; ++i) {
                h_indices[stage * dim + i] = (i ^ (1u << stage)) * sizeof(float);
                h_sign[stage * dim + i] = (i & (1u << stage)) ? -1.0f : 1.0f;
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }
};

__aicore__ inline void indexer_c8_load_row(GM_ADDR input, GM_ADDR freqs, uint32_t row,
                                           uint32_t row_stride, int64_t pos, bool freqs_fp32,
                                           const IndexerC8Buffers &ub)
{
    // Row addresses may not be 32-byte aligned.
    copy_gm_to_ubuf_align_b16(ub.bf, (__gm__ bfloat16_t *)input + (uint64_t)row * row_stride, 0, 1,
                              ub.dim * sizeof(bfloat16_t), 0, 0, 0, 0);
    if (freqs_fp32) {
        copy_gm_to_ubuf_align_b16(ub.trig, (__gm__ float *)freqs + pos * ub.rope_dim, 0, 1,
                                  ub.rope_dim * sizeof(float), 0, 0, 0, 0);
    } else {
        copy_gm_to_ubuf_align_b16(ub.trig_bf, (__gm__ bfloat16_t *)freqs + pos * ub.rope_dim, 0, 1,
                                  ub.rope_dim * sizeof(bfloat16_t), 0, 0, 0, 0);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    convert_input(ub.x, ub.bf, 2);
    if (freqs_fp32) {
        convert_output(ub.trig_bf, ub.trig, 1);
        pipe_barrier(PIPE_V);
    }
    convert_input(ub.trig, ub.trig_bf, 1);
    pipe_barrier(PIPE_V);
}

// Shared numerical transform; it does not select Q/K or write a cache.
__aicore__ inline void rope_hadamard_quant_c8(const IndexerC8Buffers &ub)
{
    uint32_t calc_repeat = DIV_ROUND_UP(ub.dim, VECTOR_MAX_NUM_OF_FP32);
    uint32_t rope_repeat = DIV_ROUND_UP(ub.rope_dim, VECTOR_MAX_NUM_OF_FP32);
    uint32_t quant_repeat = DIV_ROUND_UP(ub.dim, VECTOR_MAX_NUM_OF_FP16);

    // RoPE: adjacent pairs in the first rope_dim coordinates; tail is unchanged.
    IndexerC8Gather(ub.cos, ub.trig, ub.cos_indices, rope_repeat);
    IndexerC8Gather(ub.sin, ub.trig, ub.sin_indices, rope_repeat);
    IndexerC8Gather(ub.other, ub.x, ub.swap_indices, rope_repeat);
    pipe_barrier(PIPE_V);
    vmul(ub.tmp, ub.x, ub.cos, rope_repeat, 1, 1, 1, 8, 8, 8);
    vmul(ub.other, ub.other, ub.sin, rope_repeat, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmul(ub.other, ub.other, ub.rope_sign, rope_repeat, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(ub.x, ub.tmp, ub.other, rope_repeat, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    convert_output(ub.bf, ub.x, calc_repeat);
    pipe_barrier(PIPE_V);
    convert_input(ub.x, ub.bf, calc_repeat);
    pipe_barrier(PIPE_V);

    // Hadamard: FP32 butterflies, then normalize and round to BF16.
    for (uint32_t stage = 0; stage < ub.hadamard_stages; ++stage) {
        IndexerC8Gather(ub.other, ub.x, ub.h_indices + stage * ub.dim, calc_repeat);
        vmul(ub.tmp, ub.x, ub.h_sign + stage * ub.dim, calc_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(ub.x, ub.tmp, ub.other, calc_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    vmuls(ub.x, ub.x, float(ub.hadamard_coefficient), calc_repeat, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    convert_output(ub.bf, ub.x, calc_repeat);
    pipe_barrier(PIPE_V);
    convert_input(ub.x, ub.bf, calc_repeat);
    pipe_barrier(PIPE_V);

    // Dynamic quantization: persist the scale as FP16.
    vabs(ub.tmp, ub.x, calc_repeat, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    ReduceMax(ub.other, ub.tmp, ub.dim);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float amax = *ub.other;
    float scale = amax * (1.0f / ub.quant_max);
    float scaleRec = 0.0f;
    if (amax != 0.0f) {
        SetMask(1);
        vector_dup(ub.tmp, float(ub.quant_max), 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        // Match native dynamic quantization.
        vdiv(ub.tmp, ub.tmp, ub.other, 1, 1, 1, 1, 8, 8, 8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        scaleRec = *ub.tmp;
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
    *ub.scale = scale;
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    vmuls(ub.x, ub.x, scaleRec, calc_repeat, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    // Round directly FP32->INT32. FP32->FP16->INT8 would double-round.
    vconv_f322s32r(ub.rounded, ub.x, calc_repeat, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vconv_s322f32(ub.x, ub.rounded, calc_repeat, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vconv_f322f16(ub.half_values, ub.x, calc_repeat, 1, 1, 4, 8);
    pipe_barrier(PIPE_V);
    vconv_f162s8(ub.quant, ub.half_values, quant_repeat, 1, 1, 4, 8);
    SetMask(1);
    vconv_f322f16(ub.scale_half, ub.scale, 1, 1, 1, 4, 8);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

__aicore__ inline void indexer_c8_store_row(GM_ADDR quant_output, GM_ADDR scale_output,
                                            uint32_t dest, const IndexerC8Buffers &ub)
{
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // UB -> GM: scatter K rows into cache slots; write Q rows densely.
    copy_ubuf_to_gm_align_b8((__gm__ int8_t *)quant_output + (uint64_t)dest * ub.dim, ub.quant, 0,
                             1, ub.dim, 0, 0, 0, 0);
    // Exact 2-byte store: do not overwrite a neighbouring token's scale.
    CopyUbufToGmAligned((__gm__ half *)scale_output + dest, ub.scale_half, sizeof(half));
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

// K: one row per token, LayerNorm followed by the shared transform and slot-based stores.
__aicore__ inline void indexer_prepare_k_c8(GM_ADDR k, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs,
                                            GM_ADDR positions, GM_ADDR slots, GM_ADDR k_cache,
                                            GM_ADDR scale_cache, uint32_t tokens,
                                            uint32_t row_stride, uint32_t max_position,
                                            uint32_t capacity, bool freqs_fp32,
                                            const IndexerC8Buffers &ub)
{
    if (block_idx >= tokens) {
        return;
    }
    uint32_t norm_repeat = DIV_ROUND_UP(ub.dim, VECTOR_MAX_NUM_OF_FP32);
    float inv_norm_dim = 1.0f / ub.dim;

    copy_gm_to_ubuf_align_b16(ub.ub_weight, (__gm__ float *)weight, 0, 1, ub.dim * sizeof(float), 0,
                              0, 0, 0);
    copy_gm_to_ubuf_align_b16(ub.ub_bias, (__gm__ float *)bias, 0, 1, ub.dim * sizeof(float), 0, 0,
                              0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Preserve the round-robin row assignment (no double buffering).
    for (uint32_t row = block_idx; row < tokens; row += block_num) {
        CopyGmToUbufAligned(ub.position, (__gm__ int64_t *)positions + row, sizeof(int64_t));
        CopyGmToUbufAligned(ub.slot, (__gm__ int32_t *)slots + row, sizeof(int32_t));
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int64_t pos = *ub.position;
        int32_t dest = *ub.slot;
        // -1 is padding. Also guard invalid indices before reading RoPE or writing caches.
        if (dest < 0 || static_cast<uint32_t>(dest) >= capacity || pos < 0 ||
            static_cast<uint64_t>(pos) >= max_position) {
            continue;
        }
        indexer_c8_load_row(k, freqs, row, row_stride, pos, freqs_fp32, ub);

        // LayerNorm (K only), followed by BF16 rounding.
        vmuls(ub.tmp, ub.x, inv_norm_dim, norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        reduce_sum(ub.tmp, 1, ub.dim);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float mean = *ub.tmp;
        vadds(ub.x, ub.x, -mean, norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(ub.tmp, ub.x, ub.x, norm_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(ub.tmp, ub.tmp, inv_norm_dim, norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        reduce_sum(ub.tmp, 1, ub.dim);
        pipe_barrier(PIPE_V);
        SetMask(1);
        vadds(ub.tmp, ub.tmp, float(ub.norm_eps), 1, 1, 1, 0, 0);
        pipe_barrier(PIPE_V);
        vsqrt(ub.tmp, ub.tmp, 1, 1, 1, 0, 0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float inv_std = 1.0f / *ub.tmp;
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        vmuls(ub.x, ub.x, inv_std, norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(ub.x, ub.x, ub.ub_weight, norm_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(ub.x, ub.x, ub.ub_bias, norm_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        convert_output(ub.bf, ub.x, norm_repeat);
        pipe_barrier(PIPE_V);
        convert_input(ub.x, ub.bf, norm_repeat);
        pipe_barrier(PIPE_V);

        rope_hadamard_quant_c8(ub);
        indexer_c8_store_row(k_cache, scale_cache, dest, ub);
    }
    // Drain this core's K writes before Q reuses the same UB workspace.
    pipe_barrier(PIPE_ALL);
}

// Q: flatten [token, head], skip LayerNorm, and fold Q scales into the head weights.
__aicore__ inline void indexer_prepare_q_c8(GM_ADDR q, GM_ADDR kw, GM_ADDR freqs, GM_ADDR positions,
                                            GM_ADDR q8, GM_ADDR q_scale, GM_ADDR scaled_weights,
                                            uint32_t tokens, uint32_t heads, uint32_t max_position,
                                            bool freqs_fp32, const IndexerC8Buffers &ub)
{
    uint32_t n_rows = tokens * heads;
    for (uint32_t row = block_idx; row < n_rows; row += block_num) {
        CopyGmToUbufAligned(ub.position, (__gm__ int64_t *)positions + row / heads,
                            sizeof(int64_t));
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int64_t pos = *ub.position;
        int32_t dest = row;
        if (dest < 0 || static_cast<uint32_t>(dest) >= n_rows || pos < 0 ||
            static_cast<uint64_t>(pos) >= max_position) {
            continue;
        }
        indexer_c8_load_row(q, freqs, row, ub.dim, pos, freqs_fp32, ub);
        rope_hadamard_quant_c8(ub);
        indexer_c8_store_row(q8, q_scale, dest, ub);

        // Fold the FP16 Q scale into the head weight.
        uint64_t offset = (uint64_t)(row / heads) * (ub.dim + heads) + ub.dim + row % heads;
        CopyGmToUbufAligned(ub.bf, (__gm__ bfloat16_t *)kw + offset, sizeof(bfloat16_t));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        SetMask(1);
        convert_input(ub.tmp, ub.bf, 1);
        vconv_f162f32(ub.other, ub.scale_half, 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmul(ub.tmp, ub.tmp, ub.other, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(ub.half_values, ub.tmp, 1, 1, 1, 4, 8);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        CopyUbufToGmAligned((__gm__ half *)scaled_weights + row, ub.half_values, sizeof(half));
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}

// A single launch handles K and, for long sequences, Q. All projected inputs are read-only.
__aicore__ inline void indexer_prepare_c8(GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs,
                                          GM_ADDR positions, GM_ADDR slots, GM_ADDR k_cache,
                                          GM_ADDR k_scale_cache, GM_ADDR q, GM_ADDR q8,
                                          GM_ADDR q_scale, GM_ADDR scaled_weights, uint32_t tokens,
                                          uint32_t row_stride, uint32_t heads,
                                          uint32_t max_position, uint32_t capacity, bool freqs_fp32,
                                          bool is_long)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    if (block_idx >= (is_long ? tokens * heads : tokens)) {
        return;
    }
    IndexerC8Buffers ub;
    ub.Init();
    indexer_prepare_k_c8(kw, weight, bias, freqs, positions, slots, k_cache, k_scale_cache, tokens,
                         row_stride, max_position, capacity, freqs_fp32, ub);
    if (is_long) {
        indexer_prepare_q_c8(q, kw, freqs, positions, q8, q_scale, scaled_weights, tokens, heads,
                             max_position, freqs_fp32, ub);
    }
}

#define INDEXER_PREPARE_C8_FUNC_DEFINE(dtype)                                                      \
    extern "C" __global__ __aicore__ void indexer_prepare_c8_##dtype(                              \
        GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs, GM_ADDR positions, GM_ADDR slots, \
        GM_ADDR k_cache, GM_ADDR k_scale_cache, GM_ADDR q, GM_ADDR q8, GM_ADDR q_scale,            \
        GM_ADDR scaled_weights, uint32_t tokens, uint32_t row_stride, uint32_t heads,              \
        uint32_t max_position, uint32_t capacity, bool freqs_fp32, bool is_long)                   \
    {                                                                                              \
        indexer_prepare_c8(kw, weight, bias, freqs, positions, slots, k_cache, k_scale_cache, q,   \
                           q8, q_scale, scaled_weights, tokens, row_stride, heads, max_position,   \
                           capacity, freqs_fp32, is_long);                                         \
    }
#else
#define INDEXER_PREPARE_C8_FUNC_DEFINE(dtype)                                                      \
    extern "C" __global__ __aicore__ void indexer_prepare_c8_##dtype(                              \
        GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs, GM_ADDR positions, GM_ADDR slots, \
        GM_ADDR k_cache, GM_ADDR k_scale_cache, GM_ADDR q, GM_ADDR q8, GM_ADDR q_scale,            \
        GM_ADDR scaled_weights, uint32_t tokens, uint32_t row_stride, uint32_t heads,              \
        uint32_t max_position, uint32_t capacity, bool freqs_fp32, bool is_long)                   \
    {                                                                                              \
    }
#endif
