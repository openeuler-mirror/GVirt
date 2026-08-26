/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"
#include "norm.h"
#include "rope_complex_and_cache.h"
#include "muls.h"

#ifdef __DAV_C220_VEC__

template <typename Dtype>
__aicore__ void norm_ropex_cache_muls(GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR position,
                                      float norm_eps, GM_ADDR freqs, uint32_t n_rows,
                                      uint32_t n_cols, uint32_t norm_dim, uint32_t rope_dim,
                                      GM_ADDR kcache = nullptr, GM_ADDR slot_mapping = nullptr,
                                      uint32_t block_size = 0, uint32_t scale_dim = 0,
                                      float scale = 1.0f, uint32_t top_k = 2048,
                                      int core_offset = 0, int *next_core_offset = nullptr)
{
    /*
     * | <----------------------------- n_cols ------------------------------> |
     * | <------------------------- calc_dim ------------------------> |
     * | <-------------- norm_dim -------------> | <--- scale_dim ---> |
     * | <--- rope_dim ---> | <--- nope_dim ---> | <--- scale_dim ---> |
     *
     * NOTES:
     * 1. set `index_k_cache` to nullptr if no caching is needed
     * 2. set `scale_dim` to 0 if no scaling is needed
     * 3. currently only support LayerNorm with one attention head, assuming rope_dim > 0
     * 4. `rope` is the first half of the norm_dim, and `nope` is the second half of the norm_dim
     * 5. all dims are expected to be multiples of 64 to avoid unexpected overflows
     *
     */
    uint32_t n_rows_per_core = DIV_ROUND_UP(n_rows, block_num);
    uint32_t rel_block_idx =
        (block_idx + block_num - core_offset) % block_num;  // relative block index
    if (next_core_offset) {
        *next_core_offset = (core_offset + n_rows) % block_num;
    }
    if (rel_block_idx * n_rows_per_core >= n_rows) {
        return;
    }

    float inv_norm_dim = 1.0f / norm_dim;
    uint32_t calc_dim = norm_dim + scale_dim;
    uint32_t col_size = n_cols * sizeof(Dtype);
    uint32_t calc_size = calc_dim * sizeof(Dtype);
    uint32_t norm_size = norm_dim * sizeof(Dtype);
    uint32_t scale_size = scale_dim * sizeof(Dtype);
    bool needs_position = rope_dim > 0 || scale_dim > 0;
    bool needs_cache = (kcache != nullptr) && (slot_mapping != nullptr) && (block_size != 0);

    // bytesizes, repeats, etc.
    uint32_t ub_pos_num = MIN(n_rows_per_core, GM_UB_BURST_SIZE / sizeof(uint64_t));

    // The following UB layout is designed for NPU arch 220*
    uint64_t off = 0, invoff = UB_SIZE - UB_BANKGROUP_ROW_SIZE;
    uint32_t norm_size_row =
        ROUND_UP(norm_dim * sizeof(float) + UB_BANK_CONFLICT_OFFSET, UB_BANKGROUP_ROW_SIZE);
    __ubuf__ float *ub_weight =
        reinterpret_cast<UBA(float)>(off + UB_BANK_CONFLICT_OFFSET);  // starts at bank group 8 or 0
    off += norm_size_row;
    __ubuf__ float *ub_bias =
        reinterpret_cast<UBA(float)>(off + UB_BANK_CONFLICT_OFFSET);  // starts at bank group 8 or 0
    off += norm_size_row;
    __ubuf__ uint64_t *ub_pos = reinterpret_cast<UBA(uint64_t)>(off);  // starts at bank group 0
    off += ROUND_UP(ub_pos_num * sizeof(uint64_t), UB_BANKGROUP_ROW_SIZE);
    __ubuf__ uint32_t *ub_slot_mapping = reinterpret_cast<UBA(uint32_t)>(off);
    if (needs_cache) {
        off += ROUND_UP(ub_pos_num * sizeof(uint32_t), UB_BANKGROUP_ROW_SIZE);
    }

    // in/out buffers with Dtype, and copy in buffer for freqs with float
    uint32_t col_size_row = ROUND_UP(col_size + UB_BANK_CONFLICT_OFFSET, UB_BANKGROUP_ROW_SIZE);
    uint32_t rope_size_row =
        ROUND_UP(rope_dim * sizeof(float) + UB_BANK_CONFLICT_OFFSET, UB_BANKGROUP_ROW_SIZE);
    __ubuf__ Dtype *in0 = reinterpret_cast<UBA(Dtype)>(off);
    off += col_size_row;
    __ubuf__ Dtype *out0 = reinterpret_cast<UBA(Dtype)>(off);
    off += col_size_row;
    __ubuf__ Dtype *in1 = reinterpret_cast<UBA(Dtype)>(off);
    off += col_size_row;
    __ubuf__ Dtype *out1 = reinterpret_cast<UBA(Dtype)>(off);
    off += col_size_row;
    __ubuf__ float *ub_freqs0 = reinterpret_cast<UBA(float)>(off);
    off += rope_size_row;
    __ubuf__ float *ub_freqs1 = reinterpret_cast<UBA(float)>(off);
    off += rope_size_row;
    __ubuf__ float *calc_even_cos =
        reinterpret_cast<UBA(float)>(off);  // rope, starts at bank group 0
    off += rope_size_row;
    __ubuf__ float *calc_even_sin =
        reinterpret_cast<UBA(float)>(off);  // rope, starts at bank group 0
    off += rope_size_row;
    __ubuf__ float *calc_odd_cos =
        reinterpret_cast<UBA(float)>(off + UB_BANK_CONFLICT_OFFSET);  // group 8 or 0
    off += rope_size_row;
    __ubuf__ float *calc_odd_sin =
        reinterpret_cast<UBA(float)>(off + UB_BANK_CONFLICT_OFFSET);  // group 8 or 0
    off += rope_size_row;

    // TODO: optimize for bank conflict
    uint32_t calc_size_row =
        ROUND_UP(calc_dim * sizeof(float) + UB_BANK_CONFLICT_OFFSET, UB_BANKGROUP_ROW_SIZE);
    invoff -= calc_size_row;
    __ubuf__ float *in_float = reinterpret_cast<UBA(float)>(invoff);  // for norm + rope + scale
    invoff -= calc_size_row;
    __ubuf__ float *out_float =
        reinterpret_cast<UBA(float)>(invoff + UB_BANK_CONFLICT_OFFSET);  // for norm + rope + scale
    invoff -= norm_size_row;
    __ubuf__ float *calc_float = reinterpret_cast<UBA(float)>(invoff);  // only for norm
    invoff -= rope_size_row;
    __ubuf__ float *in_even = reinterpret_cast<UBA(float)>(
        invoff + UB_BANK_CONFLICT_OFFSET);  // only for rope, starts at group 8 or 0
    invoff -= rope_size_row;
    __ubuf__ float *in_odd =
        reinterpret_cast<UBA(float)>(invoff);  // only for rope, starts at bank group 0
    invoff -= rope_size_row;
    __ubuf__ float *cos_float =
        reinterpret_cast<UBA(float)>(invoff);  // only for rope, starts at bank group 0
    invoff -= rope_size_row;
    __ubuf__ float *sin_float = reinterpret_cast<UBA(float)>(
        invoff + UB_BANK_CONFLICT_OFFSET);  // only for rope, starts at group 8 or 0
    assert(off <= invoff);

    __ubuf__ Dtype *in[2] = {in0, in1};
    __ubuf__ Dtype *out[2] = {out0, out1};
    __ubuf__ float *ub_freqs[2] = {ub_freqs0, ub_freqs1};

    uint32_t len_burst_calc = DIV_ROUND_UP(calc_dim, BLOCK_SIZE / sizeof(Dtype));
    uint32_t len_burst_norm = DIV_ROUND_UP(norm_dim, BLOCK_SIZE / sizeof(Dtype));
    uint32_t calc_repeat = DIV_ROUND_UP(calc_dim, VECTOR_REPEAT_BYTESIZE / sizeof(float));
    uint32_t norm_repeat = DIV_ROUND_UP(norm_dim, VECTOR_REPEAT_BYTESIZE / sizeof(float));
    uint32_t rope_repeat = DIV_ROUND_UP(rope_dim, VECTOR_REPEAT_BYTESIZE / sizeof(float));
    uint32_t rope_blocks = DIV_ROUND_UP(rope_dim, BLOCK_SIZE / sizeof(float));
    uint32_t half_rope_blocks = DIV_ROUND_UP(rope_dim, BLOCK_SIZE * 2 / sizeof(float));
    uint32_t scale_repeat = DIV_ROUND_UP(scale_dim, VECTOR_REPEAT_BYTESIZE / sizeof(float));
    // assert(scale_dim == 0 || scale != 1.0f);

    uint32_t curr = 0;
    // whether to load weights/biases from GM to UB, only load once per core
    if (weight) {
        copy_gm_to_ubuf(in[curr], (__gm__ Dtype *)weight, 0, 1, len_burst_norm, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        convert_input(ub_weight, in[curr], norm_repeat);
        curr = 1 - curr;
    }
    if (bias) {
        copy_gm_to_ubuf(in[curr], (__gm__ Dtype *)bias, 0, 1, len_burst_norm, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        convert_input(ub_bias, in[curr], norm_repeat);
        curr = 1 - curr;
    }
    pipe_barrier(PIPE_V);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // copy in -> in
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);  // copy in -> in
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);  // copy in -> freqs
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);  // copy in -> freqs
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // copy out <- out
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);  // copy out <- out

    uint32_t ub_pos_start = 0, ub_pos_end = 0;  // end is exclusive
    uint32_t irow_end = MIN(n_rows, (rel_block_idx + 1) * n_rows_per_core);
    for (uint32_t irow = rel_block_idx * n_rows_per_core; irow < irow_end; irow++) {
        // move position/slot_mapping to UB if needed
        if (irow >= ub_pos_end) {
            ub_pos_start = irow;
            ub_pos_end = MIN(irow + ub_pos_num, n_rows);
            uint32_t n_rows_pos = ub_pos_end - ub_pos_start;
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            CopyGmToUbufAligned(ub_pos, ((__gm__ uint64_t *)position) + ub_pos_start,
                                n_rows_pos * sizeof(uint64_t));
            if (needs_cache) {
                CopyGmToUbufAligned(ub_slot_mapping,
                                    ((__gm__ uint32_t *)slot_mapping) + ub_pos_start,
                                    n_rows_pos * sizeof(uint32_t));
            }
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        }

        // copy in: GM -> UB
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // acquire in[curr]
        auto *input_gm = (__gm__ Dtype *)(kw) + irow * n_cols;
        copy_gm_to_ubuf(in[curr], input_gm, 0, 1, len_burst_calc, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        convert_input(in_float, in[curr], calc_repeat);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // release in[curr]

        // scalar unit fetches position/slot_mapping from UB
        uint64_t ipos = ub_pos[irow - ub_pos_start];
        auto *freqs_gm = (__gm__ float *)(freqs) + ipos * rope_dim;
        __gm__ Dtype *output_norm_gm = input_gm;
        if (needs_cache) {
            output_norm_gm =
                (__gm__ Dtype *)(kcache) + ub_slot_mapping[irow - ub_pos_start] * norm_dim;
        }

        // ub_freqs[curr]: GM -> UB
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);  // acquire ub_freqs[curr]
        copy_gm_to_ubuf(ub_freqs[curr], freqs_gm, 0, 1, rope_blocks, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
        // freqs: separate into sin/cos
        SetMask(rope_dim);
        // [cos(k/(theta^(2t/dim))]
        vreducev2((UBA(uint32_t))cos_float, (UBA(uint32_t))ub_freqs[curr], nullptr, rope_repeat, 1,
                  1, 8, 0);
        // [sin(k/(theta^(2t/dim))]
        vreducev2((UBA(uint32_t))sin_float, (UBA(uint32_t))ub_freqs[curr], nullptr, rope_repeat, 1,
                  2, 8, 0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);  // release ub_freqs[curr]

        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        // BEGIN: muls
        bool need_muls = scale_repeat > 0 && ipos > top_k;
        if (need_muls) {
            // scale: multiply by scale factor
            vmuls(out_float + norm_dim, in_float + norm_dim, scale, scale_repeat, 1, 1, 8, 8);
        }
        // DONE: muls

        // BEGIN: norm
        // x / n
        vmuls(calc_float, in_float, inv_norm_dim, norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // mean = sum (x / n)
        reduce_sum(calc_float, 1, norm_dim);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        // duplicate mean
        duplicate_item(calc_float, 1, norm_repeat, norm_dim);
        pipe_barrier(PIPE_V);
        // x - mean
        vsub(out_float, in_float, calc_float, norm_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // (x - mean)^2
        vmul(calc_float, out_float, out_float, norm_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // (x - mean)^2 / n
        vmuls(in_float, calc_float, inv_norm_dim, norm_repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // sum ((x - mean)^2 / n)
        reduce_sum(in_float, 1, norm_dim);
        pipe_barrier(PIPE_V);
        // sum ((x - mean)^2 / n) + eps
        SetMask(1);
        vadds(in_float, in_float, norm_eps, 1, 1, 1, 0, 0);
        pipe_barrier(PIPE_V);
        // sqrt(sum ((x - mean)^2 / n) + eps)
        vsqrt(in_float, in_float, 1, 1, 1, 0, 0);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        // duplicate sqrt(sum ((x - mean)^2 / n) + eps)
        duplicate_item(in_float, 1, norm_repeat, norm_dim);
        pipe_barrier(PIPE_V);
        // (x - mean) / sqrt(sum ((x - mean)^2 / n) + eps)
        vdiv(calc_float, out_float, in_float, norm_repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // (x - mean) / sqrt(sum ((x - mean)^2 / n) + eps) * weight
        if (weight) {
            vmul(bias ? in_float : out_float, calc_float, ub_weight, norm_repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        // (x - mean) / sqrt(sum ((x - mean)^2 / n) + eps) * weight + bias
        if (bias) {
            vadd(out_float, weight ? in_float : calc_float, ub_bias, norm_repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        // END: norm

        // BEGIN: rope
        // DSA indexer K layout is [rope | nope]: RoPE applies to the FIRST rope_dim elements
        SetMask(rope_dim);  // rope_fim must not be 0
        // [x0, x2, x4, ...]
        vreducev2((UBA(uint32_t))in_even, (UBA(uint32_t))out_float, nullptr, rope_repeat, 1, 1, 8,
                  0);
        // [x1, x3, x5, ...]
        vreducev2((UBA(uint32_t))in_odd, (UBA(uint32_t))out_float, nullptr, rope_repeat, 1, 2, 8,
                  0);
        pipe_barrier(PIPE_V);
        SetMask(rope_dim / 2);
        // x[0::2] * cos
        vmul(calc_even_cos, cos_float, in_even, 1, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        // x[0::2] * sin
        vmul(calc_even_sin, sin_float, in_even, 1, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        // x[1::2] * sin
        vmul(calc_odd_sin, sin_float, in_odd, 1, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        // x[1::2] * cos
        vmul(calc_odd_cos, cos_float, in_odd, 1, 1, 1, 1, half_rope_blocks, half_rope_blocks, 0);
        pipe_barrier(PIPE_V);
        // real : x[0::2] * cos - x[1::2] * sin
        vsub(out_float, calc_even_cos, calc_odd_sin, 1, 1, 1, 1, rope_blocks, half_rope_blocks,
             half_rope_blocks);
        // img : x[0::2] * sin + x[1::2] * cos
        vadd(out_float + rope_dim / 2, calc_even_sin, calc_odd_cos, 1, 1, 1, 1, rope_blocks,
             half_rope_blocks, half_rope_blocks);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        // out_float -> out[curr] ith row
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // acquire out[curr]
        convert_output(out[curr], out_float, calc_repeat);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
        // copy out: UB -> GM
        if (needs_cache) {
            CopyUbufToGmAligned(output_norm_gm, out[curr], norm_size);
            if (need_muls) {
                CopyUbufToGmAligned(input_gm + norm_dim, out[curr] + norm_dim, scale_size);
            }
        } else {
            copy_ubuf_to_gm(input_gm, out[curr], 0, 1, len_burst_calc, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // release out[curr]
        curr = 1 - curr;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // copy in -> in
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);  // copy in -> in
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);  // copy in -> freqs
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);  // copy in -> freqs
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // copy out <- out
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);  // copy out <- out

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_ALL);
}

template <typename Dtype>
__aicore__ inline void indexer_prepare(GM_ADDR kw, GM_ADDR kNorm, GM_ADDR kNormBias, GM_ADDR freqs,
                                       GM_ADDR position, uint32_t token_num,
                                       uint32_t index_head_dim, uint32_t index_n_heads,
                                       uint32_t rope_head_dim, uint32_t block_size, float norm_eps,
                                       GM_ADDR index_k_cache = nullptr,
                                       GM_ADDR slot_mapping = nullptr, GM_ADDR q = nullptr,
                                       float scale = 1.0f, uint32_t top_k = 2048,
                                       bool is_long = false)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t total_dim = index_head_dim + index_n_heads;
    int core_offset = 0;

    norm_ropex_cache_muls<Dtype>(kw, kNorm, kNormBias, position, norm_eps, freqs, token_num,
                                 total_dim, index_head_dim, rope_head_dim, index_k_cache,
                                 slot_mapping, block_size, is_long ? index_n_heads : 0, scale,
                                 top_k, core_offset, &core_offset);

    if (is_long) {
        rope_complex_and_cache<Dtype>(token_num, index_n_heads, index_head_dim, rope_head_dim, 0,
                                      rope_head_dim, q, q, index_head_dim, 0, freqs, position, 0,
                                      nullptr, nullptr, false, false, core_offset);
    }
}

#define INDEXER_PREPARE_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void indexer_prepare_##dtype(                                \
        GM_ADDR kw, GM_ADDR kNorm, GM_ADDR kNormBias, GM_ADDR freqs, GM_ADDR position,            \
        uint32_t token_num, uint32_t index_head_dim, uint32_t index_n_heads,                      \
        uint32_t rope_head_dim, uint32_t block_size, float norm_eps, GM_ADDR index_k_cache,       \
        GM_ADDR slot_mapping, GM_ADDR q, float scale, uint32_t top_k, bool is_long)               \
    {                                                                                             \
        indexer_prepare<dtype>(kw, kNorm, kNormBias, freqs, position, token_num, index_head_dim,  \
                               index_n_heads, rope_head_dim, block_size, norm_eps, index_k_cache, \
                               slot_mapping, q, scale, top_k, is_long);                           \
    }
#else
#define INDEXER_PREPARE_FUNC_DEFINE(dtype)                                                  \
    extern "C" __global__ __aicore__ void indexer_prepare_##dtype(                          \
        GM_ADDR kw, GM_ADDR kNorm, GM_ADDR kNormBias, GM_ADDR freqs, GM_ADDR position,      \
        uint32_t token_num, uint32_t index_head_dim, uint32_t index_n_heads,                \
        uint32_t rope_head_dim, uint32_t block_size, float norm_eps, GM_ADDR index_k_cache, \
        GM_ADDR slot_mapping, GM_ADDR q, float scale, uint32_t top_k, bool is_long)         \
    {                                                                                       \
    }
#endif
