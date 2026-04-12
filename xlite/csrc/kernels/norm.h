/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

template <typename Dtype>
__aicore__ inline void norm(GM_ADDR input, GM_ADDR addInOut, GM_ADDR weight, GM_ADDR bias,
                            GM_ADDR output, uint32_t token_num, uint32_t norm_dim, float norm_eps,
                            bool mean, uint32_t cnt_per_token, uint32_t in_step, uint32_t out_step,
                            uint32_t in_start_offset, uint32_t out_start_offset)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    bool has_addInOut = (addInOut != nullptr);
    float n = (float)1.0 / norm_dim;
    uint32_t total_dim = norm_dim * cnt_per_token;
    uint64_t len_burst = DIV_ROUND_UP(total_dim * sizeof(Dtype), BLOCK_SIZE);
    uint64_t len_burst_per_norm = DIV_ROUND_UP(norm_dim * sizeof(Dtype), BLOCK_SIZE);
    uint32_t inout_blocksize = ROUND_UP(total_dim * sizeof(Dtype), BLOCK_SIZE);
    uint32_t calc_blocksize = ROUND_UP(total_dim * sizeof(float), BLOCK_SIZE);
    int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    uint64_t repeat = DIV_ROUND_UP(total_dim, calcPad);
    uint64_t repeat_per_norm = DIV_ROUND_UP(norm_dim, calcPad);
    uint64_t len_burst_float_per_norm = DIV_ROUND_UP(norm_dim * sizeof(float), BLOCK_SIZE);
    uint64_t repeat_stride = DIV_ROUND_UP(norm_dim * sizeof(float), BLOCK_SIZE);
    float dupnum;

    uint64_t off = 0;
    __ubuf__ Dtype *in0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += inout_blocksize;
    __ubuf__ Dtype *in1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += inout_blocksize;
    __ubuf__ Dtype *in[2] = {in0, in1};  // event 0/1
    __ubuf__ Dtype *out0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += inout_blocksize;
    __ubuf__ Dtype *out1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += inout_blocksize;
    __ubuf__ Dtype *out[2] = {out0, out1};  // event 0/1

    __ubuf__ float *calc0 = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;
    __ubuf__ float *calc1 = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;
    __ubuf__ float *weight_calc = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;
    __ubuf__ float *bias_calc = reinterpret_cast<__ubuf__ float *>(off);
    off += calc_blocksize;

    int inCurr = 0;
    int outCurr = 0;
    copy_gm_to_ubuf(in[inCurr], (__gm__ Dtype *)weight, 0, 1, len_burst_per_norm, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
    if constexpr (std::is_same_v<Dtype, float16_t>) {
        vconv_f162f32(weight_calc, in[inCurr], repeat_per_norm, 1, 1, 8, 4);
    } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
        vconv_bf162f32(weight_calc, in[inCurr], repeat_per_norm, 1, 1, 8, 4);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    inCurr = 1 - inCurr;

    if (bias != nullptr) {
        copy_gm_to_ubuf(in[inCurr], (__gm__ Dtype *)bias, 0, 1, len_burst_per_norm, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(bias_calc, in[inCurr], repeat_per_norm, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(bias_calc, in[inCurr], repeat_per_norm, 1, 1, 8, 4);
        }
        inCurr = 1 - inCurr;
    }
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    for (uint32_t norm_idx = 1; norm_idx < cnt_per_token; norm_idx++) {
        auto weight_norm = weight_calc + norm_idx * norm_dim;
        copy_ubuf_to_ubuf(weight_norm, weight_calc, 0, 1, len_burst_float_per_norm, 0, 0);
        if (bias != nullptr) {
            auto bias_norm = bias_calc + norm_idx * norm_dim;
            copy_ubuf_to_ubuf(bias_norm, bias_calc, 0, 1, len_burst_float_per_norm, 0, 0);
        }
        if (norm_idx == cnt_per_token - 1) {
            pipe_barrier(PIPE_V);
        }
    }

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    for (uint32_t loop = block_idx; loop < token_num; loop += block_num) {
        uint32_t in_offset = in_start_offset + loop * in_step;
        uint32_t out_offset = out_start_offset + loop * out_step;
        auto gm_in = (__gm__ Dtype *)input + in_offset;
        auto gm_out = (__gm__ Dtype *)output + out_offset;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
        copy_gm_to_ubuf(in[inCurr], gm_in, 0, 1, len_burst, 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);

        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(calc0, in[inCurr], repeat, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(calc0, in[inCurr], repeat, 1, 1, 8, 4);
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
        inCurr = 1 - inCurr;

        if (has_addInOut) {
            auto gm_addInOut = (__gm__ Dtype *)addInOut + in_offset;
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
            copy_gm_to_ubuf(in[inCurr], gm_addInOut, 0, 1, len_burst, 0, 0);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + inCurr);

            // cast data type && input = input + addInOut
            if constexpr (std::is_same_v<Dtype, float16_t>) {
                vconv_f162f32(calc1, in[inCurr], repeat, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
                inCurr = 1 - inCurr;
                vadd(calc0, calc1, calc0, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
                vconv_f322f16(out[outCurr], calc0, repeat, 1, 1, 4, 8);
            } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
                vconv_bf162f32(calc1, in[inCurr], repeat, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + inCurr);
                inCurr = 1 - inCurr;
                vadd(calc0, calc1, calc0, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
                vconv_f322bf16r(out[outCurr], calc0, repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);

            copy_ubuf_to_gm(gm_addInOut, out[outCurr], 0, 1, len_burst, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
            outCurr = 1 - outCurr;
        }

        if (mean) {
            // x / n
            vmuls(calc1, calc0, n, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // mean = sum(x / n)
            if (norm_dim == 128) {
                vadd(calc1, calc1, calc1 + 64, cnt_per_token, 1, 1, 1, 16, 16, 16);
                pipe_barrier(PIPE_V);
                for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
                    auto calc_norm = calc1 + norm_idx * norm_dim;
                    vcadd(calc_norm, calc_norm, 1, 1, 1, 8, 0);
                }
            } else {
                for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
                    auto calc_norm = calc1 + norm_idx * norm_dim;
                    ReduceSum(calc_norm, calc_norm, norm_dim);
                }
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            // duplicate item
            for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
                auto calc_norm = calc1 + norm_idx * norm_dim;
                dupnum = *calc_norm;
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                vector_dup(calc_norm, dupnum, repeat_per_norm, 1, 1, 8, 1);
            }
            pipe_barrier(PIPE_V);

            // x - mean
            vsub(calc0, calc0, calc1, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        // x ^ 2
        vmul(calc1, calc0, calc0, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // x ^ 2 / n
        vmuls(calc1, calc1, n, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // sum(x ^ 2)
        if (norm_dim == 128) {
            vadd(calc1, calc1, calc1 + 64, cnt_per_token, 1, 1, 1, 16, 16, 16);
            pipe_barrier(PIPE_V);
            for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
                auto calc_norm = calc1 + norm_idx * norm_dim;
                vcadd(calc_norm, calc_norm, 1, 1, 1, 8, 0);
            }
        } else {
            for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
                auto calc_norm = calc1 + norm_idx * norm_dim;
                ReduceSum(calc_norm, calc_norm, norm_dim);
            }
        }
        pipe_barrier(PIPE_V);

        SetMask(1);
        // sum(x ^ 2) + eps
        vadds(calc1, calc1, norm_eps, cnt_per_token, 1, 1, repeat_stride, repeat_stride);
        pipe_barrier(PIPE_V);

        // sqrt(sum(x ^ 2) + eps)
        vsqrt(calc1, calc1, cnt_per_token, 1, 1, repeat_stride, repeat_stride);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        // duplicate item
        for (uint32_t norm_idx = 0; norm_idx < cnt_per_token; norm_idx++) {
            auto calc_norm = calc1 + norm_idx * norm_dim;
            dupnum = *calc_norm;
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            vector_dup(calc_norm, dupnum, repeat_per_norm, 1, 1, 8, 1);
        }
        pipe_barrier(PIPE_V);

        // x / sqrt(sum(x ^ 2) + eps)
        vdiv(calc1, calc0, calc1, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // x / sqrt(sum(x ^ 2) + eps) * weight
        vmul(calc1, weight_calc, calc1, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        if (bias != nullptr) {
            vadd(calc1, bias_calc, calc1, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }

        // cast data type
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16(out[outCurr], calc1, repeat, 1, 1, 4, 8);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(out[outCurr], calc1, repeat, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + outCurr);

        copy_ubuf_to_gm(gm_out, out[outCurr], 0, 1, len_burst, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + outCurr);
        outCurr = 1 - outCurr;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

#define NORM_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void norm_##dtype(                                           \
        GM_ADDR input, GM_ADDR addInOut, GM_ADDR weight, GM_ADDR bias, GM_ADDR out,               \
        uint32_t token_num, uint32_t norm_dim, float norm_eps, bool mean, uint32_t cnt_per_token, \
        uint32_t in_step, uint32_t out_step, uint32_t in_start_offset, uint32_t out_start_offset) \
    {                                                                                             \
        norm<dtype>(input, addInOut, weight, bias, out, token_num, norm_dim, norm_eps, mean,      \
                    cnt_per_token, in_step, out_step, in_start_offset, out_start_offset);         \
    }
#else
#define NORM_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void norm_##dtype(                                           \
        GM_ADDR input, GM_ADDR addInOut, GM_ADDR weight, GM_ADDR bias, GM_ADDR out,               \
        uint32_t token_num, uint32_t norm_dim, float norm_eps, uint32_t cnt_per_token,            \
        uint32_t in_step, uint32_t out_step, uint32_t in_start_offset, uint32_t out_start_offset) \
    {                                                                                             \
    }
#endif