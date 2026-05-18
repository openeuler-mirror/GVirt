/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#ifdef __DAV_C220_VEC__

static __aicore__ inline void DumpBufferIndex(__ubuf__ float *buf, const __gm__ char *name,
                                              int size, int step = 1)
{
    DumpBuffer(buf, name, size, step, 1, true);
}

template <typename Dtype>
class BetaDecay
{
public:
    __aicore__ inline BetaDecay()
    {
    }

    __aicore__ inline void Init(GM_ADDR b, GM_ADDR a, GM_ADDR A_log, GM_ADDR dt_bias, GM_ADDR beta,
                                GM_ADDR g, uint32_t bsz, uint32_t seqlen, uint32_t num_v_heads)
    {
        set_mask_norm();
        this->b = (__gm__ Dtype *)b;
        this->a = (__gm__ Dtype *)a;
        this->A_log = (__gm__ Dtype *)A_log;
        this->dt_bias = (__gm__ Dtype *)dt_bias;
        this->beta = (__gm__ Dtype *)beta;
        this->g = (__gm__ Dtype *)g;

        this->bsz = bsz;
        this->seqlen = seqlen;
        this->num_v_heads = num_v_heads;

        num = DIV_ROUND_UP(num_v_heads * sizeof(Dtype), 32);
        calcRepeat = DIV_ROUND_UP(num, 8);

        uint64_t off = 0;
        uint64_t stride = DIV_ROUND_UP(num_v_heads * sizeof(Dtype), 256) * 256;
        if constexpr (!(std::is_same<Dtype, float>::value)) {
            conv_b = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += stride;
            conv_a = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += stride;
            conv_A_log = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += stride;
            conv_dt_bias = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += stride;
        }
        stride = DIV_ROUND_UP(num_v_heads * sizeof(float), 256) * 256;
        b_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += stride;
        a_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += stride;
        A_log_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += stride;
        dt_bias_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += stride;
        beta_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += stride;
        g_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += stride;
        ones_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    }

    __aicore__ inline void Beta()
    {
        vector_dup(ones_buf, float(1), DIV_ROUND_UP(num_v_heads, 64), 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
        for (int i = 0; i < bsz * seqlen; i++) {
            if (i % GetBlockNum() != GetBlockIdx())
                continue;
            int bOffset = num_v_heads * i;
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            if constexpr (std::is_same<Dtype, float>::value) {
                copy_gm_to_ubuf(b_buf, b + bOffset, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE2);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            } else {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                copy_gm_to_ubuf(conv_b, b + bOffset, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE2);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f162f32(b_buf, conv_b, calcRepeat, 1, 1, 8, 8);
                } else {
                    vconv_bf162f32(b_buf, conv_b, calcRepeat, 1, 1, 8, 8);
                }
                pipe_barrier(PIPE_V);
            }

            // -x
            vmuls(b_buf, b_buf, (float)-1.0, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // e^-x
            vexp(b_buf, b_buf, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // 1 + e^-x
            vadds(b_buf, b_buf, (float)1.0, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // sigmoid(x) = 1 / (1 + e^-x)
            vdiv(b_buf, ones_buf, b_buf, calcRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            if constexpr (std::is_same<Dtype, float>::value) {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                copy_ubuf_to_gm(beta + bOffset, b_buf, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE3);
            } else {
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f322f16(conv_b, b_buf, calcRepeat, 1, 1, 8, 8);
                } else {
                    vconv_f322bf16r(conv_b, b_buf, calcRepeat, 1, 1, 8, 8);
                }
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                copy_ubuf_to_gm(beta + bOffset, conv_b, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE3);
            }
        }
    }

    __aicore__ inline void Decay()
    {
        // load A_log
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_gm_to_ubuf(A_log_buf, A_log, 0, num, 1, 0, 0);
            pipe_barrier(PIPE_MTE2);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        } else {
            copy_gm_to_ubuf(conv_A_log, A_log, 0, num, 1, 0, 0);
            pipe_barrier(PIPE_MTE2);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(A_log_buf, conv_A_log, calcRepeat, 1, 1, 8, 8);
            } else {
                vconv_bf162f32(A_log_buf, conv_A_log, calcRepeat, 1, 1, 8, 8);
            }
            pipe_barrier(PIPE_V);
        }
        // e^x
        vexp(A_log_buf, A_log_buf, calcRepeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // -(e^x)
        vmuls(A_log_buf, A_log_buf, (float)-1.0, calcRepeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // load dt_bias
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_gm_to_ubuf(dt_bias_buf, dt_bias, 0, num, 1, 0, 0);
            pipe_barrier(PIPE_MTE2);
        } else {
            copy_gm_to_ubuf(conv_dt_bias, dt_bias, 0, num, 1, 0, 0);
            pipe_barrier(PIPE_MTE2);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(dt_bias_buf, conv_dt_bias, calcRepeat, 1, 1, 8, 8);
            } else {
                vconv_bf162f32(dt_bias_buf, conv_dt_bias, calcRepeat, 1, 1, 8, 8);
            }
            pipe_barrier(PIPE_V);
        }
        // load a and calc decay
        for (int i = 0; i < bsz * seqlen; i++) {
            if (i % GetBlockNum() != GetBlockIdx())
                continue;
            int aOffset = num_v_heads * i;
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            if constexpr (std::is_same<Dtype, float>::value) {
                copy_gm_to_ubuf(a_buf, a + aOffset, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE2);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            } else {
                copy_gm_to_ubuf(conv_a, a + aOffset, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE2);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f162f32(a_buf, conv_a, calcRepeat, 1, 1, 8, 8);
                } else {
                    vconv_bf162f32(a_buf, conv_a, calcRepeat, 1, 1, 8, 8);
                }
                pipe_barrier(PIPE_V);
            }

            // x = a + dt_bias
            vadd(a_buf, a_buf, dt_bias_buf, calcRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            // e^x
            vexp(a_buf, a_buf, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // 1 + e^x
            vadds(a_buf, a_buf, (float)1.0, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // softplus(x) = ln(1 + e^x)
            vln(a_buf, a_buf, calcRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // g = -e^A_log * softplus(x)
            vmul(a_buf, a_buf, A_log_buf, calcRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            if constexpr (std::is_same<Dtype, float>::value) {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                copy_ubuf_to_gm(g + aOffset, a_buf, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE3);
            } else {
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f322f16(conv_a, a_buf, calcRepeat, 1, 1, 8, 8);
                } else {
                    vconv_f322bf16r(conv_a, a_buf, calcRepeat, 1, 1, 8, 8);
                }
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                copy_ubuf_to_gm(g + aOffset, conv_a, 0, num, 1, 0, 0);
                pipe_barrier(PIPE_MTE3);
            }
        }
    }

private:
    __gm__ Dtype *b;
    __gm__ Dtype *a;
    __gm__ Dtype *A_log;
    __gm__ Dtype *dt_bias;
    __gm__ Dtype *beta;
    __gm__ Dtype *g;

    __ubuf__ Dtype *conv_b;
    __ubuf__ Dtype *conv_a;
    __ubuf__ Dtype *conv_A_log;
    __ubuf__ Dtype *conv_dt_bias;
    __ubuf__ float *beta_buf;
    __ubuf__ float *g_buf;
    __ubuf__ float *b_buf;
    __ubuf__ float *a_buf;
    __ubuf__ float *A_log_buf;
    __ubuf__ float *dt_bias_buf;
    __ubuf__ float *ones_buf;

    uint32_t bsz;
    uint32_t seqlen;
    uint32_t num_v_heads;

    int num;
    int calcRepeat;
};

#define BETA_DECAY_FUNC_DEFINE(dtype)                                                  \
    extern "C" __global__ __aicore__ void beta_decay_##dtype(                          \
        GM_ADDR b, GM_ADDR a, GM_ADDR A_log, GM_ADDR dt_bias, GM_ADDR beta, GM_ADDR g, \
        uint32_t bsz, uint32_t seqlen, uint32_t num_v_heads)                           \
    {                                                                                  \
        BetaDecay<dtype> op;                                                           \
        op.Init(b, a, A_log, dt_bias, beta, g, bsz, seqlen, num_v_heads);              \
        op.Beta();                                                                     \
        op.Decay();                                                                    \
    }
#else
#define BETA_DECAY_FUNC_DEFINE(dtype)                                                  \
    extern "C" __global__ __aicore__ void beta_decay_##dtype(                          \
        GM_ADDR b, GM_ADDR a, GM_ADDR A_log, GM_ADDR dt_bias, GM_ADDR beta, GM_ADDR g, \
        uint32_t bsz, uint32_t seqlen, uint32_t num_v_heads)                           \
    {                                                                                  \
    }
#endif