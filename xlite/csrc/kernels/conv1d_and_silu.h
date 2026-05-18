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
class XliteConv1dAndSiLU
{
public:
    __aicore__ inline XliteConv1dAndSiLU()
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch,
                                uint32_t channels, uint32_t seqLen, uint32_t kernelDim)
    {
        set_mask_norm();
        this->input = (__gm__ Dtype *)input;
        this->weight = (__gm__ Dtype *)weight;
        this->output = (__gm__ Dtype *)output;

        this->batch = batch;
        this->channels = channels;
        this->seqLen = seqLen;
        this->kernelDim = kernelDim;

        uint64_t off = 0;
        if constexpr (!(std::is_same<Dtype, float>::value)) {
            ubuf0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += 8 * 32;
            ubuf1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += 8 * 32;
            tmp_kernel_buf = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += 8 * 32;
            tmp_out_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
            off += 8 * 32;
        }

        ubuf2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        ubuf3 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        kernel_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        mul_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        calc_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        out_buf = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    }

    __aicore__ inline void SiLU()
    {
        // -x
        vmuls(calc_buf, mul_buf, (float)-1.0, 1, 0, 0, 0, 0);
        pipe_barrier(PIPE_V);

        // e^-x
        vexp(calc_buf, calc_buf, 1, 0, 0, 0, 0);
        pipe_barrier(PIPE_V);

        // 1 + e^-x
        vadds(calc_buf, calc_buf, (float)1.0, 1, 0, 0, 0, 0);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // silu = x / (1 + e^-x)
        if constexpr (std::is_same<Dtype, float>::value) {
            vdiv(out_buf, mul_buf, calc_buf, 1, 0, 0, 0, 0, 0, 0);
        } else {
            vdiv(calc_buf, mul_buf, calc_buf, 1, 0, 0, 0, 0, 0, 0);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(out_buf, calc_buf, 1, 0, 0, 0, 0);
            } else {
                vconv_f322bf16r(out_buf, calc_buf, 1, 0, 0, 0, 0);
            }
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }

    __aicore__ inline void Process()
    {
        set_vector_mask(0, 15);
        for (int channel = 0; channel < channels; channel++) {
            if (channel % GetBlockNum() != GetBlockIdx())
                continue;
            int wOffset = channel * kernelDim;

            if constexpr (std::is_same<Dtype, float>::value) {
                copy_gm_to_ubuf(kernel_buf, weight + wOffset, 0, 1, 1, 0, 0);
                pipe_barrier(PIPE_MTE2);
            } else {
                copy_gm_to_ubuf(tmp_kernel_buf, weight + wOffset, 0, 1, 1, 0, 0);
                pipe_barrier(PIPE_MTE2);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f162f32(kernel_buf, tmp_kernel_buf, 1, 0, 0, 0, 0);
                } else {
                    vconv_bf162f32(kernel_buf, tmp_kernel_buf, 1, 0, 0, 0, 0);
                }
                pipe_barrier(PIPE_V);
            }

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            for (int i = 0; i < seqLen; i++) {
                for (int batchIdx = 0, ping = 0; batchIdx < batch; batchIdx++) {
                    int inOffset = (batchIdx * channels + channel) * (seqLen + kernelDim) + i + 1;
                    qkv_buf = ping ? ubuf3 : ubuf2;
                    auto event_id = ping ? EVENT_ID1 : EVENT_ID0;

                    wait_flag(PIPE_V, PIPE_MTE2, event_id);
                    if constexpr (std::is_same<Dtype, float>::value) {
                        copy_gm_to_ubuf(qkv_buf, input + inOffset, 0, 1, 1, 0, 0);
                        pipe_barrier(PIPE_MTE2);
                        set_flag(PIPE_MTE2, PIPE_V, event_id);
                        wait_flag(PIPE_MTE2, PIPE_V, event_id);
                    } else {
                        tmp_qkv_buf = ping ? ubuf1 : ubuf0;
                        copy_gm_to_ubuf(tmp_qkv_buf, input + inOffset, 0, 1, 1, 0, 0);
                        pipe_barrier(PIPE_MTE2);
                        set_flag(PIPE_MTE2, PIPE_V, event_id);
                        wait_flag(PIPE_MTE2, PIPE_V, event_id);
                        if constexpr (std::is_same<Dtype, float16_t>::value) {
                            vconv_f162f32(qkv_buf, tmp_qkv_buf, 1, 0, 0, 0, 0);
                        } else {
                            vconv_bf162f32(qkv_buf, tmp_qkv_buf, 1, 0, 0, 0, 0);
                        }
                        pipe_barrier(PIPE_V);
                    }

                    vmul(mul_buf, qkv_buf, kernel_buf, 1, 0, 0, 0, 0, 0, 0);
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE2, event_id);

                    vcadd(mul_buf, mul_buf, 1, 0, 0, 0, 0);
                    pipe_barrier(PIPE_V);

                    SiLU();

                    DumpBuffer(kernel_buf, "kernel", 4);
                    DumpBuffer(qkv_buf, "qkv", 4);
                    DumpBuffer(mul_buf, "mul", 4);
                    DumpBuffer(out_buf, "out", 1);

                    int outOffset = (batchIdx * channels + channel) * seqLen + i;
                    copy_ubuf_to_gm(output + outOffset, out_buf, 0, 1, sizeof(Dtype), 0, 0,
                                    BM_ENABLE);
                    pipe_barrier(PIPE_MTE3);
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    ping = 1 - ping;
                }
            }
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }

private:
    __gm__ Dtype *input;
    __gm__ Dtype *weight;
    __gm__ Dtype *output;

    __ubuf__ Dtype *ubuf0;
    __ubuf__ Dtype *ubuf1;
    __ubuf__ Dtype *tmp_qkv_buf;
    __ubuf__ Dtype *tmp_kernel_buf;
    __ubuf__ float *tmp_out_buf;

    __ubuf__ float *ubuf2;
    __ubuf__ float *ubuf3;
    __ubuf__ float *qkv_buf;
    __ubuf__ float *kernel_buf;
    __ubuf__ float *mul_buf;
    __ubuf__ float *calc_buf;
    __ubuf__ Dtype *out_buf;

    uint32_t batch;
    uint32_t channels;
    uint32_t seqLen;
    uint32_t kernelDim;
};

#define CONV1D_AND_SILU_FUNC_DEFINE(dtype)                                                \
    extern "C" __global__ __aicore__ void conv1d_and_silu_##dtype(                        \
        GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch, uint32_t channels, \
        uint32_t seqLen, uint32_t kernelDim)                                              \
    {                                                                                     \
        XliteConv1dAndSiLU<dtype> op;                                                     \
        op.Init(input, weight, output, batch, channels, seqLen, kernelDim);               \
        op.Process();                                                                     \
    }
#else
#define CONV1D_AND_SILU_FUNC_DEFINE(dtype)                                                \
    extern "C" __global__ __aicore__ void conv1d_and_silu_##dtype(                        \
        GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch, uint32_t channels, \
        uint32_t seqLen, uint32_t kernelDim)                                              \
    {                                                                                     \
    }
#endif