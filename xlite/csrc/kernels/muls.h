/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ void muls(__gm__ Dtype *input, float scale, __gm__ Dtype *output, uint64_t numel)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    constexpr int n = VECTOR_MAX_REPEAT * calcPad;

    uint64_t off = 0;
    __ubuf__ Dtype *in1 = (__ubuf__ Dtype *)off;
    off += n * sizeof(Dtype);
    __ubuf__ Dtype *in2 = (__ubuf__ Dtype *)off;
    off += n * sizeof(Dtype);
    __ubuf__ Dtype *in[2] = {in1, in2};

    __ubuf__ float *calc = (__ubuf__ float *)off;
    off += n * sizeof(float);

    __ubuf__ Dtype *out1 = (__ubuf__ Dtype *)off;
    off += n * sizeof(Dtype);
    __ubuf__ Dtype *out2 = (__ubuf__ Dtype *)off;
    off += n * sizeof(Dtype);
    __ubuf__ Dtype *out[2] = {out1, out2};

    int numelPerCore = DIV_ROUND_UP(numel, block_num);
    int offset = block_idx * numelPerCore;
    int numelCurrCore = numelPerCore;
    if (offset + numelCurrCore > numel) {
        numelCurrCore = numel - offset;
    }

    int curr = 0;
    int num = n;
    int repeat = VECTOR_MAX_REPEAT;
    int loop = DIV_ROUND_UP(numelCurrCore, n);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    for (int i = 0; i < loop; i++) {
        int off = i * n;
        if (off + n > numelCurrCore) {
            num = numelCurrCore - off;
            repeat = DIV_ROUND_UP(num, calcPad);
        }

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        if (num * sizeof(Dtype) % BLOCK_SIZE == 0) {
            copy_gm_to_ubuf(in[curr], input + offset + off, 0, 1, num * sizeof(Dtype) / BLOCK_SIZE,
                            0, 0);
        } else {
            copy_gm_to_ubuf_align_b16(in[curr], input + offset + off, 0, 1, num * sizeof(Dtype), 0,
                                      0, 0, 0);
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            vconv_bf162f32(calc, in[curr], repeat, 1, 1, 8, 4);
        } else {
            vconv_f162f32(calc, in[curr], repeat, 1, 1, 8, 4);
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

        vmuls(calc, calc, scale, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
        if constexpr (std::is_same<Dtype, half>::value) {
            vconv_f322f16r(out[curr], calc, repeat, 1, 1, 4, 8);
        } else {
            vconv_f322bf16r(out[curr], calc, repeat, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        if (num * sizeof(Dtype) % BLOCK_SIZE == 0) {
            copy_ubuf_to_gm(output + offset + off, out[curr], 0, 1,
                            num * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
        } else {
            copy_ubuf_to_gm_align_b16(output + offset + off, out[curr], 0, 1, num * sizeof(Dtype),
                                      0, 0, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
        curr = 1 - curr;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

#define MULS_FUNC_DEFINE(dtype)                                                                    \
    extern "C" __global__ __aicore__ void muls_##dtype(GM_ADDR input, float scale, GM_ADDR output, \
                                                       uint32_t numel)                             \
    {                                                                                              \
        muls((__gm__ dtype *)input, scale, (__gm__ dtype *)output, numel);                         \
    }
#else
#define MULS_FUNC_DEFINE(dtype)                                                                    \
    extern "C" __global__ __aicore__ void muls_##dtype(GM_ADDR input, float scale, GM_ADDR output, \
                                                       uint32_t numel)                             \
    {                                                                                              \
    }
#endif
