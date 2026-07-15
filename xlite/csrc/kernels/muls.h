/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ void muls(__gm__ Dtype *input, float scale, __gm__ Dtype *output, uint32_t shape0,
                     uint32_t shape1, uint32_t calcOffset, uint32_t calcNum)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);

    int actualLen = calcNum * sizeof(Dtype);
    uint64_t len = ROUND_UP(actualLen, VECTOR_MAX_BYTESIZE);
    uint64_t off = 0;
    __ubuf__ Dtype *in1 = (__ubuf__ Dtype *)off;
    off += len;
    __ubuf__ Dtype *in2 = (__ubuf__ Dtype *)off;
    off += len;
    __ubuf__ Dtype *in[2] = {in1, in2};

    __ubuf__ float *calc = (__ubuf__ float *)off;
    if constexpr (std::is_same<Dtype, float>::value) {
        off += len;
    } else {
        off += ROUND_UP(actualLen * sizeof(float), VECTOR_MAX_BYTESIZE);
    }

    __ubuf__ Dtype *out1 = (__ubuf__ Dtype *)off;
    off += len;
    __ubuf__ Dtype *out2 = (__ubuf__ Dtype *)off;
    off += len;
    __ubuf__ Dtype *out[2] = {out1, out2};

    int repeat = DIV_ROUND_UP(calcNum, calcPad);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    int curr = 0;
    for (uint32_t process = block_idx; process < shape0; process += uint32_t(block_num)) {
        __gm__ Dtype *inPtr = input + process * shape1 + calcOffset;
        __gm__ Dtype *outPtr = output + process * shape1 + calcOffset;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(in[curr], inPtr, actualLen);

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

        CopyUbufToGmAligned(outPtr, out[curr], actualLen);
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
                                                       uint32_t shape0, uint32_t shape1,           \
                                                       uint32_t calcOffset, uint32_t calcLen)      \
    {                                                                                              \
        muls((__gm__ dtype *)input, scale, (__gm__ dtype *)output, shape0, shape1, calcOffset,     \
             calcLen);                                                                             \
    }
#else
#define MULS_FUNC_DEFINE(dtype)                                                                    \
    extern "C" __global__ __aicore__ void muls_##dtype(GM_ADDR input, float scale, GM_ADDR output, \
                                                       uint32_t shape0, uint32_t shape1,           \
                                                       uint32_t calcOffset, uint32_t calcLen)      \
    {                                                                                              \
    }
#endif
