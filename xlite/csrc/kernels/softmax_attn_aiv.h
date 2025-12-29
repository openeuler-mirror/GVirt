/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_SOFTMAX_ATTN_AIV_H_
#define _XLITE_SOFTMAX_ATTN_AIV_H_

#include "kernel_operator.h"
#include "kernel_macro.h"
using namespace AscendC;

#if __DAV_C220_VEC__
/* ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE) * 2 +
 *     ROUND_UP(n * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE) +
 *     DIV_ROUND_UP(n * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE) * sizeof(CalcDtype) <= ub size(196608B)
 * float16_t: n <= 32640
 * bfloat16_t: n <= 24320
 */
template<typename Dtype, typename CalcDtype>
inline __aicore__ void RunAivSoftmax(__gm__ Dtype *buf, uint32_t m, uint32_t n, uint32_t calcLen)
{
    CalcDtype min;
    if constexpr (std::is_same<CalcDtype, half>::value) {
        min = half(-65504);
    } else {
        min = -3.4028235e+38;
    }

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint64_t off = 0;
    __ubuf__ Dtype *in = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *out = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ CalcDtype *cal = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t)off);
    off += ROUND_UP(n * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ CalcDtype *temp = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t)off);
    __ubuf__ CalcDtype *ptr;

    int calPad = VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
    int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    for (int idx = 0; idx < m; ++idx) {
        int actualCalcLen = calcLen + idx; // 每一行开始mask的位置
        if (actualCalcLen > n) {
            actualCalcLen = n;
        }
        int padCachedTokens = ROUND_UP(actualCalcLen, calPad);
        int repeat = padCachedTokens / calPad;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        copy_gm_to_ubuf(in, buf + idx * n, 0, 1,
                        DIV_ROUND_UP(actualCalcLen * sizeof(Dtype), 32), 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if constexpr (std::is_same<CalcDtype, half>::value) {
            ptr = in;
        } else {
            vconv_bf162f32((__ubuf__ float *)cal, (__ubuf__ Dtype *)in, repeat, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            ptr = cal;
        }

        // dup calPad
        if (actualCalcLen % calPad != 0) {
            SetMaskFromHighBit(calPad, calPad - actualCalcLen % calPad);
            vector_dup(ptr + ROUND_DOWN(actualCalcLen, calPad), min, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }

        // max
        ReduceMax(temp, ptr, actualCalcLen);

        // broadcast一个max标量为一个block大小的向量，避免使用scalar运算
        if constexpr (std::is_same<CalcDtype, half>::value) {
            vbrcb((__ubuf__ uint16_t*)temp, (__ubuf__ uint16_t*)temp, 0, 0, 1);
        } else {
            vbrcb((__ubuf__ uint32_t*)temp, (__ubuf__ uint32_t*)temp, 0, 0, 1);
        }
        pipe_barrier(PIPE_V);

        // QK - max
        vsub(cal, ptr, temp, repeat, 1, 1, 0, 8, 8, 0);
        pipe_barrier(PIPE_V);
        if constexpr (std::is_same<CalcDtype, half>::value) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        }

        // EXP = exp(QK-max)
        vexp(cal, cal, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // s = Reduce_sum(EXP)
        ReduceSum(temp, cal, actualCalcLen);

        // broadcast一个s = Reduce_sum(EXP)标量为一个block大小的向量，避免使用scalar运算
        if constexpr (std::is_same<CalcDtype, half>::value) {
            vbrcb((__ubuf__ uint16_t*)temp, (__ubuf__ uint16_t*)temp, 0, 0, 1);
        } else {
            vbrcb((__ubuf__ uint32_t*)temp, (__ubuf__ uint32_t*)temp, 0, 0, 1);
        }
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        if constexpr (std::is_same<CalcDtype, half>::value) {
            vdiv(out, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
        } else {
            vdiv(cal, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vconv_f322bf16r(out, cal, repeat, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
        }

        if (n > actualCalcLen) {
            if (actualCalcLen % pad != 0) {
                SetMaskFromHighBit(pad, pad - actualCalcLen % pad);
                vector_dup(out + ROUND_DOWN(actualCalcLen, pad), Dtype(0), 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }
            int last = ROUND_UP(actualCalcLen, pad);
            if (n > last) {
                vector_dup(out + last, Dtype(0), (n - last) / pad, 1, 1, 8, 0);
            }
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if ((n * sizeof(Dtype)) % BLOCK_SIZE == 0) {
            copy_ubuf_to_gm(buf + idx * n, out, 0, 1, n * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
        } else {
            copy_ubuf_to_gm_align_b16(buf + idx * n, out, 0, 1, n * sizeof(Dtype), 0, 0, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    pipe_barrier(PIPE_ALL);
}
#endif
#endif