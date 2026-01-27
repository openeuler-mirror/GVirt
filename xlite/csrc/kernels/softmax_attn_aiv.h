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
inline __aicore__ void RunAivSoftmaxOneLoop(__gm__ Dtype *buf, uint32_t m, uint32_t n, uint32_t calcLen)
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
    int calInstPad = VECTOR_MAX_REPEAT * calPad;
    int instPad = VECTOR_MAX_REPEAT * pad;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    for (int idx = 0; idx < m; ++idx) {
        int actualCalcLen = calcLen + idx; // 每一行开始mask的位置
        if (actualCalcLen > n) {
            actualCalcLen = n;
        }
        int padCachedTokens = ROUND_UP(actualCalcLen, calPad);
        int repeat = padCachedTokens / calPad;
        int instNum = DIV_ROUND_UP(repeat, VECTOR_MAX_REPEAT);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        copy_gm_to_ubuf(in, buf + idx * n, 0, 1,
                        DIV_ROUND_UP(actualCalcLen * sizeof(Dtype), 32), 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if constexpr (std::is_same<CalcDtype, Dtype>::value) {
            ptr = in;
        } else {
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                if constexpr (std::is_same<Dtype, half>::value) {
                    vconv_f162f32(cal + i * calInstPad, in + i * calInstPad, currRepeat, 1, 1, 8, 4);
                } else {
                    vconv_bf162f32(cal + i * calInstPad, in + i * calInstPad, currRepeat, 1, 1, 8, 4);
                }
            }
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
        for (int i = 0; i < instNum; i++) {
            int currRepeat = VECTOR_MAX_REPEAT;
            if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                currRepeat = repeat - i * VECTOR_MAX_REPEAT;
            }
            vsub(cal + i * calInstPad, ptr + i * calInstPad, temp, currRepeat, 1, 1, 0, 8, 8, 0);
        }
        pipe_barrier(PIPE_V);
        if constexpr (std::is_same<CalcDtype, half>::value) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        }

        // EXP = exp(QK-max)
        for (int i = 0; i < instNum; i++) {
            int currRepeat = VECTOR_MAX_REPEAT;
            if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                currRepeat = repeat - i * VECTOR_MAX_REPEAT;
            }
            vexp(cal + i * calInstPad, cal + i * calInstPad, currRepeat, 1, 1, 8, 8);
        }
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
        if constexpr (std::is_same<CalcDtype, Dtype>::value) {
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                vdiv(out + i * instPad, cal + i * calInstPad, temp, currRepeat, 1, 1, 0, 8, 8, 0);
            }
            pipe_barrier(PIPE_V);
        } else {
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                vdiv(cal + i * calInstPad, cal + i * calInstPad, temp, currRepeat, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
                if constexpr (std::is_same<Dtype, half>::value) {
                    vconv_f322f16r(out + i * calInstPad, cal + i * calInstPad, currRepeat, 1, 1, 4, 8);
                } else {
                    vconv_f322bf16r(out + i * calInstPad, cal + i * calInstPad, currRepeat, 1, 1, 4, 8);
                }
                pipe_barrier(PIPE_V);
            }
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
                vector_dup(out + last, Dtype(0), DIV_ROUND_UP(n - last, pad), 1, 1, 8, 0);
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

/*
 * (VECTOR_MAX_REPEAT - 1) * VECTOR_MAX_BYTESIZE * 2 + VECTOR_MAX_BYTESIZE * DIV_ROUND_UP(VECTOR_MAX_REPEAT - 1, calcPad) + VECTOR_MAX_BYTESIZE * (subBlockNum + 1) * 2 < UB SIZE(196608)
 * float16_t: subBlockNum <= 128
 * float16_t: n <= 4161536
 * bfloat16_t: subBlockNum <= 127
 * bfloat16_t: n <= 2064512
 */
template<typename Dtype, typename CalcDtype>
inline __aicore__ void RunAivSoftmaxLong(__gm__ Dtype *buf, int32_t m, uint32_t n, uint32_t calcLen)
{
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    // VECTOR_MAX_REPEAT - 1 for Dtype size 2, CalcDtype size 4
    const uint32_t MAX_SUB_CONTEXT_SIZE = (VECTOR_MAX_REPEAT - 1) * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
    uint32_t totalSubBlockNum = DIV_ROUND_UP(n, MAX_SUB_CONTEXT_SIZE);
    int calcPad = VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
    int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);

    uint32_t off = 0;
    auto *in = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t) off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(Dtype);
    auto *out = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t) off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(Dtype);
    __ubuf__ CalcDtype *calc;
    if constexpr (!std::is_same<Dtype, CalcDtype>::value) {
        calc = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t) off);
        off += MAX_SUB_CONTEXT_SIZE * sizeof(CalcDtype);
    } else {
        calc = in;
    }
    __ubuf__ CalcDtype *calcDst = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t) off);
    off += DIV_ROUND_UP(VECTOR_MAX_REPEAT - 1, calcPad) * VECTOR_MAX_BYTESIZE;
    auto *max = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t) off);
    off += (totalSubBlockNum + 1) * VECTOR_MAX_BYTESIZE;
    auto *sum = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t) off);
    off += (totalSubBlockNum + 1) * VECTOR_MAX_BYTESIZE;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    for (int idx = 0; idx < m; idx++) {
        int actualCalcLen = calcLen + idx; // 每一行开始mask的位置
        if (actualCalcLen > n) {
            actualCalcLen = n;
        }
        uint64_t calcOutLen = ROUND_UP(actualCalcLen, MAX_SUB_CONTEXT_SIZE);
        if (calcOutLen > n) {
            calcOutLen = n;
        }

        // stage1: max(x) & sum(exp(x - max(x)))
        CalcDtype min;
        if constexpr (std::is_same<CalcDtype, float>::value) {
            min = -3.4028235e+38;
        } else {
            min = CalcDtype(-65504);
        }
        uint32_t subBlockNum = DIV_ROUND_UP(actualCalcLen, MAX_SUB_CONTEXT_SIZE);
        auto *totalMax = max + subBlockNum * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
        auto *totalSum = sum + subBlockNum * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
        __gm__ Dtype *curBuf = buf + idx * n;
        bool maxFlag[129] = {false};
        int block;
        for (block = 0; block < subBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            uint32_t curRepeat = VECTOR_MAX_REPEAT;
            if (block == subBlockNum - 1) {
                curLen = actualCalcLen - offset;
                curRepeat = DIV_ROUND_UP(curLen * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE);
            }
            auto *curMax = max + block * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
            auto *curSum = sum + block * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            if (curLen * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(in, curBuf + offset, 0, 1, DIV_ROUND_UP(curLen * sizeof(Dtype), BLOCK_SIZE), 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(in, curBuf + offset, 0, 1, curLen * sizeof(Dtype), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

            if constexpr (!std::is_same<CalcDtype, Dtype>::value) {
                if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                    vconv_bf162f32(calc, in, curRepeat, 1, 1, 8, 4);
                } else {
                    vconv_f162f32(calc, in, curRepeat, 1, 1, 8, 4);
                }
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }

            ReduceMax(calcDst, calc, curLen);
            pipe_barrier(PIPE_V);

            if constexpr (std::is_same<CalcDtype, float16_t>::value) {
                vbrcb((__ubuf__ uint16_t*)calcDst, (__ubuf__ uint16_t*)calcDst, 0, 0, 1);
            } else {
                vbrcb((__ubuf__ uint32_t*)calcDst, (__ubuf__ uint32_t*)calcDst, 0, 0, 1);
            }
            pipe_barrier(PIPE_V);

            if (block == 0) {                
                vector_dup(max, min, subBlockNum + 1, 1, 1, 8, 0);
                vector_dup(sum, CalcDtype(0), subBlockNum + 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
            }

            set_flag(PIPE_V, PIPE_S, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
            float curMaxVal = *calcDst;
            float totalMaxVal = *totalMax;
            if (curMaxVal > totalMaxVal) {
                for (int i = 0; i < block; i++) {
                    maxFlag[i] = 1;
                }
                copy_ubuf_to_ubuf(totalMax, calcDst, 0, 1, 8, 1, 1);
                pipe_barrier(PIPE_V);
            }
            copy_ubuf_to_ubuf(curMax, totalMax, 0, 1, 8, 1, 1);
            copy_ubuf_to_ubuf(calcDst, totalMax, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);

            vsub(calc, calc, calcDst, curRepeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same<Dtype, CalcDtype>::value) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                vexp(out, calc, curRepeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                ReduceSum(calcDst, out, curLen);
                pipe_barrier(PIPE_V);
            } else {
                vexp(calc, calc, curRepeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                ReduceSum(calcDst, calc, curLen);
                pipe_barrier(PIPE_V);
            }

            if constexpr (std::is_same_v<CalcDtype, half>) {
                vbrcb((__ubuf__ uint16_t*)calcDst, (__ubuf__ uint16_t*)calcDst, 0, 0, 1);
            } else {
                vbrcb((__ubuf__ uint32_t*)calcDst, (__ubuf__ uint32_t*)calcDst, 0, 0, 1);
            }
            pipe_barrier(PIPE_V);

            if (block > 0) {
                auto *lastMax = max + (block - 1) * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
                auto *lastSum = sum + (block - 1) * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
                if (curMaxVal > totalMaxVal) {
                    auto *calcTemp = calcDst + VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
                    vsub(calcTemp, lastMax, totalMax, 1, 1, 1, 1, 8, 8, 1);
                    pipe_barrier(PIPE_V);
                    vexp(calcTemp, calcTemp, 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    vmul(calcTemp, lastSum, calcTemp, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    vadd(calcDst, calcDst, calcTemp, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                } else {
                    vadd(calcDst, calcDst, lastSum, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
            }
            copy_ubuf_to_ubuf(curSum, calcDst, 0, 1, 8, 1, 1);
            copy_ubuf_to_ubuf(totalSum, calcDst, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);

            if constexpr (!std::is_same<CalcDtype, Dtype>::value) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                    vconv_f322bf16r(out, calc, curRepeat, 1, 1, 4, 8);
                } else {
                    vconv_f322f16r(out, calc, curRepeat, 1, 1, 4, 8);
                }
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            }

            if (curLen * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(curBuf + offset, out, 0, 1, DIV_ROUND_UP(curLen * sizeof(Dtype), BLOCK_SIZE), 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        
        // stage2: exp(x - max(x)) / sum(exp(x - max(x)))
        for (block = 0; block < subBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            uint32_t curRepeat = VECTOR_MAX_REPEAT;
            if (block == subBlockNum - 1) {
                curLen = actualCalcLen - offset;
                curRepeat = DIV_ROUND_UP(curLen * sizeof(CalcDtype), VECTOR_MAX_BYTESIZE);
            }
            auto *curMax = max + block * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
            auto *curSum = sum + block * VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            if (curLen * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(in, curBuf + offset, 0, 1, DIV_ROUND_UP(curLen * sizeof(Dtype), BLOCK_SIZE), 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(in, curBuf + offset, 0, 1, curLen * sizeof(Dtype), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

            if constexpr (!std::is_same<CalcDtype, Dtype>::value) {
                if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                    vconv_bf162f32(calc, in, curRepeat, 1, 1, 8, 4);
                } else {
                    vconv_f162f32(calc, in, curRepeat, 1, 1, 8, 4);
                }
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }

            uint32_t remain = curLen % calcPad;
            if (remain) {
                uint32_t start = ROUND_DOWN(curLen, calcPad);
                SetMaskFromHighBit(calcPad, calcPad - remain);
                vector_dup(calc + start, min, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }

            if (maxFlag[block]) {
                auto *calcTemp = calcDst + VECTOR_MAX_BYTESIZE / sizeof(CalcDtype);
                vsub(calcTemp, curMax, totalMax, 1, 1, 1, 1, 8, 8, 1);
                pipe_barrier(PIPE_V);
                vexp(calcTemp, calcTemp, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vmul(calc, calc, calcTemp, curRepeat, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
            }

            if constexpr (!std::is_same<CalcDtype, Dtype>::value) {
                vdiv(calc, calc, totalSum, curRepeat, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                    vconv_f322bf16r(out, calc, curRepeat, 1, 1, 4, 8);
                } else {
                    vconv_f322f16r(out, calc, curRepeat, 1, 1, 4, 8);
                }
                pipe_barrier(PIPE_V);
            } else {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                vdiv(out, calc, totalSum, curRepeat, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }

            if (block == subBlockNum - 1 && calcOutLen > actualCalcLen) {
                if (curLen % pad != 0) {
                    SetMaskFromHighBit(pad, pad - curLen % pad);
                    vector_dup(out + ROUND_DOWN(curLen, pad), Dtype(0), 1, 1, 1, 8, 0);
                    pipe_barrier(PIPE_V);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                int last = ROUND_UP(curLen, pad);
                if (calcOutLen - offset > last) {
                    vector_dup(out + last, Dtype(0), DIV_ROUND_UP(calcOutLen - offset - last, pad), 1, 1, 8, 0);
                    pipe_barrier(PIPE_V);
                }
                curLen = calcOutLen - offset;
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            if ((curLen * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(curBuf + offset, out, 0, 1, DIV_ROUND_UP(curLen * sizeof(Dtype), BLOCK_SIZE), 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        if (block < totalSubBlockNum) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            vector_dup(out, Dtype(0), DIV_ROUND_UP(MAX_SUB_CONTEXT_SIZE, pad), 1, 1, 8, 0);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        }

        for (; block < totalSubBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            if (block == totalSubBlockNum - 1) {
                curLen = n - offset;
            }
            if ((curLen * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(curBuf + offset, out, 0, 1, DIV_ROUND_UP(curLen * sizeof(Dtype), BLOCK_SIZE), 0, 0);
            } else {
                copy_ubuf_to_gm_align_b16(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype), 0, 0, 0, 0);
            }
            if (block == totalSubBlockNum - 1) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

template<typename Dtype, typename CalcDtype>
inline __aicore__ void RunAivSoftmax(__gm__ Dtype *buf, uint32_t m, uint32_t n, uint32_t calcLen)
{
    uint64_t maxNOneLoop = 0;
    if constexpr (std::is_same<Dtype, float16_t>::value) {
        maxNOneLoop = 32640;
    } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
        maxNOneLoop = 24320;
    }
    if (n <= maxNOneLoop) {
        RunAivSoftmaxOneLoop<Dtype, CalcDtype>(buf, m, n, calcLen);
    } else {
        RunAivSoftmaxLong<Dtype, CalcDtype>(buf, m, n, calcLen);
    }
}

#else
template<typename Dtype, typename CalcDtype>
inline __aicore__ void RunAivSoftmaxOneLoop(__gm__ Dtype *buf, uint32_t m, uint32_t n, uint32_t calcLen)
{
}
template<typename Dtype, typename CalcDtype>
inline __aicore__ void RunAivSoftmaxLong(__gm__ Dtype *buf, uint32_t m, uint32_t n, uint32_t calcLen)
{
}
#endif
#endif