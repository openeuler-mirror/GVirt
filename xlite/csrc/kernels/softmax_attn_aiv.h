/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_SOFTMAX_ATTN_AIV_H_
#define _XLITE_SOFTMAX_ATTN_AIV_H_

#include "kernel_operator.h"
#include "kernel_macro.h"
using namespace AscendC;

#if __DAV_C220_VEC__
/* ROUND_UP(n * sizeof(Dtype), VECTOR_MAX_BYTESIZE) * 4 +
 *     ROUND_UP(n * sizeof(float), VECTOR_MAX_BYTESIZE) +
 *     (DIV_ROUND_UP(n * sizeof(float), VECTOR_MAX_BYTESIZE) / 2) * VECTOR_MAX_BYTESIZE <= ub
 * size(196608B) float16_t or bfloat16_t: n <= 13952
 */
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxPingPong(__gm__ Dtype *buf, uint32_t m, uint32_t n,
                                             int calcLen, uint32_t outN = 0,
                                             uint32_t maskOff = 0, uint32_t maskStride = 1,
                                             __gm__ float *maxBuf = nullptr, __gm__ float *sumBuf = nullptr)
{
    if (outN == 0 || outN > n) {
        outN = n;
    }

    bool saveMaxSum = (maxBuf != nullptr && sumBuf != nullptr);

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint64_t off = 0;
    __ubuf__ Dtype *in1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *in2 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *in[PINGPONG_BUF_NUM] = {in1, in2};
    __ubuf__ Dtype *out1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *out2 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
    __ubuf__ Dtype *out[PINGPONG_BUF_NUM] = {out1, out2};
    __ubuf__ float *cal = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += ROUND_UP(outN * sizeof(float), VECTOR_MAX_BYTESIZE);
    __ubuf__ float *temp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += (DIV_ROUND_UP(outN * sizeof(float), VECTOR_MAX_BYTESIZE) / 2) * VECTOR_MAX_BYTESIZE;
    __ubuf__ float *maxOut0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *maxOut1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *maxOut[PINGPONG_BUF_NUM] = {maxOut0, maxOut1};
    __ubuf__ float *sumOut0 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *sumOut1 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += VECTOR_MAX_BYTESIZE;
    __ubuf__ float *sumOut[PINGPONG_BUF_NUM] = {sumOut0, sumOut1};

    constexpr int calPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    int curr = 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    for (int idx = 0; idx < m; ++idx) {
        int actualCalcLen = calcLen + (idx + maskOff) / maskStride;  // 每一行开始mask的位置
        if (actualCalcLen <= 0) {
            actualCalcLen = 0;
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            int last = ROUND_UP(actualCalcLen, pad);
            vector_dup(out[curr] + last, Dtype(0), DIV_ROUND_UP(outN - last, pad), 1, 1, 8, 0);
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            if ((outN * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_ubuf_to_gm_align_b16(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype), 0, 0, 0,
                                        0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        } else {
            if (actualCalcLen > outN) {
                actualCalcLen = outN;
            }
            int padCachedTokens = ROUND_UP(actualCalcLen, calPad);
            int repeat = padCachedTokens / calPad;

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);
            if (actualCalcLen * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(in[curr], buf + idx * n, 0, 1,
                                actualCalcLen * sizeof(Dtype) / BLOCK_SIZE, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b16(in[curr], buf + idx * n, 0, 1, actualCalcLen * sizeof(Dtype),
                                        0, 0, 0, 0);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f162f32(cal, in[curr], repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(cal, in[curr], repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);

            // max
            ReduceMaxV2(temp, cal, actualCalcLen);
            pipe_barrier(PIPE_V);

            // broadcast一个max标量为一个block大小的向量，避免使用scalar运算
            vbrcb((__ubuf__ uint32_t *)temp, (__ubuf__ uint32_t *)temp, 0, 0, 1);
            pipe_barrier(PIPE_V);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            if (saveMaxSum) {
                copy_ubuf_to_ubuf(maxOut[curr], temp, 0, 1, 1, 1, 1);
                pipe_barrier(PIPE_V);
            }

            // QK - max
            vsub(cal, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);

            // EXP = exp(QK-max)
            vexp(cal, cal, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // s = Reduce_sum(EXP)
            ReduceSumV2(temp, cal, actualCalcLen);
            pipe_barrier(PIPE_V);

            if (saveMaxSum) {
                copy_ubuf_to_ubuf(sumOut[curr], temp, 0, 1, 1, 1, 1);
                pipe_barrier(PIPE_V);
            }

            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f322f16r(out[curr], cal, repeat, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(out[curr], cal, repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);

            if (outN > actualCalcLen) {
                if (actualCalcLen % pad != 0) {
                    SetMaskFromHighBit(pad, pad - actualCalcLen % pad);
                    vector_dup(out[curr] + ROUND_DOWN(actualCalcLen, pad), Dtype(0), 1, 1, 1, 8, 0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                int last = ROUND_UP(actualCalcLen, pad);
                if (outN > last) {
                    vector_dup(out[curr] + last, Dtype(0), DIV_ROUND_UP(outN - last, pad), 1, 1, 8, 0);
                }
                pipe_barrier(PIPE_V);
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            if ((outN * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_ubuf_to_gm_align_b16(buf + idx * n, out[curr], 0, 1, outN * sizeof(Dtype), 0, 0, 0,
                                        0);
            }
            if (saveMaxSum) {
                copy_ubuf_to_gm_align_b16(maxBuf + idx, maxOut[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
                copy_ubuf_to_gm_align_b16(sumBuf + idx, sumOut[curr], 0, 1, sizeof(float), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        }
        curr = 1 - curr;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
}

/*
 * (VECTOR_MAX_REPEAT - 1) * VECTOR_MAX_BYTESIZE * 2 + VECTOR_MAX_BYTESIZE *
 * DIV_ROUND_UP(VECTOR_MAX_REPEAT - 1, calcPad) + VECTOR_MAX_BYTESIZE * (subBlockNum + 1) * 2 < UB
 * SIZE(196608) float16_t or bfloat16_t: subBlockNum <= 127 float16_t or bfloat16_t: n <= 2064512
 */
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxLong(__gm__ Dtype *buf, __gm__ float *expBuf, int32_t m,
                                         uint32_t n, uint32_t calcLen, uint32_t outN = 0,
                                         uint32_t maskOff = 0, uint32_t maskStride = 1)
{
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    float min = -3.4028235e+38;
    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);

    if (outN == 0 || outN > n) {
        outN = n;
    }

    // VECTOR_MAX_REPEAT - 1 for Dtype size 2, float size 4
    uint32_t maxRepeat = VECTOR_MAX_REPEAT - 1;
    const uint32_t MAX_SUB_CONTEXT_SIZE = maxRepeat * calcPad;
    uint32_t totalSubBlockNum = DIV_ROUND_UP(outN, MAX_SUB_CONTEXT_SIZE);

    uint32_t off = 0;
    auto *in = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(Dtype);
    auto *out = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(Dtype);
    __ubuf__ float *calc = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += MAX_SUB_CONTEXT_SIZE * sizeof(float);
    __ubuf__ float *calcDst = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    int useV2 = 0;
    if (totalSubBlockNum <= 65) {
        useV2 = 1;
        off += (VECTOR_MAX_REPEAT - 1) / 2 * VECTOR_MAX_BYTESIZE;
        ;
    } else {
        off += DIV_ROUND_UP(VECTOR_MAX_REPEAT - 1, calcPad) * VECTOR_MAX_BYTESIZE;
    }
    auto *max = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += (totalSubBlockNum + 1) * VECTOR_MAX_BYTESIZE;
    auto *sum = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
    off += (totalSubBlockNum + 1) * VECTOR_MAX_BYTESIZE;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // for MTE2 in
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);  // for MTE2 calc
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // for V calc
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);  // for V out
    for (int idx = 0; idx < m; idx++) {
        int actualCalcLen = calcLen + (idx + maskOff) / maskStride;  // 每一行开始mask的位置
        if (actualCalcLen > outN) {
            actualCalcLen = outN;
        }
        uint64_t calcOutLen = ROUND_UP(actualCalcLen, MAX_SUB_CONTEXT_SIZE);
        if (calcOutLen > outN) {
            calcOutLen = outN;
        }

        // stage1: max(x) & sum(exp(x - max(x)))
        uint32_t subBlockNum = DIV_ROUND_UP(actualCalcLen, MAX_SUB_CONTEXT_SIZE);
        auto *totalMax = max + subBlockNum * calcPad;
        auto *totalSum = sum + subBlockNum * calcPad;
        __gm__ Dtype *curBuf = buf + idx * n;
        bool maxFlag[128] = {false};
        int block;
        for (block = 0; block < subBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            uint32_t curRepeat = maxRepeat;
            if (block == subBlockNum - 1) {
                curLen = actualCalcLen - offset;
                curRepeat = DIV_ROUND_UP(curLen, calcPad);
            }
            float curMaxVal, totalMaxVal;

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            if (curLen * sizeof(Dtype) % BLOCK_SIZE == 0) {
                copy_gm_to_ubuf(in, curBuf + offset, 0, 1, curLen * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_gm_to_ubuf_align_b16(in, curBuf + offset, 0, 1, curLen * sizeof(Dtype), 0, 0,
                                          0, 0);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                vconv_bf162f32(calc, in, curRepeat, 1, 1, 8, 4);
            } else {
                vconv_f162f32(calc, in, curRepeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            if (useV2) {
                ReduceMaxV2(calcDst, calc, curLen);
                pipe_barrier(PIPE_V);
            } else {
                ReduceMax(calcDst, calc, curLen);
            }

            vbrcb((__ubuf__ uint32_t *)calcDst, (__ubuf__ uint32_t *)calcDst, 0, 0, 1);

            if (subBlockNum > 1 && block == 0) {
                vector_dup(max, min, subBlockNum + 1, 1, 1, 8, 0);
                vector_dup(sum, float(0), subBlockNum + 1, 1, 1, 8, 0);
            }
            pipe_barrier(PIPE_V);

            if (subBlockNum > 1) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID2);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
                curMaxVal = *calcDst;
                totalMaxVal = *totalMax;
                auto *curMax = max + block * calcPad;
                if (curMaxVal > totalMaxVal) {
                    for (int i = 0; i < block; i++) {
                        maxFlag[i] = 1;
                    }
                    copy_ubuf_to_ubuf(curMax, calcDst, 0, 1, 8, 1, 1);
                    copy_ubuf_to_ubuf(totalMax, calcDst, 0, 1, 8, 1, 1);
                } else {
                    copy_ubuf_to_ubuf(curMax, totalMax, 0, 1, 8, 1, 1);
                    copy_ubuf_to_ubuf(calcDst, totalMax, 0, 1, 8, 1, 1);
                    pipe_barrier(PIPE_V);
                }
            }

            vsub(calc, calc, calcDst, curRepeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            vexp(calc, calc, curRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            if (subBlockNum > 1) {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            }

            if (useV2) {
                ReduceSumV2(calcDst, calc, curLen);
                pipe_barrier(PIPE_V);
            } else {
                ReduceSum(calcDst, calc, curLen);
            }

            vbrcb((__ubuf__ uint32_t *)calcDst, (__ubuf__ uint32_t *)calcDst, 0, 0, 1);
            pipe_barrier(PIPE_V);

            if (subBlockNum > 1) {
                if (block > 0) {
                    auto *lastMax = max + (block - 1) * calcPad;
                    auto *lastSum = sum + (block - 1) * calcPad;
                    if (curMaxVal > totalMaxVal) {
                        auto *calcTemp = calcDst + calcPad;
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
                auto *curSum = sum + block * calcPad;
                copy_ubuf_to_ubuf(curSum, calcDst, 0, 1, 8, 1, 1);

                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
                if (curLen * sizeof(float) % BLOCK_SIZE == 0) {
                    copy_ubuf_to_gm(expBuf + offset, calc, 0, 1,
                                    curLen * sizeof(float) / BLOCK_SIZE, 0, 0);
                } else {
                    copy_ubuf_to_gm_align_b16(expBuf + offset, calc, 0, 1, curLen * sizeof(float),
                                              0, 0, 0, 0);
                }
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            copy_ubuf_to_ubuf(totalSum, calcDst, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);
        }

        if (subBlockNum > 1) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        }

        // stage2: exp(x - max(x)) / sum(exp(x - max(x)))
        for (block = 0; block < subBlockNum; block++) {
            uint32_t offset = block * MAX_SUB_CONTEXT_SIZE;
            uint32_t curLen = MAX_SUB_CONTEXT_SIZE;
            uint32_t curRepeat = maxRepeat;
            if (block == subBlockNum - 1) {
                curLen = actualCalcLen - offset;
                curRepeat = DIV_ROUND_UP(curLen, calcPad);
            }

            if (subBlockNum > 1) {
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                if (curLen * sizeof(float) % BLOCK_SIZE == 0) {
                    copy_gm_to_ubuf(calc, expBuf + offset, 0, 1,
                                    curLen * sizeof(float) / BLOCK_SIZE, 0, 0);
                } else {
                    copy_gm_to_ubuf_align_b16(calc, expBuf + offset, 0, 1, curLen * sizeof(float),
                                              0, 0, 0, 0);
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

                if (maxFlag[block]) {
                    auto *calcTemp = calcDst + calcPad;
                    auto *curMax = max + block * calcPad;
                    vsub(calcTemp, curMax, totalMax, 1, 1, 1, 1, 8, 8, 1);
                    pipe_barrier(PIPE_V);
                    vexp(calcTemp, calcTemp, 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    vmul(calc, calc, calcTemp, curRepeat, 1, 1, 0, 8, 8, 0);
                    pipe_barrier(PIPE_V);
                }
            }

            vdiv(calc, calc, totalSum, curRepeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                vconv_f322bf16r(out, calc, curRepeat, 1, 1, 4, 8);
            } else {
                vconv_f322f16r(out, calc, curRepeat, 1, 1, 4, 8);
            }
            if (subBlockNum > 1) {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            }
            pipe_barrier(PIPE_V);

            if (block == subBlockNum - 1 && calcOutLen > actualCalcLen) {
                if (curLen % pad != 0) {
                    SetMaskFromHighBit(pad, pad - curLen % pad);
                    vector_dup(out + ROUND_DOWN(curLen, pad), Dtype(0), 1, 1, 1, 8, 0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                int last = ROUND_UP(curLen, pad);
                if (calcOutLen - offset > last) {
                    vector_dup(out + last, Dtype(0), DIV_ROUND_UP(calcOutLen - offset - last, pad),
                               1, 1, 8, 0);
                }
                pipe_barrier(PIPE_V);
                curLen = calcOutLen - offset;
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
            if ((curLen * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_ubuf_to_gm_align_b16(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype), 0, 0,
                                          0, 0);
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
                curLen = outN - offset;
            }
            if ((curLen * sizeof(Dtype)) % BLOCK_SIZE == 0) {
                copy_ubuf_to_gm(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype) / BLOCK_SIZE, 0,
                                0);
            } else {
                copy_ubuf_to_gm_align_b16(curBuf + offset, out, 0, 1, curLen * sizeof(Dtype), 0, 0,
                                          0, 0);
            }
            if (block == totalSubBlockNum - 1) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

template <typename Dtype>
inline __aicore__ void RunAivSoftmax(__gm__ Dtype *buf, __gm__ float *expBuf, uint32_t m,
                                     uint32_t n, uint32_t calcLen, uint32_t outN = 0,
                                     uint32_t maskOff = 0, uint32_t maskStride = 1)
{
    RunAivSoftmaxLong<Dtype>(buf, expBuf, m, n, calcLen, outN, maskOff, maskStride);
}

#else
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxPingPong(__gm__ Dtype *buf, uint32_t m, uint32_t n,
                                             int calcLen, uint32_t outN = 0,
                                             uint32_t maskOff = 0, uint32_t maskStride = 1,
                                             __gm__ float *maxBuf = nullptr, __gm__ float *sumBuf = nullptr)
{
}
template <typename Dtype>
inline __aicore__ void RunAivSoftmaxLong(__gm__ Dtype *buf, __gm__ float *expBuf, uint32_t m,
                                         uint32_t n, uint32_t calcLen, uint32_t outN = 0,
                                         uint32_t maskOff = 0, uint32_t maskStride = 1)
{
}
#endif
#endif