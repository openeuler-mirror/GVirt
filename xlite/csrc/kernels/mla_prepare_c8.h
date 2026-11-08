/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "mla_cache_c8.h"
#include "norm.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ inline void mla_prepare_c8(GM_ADDR attnQkvc, GM_ADDR qNorm, GM_ADDR qNormBias,
                                      GM_ADDR attnNormQc, GM_ADDR kvNorm, GM_ADDR kvNormBias,
                                      GM_ADDR attnNormKvc, GM_ADDR freqs, GM_ADDR position,
                                      GM_ADDR cache, GM_ADDR peCache, GM_ADDR scaleCache,
                                      GM_ADDR slots, uint32_t tokens, uint32_t qLoraRank,
                                      uint32_t latentDim, uint32_t ropeDim, float normEps,
                                      uint32_t tpSize)
{
    uint32_t totalDim = qLoraRank + latentDim + ropeDim;
    int coreOffset = 0;
    auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::Rms);
    norm<Dtype>(attnQkvc, nullptr, qNorm, qNormBias, attnNormQc, tokens, qLoraRank, normEps, kind,
                1, totalDim, qLoraRank, 0, 0, true, nullptr, tpSize, nullptr, nullptr, 0, 0, 0,
                &coreOffset);

    uint32_t first = (block_idx + block_num - coreOffset) % block_num;
    if (first >= tokens) {
        return;
    }
    constexpr uint32_t groupSize = 128;
    uint32_t groupNum = latentDim / groupSize;
    uint32_t scaleOffset = latentDim + ropeDim * sizeof(Dtype);
    uint32_t cacheBytes = scaleOffset + groupNum * sizeof(float);
    uint32_t repeat = latentDim / VECTOR_MAX_NUM_OF_FP32;
    uint32_t ropeRepeat = DIV_ROUND_UP(ropeDim, VECTOR_MAX_NUM_OF_FP32);
    float invDim = 1.0f / latentDim;

    __ubuf__ Dtype *inputs[PINGPONG_BUF_NUM];
    __ubuf__ Dtype *normalized[PINGPONG_BUF_NUM];
    __ubuf__ int8_t *packed[PINGPONG_BUF_NUM];
    __ubuf__ float *trig[PINGPONG_BUF_NUM];
    uint64_t off = 0;
    for (uint32_t curr = 0; curr < PINGPONG_BUF_NUM; ++curr) {
        inputs[curr] = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += ROUND_UP((latentDim + ropeDim) * sizeof(Dtype), VECTOR_REPEAT_BYTESIZE);
        normalized[curr] = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += ROUND_UP(latentDim * sizeof(Dtype), VECTOR_REPEAT_BYTESIZE);
        packed[curr] = reinterpret_cast<__ubuf__ int8_t *>(off);
        off += ROUND_UP(cacheBytes, VECTOR_REPEAT_BYTESIZE);
        trig[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(ropeDim * sizeof(float), VECTOR_REPEAT_BYTESIZE);
    }
    uint32_t calcBytes = latentDim * sizeof(float);
    auto x = reinterpret_cast<__ubuf__ float *>(off);
    off += MAX(calcBytes, ROUND_UP(ropeDim * sizeof(float), VECTOR_REPEAT_BYTESIZE));
    auto tmp = reinterpret_cast<__ubuf__ float *>(off);
    off += calcBytes;
    auto weight = reinterpret_cast<__ubuf__ float *>(off);
    off += calcBytes;
    auto bias = reinterpret_cast<__ubuf__ float *>(off);
    off += calcBytes;
    __ubuf__ float *rope[8];
    for (uint32_t i = 0; i < 8; ++i) {
        rope[i] = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(ropeDim * sizeof(float), VECTOR_REPEAT_BYTESIZE);
    }
    auto amaxBuf = reinterpret_cast<__ubuf__ float *>(off);
    off += VECTOR_REPEAT_BYTESIZE;
    auto factorBuf = reinterpret_cast<__ubuf__ float *>(off);
    off += VECTOR_REPEAT_BYTESIZE;
    auto halfValues = reinterpret_cast<__ubuf__ half *>(off);
    off += groupSize * sizeof(half);
    assert(off <= UB_SIZE);

    if (kvNorm) {
        CopyGmToUbufAligned(inputs[0], (__gm__ Dtype *)kvNorm, latentDim * sizeof(Dtype));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        convert_input(weight, inputs[0], repeat);
    }
    if (kvNormBias) {
        CopyGmToUbufAligned(inputs[1], (__gm__ Dtype *)kvNormBias, latentDim * sizeof(Dtype));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        convert_input(bias, inputs[1], repeat);
    }
    // Reuse the weight-loading buffers for token inputs.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (uint32_t curr = 0; curr < PINGPONG_BUF_NUM; ++curr) {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
    }
    int32_t slot = ((__gm__ int32_t *)slots)[first];
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    if (slot >= 0 || attnNormKvc) {
        CopyGmToUbufAligned(inputs[0],
                            (__gm__ Dtype *)attnQkvc + (uint64_t)first * totalDim + qLoraRank,
                            (latentDim + ropeDim) * sizeof(Dtype));
    }
    if (slot >= 0) {
        int64_t pos = ((__gm__ int64_t *)position)[first];
        CopyGmToUbufAligned(trig[0], (__gm__ float *)freqs + pos * ropeDim,
                            ropeDim * sizeof(float));
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

    uint32_t taskNum = DIV_ROUND_UP(tokens - first, block_num);
    for (uint32_t task = 0; task < taskNum; ++task) {
        uint32_t curr = task % PINGPONG_BUF_NUM;
        uint32_t row = first + task * block_num;
        int32_t nextSlot = -1;
        if (task + 1 < taskNum) {
            uint32_t next = 1 - curr;
            uint32_t nextRow = row + block_num;
            nextSlot = ((__gm__ int32_t *)slots)[nextRow];
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + next);
            if (nextSlot >= 0 || attnNormKvc) {
                CopyGmToUbufAligned(
                    inputs[next],
                    (__gm__ Dtype *)attnQkvc + (uint64_t)nextRow * totalDim + qLoraRank,
                    (latentDim + ropeDim) * sizeof(Dtype));
            }
            if (nextSlot >= 0) {
                int64_t pos = ((__gm__ int64_t *)position)[nextRow];
                CopyGmToUbufAligned(trig[next], (__gm__ float *)freqs + pos * ropeDim,
                                    ropeDim * sizeof(float));
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + next);
        }
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
        if (slot >= 0 || attnNormKvc) {
            // Match norm.h, including the BF16 boundary before quantization.
            convert_input(x, inputs[curr], repeat);
            pipe_barrier(PIPE_V);
            vmul(tmp, x, x, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vmuls(tmp, tmp, invDim, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            reduce_sum(tmp, 1, latentDim);
            pipe_barrier(PIPE_V);
            SetMask(1);
            vadds(tmp, tmp, normEps, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vsqrt(tmp, tmp, 1, 1, 1, 8, 8);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            duplicate_item(tmp, 1, repeat, latentDim);
            pipe_barrier(PIPE_V);
            vdiv(tmp, x, tmp, repeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if (kvNorm) {
                vmul(tmp, weight, tmp, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
            if (kvNormBias) {
                vadd(tmp, bias, tmp, repeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
            convert_output(normalized[curr], tmp, repeat);
        }
        if (slot >= 0) {
            // RoPE: interleaved input to [real | imag], as in rope_complex_and_cache.
            auto even = rope[0], odd = rope[1], cos = rope[2], sin = rope[3];
            auto evenCos = rope[4], oddSin = rope[5], evenSin = rope[6], oddCos = rope[7];
            convert_input(x, inputs[curr] + latentDim, ropeRepeat);
            pipe_barrier(PIPE_V);
            vreducev2((__ubuf__ uint32_t *)even, (__ubuf__ uint32_t *)x, (__ubuf__ uint32_t *)x,
                      ropeRepeat, 1, 1, 8, 0);
            vreducev2((__ubuf__ uint32_t *)odd, (__ubuf__ uint32_t *)x, (__ubuf__ uint32_t *)x,
                      ropeRepeat, 1, 2, 8, 0);
            vreducev2((__ubuf__ uint32_t *)cos, (__ubuf__ uint32_t *)trig[curr],
                      (__ubuf__ uint32_t *)trig[curr], ropeRepeat, 1, 1, 8, 0);
            vreducev2((__ubuf__ uint32_t *)sin, (__ubuf__ uint32_t *)trig[curr],
                      (__ubuf__ uint32_t *)trig[curr], ropeRepeat, 1, 2, 8, 0);
            pipe_barrier(PIPE_V);
            set_mask_count();
            set_vector_mask(0, ropeDim / 2);
            vmul(evenCos, even, cos, 1, 1, 1, 1, 8, 8, 8);
            vmul(oddSin, odd, sin, 1, 1, 1, 1, 8, 8, 8);
            vmul(evenSin, even, sin, 1, 1, 1, 1, 8, 8, 8);
            vmul(oddCos, odd, cos, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vsub(x, evenCos, oddSin, 1, 1, 1, 1, 8, 8, 8);
            vadd(x + ropeDim / 2, evenSin, oddCos, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0, ropeDim);
            convert_output((__ubuf__ Dtype *)(packed[curr] + latentDim), x, 1);
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            for (uint32_t group = 0; group < groupNum; ++group) {
                convert_input(x, normalized[curr] + group * groupSize,
                              groupSize / VECTOR_MAX_NUM_OF_FP32);
                pipe_barrier(PIPE_V);
                mla_cache_c8_quant(x, tmp, amaxBuf, factorBuf,
                                   (__ubuf__ float *)(packed[curr] + scaleOffset) + group,
                                   groupSize);
                vconv_f322f16o(halfValues, x, groupSize / VECTOR_MAX_NUM_OF_FP32, 1, 1, 4, 8);
                pipe_barrier(PIPE_V);
                vconv_f162s8r(packed[curr] + group * groupSize, halfValues,
                              groupSize / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
            }
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (attnNormKvc) {
            CopyUbufToGmAligned((__gm__ Dtype *)attnNormKvc + (uint64_t)row * latentDim,
                                normalized[curr], latentDim * sizeof(Dtype));
        }
        if (slot >= 0) {
            CopyUbufToGmAligned((__gm__ int8_t *)cache + (uint64_t)slot * latentDim, packed[curr],
                                latentDim);
            CopyUbufToGmAligned((__gm__ Dtype *)peCache + (uint64_t)slot * ropeDim,
                                packed[curr] + latentDim, ropeDim * sizeof(Dtype));
            CopyUbufToGmAligned((__gm__ float *)scaleCache + (uint64_t)slot * groupNum,
                                packed[curr] + scaleOffset, groupNum * sizeof(float));
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
        slot = nextSlot;
    }
    for (uint32_t curr = 0; curr < PINGPONG_BUF_NUM; ++curr) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
    }
}

#define MLA_PREPARE_C8_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void mla_prepare_c8_##dtype(                                \
        GM_ADDR attnQkvc, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR attnNormQc, GM_ADDR kvNorm,  \
        GM_ADDR kvNormBias, GM_ADDR attnNormKvc, GM_ADDR freqs, GM_ADDR position, GM_ADDR cache, \
        GM_ADDR peCache, GM_ADDR scaleCache, GM_ADDR slots, uint32_t tokens, uint32_t qLoraRank, \
        uint32_t latentDim, uint32_t ropeDim, float normEps, uint32_t tpSize)                    \
    {                                                                                            \
        mla_prepare_c8<dtype>(attnQkvc, qNorm, qNormBias, attnNormQc, kvNorm, kvNormBias,        \
                              attnNormKvc, freqs, position, cache, peCache, scaleCache, slots,   \
                              tokens, qLoraRank, latentDim, ropeDim, normEps, tpSize);           \
    }
#else
#define MLA_PREPARE_C8_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void mla_prepare_c8_##dtype(                                \
        GM_ADDR attnQkvc, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR attnNormQc, GM_ADDR kvNorm,  \
        GM_ADDR kvNormBias, GM_ADDR attnNormKvc, GM_ADDR freqs, GM_ADDR position, GM_ADDR cache, \
        GM_ADDR peCache, GM_ADDR scaleCache, GM_ADDR slots, uint32_t tokens, uint32_t qLoraRank, \
        uint32_t latentDim, uint32_t ropeDim, float normEps, uint32_t tpSize)                    \
    {                                                                                            \
    }
#endif
