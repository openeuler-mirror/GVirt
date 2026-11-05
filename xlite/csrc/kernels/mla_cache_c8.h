/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ inline void mla_cache_c8_quant(__ubuf__ Dtype *input, __ubuf__ int8_t *quant,
                                          __ubuf__ float *scale, __ubuf__ float *x,
                                          __ubuf__ float *tmp, __ubuf__ float *amaxBuf,
                                          __ubuf__ float *factorBuf, __ubuf__ half *halfValues)
{
    convert_input(x, input, 2);
    pipe_barrier(PIPE_V);
    vabs(tmp, x, 2, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    ReduceMax(amaxBuf, tmp, 128);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float amax = *amaxBuf;
    float factor = 0.0f;
    if (amax != 0.0f) {
        SetMask(1);
        vector_dup(factorBuf, 1.0f, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vdiv(factorBuf, factorBuf, amaxBuf, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(factorBuf, factorBuf, 127.0f, 1, 1, 1, 8, 8);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        factor = *factorBuf;
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
    *scale = amax * (1.0f / 127.0f);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    vmuls(x, x, factor, 2, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    // FP32 -> FP16 (ODD) -> INT8 (RINT) avoids double rounding.
    vconv_f322f16o(halfValues, x, 2, 1, 1, 4, 8);
    pipe_barrier(PIPE_V);
    vconv_f162s8r(quant, halfValues, 1, 1, 1, 4, 8);
}

template <typename Dtype>
__aicore__ inline void mla_cache_c8(GM_ADDR latent, GM_ADDR pe, GM_ADDR slots, GM_ADDR cache,
                                    GM_ADDR peCache, GM_ADDR scaleCache, uint32_t tokens,
                                    uint32_t latentDim, uint32_t ropeDim)
{
    constexpr uint32_t groupSize = 128;
    uint32_t groupNum = latentDim / groupSize;
    uint32_t scaleOffset = latentDim * sizeof(int8_t) + ropeDim * sizeof(Dtype);
    uint32_t cacheBytes = scaleOffset + groupNum * sizeof(float);
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t inputSize = ROUND_UP(latentDim * sizeof(Dtype), VECTOR_REPEAT_BYTESIZE);
    uint32_t packedSize = ROUND_UP(cacheBytes, VECTOR_REPEAT_BYTESIZE);
    constexpr uint32_t calcSize = ROUND_UP(groupSize * sizeof(float), UB_BUF_ALIGN_SIZE);
    __ubuf__ Dtype *inputs[PINGPONG_BUF_NUM];
    __ubuf__ int8_t *packed[PINGPONG_BUF_NUM];
    __ubuf__ float *xBuf[PINGPONG_BUF_NUM];
    __ubuf__ float *tmpBuf[PINGPONG_BUF_NUM];
    __ubuf__ float *amaxBufs[PINGPONG_BUF_NUM];
    __ubuf__ float *factorBufs[PINGPONG_BUF_NUM];
    __ubuf__ half *halfBuf[PINGPONG_BUF_NUM];
    uint64_t off = 0;
    for (uint32_t curr = 0; curr < PINGPONG_BUF_NUM; ++curr) {
        inputs[curr] = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += inputSize;
        packed[curr] = reinterpret_cast<__ubuf__ int8_t *>(off);
        off += packedSize;
        xBuf[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += calcSize;
        tmpBuf[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += calcSize;
        amaxBufs[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_REPEAT_BYTESIZE;
        factorBufs[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_REPEAT_BYTESIZE;
        halfBuf[curr] = reinterpret_cast<__ubuf__ half *>(off);
        off += ROUND_UP(groupSize * sizeof(half), UB_BUF_ALIGN_SIZE);
        off = ROUND_UP(off, UB_BANKGROUP_ROW_SIZE);
    }
    assert(off <= UB_SIZE);

    // Each bank is reusable after all cache writes complete.
    for (uint32_t curr = 0; curr < PINGPONG_BUF_NUM; ++curr) {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
    }
    int32_t slot = -1;
    uint32_t taskNum = DIV_ROUND_UP(tokens - block_idx, block_num);
    for (uint32_t task = 0; task <= taskNum; ++task) {
        int32_t nextSlot = -1;
        if (task < taskNum) {
            uint32_t next = task % PINGPONG_BUF_NUM;
            uint32_t nextRow = block_idx + task * block_num;
            nextSlot = ((__gm__ int32_t *)slots)[nextRow];
            if (nextSlot >= 0) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + next);
                CopyGmToUbufAligned(inputs[next],
                                    (__gm__ Dtype *)latent + (uint64_t)nextRow * latentDim,
                                    latentDim * sizeof(Dtype));
                CopyGmToUbufAligned((__ubuf__ Dtype *)(packed[next] + latentDim * sizeof(int8_t)),
                                    (__gm__ Dtype *)pe + (uint64_t)nextRow * ropeDim,
                                    ropeDim * sizeof(Dtype));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + next);
            }
        }
        if (task == 0) {
            slot = nextSlot;
            continue;
        }
        uint32_t curr = (task - 1) % PINGPONG_BUF_NUM;
        if (slot >= 0) {
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
            auto x = xBuf[curr];
            auto tmp = tmpBuf[curr];
            auto amaxBuf = amaxBufs[curr];
            auto factorBuf = factorBufs[curr];
            auto halfValues = halfBuf[curr];
            for (uint32_t group = 0; group < groupNum; ++group) {
                auto input = inputs[curr] + group * groupSize;
                auto quant = packed[curr] + group * groupSize;
                auto scale = (__ubuf__ float *)(packed[curr] + scaleOffset) + group;
                mla_cache_c8_quant(input, quant, scale, x, tmp, amaxBuf, factorBuf, halfValues);
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
            CopyUbufToGmAligned((__gm__ int8_t *)cache + (uint64_t)slot * latentDim, packed[curr],
                                latentDim);
            CopyUbufToGmAligned((__gm__ Dtype *)peCache + (uint64_t)slot * ropeDim,
                                packed[curr] + latentDim, ropeDim * sizeof(Dtype));
            CopyUbufToGmAligned((__gm__ float *)scaleCache + (uint64_t)slot * groupNum,
                                packed[curr] + scaleOffset, groupNum * sizeof(float));
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
        }
        slot = nextSlot;
    }
    for (uint32_t curr = 0; curr < PINGPONG_BUF_NUM; ++curr) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
    }
}

#define MLA_CACHE_C8_FUNC_DEFINE(dtype)                                                       \
    extern "C" __global__ __aicore__ void mla_cache_c8_##dtype(                               \
        GM_ADDR latent, GM_ADDR pe, GM_ADDR slots, GM_ADDR cache, GM_ADDR peCache,            \
        GM_ADDR scaleCache, uint32_t tokens, uint32_t latentDim, uint32_t ropeDim)            \
    {                                                                                         \
        mla_cache_c8<dtype>(latent, pe, slots, cache, peCache, scaleCache, tokens, latentDim, \
                            ropeDim);                                                         \
    }
#else
#define MLA_CACHE_C8_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void mla_cache_c8_##dtype(                    \
        GM_ADDR latent, GM_ADDR pe, GM_ADDR slots, GM_ADDR cache, GM_ADDR peCache, \
        GM_ADDR scaleCache, uint32_t tokens, uint32_t latentDim, uint32_t ropeDim) \
    {                                                                              \
    }
#endif
