/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"

#ifdef __DAV_C220_VEC__
// Per-group quantization: vabs → ReduceMax → factor → vmuls.
// convert_input (BF16→FP32) and vconv (FP32→INT8) are hoisted to the caller
// so they run once for the entire row instead of per group.
__aicore__ inline void mla_cache_c8_quant(__ubuf__ float *x, __ubuf__ float *tmp,
                                          __ubuf__ float *amaxBuf, __ubuf__ float *factorBuf,
                                          __ubuf__ float *scale, uint32_t groupSize)
{
    uint16_t fp32Rep = groupSize / VECTOR_MAX_NUM_OF_FP32;
    vabs(tmp, x, fp32Rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    ReduceMax(amaxBuf, tmp, groupSize);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float amax = *amaxBuf;
    *scale = amax * (1.0f / 127.0f);
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
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    vmuls(x, x, factor, fp32Rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
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
    uint32_t xBufSize = ROUND_UP(latentDim * sizeof(float), UB_BUF_ALIGN_SIZE);
    uint32_t halfBufSize = ROUND_UP(latentDim * sizeof(half), UB_BUF_ALIGN_SIZE);
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
        off += xBufSize;
        tmpBuf[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += calcSize;
        amaxBufs[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_REPEAT_BYTESIZE;
        factorBufs[curr] = reinterpret_cast<__ubuf__ float *>(off);
        off += VECTOR_REPEAT_BYTESIZE;
        halfBuf[curr] = reinterpret_cast<__ubuf__ half *>(off);
        off += halfBufSize;
        off = ROUND_UP(off, UB_BANKGROUP_ROW_SIZE);
    }
    assert(off <= UB_SIZE);

    // Each bank is reusable after all cache writes complete.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    uint32_t eid = 1;
    for (uint32_t row = block_idx; row < tokens; row += block_num) {
        int32_t slot = ((__gm__ int32_t *)slots)[row];
        if (slot < 0) {
            continue;
        }
        eid = 1 - eid;

        // MTE2: load latent + pe into buffer[eid]
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + eid);
        CopyGmToUbufAligned(inputs[eid], (__gm__ Dtype *)latent + (uint64_t)row * latentDim,
                            latentDim * sizeof(Dtype));
        CopyGmToUbufAligned((__ubuf__ Dtype *)(packed[eid] + latentDim * sizeof(int8_t)),
                            (__gm__ Dtype *)pe + (uint64_t)row * ropeDim, ropeDim * sizeof(Dtype));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eid);

        // V: convert entire row BF16 → FP32 once
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eid);
        convert_input(xBuf[eid], inputs[eid], latentDim / VECTOR_MAX_NUM_OF_FP32);
        pipe_barrier(PIPE_V);

        // V: per-group quantization (vabs → ReduceMax → factor → vmuls)
        auto tmp = tmpBuf[eid];
        auto amaxBuf = amaxBufs[eid];
        auto factorBuf = factorBufs[eid];
        for (uint32_t group = 0; group < groupNum; ++group) {
            auto x = xBuf[eid] + group * groupSize;
            auto scale = (__ubuf__ float *)(packed[eid] + scaleOffset) + group;
            mla_cache_c8_quant(x, tmp, amaxBuf, factorBuf, scale, groupSize);
        }

        // V: convert entire row FP32 → FP16(ODD) → INT8(RINT) once
        // FP32 -> FP16 (ODD) -> INT8 (RINT) avoids double rounding.
        vconv_f322f16o(halfBuf[eid], xBuf[eid], latentDim / VECTOR_MAX_NUM_OF_FP32, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_f162s8r(packed[eid], halfBuf[eid], latentDim / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eid);

        // MTE3: write quantized cache + pe + scale back to GM
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eid);
        CopyUbufToGmAligned((__gm__ int8_t *)cache + (uint64_t)slot * latentDim, packed[eid],
                            latentDim);
        CopyUbufToGmAligned((__gm__ Dtype *)peCache + (uint64_t)slot * ropeDim,
                            packed[eid] + latentDim, ropeDim * sizeof(Dtype));
        CopyUbufToGmAligned((__gm__ float *)scaleCache + (uint64_t)slot * groupNum,
                            packed[eid] + scaleOffset, groupNum * sizeof(float));
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + eid);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
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
