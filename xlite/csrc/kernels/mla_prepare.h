/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"
#include "norm.h"
#include "rope_complex_and_cache.h"

#ifdef __DAV_C220_VEC__

template <typename Dtype>
__aicore__ void mla_prepare(GM_ADDR attnQkvc, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR attnNormQc,
                            GM_ADDR kvNorm, GM_ADDR kvNormBias, GM_ADDR attnNormKvc, GM_ADDR freqs,
                            GM_ADDR position, GM_ADDR kCache, GM_ADDR peCache, GM_ADDR slotMapping,
                            uint32_t token_num, uint32_t qLoraRank, uint32_t kvLoraRank,
                            uint32_t ropeHeadDim, uint32_t blockSize, float normEps,
                            uint32_t tpSize)
{
    auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::Rms);
    uint32_t totalDim = qLoraRank + kvLoraRank + ropeHeadDim;
    int coreOffset = 0;
    int nextCoreOffset = 0;

    norm<Dtype>(attnQkvc, nullptr, qNorm, qNormBias, attnNormQc, token_num, qLoraRank, normEps,
                kind, 1, totalDim, qLoraRank, 0, 0, true, nullptr, tpSize, nullptr, nullptr, 0,
                coreOffset, &nextCoreOffset);

    coreOffset = nextCoreOffset;
    norm<Dtype>(attnQkvc, nullptr, kvNorm, kvNormBias, attnNormKvc, token_num, kvLoraRank, normEps,
                kind, 1, totalDim, kvLoraRank, qLoraRank, 0, true, nullptr, tpSize, kCache,
                slotMapping, blockSize, coreOffset, &nextCoreOffset);

    coreOffset = nextCoreOffset;
    rope_complex_and_cache<Dtype>(token_num, 1, totalDim, ropeHeadDim, qLoraRank + kvLoraRank,
                                  kvLoraRank, ropeHeadDim, attnQkvc, freqs, position, blockSize,
                                  nullptr, nullptr, peCache, slotMapping, coreOffset,
                                  &nextCoreOffset);
}

#define MLA_PREPARE_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void mla_prepare_##dtype(                                    \
        GM_ADDR attnQkvc, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR attnNormQc, GM_ADDR kvNorm,   \
        GM_ADDR kvNormBias, GM_ADDR attnNormKvc, GM_ADDR freqs, GM_ADDR position, GM_ADDR kCache, \
        GM_ADDR peCache, GM_ADDR slotMapping, uint32_t token_num, uint32_t qLoraRank,             \
        uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t blockSize, float normEps,             \
        uint32_t tpSize)                                                                          \
    {                                                                                             \
        mla_prepare<dtype>(attnQkvc, qNorm, qNormBias, attnNormQc, kvNorm, kvNormBias,            \
                           attnNormKvc, freqs, position, kCache, peCache, slotMapping, token_num, \
                           qLoraRank, kvLoraRank, ropeHeadDim, blockSize, normEps, tpSize);       \
    }
#else
#define MLA_PREPARE_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void mla_prepare_##dtype(                                    \
        GM_ADDR attnQkvc, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR attnNormQc, GM_ADDR kvNorm,   \
        GM_ADDR kvNormBias, GM_ADDR attnNormKvc, GM_ADDR freqs, GM_ADDR position, GM_ADDR kCache, \
        GM_ADDR peCache, GM_ADDR slotMapping, uint32_t token_num, uint32_t qLoraRank,             \
        uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t blockSize, float normEps,             \
        uint32_t tpSize)                                                                          \
    {                                                                                             \
    }
#endif
