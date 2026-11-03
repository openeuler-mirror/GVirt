/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "gather_sparse_kv_cache.h"

#ifdef __DAV_C220_VEC__
#define GATHER_SPARSE_KV_AND_DEQUANT_FUNC_DEFINE(dtype)                                           \
    extern "C" __global__ __aicore__ void gather_sparse_kv_and_dequant_##dtype(                   \
        GM_ADDR cache, GM_ADDR peCache, GM_ADDR scaleCache, GM_ADDR blockTables, GM_ADDR indices, \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR latent, GM_ADDR pe, \
        uint32_t batch, uint32_t outputLen, uint32_t blockSize, uint32_t blockTableStride,        \
        uint32_t latentDim, uint32_t ropeDim, uint32_t burstCopySize)                             \
    {                                                                                             \
        gather_sparse_kv_cache<dtype, bfloat16_t>(                                                \
            cache, peCache, blockTables, indices, queryStartLoc, queryLens, cachedLens, latent,   \
            pe, batch, outputLen, blockSize, blockTableStride, latentDim, ropeDim, 0, 0, 1,       \
            burstCopySize, scaleCache);                                                           \
    }
#else
#define GATHER_SPARSE_KV_AND_DEQUANT_FUNC_DEFINE(dtype)                                           \
    extern "C" __global__ __aicore__ void gather_sparse_kv_and_dequant_##dtype(                   \
        GM_ADDR cache, GM_ADDR peCache, GM_ADDR scaleCache, GM_ADDR blockTables, GM_ADDR indices, \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR latent, GM_ADDR pe, \
        uint32_t batch, uint32_t outputLen, uint32_t blockSize, uint32_t blockTableStride,        \
        uint32_t latentDim, uint32_t ropeDim, uint32_t burstCopySize)                             \
    {                                                                                             \
    }
#endif
