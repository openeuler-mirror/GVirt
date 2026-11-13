/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "gather_sparse_kv_cache_per_query.h"

extern "C" __global__ __aicore__ void gather_sparse_kv_cache_per_query_bfloat16_t(
    GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,
    GM_ADDR groupCachedLens, GM_ADDR groupRequestIds, GM_ADDR kDenseCache, GM_ADDR peDenseCache,
    uint32_t groupLen, uint32_t topK, uint32_t blockSize, uint32_t blockTableStride,
    uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride, uint32_t peCacheStride)
{
#ifdef __DAV_C220_VEC__
    gather_sparse_kv_cache_per_query<bfloat16_t>(
        kCache, peCache, blockTables, topkIndices, groupCachedLens, groupRequestIds, kDenseCache,
        peDenseCache, groupLen, topK, blockSize, blockTableStride, kvLoraRank, ropeHeadDim,
        kCacheStride, peCacheStride);
#endif
}
