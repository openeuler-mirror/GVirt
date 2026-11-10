/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void dsa_cp_metadata(
    GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR localQueryStartLoc,
    GM_ADDR localQueryLens, GM_ADDR localCachedLens, GM_ADDR localSlotMapping, uint32_t batch,
    uint32_t shardBegin, uint32_t shardEnd, uint32_t localTokenCapacity)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    // Scalar GM accesses use the data cache. Metadata may have been updated by
    // DMA since the previous layer/step; publish scalar writes for vector DMA.
    for (uint32_t i = 0; i < batch; i += 16) {
        DataCacheCleanAndInvalid((__gm__ int32_t *)queryStartLoc + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)queryLens + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)cachedLens + i);
    }
    // Keep the global request/block-table order, including zero-length intersections.
    for (uint32_t i = block_idx; i < batch; i += block_num) {
        uint32_t requestBegin = ((__gm__ int32_t *)queryStartLoc)[i];
        uint32_t requestLen = ((__gm__ int32_t *)queryLens)[i];
        uint32_t intersectionBegin = MAX(requestBegin, shardBegin);
        uint32_t intersectionEnd = MIN(requestBegin + requestLen, shardEnd);
        uint32_t localQueryLen =
            intersectionEnd > intersectionBegin ? intersectionEnd - intersectionBegin : 0;
        ((__gm__ int32_t *)localQueryStartLoc)[i] =
            localQueryLen ? intersectionBegin - shardBegin : 0;
        ((__gm__ int32_t *)localQueryLens)[i] = localQueryLen;
        ((__gm__ int32_t *)localCachedLens)[i] =
            ((__gm__ int32_t *)cachedLens)[i] +
            (localQueryLen ? intersectionBegin - requestBegin : 0);
    }
    for (uint32_t i = block_idx; i < localTokenCapacity; i += block_num) {
        ((__gm__ int32_t *)localSlotMapping)[i] = i;
    }
    for (uint32_t i = 0; i < batch; i += 16) {
        DataCacheCleanAndInvalid((__gm__ int32_t *)localQueryStartLoc + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)localQueryLens + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)localCachedLens + i);
    }
    for (uint32_t i = 0; i < localTokenCapacity; i += 16) {
        DataCacheCleanAndInvalid((__gm__ int32_t *)localSlotMapping + i);
    }
}
#endif
