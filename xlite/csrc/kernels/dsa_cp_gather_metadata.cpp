/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void dsa_cp_gather_metadata(
    GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR groupQueryStartLoc,
    GM_ADDR groupQueryLens, GM_ADDR groupCachedLens, GM_ADDR groupRequestIds, uint32_t batch,
    uint32_t groupBegin, uint32_t groupEnd)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    set_atomic_none();
    const uint32_t groupLen = groupEnd - groupBegin;
    // One vector core owns the small metadata arrays, including their cache lines.
    for (uint32_t i = 0; i < batch; i += 8) {
        DataCacheCleanAndInvalid((__gm__ int32_t *)queryStartLoc + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)queryLens + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)cachedLens + i);
    }
    for (uint32_t request = 0; request < batch; ++request) {
        uint32_t requestBegin = ((__gm__ int32_t *)queryStartLoc)[request];
        uint32_t requestLen = ((__gm__ int32_t *)queryLens)[request];
        uint32_t intersectionBegin = MAX(requestBegin, groupBegin);
        uint32_t intersectionEnd = MIN(requestBegin + requestLen, groupEnd);
        for (uint32_t query = intersectionBegin; query < intersectionEnd; ++query) {
            uint32_t row = query - groupBegin;
            ((__gm__ int32_t *)groupQueryStartLoc)[row] = row;
            ((__gm__ int32_t *)groupQueryLens)[row] = 1;
            // A query's prefix includes earlier tokens in this prefill chunk.
            ((__gm__ int32_t *)groupCachedLens)[row] =
                ((__gm__ int32_t *)cachedLens)[request] + query - requestBegin;
            ((__gm__ int32_t *)groupRequestIds)[row] = request;
        }
    }
    for (uint32_t i = 0; i < groupLen; i += 8) {
        DataCacheCleanAndInvalid((__gm__ int32_t *)groupQueryStartLoc + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)groupQueryLens + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)groupCachedLens + i);
        DataCacheCleanAndInvalid((__gm__ int32_t *)groupRequestIds + i);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
