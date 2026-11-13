/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void dsa_cp_write_kv_cache_by_slot(
    GM_ADDR input, GM_ADDR cache, GM_ADDR slotMapping, uint32_t numTokens, uint32_t tokenBytes,
    uint64_t inputStrideBytes, uint32_t blockSize, uint64_t blockStrideBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    set_atomic_none();
    auto buffer = reinterpret_cast<__ubuf__ uint8_t *>(0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (uint32_t t = block_idx; t < numTokens; t += block_num) {
        DataCacheCleanAndInvalid((__gm__ int32_t *)slotMapping + t);
        int32_t slot = ((__gm__ int32_t *)slotMapping)[t];
        if (slot < 0) {
            continue;
        }
        uint64_t cacheOffset =
            (slot / blockSize) * blockStrideBytes + (slot % blockSize) * tokenBytes;
        for (uint32_t offset = 0; offset < tokenBytes; offset += UB_SIZE) {
            uint32_t copyBytes = MIN(UB_SIZE, tokenBytes - offset);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            CopyGmToUbufAligned(buffer,
                                (__gm__ uint8_t *)input + uint64_t(t) * inputStrideBytes + offset,
                                copyBytes);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            CopyUbufToGmAligned((__gm__ uint8_t *)cache + cacheOffset + offset, buffer, copyBytes);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
