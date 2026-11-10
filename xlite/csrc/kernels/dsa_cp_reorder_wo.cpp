/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void dsa_cp_reorder_wo(GM_ADDR input, GM_ADDR output,
                                                        uint32_t ranks, uint32_t columnBlocks,
                                                        uint64_t blockBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    set_atomic_none();
    auto buffer = reinterpret_cast<__ubuf__ uint8_t *>(0);
    const uint64_t tilesPerBlock = DIV_ROUND_UP(blockBytes, UB_SIZE);
    const uint64_t tasks = uint64_t(ranks) * columnBlocks * tilesPerBlock;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (uint64_t task = block_idx; task < tasks; task += block_num) {
        const uint64_t dstBlock = task / tilesPerBlock;
        const uint64_t offset = (task % tilesPerBlock) * UB_SIZE;
        const uint64_t rank = dstBlock % ranks;
        const uint64_t column = dstBlock / ranks;
        const uint64_t srcBlock = rank * columnBlocks + column;
        const uint32_t copyBytes = MIN(uint64_t(UB_SIZE), blockBytes - offset);
        // AllGather produces [rank, N1, local K1, 16, C0]; matmul needs
        // [N1, rank, local K1, 16, C0]. Copy whole NZ blocks without casting.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        CopyGmToUbufAligned(buffer, (__gm__ uint8_t *)input + srcBlock * blockBytes + offset,
                            copyBytes);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        CopyUbufToGmAligned((__gm__ uint8_t *)output + dstBlock * blockBytes + offset, buffer,
                            copyBytes);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
