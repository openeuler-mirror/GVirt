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
    // Split UB into two staging halves so the MTE2 (GM->UB) load of task k+1
    // overlaps the MTE3 (UB->GM) store of task k (split.cpp protocol). NZ
    // blockBytes is always a whole multiple of 32 bytes (16 rows x 32B column
    // bytes), and bufBytes is 32B-aligned, so every sub-chunk copy stays on
    // the block-count DMA path.
    const uint32_t bufBytes = ROUND_DOWN(UB_SIZE / 2, BLOCK_SIZE);
    __ubuf__ uint8_t *buffer[2];
    buffer[0] = reinterpret_cast<__ubuf__ uint8_t *>(0);
    buffer[1] = buffer[0] + bufBytes;
    const uint64_t tilesPerBlock = DIV_ROUND_UP(blockBytes, bufBytes);
    const uint64_t tasks = uint64_t(ranks) * columnBlocks * tilesPerBlock;
    // Seed both halves' MTE3->MTE2 flags so the first wait of each can consume.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    uint32_t curr = 0;
    bool hasPending = false;
    __gm__ uint8_t *pendDst = nullptr;
    uint32_t pendBytes = 0;
    for (uint64_t task = block_idx; task < tasks; task += block_num) {
        const uint64_t dstBlock = task / tilesPerBlock;
        const uint64_t offset = (task % tilesPerBlock) * bufBytes;
        const uint64_t rank = dstBlock % ranks;
        const uint64_t column = dstBlock / ranks;
        const uint64_t srcBlock = rank * columnBlocks + column;
        const uint32_t copyBytes = MIN(uint64_t(bufBytes), blockBytes - offset);
        // AllGather produces [rank, N1, local K1, 16, C0]; matmul needs
        // [N1, rank, local K1, 16, C0]. Copy whole NZ blocks without casting.
        // Store the previous task's sub-chunk from the other half.
        if (hasPending) {
            const uint32_t wbuf = 1 - curr;
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
            CopyUbufToGmAligned(pendDst, buffer[wbuf], pendBytes);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
        }
        // Load this task's sub-chunk into the current half. The deferred store
        // crosses task boundaries: input is only read and output only written
        // (disjoint GM tensors), so the event flags alone guard UB reuse.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(buffer[curr], (__gm__ uint8_t *)input + srcBlock * blockBytes + offset,
                            copyBytes);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
        pendDst = (__gm__ uint8_t *)output + dstBlock * blockBytes + offset;
        pendBytes = copyBytes;
        hasPending = true;
        curr = 1 - curr;
    }
    // Flush the last buffered sub-chunk.
    if (hasPending) {
        const uint32_t wbuf = 1 - curr;
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);
        CopyUbufToGmAligned(pendDst, buffer[wbuf], pendBytes);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);
    }
    // Drain both staging halves before kernel exit.
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}
#endif
