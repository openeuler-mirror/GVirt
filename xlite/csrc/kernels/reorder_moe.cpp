/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

__aicore__ inline void reorder_moe_kernel(GM_ADDR input, GM_ADDR output, GM_ADDR counts,
                                          uint32_t moeEpSize, uint32_t nRoutedExperts,
                                          uint32_t hiddenSize, uint32_t localStart,
                                          uint32_t localEnd, uint32_t forward, uint32_t elemBytes)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t nLocalExperts = localEnd - localStart;
    uint32_t numPairs = moeEpSize * nLocalExperts;
    if (numPairs == 0)
        return;

    // distribute pairs across blocks
    uint32_t pairStart, pairEnd;
    uint32_t remain = numPairs % block_num;
    uint32_t avg = numPairs / block_num;
    if (block_idx < remain) {
        pairStart = block_idx * avg + block_idx;
        pairEnd = pairStart + avg + 1;
    } else {
        pairStart = block_idx * avg + remain;
        pairEnd = pairStart + avg;
    }

    if (pairStart >= numPairs)
        return;

    // --- load counts into UB ---
    uint32_t countsBytes = moeEpSize * nRoutedExperts * sizeof(int32_t);

    uint64_t off = 0;
    __ubuf__ int32_t *countsBuf = (__ubuf__ int32_t *)((uintptr_t)off);
    off += ROUND_UP(countsBytes, VECTOR_MAX_BYTESIZE);
    // expertOffsets[0..nLocalExperts): running offset of each expert in expert-grouped layout
    __ubuf__ int32_t *expertOffsets = (__ubuf__ int32_t *)((uintptr_t)off);
    off += ROUND_UP(nLocalExperts * sizeof(int32_t), VECTOR_MAX_BYTESIZE);
    // chunkOffsets[0..moeEpSize): token offset of each source chunk in source-grouped layout
    __ubuf__ int32_t *chunkOffsets = (__ubuf__ int32_t *)((uintptr_t)off);
    off += ROUND_UP(moeEpSize * sizeof(int32_t), VECTOR_MAX_BYTESIZE);
    __ubuf__ uint8_t *dataBuf = (__ubuf__ uint8_t *)((uintptr_t)off);
    uint32_t dataBufSize = UB_SIZE - (uint32_t)off;

    copy_gm_to_ubuf(countsBuf, (__gm__ int32_t *)counts, 0, 1,
                    DIV_ROUND_UP(countsBytes, BLOCK_SIZE), 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    // --- compute chunkOffsets and expertOffsets from counts ---
    int32_t chunkAccum = 0;
    for (uint32_t i = 0; i < moeEpSize; i++) {
        chunkOffsets[i] = chunkAccum;
        for (uint32_t e = localStart; e < localEnd; e++) {
            chunkAccum += countsBuf[i * nRoutedExperts + e];
        }
    }
    int32_t running = 0;
    for (uint32_t e = localStart; e < localEnd; e++) {
        expertOffsets[e - localStart] = running;
        for (uint32_t i = 0; i < moeEpSize; i++) {
            running += countsBuf[i * nRoutedExperts + e];
        }
    }
    pipe_barrier(PIPE_V);

    // --- iterate all pairs, each block copies its assigned pairs ---
    uint32_t tileBytes = hiddenSize * elemBytes;
    uint32_t p = 0;
    for (uint32_t i = 0; i < moeEpSize; i++) {
        int32_t chunkRun = chunkOffsets[i];
        for (uint32_t e = localStart; e < localEnd; e++, p++) {
            int32_t cnt = countsBuf[i * nRoutedExperts + e];
            if (cnt == 0)
                continue;

            int32_t srcOff, dstOff;
            if (forward) {
                srcOff = chunkRun;
                dstOff = expertOffsets[e - localStart];
            } else {
                srcOff = expertOffsets[e - localStart];
                dstOff = chunkRun;
            }

            if (p >= pairStart && p < pairEnd) {
                uint32_t srcByteOff = (uint32_t)srcOff * tileBytes;
                uint32_t dstByteOff = (uint32_t)dstOff * tileBytes;
                uint32_t totalBytes = (uint32_t)cnt * tileBytes;

                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                for (uint32_t bo = 0; bo < totalBytes; bo += dataBufSize) {
                    uint32_t chunk =
                        (bo + dataBufSize > totalBytes) ? totalBytes - bo : dataBufSize;
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

                    copy_gm_to_ubuf(dataBuf, (__gm__ uint8_t *)input + srcByteOff + bo, 0, 1,
                                    DIV_ROUND_UP(chunk, BLOCK_SIZE), 0, 0);
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

                    copy_ubuf_to_gm_align_b32((__gm__ uint8_t *)output + dstByteOff + bo, dataBuf,
                                              0, 1, chunk, 0, 0, 0, 0);

                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                }
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }

            // always advance offsets for the next pair
            expertOffsets[e - localStart] += cnt;
            chunkRun += cnt;
        }
    }
}

extern "C" __global__ __aicore__ void reorder_moe(GM_ADDR input, GM_ADDR output, GM_ADDR counts,
                                                  uint32_t moeEpSize, uint32_t nRoutedExperts,
                                                  uint32_t hiddenSize, uint32_t localStart,
                                                  uint32_t localEnd, uint32_t forward,
                                                  uint32_t elemBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    reorder_moe_kernel(input, output, counts, moeEpSize, nRoutedExperts, hiddenSize, localStart,
                       localEnd, forward, elemBytes);
}
#endif
