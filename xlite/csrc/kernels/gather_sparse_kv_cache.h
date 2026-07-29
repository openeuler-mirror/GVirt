/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_param.h"
#include "kernel_operator.h"
// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#define CNT_ONE_LOOP 16

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ __inline__ void gather_sparse_kv_cache(GM_ADDR kCache, GM_ADDR peCache,
                                                  GM_ADDR blockTables, GM_ADDR topkIndices,
                                                  GM_ADDR queryLens, GM_ADDR cachedLens,
                                                  GM_ADDR kDenseCache, GM_ADDR peDenseCache,
                                                  uint32_t batch, uint32_t indexTopK,
                                                  uint32_t blockSize, uint32_t maxNumBlocks,
                                                  uint32_t kvLoraRank, uint32_t ropeHeadDim)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t kHeadBytes = kvLoraRank * sizeof(Dtype);
    uint32_t peHeadBytes = ropeHeadDim * sizeof(Dtype);
    uint32_t kHeadBytesPad = ROUND_UP(kHeadBytes, BLOCK_SIZE) * CNT_ONE_LOOP;
    uint32_t peHeadBytesPad = ROUND_UP(peHeadBytes, BLOCK_SIZE) * CNT_ONE_LOOP;

    uint64_t off = 0;
    __ubuf__ Dtype *kBuf0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += kHeadBytesPad;
    __ubuf__ Dtype *peBuf0 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += peHeadBytesPad;
    __ubuf__ Dtype *kBuf1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += kHeadBytesPad;
    __ubuf__ Dtype *peBuf1 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    off += peHeadBytesPad;
    __ubuf__ uint32_t *topks0 = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
    off += CNT_ONE_LOOP * sizeof(uint32_t);
    __ubuf__ uint32_t *topks1 = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
    off += CNT_ONE_LOOP * sizeof(uint32_t);
    assert(off <= UB_SIZE);
    __ubuf__ uint32_t *topks[PINGPONG_BUF_NUM] = {topks0, topks1};
    __ubuf__ Dtype *kBuf[PINGPONG_BUF_NUM] = {kBuf0, kBuf1};
    __ubuf__ Dtype *peBuf[PINGPONG_BUF_NUM] = {peBuf0, peBuf1};

    __gm__ Dtype *kCacheGm = (__gm__ Dtype *)kCache;
    __gm__ Dtype *peCacheGm = (__gm__ Dtype *)peCache;
    __gm__ int32_t *blockTablesGm = (__gm__ int32_t *)blockTables;
    __gm__ int32_t *topkIndicesGm = (__gm__ int32_t *)topkIndices;
    __gm__ int32_t *queryLensGm = (__gm__ int32_t *)queryLens;
    __gm__ int32_t *cachedLensGm = (__gm__ int32_t *)cachedLens;
    __gm__ Dtype *kDenseGm = (__gm__ Dtype *)kDenseCache;
    __gm__ Dtype *peDenseGm = (__gm__ Dtype *)peDenseCache;

    uint32_t indexCnt = DIV_ROUND_UP(indexTopK, CNT_ONE_LOOP);
    uint32_t totalTasks = batch * indexCnt;
    uint32_t curr = 0;
    uint32_t physBlock[CNT_ONE_LOOP];
    uint32_t rem[CNT_ONE_LOOP];
    uint32_t lastBatch = (uint32_t)-1;
    int32_t queryLen;
    int32_t cacheLen;
    int32_t totalLen;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID3);
    for (uint32_t task = block_idx; task < totalTasks; task += block_num) {
        uint32_t batch = task / indexCnt;
        uint32_t offsetInBatch = (task % indexCnt) * CNT_ONE_LOOP;
        int offset = task * CNT_ONE_LOOP;
        __gm__ int32_t *blockTableGm = blockTablesGm + batch * maxNumBlocks;
        if (batch != lastBatch) {
            queryLen = queryLensGm[batch];
            cacheLen = cachedLensGm[batch];
            totalLen = queryLen + cacheLen;
            lastBatch = batch;
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
        }
        int cnt = CNT_ONE_LOOP;
        if (offsetInBatch + cnt > totalLen) {
            cnt = totalLen - offsetInBatch;
        }
        if (cnt <= 0) {
            continue;
        }
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2 + curr);
        copy_gm_to_ubuf_align_b16(topks[curr], topkIndicesGm + offset, 0, 1, cnt * sizeof(uint32_t),
                                  0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
        for (int i = 0; i < cnt; i++) {
            uint32_t tok = topks[curr][i];
            uint32_t blockId = tok / blockSize;
            physBlock[i] = blockTableGm[blockId];
            rem[i] = tok % blockSize;
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2 + curr);
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);

        // GM -> UB
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);

        for (int i = 0; i < cnt; i++) {
            uint32_t srcOffset = physBlock[i] * blockSize + rem[i];
            CopyGmToUbufAligned(kBuf[curr] + i * kvLoraRank, kCacheGm + srcOffset * kvLoraRank,
                                kHeadBytes);
            CopyGmToUbufAligned(peBuf[curr] + i * ropeHeadDim, peCacheGm + srcOffset * ropeHeadDim,
                                peHeadBytes);
        }

        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);

        // UB -> GM
        uint32_t kDstOffset = offset * kvLoraRank;
        uint32_t peDstOffset = offset * ropeHeadDim;
        CopyUbufToGmAligned(kDenseGm + kDstOffset, kBuf[curr], kHeadBytes * cnt);
        CopyUbufToGmAligned(peDenseGm + peDstOffset, peBuf[curr], peHeadBytes * cnt);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);

        curr = 1 - curr;
    }
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

#define GATHER_SPARSE_KV_CACHE_FUNC_DEFINE(dtype)                                              \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_##dtype(                      \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,             \
        GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache, GM_ADDR peDenseCache,      \
        uint32_t batch, uint32_t indexTopK, uint32_t blockSize, uint32_t maxNumBlocks,         \
        uint32_t kvLoraRank, uint32_t ropeHeadDim)                                             \
    {                                                                                          \
        gather_sparse_kv_cache<dtype>(kCache, peCache, blockTables, topkIndices, queryLens,    \
                                      cachedLens, kDenseCache, peDenseCache, batch, indexTopK, \
                                      blockSize, maxNumBlocks, kvLoraRank, ropeHeadDim);       \
    }
#else
#define GATHER_SPARSE_KV_CACHE_FUNC_DEFINE(dtype)                                         \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_##dtype(                 \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,        \
        GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache, GM_ADDR peDenseCache, \
        uint32_t batch, uint32_t indexTopK, uint32_t blockSize, uint32_t maxNumBlocks,    \
        uint32_t kvLoraRank, uint32_t ropeHeadDim)                                        \
    {                                                                                     \
    }
#endif
