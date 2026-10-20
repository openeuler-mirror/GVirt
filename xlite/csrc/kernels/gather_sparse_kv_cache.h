/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_param.h"
#include "kernel_operator.h"
#include "debug.h"

#define CNT_ONE_LOOP GATHER_SPARSE_KV_CACHE_CNT_ONE_LOOP
// Optimal GM->UB burst size (bytes) for the tiling gather; bursts must stay small
// enough to interleave with the copy-outs in flight on the other chunk buffers.
// The runtime burstCopySize parameter can only lower this.
#define GM_UB_BURST_SIZE_TILING 8192

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ __inline__ void gather_sparse_kv_cache(
    GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,
    GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,
    GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize, uint32_t maxNumBlocks,
    uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride = 0,
    uint32_t peCacheStride = 0, uint32_t compressRatio = 1,
    uint32_t burstCopySize = GM_UB_BURST_SIZE)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ Dtype *kCacheGm = (__gm__ Dtype *)kCache;
    __gm__ int32_t *blockTablesGm = (__gm__ int32_t *)blockTables;
    __gm__ int32_t *topkIndicesGm = (__gm__ int32_t *)topkIndices;
    __gm__ Dtype *kDenseGm = (__gm__ Dtype *)kDenseCache;
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE  // NOPE: no position embedding, i.e., only gather K
    __gm__ Dtype *peCacheGm = (__gm__ Dtype *)peCache;
    __gm__ Dtype *peDenseGm = (__gm__ Dtype *)peDenseCache;
#endif

    uint32_t perKBytes = kvLoraRank * sizeof(Dtype);
    uint32_t kHeadBytes = perKBytes * CNT_ONE_LOOP;
    uint32_t kHeadBytesPad = ROUND_UP(kHeadBytes, BLOCK_SIZE);
    uint32_t topkBytesPad = ROUND_UP(CNT_ONE_LOOP * sizeof(uint32_t), BLOCK_SIZE);
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
    uint32_t perPeBytes = ropeHeadDim * sizeof(Dtype);
    uint32_t peHeadBytes = perPeBytes * CNT_ONE_LOOP;
    uint32_t peHeadBytesPad = ROUND_UP(peHeadBytes, BLOCK_SIZE);
#endif

    // burstCopySize only confines the per-burst size (compile-time optimum
    // GM_UB_BURST_SIZE_TILING); the count derives from the K channel so K and PE
    // bursts cover the same run; a burst never crosses a chunk loop nor a cache
    // block; burstCopyCnt == 1 degrades to token-by-token copy, even for dense rows.
    uint32_t burstSize = MIN(MIN(burstCopySize, GM_UB_BURST_SIZE), GM_UB_BURST_SIZE_TILING);
    uint32_t burstCopyCnt = CLAMP(burstSize / perKBytes, 1u, MIN(CNT_ONE_LOOP, blockSize));

    uint64_t off = 0;
    __ubuf__ Dtype *kBufs[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    __ubuf__ uint32_t *topks[GATHER_SPARSE_KV_CACHE_BUF_NUM];
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
    __ubuf__ Dtype *peBufs[GATHER_SPARSE_KV_CACHE_BUF_NUM];
#endif
    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        kBufs[b] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kHeadBytesPad;
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
        peBufs[b] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += peHeadBytesPad;
#endif
        topks[b] = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += topkBytesPad;
    }
    uint32_t windowBytes = ROUND_DOWN((UB_SIZE - off) / 3u, BLOCK_SIZE);
    uint32_t maxBatchSize = windowBytes / sizeof(int32_t);
    __ubuf__ int32_t *queryStartLocUb = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    off += windowBytes;
    __ubuf__ int32_t *queryLensUb = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    off += windowBytes;
    __ubuf__ int32_t *cachedLensUb = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    off += windowBytes;
    uint32_t batchStart = 0, batchEnd = 0;  // end is exclusive
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
    assert(off + BLOCK_SIZE <= UB_SIZE && perKBytes % BLOCK_SIZE == 0 && topK > 0 &&
           compressRatio > 0 && blockSize % CNT_ONE_LOOP == 0 && kvLoraRank >= ropeHeadDim &&
           perPeBytes % BLOCK_SIZE == 0);
#else
    assert(off + BLOCK_SIZE <= UB_SIZE && perKBytes % BLOCK_SIZE == 0 && topK > 0 &&
           compressRatio > 0 && blockSize % CNT_ONE_LOOP == 0);
#endif

    uint32_t indexCnt = DIV_ROUND_UP(topK, CNT_ONE_LOOP);
    uint32_t curr = 0;
    uint32_t physBlock[CNT_ONE_LOOP];
    uint32_t physInBlockOffset[CNT_ONE_LOOP];
    int32_t queryStart, queryLen, cachedLen;
    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + b);     // topks[b]
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + b);  // kBufs[b]/peBufs[b]
    }
    uint32_t validIdx = 0;
    for (uint32_t curBatch = 0; curBatch < batch; curBatch++) {
        if (curBatch >= batchEnd) {
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
            batchStart = curBatch;
            batchEnd = MIN(batchStart + maxBatchSize, batch);
            CopyGmToUbufAligned(queryStartLocUb, (__gm__ int32_t *)queryStartLoc + batchStart,
                                (batchEnd - batchStart) * sizeof(int32_t));
            CopyGmToUbufAligned(queryLensUb, (__gm__ int32_t *)queryLens + batchStart,
                                (batchEnd - batchStart) * sizeof(int32_t));
            CopyGmToUbufAligned(cachedLensUb, (__gm__ int32_t *)cachedLens + batchStart,
                                (batchEnd - batchStart) * sizeof(int32_t));
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID4);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID4);
        }
        queryStart = queryStartLocUb[curBatch - batchStart];
        queryLen = queryLensUb[curBatch - batchStart];
        cachedLen = cachedLensUb[curBatch - batchStart];
        uint32_t tasks = queryLen * indexCnt;
        for (uint32_t idx = 0; idx < tasks;) {
            uint32_t offsetInQuery = idx / indexCnt;
            uint32_t offsetInTopk = (idx % indexCnt) * CNT_ONE_LOOP;
            // +1: the current query token's own KV is a candidate (causal self-attention),
            // matching the indexer's per-query range MapPosToKv(cachedLen+offsetInQuery)+1.
            int32_t numCandidates = (cachedLen + offsetInQuery + 1) / compressRatio;
            bool denseRow = numCandidates <= topK;
            int32_t cnt =
                MIN(CNT_ONE_LOOP, MIN(numCandidates, (int32_t)topK) - (int32_t)offsetInTopk);
            if (cnt <= 0) {
                idx = ROUND_UP(idx + 1, indexCnt);
                continue;
            }
            idx++;
            validIdx++;
            if (validIdx % block_num != block_idx) {  // assign task to vector core
                continue;
            }

            uint32_t inOffset = queryStart + offsetInQuery;
            // Address computation (scalar core)
            __gm__ int32_t *blkTbGm = blockTablesGm + maxNumBlocks * curBatch;
            if (denseRow) {
                // Identity template: a dense row's topk result is [0, pos] + dummies, so
                // slot i maps to token offsetInTopk + i; topkIndices is not read, and
                // blockSize % CNT_ONE_LOOP keeps the whole chunk in one physical block.
                uint32_t blk = blkTbGm[offsetInTopk / blockSize];
                uint32_t base = offsetInTopk % blockSize;
                for (int i = 0; i < cnt; i++) {
                    physBlock[i] = blk;
                    physInBlockOffset[i] = base + i;
                }
            } else {
                wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);  // acquire `topks[curr]`
                // the GM source is only 4B-aligned in general (topK * inOffset + offsetInTopk
                // is an int32 element offset); CopyGmToUbufAligned would pick the 32B-block
                // variant by size alone
                copy_gm_to_ubuf_align_b16(topks[curr],
                                          topkIndicesGm + topK * inOffset + offsetInTopk, 0, 1,
                                          cnt * sizeof(int32_t), 0, 0, 0, 0);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
                for (int i = 0; i < cnt; i++) {
                    uint32_t tokPos = topks[curr][i];
                    physBlock[i] = blkTbGm[tokPos / blockSize];  // DCache reduces duplicate reads
                    physInBlockOffset[i] = tokPos % blockSize;
                }
                set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);  // release `topks[curr]`
            }

            // Burst copy-in: consecutive ascending/descending indices in the same block
            // are fetched as one run in ascending order; a descending run lands reversed,
            // which is fine — consumers treat the gathered cache as a set. No UB -> UB copies.
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);  // acquire chunk buffers
            for (int i = 0, iBurst; i < cnt; i += iBurst) {
                uint32_t blk = physBlock[i], blkOffset = physInBlockOffset[i];
                int dir = 0;  // +1: ascending run, -1: descending run, 0: single index
                if (i + 1 < cnt && physBlock[i + 1] == blk) {
                    if (physInBlockOffset[i + 1] == blkOffset + 1) {
                        dir = 1;
                    } else if (physInBlockOffset[i + 1] + 1 == blkOffset) {
                        dir = -1;
                    }
                }
                iBurst = 1;
                while (dir != 0 && i + iBurst < cnt && iBurst < burstCopyCnt &&
                       physBlock[i + iBurst] == blk &&
                       (int32_t)physInBlockOffset[i + iBurst] ==
                           (int32_t)blkOffset + dir * iBurst) {
                    iBurst++;
                }
                // Source token range: [srcTok, srcTok + iBurst) of `blk`, ascending;
                // `blk` is a physical block id, so its GM pitch is the cache's dim-0 stride
                uint32_t srcTok = dir >= 0 ? blkOffset : blkOffset + 1 - iBurst;
                CopyGmToUbufAligned(kBufs[curr] + i * kvLoraRank,
                                    kCacheGm +
                                        (uint64_t)blk * MAX(kCacheStride, blockSize * kvLoraRank) +
                                        (uint64_t)srcTok * kvLoraRank,
                                    iBurst * perKBytes);
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
                CopyGmToUbufAligned(
                    peBufs[curr] + i * ropeHeadDim,
                    peCacheGm + (uint64_t)blk * MAX(peCacheStride, blockSize * ropeHeadDim) +
                        (uint64_t)srcTok * ropeHeadDim,
                    iBurst * perPeBytes);
#endif
            }
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);

            // Burst copy-out (UB -> GM): the whole chunk as one contiguous burst
            CopyUbufToGmAligned(kDenseGm + (uint64_t)(inOffset * topK + offsetInTopk) * kvLoraRank,
                                kBufs[curr], cnt * perKBytes);
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
            CopyUbufToGmAligned(
                peDenseGm + (uint64_t)(inOffset * topK + offsetInTopk) * ropeHeadDim, peBufs[curr],
                cnt * perPeBytes);
#endif
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);  // release chunk buffers

            curr = (curr + 1) % GATHER_SPARSE_KV_CACHE_BUF_NUM;
        }
    }

    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + b);     // topks[b]
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + b);  // chunk buffers
    }
}

#define GATHER_SPARSE_KV_CACHE_FUNC_DEFINE(dtype)                                                 \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_##dtype(                         \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,                \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,        \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,                  \
        uint32_t maxNumBlocks, uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride,  \
        uint32_t peCacheStride, uint32_t compressRatio, uint32_t burstCopySize)                   \
    {                                                                                             \
        gather_sparse_kv_cache<dtype>(kCache, peCache, blockTables, topkIndices, queryStartLoc,   \
                                      queryLens, cachedLens, kDenseCache, peDenseCache, batch,    \
                                      topK, blockSize, maxNumBlocks, kvLoraRank, ropeHeadDim,     \
                                      kCacheStride, peCacheStride, compressRatio, burstCopySize); \
    }
#define GATHER_SPARSE_KV_CACHE_NOPE_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_nope_##dtype(                    \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,                \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,        \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,                  \
        uint32_t maxNumBlocks, uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride,  \
        uint32_t peCacheStride, uint32_t compressRatio, uint32_t burstCopySize)                   \
    {                                                                                             \
        gather_sparse_kv_cache<dtype>(kCache, peCache, blockTables, topkIndices, queryStartLoc,   \
                                      queryLens, cachedLens, kDenseCache, peDenseCache, batch,    \
                                      topK, blockSize, maxNumBlocks, kvLoraRank, ropeHeadDim,     \
                                      kCacheStride, peCacheStride, compressRatio, burstCopySize); \
    }
#else
#define GATHER_SPARSE_KV_CACHE_FUNC_DEFINE(dtype)                                                \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_##dtype(                        \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,               \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,       \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,                 \
        uint32_t maxNumBlocks, uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride, \
        uint32_t peCacheStride, uint32_t compressRatio, uint32_t burstCopySize)                  \
    {                                                                                            \
    }
#define GATHER_SPARSE_KV_CACHE_NOPE_FUNC_DEFINE(dtype)                                           \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_nope_##dtype(                   \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,               \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,       \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,                 \
        uint32_t maxNumBlocks, uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride, \
        uint32_t peCacheStride, uint32_t compressRatio, uint32_t burstCopySize)                  \
    {                                                                                            \
    }
#endif
