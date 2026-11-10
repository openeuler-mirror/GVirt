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
#define GM_UB_BURST_SIZE_TILING 8192

#ifdef __DAV_C220_VEC__
// groupCachedLens gives the causal prefix length before each query; groupRequestIds maps
// that query to a request-level blockTables row. Gather each query's Top-K KV
// into a dense row, shared by all of its attention heads.
template <typename Dtype>
__aicore__ __inline__ void gather_sparse_kv_cache_per_query(
    GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,
    GM_ADDR groupCachedLens, GM_ADDR groupRequestIds, GM_ADDR kDenseCache, GM_ADDR peDenseCache,
    uint32_t groupLen, uint32_t topK, uint32_t blockSize, uint32_t blockTableStride,
    uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride, uint32_t peCacheStride)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ Dtype *kCacheGm = (__gm__ Dtype *)kCache;
    __gm__ int32_t *blockTablesGm = (__gm__ int32_t *)blockTables;
    __gm__ int32_t *topkIndicesGm = (__gm__ int32_t *)topkIndices;
    __gm__ Dtype *kDenseGm = (__gm__ Dtype *)kDenseCache;
    // The request map is regenerated into the same workspace for each group.
    for (uint32_t i = 0; i < groupLen; i += BLOCK_SIZE / sizeof(int32_t)) {
        DataCacheCleanAndInvalid((__gm__ int32_t *)groupRequestIds + i);
    }
    __gm__ Dtype *peCacheGm = (__gm__ Dtype *)peCache;
    __gm__ Dtype *peDenseGm = (__gm__ Dtype *)peDenseCache;

    uint32_t perKBytes = kvLoraRank * sizeof(Dtype);
    uint32_t kHeadBytes = perKBytes * CNT_ONE_LOOP;
    uint32_t kHeadBytesPad = ROUND_UP(kHeadBytes, BLOCK_SIZE);
    uint32_t topkBytesPad = ROUND_UP(CNT_ONE_LOOP * sizeof(uint32_t), BLOCK_SIZE);
    uint32_t perPeBytes = ropeHeadDim * sizeof(Dtype);
    uint32_t peHeadBytes = perPeBytes * CNT_ONE_LOOP;
    uint32_t peHeadBytesPad = ROUND_UP(peHeadBytes, BLOCK_SIZE);

    // Coalesce adjacent indices within one cache block, keeping bursts small
    // enough to overlap with writes from the other chunk buffers.
    uint32_t burstSize = MIN(GM_UB_BURST_SIZE, GM_UB_BURST_SIZE_TILING);
    uint32_t burstCopyCnt = CLAMP(burstSize / perKBytes, 1u, MIN(CNT_ONE_LOOP, blockSize));

    uint64_t off = 0;
    __ubuf__ Dtype *kBufs[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    __ubuf__ uint32_t *topkIndicesUb[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    __ubuf__ Dtype *peBufs[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        kBufs[b] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kHeadBytesPad;
        peBufs[b] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += peHeadBytesPad;
        topkIndicesUb[b] = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += topkBytesPad;
    }
    assert(off + BLOCK_SIZE <= UB_SIZE && perKBytes % BLOCK_SIZE == 0 && topK > 0 &&
           blockSize % CNT_ONE_LOOP == 0 && kvLoraRank >= ropeHeadDim &&
           perPeBytes % BLOCK_SIZE == 0);
    uint32_t queryWindowCapacity = (UB_SIZE - off) / sizeof(int32_t);
    __ubuf__ int32_t *groupCachedLensUb = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    uint32_t queryWindowBegin = 0, queryWindowEnd = 0;

    uint32_t topkChunks = DIV_ROUND_UP(topK, CNT_ONE_LOOP);
    uint32_t curr = 0;
    uint32_t physBlock[CNT_ONE_LOOP];
    uint32_t physInBlockOffset[CNT_ONE_LOOP];
    int32_t cachedLen = 0;
    uint32_t lastQuery = groupLen;
    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + b);     // topkIndicesUb[b]
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + b);  // kBufs[b]/peBufs[b]
    }
    // Assign query/KV chunks directly to vector cores, including padding chunks.
    for (uint32_t task = block_idx; task < groupLen * topkChunks; task += block_num) {
        uint32_t query = task / topkChunks;
        uint32_t offsetInTopk = (task % topkChunks) * CNT_ONE_LOOP;
        if (query != lastQuery) {
            if (query >= queryWindowEnd) {
                set_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
                wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID4);
                queryWindowBegin = query;
                queryWindowEnd = MIN(queryWindowBegin + queryWindowCapacity, groupLen);
                CopyGmToUbufAligned(groupCachedLensUb,
                                    (__gm__ int32_t *)groupCachedLens + queryWindowBegin,
                                    (queryWindowEnd - queryWindowBegin) * sizeof(int32_t));
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID4);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID4);
            }
            cachedLen = groupCachedLensUb[query - queryWindowBegin];
            lastQuery = query;
        }
        // Metadata already includes preceding tokens within this query's request.
        int32_t numCandidates = cachedLen + 1;
        bool denseRow = numCandidates <= topK;
        int32_t chunkLen =
            MAX(0, MIN(CNT_ONE_LOOP, MIN(numCandidates, (int32_t)topK) - (int32_t)offsetInTopk));
        uint32_t request = ((__gm__ int32_t *)groupRequestIds)[query];
        __gm__ int32_t *blkTbGm = blockTablesGm + uint64_t(blockTableStride) * request;
        if (denseRow && chunkLen > 0) {
            // Identity template: a dense row's topk result is [0, pos] + dummies, so
            // slot i maps to token offsetInTopk + i; topkIndices is not read, and
            // blockSize % CNT_ONE_LOOP keeps the whole chunk in one physical block.
            uint32_t blk = blkTbGm[offsetInTopk / blockSize];
            uint32_t base = offsetInTopk % blockSize;
            for (int i = 0; i < chunkLen; i++) {
                physBlock[i] = blk;
                physInBlockOffset[i] = base + i;
            }
        } else if (chunkLen > 0) {
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);  // acquire `topkIndicesUb[curr]`
            // the GM source is only 4B-aligned in general (topK * query + offsetInTopk
            // is an int32 element offset); CopyGmToUbufAligned would pick the 32B-block
            // variant by size alone
            copy_gm_to_ubuf_align_b16(topkIndicesUb[curr],
                                      topkIndicesGm + topK * query + offsetInTopk, 0, 1,
                                      chunkLen * sizeof(int32_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
            for (int i = 0; i < chunkLen; i++) {
                uint32_t tokPos = topkIndicesUb[curr][i];
                physBlock[i] = blkTbGm[tokPos / blockSize];  // DCache reduces duplicate reads
                physInBlockOffset[i] = tokPos % blockSize;
            }
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);  // release `topkIndicesUb[curr]`
        }

        // Burst copy-in: consecutive ascending/descending indices in the same block
        // are fetched as one run in ascending order; a descending run lands reversed,
        // which is fine — consumers treat the gathered cache as a set. No UB -> UB copies.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);  // acquire chunk buffers
        if (chunkLen < CNT_ONE_LOOP) {
            // SV may read aligned padding. Keep it finite without clearing
            // the full expanded KV buffer before every group.
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
            vector_dup(kBufs[curr], Dtype(0), kHeadBytesPad / VECTOR_MAX_BYTESIZE, 1, 1, 8, 0);
            vector_dup(peBufs[curr], Dtype(0), peHeadBytesPad / VECTOR_MAX_BYTESIZE, 1, 1, 8, 0);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        }
        for (int i = 0, iBurst; i < chunkLen; i += iBurst) {
            uint32_t blk = physBlock[i], blkOffset = physInBlockOffset[i];
            int dir = 0;  // +1: ascending run, -1: descending run, 0: single index
            if (i + 1 < chunkLen && physBlock[i + 1] == blk) {
                if (physInBlockOffset[i + 1] == blkOffset + 1) {
                    dir = 1;
                } else if (physInBlockOffset[i + 1] + 1 == blkOffset) {
                    dir = -1;
                }
            }
            iBurst = 1;
            while (dir != 0 && i + iBurst < chunkLen && iBurst < burstCopyCnt &&
                   physBlock[i + iBurst] == blk &&
                   (int32_t)physInBlockOffset[i + iBurst] == (int32_t)blkOffset + dir * iBurst) {
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
            CopyGmToUbufAligned(peBufs[curr] + i * ropeHeadDim,
                                peCacheGm +
                                    (uint64_t)blk * MAX(peCacheStride, blockSize * ropeHeadDim) +
                                    (uint64_t)srcTok * ropeHeadDim,
                                iBurst * perPeBytes);
        }
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);

        // Burst copy-out (UB -> GM): the whole chunk as one contiguous burst
        uint32_t outCnt = MIN(CNT_ONE_LOOP, topK - offsetInTopk);
        CopyUbufToGmAligned(kDenseGm + (uint64_t)(query * topK + offsetInTopk) * kvLoraRank,
                            kBufs[curr], outCnt * perKBytes);
        CopyUbufToGmAligned(peDenseGm + (uint64_t)(query * topK + offsetInTopk) * ropeHeadDim,
                            peBufs[curr], outCnt * perPeBytes);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);  // release chunk buffers

        curr = (curr + 1) % GATHER_SPARSE_KV_CACHE_BUF_NUM;
    }

    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + b);     // topkIndicesUb[b]
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + b);  // chunk buffers
    }
}

#endif
