/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_param.h"
#include "kernel_operator.h"
#include "debug.h"

#define CNT_ONE_LOOP GATHER_SPARSE_KV_CACHE_CNT_ONE_LOOP
#define C8_CNT_ONE_LOOP 16
// Optimal GM->UB burst size (bytes) for the tiling gather; bursts must stay small
// enough to interleave with the copy-outs in flight on the other chunk buffers.
// The runtime burstCopySize parameter can only lower this.
#define GM_UB_BURST_SIZE_TILING 8192

#ifdef __DAV_C220_VEC__
template <typename OutDtype>
__aicore__ inline void dequant_c8_rows(__ubuf__ OutDtype *dst, __ubuf__ int8_t *input,
                                       __ubuf__ half *halfValues, __ubuf__ float *x, int32_t count,
                                       uint32_t kvLoraRank, uint32_t inputStride,
                                       uint32_t scaleOffset)
{
    constexpr uint32_t groupSize = 128;
    uint32_t groupNum = kvLoraRank / groupSize;

    pipe_barrier(PIPE_V);
    for (uint32_t group = 0; group < groupNum; ++group) {
        vconv_s82f16(halfValues + group * groupSize, input + group * groupSize, count, 1, 1,
                     kvLoraRank * sizeof(half) / BLOCK_SIZE, inputStride / BLOCK_SIZE);
    }
    pipe_barrier(PIPE_V);
    uint32_t repeats = count * kvLoraRank / VECTOR_MAX_NUM_OF_FP32;
    convert_input(x, halfValues, repeats);
    pipe_barrier(PIPE_V);
    for (int i = 0; i < count; ++i) {
        auto scales = reinterpret_cast<__ubuf__ float *>(
            input + i * ROUND_UP(groupNum * sizeof(float), BLOCK_SIZE) + scaleOffset);
        for (uint32_t group = 0; group < groupNum; ++group) {
            float s = scales[group];
            uint32_t valueOffset = i * kvLoraRank + group * groupSize;
            vmuls(x + valueOffset, x + valueOffset, s, 2, 1, 1, 8, 8);
        }
    }
    pipe_barrier(PIPE_V);
    convert_output(dst, x, repeats);
}

template <typename Dtype, typename OutDtype = Dtype>
__aicore__ __inline__ void gather_sparse_kv_cache(
    GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,
    GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,
    GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,
    uint32_t blockTableStride, uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kCacheStride = 0,
    uint32_t peCacheStride = 0, uint32_t compressRatio = 1,
    uint32_t burstCopySize = GM_UB_BURST_SIZE, GM_ADDR scaleCache = nullptr)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ Dtype *kCacheGm = (__gm__ Dtype *)kCache;
    __gm__ int32_t *blockTablesGm = (__gm__ int32_t *)blockTables;
    __gm__ int32_t *topkIndicesGm = (__gm__ int32_t *)topkIndices;
    __gm__ OutDtype *kDenseGm = (__gm__ OutDtype *)kDenseCache;
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE  // NOPE: no position embedding, i.e., only gather K
    __gm__ OutDtype *peCacheGm = (__gm__ OutDtype *)peCache;
    __gm__ OutDtype *peDenseGm = (__gm__ OutDtype *)peDenseCache;
#endif

    constexpr uint32_t tokensPerTask =
        std::is_same<Dtype, int8_t>::value ? C8_CNT_ONE_LOOP : CNT_ONE_LOOP;
    uint32_t perKBytes = kvLoraRank * sizeof(OutDtype);
    uint32_t kHeadBytes = perKBytes * tokensPerTask;
    uint32_t kHeadBytesPad = ROUND_UP(kHeadBytes, BLOCK_SIZE);
    uint32_t topkBytesPad = ROUND_UP(tokensPerTask * sizeof(uint32_t), BLOCK_SIZE);
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
    uint32_t perPeBytes = ropeHeadDim * sizeof(OutDtype);
    uint32_t peHeadBytes = perPeBytes * tokensPerTask;
    uint32_t peHeadBytesPad = ROUND_UP(peHeadBytes, BLOCK_SIZE);
#endif

    constexpr uint32_t groupSize = 128;
    uint32_t groupNum = kvLoraRank / groupSize;
    // Preserve the input UB reservation from the packed-cache layout.
    uint32_t inputBufBytes =
        tokensPerTask * ROUND_UP(kvLoraRank * sizeof(int8_t) + ropeHeadDim * sizeof(bfloat16_t) +
                                     groupNum * sizeof(float),
                                 VECTOR_REPEAT_BYTESIZE);
    uint32_t splitScaleOffset = tokensPerTask * kvLoraRank;
    uint32_t scaleStride = ROUND_UP(groupNum * sizeof(float), BLOCK_SIZE);

    // burstCopySize only confines the per-burst size (compile-time optimum
    // GM_UB_BURST_SIZE_TILING); the count uses the input K bytes per token.
    // K and PE cover the same run; a burst never crosses a chunk loop nor a cache
    // block; burstCopyCnt == 1 degrades to token-by-token copy, even for dense rows.
    uint32_t burstSize = MIN(MIN(burstCopySize, GM_UB_BURST_SIZE), GM_UB_BURST_SIZE_TILING);
    uint32_t burstCopyCnt =
        CLAMP(burstSize / (kvLoraRank * sizeof(Dtype)), 1u, MIN(tokensPerTask, blockSize));

    uint64_t off = 0;
    __ubuf__ OutDtype *kBufs[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    __ubuf__ uint32_t *topks[GATHER_SPARSE_KV_CACHE_BUF_NUM];
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
    __ubuf__ OutDtype *peBufs[GATHER_SPARSE_KV_CACHE_BUF_NUM];
#endif
    __ubuf__ int8_t *inputBuf[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    __ubuf__ half *halfBuf[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    __ubuf__ float *xBuf[GATHER_SPARSE_KV_CACHE_BUF_NUM];
    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        if constexpr (std::is_same<Dtype, int8_t>::value) {
            off = ROUND_UP(off, UB_BANKGROUP_ROW_SIZE);
            inputBuf[b] = reinterpret_cast<__ubuf__ int8_t *>(off);
            off += inputBufBytes;
            halfBuf[b] = reinterpret_cast<__ubuf__ half *>(off);
            off += tokensPerTask * kvLoraRank * sizeof(half);
            xBuf[b] = reinterpret_cast<__ubuf__ float *>(off);
            off += tokensPerTask * kvLoraRank * sizeof(float);
        }
        kBufs[b] = reinterpret_cast<__ubuf__ OutDtype *>((uintptr_t)off);
        off += kHeadBytesPad;
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
        peBufs[b] = reinterpret_cast<__ubuf__ OutDtype *>((uintptr_t)off);
        off += peHeadBytesPad;
#endif
        topks[b] = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += topkBytesPad;
    }
    if constexpr (std::is_same<Dtype, int8_t>::value) {
        assert(off + 3 * BLOCK_SIZE <= UB_SIZE);
    }
    uint32_t windowBytes = ROUND_DOWN((UB_SIZE - off) / 3u, BLOCK_SIZE);
    uint32_t maxBatchSize = windowBytes / sizeof(int32_t);
    __ubuf__ int32_t *queryStartLocUb = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    off += windowBytes;
    __ubuf__ int32_t *queryLensUb = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    off += windowBytes;
    __ubuf__ int32_t *cachedLensUb = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    off += windowBytes;
    if constexpr (!std::is_same<Dtype, int8_t>::value) {
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
        assert(off + BLOCK_SIZE <= UB_SIZE && perKBytes % BLOCK_SIZE == 0 && topK > 0 &&
               compressRatio > 0 && blockSize % CNT_ONE_LOOP == 0 && kvLoraRank >= ropeHeadDim &&
               perPeBytes % BLOCK_SIZE == 0);
#else
        assert(off + BLOCK_SIZE <= UB_SIZE && perKBytes % BLOCK_SIZE == 0 && topK > 0 &&
               compressRatio > 0 && blockSize % CNT_ONE_LOOP == 0);
#endif
    }
    uint32_t batchStart = 0, batchEnd = 0;  // end is exclusive

    const bool sequential = std::is_same<Dtype, int8_t>::value && topkIndices == nullptr;
    uint32_t indexCnt = DIV_ROUND_UP(topK, tokensPerTask);
    uint32_t curr = 0;
    uint32_t physBlock[tokensPerTask];
    uint32_t physInBlockOffset[tokensPerTask];
    int32_t queryStart, queryLen, cachedLen;
    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        if constexpr (std::is_same<Dtype, int8_t>::value) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + b);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4 + b);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID4 + b);
        } else {
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + b);     // topks[b]
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + b);  // kBufs[b]/peBufs[b]
        }
    }
    // TODO: Recheck C8 scheduling performance and token grouping after tail skipping.
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
        uint64_t tasks = (uint64_t)(sequential ? 1 : queryLen) * indexCnt;
        for (uint64_t idx = 0; idx < tasks;) {
            uint32_t offsetInQuery = idx / indexCnt;
            uint32_t offsetInTopk = (idx % indexCnt) * tokensPerTask;
            int32_t numCandidates;
            // +1: the current query token's own KV is a candidate (causal self-attention),
            // matching the indexer's per-query range MapPosToKv(cachedLen+offsetInQuery)+1.
            numCandidates =
                (cachedLen + (sequential ? queryLen : offsetInQuery + 1)) / compressRatio;
            bool denseRow = sequential || numCandidates <= topK;
            int32_t cnt = MIN((int32_t)tokensPerTask,
                              MIN(numCandidates, (int32_t)topK) - (int32_t)offsetInTopk);
            if (cnt <= 0) {
                idx = ROUND_UP(idx + 1, indexCnt);
                continue;
            }
            idx++;
            validIdx++;
            if (validIdx % block_num != block_idx) {  // assign task to vector core
                continue;
            }

            uint32_t inOffset = sequential ? curBatch : queryStart + offsetInQuery;
            uint64_t outputOffset = (uint64_t)inOffset * topK + offsetInTopk;
            if constexpr (std::is_same<Dtype, int8_t>::value) {
                // Inputs and outputs share the bank until its output DMA completes.
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4 + curr);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID4 + curr);
            }
            // Address computation (scalar core)
            __gm__ int32_t *blkTbGm = blockTablesGm + blockTableStride * curBatch;
            if (denseRow && !std::is_same<Dtype, int8_t>::value) {
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
                if (!denseRow) {
                    if constexpr (!std::is_same<Dtype, int8_t>::value) {
                        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);  // acquire `topks[curr]`
                    }
                    // the GM source is only 4B-aligned in general (topK * inOffset +
                    // offsetInTopk is an int32 element offset); CopyGmToUbufAligned would pick
                    // the 32B-block variant by size alone
                    copy_gm_to_ubuf_align_b16(topks[curr], topkIndicesGm + outputOffset, 0, 1,
                                              cnt * sizeof(int32_t), 0, 0, 0, 0);
                    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
                    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0 + curr);
                }
                for (int i = 0; i < cnt; i++) {
                    uint32_t tokPos = denseRow ? offsetInTopk + i : topks[curr][i];
                    physBlock[i] = blkTbGm[tokPos / blockSize];  // DCache reduces duplicate reads
                    physInBlockOffset[i] = tokPos % blockSize;
                }
                if constexpr (!std::is_same<Dtype, int8_t>::value) {
                    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + curr);  // release `topks[curr]`
                }
            }

            __ubuf__ Dtype *kInput;
            if constexpr (std::is_same<Dtype, int8_t>::value) {
                kInput = inputBuf[curr];
            } else {
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);  // acquire chunk buffers
                kInput = kBufs[curr];
            }
            // Burst copy-in: consecutive ascending/descending indices in the same block
            // are fetched as one run in ascending order; a descending run lands reversed,
            // which is fine — consumers treat the gathered cache as a set. No UB -> UB copies.
            for (int i = 0, iBurst; i < cnt; i += iBurst) {
                iBurst = 1;
                uint32_t blk = physBlock[i], blkOffset = physInBlockOffset[i];
                int dir = 0;  // +1: ascending run, -1: descending run, 0: single index
                if (i + 1 < cnt && physBlock[i + 1] == blk) {
                    if (physInBlockOffset[i + 1] == blkOffset + 1) {
                        dir = 1;
                    } else if (physInBlockOffset[i + 1] + 1 == blkOffset) {
                        dir = -1;
                    }
                }
                while (dir != 0 && i + iBurst < cnt && iBurst < burstCopyCnt &&
                       physBlock[i + iBurst] == blk &&
                       (int32_t)physInBlockOffset[i + iBurst] ==
                           (int32_t)blkOffset + dir * iBurst) {
                    iBurst++;
                }
                // Source token range: [srcTok, srcTok + iBurst) of `blk`, ascending;
                // `blk` is a physical block id, so its GM pitch is the cache's dim-0 stride
                uint32_t srcTok = dir >= 0 ? blkOffset : blkOffset + 1 - iBurst;
                CopyGmToUbufAligned(
                    kInput + i * kvLoraRank,
                    kCacheGm + (uint64_t)blk * MAX(kCacheStride, (uint64_t)blockSize * kvLoraRank) +
                        (uint64_t)srcTok * kvLoraRank,
                    iBurst * kvLoraRank * sizeof(Dtype));
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
                CopyGmToUbufAligned(
                    peBufs[curr] + i * ropeHeadDim,
                    peCacheGm +
                        (uint64_t)blk * MAX(peCacheStride, (uint64_t)blockSize * ropeHeadDim) +
                        (uint64_t)srcTok * ropeHeadDim,
                    iBurst * perPeBytes);
#endif
                if constexpr (std::is_same<Dtype, int8_t>::value) {
                    uint64_t slot = (uint64_t)blk * blockSize + srcTok;
                    copy_gm_to_ubuf_align_b32(inputBuf[curr] + splitScaleOffset + i * scaleStride,
                                              (__gm__ float *)scaleCache + slot * groupNum, 0,
                                              iBurst, groupNum * sizeof(float), 0, 0, 0, 0);
                }
            }
            if constexpr (std::is_same<Dtype, int8_t>::value) {
                auto input = inputBuf[curr];
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                dequant_c8_rows(kBufs[curr], input, halfBuf[curr], xBuf[curr], cnt, kvLoraRank,
                                kvLoraRank, splitScaleOffset);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            } else {
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
            }

            // Burst copy-out (UB -> GM): the whole chunk as one contiguous burst
            CopyUbufToGmAligned(kDenseGm + outputOffset * kvLoraRank, kBufs[curr], cnt * perKBytes);
#ifndef XLITE_GATHER_SPARSE_KV_CACHE_NOPE
            CopyUbufToGmAligned(peDenseGm + outputOffset * ropeHeadDim, peBufs[curr],
                                cnt * perPeBytes);
#endif
            if constexpr (std::is_same<Dtype, int8_t>::value) {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + curr);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4 + curr);
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID4 + curr);
            } else {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);  // release chunk buffers
            }

            curr = (curr + 1) % GATHER_SPARSE_KV_CACHE_BUF_NUM;
        }
    }

    for (uint32_t b = 0; b < GATHER_SPARSE_KV_CACHE_BUF_NUM; b++) {
        if constexpr (std::is_same<Dtype, int8_t>::value) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4 + b);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4 + b);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID4 + b);
        } else {
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0 + b);     // topks[b]
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + b);  // chunk buffers
        }
    }
}

#define GATHER_SPARSE_KV_CACHE_FUNC_DEFINE(dtype)                                                 \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_##dtype(                         \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,                \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,        \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,                  \
        uint32_t blockTableStride, uint32_t kvLoraRank, uint32_t ropeHeadDim,                     \
        uint32_t kCacheStride, uint32_t peCacheStride, uint32_t compressRatio,                    \
        uint32_t burstCopySize)                                                                   \
    {                                                                                             \
        gather_sparse_kv_cache<dtype>(kCache, peCache, blockTables, topkIndices, queryStartLoc,   \
                                      queryLens, cachedLens, kDenseCache, peDenseCache, batch,    \
                                      topK, blockSize, blockTableStride, kvLoraRank, ropeHeadDim, \
                                      kCacheStride, peCacheStride, compressRatio, burstCopySize); \
    }
#define GATHER_SPARSE_KV_CACHE_NOPE_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_nope_##dtype(                    \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,                \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache,        \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,                  \
        uint32_t blockTableStride, uint32_t kvLoraRank, uint32_t ropeHeadDim,                     \
        uint32_t kCacheStride, uint32_t peCacheStride, uint32_t compressRatio,                    \
        uint32_t burstCopySize)                                                                   \
    {                                                                                             \
        gather_sparse_kv_cache<dtype>(kCache, peCache, blockTables, topkIndices, queryStartLoc,   \
                                      queryLens, cachedLens, kDenseCache, peDenseCache, batch,    \
                                      topK, blockSize, blockTableStride, kvLoraRank, ropeHeadDim, \
                                      kCacheStride, peCacheStride, compressRatio, burstCopySize); \
    }
#else
#define GATHER_SPARSE_KV_CACHE_FUNC_DEFINE(dtype)                                          \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_##dtype(                  \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,         \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache, \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,           \
        uint32_t blockTableStride, uint32_t kvLoraRank, uint32_t ropeHeadDim,              \
        uint32_t kCacheStride, uint32_t peCacheStride, uint32_t compressRatio,             \
        uint32_t burstCopySize)                                                            \
    {                                                                                      \
    }
#define GATHER_SPARSE_KV_CACHE_NOPE_FUNC_DEFINE(dtype)                                     \
    extern "C" __global__ __aicore__ void gather_sparse_kv_cache_nope_##dtype(             \
        GM_ADDR kCache, GM_ADDR peCache, GM_ADDR blockTables, GM_ADDR topkIndices,         \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR kDenseCache, \
        GM_ADDR peDenseCache, uint32_t batch, uint32_t topK, uint32_t blockSize,           \
        uint32_t blockTableStride, uint32_t kvLoraRank, uint32_t ropeHeadDim,              \
        uint32_t kCacheStride, uint32_t peCacheStride, uint32_t compressRatio,             \
        uint32_t burstCopySize)                                                            \
    {                                                                                      \
    }
#endif
