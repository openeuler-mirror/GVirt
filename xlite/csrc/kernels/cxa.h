/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_param.h"
#include "kernel_operator.h"
// #define XLITE_KERNEL_DEBUG
#include "debug.h"
#include "softmax_attn_aiv.h"
#include "cxa_aic_helper.h"

// CXA (C4A and C128A) attention kernel framework.
//
// Unified kernel covering two compress-KV cache layouts (dense flag); the SWA
// cache is always paged and walked via swaBlockTables in both modes:
//   - sparse: paged compressKCache walked via compressBlockTables, with
//     optional top-k token selection (DSA) driven by topkIndices.
//   - dense : per-batch contiguous compressKCache; batch b occupies compressed
//     tokens [b * maxSeqLen, (b + 1) * maxSeqLen) with maxSeqLen == indexTopK,
//     and compressBlockTables is unused (block id == logical index). The
//     compress segment is attended causally without top-k masking, clamped to
//     maxSeqLen.
//
// Inputs:
//   q:    [totalQ, nLocalHeads, headDim]    query
//   swaKCache:    [block, swaBlockSize, headDim]    sliding-window KV cache
//   compressKCache:    [block, compressBlockSize, headDim]    compressed KV cache
//   swaBlockTables:    [batch, swaMaxNumBlocks] int32    block table for sliding-window cache
//   compressBlockTables:    [batch, compressMaxNumBlocks] int32    block table for compressed cache
//   attnSink:    [nHeads] fp32    per-head attention sink bias
//   scores:    [aicNum * XLITE_MAX_M0 * 2, swaSegWidth + kvSize]    workspace for QK scores
//              where swaSegWidth = windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B (0 when
//              windowSize is 0) is a fixed host/device contract (see Init).
//   queryStartLoc:    [batch] int32    cumulative query start offsets
//   lens:    [batch] int32    query lengths per batch
//   cachedLens:    [batch] int32    cached lengths per batch
//   topkIndices:    [totalQ, indexTopK] int32 (optional)    top-k indices in compressed cache
// Outputs:
//   output:    [totalQ, nLocalHeads, headDim]     attention output
template <typename Dtype>
class CXA
{
public:
    __aicore__ inline CXA()
    {
    }

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR swaKCache, GM_ADDR compressKCache,
                                GM_ADDR swaBlockTables, GM_ADDR compressBlockTables,
                                uint32_t swaBlockSize, uint32_t compressBlockSize,
                                uint32_t swaMaxNumBlocks, uint32_t compressMaxNumBlocks,
                                GM_ADDR attnSink, GM_ADDR scores, GM_ADDR output, uint32_t batch,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                uint32_t nHeads, uint32_t headDim, float scale, uint32_t windowSize,
                                uint32_t kvSize, uint32_t compressRatio, uint32_t indexTopK,
                                GM_ADDR topkIndices, uint32_t dense)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->q.SetGlobalBuffer((__gm__ Dtype *)q);
        this->swaKCache.SetGlobalBuffer((__gm__ Dtype *)swaKCache);
        this->compressKCache.SetGlobalBuffer((__gm__ Dtype *)compressKCache);
        this->output.SetGlobalBuffer((__gm__ Dtype *)output);

        this->swaBlockTables = (__gm__ int32_t *)swaBlockTables;
        this->compressBlockTables = (__gm__ int32_t *)compressBlockTables;
        this->attnSink = (__gm__ float *)attnSink;
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->topkIndices = (__gm__ int32_t *)topkIndices;

        this->batch = batch;
        this->nHeads = nHeads;
        this->headDim = headDim;
        this->scale = scale;
        this->compressRatio = compressRatio;
        this->dense = dense != 0;
        // top-k masking is a sparse-mode feature: in dense mode the compress cache
        // is attended causally, so topkIndices is ignored.
        this->indexTopK = (!this->dense && topkIndices != nullptr) ? indexTopK : 0;
        this->maxSeqLen = indexTopK;
        this->swaMaxNumBlocks = swaMaxNumBlocks;
        this->compressMaxNumBlocks = this->dense ? 0 : compressMaxNumBlocks;
        this->windowSize = windowSize;
        // the causal-window union of one query tile is windowSize + XLITE_MAX_M0 wide;
        // +K_BLOCK_SIZE_2B absorbs the windowStart round-down lead-in.
        this->swaSegWidth = windowSize == 0 ? 0 : windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B;
        this->qkStride = swaSegWidth + kvSize;

        this->scores[0].SetGlobalBuffer((__gm__ Dtype *)scores +
                                        block_idx * XLITE_MAX_M0 * qkStride);
        this->scores[1].SetGlobalBuffer((__gm__ Dtype *)scores +
                                        block_idx * XLITE_MAX_M0 * qkStride +
                                        block_num * XLITE_MAX_M0 * qkStride);

        this->svk0 =
            aicHelper.Init(nHeads, headDim, swaBlockSize, this->dense ? 0 : compressBlockSize,
                           windowSize, compressRatio, qkStride, swaSegWidth, this->dense);
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskOffset, lastQueryTaskLen, last, lastMhOffset, lastCalcLen;
        int lastAbsQueryStart = 0;
        __gm__ uint32_t *lastSwaBt;
        __gm__ uint32_t *lastCompressBt;
        GlobalTensor<Dtype> lastCompressKCache;

        int needDoSV = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int coreOffset = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *swaBt =
                (__gm__ uint32_t *)((uint64_t)swaBlockTables +
                                    batchIdx * swaMaxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            // per-batch compress KV view: dense mode takes a subview of the contiguous
            // cache (batch b starts at b * maxSeqLen compressed tokens, maxSeqLen ==
            // indexTopK); sparse mode walks the full paged cache through this batch's
            // block table.
            GlobalTensor<Dtype> batchCompressKCache = compressKCache;
            __gm__ uint32_t *compressBt = nullptr;
            if (dense) {
                batchCompressKCache = compressKCache[batchIdx * maxSeqLen * headDim];
            } else {
                compressBt =
                    (__gm__ uint32_t *)((uint64_t)compressBlockTables +
                                        batchIdx * compressMaxNumBlocks * sizeof(uint32_t));
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryTileSize = m0 / nHeads;
            if (queryTileSize == 0) {
                queryTileSize = m0;
            }
            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum;
            int firstCore = (GetBlockIdx() + GetBlockNum() - coreOffset) % GetBlockNum();
            for (int idx = firstCore; idx < taskNum; idx += GetBlockNum()) {
                int queryIdx = idx;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t absQueryStart = cachedLen + queryTaskStart;
                uint32_t calcLen = absQueryStart + queryTaskLen;
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                // do queryIdx & (0, nHeads)'s QK
                uint32_t mhOffset = queryTaskOffset * nHeads * headDim;

                dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u)}"
                           " use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           nHeads, curr);
                aicHelper.RunAicQK(q[mhOffset], swaKCache, batchCompressKCache, swaBt, compressBt,
                                   absQueryStart, queryTaskLen, 0, calcLen, calcLen, maxSeqLen,
                                   scores[curr]);
                ffts_cross_core_sync(PIPE_FIX, config);

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(1);
                    // do softmax * V
                    dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u)}"
                               " use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, nHeads, last);
                    aicHelper.RunAicSV(scores[last], swaKCache, lastCompressKCache, lastSwaBt,
                                       lastCompressBt, lastAbsQueryStart, lastQueryTaskLen, 0,
                                       lastCalcLen, lastCalcLen, maxSeqLen, output[lastMhOffset]);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastMhOffset = mhOffset;
                lastQueryTaskLen = queryTaskLen;
                lastSwaBt = swaBt;
                lastCompressBt = compressBt;
                lastCompressKCache = batchCompressKCache;
                lastCalcLen = calcLen;
                lastAbsQueryStart = absQueryStart;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            coreOffset = (coreOffset + taskNum) % GetBlockNum();
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(1);
            dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u)}"
                       " use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, nHeads, last);
            aicHelper.RunAicSV(scores[last], swaKCache, lastCompressKCache, lastSwaBt,
                               lastCompressBt, lastAbsQueryStart, lastQueryTaskLen, 0, lastCalcLen,
                               lastCalcLen, maxSeqLen, output[lastMhOffset]);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t flagIdx = 1;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int coreOffset = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryTileSize = m0 / nHeads;
            if (queryTileSize == 0) {
                queryTileSize = m0;
            }
            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum;
            int firstCore = (block_idx + block_num - coreOffset) % block_num;
            for (int idx = firstCore; idx < taskNum; idx += block_num) {
                int queryIdx = idx;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                int nWork = queryTaskLen * nHeads;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                uint32_t subIdx = get_subblockid();
                int nWorkStart = subIdx * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t qkOffset = nWorkStart * qkStride;
                uint32_t calcSoftmaxLen = cachedLen + queryTaskStart + 1;
                // winCalcLen: exclusive end of the causally visible SWA columns for this
                // tile's FIRST query row, in scores columns. scores column 0 = abs pos
                // alignStart = ROUND_DOWN(windowStart, kBlockSize) (see RunAicQK), so
                // winCalcLen = calcSoftmaxLen - alignStart; when the window is sliding
                // this exceeds windowSize by the lead-in (< 16).
                int winStart = calcSoftmaxLen > (int)windowSize
                                   ? (int)calcSoftmaxLen - (int)windowSize
                                   : 0;  // this tile's causal window start (abs coords)
                uint32_t winCalcLen = calcSoftmaxLen - ROUND_DOWN(winStart, K_BLOCK_SIZE_2B);
                // compress-segment causal length in compressed tokens, clamped to the
                // dense cache size maxSeqLen (= indexTopK) in dense mode; the SWA
                // segment keeps its own unclamped causal window semantics.
                uint32_t ncCalcLen = compressRatio == 0 ? 0 : calcLen / compressRatio;
                if (dense && ncCalcLen > maxSeqLen) {
                    ncCalcLen = maxSeqLen;
                }
                uint32_t outN =
                    swaSegWidth + (compressRatio == 0 ? 0 : ROUND_UP(ncCalcLen, 4 * svk0));
                if (outN > qkStride) {
                    outN = qkStride;
                }

                // wait aic qk done
                wait_flag_dev(0);

                // do softmax
                int dbgBlockIdx = block_idx;
                dbg_printf("block%d subblock%u: {batch %d, query [%u - %u) "
                           "query x head group [%u - "
                           "%u)} calcSoftmaxLen %u, off %u, stride %u, outN %u, use %d temp buf: "
                           "SOFTMAX\n",
                           dbgBlockIdx, subIdx, batchIdx, queryTaskOffset,
                           queryTaskOffset + queryTaskLen, nWorkStart, nWorkStart + nWorkCurCore,
                           calcSoftmaxLen, nWorkStart, nHeads, outN, curr);
                if (indexTopK == 0) {
                    RunAivSoftmax((__gm__ Dtype *)scores[curr][qkOffset].GetPhyAddr(),
                                  m0 > (XLITE_MAX_M0 - 4)
                                      ? 0
                                      : (__gm__ float *)scores[curr][(m0 + subIdx * 2) * qkStride]
                                            .GetPhyAddr(),
                                  nWorkCurCore, qkStride, calcSoftmaxLen, outN, true, nWorkStart,
                                  nHeads, true, scale, 0, 0, nullptr, windowSize, winCalcLen,
                                  compressRatio == 0 ? 1 : compressRatio, attnSink, swaSegWidth,
                                  dense && compressRatio != 0 ? maxSeqLen : 0);
                } else {
                    RunAivSoftmaxPingPong(
                        (__gm__ Dtype *)scores[curr][qkOffset].GetPhyAddr(), nWorkCurCore, qkStride,
                        calcSoftmaxLen, outN, true, nWorkStart, nHeads, nullptr, nullptr, true,
                        scale, 0, calcLen > indexTopK ? indexTopK : 0,
                        (calcLen > indexTopK && indexTopK > 0)
                            ? topkIndices + indexTopK * queryTaskOffset
                            : nullptr,
                        windowSize, winCalcLen, compressRatio == 0 ? 1 : compressRatio, attnSink,
                        swaSegWidth, dense && compressRatio != 0 ? maxSeqLen : 0);
                }

                ffts_cross_core_sync(PIPE_MTE3, config);
                curr = 1 - curr;
            }
            coreOffset = (coreOffset + taskNum) % block_num;
            queryStart = -1;
            cachedLen = -1;
        }
    }

    __aicore__ inline void Run()
    {
#ifdef __DAV_C220_CUBE__
        RunAic();
#elif __DAV_C220_VEC__
        RunAiv();
#endif
    }

private:
    GlobalTensor<Dtype> q;
    GlobalTensor<Dtype> swaKCache;
    GlobalTensor<Dtype> compressKCache;
    GlobalTensor<Dtype> scores[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> output;

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *swaBlockTables;
    __gm__ int32_t *compressBlockTables;
    __gm__ float *attnSink;
    __gm__ int32_t *topkIndices;

    uint32_t batch;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t compressRatio;
    uint32_t indexTopK;
    uint32_t maxSeqLen;
    uint32_t kvSize;
    uint32_t qkStride;
    uint32_t swaSegWidth;
    uint32_t swaMaxNumBlocks;
    uint32_t compressMaxNumBlocks;
    uint32_t windowSize;
    float scale;
    bool dense;
    int svk0;

    CxaAicHelper<Dtype> aicHelper;
};

#define CXA_FUNC_DEFINE(dtype)                                                                     \
    extern "C" __global__ __aicore__ void cxa_##dtype(                                             \
        GM_ADDR q, GM_ADDR swaKCache, GM_ADDR compressKCache, GM_ADDR swaBlockTables,              \
        GM_ADDR compressBlockTables, uint32_t swaBlockSize, uint32_t compressBlockSize,            \
        uint32_t swaMaxNumBlocks, uint32_t compressMaxNumBlocks, GM_ADDR attnSink, GM_ADDR scores, \
        GM_ADDR output, uint32_t batch, GM_ADDR queryStartLoc, GM_ADDR queryLens,                  \
        GM_ADDR cachedLens, uint32_t nHeads, uint32_t headDim, float scale, uint32_t windowSize,   \
        uint32_t kvSize, uint32_t compressRatio, uint32_t indexTopK, GM_ADDR topkIndices,          \
        uint32_t dense)                                                                            \
    {                                                                                              \
        CXA<dtype> op;                                                                             \
        op.Init(q, swaKCache, compressKCache, swaBlockTables, compressBlockTables, swaBlockSize,   \
                compressBlockSize, swaMaxNumBlocks, compressMaxNumBlocks, attnSink, scores,        \
                output, batch, queryStartLoc, queryLens, cachedLens, nHeads, headDim, scale,       \
                windowSize, kvSize, compressRatio, indexTopK, topkIndices, dense);                 \
        op.Run();                                                                                  \
    }