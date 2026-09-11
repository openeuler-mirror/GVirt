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
#include "ring_sync.h"
#include "cxa_aic_helper.h"

// FlashCXA: flash (online-softmax) variant of CXA. Tiles the compress-token
// dimension into tileSizeOfCachedKV chunks merged online.
//
// Per-tile scores row [SWA | compress] is tile-local: SWA (cols [0, swaSegWidth))
// only on the first kv tile.
template <typename Dtype>
class FlashCXA
{
public:
    __aicore__ inline FlashCXA()
    {
    }

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR swaKCache, GM_ADDR compressKCache,
                                GM_ADDR swaBlockTables, GM_ADDR compressBlockTables,
                                uint32_t swaBlockSize, uint32_t compressBlockSize,
                                uint32_t swaMaxNumBlocks, uint32_t compressMaxNumBlocks,
                                GM_ADDR attnSink, GM_ADDR qk, GM_ADDR sv, GM_ADDR max, GM_ADDR sum,
                                GM_ADDR lastMax, GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output,
                                uint32_t batch, GM_ADDR queryStartLoc, GM_ADDR queryLens,
                                GM_ADDR cachedLens, uint32_t nHeads, uint32_t headDim, float scale,
                                uint32_t windowSize, uint32_t compressRatio, uint32_t indexTopK,
                                GM_ADDR topkIndices, uint32_t tileSizeOfCachedKV)
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
        this->indexTopK = (topkIndices != nullptr) ? indexTopK : 0;
        this->swaMaxNumBlocks = swaMaxNumBlocks;
        this->compressMaxNumBlocks = compressMaxNumBlocks;
        this->windowSize = windowSize;
        this->swaSegWidth = windowSize == 0 ? 0 : windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B;
        this->tileSizeOfCachedKV = tileSizeOfCachedKV;
        this->qkStride = swaSegWidth + tileSizeOfCachedKV;
        ringSync.Init(sync);

        this->qk[0].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride);
        this->qk[1].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride +
                                    block_num * XLITE_MAX_M0 * qkStride);
        this->sv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * headDim);
        this->sv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * headDim +
                                    block_num * XLITE_MAX_M0 * headDim);
        this->max[0].SetGlobalBuffer(((__gm__ float *)max) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0);
        this->max[1].SetGlobalBuffer(((__gm__ float *)max) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0 +
                                     block_num * XLITE_MAX_M0 * 2);
        this->sum[0].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0);
        this->sum[1].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0 +
                                     block_num * XLITE_MAX_M0 * 2);
        this->lastMax.SetGlobalBuffer((__gm__ float *)lastMax);
        this->lastSum.SetGlobalBuffer((__gm__ float *)lastSum);

        svk0 = aicHelper.Init(nHeads, headDim, swaBlockSize, compressBlockSize, windowSize,
                              compressRatio, qkStride, swaSegWidth, false);
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t softmaxConfig = 1 | (mode << 4) | (flagIdx << 8);
        flagIdx = 1;
        uint64_t updateConfig = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskOffset, lastQueryTaskLen, last;
        int lastAbsQueryStart = 0;
        int lastKvOffset, lastKvLen, lastCalcLen;
        int lastHasSwa = 0;
        __gm__ uint32_t *lastSwaBt;
        __gm__ uint32_t *lastCompressBt;
        GlobalTensor<Dtype> lastCompressKCache;

        int queryTileSize = XLITE_MAX_M0 / nHeads;
        if (queryTileSize == 0) {
            queryTileSize = XLITE_MAX_M0;
        }
        int needDoSV = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            int cachedLen = cachedLens[batchIdx];
            __gm__ uint32_t *swaBt =
                (__gm__ uint32_t *)((uint64_t)swaBlockTables +
                                    batchIdx * swaMaxNumBlocks * sizeof(uint32_t));
            __gm__ uint32_t *compressBt =
                (__gm__ uint32_t *)((uint64_t)compressBlockTables +
                                    batchIdx * compressMaxNumBlocks * sizeof(uint32_t));

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int compressTotalLen = (cachedLen + queryLen) / compressRatio;
            int kvNum = DIV_ROUND_UP(compressTotalLen, tileSizeOfCachedKV);
            // kv tile 0 must always run to carry the SWA pass even if compress is empty.
            if (windowSize != 0 && kvNum == 0) {
                kvNum = 1;
            }
            int taskNum = queryNum * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int queryIdx = idx / kvNum;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * tileSizeOfCachedKV * compressRatio;
                if (calcLen <= (uint32_t)kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = tileSizeOfCachedKV * compressRatio;
                if (kvOffset + kvLen > (int)calcLen) {
                    kvLen = calcLen - kvOffset;
                }
                // SWA segment only in the first kv tile (swaSegWidth << tileSizeOfCachedKV).
                bool hasSwa = (windowSize != 0) && (kvIdx == 0);

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;
                uint32_t absQueryStart = cachedLen + queryTaskStart;
                uint32_t mhOffset = queryTaskOffset * nHeads * headDim;

                dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), "
                           "kv [%u - %u)} use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           nHeads, kvOffset, kvOffset + kvLen, curr);
                aicHelper.RunAicQK(q[mhOffset], swaKCache, compressKCache, swaBt, compressBt,
                                   absQueryStart, queryTaskLen, kvOffset, kvLen, calcLen, calcLen,
                                   qk[curr], hasSwa);
                ffts_cross_core_sync(PIPE_FIX, softmaxConfig);

                if (needDoSV != 0) {
                    // wait aiv softmax done, then SV for the previous tile
                    wait_flag_dev(2);
                    dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), "
                               "kv [%u - %u)} use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, nHeads, lastKvOffset,
                               lastKvOffset + lastKvLen, last);
                    aicHelper.RunAicSV(qk[last], swaKCache, lastCompressKCache, lastSwaBt,
                                       lastCompressBt, lastAbsQueryStart, lastQueryTaskLen,
                                       lastKvOffset, lastKvLen, lastCalcLen, lastCalcLen, sv[last],
                                       lastHasSwa != 0);
                    ffts_cross_core_sync(PIPE_FIX, updateConfig);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastSwaBt = swaBt;
                lastCompressBt = compressBt;
                lastCompressKCache = compressKCache;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                lastCalcLen = calcLen;
                lastAbsQueryStart = absQueryStart;
                lastHasSwa = hasSwa ? 1 : 0;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
        }

        // do last SV
        if (needDoSV != 0) {
            wait_flag_dev(2);
            dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), "
                       "kv [%u - %u)} use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, nHeads, lastKvOffset,
                       lastKvOffset + lastKvLen, last);
            aicHelper.RunAicSV(qk[last], swaKCache, lastCompressKCache, lastSwaBt, lastCompressBt,
                               lastAbsQueryStart, lastQueryTaskLen, lastKvOffset, lastKvLen,
                               lastCalcLen, lastCalcLen, sv[last], lastHasSwa != 0);
            ffts_cross_core_sync(PIPE_FIX, updateConfig);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t flagIdx = 2;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int dbgBlockIdx = block_idx;

        int lastBatchIdx, lastQueryTaskLen, last, lastKvOffset, lastKvLen, lastQueryTaskOffset,
            lastWorkStart, lastWorkCurCore, lastActualCalcSoftmaxLen;
        int lastKvIdx;
        int lastIsLastKvTile;
        uint32_t lastOutOffset;

        int queryTileSize = XLITE_MAX_M0 / nHeads;
        if (queryTileSize == 0) {
            queryTileSize = XLITE_MAX_M0;
        }
        int needDoUpdate = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int resetPrevCore = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            int cachedLen = cachedLens[batchIdx];

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int compressTotalLen = (cachedLen + queryLen) / compressRatio;
            int kvNum = DIV_ROUND_UP(compressTotalLen, tileSizeOfCachedKV);
            if (windowSize != 0 && kvNum == 0) {
                kvNum = 1;
            }
            int taskNum = queryNum * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int queryIdx = idx / kvNum;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                uint32_t calcCompressLen = calcLen / compressRatio;
                int kvOffset = kvIdx * tileSizeOfCachedKV * compressRatio;
                if (calcLen <= kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = tileSizeOfCachedKV * compressRatio;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                bool hasSwa = (windowSize != 0) && (kvIdx == 0);

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;
                uint32_t outOffset = queryTaskOffset * nHeads;

                int isLastKvTile = (kvOffset + kvLen == calcLen) ? 1 : 0;

                int nWork = queryTaskLen * nHeads;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = get_subblockid() * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t qkOffset = nWorkStart * qkStride;
                uint32_t calcSoftmaxLen = cachedLen + queryTaskStart + 1;
                int actualCalcSoftmaxLen = calcSoftmaxLen - kvOffset;
                if (actualCalcSoftmaxLen > kvLen) {
                    actualCalcSoftmaxLen = kvLen;
                }
                // SWA causal window (absolute); only meaningful for the hasSwa tile.
                int winStart =
                    calcSoftmaxLen > (int)windowSize ? (int)calcSoftmaxLen - (int)windowSize : 0;
                uint32_t winCalcLen = calcSoftmaxLen - ROUND_DOWN(winStart, K_BLOCK_SIZE_2B);
                uint32_t swaSegWidthEff = hasSwa ? swaSegWidth : 0;
                uint32_t outN = swaSegWidthEff + ROUND_UP(kvLen / compressRatio, 4 * svk0);
                if (outN > qkStride) {
                    outN = qkStride;
                }

                // wait aic qk done
                wait_flag_dev(0);

                dbg_printf(
                    "block%d subblock%u: {batch %d, query [%u - %u) "
                    "query x head group [%u - "
                    "%u) "
                    "kv [%u - %u)} calcSoftmaxLen %u, off %u, stride %u, outN %u, use %d temp buf: "
                    "SOFTMAX\n",
                    dbgBlockIdx, get_subblockid(), batchIdx, queryTaskOffset,
                    queryTaskOffset + queryTaskLen, nWorkStart, nWorkStart + nWorkCurCore, kvOffset,
                    kvOffset + kvLen, actualCalcSoftmaxLen, nWorkStart, nHeads, outN, curr);
                RunAivSoftmaxPingPong(
                    (__gm__ Dtype *)qk[curr][qkOffset].GetPhyAddr(), nWorkCurCore, qkStride,
                    actualCalcSoftmaxLen, outN, true, nWorkStart, nHeads,
                    (__gm__ float *)max[curr][nWorkStart].GetPhyAddr(),
                    (__gm__ float *)sum[curr][nWorkStart].GetPhyAddr(), true, scale, kvOffset,
                    (calcCompressLen > indexTopK) ? indexTopK : 0,
                    (calcCompressLen > indexTopK && indexTopK > 0)
                        ? topkIndices + indexTopK * queryTaskOffset
                        : nullptr,
                    hasSwa ? windowSize : 0, hasSwa ? winCalcLen : 0, compressRatio,
                    kvIdx == 0 ? attnSink : nullptr, swaSegWidthEff);
                ffts_cross_core_sync(PIPE_MTE3, config);

                if (needDoUpdate != 0) {
                    // wait aic sv done, then online-softmax merge for the previous tile.
                    wait_flag_dev(1);
                    if (lastKvOffset != 0) {
                        ringSync.WaitPrevCore();
                        resetPrevCore = 1;
                    }
                    dbg_printf("block%d subblock%u: {batch %d, query [%u - %u)"
                               "query x head group [%u "
                               "- %u) "
                               "kv [%u - %u)} use %d temp buf: UPDATE\n",
                               dbgBlockIdx, get_subblockid(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                               lastWorkStart + lastWorkCurCore, lastKvOffset,
                               lastKvOffset + lastKvLen, last);
                    RunAivSoftmaxUpdate(
                        (__gm__ Dtype *)sv[last][lastWorkStart * headDim].GetPhyAddr(),
                        (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ Dtype *)output[lastOutOffset * headDim].GetPhyAddr(),
                        (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                        (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(), lastWorkCurCore,
                        nHeads, headDim, lastKvOffset == 0, lastActualCalcSoftmaxLen, true,
                        lastWorkStart, nHeads, compressRatio,
                        ((windowSize != 0) && (lastKvOffset == 0)) ? swaSegWidth : 0);
                    if (!lastIsLastKvTile) {
                        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        ringSync.SetNextCore();
                    }
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastWorkStart = nWorkStart;
                lastWorkCurCore = nWorkCurCore;
                lastOutOffset = outOffset;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                lastIsLastKvTile = isLastKvTile;
                lastActualCalcSoftmaxLen = actualCalcSoftmaxLen;
                last = curr;
                needDoUpdate = 1;
                curr = 1 - curr;
            }
            queryStart = -1;
        }

        // do last update
        if (needDoUpdate != 0) {
            wait_flag_dev(1);
            if (lastKvOffset != 0) {
                ringSync.WaitPrevCore();
                resetPrevCore = 1;
            }
            dbg_printf("block%d subblock%u: {batch %d, query [%u - %u)"
                       "query x head group [%u "
                       "- %u) "
                       "kv [%u - %u)} use %d temp buf: UPDATE\n",
                       dbgBlockIdx, get_subblockid(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                       lastWorkStart + lastWorkCurCore, lastKvOffset, lastKvOffset + lastKvLen,
                       last);
            RunAivSoftmaxUpdate(
                (__gm__ Dtype *)sv[last][lastWorkStart * headDim].GetPhyAddr(),
                (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                (__gm__ Dtype *)output[lastOutOffset * headDim].GetPhyAddr(),
                (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(), lastWorkCurCore, nHeads,
                headDim, lastKvOffset == 0, lastActualCalcSoftmaxLen, true, lastWorkStart, nHeads,
                compressRatio, ((windowSize != 0) && (lastKvOffset == 0)) ? swaSegWidth : 0);
            if (!lastIsLastKvTile) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                ringSync.SetNextCore();
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ringSync.ResetPrevCore();
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
    RingSync<Dtype> ringSync;
    GlobalTensor<Dtype> q;
    GlobalTensor<Dtype> swaKCache;
    GlobalTensor<Dtype> compressKCache;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> sv[PINGPONG_BUF_NUM];
    GlobalTensor<float> max[PINGPONG_BUF_NUM];
    GlobalTensor<float> sum[PINGPONG_BUF_NUM];
    GlobalTensor<float> lastMax;
    GlobalTensor<float> lastSum;
    GlobalTensor<Dtype> output;

    CxaAicHelper<Dtype> aicHelper;

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
    uint32_t swaMaxNumBlocks;
    uint32_t compressMaxNumBlocks;
    uint32_t windowSize;
    uint32_t swaSegWidth;
    uint32_t qkStride;
    uint32_t tileSizeOfCachedKV;
    float scale;
    int svk0;
};

#define FLASH_CXA_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void flash_cxa_##dtype(                                      \
        GM_ADDR q, GM_ADDR swaKCache, GM_ADDR compressKCache, GM_ADDR swaBlockTables,             \
        GM_ADDR compressBlockTables, uint32_t swaBlockSize, uint32_t compressBlockSize,           \
        uint32_t swaMaxNumBlocks, uint32_t compressMaxNumBlocks, GM_ADDR attnSink, GM_ADDR qk,    \
        GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum, GM_ADDR sync,     \
        GM_ADDR output, uint32_t batch, GM_ADDR queryStartLoc, GM_ADDR queryLens,                 \
        GM_ADDR cachedLens, uint32_t nHeads, uint32_t headDim, float scale, uint32_t windowSize,  \
        uint32_t compressRatio, uint32_t indexTopK, GM_ADDR topkIndices,                          \
        uint32_t tileSizeOfCachedKV)                                                              \
    {                                                                                             \
        FlashCXA<dtype> op;                                                                       \
        op.Init(q, swaKCache, compressKCache, swaBlockTables, compressBlockTables, swaBlockSize,  \
                compressBlockSize, swaMaxNumBlocks, compressMaxNumBlocks, attnSink, qk, sv, max,  \
                sum, lastMax, lastSum, sync, output, batch, queryStartLoc, queryLens, cachedLens, \
                nHeads, headDim, scale, windowSize, compressRatio, indexTopK, topkIndices,        \
                tileSizeOfCachedKV);                                                              \
        op.Run();                                                                                 \
    }
