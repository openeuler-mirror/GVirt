/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"
#include "flash_attention_param.h"
#include "softmax_attn_aiv.h"

#define MAX_M0 128
#define XLITE_KERNEL_DEBUG

template <typename Dtype>
class FlashAttention
{
public:
    __aicore__ inline FlashAttention()
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk,
                                GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR sync,
                                GM_ADDR output, GM_ADDR queryStartLoc, GM_ADDR queryLens,
                                GM_ADDR cachedLens, GM_ADDR blockTables, uint32_t nHeads,
                                uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize,
                                uint32_t batch, uint32_t maxNumBlocks)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->input.SetGlobalBuffer((__gm__ Dtype *)input);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->vCache.SetGlobalBuffer((__gm__ Dtype *)vCache);
        this->output.SetGlobalBuffer((__gm__ Dtype *)output);

        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;

        this->nHeads = nHeads;
        this->nKVHeads = nKVHeads;
        this->nQKVHeads = nHeads + 2 * nKVHeads;
        this->headNumInGroup = nHeads / nKVHeads;
        this->headSize = headSize;
        this->blockSize = blockSize;
        this->batch = batch;
        this->maxNumBlocks = maxNumBlocks;
        this->maxSeqLen = maxNumBlocks * blockSize;
        this->qMemSize = nHeads * headSize;
        this->kvMemSize = nKVHeads * headSize;
        this->qkvMemSize = nQKVHeads * headSize;
        this->groupMemSize = headNumInGroup * headSize;
        this->blockMemSize = blockSize * kvMemSize;

        this->qk[0].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * MAX_M0 * TILESIZE_OF_CACHED_KV);
        this->qk[1].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * MAX_M0 * TILESIZE_OF_CACHED_KV +
                                    block_num * MAX_M0 * TILESIZE_OF_CACHED_KV);
        this->sv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * MAX_M0 * headSize);
        this->sv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * MAX_M0 * headSize +
                                    block_num * MAX_M0 * headSize);
        this->max[0].SetGlobalBuffer(((__gm__ float *)max) + block_idx * MAX_M0);
        this->max[1].SetGlobalBuffer(((__gm__ float *)max) + block_idx * MAX_M0 + block_num * MAX_M0);
        this->sum[0].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * MAX_M0);
        this->sum[1].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * MAX_M0 + block_num * MAX_M0);
        this->sync1.SetGlobalBuffer((__gm__ int32_t *)sync);
        this->sync2.SetGlobalBuffer((__gm__ int32_t *)sync + block_num * 2);

        // 分配L1/L0
        uint64_t l1ATileBytes =
            MAX_M0 * (headSize > blockSize ? headSize : blockSize) * sizeof(Dtype);
        uint64_t l1BTileBytes = blockSize * headSize * sizeof(Dtype);
        uint64_t off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            l1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1ATileBytes;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BTileBytes;
        }
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            l0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1ATileBytes;
        }
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BTileBytes;
        }
        off = 0;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    /*
     * m: tokens
     * n: cachedTokens
     * k: headSize
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> query, int queryLen, int kvHeadIdx,
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> qk)
    {
    }

    /*
     * m: tokens
     * n: headSize
     * k: cachedTokens
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, int queryLen, int kvHeadIdx,
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> sv)
    {
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

        int lastBatchIdx, lastQueryTaskLen, lastkvHeadIdx, last, lastKvOffset, lastKvLen, lastQueryTaskOffset;
        __gm__ uint32_t *lastBlockTable;

        int needDoSV = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = MAX_M0;
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = 1;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryTileSize, TILESIZE_OF_CACHED_KV);
            int taskNum = queryNum * nKVHeads * kvNum;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int kvIdx = idx % kvNum;
                int kvHeadIdx = (idx / kvNum) % nKVHeads;
                int queryIdx = idx / (kvNum * nKVHeads);
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;
                int kvHeadOffset = kvHeadIdx * groupMemSize;

                // do queryIdx & kvHeadIdx & kvIdx's QK
                uint32_t qOffset = queryTaskOffset * headSize * nQKVHeads + kvHeadOffset;
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * TILESIZE_OF_CACHED_KV;
                int kvLen = TILESIZE_OF_CACHED_KV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }
#ifdef XLITE_KERNEL_DEBUG
                printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                       "kv [%u - %u)} use %d temp buf: QK\n",
                       GetBlockIdx(), batchIdx, queryTaskOffset,
                       queryTaskOffset + queryTaskLen, kvHeadIdx, kvOffset, kvOffset + kvLen, curr);
#endif
                RunAicQK(input[qOffset], queryTaskLen, kvHeadIdx, blockTable, kvOffset, kvLen,
                         qk[curr]);
                ffts_cross_core_sync(PIPE_FIX, softmaxConfig);

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(2);
                    // do softmax * V
#ifdef XLITE_KERNEL_DEBUG
                    printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                           "kv [%u - %u)} use %d temp buf: SV\n", GetBlockIdx(),
                           lastBatchIdx, lastQueryTaskOffset, lastQueryTaskOffset + lastQueryTaskLen,
                           lastkvHeadIdx, lastKvOffset, lastKvOffset + lastKvLen, last);
#endif
                    RunAicSV(qk[last], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable,
                             lastKvOffset, lastKvLen, sv[last]);
                    ffts_cross_core_sync(PIPE_FIX, updateConfig);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastkvHeadIdx = kvHeadIdx;
                lastBlockTable = blockTable;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(2);
#ifdef XLITE_KERNEL_DEBUG
            printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                   "kv [%u - %u)} use %d temp buf: SV\n", GetBlockIdx(),
                   lastBatchIdx, lastQueryTaskOffset, lastQueryTaskOffset + lastQueryTaskLen,
                   lastkvHeadIdx, lastKvOffset, lastKvOffset + lastKvLen, last);
#endif
            RunAicSV(qk[last], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable, lastKvOffset, lastKvLen,
                     sv[last]);
            ffts_cross_core_sync(PIPE_FIX, updateConfig);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        uint64_t flagIdx = 2;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskLen, lastkvHeadIdx, last, lastKvOffset, lastKvLen, lastQueryTaskOffset, lastWorkStart, lastWorkCurCore;
        __gm__ uint32_t *lastBlockTable;

        int needDoUpdate = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int dbgBlockIdx = block_idx;
        uint32_t subIdx = get_subblockid();
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = MAX_M0;
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = 1;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryTileSize, TILESIZE_OF_CACHED_KV);
            int taskNum = queryNum * nKVHeads * kvNum;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int kvIdx = idx % kvNum;
                int kvHeadIdx = (idx / kvNum) % nKVHeads;
                int queryIdx = idx / (kvNum * nKVHeads);
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;
                int kvHeadOffset = kvHeadIdx * groupMemSize;

                uint32_t qOffset = queryTaskOffset * headSize * nQKVHeads + kvHeadOffset;
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * TILESIZE_OF_CACHED_KV;
                int kvLen = TILESIZE_OF_CACHED_KV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                int nWork = queryTaskLen * headNumInGroup;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = subIdx * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }

                // wait aic qk done
                wait_flag_dev(0);

#ifdef XLITE_KERNEL_DEBUG
                printf("block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - %u), kvHeadIdx %u "
                       "kv [%u - %u)} use %d temp buf: SOFTMAX\n",
                       dbgBlockIdx, subIdx, batchIdx, queryTaskOffset,
                       queryTaskOffset + queryTaskLen, nWorkStart, nWorkStart + nWorkCurCore, kvHeadIdx, kvOffset, kvOffset + kvLen, curr);
#endif
                ffts_cross_core_sync(PIPE_MTE3, config);

                if (needDoUpdate != 0) {
                    // wait aic sv done
                    wait_flag_dev(1);
                    // do update
#ifdef XLITE_KERNEL_DEBUG
                    printf("block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - %u), kvHeadIdx %u "
                           "kv [%u - %u)} use %d temp buf: UPDATE\n", dbgBlockIdx, subIdx,
                           lastBatchIdx, lastQueryTaskOffset, lastQueryTaskOffset + lastQueryTaskLen,
                           lastWorkStart, lastWorkStart + lastWorkCurCore,
                           lastkvHeadIdx, lastKvOffset, lastKvOffset + lastKvLen, last);
#endif
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastWorkStart = nWorkStart;
                lastWorkCurCore = nWorkCurCore;
                lastkvHeadIdx = kvHeadIdx;
                lastBlockTable = blockTable;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                last = curr;
                needDoUpdate = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last update
        if (needDoUpdate != 0) {
            wait_flag_dev(1);
#ifdef XLITE_KERNEL_DEBUG
            printf("block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - %u), kvHeadIdx %u "
                   "kv [%u - %u)} use %d temp buf: UPDATE\n", dbgBlockIdx, subIdx,
                   lastBatchIdx, lastQueryTaskOffset, lastQueryTaskOffset + lastQueryTaskLen,
                   lastWorkStart, lastWorkStart + lastWorkCurCore,
                   lastkvHeadIdx, lastKvOffset, lastKvOffset + lastKvLen, last);
#endif
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
    GlobalTensor<Dtype> input;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> vCache;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> sv[PINGPONG_BUF_NUM];
    GlobalTensor<float> max[PINGPONG_BUF_NUM];
    GlobalTensor<float> sum[PINGPONG_BUF_NUM];
    GlobalTensor<int32_t> sync1;
    GlobalTensor<int32_t> sync2;
    GlobalTensor<Dtype> output;

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;

    uint32_t nHeads;
    uint32_t nKVHeads;
    uint32_t nQKVHeads;
    uint32_t headNumInGroup;
    uint32_t headSize;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlocks;
    uint32_t maxSeqLen;
    uint32_t qMemSize;
    uint32_t kvMemSize;
    uint32_t qkvMemSize;
    uint32_t groupMemSize;
    uint32_t blockMemSize;

    LocalTensor<Dtype> l1aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;
};

#define FLASH_ATTN_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void flash_attention_##dtype(                                 \
        GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache,                                             \
        GM_ADDR qk, GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR sync, GM_ADDR output,            \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,         \
        uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize, uint32_t batch, \
        uint32_t maxNumBlocks)                                                                     \
    {                                                                                              \
        FlashAttention<dtype> op;                                                                  \
        op.Init(input, kCache, vCache, qk, sv, max, sum, sync,                                     \
                output, queryStartLoc, queryLens, cachedLens,                                      \
                blockTables, nHeads, nKVHeads, headSize, blockSize, batch, maxNumBlocks);          \
        op.Run();                                                                                  \
    }