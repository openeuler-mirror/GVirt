/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"
#include "debug.h"

#pragma once
#define MAX_N0 XLITE_MAX_M0
#define INDEXER_KV_TILE_LEN 4096

template <typename Dtype>
class IndexerTopK
{
public:
    __aicore__ inline IndexerTopK()
    {
    }

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc,
                                GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,
                                GM_ADDR scores, GM_ADDR lastTopk, GM_ADDR indices,
                                GM_ADDR topkIndices, GM_ADDR sync, uint32_t nHeads,
                                uint32_t headDim, uint32_t blockSize, uint32_t batch,
                                uint32_t maxNumBlock, uint32_t topK)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->q.SetGlobalBuffer((__gm__ Dtype *)q);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->weight.SetGlobalBuffer((__gm__ Dtype *)weight);
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;
        this->lastTopk = (__gm__ uint32_t *)lastTopk;
        this->indices = (__gm__ uint32_t *)indices;
        this->topkIndices = (__gm__ uint32_t *)topkIndices;
        this->nHeads = nHeads;
        this->headDim = headDim;
        this->blockSize = blockSize;
        this->batch = batch;
        this->maxNumBlock = maxNumBlock;
        this->topK = topK;
        this->tileSizeOfCachedKV = ROUND_UP(INDEXER_KV_TILE_LEN, SORT_BLOCK_SIZE);
        this->blockIdx = block_idx;
        this->subBlockIdx = get_subblockid();
        this->nextBlockIdx = (blockIdx + 1) % block_num;
        this->prevBlockIdx = blockIdx == 0 ? (block_num - 1) : (blockIdx - 1);
        this->setNextGeneration = 1;
        this->waitPrevGeneration = 1;
        this->resetPrevCore = 0;

        this->scores[0].SetGlobalBuffer(((__gm__ Dtype *)scores) +
                                        block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->scores[1].SetGlobalBuffer(((__gm__ Dtype *)scores) +
                                        block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV +
                                        block_num * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->setNextSync = (__gm__ int32_t *)sync + blockIdx * 2 + subBlockIdx;
        this->waitPrevSync = (__gm__ int32_t *)sync + prevBlockIdx * 2 + subBlockIdx;
        assert(this->tileSizeOfCachedKV >= this->topK);

#ifdef __DAV_C220_CUBE__
        /*
         * scores = k * q
         *     m: cachedTokens, n: queryLen * nHeads, k: headDim
         *     m0: blockSize, n0: MAX_N0, k0: MAX_K0
         * index_scores = weights * scores
         *     m: queryLen, n: cachedTokens, k: nHeads
         *     m0: 16, n0: blockSize, k0: MAX_K0
         */
        constexpr uint32_t k0 = 256 / sizeof(Dtype);
        assert(headDim <= k0 && nHeads <= k0);
        uint64_t off = 0;
        uint64_t kl1Size = blockSize * headDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kl1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kl1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kl1Size;
        }

        uint64_t ql1Size = MAX_N0 * headDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            ql1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            ql1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ql1Size;
        }

        uint64_t wl1Size = XLITE_MAX_M0 * nHeads * sizeof(Dtype);
        wl1Buf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        wl1Buf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += wl1Size;

        uint64_t kql1Size = blockSize * MAX_N0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kql1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kql1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kql1Size;
        }

        off = 0;
        uint64_t l0aSize = XLITE_MAX_M0 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            l0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0aSize;
        }

        off = 0;
        uint64_t l0bSize = MAX_N0 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0bSize;
        }

        off = 0;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
#endif
    }

    __aicore__ inline void SetNextCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
        dbg_printf("block%d subblock%u set block%d subblock%u %u\n", blockIdx, subBlockIdx,
                   nextBlockIdx, subBlockIdx, setNextGeneration);
        *val = setNextGeneration;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b16(setNextSync, val, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
        setNextGeneration++;
    }

    __aicore__ inline void WaitPrevCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
        dbg_printf("block%d subblock%u wait block%d subblock%u %u\n", blockIdx, subBlockIdx,
                   prevBlockIdx, subBlockIdx, waitPrevGeneration);
        do {
            copy_gm_to_ubuf_align_b16(val, waitPrevSync, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        } while (*val < waitPrevGeneration);
        waitPrevGeneration++;
    }

    __aicore__ inline void ResetPrevCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
        dbg_printf("block%d subblock%u reset block%d subblock%u\n", blockIdx, subBlockIdx,
                   prevBlockIdx, subBlockIdx);
        *val = 0;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b16(waitPrevSync, val, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
    }

#ifdef __DAV_C220_CUBE__
    __aicore__ inline void RunAicIndexerScores(GlobalTensor<Dtype> query,
                                               GlobalTensor<Dtype> weight, int queryLen,
                                               __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                               GlobalTensor<Dtype> scores)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mIdxStart = kvOffset / blockSize;
        int mLoop = DIV_ROUND_UP(kvLen, blockSize);
        int mSize = blockSize;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = queryLen * nHeads;
        int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int kBlockNum = headDim / kBlockSize;

        int wmSize = queryLen;
        int wmBlockPad = ROUND_UP(wmSize, MBLOCKSIZE);
        int wmBlockNum = wmBlockPad / MBLOCKSIZE;
        int wnSize = blockSize;
        int wnBlockPad = ROUND_UP(wnSize, NBLOCKSIZE);
        int wnBlockNum = wnBlockPad / NBLOCKSIZE;
        int wkBlockNum = nHeads / kBlockSize;

        // copy weight (queryTaskLen, nHeads) to L1
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        CopyGmToL1Nd2Nz(wl1Buf, weight, wmSize, nHeads, (headDim + nHeads), wmBlockPad);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);

        int curr = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        // do queryTileSize's QK^T * weight
        for (int mIdx = 0; mIdx < mLoop; mIdx++) {  // kvLen
            int mOffset = mIdx * blockSize;
            if (mOffset + mSize > kvLen) {
                mSize = kvLen - mOffset;
                mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
                mBlockNum = mBlockPad / MBLOCKSIZE;
                wnSize = mSize;
                wnBlockPad = ROUND_UP(wnSize, NBLOCKSIZE);
                wnBlockNum = wnBlockPad / NBLOCKSIZE;
            }

            // copy k (blockSize, headDim) to L1
            uint32_t block = blockTable[mIdx + mIdxStart];
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);
            CopyGmToL1Nd2Nz(kl1Buf[curr], kCache[block * blockSize * headDim], mSize, headDim,
                            headDim, mBlockPad);
            // copy q (queryTaskLen, nHeads, headDim) to L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
            CopyGmToL1Nd2Nz(ql1Buf[curr], query, nSize, headDim, headDim, nBlockPad);

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            CopyToL0ACol(l0aBuf[curr], kl1Buf[curr], mBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);

            CopyToL0BCol(l0bBuf[curr], ql1Buf[curr], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

            // mmad scores (blockSize, queryTaskLen, nHeads)
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, headDim, true);
            if (mBlockNum * nBlockNum < 10) {
                PipeBarrier<PIPE_M>();
            }
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);

            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy scores (blockSize, queryTaskLen, nHeads) from L0C to L1 with ReLU filter.
            WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);
            CopyL0CToL1(kql1Buf[curr], l0cBuf, mBlockPad, nBlockPad, mBlockPad,
                        mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE, /*reluEn=*/1);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);

            SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);

            for (int q = 0; q < queryLen; q++) {
                // copy weight (1, nHeads) to L0A
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], wl1Buf[0][q * kBlockSize], 1, 0, wkBlockNum);
                // copy scores (blockSize, nHeads) to L0B
                CopyToL0BCol(l0bBuf[curr], kql1Buf[curr][q * mBlockPad * nHeads], wnBlockNum, 0,
                             wkBlockNum);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad index_scores (1, blockSize)
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], MBLOCKSIZE, wnBlockPad, nHeads, true);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                if (wnBlockNum < 10) {
                    PipeBarrier<PIPE_M>();
                }

                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

                // copy index_scores (1, blockSize) from L0C to GM
                CopyToGm(scores[q * tileSizeOfCachedKV + mOffset], l0cBuf, 1, mSize, MBLOCKSIZE,
                         tileSizeOfCachedKV);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
            SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);
            curr = 1 - curr;
        }
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
#endif

#ifdef __DAV_C220_VEC__
    __aicore__ inline void RunAivTopk(__gm__ Dtype *scores, __gm__ uint32_t *lastTopk,
                                      __gm__ uint32_t *indices, int queryLen, int kvOffset,
                                      int kvLen, uint32_t topK, __gm__ uint32_t *topkIndices,
                                      int queryPosBase)
    {
        assert(kvLen <= MAX_INDEXER_KV_TILE_LEN && topK <= MAX_TOPK_NUM &&
               topK <= queryPosBase + queryLen);
        constexpr float min = FLOAT_MIN;

        // total sort & WaitPrevCore & SetNextCore use
        uint64_t off = 0;
        __ubuf__ float *totalSortPhysical = reinterpret_cast<__ubuf__ float *>(off);
        __ubuf__ float *totalSort = totalSortPhysical;  // can point elsewhere to reduce memcpy
        off += ROUND_UP(MAX_TOPK_NUM * 4 * sizeof(float), VECTOR_MAX_BYTESIZE);
        // see `mrgSortBuf0`; separating the two mrgSort buffers to avoid bank conflict in A2/A3
        __ubuf__ float *mrgSortBuf1 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);

        // in
        __ubuf__ Dtype *in0 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
        __ubuf__ Dtype *in1 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
        __ubuf__ Dtype *in[PINGPONG_BUF_NUM] = {in0, in1};
        __ubuf__ uint32_t *sortIndices0 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *sortIndices1 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *sortIndices[PINGPONG_BUF_NUM] = {sortIndices0, sortIndices1};
        __ubuf__ float *lastSort0 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        __ubuf__ float *lastSort1 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        __ubuf__ float *lastSort[PINGPONG_BUF_NUM] = {lastSort0, lastSort1};

        // out
        __ubuf__ uint32_t *out0 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *out1 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *out[PINGPONG_BUF_NUM] = {out0, out1};

        // calc
        uint64_t invoff =
            UB_SIZE - ROUND_UP(MAX_INDEXER_KV_TILE_LEN * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        __ubuf__ float *mrgSortBuf0 = reinterpret_cast<__ubuf__ float *>(invoff);
        assert(off <= invoff);

        constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
        constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
        int topKSortRepeat = DIV_ROUND_UP(topK, SORT_BLOCK_SIZE);

        int curr = 0;
        bool waitCoreTriggered = false;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        for (int idx = 0; idx < queryLen; idx++) {
            int p0 = queryPosBase + idx;  // position of the current token in the sequence
            int validKvLen = MIN(p0 - kvOffset + 1, kvLen);  // per position valid kvLen
            if (validKvLen <= 0) {  // the last kv chunk should have concluded the topK merge
                continue;
            }
            bool isFirst = kvOffset <= 0;
            bool isFinal = kvOffset + validKvLen > p0;
            int repeat = DIV_ROUND_UP(validKvLen, calcPad);
            int fullRepeat = validKvLen / calcPad;
            int sortRepeat = DIV_ROUND_UP(validKvLen, SORT_BLOCK_SIZE);
            pipe_barrier(PIPE_V);

            // pad the tail of `mrgSortBuf0` (incoming scores) with `FLOAT_MIN`
            int sortLen = sortRepeat * SORT_BLOCK_SIZE;
            if (fullRepeat < repeat || sortLen > repeat * calcPad) {
                uint16_t repeatNum = sortLen / calcPad - fullRepeat + 1;
                vector_dup(mrgSortBuf0 + fullRepeat * calcPad, float(min), repeatNum, 1, 1, 8, 1);
                pipe_barrier(PIPE_V);
            }

            // copy scores to in
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // acquire `in[curr]`
            CopyGmToUbufAligned(in[curr], scores + idx * tileSizeOfCachedKV,
                                validKvLen * sizeof(Dtype));
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            // convert in to float
            convert_input(mrgSortBuf0, in[curr], fullRepeat, validKvLen % calcPad);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // release `in[curr]`

            // copy indices to sortIndices
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);  // acquire `sortIndices[curr]`
            // TODO: check if the indices can be generated in ubuf instead of copying from GM
            CopyGmToUbufAligned(sortIndices[curr], indices + kvOffset,
                                validKvLen * sizeof(uint32_t));
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
            // sort local
            vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices[curr], sortRepeat);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);  // release `sortIndices[curr]`

            uint32_t dstBufIdx = 0;
            MrgSort(mrgSortBuf1, mrgSortBuf0, sortRepeat, &dstBufIdx, topK);
            __ubuf__ float *localSort = dstBufIdx == 0 ? mrgSortBuf1 : mrgSortBuf0;
            pipe_barrier(PIPE_V);

            // sort local & last
            // Drain the previous iteration's async MTE3 reads of `totalSort` (lastTopk copy /
            // finalize output) before the V pipe rewrites it.
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            totalSort = totalSortPhysical;
            uint64_t totalSortLen = MIN(topK, validKvLen);
            if (!isFirst) {
                if (!waitCoreTriggered) {
                    WaitPrevCore();
                    waitCoreTriggered = true;
                    resetPrevCore = 1;
                }
                // copy last intermediate sort results (score + index) to `lastSort[curr]`
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4 + curr);  // acquire `lastSort[curr]`
                uint64_t lastSortLen = MIN(topK, kvOffset);
                CopyGmToUbufAligned(lastSort[curr], lastTopk + idx * 2 * topK,
                                    lastSortLen * 2 * sizeof(uint32_t));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4 + curr);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4 + curr);
                // TODO: check list order & config
                __ubuf__ float *addrs[4] = {localSort, lastSort[curr]};
                vmrgsort4(totalSort, addrs, totalSortLen | (lastSortLen << 16),
                          1ull | (0x3ull << MGR_SORT_VALID_BITS_OFFSET));
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4 + curr);  // release `lastSort[curr]`
                totalSortLen += lastSortLen;
            } else {
                // isFirst: materialize localSort into `totalSortPhysical` so the MTE3 lastTopk
                // copy / finalize reads a stable buffer not aliased to `mrgSortBuf*`.
                copy_ubuf_to_ubuf(totalSort, localSort, 0, 1,
                                  DIV_ROUND_UP(topK * 2 * sizeof(uint32_t), BLOCK_SIZE), 1, 1);
                pipe_barrier(PIPE_V);
            }
            pipe_barrier(PIPE_V);

            if (!isFinal) {
                // copy `totalSort` (intermediate score+index results) to GM for next core to merge
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                // copy `totalSort` (intermediate score+index results) to last
                CopyUbufToGmAligned(lastTopk + idx * topK * 2, totalSort,
                                    MIN(topK, totalSortLen) * 2 * sizeof(uint32_t));
                if (idx == queryLen - 1) {
                    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                    SetNextCore();
                }
            } else {  // isFinal
                // aggregate topK indices from totalSort, sort them from largest index to
                // smallest, and copy to GM. `totalSort` is `totalSortPhysical` (never aliases
                // `mrgSortBuf*`), so both calc buffers are free here.
                __ubuf__ uint32_t *index0 = (__ubuf__ uint32_t *)mrgSortBuf0;
                vreducev2(index0, (__ubuf__ uint32_t *)totalSort, (__ubuf__ uint32_t *)totalSort,
                          DIV_ROUND_UP(topK, calcPad / 2), 1, 2, 8, 0);
                pipe_barrier(PIPE_V);
                // assuming `index0`'s hightest bit is 0, we can directly cast it to float for
                // vbitsort while preserving the order of the indices
                vbitsort(mrgSortBuf1, mrgSortBuf0, index0, topKSortRepeat);
                pipe_barrier(PIPE_V);
                uint32_t outDstBufIdx = 0;
                MrgSort(mrgSortBuf1, mrgSortBuf0, topKSortRepeat, &outDstBufIdx, topK);
                __ubuf__ uint32_t *indexSorted =
                    (__ubuf__ uint32_t *)(outDstBufIdx == 0 ? mrgSortBuf1 : mrgSortBuf0);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // acquire `out[curr]`
                vreducev2(out[curr], indexSorted, indexSorted, DIV_ROUND_UP(topK, calcPad / 2), 1,
                          2, 8, 0);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
                CopyUbufToGmAligned(topkIndices + idx * topK, out[curr], topK * sizeof(uint32_t));
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // release `out[curr]`
            }
            // Release `totalSort` for the next iteration's V-pipe writes: the async MTE3 reads
            // issued this iteration (lastTopk copy / finalize output) are now enqueued.
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            curr = 1 - curr;
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
#endif

    __aicore__ inline void Run()
    {
        uint64_t flagIdx0 = 0;
        uint64_t flagIdx1 = 1;
        uint64_t flagIdx2 = 2;
        uint64_t flagIdx3 = 3;
        uint64_t mode = 2;  // inner-group aic/aiv sync
#ifdef __DAV_C220_CUBE__
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);
        uint64_t sync0 = 1 | (mode << 4) | (flagIdx0 << 8);
        uint64_t sync1 = 1 | (mode << 4) | (flagIdx1 << 8);
        uint64_t a2vSyncFlag[PINGPONG_BUF_NUM] = {sync0, sync1};
        uint64_t v2aSyncFlag[PINGPONG_BUF_NUM] = {flagIdx2, flagIdx3};
#elif __DAV_C220_VEC__
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t sync2 = 1 | (mode << 4) | (flagIdx2 << 8);
        uint64_t sync3 = 1 | (mode << 4) | (flagIdx3 << 8);
        uint64_t a2vSyncFlag[PINGPONG_BUF_NUM] = {flagIdx0, flagIdx1};
        uint64_t v2aSyncFlag[PINGPONG_BUF_NUM] = {sync2, sync3};
#endif

        int queryTileSize = XLITE_MAX_M0 / nHeads;
        if (queryTileSize == 0) {
            queryTileSize = XLITE_MAX_M0;
        }

        int totalIdx = 0;
        int curr = 0;
#ifdef __DAV_C220_VEC__
        ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[0]);
        ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[1]);
#endif
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {  // each loop handles one sequence
            int queryLen = queryLens[batchIdx];
            int cachedLen = cachedLens[batchIdx];
            int queryStart = queryStartLoc[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlock * sizeof(uint32_t));

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNumMax = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV);
            int taskNum = queryNum * kvNumMax;
            for (int idx = 0; idx < taskNum;) {  // each loop handles a chunk of one sequence/batch
                // tiling infor for the indexer query matrix
                int queryIdx = idx / kvNumMax;
                int queryTaskLen = queryTileSize;
                int queryOffset = queryIdx * queryTileSize;
                if (queryOffset + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryOffset;
                }
                int queryTaskOffset = queryStart + queryOffset;
                int queryPosBase = cachedLen + queryOffset;
                int queryPosEnd = queryPosBase + queryTaskLen - 1;  // inclusive
                // per-chunk token position info, skip if this chunk's attention is dense
                if (queryPosEnd < topK) {
                    idx = NEXT_MULTIPLE(idx, kvNumMax);  // skip KV chunks; jump to next query chunk
                    continue;
                }
                // tiling infor for the indexer key matrix
                int kvIdx = idx % kvNumMax;
                int kvOffset = kvIdx * tileSizeOfCachedKV;
                // skip score calculation if per-chunk last token pos < KV chunk's smallest pos
                if (queryPosEnd < kvOffset) {
                    idx = NEXT_MULTIPLE(idx, kvNumMax);  // skip KV chunks; jump to next query chunk
                    continue;
                }
                int kvLen = MIN(tileSizeOfCachedKV, queryPosEnd - kvOffset + 1);
                idx++;
                totalIdx++;  // only increase the global task index when the current task is valid
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
#ifdef __DAV_C220_CUBE__
                uint32_t qOffset = queryTaskOffset * nHeads * headDim;
                uint32_t wOffset = queryTaskOffset * (headDim + nHeads) + headDim;
                dbg_printf("block%d: {batch %d, query start loc %u, query [%u - %u), index k [%u - "
                           "%u)} use %d buf\n",
                           blockIdx, batchIdx, queryStart, queryOffset, queryOffset + queryTaskLen,
                           kvOffset, kvOffset + kvLen, curr);
                wait_flag_dev(v2aSyncFlag[curr]);
                RunAicIndexerScores(q[qOffset], weight[wOffset], queryTaskLen, blockTable, kvOffset,
                                    kvLen, scores[curr]);
                ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr]);
#else
                int nWork = queryTaskLen;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = subBlockIdx * nWorkCurCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t outOffset = (queryTaskOffset + nWorkStart) * topK;
                wait_flag_dev(a2vSyncFlag[curr]);
                dbg_printf("block%d subblock%u: {batch %d, query start loc %u, query [%u - %u), "
                           "index k [%u - %u)} use %d buf\n",
                           blockIdx, subBlockIdx, batchIdx, queryStart, queryOffset + nWorkStart,
                           queryOffset + nWorkStart + nWorkCurCore, kvOffset, kvOffset + kvLen,
                           curr);
                RunAivTopk(
                    (__gm__ Dtype *)scores[curr][nWorkStart * tileSizeOfCachedKV].GetPhyAddr(),
                    lastTopk + outOffset * 2, indices, nWorkCurCore, kvOffset, kvLen, topK,
                    topkIndices + outOffset, queryPosBase + nWorkStart);
                ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[curr]);
#endif
                curr = 1 - curr;
            }
        }
#ifdef __DAV_C220_CUBE__
        wait_flag_dev(v2aSyncFlag[0]);
        wait_flag_dev(v2aSyncFlag[1]);
#else
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ResetPrevCore();
        }
#endif
    }

private:
    GlobalTensor<Dtype> q;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> weight;
    GlobalTensor<Dtype> scores[PINGPONG_BUF_NUM];
    __gm__ int32_t *setNextSync;
    __gm__ int32_t *waitPrevSync;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;
    __gm__ uint32_t *lastTopk;
    __gm__ uint32_t *indices;
    __gm__ uint32_t *topkIndices;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlock;
    uint32_t topK;
    uint32_t tileSizeOfCachedKV;
    int blockIdx;
    int subBlockIdx;
    int nextBlockIdx;
    int prevBlockIdx;
    uint32_t setNextGeneration;
    uint32_t waitPrevGeneration;
    int resetPrevCore;

    LocalTensor<Dtype> kl1Buf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> ql1Buf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<Dtype> wl1Buf;                     // event 4
    LocalTensor<Dtype> kql1Buf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;  // event 0
};

#define INDEXER_TOPK_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void indexer_topk_##dtype(                                \
        GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc, GM_ADDR queryLens,   \
        GM_ADDR cachedLens, GM_ADDR blockTables, GM_ADDR scores, GM_ADDR lastTopk,             \
        GM_ADDR indices, GM_ADDR topkIndices, GM_ADDR sync, uint32_t nHeads, uint32_t headDim, \
        uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock, uint32_t topK)               \
    {                                                                                          \
        IndexerTopK<dtype> op;                                                                 \
        op.Init(q, kCache, weight, queryStartLoc, queryLens, cachedLens, blockTables, scores,  \
                lastTopk, indices, topkIndices, sync, nHeads, headDim, blockSize, batch,       \
                maxNumBlock, topK);                                                            \
        op.Run();                                                                              \
    }