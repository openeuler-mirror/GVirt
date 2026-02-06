/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_macro.h"
#include "kernel_operator.h"
#include "softmax_attn_aiv.h"

#define MAX_M0 128
#define MBLOCKSIZE 16
#define NBLOCKSIZE 16
#define SEQLEN_64 64
#define SEQLEN_8K 8192
#define SEQLEN_19K 19456
//#define XLITE_KERNEL_DEBUG

template<typename Dtype, typename CalcDtype>
class Attention {
public:
    __aicore__ inline Attention() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk, GM_ADDR output,
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,
        uint32_t nHeads, uint32_t nKVHeads,
        uint32_t headSize, uint32_t blockSize, uint32_t batch,
        uint32_t maxNumBlocks)
    {
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

        this->qk[0].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * MAX_M0 * maxSeqLen);
        this->qk[1].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * MAX_M0 * maxSeqLen + block_num * MAX_M0 * maxSeqLen);

        // 分配L1/L0
        uint64_t l1ATileBytes = MAX_M0 * (headSize > blockSize ? headSize : blockSize) * sizeof(Dtype);
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
                                    __gm__ uint32_t *blockTable, int totalLen,
                                    GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int nLoop = DIV_ROUND_UP(totalLen, blockSize);
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nBlockPad = ROUND_UP(blockSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int kBlockNum = DIV_ROUND_UP(headSize, kBlockSize);
        int kvHeadOffset = kvHeadIdx * headSize;

        Nd2NzParams nd2nzParams(1 /* NdNum */, queryLen /* nValue */, headSize /* dValue */,
            0 /* srcNdMatrixStride */, qkvMemSize /* srcDValue */, mBlockPad /* dstNzC0Stride */,
            headNumInGroup /* dstNzNStride */, 0 /* dstNzMatrixStride */);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        for (int h = 0; h < headNumInGroup; h++) {
            DataCopy(l1aBuf[0][MBLOCKSIZE * h], query[headSize * h], nd2nzParams);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        CopyToL0ACol(l0aBuf[0], l1aBuf[0], mBlockNum, 0, kBlockNum);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

        int curIdx = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {
            uint32_t blockIdx = blockTable[nIdx];
            int nSize = blockSize;
            if (nIdx * blockSize + nSize > totalLen) {
                nSize = totalLen - nIdx * blockSize;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx], kCache[blockIdx * blockMemSize + kvHeadOffset],
                            nSize, headSize, kvMemSize, nBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            CopyToL0BCol(l0bBuf[curIdx], l1bBuf[curIdx], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            CalMmad(l0cBuf, l0aBuf[0], l0bBuf[curIdx], mBlockPad, nBlockPad, headSize, true);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);

            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            CopyToGm(qk[nIdx * blockSize], l0cBuf, mActual, nSize, mBlockPad, maxSeqLen);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            PipeBarrier<PIPE_M>();
            curIdx = 1 - curIdx;
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

    /*
     * m: tokens
     * n: headSize
     * k: cachedTokens
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, int queryLen, int kvHeadIdx,
                                    __gm__ uint32_t *blockTable, int totalLen,
                                    GlobalTensor<Dtype> output)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int kLoop = DIV_ROUND_UP(totalLen, blockSize);
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nBlockNum = DIV_ROUND_UP(headSize, NBLOCKSIZE);
        int kBlockPad = ROUND_UP(blockSize, kBlockSize);
        int kBlockNum = kBlockPad / kBlockSize;
        int kvHeadOffset = kvHeadIdx * headSize;

        int curIdx = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int kIdx = 0; kIdx < kLoop; kIdx++) {
            int kSize = blockSize;
            if (kIdx * blockSize + kSize > totalLen) {
                kSize = totalLen - kIdx * blockSize;
                kBlockPad = ROUND_UP(kSize, kBlockSize);
                kBlockNum = kBlockPad / kBlockSize;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curIdx);
            CopyGmToL1Nd2Nz(l1aBuf[curIdx], qk[kIdx * blockSize],
                            mActual, kSize, maxSeqLen, mBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curIdx);

            uint32_t blockIdx = blockTable[kIdx];
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx], vCache[blockIdx * blockMemSize + kvHeadOffset],
                            kSize, headSize, kvMemSize, kBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curIdx);
            CopyToL0ACol(l0aBuf[curIdx], l1aBuf[curIdx], mBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            CopyToL0BTCol(l0bBuf[curIdx], l1bBuf[curIdx], nBlockNum, 0, kBlockNum, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curIdx);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);
            CalMmad(l0cBuf, l0aBuf[curIdx], l0bBuf[curIdx], mBlockPad, headSize, kBlockPad, kIdx == 0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curIdx);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            PipeBarrier<PIPE_M>();
            curIdx = 1 - curIdx;
        }
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        if (headNumInGroup == 1) {
            CopyToGm(output, l0cBuf, queryLen, headSize, mBlockPad, qMemSize);
        } else {
            int l0cOffset = headNumInGroup * kBlockSize;
            for (int i = 0; i < queryLen; i++) {
                CopyToGm(output[i * qMemSize], l0cBuf[i * l0cOffset], headNumInGroup, headSize, mBlockPad, headSize);
            }
        }
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2; // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        uint32_t lastOutOffset, lastCalcLen;
        int lastQueryTaskLen, lastkvHeadIdx, lastQkIdx;
        __gm__ uint32_t *lastBlockTable;

        int needDoSV = 0;
        int totalIdx = 0;
        int currQkIdx = 0;
        int queryStart = -1;
        int cachedLen = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable = (__gm__ uint32_t *)((uint64_t)blockTables + batchIdx * maxNumBlocks * sizeof(uint32_t));

            uint32_t m0 = MAX_M0;
            if (queryLen <= SEQLEN_64) {
                m0 = 16;
            } else {
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                int totalLen = queryLen + cachedLen;
                if (totalLen > SEQLEN_19K) {
                    m0 = 32;
                } else if (totalLen > SEQLEN_8K) {
                    m0 = 64;
                }
            }
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = 1;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum * nKVHeads;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int kvHeadIdx = idx % nKVHeads;
                int queryIdx = idx / nKVHeads;
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

                // do queryIdx & kvHeadIdx's QK
                uint32_t qOffset = queryTaskOffset * headSize * nQKVHeads + kvHeadOffset;
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
#ifdef XLITE_KERNEL_DEBUG
                printf("block%d: batch %d query start %u query [%u - %u) do QK kvHeadIdx %u calcLen %u, use %d qk buf\n",
                    GetBlockIdx(), batchIdx, queryStart, queryTaskOffset,
                    queryTaskOffset + queryTaskLen, kvHeadIdx, calcLen, currQkIdx);
#endif
                RunAicQK(input[qOffset], queryTaskLen, kvHeadIdx, blockTable, calcLen, qk[currQkIdx]);
                ffts_cross_core_sync(PIPE_FIX, config);

                if (needDoSV != 0) {
                    // wait vector softmax done                
                    wait_flag_dev(1);
                    // do softmax * V
#ifdef XLITE_KERNEL_DEBUG
                    printf("block%d: do SV kvHeadIdx %u calcLen %u, use %d qk buf\n",
                        GetBlockIdx(), lastkvHeadIdx, lastCalcLen, lastQkIdx);
#endif
                    RunAicSV(qk[lastQkIdx], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable, lastCalcLen, output[lastOutOffset]);
                }

                lastOutOffset = queryTaskOffset * headSize * nHeads + kvHeadOffset;
                lastQueryTaskLen = queryTaskLen;
                lastkvHeadIdx = kvHeadIdx;
                lastBlockTable = blockTable;
                lastCalcLen = calcLen;
                lastQkIdx = currQkIdx;
                needDoSV = 1;

                currQkIdx = 1 - currQkIdx;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(1);
#ifdef XLITE_KERNEL_DEBUG
            printf("block%d: do SV kvHeadIdx %u calcLen %u, use %d qk buf\n",
                GetBlockIdx(), lastkvHeadIdx, lastCalcLen, lastQkIdx);
#endif
            RunAicSV(qk[lastQkIdx], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable, lastCalcLen, output[lastOutOffset]);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    
        uint64_t flagIdx = 1;
        uint64_t mode = 2; // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int totalIdx = 0;
        int currQkIdx = 0;
        int cachedLen = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable = (__gm__ uint32_t *)((uint64_t)blockTables + batchIdx * maxNumBlocks * sizeof(uint32_t));

            uint32_t m0 = MAX_M0;
            if (queryLen <= SEQLEN_64) {
                m0 = 16;
            } else {
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                int totalLen = queryLen + cachedLen;
                if (totalLen > SEQLEN_19K) {
                    m0 = 32;
                } else if (totalLen > SEQLEN_8K) {
                    m0 = 64;
                }
            }
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = 1;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum * nKVHeads;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int kvHeadIdx = idx % nKVHeads;
                int queryIdx = idx / nKVHeads;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }

                int nSoftmax = queryTaskLen * headNumInGroup;
                int nSoftmaxPerCore = DIV_ROUND_UP(nSoftmax, 2);
                int nSoftmaxCurCore = nSoftmaxPerCore;
                uint32_t subIdx = get_subblockid();
                int nSoftmaxStart = subIdx * nSoftmaxPerCore;
                if (nSoftmaxStart + nSoftmaxCurCore > nSoftmax) {
                    nSoftmaxCurCore = nSoftmax - nSoftmaxStart;
                }
                uint32_t qkOffset = nSoftmaxStart * maxSeqLen;
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + nSoftmaxStart / headNumInGroup + 1;
                uint32_t outN = ROUND_UP(cachedLen + queryTaskStart + queryTaskLen, blockSize);

                // wait aic qk done
                wait_flag_dev(0);

                // do softmax
#ifdef XLITE_KERNEL_DEBUG
                int dbgBlockIdx = block_idx;
                printf("block%d subblock%u: batch %d do softmax kvHeadIdx %u m %d calcLen %u outN %u mask off %u, use %d qk buf\n",
                    dbgBlockIdx, subIdx, batchIdx, kvHeadIdx, nSoftmaxCurCore, calcLen, outN, nSoftmaxStart % headNumInGroup, currQkIdx);
#endif
                RunAivSoftmax<Dtype, CalcDtype>((__gm__ Dtype*)qk[currQkIdx][qkOffset].GetPhyAddr(),
                    nSoftmaxCurCore, maxSeqLen, calcLen, outN,
                    nSoftmaxStart % headNumInGroup, headNumInGroup);

                ffts_cross_core_sync(PIPE_MTE3, config);
                currQkIdx = 1 - currQkIdx;
            }
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
    GlobalTensor<Dtype> input;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> vCache;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
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

#define ATTN_FUNC_DEFINE(dtype, calcDtype) \
extern "C" __global__ __aicore__ void attention_##dtype( \
    GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk, GM_ADDR output, \
    GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables, \
    uint32_t nHeads, uint32_t nKVHeads, \
    uint32_t headSize, uint32_t blockSize, uint32_t batch, \
    uint32_t maxNumBlocks) \
{ \
    Attention<dtype, calcDtype> op; \
    op.Init(input, kCache, vCache, qk, output, \
        queryStartLoc, queryLens, cachedLens, blockTables, \
        nHeads, nKVHeads, \
        headSize, blockSize, batch, \
        maxNumBlocks); \
    op.Run(); \
}