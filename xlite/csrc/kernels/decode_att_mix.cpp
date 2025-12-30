/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_macro.h"
#include "kernel_operator.h"
#include "softmax_attn_aiv.h"

constexpr uint32_t PINGPONG_BUF_NUM = 2;
constexpr int CUBE_BLOCK_SIZE = 16;
constexpr uint32_t MAX_CONTEXT_BLOCK_LEN = 8192;
constexpr uint32_t MAX_CONTEXT_REPEAT_TIMES = MAX_CONTEXT_BLOCK_LEN / CUBE_BLOCK_SIZE;
constexpr uint32_t QK_RESULT_TEMP_NUM = 4; // 1-3 for subBlock, 4 for whole
constexpr uint32_t QK_RESULT_TEMP_SIZE = QK_RESULT_TEMP_NUM * VECTOR_MAX_BYTESIZE;
constexpr uint32_t MAX_UB_SIZE = 192 * 1024;

// 本算子由小艺团队贡献，参考论文《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads》 [ASPLOS 2026]
template<typename Dtype, typename CalcDtype>
class DecodeAttn {
public:
    __aicore__ inline DecodeAttn()
    {
    }

    __aicore__ inline void Init(GM_ADDR a2v, GM_ADDR v2a, GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR cachedLens, GM_ADDR mapping, GM_ADDR qk, GM_ADDR o,
                                GM_ADDR decodeIndex, GM_ADDR cumPromptLen,
                                uint32_t numTokens, uint32_t numHeads, uint32_t headSize,
                                uint32_t blockSize, uint32_t mappingLen, uint32_t numKVHeads,
                                uint32_t maxContextLen, uint32_t numQKVHeads,
                                uint32_t mOffset, uint32_t mSlice)
    {
        qGmBuf.SetGlobalBuffer((__gm__ Dtype*)q);
        kGmBuf.SetGlobalBuffer((__gm__ Dtype*)k);
        vGmBuf.SetGlobalBuffer((__gm__ Dtype*)v);
        qkGmBuf.SetGlobalBuffer((__gm__ Dtype*)qk);
        outGmBuf.SetGlobalBuffer((__gm__ Dtype*)o);

        this->a2v = (__gm__ uint32_t*)a2v;
        this->v2a = (__gm__ uint32_t*)v2a;
        this->cumPromptLen = (__gm__ int32_t*)cumPromptLen;
        this->mapping = (__gm__ int32_t*)mapping;
        this->decodeIndex = (__gm__ int32_t*)decodeIndex;
        this->cachedLens = (__gm__ int32_t*)cachedLens;

        this->headSize = headSize;
        this->nHeads = numHeads;
        this->nTokens = numTokens;
        this->nKVHeads = numKVHeads;
        this->blockSize = blockSize;
        this->mappingLen = mappingLen;
        this->maxContextLen = maxContextLen;
        this->nQKVHeads = numQKVHeads;
        this->mOffset = mOffset;
        this->mSlice = mSlice;
        this->headNumInGroup = numHeads / numKVHeads;
        this->blockMemSize = numKVHeads * blockSize * headSize;
    }

    inline __aicore__ void Run()
    {
#ifdef __DAV_C220_CUBE__
        RunAic();
#elif __DAV_C220_VEC__
        RunAiv();
#endif
    }

private:
    inline __aicore__ void RunAic()
    {
        InitAicBuf();

        RunAicQK();
        RunAicSV();
    }

    inline __aicore__ void InitAicBuf()
    {
        uint64_t off = 0;
        l1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        l1aBuf.address_.bufferAddr = off;
        l0aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf.address_.bufferAddr = off;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = off;

        for (uint32_t i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = off;
            off += blockSize * headSize * sizeof(Dtype);
        }
    }

    inline __aicore__ void RunAicQK()
    {
        uint64_t off = headSize * sizeof(Dtype) * CUBE_BLOCK_SIZE;
        for (uint32_t i = 0; i < PINGPONG_BUF_NUM; ++i) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = off;
            off += blockSize * headSize * sizeof(Dtype);
        }

        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE2>(EVENT_ID0);
        uint32_t totalTaskNum = nTokens * nKVHeads;
        uint32_t blockNum = GetBlockNum();
        for (uint32_t gqaHeadIdx = GetBlockIdx(); gqaHeadIdx < totalTaskNum; gqaHeadIdx += blockNum) {
            uint32_t qHeadIdxStart = gqaHeadIdx * headNumInGroup;
            uint32_t kvHeadIdx = qHeadIdxStart % nHeads / headNumInGroup;
            uint32_t tokenIdx = qHeadIdxStart / nHeads;
            uint32_t cumM = (uint32_t)*(cumPromptLen + tokenIdx);
            if (mSlice > 0 && (cumM >= mOffset + mSlice || cumM < mOffset)) {
                continue;
            }
            uint32_t headOffset = kvHeadIdx * headSize;
            WaitFlag<HardEvent::M_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            CopyGmToL1Nd2Nz(l1aBuf, qGmBuf[(cumM * nQKVHeads + qHeadIdxStart % nHeads) * headSize],
                            headNumInGroup, headSize, headSize, CUBE_BLOCK_SIZE);

            if (gqaHeadIdx >= blockNum) {
                SetAicAivFlag(a2v, headNumInGroup, (gqaHeadIdx - blockNum) * headNumInGroup);
            }
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
            uint32_t blockTableId = (uint32_t)*(mapping + tokenIdx * mappingLen);
            CopyGmToL1Nd2Nz(l1bBuf[0], kGmBuf[blockTableId * blockMemSize + headOffset],
                            blockSize, headSize, headSize * nKVHeads, blockSize);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            uint8_t repeat = headSize / CUBE_BLOCK_SIZE;
            LoadData2dParams params(0, repeat, 1, 0, 0, 0, inc);
            LoadData(l0aBuf, l1aBuf, params);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

            ComputeQK(tokenIdx, headOffset, qHeadIdxStart);

            if (gqaHeadIdx + blockNum >= totalTaskNum) {
                SetAicAivFlag(a2v, headNumInGroup, qHeadIdxStart);
            }
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        }
        WaitFlag<HardEvent::M_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
    }

    inline __aicore__ void ComputeQK(uint32_t tokenIdx, uint32_t headOffset, uint32_t qHeadIdxStart)
    {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_S>(EVENT_ID1);

        uint32_t contextLen = (uint32_t)*(cachedLens + tokenIdx) + 1;
        uint32_t numIters = DIV_ROUND_UP(contextLen, blockSize);
        uint32_t curIdx = 0;
        for (uint32_t kIdx = 0; kIdx < numIters; kIdx++) {
            uint32_t nextIdx = 1 - curIdx;
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            if (kIdx + 1 < numIters) {
                uint32_t blockTableId = (uint32_t)*(mapping + tokenIdx * mappingLen + kIdx + 1);
                CopyGmToL1Nd2Nz(l1bBuf[nextIdx], kGmBuf[blockTableId * blockMemSize + headOffset],
                                blockSize, headSize, nKVHeads * headSize, blockSize);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + nextIdx);
            }
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + curIdx);
            uint32_t cubeSize = 512 / sizeof(Dtype);
            uint32_t nBlockNum = headSize / CUBE_BLOCK_SIZE;
            uint32_t kBlockNum = blockSize / CUBE_BLOCK_SIZE;
            LoadData2dParams params(0, static_cast<uint8_t>(nBlockNum), 1, 0, 0, 0, inc);
            for (int k = 0; k < kBlockNum; k++) {
                LoadData(l0bBuf[curIdx][k * nBlockNum * cubeSize], l1bBuf[curIdx][k * nBlockNum * cubeSize], params);
            }
            SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            CalMmad(l0cBuf, l0aBuf, l0bBuf[curIdx], CUBE_BLOCK_SIZE, blockSize, headSize, true, 0);

            if (kIdx == numIters - 1) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::M_MTE2>(EVENT_ID0);
            }
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            SetFlag<HardEvent::M_FIX>(EVENT_ID1);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID1);

            WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
            uint64_t config = 0x1;
            set_nd_para(config);
            int nSize = (kIdx + 1) == numIters ? contextLen - kIdx * blockSize : blockSize;
            CopyToGm(qkGmBuf[qHeadIdxStart * maxContextLen + kIdx * blockSize], l0cBuf,
                     headNumInGroup, nSize, CUBE_BLOCK_SIZE, maxContextLen);
            SetFlag<HardEvent::FIX_S>(EVENT_ID1);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            curIdx = 1 - curIdx;
        }
        WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
    }


    inline __aicore__ void RunAicSV()
    {
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);

        uint32_t off = MAX_CONTEXT_BLOCK_LEN * sizeof(Dtype) * CUBE_BLOCK_SIZE;
        for (uint32_t i = 0; i < PINGPONG_BUF_NUM; ++i) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = off;
            off += blockSize * headSize * sizeof(Dtype);
        }

        int blockNum = static_cast<int>(GetBlockNum());
        int totalTaskNum = nTokens * nKVHeads;
        int startOffset = totalTaskNum % blockNum;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        for (int idx = static_cast<int>(GetBlockIdx()) - startOffset; idx < totalTaskNum; idx += blockNum) {
            if (idx < 0) {
                continue;
            }
            uint32_t kvHeadIdx = idx % nKVHeads;
            uint32_t tokenIdx = idx / nKVHeads;
            uint32_t realTokenIdx = (uint32_t)*(decodeIndex + tokenIdx);
            uint32_t cumM = (uint32_t)*(cumPromptLen + realTokenIdx);
            if (mSlice > 0 && (cumM >= mOffset + mSlice || cumM < mOffset)) {
                continue;
            }
            uint32_t qOffset = idx * headNumInGroup;
            uint32_t headOffset = kvHeadIdx * headSize;
            WaitAicAivFlag(v2a, headNumInGroup, qOffset);

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            DataCopyParams repeatParams(MAX_CONTEXT_REPEAT_TIMES, 1, 0, 16 - 1);
            for (uint32_t i = 0; i < headNumInGroup; i++) {
                DataCopy(l1aBuf[i * CUBE_BLOCK_SIZE], qkGmBuf[(qOffset + i) * maxContextLen], repeatParams);
            }
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

            uint32_t blockTableId = (uint32_t)*(mapping + tokenIdx * mappingLen);
            CopyGmToL1Nd2Nz(l1bBuf[0], vGmBuf[blockTableId * blockMemSize + headOffset],
                            blockSize, headSize, nKVHeads * headSize, blockSize);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

            ComputeSV(realTokenIdx, headOffset, qOffset);

            SetFlag<HardEvent::FIX_S>(EVENT_ID1);
            WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
            CopyToGm(outGmBuf[cumM * nHeads * headSize + kvHeadIdx * headNumInGroup * headSize], l0cBuf,
                     headNumInGroup, headSize, CUBE_BLOCK_SIZE, headSize, 3);

            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            ResetAicAivFlag(v2a, headNumInGroup, qOffset);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        PipeBarrier<PIPE_ALL>();
    }


    inline __aicore__ void ComputeSV(uint32_t tokenIdx, uint32_t headOffset, uint32_t qOffset)
    {
        __gm__ int32_t *blockTableMap = mapping + tokenIdx * mappingLen;

        uint32_t contextLen = (uint32_t)*(cachedLens + tokenIdx) + 1;
        uint32_t curCtxBlock = 0;
        uint32_t maxCtxBlockNum = MAX_CONTEXT_BLOCK_LEN / blockSize;
        uint32_t kTileFactor = blockSize / CUBE_BLOCK_SIZE;
        uint32_t nTileFactor = headSize / CUBE_BLOCK_SIZE;

        uint32_t curIdx = 0;
        uint32_t unitFlag = 2;
        int numIters = DIV_ROUND_UP(contextLen, blockSize);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        for (int iter = 0; iter < numIters; iter++) {
            uint32_t nextIdx = 1 - curIdx;
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            if (iter + 1 < numIters) {
                uint32_t blockTableId = (uint32_t)*(blockTableMap + iter + 1);
                CopyGmToL1Nd2Nz(l1bBuf[nextIdx], vGmBuf[blockTableId * blockMemSize + headOffset],
                                blockSize, headSize, nKVHeads * headSize, blockSize);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + nextIdx);
            }
            if (iter == 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            }
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            CopyToL0ACol(l0aBuf, l1aBuf[CUBE_BLOCK_SIZE * (iter % maxCtxBlockNum) * blockSize],
                         headSize / CUBE_BLOCK_SIZE, 0, 1);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            if (iter + 1 < numIters) {
                uint32_t nextCtxBlock = (iter + 1) * blockSize / MAX_CONTEXT_BLOCK_LEN;
                if (nextCtxBlock != curCtxBlock) {
                    curCtxBlock = nextCtxBlock;
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + nextIdx);

                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
                    DataCopyParams repeatParams(MAX_CONTEXT_REPEAT_TIMES, 1, 0, 16 - 1);
                    uint32_t qkOffset = qOffset * maxContextLen + MAX_CONTEXT_BLOCK_LEN * nextCtxBlock;
                    for (int i = 0; i < headNumInGroup; i++) {
                        DataCopy(l1aBuf[i * CUBE_BLOCK_SIZE], qkGmBuf[qkOffset + i * maxContextLen], repeatParams);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + nextIdx);

                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
                }
            }
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + curIdx);
            CopyToL0BTCol(l0bBuf[curIdx], l1bBuf[curIdx], nTileFactor, 0, kTileFactor);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
            if (iter == numIters - 1) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                unitFlag = 3;
            }
            int kSize = (iter + 1) == numIters ? contextLen - iter * blockSize : blockSize;
            CalMmad(l0cBuf, l0aBuf, l0bBuf[curIdx], CUBE_BLOCK_SIZE, headSize, kSize, iter == 0, unitFlag);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            curIdx = 1 - curIdx;
        }
    }

#ifdef __DAV_C220_VEC__
    inline __aicore__ void RunAiv()
    {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        uint64_t maxNOneLoop = 0;
        if constexpr (std::is_same<Dtype, float16_t>::value) {
            maxNOneLoop = 32640;
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            maxNOneLoop = 24320;
        }

        uint32_t processNum = nTokens * nHeads;
        uint32_t aivBlockNum = block_num * 2;
        uint32_t id = get_block_idx() * 2 + get_subblockid();
        for (uint32_t process = id; process < processNum; process += aivBlockNum) {
            uint32_t tokenIdx = process / nHeads;
            uint32_t realTokenIdx = (uint32_t)*(decodeIndex + tokenIdx);
            uint32_t cumM = (uint32_t)*(cumPromptLen + realTokenIdx);
            if (mSlice > 0 && (cumM >= mOffset + mSlice || cumM < mOffset)) {
                continue;
            }
            __gm__ Dtype *qk = (__gm__ Dtype*)qkGmBuf.GetPhyAddr() + process * maxContextLen;
            uint32_t contextLen = (uint32_t)*(cachedLens + realTokenIdx) + 1;

            WaitAicAivFlag(a2v, 1, process);
            if (contextLen <= maxNOneLoop) {
                RunAivSoftmax<Dtype, CalcDtype>(qk, 1, ROUND_UP(contextLen, blockSize), contextLen);
            } else {
                RunAivSoftmaxLong<Dtype, CalcDtype>(qk, contextLen);
            }
            pipe_barrier(PIPE_ALL); // 此处的PIPE_ALL必须要，是用于核间同步的，保证结果写入到GM，如果用硬件同步可能可以去掉

            ResetAicAivFlag(a2v, 1, process);
            SetAicAivFlag(v2a, 1, process);
        }
        pipe_barrier(PIPE_ALL);
    }

#endif

private:
    inline __aicore__ void DataCacheCleanAndInvalid(__gm__ void *__restrict__ gm)
    {
        __asm__ __volatile__("");
        dcci(gm, 0 /*SINGLE_CACHE_LINE*/);
        __asm__ __volatile__("");
    }

    inline __aicore__ void WaitAicAivFlag(__gm__ uint32_t *flagGm, uint32_t headNum, uint32_t qOffset)
    {
        for (int subIdx = 0; subIdx < headNum; subIdx++) {
            uint32_t headIdx = qOffset + subIdx;
            __gm__ uint32_t *currAddr = flagGm + headIdx * headSize;
            DataCacheCleanAndInvalid(currAddr);
            uint32_t flag = *currAddr;
            while (flag != headIdx + 1) {
                DataCacheCleanAndInvalid(currAddr);
                flag = *currAddr;
            }
        }
    }

    inline __aicore__ void SetAicAivFlag(__gm__ uint32_t *flagGm, uint32_t headNum, uint32_t qOffset)
    {
        for (int subIdx = 0; subIdx < headNum; subIdx++) {
            uint32_t headIdx = qOffset + subIdx;
            __gm__ uint32_t *currAddr = flagGm + headIdx * headSize;
            *currAddr = headIdx + 1;
            DataCacheCleanAndInvalid(currAddr);
        }
    }

    inline __aicore__ void ResetAicAivFlag(__gm__ uint32_t *flagGm, uint32_t headNum, uint32_t qOffset)
    {
        for (int subIdx = 0; subIdx < headNum; subIdx++) {
            uint32_t headIdx = qOffset + subIdx;
            __gm__ uint32_t *currAddr = flagGm + headIdx * headSize;
            *currAddr = 0;
            DataCacheCleanAndInvalid(currAddr);
        }
    }

private:
    GlobalTensor<Dtype> qGmBuf;
    GlobalTensor<Dtype> kGmBuf;
    GlobalTensor<Dtype> vGmBuf;
    GlobalTensor<Dtype> qkGmBuf;
    GlobalTensor<Dtype> outGmBuf;
    __gm__ uint32_t *v2a;
    __gm__ uint32_t *a2v;
    __gm__ int32_t *cumPromptLen;
    __gm__ int32_t *mapping;
    __gm__ int32_t *decodeIndex;
    __gm__ int32_t *cachedLens;

    uint32_t headSize;
    uint32_t nHeads;
    uint32_t nTokens;
    uint32_t nKVHeads;
    uint32_t blockSize;
    uint32_t mappingLen;
    uint32_t maxContextLen;
    uint32_t nQKVHeads;
    uint32_t mOffset;
    uint32_t mSlice;
    uint32_t headNumInGroup;
    uint32_t blockMemSize;

private:
    LocalTensor<Dtype> l1aBuf;
    LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0aBuf;
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;
#ifdef __DAV_C220_VEC__
    __ubuf__ Dtype *in;
    __ubuf__ Dtype *out;
    __ubuf__ CalcDtype *qkTemp;
    __ubuf__ CalcDtype *reduceSum;
    __ubuf__ CalcDtype *maxQK;
    __ubuf__ CalcDtype *calc;

    uint32_t maxSubCtxSize;
    uint32_t maxSubBlockIter;
    uint32_t maxSubCtxLen;
    CalcDtype calcMin;
    int32_t calcPad;
    int32_t srcPad;

#endif
};

#define DECODE_ATTN_FUNC_DEFINE(dtype, calcDtype) \
extern "C" __global__ __aicore__ void decode_att_##dtype(GM_ADDR a2v, GM_ADDR v2a, GM_ADDR q, GM_ADDR k, GM_ADDR v, \
                                                         GM_ADDR cachedLens, GM_ADDR mapping, GM_ADDR qk, GM_ADDR o, \
                                                         GM_ADDR decodeIndex, GM_ADDR cumPromptLen, \
                                                         uint32_t numTokens, uint32_t numHeads, uint32_t headSize, \
                                                         uint32_t blockSize, uint32_t mappingLen, uint32_t numKVHeads, \
                                                         uint32_t maxContextLen, uint32_t numQKVHeads, \
                                                         int32_t mOffset, uint32_t mSlice) \
{ \
    DecodeAttn<dtype, calcDtype> op; \
    op.Init(a2v, v2a, q, k, v, cachedLens, mapping, qk, o, decodeIndex, cumPromptLen, \
            numTokens, numHeads, headSize, blockSize, mappingLen, numKVHeads, maxContextLen, \
            numQKVHeads, mOffset, mSlice); \
    op.Run(); \
}

DECODE_ATTN_FUNC_DEFINE(float16_t, float16_t);
DECODE_ATTN_FUNC_DEFINE(bfloat16_t, float);
