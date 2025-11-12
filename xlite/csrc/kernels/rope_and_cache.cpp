/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */

#include "kernel_operator.h"
#include "kernel_macro.h"

using namespace AscendC;

#define UB_SIZE 196608
#define PINGPONG_BUF_NUM 2

template <typename Dtype>
class RopeAndCache {
public:
    __aicore__ inline RopeAndCache() {}
    __aicore__ inline void Init(GM_ADDR positions, GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                GM_ADDR cossinCache, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR slotMapping,
                                uint32_t numTokens, uint32_t rotDim, uint32_t queryStride, uint32_t keyStride,
                                uint32_t valueStride, uint32_t numHeads, uint32_t numKVHeads,
                                uint32_t headDim, uint32_t blockSize, float scaleIn)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        this->queryGmBuf.SetGlobalBuffer((__gm__ Dtype *)query);
        this->keyGmBuf.SetGlobalBuffer((__gm__ Dtype *)key);
        this->valueGmBuf.SetGlobalBuffer((__gm__ Dtype *)value);
        this->cossinCacheGmBuf.SetGlobalBuffer((__gm__ Dtype *)cossinCache);
        this->keyCacheGmBuf.SetGlobalBuffer((__gm__ Dtype *)keyCache);
        this->valueCacheGmBuf.SetGlobalBuffer((__gm__ Dtype *)valueCache);
        this->posGmBuf.SetGlobalBuffer((__gm__ uint32_t *)positions);
        this->slotGmBuf.SetGlobalBuffer((__gm__ uint32_t *)slotMapping);

        this->numTokens = numTokens;
        this->rotDim = rotDim;
        this->queryStride = queryStride;
        this->keyStride = keyStride;
        this->valueStride = valueStride;
        this->numHeads = numHeads;
        this->numKVHeads = numKVHeads;
        this->headDim = headDim;
        this->blockSize = blockSize;
        this->scaleFp32 = scaleIn;
        this->embedDim = rotDim / 2;
        this->qSize = numHeads * headDim;
        this->kvSize = numKVHeads * headDim;
        this->slotBlockSize = kvSize * blockSize;

        uint32_t qBlockSize = qSize * sizeof(Dtype) + embedDim * sizeof(Dtype);
        uint32_t kvBlockSize = kvSize * sizeof(Dtype) + embedDim * sizeof(Dtype);
        uint32_t cossinBlockSize = rotDim * sizeof(Dtype);

        uint64_t off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            queryBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            queryBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qBlockSize;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            keyBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            keyBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kvBlockSize;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            valueBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            valueBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kvBlockSize;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            cosBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            cosBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += cossinBlockSize;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            sinBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            sinBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += cossinBlockSize;
        }

        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            off = ROUND_UP(off, BLOCK_SIZE);
            uint32_t qFp32BlockSize = ROUND_UP(qSize * sizeof(float) + embedDim * sizeof(float), BLOCK_SIZE);
            uint32_t kvFp32BlockSize = ROUND_UP(kvSize * sizeof(float) + embedDim * sizeof(float), BLOCK_SIZE);
            uint32_t cossinFp32BlockSize = ROUND_UP(rotDim * sizeof(float), BLOCK_SIZE);

            // query/key calc ubuf (Dtype)
            queryCalcBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            queryCalcBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += MAX(qBlockSize, kvBlockSize);

            for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
                queryFp32Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
                queryFp32Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
                off += qFp32BlockSize;
            }
            for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
                keyFp32Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
                keyFp32Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
                off += kvFp32BlockSize;
            }
            for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
                valueFp32Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
                valueFp32Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
                off += kvFp32BlockSize;
            }
            for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
                cosFp32Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
                cosFp32Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
                off += cossinFp32BlockSize;
            }
            for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
                sinFp32Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
                sinFp32Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
                off += cossinFp32BlockSize;
            }

            // query calc ubuf (Fp32)
            queryCalcFp32Buf.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            queryCalcFp32Buf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qFp32BlockSize;
            // key calc ubuf (Fp32)
            keyCalcFp32Buf.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            keyCalcFp32Buf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kvFp32BlockSize;
        } else {
            this->scale = (Dtype)scaleIn;
            // query calc ubuf
            queryCalcBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            queryCalcBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qBlockSize;
            // key calc ubuf
            keyCalcBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
            keyCalcBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kvBlockSize;
        }

        // pos and slot ubuf
        this->iterPosSlotNum = (UB_SIZE - off) / (sizeof(uint32_t) * 2);
        this->iterPosSlotNum = MIN(numTokens, this->iterPosSlotNum);
        this->posslotIters = DIV_ROUND_UP(numTokens, this->iterPosSlotNum);
        this->posslotBlockSize = ROUND_UP(this->iterPosSlotNum * sizeof(uint32_t), BLOCK_SIZE);
        posBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        posBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += this->posslotBlockSize;
        slotBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        slotBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void CopyPosSlotGmToUbuf(LocalTensor<uint32_t> dst, GlobalTensor<uint32_t> src, int numElem)
    {
        int blockLen = DIV_ROUND_UP(numElem, BLOCK_SIZE);
        DataCopyParams repeatParams(1, blockLen, 0, 0);
        DataCopy(dst, src, repeatParams);
    }

    __aicore__ inline void CopyGmToUbuf(LocalTensor<Dtype> dst, GlobalTensor<Dtype> src, int blockCount, int numElem, uint16_t dstStride)
    {
        int blockLen = DIV_ROUND_UP(numElem * sizeof(Dtype), BLOCK_SIZE);
        DataCopyParams repeatParams(blockCount, blockLen, 0, dstStride);
        DataCopy(dst, src, repeatParams);
    }

    __aicore__ inline void CopyUbufToGm(GlobalTensor<Dtype> dst, LocalTensor<Dtype> src, int blockCount, int numElem, uint16_t dstStride)
    {
        int blockLen = DIV_ROUND_UP(numElem * sizeof(Dtype), BLOCK_SIZE);
        DataCopyParams repeatParams(blockCount, blockLen, 0, dstStride);
        DataCopy(dst, src, repeatParams);
    }

    __aicore__ inline void CalcCosSin(GlobalTensor<Dtype> gmBufLoop, LocalTensor<Dtype> localBuf,
                                      LocalTensor<Dtype> calcBuf, uint32_t calcSize, int eventId)
    {
        uint64_t repeatTimes = DIV_ROUND_UP(calcSize, VECTOR_MAX_BYTESIZE / sizeof(Dtype));

        // copy query/key
        CopyGmToUbuf(localBuf, gmBufLoop, 1, calcSize, 0);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        // q * sin (q : numHeads * headDim  sin: 1 * headDim)
        SetMaskNorm();
        SetVectorMask<Dtype, MaskMode::NORMAL>(UINT64_MAX, UINT64_MAX);
        Mul<Dtype, false>(calcBuf, localBuf, this->sinBuf[eventId], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 8, 8, 0});

        // q * cos (q : numHeads * headDim  cos: 1 * headDim)
        Mul<Dtype, false>(localBuf, localBuf, this->cosBuf[eventId], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 8, 8, 0});

        // q * cos - q * sin > q[:half] * cos - q[half:] * sin
        SetVectorMask<Dtype, MaskMode::NORMAL>(0, UINT64_MAX);
        Sub<Dtype, false>(localBuf, localBuf, calcBuf[this->embedDim], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 8, 8, 8});

        // q * cos + q * sin > q[half:] * cos + q[:half] * sin
        Add<Dtype, false>(localBuf[this->embedDim], localBuf[this->embedDim], calcBuf, MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 8, 8, 8});

        ResetMask();
    }

    __aicore__ inline void CalcCosSinCast(GlobalTensor<Dtype> gmBufLoop, LocalTensor<Dtype> localBuf,
                                          LocalTensor<float> localFp32Buf, LocalTensor<float> calcFp32Buf,
                                          uint32_t calcSize, int eventId)
    {
        uint64_t repeatTimes = calcSize / this->headDim;

        // copy query/key
        CopyGmToUbuf(localBuf, gmBufLoop, 1, calcSize, 0);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        Cast(localFp32Buf, localBuf, RoundMode::CAST_NONE, calcSize);
        PipeBarrier<PIPE_V>();

        Cast(this->sinFp32Buf[eventId], this->sinBuf[eventId], RoundMode::CAST_NONE, this->rotDim);
        PipeBarrier<PIPE_V>();

        Cast(this->cosFp32Buf[eventId], this->cosBuf[eventId], RoundMode::CAST_NONE, this->rotDim);
        PipeBarrier<PIPE_V>();

        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(0, UINT64_MAX);

        // q * sin (q : numHeads * headDim  sin: 1 * headDim)
        Mul<float, false>(calcFp32Buf, localFp32Buf, this->sinFp32Buf[eventId], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 16, 16, 0});
        Mul<float, false>(calcFp32Buf[this->embedDim], localFp32Buf[this->embedDim], this->sinFp32Buf[eventId][this->embedDim], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 16, 16, 0});

        // q * cos (q : numHeads * headDim  cos: 1 * headDim)
        Mul<float, false>(localFp32Buf, localFp32Buf, this->cosFp32Buf[eventId], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 16, 16, 0});
        Mul<float, false>(localFp32Buf[this->embedDim], localFp32Buf[this->embedDim], this->cosFp32Buf[eventId][this->embedDim], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 16, 16, 0});

        Cast(this->queryCalcBuf, calcFp32Buf, RoundMode::CAST_RINT, calcSize);
        PipeBarrier<PIPE_V>();

        Cast(localBuf, localFp32Buf, RoundMode::CAST_RINT, calcSize);
        PipeBarrier<PIPE_V>();

        Cast(calcFp32Buf, this->queryCalcBuf, RoundMode::CAST_NONE, calcSize);
        PipeBarrier<PIPE_V>();

        Cast(localFp32Buf, localBuf, RoundMode::CAST_NONE, calcSize);
        PipeBarrier<PIPE_V>();

        // q * cos - q * sin > q[:half] * cos - q[half:] * sin
        Sub<float, false>(localFp32Buf, localFp32Buf, calcFp32Buf[this->embedDim], MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 16, 16, 16});

        // q * cos + q * sin > q[half:] * cos + q[:half] * sin
        Add<float, false>(localFp32Buf[this->embedDim], localFp32Buf[this->embedDim], calcFp32Buf, MASK_PLACEHOLDER, repeatTimes, {1, 1, 1, 16, 16, 16});
        PipeBarrier<PIPE_V>();

        Cast(localBuf, localFp32Buf, RoundMode::CAST_RINT, calcSize);
        PipeBarrier<PIPE_V>();
        ResetMask();
    }

    __aicore__ inline void CalcCosSinKey(GlobalTensor<Dtype> gmBufLoop, int eventId)
    {
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            CalcCosSinCast(gmBufLoop, this->keyBuf[eventId], this->keyFp32Buf[eventId],
                           this->keyCalcFp32Buf, this->kvSize, eventId);
        } else {
            CalcCosSin(gmBufLoop, this->keyBuf[eventId], this->keyCalcBuf, this->kvSize, eventId);
        }
    }

    __aicore__ inline void CalcCosSinQuery(GlobalTensor<Dtype> gmBufLoop, int eventId)
    {
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            CalcCosSinCast(gmBufLoop, this->queryBuf[eventId], this->queryFp32Buf[eventId],
                           this->queryCalcFp32Buf, this->qSize, eventId);
            Cast(this->queryFp32Buf[eventId], this->queryBuf[eventId], RoundMode::CAST_NONE, this->qSize);
            PipeBarrier<PIPE_V>();
            Muls(this->queryFp32Buf[eventId], this->queryFp32Buf[eventId], this->scaleFp32, this->qSize);
            PipeBarrier<PIPE_V>();
            Cast(this->queryBuf[eventId], this->queryFp32Buf[eventId], RoundMode::CAST_RINT, this->qSize);
            PipeBarrier<PIPE_V>();
        } else {
            CalcCosSin(gmBufLoop, this->queryBuf[eventId], this->queryCalcBuf, this->qSize, eventId);
            Muls(this->queryBuf[eventId], this->queryBuf[eventId], this->scale, this->qSize);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t cosShift, sinShift, slotblockIdx, slotblockShift, slotStartIdx, posslotIdx;
        uint32_t processedNumTokens, posslotCopyByteSize, iterPosSlotNumLoop;
        int pingpong = 1;

        for (uint32_t loop0 = 0; loop0 < this->posslotIters; loop0 += 1) {
            processedNumTokens = loop0 * this->iterPosSlotNum;
            posslotCopyByteSize = this->posslotBlockSize;
            iterPosSlotNumLoop = this->iterPosSlotNum;
            if (loop0 == this->posslotIters - 1) {
                iterPosSlotNumLoop = this->numTokens - processedNumTokens;
                posslotCopyByteSize = ROUND_UP(iterPosSlotNumLoop * sizeof(uint32_t), BLOCK_SIZE);
            }
            CopyPosSlotGmToUbuf(this->posBuf, this->posGmBuf[processedNumTokens], posslotCopyByteSize);
            CopyPosSlotGmToUbuf(this->slotBuf, this->slotGmBuf[processedNumTokens], posslotCopyByteSize);
            PipeBarrier<PIPE_ALL>();

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t loop1 = (GetBlockIdx() + processedNumTokens); loop1 < (processedNumTokens + iterPosSlotNumLoop); loop1 += GetBlockNum()) {
                auto eventId = EVENT_ID0 + pingpong;
                posslotIdx = loop1 - processedNumTokens;

                // qkv
                auto queryGmBufLoop = this->queryGmBuf[loop1 * this->queryStride];
                auto keyGmBufLoop = this->keyGmBuf[loop1 * this->keyStride];
                auto valueGmBufLoop = this->valueGmBuf[loop1 * this->valueStride];
                // cos sin
                cosShift = this->posBuf.GetValue(posslotIdx) * this->rotDim;
                sinShift = cosShift + this->embedDim;
                auto cosGmBufLoop = this->cossinCacheGmBuf[cosShift];
                auto sinGmBufLoop = this->cossinCacheGmBuf[sinShift];
                // kcache vcache
                slotblockIdx = this->slotBuf.GetValue(posslotIdx) / this->blockSize;
                slotblockShift = this->slotBuf.GetValue(posslotIdx) % this->blockSize;
                slotStartIdx = slotblockIdx * this->slotBlockSize + slotblockShift * this->headDim;
                auto kcacheGmBufLoop = this->keyCacheGmBuf[slotStartIdx];
                auto vcacheGmBufLoop = this->valueCacheGmBuf[slotStartIdx];

                SetFlag<HardEvent::S_MTE2>(eventId);
                WaitFlag<HardEvent::S_MTE2>(eventId);

                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                // Cache value
                CopyGmToUbuf(this->valueBuf[eventId], valueGmBufLoop, 1, this->kvSize, 0);
                SetFlag<HardEvent::MTE2_MTE3>(eventId);
                WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                CopyUbufToGm(vcacheGmBufLoop, this->valueBuf[eventId], this->numKVHeads, this->headDim,
                             DIV_ROUND_UP((this->blockSize - 1) * this->headDim * sizeof(Dtype), BLOCK_SIZE));
                
                // Copy cos and sin, repeat for d/2
                CopyGmToUbuf(this->cosBuf[eventId], cosGmBufLoop, 1, this->embedDim, 0);
                CopyGmToUbuf(this->cosBuf[eventId][this->embedDim], cosGmBufLoop, 1, this->embedDim, 0);
                CopyGmToUbuf(this->sinBuf[eventId], sinGmBufLoop, 1, this->embedDim, 0);
                CopyGmToUbuf(this->sinBuf[eventId][this->embedDim], sinGmBufLoop, 1, this->embedDim, 0);

                // Calc key
                CalcCosSinKey(keyGmBufLoop, eventId);
                SetFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);

                // Update key and cache key
                CopyUbufToGm(keyGmBufLoop, this->keyBuf[eventId], 1, this->kvSize, 0);
                CopyUbufToGm(kcacheGmBufLoop, this->keyBuf[eventId], this->numKVHeads, this->headDim,
                             DIV_ROUND_UP((this->blockSize - 1) * this->headDim * sizeof(Dtype), BLOCK_SIZE));

                // Calc query
                CalcCosSinQuery(queryGmBufLoop, eventId);
                SetFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);

                // Update query
                CopyUbufToGm(queryGmBufLoop, this->queryBuf[eventId], 1, this->qSize, 0);

                SetFlag<HardEvent::MTE3_MTE2>(eventId);
                pingpong ^= 1;
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<Dtype> queryGmBuf;
    GlobalTensor<Dtype> keyGmBuf;
    GlobalTensor<Dtype> valueGmBuf;
    GlobalTensor<Dtype> cossinCacheGmBuf;
    GlobalTensor<Dtype> keyCacheGmBuf;
    GlobalTensor<Dtype> valueCacheGmBuf;
    GlobalTensor<uint32_t> posGmBuf;
    GlobalTensor<uint32_t> slotGmBuf;
    LocalTensor<Dtype> queryBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> keyBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> valueBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> cosBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> sinBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> queryCalcBuf;
    LocalTensor<Dtype> keyCalcBuf;
    LocalTensor<float> queryFp32Buf[PINGPONG_BUF_NUM];
    LocalTensor<float> keyFp32Buf[PINGPONG_BUF_NUM];
    LocalTensor<float> valueFp32Buf[PINGPONG_BUF_NUM];
    LocalTensor<float> cosFp32Buf[PINGPONG_BUF_NUM];
    LocalTensor<float> sinFp32Buf[PINGPONG_BUF_NUM];
    LocalTensor<float> queryCalcFp32Buf;
    LocalTensor<float> keyCalcFp32Buf;
    LocalTensor<uint32_t> posBuf;
    LocalTensor<uint32_t> slotBuf;
    uint32_t numTokens;
    uint32_t rotDim;
    uint32_t queryStride;
    uint32_t keyStride;
    uint32_t valueStride;
    uint32_t numHeads;
    uint32_t numKVHeads;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t embedDim;
    uint32_t iterPosSlotNum;
    uint32_t posslotBlockSize;
    uint32_t posslotIters;
    uint32_t slotBlockSize;
    uint32_t kvSize;
    uint32_t qSize;
    float scaleFp32;
    Dtype scale;
};


#define ROPEANDCACHE_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void rope_and_cache_##dtype( \
                GM_ADDR positions, GM_ADDR query, GM_ADDR key, GM_ADDR value, \
                GM_ADDR cossinCache, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR slotMapping, \
                uint32_t numTokens, uint32_t rotDim, uint32_t queryStride, uint32_t keyStride, uint32_t valueStride, \
                uint32_t numHeads, uint32_t numKVHeads, uint32_t headDim, uint32_t blockSize, float scaleIn) \
{ \
    RopeAndCache<dtype> op; \
    op.Init(positions, query, key, value, cossinCache, keyCache, valueCache, slotMapping, numTokens, \
            rotDim, queryStride, keyStride, valueStride, numHeads, numKVHeads, headDim, blockSize, scaleIn); \
    op.Process(); \
}

ROPEANDCACHE_FUNC_DEFINE(float16_t);
ROPEANDCACHE_FUNC_DEFINE(bfloat16_t);