/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY of even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once
#include "kernel_macro.h"

// Cube-side QK/SV matmul helper shared by attention / flash_attention.
// Owns the L1/L0 buffers and lays out their addresses in Init(); the GM KV
// tensors are passed per-call to RunAicQK/RunAicSV. Instantiated as a member
// (composition, like RingSync).
template <typename Dtype>
class AicHelper
{
public:
    __aicore__ inline AicHelper()
    {
    }

    __aicore__ inline void Init(uint32_t nHeadsV, uint32_t nKVHeadsV, uint32_t headNumInGroupV,
                                uint32_t headSizeV, uint32_t blockSizeV, uint32_t qkStrideV)
    {
        nHeads = nHeadsV;
        nKVHeads = nKVHeadsV;
        headNumInGroup = headNumInGroupV;
        headSize = headSizeV;
        blockSize = blockSizeV;
        kvMemSize = nKVHeads * headSize;
        qkvMemSize = (nHeads + 2 * nKVHeads) * headSize;
        blockMemSize = blockSize * kvMemSize;
        qkStride = qkStrideV;

        // Shrink the Cube KV tile so 2 * tile * headSize * sizeof(Dtype) fits in
        // 64KB L0 with pingpong. Not tied to blockSize/headDim — otherwise
        // headDim=256 & blockSize=128 fills L0 with one buffer and pingpong
        // overflows (CCU address check).
        const uint64_t dtypeBytes = sizeof(Dtype);
        cubeKvTile = blockSize;
        while (cubeKvTile > static_cast<uint32_t>(NBLOCKSIZE) &&
               (2ull * cubeKvTile * headSize * dtypeBytes > ASCEND_L0_BYTES ||
                2ull * XLITE_MAX_M0 * cubeKvTile * dtypeBytes > ASCEND_L0_BYTES)) {
            cubeKvTile >>= 1;
        }

        uint64_t l1ATileBytes = static_cast<uint64_t>(XLITE_MAX_M0) *
                                (headSize > cubeKvTile ? headSize : cubeKvTile) * dtypeBytes;
        uint64_t l1BTileBytes = static_cast<uint64_t>(cubeKvTile) * headSize * dtypeBytes;
        uint64_t l0aSvBytes = static_cast<uint64_t>(XLITE_MAX_M0) * cubeKvTile * dtypeBytes;
        uint64_t l0bBytes = l1BTileBytes;

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
        // L0A: buf0 sized for Q (M0 x headSize). SV pingpong places buf1 at
        // l0aSvBytes inside the same 64KB — QK and SV are sequential phases.
        off = 0;
        l0aBuf[0].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf[0].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        l0aBuf[1].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf[1].address_.bufferAddr = reinterpret_cast<uint64_t>(off + l0aSvBytes);
        // L0B: full pingpong on cubeKvTile x headSize tiles.
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0bBytes;
        }
        off = 0;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    /*
     * m: tokens (queryLen * headNumInGroup)
     * n: headSize
     * k: cachedTokens (kvLen, from kvOffset)
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> query, GlobalTensor<Dtype> kCache,
                                    int queryLen, int kvHeadIdx, __gm__ uint32_t *blockTable,
                                    int kvOffset, int kvLen, GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int kBlockNum = DIV_ROUND_UP(headSize, kBlockSize);
        int kvHeadOffset = kvHeadIdx * headSize;
        int tile = static_cast<int>(cubeKvTile);

        Nd2NzParams nd2nzParams(1 /* NdNum */, queryLen /* nValue */, headSize /* dValue */,
                                0 /* srcNdMatrixStride */, qkvMemSize /* srcDValue */,
                                mBlockPad /* dstNzC0Stride */, headNumInGroup /* dstNzNStride */,
                                0 /* dstNzMatrixStride */);
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
        for (int local = 0; local < kvLen; local += tile) {
            int absPos = kvOffset + local;
            int rowInBlock = absPos % static_cast<int>(blockSize);
            int nSize = tile;
            if (rowInBlock + nSize > static_cast<int>(blockSize)) {
                nSize = static_cast<int>(blockSize) - rowInBlock;
            }
            if (local + nSize > kvLen) {
                nSize = kvLen - local;
            }
            int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
            int nBlockNum = nBlockPad / NBLOCKSIZE;
            uint32_t block = blockTable[absPos / static_cast<int>(blockSize)];

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx],
                            kCache[block * blockMemSize + rowInBlock * kvMemSize + kvHeadOffset],
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
            CopyToGm(qk[local], l0cBuf, mActual, nSize, mBlockPad, qkStride);
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
     * m: tokens (queryLen * headNumInGroup)
     * n: headSize
     * k: cachedTokens (kvLen, from kvOffset)
     * gqaScatter: true = attention-style per-query scattered write (dstStride = nHeads*headSize);
     *             false = flash-style compact write (dstStride = headSize)
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, GlobalTensor<Dtype> vCache,
                                    int queryLen, int kvHeadIdx, __gm__ uint32_t *blockTable,
                                    int kvOffset, int kvLen, GlobalTensor<Dtype> out,
                                    bool gqaScatter)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nBlockNum = DIV_ROUND_UP(headSize, NBLOCKSIZE);
        int kvHeadOffset = kvHeadIdx * headSize;
        int tile = static_cast<int>(cubeKvTile);

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
        int first = 1;
        for (int local = 0; local < kvLen; local += tile) {
            int absPos = kvOffset + local;
            int rowInBlock = absPos % static_cast<int>(blockSize);
            int kSize = tile;
            if (rowInBlock + kSize > static_cast<int>(blockSize)) {
                kSize = static_cast<int>(blockSize) - rowInBlock;
            }
            if (local + kSize > kvLen) {
                kSize = kvLen - local;
            }
            int kBlockPad = ROUND_UP(kSize, kBlockSize);
            int kBlockNum = kBlockPad / kBlockSize;
            uint32_t block = blockTable[absPos / static_cast<int>(blockSize)];

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curIdx);
            CopyGmToL1Nd2Nz(l1aBuf[curIdx], qk[local], mActual, kBlockPad, qkStride, mBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curIdx);

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx],
                            vCache[block * blockMemSize + rowInBlock * kvMemSize + kvHeadOffset],
                            kBlockPad, headSize, kvMemSize, kBlockPad);
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
            CalMmad(l0cBuf, l0aBuf[curIdx], l0bBuf[curIdx], mBlockPad, headSize, kBlockPad,
                    first != 0);
            first = 0;
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
        if (gqaScatter) {
            int outStride = static_cast<int>(nHeads * headSize);
            if (headNumInGroup == 1) {
                CopyToGm(out, l0cBuf, queryLen, headSize, mBlockPad, outStride);
            } else {
                int l0cOffset = headNumInGroup * kBlockSize;
                for (int i = 0; i < queryLen; i++) {
                    CopyToGm(out[i * outStride], l0cBuf[i * l0cOffset], headNumInGroup, headSize,
                             mBlockPad, headSize);
                }
            }
        } else {
            CopyToGm(out, l0cBuf, mActual, headSize, mBlockPad, static_cast<int>(headSize));
        }
    }

private:
    LocalTensor<Dtype> l1aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;
    uint32_t nHeads;
    uint32_t nKVHeads;
    uint32_t headNumInGroup;
    uint32_t headSize;
    uint32_t blockSize;
    uint32_t cubeKvTile;
    uint32_t kvMemSize;
    uint32_t blockMemSize;
    uint32_t qkvMemSize;
    uint32_t qkStride;
};
