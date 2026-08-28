/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"

#ifndef MAX_N0
#define MAX_N0 128
#endif

template <typename Dtype>
class CxaAicHelper
{
public:
    __aicore__ inline CxaAicHelper() = default;

    __aicore__ inline int Init(uint32_t nHeadsV, uint32_t headDimV, uint32_t swaBlockSizeV,
                               uint32_t compressBlockSizeV, uint32_t windowSizeV,
                               uint32_t compressRatioV, uint32_t qkStrideV, uint32_t swaSegWidthV,
                               bool denseV)
    {
        nHeads = nHeadsV;
        headDim = headDimV;
        swaBlockSize = swaBlockSizeV;
        compressBlockSize = compressBlockSizeV;
        windowSize = windowSizeV;
        compressRatio = compressRatioV;
        qkStride = qkStrideV;
        swaSegWidth = swaSegWidthV;
        dense = denseV;

        qkwn0 = swaBlockSize;
        qkcn0 = dense ? MAX_N0 : compressBlockSizeV;
        qkk0 = 256 / sizeof(Dtype);
        svn0 = 256;
        svwk0 = swaBlockSize;
        svck0 = CXA_SVCK0;
        uint64_t off = 0;

        // QK
        uint64_t qSize = XLITE_MAX_M0 * headDim * sizeof(Dtype);
        ql1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        ql1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qSize;

        uint64_t kSize = (qkwn0 > qkcn0 ? qkwn0 : qkcn0) * 4 * qkk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kSize;
        }
        uint64_t total_qk = off;
        dbg_printf("QK buf: qSize %lu, kSize %lu x 2, total %lu\n", qSize, kSize, total_qk);

        off = 0;
        uint64_t l0aSize = XLITE_MAX_M0 * qkk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            qkl0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0aSize;
        }

        off = 0;
        uint64_t l0bSize = MAX_N0 * qkk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            qkl0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0bSize;
        }

        off = 0;
        uint64_t l0cSize = XLITE_MAX_M0 * MAX_N0 * sizeof(float);
        qkl0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        qkl0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += l0cSize;

        int svwkMax = svwk0 > svck0 ? svwk0 : svck0;

        off = 0;
        uint64_t scoresL1Size = XLITE_MAX_M0 * 4 * svwkMax * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            scoresl1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            scoresl1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += scoresL1Size;
        }
        uint64_t ktL1Size = (svwk0 > 2 * svck0 ? svwk0 : 2 * svck0) * svn0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            ktl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            ktl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ktL1Size;
        }
        uint64_t total_sv_l1 = off;
        dbg_printf("SV L1 buf: scoresL1Size %lu x 2, ktL1Size %lu x 2, total %lu\n", scoresL1Size,
                   ktL1Size, total_sv_l1);

        off = 0;
        uint64_t svL0aSize = XLITE_MAX_M0 * svwkMax * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            svl0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            svl0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += svL0aSize;
        }

        off = 0;
        uint64_t svL0bSize = svn0 * svwkMax * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            svl0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            svl0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += svL0bSize;
        }

        off = 0;
        uint64_t svL0cSize = XLITE_MAX_M0 * svn0 * sizeof(float);
        svl0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        svl0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += svL0cSize;

        return svck0;
    }

    /*
     * scores = Q * K
     *          Q: (queryTokens * nHeads, headDim)
     *          K: (windowSize + kvLen, headDim)
     *          m0: XLITE_MAX_M0, n0: qkwn0 qkcn0, k0: qkk0
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> q, GlobalTensor<Dtype> swaKCache,
                                    GlobalTensor<Dtype> compressKCache,
                                    __gm__ uint32_t *swaBlockTables,
                                    __gm__ uint32_t *compressBlockTables, uint32_t queryOffset,
                                    uint32_t queryLen, uint32_t kvOffset, uint32_t kvLen,
                                    GlobalTensor<Dtype> scores)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mSize = queryLen * nHeads;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int totalLen = kvOffset + kvLen;

        int curr = 0;
        int pingpongL1B = 0;

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // copy Q (queryTokens * nHeads, headDim) to L1
        CopyGmToL1Nd2Nz(ql1aBuf, q, mSize, headDim, headDim, mBlockPad);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        // window [alignStart, windowEnd]. alignStart is windowStart rounded down to a
        // kBlockSize multiple so scores column 0 (= alignStart) keeps every tile's nSize
        // a multiple of kBlockSize: no column straddles a tile boundary, so the SV path
        // cannot double-count the boundary token. The lead-in columns
        // [0, windowStart - alignStart) are masked to -inf by RunAivSoftmax.
        int windowStart =
            (int)queryOffset > (int)windowSize - 1 ? (int)queryOffset - (int)windowSize + 1 : 0;
        int windowEnd = queryOffset + queryLen - 1;
        int alignStart = ROUND_DOWN(windowStart, kBlockSize);
        int nwIdxStart = alignStart / qkwn0;
        int nwIdxEnd = windowEnd / qkwn0;
        int nwBlockOffset = alignStart % swaBlockSize;
        int nwOffset = alignStart % qkwn0;
        int kSize = qkk0;
        int kBlockPad = qkk0;
        int kBlockNum = qkk0 / kBlockSize;
        int kLoop = DIV_ROUND_UP(headDim, qkk0);
        if (windowSize != 0) {
            for (int nIdx = nwIdxStart; nIdx < nwIdxEnd + 1; nIdx++) {  // window size
                uint32_t block = swaBlockTables[nIdx];
                uint32_t blockOffset = nIdx == nwIdxStart ? nwBlockOffset : 0;
                int nOffset = nIdx * qkwn0;
                // first tile clipped by alignStart, last tile by totalLen
                int nSize = qkwn0;
                if (nIdx == nwIdxStart) {
                    nOffset += nwOffset;
                    nSize = qkwn0 - nwOffset;
                }
                if (nOffset + nSize > totalLen) {
                    nSize = totalLen - nOffset;
                }
                int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                int nBlockNum = nBlockPad / NBLOCKSIZE;

                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

                kSize = qkk0;
                kBlockPad = qkk0;
                kBlockNum = qkk0 / kBlockSize;
                for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // headDim
                    int kIdx4 = kIdx % 4;
                    int kOffset = kIdx * qkk0;
                    if (kOffset + kSize > headDim) {
                        kSize = headDim - kOffset;
                        kBlockPad = ROUND_UP(kSize, kBlockSize);
                        kBlockNum = kBlockPad / kBlockSize;
                    }
                    // copy SWK (qkwn0, 4 * qkk0) to L1
                    if (kIdx4 == 0) {
                        int kRemSize = 4 * qkk0;
                        if (kOffset + kRemSize > headDim) {
                            kRemSize = headDim - kOffset;
                        }
                        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                        CopyGmToL1Nd2Nz(
                            kl1bBuf[pingpongL1B],
                            swaKCache[(block * qkwn0 + blockOffset) * headDim + kOffset], nSize,
                            kRemSize, headDim, nBlockPad);
                        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                    }

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    CopyToL0BCol(qkl0bBuf[curr], kl1bBuf[pingpongL1B], nBlockNum,
                                 kIdx4 * qkk0 / kBlockSize, kBlockNum);
                    if (kIdx4 == 3) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                        pingpongL1B ^= 1;
                    }
                    CopyToL0ACol(qkl0aBuf[curr], ql1aBuf[kOffset * mBlockPad], mBlockNum, 0,
                                 kBlockNum);

                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                    // mmad scores(queryTokens * nHeads, qkwn0) = Q * SWK
                    CalMmad(qkl0cBuf, qkl0aBuf[curr], qkl0bBuf[curr], mBlockPad, nBlockPad,
                            kBlockPad, kIdx == 0);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    PipeBarrier<PIPE_M>();
                    curr = 1 - curr;
                }
                if (kLoop % 4 != 0) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    pingpongL1B ^= 1;
                }

                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
                // copy scores (queryTokens * nHeads, qkwn0) from L0C to GM
                CopyToGm(scores[nOffset - alignStart], qkl0cBuf, mSize, nSize, mBlockPad, qkStride);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
        }

        if (compressRatio != 0) {
            int ncTotalLen = kvLen / compressRatio;
            int nIdxStart = kvOffset / qkcn0;
            int nSize = qkcn0;
            int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
            int nBlockNum = nBlockPad / NBLOCKSIZE;
            int nLoop = DIV_ROUND_UP(ncTotalLen, qkcn0);
            for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // compress kv len
                uint32_t block = !dense ? compressBlockTables[nIdx + nIdxStart] : nIdx + nIdxStart;
                int nOffset = nIdx * qkcn0;
                nSize = qkcn0;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
                if (nOffset + nSize > ncTotalLen) {
                    nSize = ncTotalLen - nOffset;
                    nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                    nBlockNum = nBlockPad / NBLOCKSIZE;
                }

                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

                kSize = qkk0;
                kBlockPad = qkk0;
                kBlockNum = qkk0 / kBlockSize;
                for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // headDim
                    int kIdx4 = kIdx % 4;
                    int kOffset = kIdx * qkk0;
                    if (kOffset + kSize > headDim) {
                        kSize = headDim - kOffset;
                        kBlockPad = ROUND_UP(kSize, kBlockSize);
                        kBlockNum = kBlockPad / kBlockSize;
                    }
                    // copy CK (qkcn0, 4 * qkk0) to L1
                    if (kIdx4 == 0) {
                        int kRemSize = 4 * qkk0;
                        if (kOffset + kRemSize > headDim) {
                            kRemSize = headDim - kOffset;
                        }
                        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                        CopyGmToL1Nd2Nz(kl1bBuf[pingpongL1B],
                                        compressKCache[block * qkcn0 * headDim + kOffset], nSize,
                                        kRemSize, headDim, nBlockPad);
                        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                    }

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    CopyToL0BCol(qkl0bBuf[curr], kl1bBuf[pingpongL1B], nBlockNum,
                                 kIdx4 * qkk0 / kBlockSize, kBlockNum);
                    if (kIdx4 == 3) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                        pingpongL1B ^= 1;
                    }
                    CopyToL0ACol(qkl0aBuf[curr], ql1aBuf[kOffset * mBlockPad], mBlockNum, 0,
                                 kBlockNum);

                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                    // mmad scores(queryTokens * nHeads, qkcn0) = Q * CK
                    CalMmad(qkl0cBuf, qkl0aBuf[curr], qkl0bBuf[curr], mBlockPad, nBlockPad,
                            kBlockPad, kIdx == 0);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    PipeBarrier<PIPE_M>();
                    curr = 1 - curr;
                }
                if (kLoop % 4 != 0) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    pingpongL1B ^= 1;
                }

                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
                // copy scores (queryTokens * nHeads, qkcn0) from L0C to GM
                CopyToGm(scores[swaSegWidth + nOffset], qkl0cBuf, mSize, nSize, mBlockPad,
                         qkStride);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

    /*
     * output = scores * K(T)
     *          scores: (queryTokens * nHeads, windowSize + kvLen)
     *          K: (windowSize + kvLen, headDim)
     *          m0: XLITE_MAX_M0, n0: svn0, k0: svk0
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> scores, GlobalTensor<Dtype> swaKCache,
                                    GlobalTensor<Dtype> compressKCache,
                                    __gm__ uint32_t *swaBlockTables,
                                    __gm__ uint32_t *compressBlockTables, uint32_t queryOffset,
                                    uint32_t queryLen, uint32_t kvOffset, uint32_t kvLen,
                                    GlobalTensor<Dtype> output)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mSize = queryLen * nHeads;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = svn0;
        int nBlockPad = svn0;
        int nBlockNum = svn0 / NBLOCKSIZE;
        int nLoop = DIV_ROUND_UP(headDim, svn0);
        int totalLen = kvOffset + kvLen;

        // window [alignStart, windowEnd], matching RunAicQK's SWA segment layout (scores
        // column 0 = abs pos alignStart). With every tile boundary a kBlockSize multiple,
        // the [kOffset, kOffset + kBlockPad) spans this loop walks are disjoint; an
        // unaligned windowStart would make the first tile's kBlockPad overrun the next
        // tile's first token and double-count it.
        int windowStart =
            (int)queryOffset > (int)windowSize - 1 ? (int)queryOffset - (int)windowSize + 1 : 0;
        int windowEnd = queryOffset + queryLen - 1;
        int alignStart = ROUND_DOWN(windowStart, kBlockSize);
        int kwIdxStart = alignStart / svwk0;
        int kwIdxEnd = windowEnd / svwk0;
        int kwBlockOffset = alignStart % swaBlockSize;
        int kwOffset = alignStart % svwk0;
        int kcIdxStart = !dense ? kvOffset / compressBlockSize : 0;
        int kcTotalLen = compressRatio == 0 ? 0 : kvLen / compressRatio;
        int kcLoop = DIV_ROUND_UP(kcTotalLen, svck0);

        if (windowSize == 0 && kcTotalLen == 0) {
            return;
        }

        int curr = 0;
        int pingpongL1A = 0;
        int pingpongL1B = 0;
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // headDim
            int nOffset = nIdx * svn0;
            if (nOffset + nSize > headDim) {
                nSize = headDim - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            bool init = true;

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            if (windowSize != 0) {
                for (int kIdx = kwIdxStart; kIdx < kwIdxEnd + 1; kIdx++) {  // window size
                    uint32_t block = swaBlockTables[kIdx];
                    uint32_t blockOffset = kIdx == kwIdxStart ? kwBlockOffset : 0;
                    int kOffset = kIdx * svwk0;
                    // first tile clipped by alignStart, last tile by totalLen
                    int kSize = svwk0;
                    if (kIdx == kwIdxStart) {
                        kOffset += kwOffset;
                        kSize = svwk0 - kwOffset;
                    }
                    if (kOffset + kSize > totalLen) {
                        kSize = totalLen - kOffset;
                    }
                    int kBlockPad = ROUND_UP(kSize, kBlockSize);
                    int kBlockNum = kBlockPad / kBlockSize;

                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    CopyGmToL1Nd2Nz(scoresl1aBuf[pingpongL1A], scores[kOffset - alignStart], mSize,
                                    kBlockPad, qkStride, mBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);

                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    // copy SWK(T) (svwn0, kSize) to L1
                    CopyGmToL1Nd2Nz(ktl1bBuf[pingpongL1B],
                                    swaKCache[(block * svwk0 + blockOffset) * headDim + nOffset],
                                    kBlockPad, nSize, headDim, kBlockPad);

                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    CopyToL0ACol(svl0aBuf[curr], scoresl1aBuf[pingpongL1A], mBlockNum, 0,
                                 kBlockNum);
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    pingpongL1A ^= 1;
                    CopyToL0BTCol(svl0bBuf[curr], ktl1bBuf[pingpongL1B], nBlockNum, 0, kBlockNum,
                                  kBlockNum);
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    pingpongL1B ^= 1;

                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                    // mmad output (queryTokens * nHeads, qkwn0) = scores * SWK(T)
                    CalMmad(svl0cBuf, svl0aBuf[curr], svl0bBuf[curr], mBlockPad, nBlockPad,
                            kBlockPad, init);
                    init = false;
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    PipeBarrier<PIPE_M>();
                    curr = 1 - curr;
                }
            }

            if (compressRatio != 0) {
                int L1BkRemBlockPad = ROUND_UP(2 * svck0, kBlockSize);
                int L1BkRemBlockNum = L1BkRemBlockPad / kBlockSize;
                for (int kIdx = 0; kIdx < kcLoop; kIdx++) {
                    int kIdx4 = kIdx % 4;
                    int kIdx2 = kIdx % 2;
                    int kOffset = kIdx * svck0;
                    int kSize = svck0;
                    int kBlockPad = svck0;
                    int kBlockNum = svck0 / kBlockSize;
                    if (kOffset + kSize > kcTotalLen) {
                        kSize = kcTotalLen - kOffset;
                        kBlockPad = ROUND_UP(kSize, kBlockSize);
                        kBlockNum = kBlockPad / kBlockSize;
                    }
                    int blockOffset = !dense ? kOffset / compressBlockSize + kcIdxStart : 0;
                    int blockRemainder = !dense ? kOffset % compressBlockSize : 0;

                    if (kIdx4 == 0) {
                        int kRemSize = 4 * svck0;
                        int kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                        if (kOffset + kRemSize > kcTotalLen) {
                            kRemSize = kcTotalLen - kOffset;
                            kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                        }
                        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                        // copy scores (queryTokens * nHeads, 4 * svck0) to L1
                        CopyGmToL1Nd2Nz(scoresl1aBuf[pingpongL1A], scores[swaSegWidth + kOffset],
                                        mSize, kRemBlockPad, qkStride, mBlockPad);
                        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                    }

                    if (kIdx2 == 0) {
                        int kRemSize = 2 * svck0;
                        int kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                        if (kOffset + kRemSize > kcTotalLen) {
                            kRemSize = kcTotalLen - kOffset;
                            kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                        }
                        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                        // copy CK(T) (nSize, kRemSize) to L1
                        if (dense) {
                            // dense cache: contiguous layout, single copy
                            CopyGmToL1Nd2Nz(
                                ktl1bBuf[pingpongL1B],
                                compressKCache[(kvOffset + kOffset) * headDim + nOffset],
                                kRemBlockPad, nSize, headDim, L1BkRemBlockPad);
                        } else {
                            // paged cache: per-block lookup, multi-block copy
                            for (int bid = 0; bid < DIV_ROUND_UP(kRemSize, compressBlockSize);
                                 bid++) {
                                int kOffsetTmp = bid * compressBlockSize;
                                uint32_t block = compressBlockTables[blockOffset + bid];
                                int kRemSizeTmp = compressBlockSize;
                                int kRemBlockPadTmp = ROUND_UP(kRemSizeTmp, kBlockSize);
                                if (kOffsetTmp + kRemSizeTmp > kRemSize) {
                                    kRemSizeTmp = kRemSize - kOffsetTmp;
                                    kRemBlockPadTmp = ROUND_UP(kRemSizeTmp, kBlockSize);
                                }
                                CopyGmToL1Nd2Nz(
                                    ktl1bBuf[pingpongL1B][bid * compressBlockSize * kBlockSize],
                                    compressKCache[(block * compressBlockSize + blockRemainder) *
                                                       headDim +
                                                   nOffset],
                                    kRemBlockPadTmp, nSize, headDim, L1BkRemBlockPad);
                            }
                        }

                        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                    }

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    CopyToL0ACol(svl0aBuf[curr], scoresl1aBuf[pingpongL1A], mBlockNum,
                                 kIdx4 * svck0 / kBlockSize, kBlockNum);
                    if (kIdx4 == 3) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                        pingpongL1A ^= 1;
                    }

                    CopyToL0BTCol(svl0bBuf[curr], ktl1bBuf[pingpongL1B], nBlockNum,
                                  kIdx2 * svck0 / kBlockSize, kBlockNum, L1BkRemBlockNum);
                    if (kIdx2 == 1) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                        pingpongL1B ^= 1;
                    }

                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                    // mmad output (queryTokens * nHeads, qkwn0) = scores * CK(T)
                    CalMmad(svl0cBuf, svl0aBuf[curr], svl0bBuf[curr], mBlockPad, nBlockPad,
                            kBlockPad, init);
                    init = false;
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    PipeBarrier<PIPE_M>();
                    curr = 1 - curr;
                }
                if (kcLoop % 4 != 0) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    pingpongL1A ^= 1;
                }
                if (kcLoop % 2 != 0) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    pingpongL1B ^= 1;
                }
            }
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy output (queryTokens * nHeads, nSize) from L0C to GM
            CopyToGm(output[nOffset], svl0cBuf, mSize, nSize, mBlockPad, headDim);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t swaBlockSize;
    uint32_t compressBlockSize;
    uint32_t windowSize;
    uint32_t compressRatio;
    uint32_t qkStride;
    uint32_t swaSegWidth;
    bool dense;

    int qkwn0;
    int qkcn0;
    int qkk0;
    int svn0;
    int svwk0;
    int svck0;

    LocalTensor<Dtype> ql1aBuf;                     // event 0
    LocalTensor<Dtype> kl1bBuf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<Dtype> qkl0aBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> qkl0bBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<float> qkl0cBuf;                    // event 0

    LocalTensor<Dtype> scoresl1aBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> ktl1bBuf[PINGPONG_BUF_NUM];      // event 2/3
    LocalTensor<Dtype> svl0aBuf[PINGPONG_BUF_NUM];      // event 0/1
    LocalTensor<Dtype> svl0bBuf[PINGPONG_BUF_NUM];      // event 0/1
    LocalTensor<float> svl0cBuf;                        // event 0
};