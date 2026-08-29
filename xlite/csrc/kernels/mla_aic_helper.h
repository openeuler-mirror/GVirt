/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"

#ifndef MAX_N0
#define MAX_N0 128
#endif

// Cube-side QK/SV matmul helper shared by mla_v2 / flash_mla_v2.
//
// MLA path: Q is split into qAbsorb (kvLoraRank) + qr (ropeHeadDim),
//           K is split into kCache (kvLoraRank) + peCache (ropeHeadDim).
//   RunAicQK: C = qAbsorb * kCache, R = qr * peCache, QK = C + R.
//   RunAicSV: oAbsorb = QK * kCache(T).
//
// This helper owns the 12 L1/L0 pingpong buffers and their address layout
// (bound in Init()), plus scalar geometry. GM KV tensors (kCache/peCache) are
// already bound by the kernel's own Init() and are passed into RunAicQK/RunAicSV
// as parameters. Instantiated as a member (composition, like RingSync).
template <typename Dtype>
class MlaAicHelper
{
public:
    __aicore__ inline MlaAicHelper() = default;

    // Bind L1/L0 buffers and scalar geometry; return svk0 for the kernel's RunAiv.
    // dense: dense (contiguous) KV cache and blockTable is unused (block id == logical index).
    __aicore__ inline int Init(uint32_t nHeadsV, uint32_t ropeHeadDimV, uint32_t kvLoraRankV,
                               uint32_t blockSizeV, uint32_t qkStrideV, bool denseV)
    {
        nHeads = nHeadsV;
        ropeHeadDim = ropeHeadDimV;
        kvLoraRank = kvLoraRankV;
        dense = denseV;
        qkStride = qkStrideV;
        blockSize = blockSizeV;
        qkn0 = dense ? MAX_N0 : blockSizeV;
        qkk0 = 256 / sizeof(Dtype);
        svn0 = 256;
        svk0 = 64;
        uint64_t off = 0;

        // QK
        off = 0;
        uint64_t qnSize = XLITE_MAX_M0 * kvLoraRank * sizeof(Dtype);
        aqnl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        aqnl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qnSize;

        uint64_t qrSize = XLITE_MAX_M0 * ropeHeadDim * sizeof(Dtype);
        aqrl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        aqrl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qrSize;

        uint64_t kSize = qkn0 * 4 * qkk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kSize;
        }

        uint64_t krSize = qkn0 * ropeHeadDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akrl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akrl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += krSize;
        }
        uint64_t total_qk = off;
        dbg_printf("QK buf: qnSize %lu, qrSize %lu, kSize %lu x 2, krSize %lu x 2, total %lu\n",
                   qnSize, qrSize, kSize, krSize, total_qk);

        // l0
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
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl0cBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
            qkl0cBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0cSize;
        }

        // SV
        off = 0;
        uint64_t qkSize = XLITE_MAX_M0 * 4 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            aqkl1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            aqkl1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qkSize;
        }

        uint64_t ktSize = svn0 * 2 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            aktl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            aktl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ktSize;
        }
        uint64_t total_sv = off;
        dbg_printf("SV buf: ktSize %lu x 2, qkSize %lu x 2, total %lu\n", ktSize, qkSize, total_sv);

        // l0
        off = 0;
        l0aSize = XLITE_MAX_M0 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            svl0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            svl0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0aSize;
        }

        off = 0;
        l0bSize = svn0 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            svl0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            svl0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0bSize;
        }

        off = 0;
        l0cSize = XLITE_MAX_M0 * svn0 * sizeof(float);
        svl0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        svl0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);

        return svk0;
    }

    /*
     * C = Absorb * K
     *     Absorb: (queryTokens * nHeads, kvLoraRank)
     *     K: (cachedTokens, kvLoraRank)
     *     C: (queryTokens * nHeads, cachedTokens)
     *     m0: XLITE_MAX_M0, n0: qkn0, k0: qkk0
     * R = QR * KR
     *     QR: (queryTokens * nHeads, ropeHeadDim)
     *     K: (cachedTokens, ropeHeadDim)
     *     R: (queryTokens * nHeads, cachedTokens)
     *     m0: XLITE_MAX_M0, n0: qkn0, k0: qkk0
     * QK = C + R
     *
     * kvOffset/kvLen express a KV tile; mla_v2 passes kvOffset=0, kvLen=calcLen
     * for the full-range case, which is exactly the legacy mla_v2 behavior.
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> qAbsorb, GlobalTensor<Dtype> qr,
                                    GlobalTensor<Dtype> kCache, GlobalTensor<Dtype> peCache,
                                    uint32_t queryTaskLen, __gm__ uint32_t *blockTable,
                                    uint32_t kvOffset, uint32_t kvLen, GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mSize = queryTaskLen * nHeads;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nIdxStart = kvOffset / qkn0;
        int nSize = qkn0;
        int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int nLoop = DIV_ROUND_UP(kvLen, qkn0);
        int kSize = qkk0;
        int kBlockPad = qkk0;
        int kBlockNum = qkk0 / kBlockSize;
        int kLoop = DIV_ROUND_UP(kvLoraRank, qkk0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // copy Absorb (queryTokens * nHeads, kvLoraRank) to L1
        CopyGmToL1Nd2Nz(aqnl1aBuf, qAbsorb, mSize, kvLoraRank, kvLoraRank, mBlockPad);
        // copy QR (queryTokens * nHeads, ropeHeadDim) to L1
        CopyGmToL1Nd2Nz(aqrl1aBuf, qr, mSize, ropeHeadDim, ropeHeadDim, mBlockPad);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        int curr = 0;
        int pingpongL1B = 0;
        int pingpongL0C = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // kvLen
            uint32_t block = !dense ? blockTable[nIdx + nIdxStart] : nIdx + nIdxStart;
            int nOffset = nIdx * qkn0;
            if (nOffset + nSize > kvLen) {
                nSize = kvLen - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0 + pingpongL0C);

            kSize = qkk0;
            kBlockPad = qkk0;
            kBlockNum = qkk0 / kBlockSize;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLoraRank
                int kIdx4 = kIdx % 4;
                int kOffset = kIdx * qkk0;
                if (kOffset + kSize > kvLoraRank) {
                    kSize = kvLoraRank - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }
                // copy K (qkn0, 4 * qkk0) to L1
                if (kIdx4 == 0) {
                    int kRemSize = 4 * qkk0;
                    if (kOffset + kRemSize > kvLoraRank) {
                        kRemSize = kvLoraRank - kOffset;
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    CopyGmToL1Nd2Nz(akl1bBuf[pingpongL1B],
                                    kCache[block * qkn0 * kvLoraRank + kOffset], nSize, kRemSize,
                                    kvLoraRank, nBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                }

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0BCol(qkl0bBuf[curr], akl1bBuf[pingpongL1B], nBlockNum,
                             kIdx4 * qkk0 / kBlockSize, kBlockNum);
                if (kIdx4 == 3) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    pingpongL1B ^= 1;
                }
                CopyToL0ACol(qkl0aBuf[curr], aqnl1aBuf[kOffset * mBlockPad], mBlockNum, 0,
                             kBlockNum);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad C (queryTokens * nHeads, qkn0) = Absorb * K
                CalMmad(qkl0cBuf[pingpongL0C], qkl0aBuf[curr], qkl0bBuf[curr], mBlockPad, nBlockPad,
                        kBlockPad, kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            if (kLoop % 4 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                pingpongL1B ^= 1;
            }

            kSize = ropeHeadDim;
            kBlockPad = ROUND_UP(kSize, kBlockSize);
            kBlockNum = kBlockPad / kBlockSize;
            // copy KR (qkn0, ropeHeadDim) to L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
            CopyGmToL1Nd2Nz(akrl1bBuf[curr], peCache[block * qkn0 * ropeHeadDim], nSize,
                            ropeHeadDim, ropeHeadDim, nBlockPad);

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            CopyToL0ACol(qkl0aBuf[curr], aqrl1aBuf, mBlockNum, 0, kBlockNum);
            CopyToL0BCol(qkl0bBuf[curr], akrl1bBuf[curr], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

            // mmad R (queryTokens * nHeads, qkn0) = QR * KR
            CalMmad(qkl0cBuf[pingpongL0C], qkl0aBuf[curr], qkl0bBuf[curr], mBlockPad, nBlockPad,
                    kBlockPad, false);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_FIX>(EVENT_ID0 + pingpongL0C);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0 + pingpongL0C);

            // copy final QK(queryTokens * nHeads, nSize) from L0C to GM
            CopyToGm(qk[nIdx * qkn0], qkl0cBuf[pingpongL0C], mSize, nSize, mBlockPad, qkStride);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0 + pingpongL0C);
            curr = 1 - curr;
            pingpongL0C ^= 1;
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

    /*
     * Absorb = QK * K(T)
     *     QK: (queryTokens * nHeads, cachedTokens)
     *     K: (cachedTokens, kvLoraRank)
     *     Absorb: (queryTokens * nHeads, kvLoraRank)
     *     m0: XLITE_MAX_M0, n0: svn0, k0: svk0
     *
     * kvOffset/kvLen express a KV tile; mla_v2 passes kvOffset=0, kvLen=calcLen.
     * `out` is the output GM tensor (already offset by the caller).
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, GlobalTensor<Dtype> kCache,
                                    uint32_t queryTaskLen, __gm__ uint32_t *blockTable,
                                    uint32_t kvOffset, uint32_t kvLen, GlobalTensor<Dtype> out)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mSize = queryTaskLen * nHeads;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = svn0;
        int nBlockPad = svn0;
        int nBlockNum = svn0 / NBLOCKSIZE;
        int nLoop = DIV_ROUND_UP(kvLoraRank, svn0);
        int kIdxStart = !dense ? kvOffset / blockSize : 0;
        int kSize = svk0;
        int kBlockPad = svk0;
        int kBlockNum = svk0 / kBlockSize;
        int kLoop = DIV_ROUND_UP(kvLen, svk0);

        int curr = 0;
        int pingpongL1A = 0;
        int pingpongL1B = 0;
        int L1BkRemBlockPad = ROUND_UP(2 * svk0, kBlockSize);
        int L1BkRemBlockNum = L1BkRemBlockPad / kBlockSize;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // kvLoraRank
            int nOffset = nIdx * svn0;
            if (nOffset + nSize > kvLoraRank) {
                nSize = kvLoraRank - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            kSize = svk0;
            kBlockPad = svk0;
            kBlockNum = svk0 / kBlockSize;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLen
                int kIdx4 = kIdx % 4;
                int kIdx2 = kIdx % 2;
                int kOffset = kIdx * svk0;
                if (kOffset + kSize > kvLen) {
                    kSize = kvLen - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }
                int blockOffset = !dense ? kOffset / blockSize + kIdxStart : 0;
                int blockRemainder = !dense ? kOffset % blockSize : 0;

                if (kIdx4 == 0) {
                    int kRemSize = 4 * svk0;
                    int kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    if (kOffset + kRemSize > kvLen) {
                        kRemSize = kvLen - kOffset;
                        kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    // copy QK (queryTokens * nHeads, 4 * svk0) to L1
                    CopyGmToL1Nd2Nz(aqkl1aBuf[pingpongL1A], qk[kOffset], mSize, kRemBlockPad,
                                    qkStride, mBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                }

                if (kIdx2 == 0) {
                    int kRemSize = 2 * svk0;
                    int kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    if (kOffset + kRemSize > kvLen) {
                        kRemSize = kvLen - kOffset;
                        kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    // copy K(T) (nSize, kRemSize) to L1
                    if (dense) {
                        // dense cache: contiguous layout, single copy
                        CopyGmToL1Nd2Nz(aktl1bBuf[pingpongL1B],
                                        kCache[(kvOffset + kOffset) * kvLoraRank + nOffset],
                                        kRemBlockPad, nSize, kvLoraRank, L1BkRemBlockPad);
                    } else {
                        // paged cache: per-block lookup, multi-block copy
                        for (int bid = 0; bid < DIV_ROUND_UP(kRemSize, blockSize); bid++) {
                            int kOffsetTmp = bid * blockSize;
                            uint32_t block = blockTable[blockOffset + bid];
                            int kRemSizeTmp = blockSize;
                            int kRemBlockPadTmp = ROUND_UP(kRemSizeTmp, kBlockSize);
                            if (kOffsetTmp + kRemSizeTmp > kRemSize) {
                                kRemSizeTmp = kRemSize - kOffsetTmp;
                                kRemBlockPadTmp = ROUND_UP(kRemSizeTmp, kBlockSize);
                            }
                            CopyGmToL1Nd2Nz(
                                aktl1bBuf[pingpongL1B][bid * blockSize * kBlockSize],
                                kCache[(block * blockSize + blockRemainder) * kvLoraRank + nOffset],
                                kRemBlockPadTmp, nSize, kvLoraRank, L1BkRemBlockPad);
                        }
                    }

                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                }

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(svl0aBuf[curr], aqkl1aBuf[pingpongL1A], mBlockNum,
                             kIdx4 * svk0 / kBlockSize, kBlockNum);
                if (kIdx4 == 3) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    pingpongL1A ^= 1;
                }

                CopyToL0BTCol(svl0bBuf[curr], aktl1bBuf[pingpongL1B], nBlockNum,
                              kIdx2 * svk0 / kBlockSize, kBlockNum, L1BkRemBlockNum);
                if (kIdx2 == 1) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    pingpongL1B ^= 1;
                }

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad Absorb (queryTokens, nSize) = QK * K（T）
                CalMmad(svl0cBuf, svl0aBuf[curr], svl0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            if (kLoop % 4 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                pingpongL1A ^= 1;
            }
            if (kLoop % 2 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                pingpongL1B ^= 1;
            }
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy Absorb (queryTokens * nHeads, nSize) to GM
            CopyToGm(out[nOffset], svl0cBuf, mSize, nSize, mBlockPad, kvLoraRank);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }

private:
    LocalTensor<Dtype> aqnl1aBuf;                    // event 0
    LocalTensor<Dtype> aqrl1aBuf;                    // event 0
    LocalTensor<Dtype> akl1bBuf[PINGPONG_BUF_NUM];   // event 4/5
    LocalTensor<Dtype> akrl1bBuf[PINGPONG_BUF_NUM];  // event 2/3
    LocalTensor<Dtype> aqkl1aBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> aktl1bBuf[PINGPONG_BUF_NUM];  // event 4/5
    LocalTensor<Dtype> qkl0aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> qkl0bBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<float> qkl0cBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> svl0aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> svl0bBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<float> svl0cBuf;                     // event 0

    uint32_t nHeads;
    uint32_t ropeHeadDim;
    uint32_t kvLoraRank;
    uint32_t blockSize;
    uint32_t qkStride;
    bool dense;
    int qkn0;
    int qkk0;
    int svn0;
    int svk0;
};
