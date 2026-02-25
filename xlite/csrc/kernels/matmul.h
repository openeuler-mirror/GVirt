/*
 * @file matmul.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef _XLITE_MATMUL_H_
#define _XLITE_MATMUL_H_

#include "kernel_operator.h"
#include "kernel_macro.h"
using namespace AscendC;

template <typename Dtype>
class Matmul
{
public:
    __aicore__ inline Matmul()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t m, uint64_t n, uint64_t k,
                                uint64_t nz, uint64_t transpose, uint64_t m0, uint64_t n0,
                                uint64_t k0, uint64_t swizzl, uint32_t curBlock = 0,
                                uint32_t curCount = 0, uint32_t remain = 0)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

        SetHF32Mode(false);
        SetMaskNorm();
        SetAtomicNone();

        this->swizzlCount = (swizzl >> 8);
        this->swizzleDirection = (swizzl & 0xFF);
        this->aGmBuf.SetGlobalBuffer((__gm__ Dtype *)x);
        this->bGmBuf.SetGlobalBuffer((__gm__ Dtype *)y);
        this->cGmBuf.SetGlobalBuffer((__gm__ Dtype *)z);
        this->m = m;
        this->n = n;
        this->k = k;
        this->nz = nz;
        this->transpose = transpose;

        if (m0 == (uint64_t)-1) {
            m0 = ROUND_UP(m, 32);
            if (m0 > 128) {
                m0 = 128;
            }
            n0 = 256;
            k0 = 512 / sizeof(Dtype);
        }

        this->m0 = m0;
        this->n0 = n0;
        this->k0 = k0;
        this->curBlock = curBlock;
        this->curCount = curCount;
        this->remain = remain;

        this->mBlockSize = 16;
        this->nBlockSize = 16;
        if constexpr (std::is_same<Dtype, float>::value) {
            this->kBlockSize = 8;
        } else if constexpr (std::is_same<Dtype, float16_t>::value) {
            this->kBlockSize = 16;
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            this->kBlockSize = 16;
        }

        kDtileSize = k0 << 1;
        kQtileSize = k0 >> 2;

        int l1ATileBytes = m0 * kDtileSize * sizeof(Dtype);
        int l1BTileBytes = n0 * k0 * sizeof(Dtype);
        int l0ATileBytes = m0 * kQtileSize * sizeof(Dtype);
        int l0BTileBytes = n0 * kQtileSize * sizeof(Dtype);
        int l0CTileBytes = m0 * n0 * sizeof(Dtype);
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
            off += l0ATileBytes;
        }
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0BTileBytes;
        }
        off = 0;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void GetMNBlockIdx(int32_t loopIdx, int32_t mLoop, int32_t nLoop,
                                         int32_t swizzlDirection, int32_t swizzlCount,
                                         int64_t &mIdx, int64_t &nIdx)
    {
        // Adjust the traversal order of matmul block by setting swizzlCount(upper 8 bits) and
        // swizzleDirection(lower 8 bits). Adjusting the swizzle strategy helps improve cache hit
        // rate and reduce data read overhead, thereby enhancing the overall computational
        // efficiency of matmul.
        uint32_t tileBlockLoop, tileBlockIdx, inTileBlockIdx;
        uint32_t inBatchIdx = loopIdx % (mLoop * nLoop);
        if (swizzlDirection == 0) {  // Zn
            tileBlockLoop = (mLoop + swizzlCount - 1) / swizzlCount;
            tileBlockIdx = inBatchIdx / (swizzlCount * nLoop);
            inTileBlockIdx = inBatchIdx % (swizzlCount * nLoop);

            uint32_t nRow = swizzlCount;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = mLoop - swizzlCount * tileBlockIdx;
            }
            mIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % nRow;
            nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 != 0) {
                nIdx = nLoop - nIdx - 1;
            }
        } else if (swizzlDirection == 1) {  // Nz
            tileBlockLoop = (nLoop + swizzlCount - 1) / swizzlCount;
            tileBlockIdx = inBatchIdx / (swizzlCount * mLoop);
            inTileBlockIdx = inBatchIdx % (swizzlCount * mLoop);

            uint32_t nCol = swizzlCount;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = nLoop - swizzlCount * tileBlockIdx;
            }
            mIdx = inTileBlockIdx / nCol;
            nIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 != 0) {
                mIdx = mLoop - mIdx - 1;
            }
        }
    }

    __aicore__ inline void Run()
    {
        int kQtileBlockNum = kQtileSize / kBlockSize;
        int kLoop = DIV_ROUND_UP(k, kQtileSize);
        int nStride = ROUND_UP(n, nBlockSize);
        int kStride = ROUND_UP(k, kBlockSize);

        int pingpongL1A = 0;
        int pingpongL1B = 0;

        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);

        int nLoop = DIV_ROUND_UP(n, n0);
        int mLoop = DIV_ROUND_UP(m, m0);
        int coreLoop = nLoop * mLoop;
        curCount = (curCount == 0) ? coreLoop : curCount;

        for (int32_t loopIdx = curBlock; loopIdx < curCount; loopIdx++) {
            if (loopIdx % GetBlockNum() != GetBlockIdx()) {
                continue;
            }
            int64_t midx = (loopIdx - remain) / nLoop;
            int64_t nidx = (loopIdx - remain) % nLoop;
            GetMNBlockIdx(loopIdx, mLoop, nLoop, swizzleDirection, swizzlCount, midx, nidx);
            int nOffset = nidx * n0;
            int mOffset = midx * m0;

            int mActual = m0;
            if (mOffset + m0 > m) {
                mActual = m - mOffset;
            }
            int mActualBlockPad = ROUND_UP(mActual, mBlockSize);
            int mActualBlockNum = DIV_ROUND_UP(mActual, mBlockSize);

            int nActual = n0;
            if (nOffset + n0 > n) {
                nActual = n - nOffset;
            }
            int nActualBlockPad = ROUND_UP(nActual, nBlockSize);
            int nActualBlockNum = DIV_ROUND_UP(nActual, nBlockSize);

            GlobalTensor<Dtype> outGm = cGmBuf[mOffset * n + nOffset];

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            int kOffset = 0;
            int kIdx = 0;
            for (; kIdx < kLoop; kIdx++) {
                int kIdx8 = kIdx % 8;
                int kIdx4 = kIdx % 4;
                int kIdx2 = kIdx % 2;

                /* A GM -> L1 */
                if (kIdx8 == 0) {
                    int kRemSize = kDtileSize;
                    if (kOffset + kRemSize > k) {
                        kRemSize = k - kOffset;
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    CopyGmToL1Nd2Nz(l1aBuf[pingpongL1A], aGmBuf[mOffset * k + kOffset], mActual,
                                    kRemSize, k, mActualBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                }

                /* B GM -> L1 */
                int k0ActualBlockNum;
                if (kIdx4 == 0) {
                    int kRemSize = k0;
                    if (kOffset + kRemSize > k) {
                        kRemSize = k - kOffset;
                    }
                    k0ActualBlockNum = DIV_ROUND_UP(kRemSize, kBlockSize);
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    if (transpose == 0 && nz == 0) {
                        CopyGmToL1Nd2Nz(l1bBuf[pingpongL1B], bGmBuf[nOffset * k + kOffset], nActual,
                                        kRemSize, k, nActualBlockPad);
                    } else if (transpose == 0 && nz == 1) {
                        CopyGmToL1(l1bBuf[pingpongL1B],
                                   bGmBuf[kOffset * nStride + nOffset * kBlockSize], nActual,
                                   k0ActualBlockNum, nStride);
                    } else if (transpose == 1 && nz == 0) {
                        CopyGmToL1Nd2Nz(l1bBuf[pingpongL1B], bGmBuf[kOffset * n + nOffset],
                                        kRemSize, nActual, n, ROUND_UP(kRemSize, kBlockSize));
                    } else if (transpose == 1 && nz == 1) {
                        CopyGmToL1(l1bBuf[pingpongL1B],
                                   bGmBuf[nOffset * kStride + kOffset * nBlockSize], kRemSize,
                                   DIV_ROUND_UP(nActual, nBlockSize), kStride);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                }

                int kActual = kQtileSize;
                if (kOffset + kActual > k) {
                    kActual = k - kOffset;
                }
                int kActualBlockPad = ROUND_UP(kActual, kBlockSize);
                int kActualBlockNum = DIV_ROUND_UP(kActual, kBlockSize);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + kIdx2);

                /* A L1 -> L0A */
                if (kIdx8 == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                }
                CopyToL0ACol(l0aBuf[kIdx2], l1aBuf[pingpongL1A], mActualBlockNum,
                             kIdx8 * kQtileBlockNum, kActualBlockNum);
                if (kIdx8 == 7) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    pingpongL1A ^= 1;
                }

                /* B L1 -> L0B */
                if (kIdx4 == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                }
                if (transpose) {
                    CopyToL0BTCol(l0bBuf[kIdx2], l1bBuf[pingpongL1B], nActualBlockNum,
                                  kIdx4 * kQtileBlockNum, kActualBlockNum, k0ActualBlockNum);
                } else {
                    CopyToL0BCol(l0bBuf[kIdx2], l1bBuf[pingpongL1B], nActualBlockNum,
                                 kIdx4 * kQtileBlockNum, kActualBlockNum);
                }
                if (kIdx4 == 3) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    pingpongL1B ^= 1;
                }

                /* Mmad L0A L0B -> L0C */
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                PipeBarrier<PIPE_M>();
                CalMmad(l0cBuf, l0aBuf[kIdx2], l0bBuf[kIdx2], m0, nActualBlockPad, kActualBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + kIdx2);
                kOffset += kActual;
            }

            if (kIdx % 8 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
            }
            if (kIdx % 4 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
            }

            /* C L0C -> GM */
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            CopyToGm(outGm, l0cBuf, mActual, nActual, m0, n);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }  // M * N

        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

private:
    GlobalTensor<Dtype> aGmBuf;
    GlobalTensor<Dtype> bGmBuf;
    GlobalTensor<Dtype> cGmBuf;
    LocalTensor<Dtype> l1aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;

    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t nz;
    uint64_t transpose;
    uint64_t m0;
    uint64_t n0;
    uint64_t k0;
    uint64_t mBlockSize;
    uint64_t nBlockSize;
    uint64_t kBlockSize;
    int kDtileSize;
    int kQtileSize;
    int32_t swizzlCount;
    int32_t swizzleDirection;
    uint32_t curBlock;
    uint32_t curCount;
    uint32_t remain;
};

#define MATMUL_FUNC_DEFINE(dtype)                                                         \
    extern "C" __global__ __aicore__ void matmul_##dtype(                                 \
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t m, uint64_t n, uint64_t k, uint64_t nz, \
        uint64_t transpose, uint64_t m0, uint64_t n0, uint64_t k0, uint64_t swizzl)       \
    {                                                                                     \
        Matmul<dtype> op;                                                                 \
        op.Init(x, y, z, m, n, k, nz, transpose, m0, n0, k0, swizzl);                     \
        op.Run();                                                                         \
    }

#endif
