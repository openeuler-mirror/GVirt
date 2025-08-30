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

#define PINGPONG_BUF_NUM 2

template<typename Dtype>
class Matmul {
public:
    __aicore__ inline Matmul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint64_t m, uint64_t n, uint64_t k, uint64_t nz,
                                uint64_t m0, uint64_t n0, uint64_t k0)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

        SetHF32Mode(false);
        SetMaskNorm();
        SetAtomicNone();

        this->aGmBuf.SetGlobalBuffer((__gm__ Dtype *)x);
        this->bGmBuf.SetGlobalBuffer((__gm__ Dtype *)y);
        this->cGmBuf.SetGlobalBuffer((__gm__ Dtype *)z);
        this->m = m;
        this->n = n;
        this->k = k;
        this->nz = nz;

        if (m0 == (uint64_t)-1) {
            m0 = 128;
            n0 = n < 2048 ? 128 : 256;
            k0 = 512 / sizeof(Dtype);
            if (n < 1024) {
                m0 = 64;
                n0 = 64;
            }
        }

        if (nz) {
            /* same n0 value with tests/models/weight_utils.py 's matrix_nd2nz func */
            m0 = 128;
            n0 = 256;
            k0 = 512 / sizeof(Dtype);
        }

        this->m0 = m0;
        this->n0 = n0;
        this->k0 = k0;

        this->mBlockSize = 16;
        this->nBlockSize = 16;
        if constexpr (std::is_same<Dtype, float>::value) {
            this->kBlockSize = 8;
        } else if constexpr (std::is_same<Dtype, float16_t>::value) {
            this->kBlockSize = 16;
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            this->kBlockSize = 16;
        }
        this->cubeBlockSize = mBlockSize * kBlockSize;

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

    __aicore__ inline void CopyGmToL1Nd2Nz(LocalTensor<Dtype> dst, GlobalTensor<Dtype> src, int nValue, int dValue, int srcDValue, int dstNzC0Stride)
    {
        Nd2NzParams nd2nzParams(1 /* NdNum */, nValue, dValue, 0 /* srcNdMatrixStride */, srcDValue, dstNzC0Stride, 1 /* dstNzNStride */, 0 /* dstNzMatrixStride */);
        DataCopy(dst, src, nd2nzParams);
    }

    __aicore__ inline void CopyGmToL1(LocalTensor<Dtype> dst, GlobalTensor<Dtype> src, int nElem)
    {
        int burstLen = DIV_ROUND_UP(nElem * sizeof(Dtype), 32);
        DataCopyParams repeatParams(1, burstLen, 0, 0);
        DataCopy(dst, src, repeatParams);
    }

    __aicore__ inline void CopyToL0ACol(LocalTensor<Dtype> dst, LocalTensor<Dtype> src, int mBlockNum, int kBlockStart, int kBlockNum)
    {
        LoadData2dParams params(0 /* startIndex */, mBlockNum /* repeatTimes */, 1 /* srcStride */, 0 /* sid */, kBlockNum - 1 /* dstGap */, 0, inc);
        for (int k = kBlockStart; k < kBlockStart + kBlockNum; k++) {
            LoadData(dst[(k - kBlockStart) * cubeBlockSize], src[k * mBlockNum * cubeBlockSize], params);
        }
    }

    __aicore__ inline void CopyToL0BTCol(LocalTensor<Dtype> dst, LocalTensor<Dtype> src, int nBlockNum, int kBlockStart, int kBlockNum)
    {
        LoadData2dParams params(0, kBlockNum * nBlockNum, 1, 0, 0, 0, inc);
        LoadData(dst, src[kBlockStart * nBlockNum * cubeBlockSize], params);
    }

    __aicore__ inline void CalMmad(LocalTensor<float> c, LocalTensor<Dtype> a, LocalTensor<Dtype> b, int m, int n, int k, bool init)
    {
        MmadParams params;
        params.m = m;
        params.n = n;
        params.k = k;
        params.cmatrixInitVal = init;
        Mmad(c, a, b, params);
    }

    __aicore__ inline void CopyToGm(GlobalTensor<Dtype> dst, LocalTensor<float> src, int mSize, int nSize, int srcStride, int dstStride)
    {
        QuantMode_t mode;
        if constexpr (std::is_same<Dtype, float>::value) {
            mode = NoQuant;
        } else if constexpr (std::is_same<Dtype, float16_t>::value) {
            mode = F322F16;
        } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            mode = F322BF16;
        }
        DataCopyCO12DstParams param(nSize, mSize, dstStride, srcStride, mode, 0, 0, 1);
        SetFixpipeNz2ndFlag(1, 1, 1);
        DataCopy(dst, src, param);
    }

    __aicore__ inline void Run()
    {
        int kQtileBlockNum = kQtileSize / kBlockSize;
        int kLoop = DIV_ROUND_UP(k, kQtileSize);

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

        for (int32_t loopIdx = 0; loopIdx < coreLoop; loopIdx++) {
            if (loopIdx % GetBlockNum() != GetBlockIdx()) {
                continue;
            }
            int64_t midx = loopIdx / nLoop;
            int64_t nidx = loopIdx % nLoop;
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
                    CopyGmToL1Nd2Nz(l1aBuf[pingpongL1A], aGmBuf[mOffset * k + kOffset], mActual, kRemSize, k, mActualBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                }

                /* B GM -> L1 */
                if (kIdx4 == 0) {
                    int kRemSize = k0;
                    if (kOffset + kRemSize > k) {
                        kRemSize = k - kOffset;
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    if (nz == 0) {
                        CopyGmToL1Nd2Nz(l1bBuf[pingpongL1B], bGmBuf[nOffset * k + kOffset], nActual, kRemSize, k, nActualBlockPad);
                    } else {
                        CopyGmToL1(l1bBuf[pingpongL1B], bGmBuf[nOffset * k + nActualBlockPad * kOffset], nActual * kRemSize);
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
                CopyToL0ACol(l0aBuf[kIdx2], l1aBuf[pingpongL1A], mActualBlockNum, kIdx8 * kQtileBlockNum, kActualBlockNum);
                if (kIdx8 == 7) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    pingpongL1A ^= 1;
                }

                /* B L1 -> L0B */
                if (kIdx4 == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                }
                CopyToL0BTCol(l0bBuf[kIdx2], l1bBuf[pingpongL1B], nActualBlockNum, kIdx4 * kQtileBlockNum, kActualBlockNum);
                if (kIdx4 == 3) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    pingpongL1B ^= 1;
                }

                /* Mmad L0A L0B -> L0C */
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                CalMmad(l0cBuf, l0aBuf[kIdx2], l0bBuf[kIdx2], m0, nActualBlockPad, kActualBlockPad, kIdx == 0);
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
        } // M * N

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
    uint64_t m0;
    uint64_t n0;
    uint64_t k0;
    uint64_t mBlockSize;
    uint64_t nBlockSize;
    uint64_t kBlockSize;
    uint64_t cubeBlockSize;
    int kDtileSize;
    int kQtileSize;
};

#define MATMUL_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void matmul_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z, \
                                                     uint64_t m, uint64_t n, uint64_t k, uint64_t nz, \
                                                     uint64_t m0 = -1, uint64_t n0 = -1, uint64_t k0 = -1) \
{ \
    Matmul<dtype> op; \
    op.Init(x, y, z, m, n, k, nz, m0, n0, k0); \
    op.Run(); \
}

#endif