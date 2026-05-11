/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"
// #define XLITE_KERNEL_DEBUG
#include "debug.h"
#include "softmax_attn_aiv.h"

#define MAX_M0 128
#define MAX_N0 128
#define MAX_K0 128
#define MBLOCKSIZE 16
#define NBLOCKSIZE 16
#define SEQLEN_64 64
#define SEQLEN_12K 12288
#define SEQLEN_20K 20480
#define SEQLEN_24K 24576
#define SEQLEN_30K 30720
#define SEQLEN_48K 49152
#define SEQLEN_60K 61440
#define SEQLEN_96K 98304

template <typename Dtype>
class MLA
{
public:
    __aicore__ inline MLA()
    {
    }

    __aicore__ inline void Init(GM_ADDR qWithQr, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR wkvb,
                                GM_ADDR topkIndices, GM_ADDR qk, GM_ADDR output,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                                uint32_t nopeHeadDim, uint32_t vHeadDim, uint32_t kvLoraRank,
                                uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock,
                                float scale, uint32_t topK)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->qWithQr.SetGlobalBuffer((__gm__ Dtype *)qWithQr);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->vCache.SetGlobalBuffer((__gm__ Dtype *)vCache);
        this->wkvb.SetGlobalBuffer((__gm__ Dtype *)wkvb);
        this->topkIndices = (__gm__ int32_t *)topkIndices;
        this->output.SetGlobalBuffer((__gm__ Dtype *)output);

        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;

        this->nHeads = nHeads;
        this->ropeHeadDim = ropeHeadDim;
        this->nopeHeadDim = nopeHeadDim;
        this->vHeadDim = vHeadDim;
        this->kvLoraRank = kvLoraRank;
        this->blockSize = blockSize;
        this->batch = batch;
        this->maxNumBlock = maxNumBlock;
        this->maxSeqLen = maxNumBlock * blockSize;
        this->scale = scale;
        this->topK = (topkIndices == nullptr) ? 0 : topK;
        this->qkStride = this->topK > 0 ? this->topK : this->maxSeqLen;

        this->qk[0].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * MAX_M0 * qkStride);
        this->qk[1].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * MAX_M0 * qkStride +
                                    block_num * MAX_M0 * qkStride);

        k0 = 256 / sizeof(Dtype);
        uint64_t off = 0;

        off = 0;
        uint64_t l0aSize = MAX_M0 * k0 * sizeof(Dtype);
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
        uint64_t l0cSize = MAX_M0 * MAX_N0 * sizeof(float);
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += l0cSize;
        qksvl0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        qksvl0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);

        // 分配L1
        InitBuf();
        InitAbsorbBuf();
    }

    __aicore__ inline void InitBuf()
    {
        /*
         * KC = K * WUK:
         *     m: cachedTokens, n: nopeHeadDim, k: kvLoraRank
         *     m0: blockSize, n0: MAX_N0, k0: MAX_K0
         * QR * KR:
         *     m: queryTokens, n: cachedTokens, k: ropeHeadDim
         *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
         * QC * KC:
         *     m: queryTokens, n: cachedTokens, k: nopeHeadDim
         *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
         *
         * V = K * WUV:
         *     m: cachedTokens, n: vHeadDim, k: kvLoraRank
         *     m0: blockSize, n0: MAX_N0, k0: MAX_K0
         * QK * V:
         *     m: queryTokens, n: vHeadDim, k: cachedTokens
         *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
         */

        uint64_t off = 0;
        uint64_t kl1aSize = blockSize * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kl1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kl1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kl1aSize;
        }
        uint64_t sharel1Size = off;

        // QK
        uint64_t wukSize = nopeHeadDim * kvLoraRank * sizeof(Dtype);
        wukl1bBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        wukl1bBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += wukSize;

        uint64_t qcSize = MAX_M0 * nopeHeadDim * sizeof(Dtype);
        qcl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        qcl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qcSize;

        uint64_t qrSize = MAX_M0 * ropeHeadDim * sizeof(Dtype);
        qrl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        qrl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qrSize;

        uint64_t kcSize = blockSize * MAX_N0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kcl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kcl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kcSize;
        }

        uint64_t krSize = blockSize * ropeHeadDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            krl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            krl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += krSize;
        }

        // SV
        off = sharel1Size;
        uint64_t wuvSize = MAX_M0 * kvLoraRank * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            wuvl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            wuvl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += wuvSize;
        }

        uint64_t qkSize = MAX_M0 * blockSize * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            qkl1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qkSize;
        }

        uint64_t vSize = MAX_N0 * blockSize * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            vl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            vl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += vSize;
        }
    }

    __aicore__ inline void InitAbsorbBuf()
    {
        /* (absorb)
         * Absorb = QC * WUK(T)
         *     m: queryTokens, n: kvLoraRank, k: nopeHeadDim
         *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
         * C = Absorb * K
         *     m: queryTokens, n: cachedTokens, k: kvLoraRank
         *     m0: MAX_M0, n0: (blockSize > MAX_N0 ? blockSize : MAX_N0), k0: MAX_K0
         * R = QR * KR
         *     m: queryTokens, n: cachedTokens, k: ropeHeadDim
         *     m0: MAX_M0, n0: (blockSize > MAX_N0 ? blockSize : MAX_N0), k0: MAX_K0
         *
         * Absorb = QK * K（T）
         *     m: queryTokens, n: kvLoraRank, k: cachedTokens
         *     m0: MAX_M0, n0: MAX_N0, k0: (blockSize > k0 ? blockSize : k0)
         * Absorb * WUV
         *     m: queryTokens, n: vHeadDim, k: kvLoraRank
         *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
         */
        uint64_t off = 0;
        uint64_t absorbSize = MAX_M0 * kvLoraRank * sizeof(Dtype);
        absorbl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        absorbl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += absorbSize;
        uint64_t sharel1Size = off;

        // QK (absorb)
        uint64_t qcSize = MAX_M0 * nopeHeadDim * sizeof(Dtype);
        aqcl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        aqcl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qcSize;

        uint64_t wukSize = MAX_N0 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            awukl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            awukl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += wukSize;
        }

        uint64_t kSize = (blockSize > MAX_N0 ? blockSize : MAX_N0) * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kSize;
        }

        uint64_t qrSize = MAX_M0 * ropeHeadDim * sizeof(Dtype);
        aqrl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        aqrl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qrSize;

        uint64_t krSize = (blockSize > MAX_N0 ? blockSize : MAX_N0) * ropeHeadDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akrl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akrl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += krSize;
        }

        // SV (absorb)
        off = sharel1Size;
        uint64_t ktSize = MAX_N0 * (blockSize > k0 ? blockSize : k0) * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            aktl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            aktl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ktSize;
        }

        uint64_t qkSize = MAX_M0 * (blockSize > k0 ? blockSize : k0) * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            aqkl1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            aqkl1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qkSize;
        }

        uint64_t wuvSize = MAX_N0 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            awuvl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            awuvl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += wuvSize;
        }
    }

    /*
     * KC = K * WUK:
     *     m: cachedTokens, n: nopeHeadDim, k: kvLoraRank
     *     m0: blockSize, n0: MAX_N0, k0: MAX_K0
     * QR * KR:
     *     m: queryTokens, n: cachedTokens, k: ropeHeadDim
     *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
     * QC * KC:
     *     m: queryTokens, n: cachedTokens, k: nopeHeadDim
     *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> query, int queryLen, int headIdx,
                                    __gm__ uint32_t *blockTable, int totalLen,
                                    GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mBlockPad = ROUND_UP(queryLen, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nLoop = DIV_ROUND_UP(totalLen, blockSize);
        int nBlockPad = ROUND_UP(blockSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int KCkLoop = DIV_ROUND_UP(kvLoraRank, k0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // copy QC (m0, nopeHeadDim) to L1
        CopyGmToL1Nd2Nz(qcl1aBuf, query, queryLen, nopeHeadDim,
                        nHeads * (nopeHeadDim + ropeHeadDim), mBlockPad);
        // copy QR (m0, ropeHeadDim) to L1
        CopyGmToL1Nd2Nz(qrl1aBuf, query[nopeHeadDim], queryLen, ropeHeadDim,
                        nHeads * (nopeHeadDim + ropeHeadDim), mBlockPad);
        // copy WUK (nopeHeadDim, kvLoraRank) to L1
        CopyGmToL1Nd2Nz(wukl1bBuf, wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank],
                        nopeHeadDim, kvLoraRank, kvLoraRank, nopeHeadDim);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        int curr = 0;
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // totalLen
            uint32_t block = blockTable[nIdx];
            int nSize = blockSize;
            if (nIdx * blockSize + nSize > totalLen) {
                nSize = totalLen - nIdx * blockSize;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID1);

            // calc KC = K * WUK
            int KCmSize = nSize;
            int KCmBlockPad = ROUND_UP(KCmSize, MBLOCKSIZE);
            int KCmBlockNum = KCmBlockPad / MBLOCKSIZE;
            int KCnLoop, KCn0;
            if (nopeHeadDim <= MAX_N0) {
                KCnLoop = 1;
                KCn0 = nopeHeadDim;
            } else {
                KCnLoop = DIV_ROUND_UP(nopeHeadDim, MAX_N0);
                KCn0 = MAX_N0;
            }
            int KCnBlockPad = ROUND_UP(KCn0, NBLOCKSIZE);
            int KCnBlockNum = KCnBlockPad / NBLOCKSIZE;
            for (int KCnIdx = 0; KCnIdx < KCnLoop; KCnIdx++) {  // nopeHeadDim
                int KCnSize = KCn0;
                int KCnOffset = KCnIdx * KCn0;
                if (KCnOffset + KCnSize > nopeHeadDim) {
                    KCnSize = nopeHeadDim - KCnOffset;
                    KCnBlockPad = ROUND_UP(KCnSize, NBLOCKSIZE);
                    KCnBlockNum = KCnBlockPad / NBLOCKSIZE;
                }
                int KCkBlockPad = ROUND_UP(k0, kBlockSize);
                int KCkBlockNum = KCkBlockPad / kBlockSize;

                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                for (int KCkIdx = 0; KCkIdx < KCkLoop; KCkIdx++) {  // kvLoraRank
                    int KCkSize = k0;
                    int KCkOffset = KCkIdx * k0;
                    if (KCkOffset + KCkSize > kvLoraRank) {
                        KCkSize = kvLoraRank - KCkOffset;
                        KCkBlockPad = ROUND_UP(KCkSize, kBlockSize);
                        KCkBlockNum = KCkBlockPad / kBlockSize;
                    }
                    // copy K (nSize, k0) to L1
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                    CopyGmToL1Nd2Nz(kl1aBuf[curr],
                                    kCache[block * blockSize * kvLoraRank + KCkOffset], KCmSize,
                                    KCkSize, kvLoraRank, KCmBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);
                    CopyToL0ACol(l0aBuf[curr], kl1aBuf[curr], KCmBlockNum, 0, KCkBlockNum);
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);

                    LoadData2dParams params(0, KCnBlockNum, 1, 0, 0, 0, inc);
                    for (int k = 0; k < KCkBlockNum; k++) {
                        uint64_t srcOffset =
                            (KCkOffset + k * kBlockSize) * nopeHeadDim + KCnOffset * kBlockSize;
                        uint64_t dstOffset = k * KCnBlockNum * (512 / sizeof(Dtype));
                        LoadData(l0bBuf[curr][dstOffset], wukl1bBuf[srcOffset], params);
                    }

                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    // do matmul to get KC (nSize, KCnSize) to L0C
                    CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], KCmBlockPad, KCnBlockPad,
                            KCkBlockPad, KCkIdx == 0);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    PipeBarrier<PIPE_M>();
                    curr = 1 - curr;
                }
                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

                // copy KC (nSize, KCnSize) from L0C to L1
                int kSize = KCnSize;
                int kBlockPad = ROUND_UP(kSize, kBlockSize);
                int kBlockNum = kBlockPad / kBlockSize;
                WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);
                CopyL0CToL1(kcl1bBuf[curr], l0cBuf, nBlockPad, kBlockPad, nBlockPad,
                            nBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE);

                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], qcl1aBuf[KCnOffset * mBlockPad], mBlockNum, 0,
                             kBlockNum);
                WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);
                CopyToL0BCol(l0bBuf[curr], kcl1bBuf[curr], nBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // do matmul QC * KC to L0C
                CalMmad(qksvl0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        KCnIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }

            // copy KR (nSize, ropeHeadDim) to L1
            int kSize = ropeHeadDim;
            int kBlockPad = ROUND_UP(kSize, kBlockSize);
            int kBlockNum = kBlockPad / kBlockSize;
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
            CopyGmToL1Nd2Nz(krl1bBuf[curr], vCache[block * blockSize * ropeHeadDim], nSize,
                            ropeHeadDim, ropeHeadDim, nBlockPad);

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            CopyToL0ACol(l0aBuf[curr], qrl1aBuf, mBlockNum, 0, kBlockNum);
            CopyToL0BCol(l0bBuf[curr], krl1bBuf[curr], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

            // do matmul QR * KR to L0C
            CalMmad(qksvl0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad, false);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_FIX>(EVENT_ID1);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID1);

            // copy final QK(m0, nSize) from L0C to GM
            CopyToGm(qk[nIdx * blockSize], qksvl0cBuf, queryLen, nSize, mBlockPad, qkStride);
            SetFlag<HardEvent::FIX_M>(EVENT_ID1);
            curr = 1 - curr;
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
    }

    /* V = k * WUV:
     *     m: cachedTokens, n: vHeadDim, k: kvLoraRank
     *     m0: blockSize, n0: MAX_N0, k0: MAX_K0
     * QK * V:
     *     m: queryTokens, n: vHeadDim, k: cachedTokens
     *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, int queryLen, int headIdx,
                                    __gm__ uint32_t *blockTable, int totalLen,
                                    GlobalTensor<Dtype> output)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int kLoop = DIV_ROUND_UP(totalLen, blockSize);
        int mBlockPad = ROUND_UP(queryLen, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int VkLoop = DIV_ROUND_UP(kvLoraRank, k0);

        int curr = 0;
        int wuvCurr = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);

        int VnLoop, Vn0;
        if (vHeadDim <= MAX_N0) {
            VnLoop = 1;
            Vn0 = vHeadDim;
        } else {
            VnLoop = DIV_ROUND_UP(vHeadDim, MAX_N0);
            Vn0 = MAX_N0;
        }
        int nBlockPad = ROUND_UP(Vn0, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        for (int VnIdx = 0; VnIdx < VnLoop; VnIdx++) {  // vHeadDim
            int VnSize = Vn0;
            int VnOffset = VnIdx * Vn0;
            if (VnOffset + VnSize > vHeadDim) {
                VnSize = vHeadDim - VnOffset;
                nBlockPad = ROUND_UP(VnSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }
            int kBlockPad = ROUND_UP(blockSize, kBlockSize);
            int kBlockNum = kBlockPad / kBlockSize;

            // copy WUV (VnSize, k0) to L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + wuvCurr);
            CopyGmToL1Nd2Nz(wuvl1bBuf[wuvCurr],
                            wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank +
                                 (nopeHeadDim + VnOffset) * kvLoraRank],
                            VnSize, kvLoraRank, kvLoraRank, nBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + wuvCurr);

            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // totalLen
                uint32_t block = blockTable[kIdx];
                int kSize = blockSize;
                if (kIdx * blockSize + kSize > totalLen) {
                    kSize = totalLen - kIdx * blockSize;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }

                // calc V = K * WUV
                int VmSize = kSize;
                int VmBlockPad = ROUND_UP(VmSize, MBLOCKSIZE);
                int VmBlockNum = VmBlockPad / MBLOCKSIZE;
                int VkBlockPad = ROUND_UP(k0, kBlockSize);
                int VkBlockNum = VkBlockPad / MBLOCKSIZE;
                for (int VkIdx = 0; VkIdx < VkLoop; VkIdx++) {  // kvLoraRank
                    int VkSize = k0;
                    int VkOffset = VkIdx * k0;
                    if (VkOffset + VkSize > kvLoraRank) {
                        VkSize = kvLoraRank - VkOffset;
                        VkBlockPad = ROUND_UP(VkSize, kBlockSize);
                        VkBlockNum = VkBlockPad / kBlockSize;
                    }
                    // copy K (kSize, k0) to L1
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                    CopyGmToL1Nd2Nz(kl1aBuf[curr],
                                    kCache[block * blockSize * kvLoraRank + VkOffset], VmBlockPad,
                                    VkSize, kvLoraRank, VmBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);
                    CopyToL0ACol(l0aBuf[curr], kl1aBuf[curr], VmBlockNum, 0, VkBlockNum);
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);

                    if (kIdx == 0 && VkIdx == 0) {
                        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + wuvCurr);
                    }
                    CopyToL0BCol(l0bBuf[curr], wuvl1bBuf[wuvCurr][VkOffset * nBlockPad], nBlockNum,
                                 0, VkBlockNum);
                    if (kIdx == kLoop - 1 && VkIdx == VkLoop - 1) {
                        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + wuvCurr);
                        wuvCurr = 1 - wuvCurr;
                    }

                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    // do matmul K * WUV to L0C
                    if (VkIdx == 0) {
                        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                    }
                    CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], VmBlockPad, nBlockPad, VkBlockPad,
                            VkIdx == 0);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    PipeBarrier<PIPE_M>();
                    curr = 1 - curr;
                }
                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

                // copy V (kSize, VnSize) from L0C to L1
                WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID2 + curr);
                CopyL0CToL1(vl1bBuf[curr], l0cBuf, kBlockPad, nBlockPad, kBlockPad,
                            kBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                SetFlag<HardEvent::FIX_MTE1>(EVENT_ID2 + curr);

                // copy QK (m0, kSize) to L1
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);
                CopyGmToL1Nd2Nz(qkl1aBuf[curr], qk[kIdx * blockSize], queryLen, kBlockPad, qkStride,
                                mBlockPad);

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], qkl1aBuf[curr], mBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);

                WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID2 + curr);
                CopyToL0BTCol(l0bBuf[curr], vl1bBuf[curr], nBlockNum, 0, kBlockNum, kBlockNum);
                SetFlag<HardEvent::MTE1_FIX>(EVENT_ID2 + curr);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                // do matmul QK * V to L0C
                if (kIdx == 0) {
                    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
                }
                CalMmad(qksvl0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                curr = 1 - curr;
            }
            // copy final output(m0, VnSize) from L0C to GM
            SetFlag<HardEvent::M_FIX>(EVENT_ID1);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
            CopyToGm(output[VnOffset], qksvl0cBuf, queryLen, VnSize, mBlockPad, nHeads * vHeadDim);
            SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }

    /* (absorb)
     * Absorb = QC * WUK(T)
     *     m: queryTokens, n: kvLoraRank, k: nopeHeadDim
     *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
     * C = Absorb * K
     *     m: queryTokens, n: cachedTokens, k: kvLoraRank
     *     m0: MAX_M0, n0: (blockSize > MAX_N0 ? blockSize : MAX_N0), k0: MAX_K0
     * R = QR * KR
     *     m: queryTokens, n: cachedTokens, k: ropeHeadDim
     *     m0: MAX_M0, n0: (blockSize > MAX_N0 ? blockSize : MAX_N0), k0: MAX_K0
     */
    __aicore__ inline void RunAicQKAbsorb(GlobalTensor<Dtype> query, int queryLen, int headIdx,
                                          __gm__ uint32_t *blockTable, int totalLen,
                                          GlobalTensor<Dtype> qk, int32_t topkOffset[])
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);

        // Absorb = QC * WUK(T)
        int mBlockPad = ROUND_UP(queryLen, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = MAX_N0;
        int nLoop = DIV_ROUND_UP(kvLoraRank, MAX_N0);
        int nBlockPad = MAX_N0;
        int nBlockNum = MAX_N0 / NBLOCKSIZE;
        int kSize = k0;
        int kLoop = DIV_ROUND_UP(nopeHeadDim, k0);
        int kBlockPad = k0;
        int kBlockNum = k0 / kBlockSize;

        bool doTopK = false;
        if (totalLen > topK && topK > 0) {
            doTopK = true;
        }

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // copy QC (queryTokens, nopeHeadDim) to L1
        CopyGmToL1Nd2Nz(aqcl1aBuf, query, queryLen, nopeHeadDim,
                        nHeads * (nopeHeadDim + ropeHeadDim), mBlockPad);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0);

        int curr = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // kvLoraRank
            int nOffset = nIdx * MAX_N0;
            if (nOffset + nSize > kvLoraRank) {
                nSize = kvLoraRank - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            kSize = k0;
            kBlockPad = k0;
            kBlockNum = k0 / kBlockSize;
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // nopeHeadDim
                int kOffset = kIdx * k0;
                if (kOffset + kSize > nopeHeadDim) {
                    kSize = nopeHeadDim - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }

                // copy WUK(T) (nSize, k0) to L1
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + curr);
                CopyGmToL1Nd2Nz(awukl1bBuf[curr],
                                wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank +
                                     kOffset * kvLoraRank + nOffset],
                                kSize, nSize, kvLoraRank, kBlockPad);

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], aqcl1aBuf[kOffset * mBlockPad], mBlockNum, 0, kBlockNum);

                CopyToL0BTCol(l0bBuf[curr], awukl1bBuf[curr], nBlockNum, 0, kBlockNum, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + curr);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad Absorb (queryTokens, nSize) = QC * WUK(T)
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            // copy Absorb (queryTokens, nSize) to L1
            CopyL0CToL1(absorbl1aBuf[nOffset * mBlockPad], l0cBuf, mBlockPad, nBlockPad, mBlockPad,
                        mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);

        // C = Absorb * K
        int n0, nTotalLen;
        if (doTopK) {
            n0 = MAX_N0;
            nTotalLen = topK;
        } else {
            n0 = blockSize;
            nTotalLen = totalLen;
        }
        nSize = n0;
        nLoop = DIV_ROUND_UP(nTotalLen, n0);
        nBlockPad = n0;
        nBlockNum = n0 / NBLOCKSIZE;
        kLoop = DIV_ROUND_UP(kvLoraRank, k0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // copy QR (queryTokens, ropeHeadDim) to L1
        CopyGmToL1Nd2Nz(aqrl1aBuf, query[nopeHeadDim], queryLen, ropeHeadDim,
                        nHeads * (nopeHeadDim + ropeHeadDim), mBlockPad);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        uint32_t block;
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // totalLen or topK
            int nOffset = nIdx * n0;
            if (nOffset + nSize > nTotalLen) {
                nSize = nTotalLen - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }
            if (!doTopK) {
                block = blockTable[nIdx];
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            kSize = k0;
            kBlockPad = k0;
            kBlockNum = k0 / kBlockSize;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLoraRank
                int kOffset = kIdx * k0;
                if (kOffset + kSize > kvLoraRank) {
                    kSize = kvLoraRank - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }

                // copy K (n0, k0) to L1
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                if (doTopK) {
                    Nd2NzParams params(1, 1, kSize, 0, kvLoraRank, nBlockPad, 1, 0);
                    for (int i = 0; i < nSize; i++) {
                        DataCopy(akl1bBuf[curr][i * kBlockSize],
                                 kCache[(topkOffset[nOffset + i]) * kvLoraRank + kOffset], params);
                    }
                } else {
                    CopyGmToL1Nd2Nz(akl1bBuf[curr],
                                    kCache[block * blockSize * kvLoraRank + kOffset], nSize, kSize,
                                    kvLoraRank, nBlockPad);
                }

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0BCol(l0bBuf[curr], akl1bBuf[curr], nBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                CopyToL0ACol(l0aBuf[curr], absorbl1aBuf[kOffset * mBlockPad], mBlockNum, 0,
                             kBlockNum);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad C (queryTokens, n0) = Absorb * K
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }

            kSize = ropeHeadDim;
            int kBlockPad = ROUND_UP(kSize, kBlockSize);
            int kBlockNum = kBlockPad / kBlockSize;
            // copy KR (n0, ropeHeadDim) to L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
            if (doTopK) {
                Nd2NzParams params(1, 1, ropeHeadDim, 0, ropeHeadDim, nBlockPad, 1, 0);
                for (int i = 0; i < nSize; i++) {
                    DataCopy(akrl1bBuf[curr][i * kBlockSize],
                             vCache[(topkOffset[nOffset + i]) * ropeHeadDim], params);
                }
            } else {
                CopyGmToL1Nd2Nz(akrl1bBuf[curr], vCache[block * blockSize * ropeHeadDim], nSize,
                                ropeHeadDim, ropeHeadDim, nBlockPad);
            }

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            CopyToL0ACol(l0aBuf[curr], aqrl1aBuf, mBlockNum, 0, kBlockNum);
            CopyToL0BCol(l0bBuf[curr], akrl1bBuf[curr], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

            // mmad R (queryTokens, n0) = QR * KR
            CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad, false);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy final QK(queryTokens, nSize) from L0C to GM
            CopyToGm(qk[nIdx * n0], l0cBuf, queryLen, nSize, mBlockPad, qkStride);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            curr = 1 - curr;
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

    /* (absorb)
     * Absorb = QK * K（T）
     *     m: queryTokens, n: kvLoraRank, k: cachedTokens
     *     m0: MAX_M0, n0: MAX_N0, k0: (blockSize > k0 ? blockSize : k0)
     * Absorb * WUV
     *     m: queryTokens, n: vHeadDim, k: kvLoraRank
     *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
     */
    __aicore__ inline void RunAicSVAbsorb(GlobalTensor<Dtype> qk, int queryLen, int headIdx,
                                          __gm__ uint32_t *blockTable, int totalLen,
                                          GlobalTensor<Dtype> output, int32_t topkOffset[])
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);

        // Absorb = QK * K（T）
        int mBlockPad = ROUND_UP(queryLen, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = MAX_N0;
        int nLoop = DIV_ROUND_UP(kvLoraRank, MAX_N0);
        int nBlockPad = MAX_N0;
        int nBlockNum = MAX_N0 / NBLOCKSIZE;
        int K0, kTotalLen, kSize, kLoop, kBlockPad, kBlockNum;

        bool doTopK = false;
        if (totalLen > topK && topK > 0) {
            doTopK = true;
            K0 = k0;
            kTotalLen = topK;
        } else {
            K0 = blockSize;
            kTotalLen = totalLen;
        }
        kSize = K0;
        kLoop = DIV_ROUND_UP(kTotalLen, K0);
        kBlockPad = K0;
        kBlockNum = K0 / kBlockSize;

        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0);

        int curr = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // kvLoraRank
            int nOffset = nIdx * MAX_N0;
            if (nOffset + nSize > kvLoraRank) {
                nSize = kvLoraRank - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            kSize = K0;
            kBlockPad = K0;
            kBlockNum = K0 / kBlockSize;
            uint32_t block;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // totalLen or topK
                int kOffset = kIdx * K0;
                if (kOffset + kSize > kTotalLen) {
                    kSize = kTotalLen - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }
                if (!doTopK) {
                    block = blockTable[kIdx];
                }

                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);
                // copy QK (queryTokens, K0) to L1
                CopyGmToL1Nd2Nz(aqkl1aBuf[curr], qk[kOffset], queryLen, kBlockPad, qkStride,
                                mBlockPad);

                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                // copy K(T) (nSize, K0) to L1
                if (doTopK) {
                    Nd2NzParams params(1, 1, nSize, 0, kvLoraRank, kBlockPad, 1, 0);
                    for (int i = 0; i < kSize; i++) {
                        DataCopy(aktl1bBuf[curr][i * kBlockSize],
                                 kCache[(topkOffset[kOffset + i]) * kvLoraRank + nOffset], params);
                    }
                } else {
                    CopyGmToL1Nd2Nz(aktl1bBuf[curr],
                                    kCache[block * blockSize * kvLoraRank + nOffset], kSize, nSize,
                                    kvLoraRank, kBlockPad);
                }

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], aqkl1aBuf[curr], mBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);

                CopyToL0BTCol(l0bBuf[curr], aktl1bBuf[curr], nBlockNum, 0, kBlockNum, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad Absorb (queryTokens, nSize) = QK * K（T）
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy Absorb (queryTokens, nSize) to L1
            CopyL0CToL1(absorbl1aBuf[nOffset * mBlockPad], l0cBuf, mBlockPad, nBlockPad, mBlockPad,
                        mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID0);

        // Absorb * WUV
        nSize = MAX_N0;
        nLoop = DIV_ROUND_UP(vHeadDim, MAX_N0);
        nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
        nBlockNum = nBlockPad / NBLOCKSIZE;
        kSize = k0;
        kLoop = DIV_ROUND_UP(kvLoraRank, k0);
        kBlockPad = k0;
        kBlockNum = k0 / kBlockSize;

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // vHeadDim
            int nOffset = nIdx * MAX_N0;
            if (nOffset + nSize > vHeadDim) {
                nSize = vHeadDim - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLoraRank
                int kOffset = kIdx * k0;
                if (kOffset + kSize > kvLoraRank) {
                    kSize = kvLoraRank - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }

                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + curr);
                // copy WUV (nSize, kSize) to L1
                CopyGmToL1Nd2Nz(awuvl1bBuf[curr],
                                wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank +
                                     (nopeHeadDim + nOffset) * kvLoraRank + kOffset],
                                nSize, kSize, kvLoraRank, nBlockPad);

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0BCol(l0bBuf[curr], awuvl1bBuf[curr], nBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + curr);
                CopyToL0ACol(l0aBuf[curr], absorbl1aBuf[kOffset * mBlockPad], mBlockNum, 0,
                             kBlockNum);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            CopyToGm(output[nOffset], l0cBuf, queryLen, nSize, mBlockPad, nHeads * vHeadDim);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
    }

    /*
     * When the totalLen exceeds MAX_SUB_CONTEXT_SIZE in the RunAivSoftmaxLong function,
     * 4 rows must be reserved for the exp buffer (float type) used in softmax.
     * Therefore, m0 must be less than or equal to MAX_M0 - 4.
     */
    __aicore__ inline uint32_t GetOptimalM0(int queryLen, int cachedLen)
    {
        if (topK > 0) {
            return 1;
        }
        if (queryLen <= SEQLEN_64) {
            return 16;
        } else {
            int totalLen = queryLen + cachedLen;
            if (totalLen <= SEQLEN_12K) {
                return 128;
            } else if (totalLen <= SEQLEN_20K) {
                return 112;
            } else if (totalLen <= SEQLEN_24K) {
                return 96;
            } else if (totalLen <= SEQLEN_30K) {
                return 80;
            } else if (totalLen <= SEQLEN_48K) {
                return 64;
            } else if (totalLen <= SEQLEN_60K) {
                return 48;
            } else if (totalLen <= SEQLEN_96K) {
                return 32;
            } else {
                return 16;
            }
        }
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskOffset, lastQueryTaskLen, lastHeadIdx, last, lastOutOffset,
            lastCalcLen;
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
                                    batchIdx * maxNumBlock * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryNum = DIV_ROUND_UP(queryLen, m0);
            int taskNum = queryNum * nHeads;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int headIdx = idx % nHeads;
                int queryIdx = idx / nHeads;
                int queryTaskLen = m0;
                int queryTaskStart = queryIdx * m0;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                // do queryIdx & headIdx's QK
                uint32_t qOffset =
                    (queryTaskOffset * nHeads + headIdx) * (ropeHeadDim + nopeHeadDim);
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;

                if (calcLen > topK && topK > 0) {
                    for (int i = 0; i < topK; i++) {
                        uint32_t topkIndex = topkIndices[queryTaskOffset * topK + i];
                        uint32_t block = blockTable[topkIndex / blockSize];
                        topkKVOffset[curr][i] = block * blockSize + (topkIndex % blockSize);
                    }
                }
                dbg_printf("block%d: {batch %d, query [%u - %u), headIdx %u}"
                           " use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           headIdx, curr);
                RunAicQKAbsorb(qWithQr[qOffset], queryTaskLen, headIdx, blockTable, calcLen,
                               qk[curr], topkKVOffset[curr]);
                ffts_cross_core_sync(PIPE_FIX, config);

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(1);
                    // do softmax * V
                    dbg_printf("block%d: {batch %d, query [%u - %u), headIdx %u}"
                               " use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, lastHeadIdx, last);
                    RunAicSVAbsorb(qk[last], lastQueryTaskLen, lastHeadIdx, lastBlockTable,
                                   lastCalcLen, output[lastOutOffset * vHeadDim],
                                   topkKVOffset[last]);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastOutOffset = queryTaskOffset * nHeads + headIdx;
                lastQueryTaskLen = queryTaskLen;
                lastHeadIdx = headIdx;
                lastBlockTable = blockTable;
                lastCalcLen = calcLen;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(1);
            dbg_printf("block%d: {batch %d, query [%u - %u), headIdx %u}"
                       " use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, lastHeadIdx, last);
            RunAicSVAbsorb(qk[last], lastQueryTaskLen, lastHeadIdx, lastBlockTable, lastCalcLen,
                           output[lastOutOffset * vHeadDim], topkKVOffset[last]);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t flagIdx = 1;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlock * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryNum = DIV_ROUND_UP(queryLen, m0);
            int taskNum = queryNum * nHeads;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int headIdx = idx % nHeads;
                int queryIdx = idx / nHeads;
                int queryTaskLen = m0;
                int queryTaskStart = queryIdx * m0;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                int nWork = queryTaskLen;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                uint32_t subIdx = get_subblockid();
                int nWorkStart = subIdx * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t qkOffset = nWorkStart * qkStride;
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcSoftmaxLen = cachedLen + queryTaskStart + nWorkStart + 1;
                uint32_t outN = ROUND_UP(cachedLen + queryTaskStart + queryTaskLen, blockSize);

                if (calcSoftmaxLen > topK && topK > 0) {
                    calcSoftmaxLen = topK;
                    outN = ROUND_UP(topK, blockSize);
                }
                // wait aic qk done
                wait_flag_dev(0);

                // do softmax
                int dbgBlockIdx = block_idx;
                dbg_printf(
                    "block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - "
                    "%u), headIdx %u} use %d temp buf: SOFTMAX\n",
                    dbgBlockIdx, subIdx, batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                    nWorkStart, nWorkStart + nWorkCurCore, headIdx, curr);
                RunAivSoftmax(
                    (__gm__ Dtype *)qk[curr][qkOffset].GetPhyAddr(),
                    m0 > (MAX_M0 - 4)
                        ? 0
                        : (__gm__ float *)qk[curr][(m0 + subIdx * 2) * qkStride].GetPhyAddr(),
                    nWorkCurCore, qkStride, calcSoftmaxLen, outN, 0, 1, true, scale);

                ffts_cross_core_sync(PIPE_MTE3, config);
                curr = 1 - curr;
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
    GlobalTensor<Dtype> qWithQr;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> vCache;
    GlobalTensor<Dtype> wkvb;
    __gm__ int32_t *topkIndices;
    GlobalTensor<Dtype> output;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
    int32_t topkKVOffset[PINGPONG_BUF_NUM][2048];

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;

    uint32_t nHeads;
    uint32_t ropeHeadDim;
    uint32_t nopeHeadDim;
    uint32_t vHeadDim;
    uint32_t kvLoraRank;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlock;
    uint32_t maxSeqLen;
    float scale;
    uint32_t topK;
    uint32_t qkStride;

    int k0;
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<float> l0cBuf;                    // event 0
    LocalTensor<float> qksvl0cBuf;                // event 1

    // QK & SV share
    LocalTensor<Dtype> kl1aBuf[PINGPONG_BUF_NUM];  // event 4/5

    // QK
    LocalTensor<Dtype> wukl1bBuf;                   // event 0
    LocalTensor<Dtype> qcl1aBuf;                    // event 0
    LocalTensor<Dtype> qrl1aBuf;                    // event 0
    LocalTensor<Dtype> kcl1bBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> krl1bBuf[PINGPONG_BUF_NUM];  // event 2/3

    // SV
    LocalTensor<Dtype> wuvl1bBuf[PINGPONG_BUF_NUM];  // event 6/7
    LocalTensor<Dtype> qkl1aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> vl1bBuf[PINGPONG_BUF_NUM];    // event 2/3

    // QK & SV (absorb) share
    LocalTensor<Dtype> absorbl1aBuf;  // event 0

    // QK (absorb)
    LocalTensor<Dtype> aqcl1aBuf;                     // event 0
    LocalTensor<Dtype> awukl1bBuf[PINGPONG_BUF_NUM];  // event 6/7
    LocalTensor<Dtype> aqrl1aBuf;                     // event 0
    LocalTensor<Dtype> akrl1bBuf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<Dtype> akl1bBuf[PINGPONG_BUF_NUM];    // event 4/5

    // SV (absorb)
    LocalTensor<Dtype> aqkl1aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> aktl1bBuf[PINGPONG_BUF_NUM];   // event 4/5
    LocalTensor<Dtype> awuvl1bBuf[PINGPONG_BUF_NUM];  // event 6/7
};

#define MLA_FUNC_DEFINE(dtype)                                                                    \
    extern "C" __global__ __aicore__ void mla_##dtype(                                            \
        GM_ADDR qWithQr, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR wkvb, GM_ADDR topkIndices,       \
        GM_ADDR qk, GM_ADDR output, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, \
        GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim, uint32_t nopeHeadDim,         \
        uint32_t vHeadDim, uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch,               \
        uint32_t maxNumBlock, float scale, uint32_t topK)                                         \
    {                                                                                             \
        MLA<dtype> op;                                                                            \
        op.Init(qWithQr, kCache, vCache, wkvb, topkIndices, qk, output, queryStartLoc, queryLens, \
                cachedLens, blockTables, nHeads, ropeHeadDim, nopeHeadDim, vHeadDim, kvLoraRank,  \
                blockSize, batch, maxNumBlock, scale, topK);                                      \
        op.Run();                                                                                 \
    }