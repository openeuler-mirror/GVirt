/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"
#include "kernel_param.h"
// #define XLITE_KERNEL_DEBUG
#include "debug.h"
#include "softmax_attn_aiv.h"

#define MAX_M0 128
#define MAX_N0 128
#define MBLOCKSIZE 16
#define NBLOCKSIZE 16

template <typename Dtype>
class FlashMLA
{
public:
    __aicore__ inline FlashMLA()
    {
    }

    __aicore__ inline void Init(GM_ADDR qWithQr, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR wkvb,
                                GM_ADDR qk, GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR lastMax,
                                GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                                uint32_t nopeHeadDim, uint32_t vHeadDim, uint32_t kvLoraRank,
                                uint32_t blockSize, uint32_t batch, uint32_t maxNumBlocks,
                                float scale, uint32_t nz)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->qWithQr.SetGlobalBuffer((__gm__ Dtype *)qWithQr);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->vCache.SetGlobalBuffer((__gm__ Dtype *)vCache);
        this->wkvb.SetGlobalBuffer((__gm__ Dtype *)wkvb);
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
        this->maxNumBlocks = maxNumBlocks;
        this->scale = scale;
        this->nz = nz;
        this->tileSizeOfCachedKV = GetTileSizeOfCachedKV(block_num);
        this->maxSeqLen = maxNumBlocks * blockSize;
        this->blockIdx = block_idx;
        this->subBlockIdx = get_subblockid();
        this->nextBlockIdx = (blockIdx + 1) % block_num;
        this->prevBlockIdx = blockIdx == 0 ? (block_num - 1) : (blockIdx - 1);
        this->setNextGeneration = 1;
        this->waitPrevGeneration = 1;
        this->nopeRopeHeadDim = nopeHeadDim + ropeHeadDim;
        this->headTileSize = nHeads;
        this->nHeadTiles = nHeads / headTileSize;

        this->qk[0].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * MAX_M0 * tileSizeOfCachedKV);
        this->qk[1].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * MAX_M0 * tileSizeOfCachedKV +
                                    block_num * MAX_M0 * tileSizeOfCachedKV);
        this->sv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * MAX_M0 * vHeadDim);
        this->sv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * MAX_M0 * vHeadDim +
                                    block_num * MAX_M0 * vHeadDim);
        this->max[0].SetGlobalBuffer(((__gm__ float *)max) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0);
        this->max[1].SetGlobalBuffer(((__gm__ float *)max) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0 + block_num * MAX_M0 * 2);
        this->sum[0].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0);
        this->sum[1].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * MAX_M0 * 2 +
                                     subBlockIdx * MAX_M0 + block_num * MAX_M0 * 2);
        this->lastMax.SetGlobalBuffer((__gm__ float *)lastMax);
        this->lastSum.SetGlobalBuffer((__gm__ float *)lastSum);
        this->setNextSync = (__gm__ int32_t *)sync + blockIdx * 2 + subBlockIdx;
        this->waitPrevSync = (__gm__ int32_t *)sync + prevBlockIdx * 2 + subBlockIdx;

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
         *     QC: (queryTokens, headTileSize, nopeHeadDim)
         *     WUK: (headTileSize, nopeHeadDim, kvLoraRank)
         *     Absorb: (headTileSize, queryTokens, kvLoraRank)
         *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
         * C = Absorb * K
         *     Absorb: (headTileSize * queryTokens, kvLoraRank)
         *     K: (cachedTokens, kvLoraRank)
         *     C: (headTileSize, queryTokens, cachedTokens)
         *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
         * R = QR * KR
         *     QR: (headTileSize * queryTokens, ropeHeadDim)
         *     K: (cachedTokens, ropeHeadDim)
         *     R: (headTileSize, queryTokens, cachedTokens)
         *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
         *
         * Absorb = QK * K(T)
         *     QK: (headTileSize, queryTokens, cachedTokens)
         *     K: (cachedTokens, kvLoraRank)
         *     Absorb: (headTileSize, queryTokens, kvLoraRank)
         *     m0: MAX_M0, n0: MAX_N0, k0: blockSize
         * Absorb * WUV
         *     Absorb: (headTileSize, queryTokens, kvLoraRank)
         *     WUV: (headTileSize, vHeadDim, kvLoraRank)
         *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
         */
        uint64_t off = 0;
        uint64_t absorbSize = MAX_M0 * kvLoraRank * sizeof(Dtype);
        absorbl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        absorbl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += absorbSize;
        uint64_t sharel1Size = off;

        // QK (absorb)
        uint64_t qcrSize = MAX_M0 * nopeRopeHeadDim * sizeof(Dtype);
        aqcrl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        aqcrl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qcrSize;
        uint32_t reuse = off;

        uint64_t wukSize = MAX_N0 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            awukl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            awukl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += wukSize;
        }

        off = reuse;
        uint64_t kSize = blockSize * 4 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kSize;
        }

        uint64_t krSize = blockSize * ropeHeadDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akrl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akrl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += krSize;
        }

        // SV (absorb)
        off = sharel1Size;
        uint64_t ktSize = MAX_N0 * blockSize * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            aktl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            aktl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ktSize;
        }

        uint64_t qkSize = MAX_M0 * blockSize * sizeof(Dtype);
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
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mBlockPad = ROUND_UP(queryLen, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nIdxStart = kvOffset / blockSize;
        int nLoop = DIV_ROUND_UP(kvLen, blockSize);
        int nBlockPad = ROUND_UP(blockSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int KCkLoop = DIV_ROUND_UP(kvLoraRank, k0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // copy QC (m0, nopeHeadDim) to L1
        CopyGmToL1Nd2Nz(qcl1aBuf, query, queryLen, nopeHeadDim, nHeads * nopeRopeHeadDim,
                        mBlockPad);
        // copy QR (m0, ropeHeadDim) to L1
        CopyGmToL1Nd2Nz(qrl1aBuf, query[nopeHeadDim], queryLen, ropeHeadDim,
                        nHeads * nopeRopeHeadDim, mBlockPad);
        // copy WUK (nopeHeadDim, kvLoraRank) to L1
        if (nz == 0) {
            CopyGmToL1Nd2Nz(wukl1bBuf, wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank],
                            nopeHeadDim, kvLoraRank, kvLoraRank, nopeHeadDim);
        } else {
            CopyGmToL1(wukl1bBuf, wkvb[headIdx * (nopeHeadDim + vHeadDim) * kBlockSize],
                       nopeHeadDim, kvLoraRank / kBlockSize, nHeads * (nopeHeadDim + vHeadDim));
        }
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
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // kvLen
            uint32_t block = blockTable[nIdx + nIdxStart];
            int nSize = blockSize;
            if (nIdx * blockSize + nSize > kvLen) {
                nSize = kvLen - nIdx * blockSize;
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
            CopyToGm(qk[nIdx * blockSize], qksvl0cBuf, queryLen, nSize, mBlockPad,
                     tileSizeOfCachedKV);
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
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> sv)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int kIdxStart = kvOffset / blockSize;
        int kLoop = DIV_ROUND_UP(kvLen, blockSize);
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
            if (nz == 0) {
                CopyGmToL1Nd2Nz(wuvl1bBuf[wuvCurr],
                                wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank +
                                     (nopeHeadDim + VnOffset) * kvLoraRank],
                                VnSize, kvLoraRank, kvLoraRank, nBlockPad);
            } else {
                CopyGmToL1(wuvl1bBuf[wuvCurr],
                           wkvb[headIdx * (nopeHeadDim + vHeadDim) * kBlockSize +
                                (nopeHeadDim + VnOffset) * kBlockSize],
                           VnSize, kvLoraRank / kBlockSize, nHeads * (nopeHeadDim + vHeadDim));
            }
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + wuvCurr);

            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLen
                uint32_t block = blockTable[kIdx + kIdxStart];
                int kSize = blockSize;
                if (kIdx * blockSize + kSize > kvLen) {
                    kSize = kvLen - kIdx * blockSize;
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
                CopyGmToL1Nd2Nz(qkl1aBuf[curr], qk[kIdx * blockSize], queryLen, kBlockPad,
                                tileSizeOfCachedKV, mBlockPad);

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
            // copy final SV(m0, VnSize) from L0C to GM
            SetFlag<HardEvent::M_FIX>(EVENT_ID1);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
            CopyToGm(sv[VnOffset], qksvl0cBuf, queryLen, VnSize, mBlockPad, vHeadDim);
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
     *     QC: (queryTokens, headTileSize, nopeHeadDim)
     *     WUK: (headTileSize, nopeHeadDim, kvLoraRank)
     *     Absorb: (headTileSize, queryTokens, kvLoraRank)
     *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
     * C = Absorb * K
     *     Absorb: (headTileSize * queryTokens, kvLoraRank)
     *     K: (cachedTokens, kvLoraRank)
     *     C: (headTileSize, queryTokens, cachedTokens)
     *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
     * R = QR * KR
     *     QR: (headTileSize * queryTokens, ropeHeadDim)
     *     K: (cachedTokens, ropeHeadDim)
     *     R: (headTileSize, queryTokens, cachedTokens)
     *     m0: MAX_M0, n0: blockSize, k0: MAX_K0
     */
    __aicore__ inline void RunAicQKAbsorb(GlobalTensor<Dtype> query, int queryLen, int headIdx,
                                          __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                          GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);

        // Absorb = QC * WUK(T)
        int mBlockPad = ROUND_UP(queryLen, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int mhSize = queryLen * headTileSize;
        int mhBlockPad = ROUND_UP(mhSize, MBLOCKSIZE);
        int mhBlockNum = mhBlockPad / MBLOCKSIZE;
        int nIdxStart = kvOffset / blockSize;
        int nSize = MAX_N0;
        int nLoop = DIV_ROUND_UP(kvLoraRank, MAX_N0);
        int nBlockPad = MAX_N0;
        int nBlockNum = MAX_N0 / NBLOCKSIZE;
        int kSize = k0;
        int kLoop = DIV_ROUND_UP(nopeHeadDim, k0);
        int kBlockPad = k0;
        int kBlockNum = k0 / kBlockSize;
        int nzHeadStride = queryLen * kBlockSize;
        int cubeSize = 512 / sizeof(Dtype);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        /* copy QC & QR (queryTokens, headTileSize, nopeHeadDim + ropeHeadDim) to L1
         * L1 (headTileSize, queryTokens, nopeHeadDim + ropeHeadDim)
         */
        for (int hIdx = 0; hIdx < headTileSize; hIdx++) {
            int nzHeadOffset = hIdx * nzHeadStride;
            CopyGmToL1Nd2Nz(aqcrl1aBuf[nzHeadOffset], query[hIdx * nopeRopeHeadDim], queryLen,
                            nopeRopeHeadDim, nHeads * nopeRopeHeadDim, mhBlockPad);
        }
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
        for (int hIdx = 0; hIdx < headTileSize; hIdx++, headIdx++) {  // headTileSize
            int nzHeadOffset = hIdx * nzHeadStride;
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
                    if (nz == 0) {
                        CopyGmToL1Nd2Nz(awukl1bBuf[curr],
                                        wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank +
                                             kOffset * kvLoraRank + nOffset],
                                        kSize, nSize, kvLoraRank, kBlockPad);
                    } else {
                        CopyGmToL1(awukl1bBuf[curr],
                                   wkvb[headIdx * (nopeHeadDim + vHeadDim) * kBlockSize +
                                        kOffset * kBlockSize +
                                        nOffset * nHeads * (nopeHeadDim + vHeadDim)],
                                   kSize, nSize / kBlockSize, nHeads * (nopeHeadDim + vHeadDim));
                    }

                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    LoadData2dParams params(0, mBlockNum, 1, 0, kBlockNum - 1, 0, inc);
                    int aqcrl1aOffset = nzHeadOffset + kOffset * mhBlockPad;
                    for (int k = 0; k < kBlockNum; k++) {
                        LoadData(l0aBuf[curr][k * cubeSize],
                                 aqcrl1aBuf[aqcrl1aOffset + k * mhBlockNum * cubeSize], params);
                    }

                    CopyToL0BTCol(l0bBuf[curr], awukl1bBuf[curr], nBlockNum, 0, kBlockNum,
                                  kBlockNum);
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
                CopyL0CToL1(absorbl1aBuf[nOffset * mhBlockPad + nzHeadOffset], l0cBuf, queryLen,
                            nBlockPad, mBlockPad, mhBlockPad);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);

        // C = Absorb * K
        nSize = blockSize;
        nLoop = DIV_ROUND_UP(kvLen, blockSize);
        nBlockPad = blockSize;
        nBlockNum = blockSize / NBLOCKSIZE;
        kLoop = DIV_ROUND_UP(kvLoraRank, k0);

        SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        int pingpongL1B = 0;
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // kvLen
            uint32_t block = blockTable[nIdx + nIdxStart];
            int nOffset = nIdx * blockSize;
            if (nOffset + nSize > kvLen) {
                nSize = kvLen - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            kSize = k0;
            kBlockPad = k0;
            kBlockNum = k0 / kBlockSize;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLoraRank
                int kIdx4 = kIdx % 4;
                int kOffset = kIdx * k0;
                if (kOffset + kSize > kvLoraRank) {
                    kSize = kvLoraRank - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }

                // copy K (blockSize, 4 * k0) to L1
                if (kIdx4 == 0) {
                    int kRemSize = 4 * k0;
                    if (kOffset + kRemSize > kvLoraRank) {
                        kRemSize = kvLoraRank - kOffset;
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    CopyGmToL1Nd2Nz(akl1bBuf[pingpongL1B],
                                    kCache[block * blockSize * kvLoraRank + kOffset], nSize,
                                    kRemSize, kvLoraRank, nBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                }

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0BCol(l0bBuf[curr], akl1bBuf[pingpongL1B], nBlockNum,
                             kIdx4 * k0 / kBlockSize, kBlockNum);
                if (kIdx4 == 3) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    pingpongL1B ^= 1;
                }
                CopyToL0ACol(l0aBuf[curr], absorbl1aBuf[kOffset * mhBlockPad], mhBlockNum, 0,
                             kBlockNum);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad C (headTileSize * queryTokens, blockSize) = Absorb * K
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mhBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            if (kLoop % 4 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                pingpongL1B ^= 1;
            }

            kSize = ropeHeadDim;
            int kBlockPad = ROUND_UP(kSize, kBlockSize);
            int kBlockNum = kBlockPad / kBlockSize;
            // copy KR (blockSize, ropeHeadDim) to L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
            CopyGmToL1Nd2Nz(akrl1bBuf[curr], vCache[block * blockSize * ropeHeadDim], nSize,
                            ropeHeadDim, ropeHeadDim, nBlockPad);

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            CopyToL0ACol(l0aBuf[curr], aqcrl1aBuf[nopeHeadDim * mhBlockPad], mhBlockNum, 0,
                         kBlockNum);
            CopyToL0BCol(l0bBuf[curr], akrl1bBuf[curr], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

            // mmad R (headTileSize * queryTokens, blockSize) = QR * KR
            CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mhBlockPad, nBlockPad, kBlockPad, false);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy final QK(headTileSize, queryTokens, nSize) from L0C to GM
            CopyToGm(qk[nIdx * blockSize], l0cBuf, mhSize, nSize, mhBlockPad, tileSizeOfCachedKV);
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
     * Absorb = QK * K(T)
     *     QK: (headTileSize, queryTokens, cachedTokens)
     *     K: (cachedTokens, kvLoraRank)
     *     Absorb: (headTileSize, queryTokens, kvLoraRank)
     *     m0: MAX_M0, n0: MAX_N0, k0: blockSize
     * Absorb * WUV
     *     Absorb: (headTileSize, queryTokens, kvLoraRank)
     *     WUV: (headTileSize, vHeadDim, kvLoraRank)
     *     m0: MAX_M0, n0: MAX_N0, k0: MAX_K0
     */
    __aicore__ inline void RunAicSVAbsorb(GlobalTensor<Dtype> qk, int queryLen, int headIdx,
                                          __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                          GlobalTensor<Dtype> sv)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int kIdxStart = kvOffset / blockSize;

        // Absorb = QK * K(T)
        int mBlockPad = ROUND_UP(queryLen, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int mhSize = queryLen * headTileSize;
        int mhBlockPad = ROUND_UP(mhSize, MBLOCKSIZE);
        int mhBlockNum = mhBlockPad / MBLOCKSIZE;
        int nSize = MAX_N0;
        int nLoop = DIV_ROUND_UP(kvLoraRank, MAX_N0);
        int nBlockPad = MAX_N0;
        int nBlockNum = MAX_N0 / NBLOCKSIZE;
        int kSize = blockSize;
        int kLoop = DIV_ROUND_UP(kvLen, blockSize);
        int kBlockPad = blockSize;
        int kBlockNum = blockSize / kBlockSize;

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

            kSize = blockSize;
            kBlockPad = blockSize;
            kBlockNum = blockSize / kBlockSize;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLen
                uint32_t block = blockTable[kIdx + kIdxStart];
                int kOffset = kIdx * blockSize;
                if (kOffset + kSize > kvLen) {
                    kSize = kvLen - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }

                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);
                // copy QK (headTileSize, queryTokens, blockSize) to L1
                CopyGmToL1Nd2Nz(aqkl1aBuf[curr], qk[kOffset], mhSize, kBlockPad, tileSizeOfCachedKV,
                                mhBlockPad);

                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                // copy K(T) (nSize, blockSize) to L1
                CopyGmToL1Nd2Nz(aktl1bBuf[curr], kCache[block * blockSize * kvLoraRank + nOffset],
                                kBlockPad, nSize, kvLoraRank, kBlockPad);

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], aqkl1aBuf[curr], mhBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);

                CopyToL0BTCol(l0bBuf[curr], aktl1bBuf[curr], nBlockNum, 0, kBlockNum, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad Absorb (queryTokens, nSize) = QK * K（T）
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mhBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy Absorb (headTileSize, queryTokens, nSize) to L1
            CopyL0CToL1(absorbl1aBuf[nOffset * mhBlockPad], l0cBuf, mhSize, nBlockPad, mhBlockPad,
                        mhBlockPad);
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
        int nzHeadStride = queryLen * kBlockSize;
        int cubeSize = 512 / sizeof(Dtype);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int hIdx = 0; hIdx < headTileSize; hIdx++, headIdx++) {
            int nzHeadOffset = hIdx * nzHeadStride;
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
                    if (nz == 0) {
                        CopyGmToL1Nd2Nz(awuvl1bBuf[curr],
                                        wkvb[headIdx * (nopeHeadDim + vHeadDim) * kvLoraRank +
                                             (nopeHeadDim + nOffset) * kvLoraRank + kOffset],
                                        nSize, kSize, kvLoraRank, nBlockPad);
                    } else {
                        CopyGmToL1(awuvl1bBuf[curr],
                                   wkvb[headIdx * (nopeHeadDim + vHeadDim) * kBlockSize +
                                        (nopeHeadDim + nOffset) * kBlockSize +
                                        kOffset * nHeads * (nopeHeadDim + vHeadDim)],
                                   nSize, kSize / kBlockSize, nHeads * (nopeHeadDim + vHeadDim));
                    }

                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + curr);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID6 + curr);

                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    CopyToL0BCol(l0bBuf[curr], awuvl1bBuf[curr], nBlockNum, 0, kBlockNum);
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID6 + curr);
                    LoadData2dParams params(0, mBlockNum, 1, 0, kBlockNum - 1, 0, inc);
                    int absorbl1aOffset = nzHeadOffset + kOffset * mhBlockPad;
                    for (int k = 0; k < kBlockNum; k++) {
                        LoadData(l0aBuf[curr][k * cubeSize],
                                 absorbl1aBuf[absorbl1aOffset + k * mhBlockNum * cubeSize], params);
                    }

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

                CopyToGm(sv[hIdx * queryLen * vHeadDim + nOffset], l0cBuf, queryLen, nSize,
                         mBlockPad, vHeadDim);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID6);
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t softmaxConfig = 1 | (mode << 4) | (flagIdx << 8);
        flagIdx = 1;
        uint64_t updateConfig = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskLen, lastHeadIdx, last, lastKvOffset, lastKvLen,
            lastQueryTaskOffset;
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
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            int queryTileSize = MAX_M0 / headTileSize;
            if (queryTileSize == 0) {
                queryTileSize = MAX_M0;
            }
            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV);
            int taskNum = queryNum * nHeadTiles * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int headTileIdx = (idx / kvNum) % nHeadTiles;
                int headIdx = headTileIdx * headTileSize;
                int queryIdx = idx / (kvNum * nHeadTiles);
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * tileSizeOfCachedKV;
                if (calcLen <= kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = tileSizeOfCachedKV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                // do queryIdx & (headIdx，headIdx + headTileSize)'s QK
                uint32_t qOffset =
                    (queryTaskOffset * nHeads + headIdx) * (ropeHeadDim + nopeHeadDim);

                dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [%u - %u), "
                           "kv [%u - %u)} use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           headIdx, headIdx + headTileSize, kvOffset, kvOffset + kvLen, curr);
                RunAicQKAbsorb(qWithQr[qOffset], queryTaskLen, headIdx, blockTable, kvOffset, kvLen,
                               qk[curr]);
                ffts_cross_core_sync(PIPE_FIX, softmaxConfig);

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(2);
                    // do softmax * V
                    dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [%u - %u)}, "
                               "kv [%u - %u)} use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, lastHeadIdx,
                               lastHeadIdx + headTileSize, lastKvOffset, lastKvOffset + lastKvLen,
                               last);
                    RunAicSVAbsorb(qk[last], lastQueryTaskLen, lastHeadIdx, lastBlockTable,
                                   lastKvOffset, lastKvLen, sv[last]);
                    ffts_cross_core_sync(PIPE_FIX, updateConfig);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastHeadIdx = headIdx;
                lastBlockTable = blockTable;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(2);
            dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [%u - %u), "
                       "kv [%u - %u)} use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, lastHeadIdx,
                       lastHeadIdx + headTileSize, lastKvOffset, lastKvOffset + lastKvLen, last);
            RunAicSVAbsorb(qk[last], lastQueryTaskLen, lastHeadIdx, lastBlockTable, lastKvOffset,
                           lastKvLen, sv[last]);
            ffts_cross_core_sync(PIPE_FIX, updateConfig);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        uint64_t flagIdx = 2;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskLen, lastHeadIdx, last, lastKvOffset, lastKvLen,
            lastQueryTaskOffset, lastWorkStart, lastWorkCurCore, lastActualCalcSoftmaxLen,
            lastHeadSubTaskStart;
        int lastIsLastKvTile;
        uint32_t lastOutOffset;
        __gm__ uint32_t *lastBlockTable;

        int needDoUpdate = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int generation = 0;
        int resetNextCore = 0;
        int resetPrevCore = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            int queryTileSize = MAX_M0 / headTileSize;
            if (queryTileSize == 0) {
                queryTileSize = MAX_M0;
            }
            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV);
            int taskNum = queryNum * nHeadTiles * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int headTileIdx = (idx / kvNum) % nHeadTiles;
                int headIdx = headTileIdx * headTileSize;
                int queryIdx = idx / (kvNum * nHeadTiles);
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * tileSizeOfCachedKV;
                if (calcLen <= kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = tileSizeOfCachedKV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                uint32_t outOffset = queryTaskOffset * nHeads + headIdx;

                int isLastKvTile = (kvOffset + kvLen == calcLen) ? 1 : 0;

                int nWork = queryTaskLen * headTileSize;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = subBlockIdx * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                int querySubTaskStart = nWorkStart % queryTaskLen;
                int headSubTaskStart = nWorkStart / queryTaskLen;
                uint32_t calcSoftmaxLen = cachedLen + queryTaskStart + 1;
                int actualCalcSoftmaxLen = calcSoftmaxLen - kvOffset;
                if (actualCalcSoftmaxLen > kvLen) {
                    actualCalcSoftmaxLen = kvLen;
                }
                uint32_t outN = ROUND_UP(kvLen, blockSize);
                // wait aic qk done
                wait_flag_dev(0);

                dbg_printf("block%d subblock%u: {batch %d, query [%u - %u), headIdx start %u, "
                           "query x head group [%u - "
                           "%u) "
                           "kv [%u - %u)} calcSoftmaxLen %u, off %u, stride %u, outN %u, use %d "
                           "temp buf: SOFTMAX\n",
                           blockIdx, subBlockIdx, batchIdx, queryTaskOffset,
                           queryTaskOffset + queryTaskLen, headIdx + headSubTaskStart, nWorkStart,
                           nWorkStart + nWorkCurCore, kvOffset, kvOffset + kvLen,
                           actualCalcSoftmaxLen, nWorkStart, queryTaskLen, outN, curr);
                RunAivSoftmaxPingPong(
                    (__gm__ Dtype *)qk[curr][nWorkStart * tileSizeOfCachedKV].GetPhyAddr(),
                    nWorkCurCore, tileSizeOfCachedKV, actualCalcSoftmaxLen, outN, false, nWorkStart,
                    queryTaskLen, (__gm__ float *)max[curr][nWorkStart].GetPhyAddr(),
                    (__gm__ float *)sum[curr][nWorkStart].GetPhyAddr(), true, scale);
                ffts_cross_core_sync(PIPE_MTE3, config);

                if (needDoUpdate != 0) {
                    // wait aic sv done
                    wait_flag_dev(1);
                    if (lastKvOffset != 0) {
                        WaitPrevCore();
                        resetPrevCore = 1;
                    }
                    dbg_printf("block%d subblock%u: {batch %d, query [%u - %u), headIdx start %u, "
                               "query x head group [%u "
                               "- %u) "
                               "kv [%u - %u)} use %d temp buf: UPDATE\n",
                               blockIdx, subBlockIdx, lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen,
                               lastHeadIdx + lastHeadSubTaskStart, lastWorkStart,
                               lastWorkStart + lastWorkCurCore, lastKvOffset,
                               lastKvOffset + lastKvLen, last);
                    // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] &
                    // max[last]
                    RunAivSoftmaxUpdate(
                        (__gm__ Dtype *)sv[last][lastWorkStart * vHeadDim].GetPhyAddr(),
                        (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ Dtype *)output[lastOutOffset * vHeadDim].GetPhyAddr(),
                        (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                        (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(), lastWorkCurCore,
                        nHeads, vHeadDim, lastKvOffset == 0, lastActualCalcSoftmaxLen, false,
                        lastWorkStart, lastQueryTaskLen);
                    if (!lastIsLastKvTile) {
                        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        SetNextCore();
                    }
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastWorkStart = nWorkStart;
                lastWorkCurCore = nWorkCurCore;
                lastHeadIdx = headIdx;
                lastOutOffset = outOffset;
                lastBlockTable = blockTable;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                lastIsLastKvTile = isLastKvTile;
                lastActualCalcSoftmaxLen = actualCalcSoftmaxLen;
                lastHeadSubTaskStart = headSubTaskStart;
                last = curr;
                needDoUpdate = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last update
        if (needDoUpdate != 0) {
            wait_flag_dev(1);
            if (lastKvOffset != 0) {
                WaitPrevCore();
                resetPrevCore = 1;
            }
            dbg_printf("block%d subblock%u: {batch %d, query [%u - %u), headIdx start %u, query x "
                       "head group [%u - %u), "
                       "kv [%u - %u)} use %d temp buf: UPDATE\n",
                       blockIdx, subBlockIdx, lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, lastHeadIdx + lastHeadSubTaskStart,
                       lastWorkStart, lastWorkStart + lastWorkCurCore, lastKvOffset,
                       lastKvOffset + lastKvLen, last);
            // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] & max[last]
            RunAivSoftmaxUpdate((__gm__ Dtype *)sv[last][lastWorkStart * vHeadDim].GetPhyAddr(),
                                (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                                (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                                (__gm__ Dtype *)output[lastOutOffset * vHeadDim].GetPhyAddr(),
                                (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                                (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(),
                                lastWorkCurCore, nHeads, vHeadDim, lastKvOffset == 0,
                                lastActualCalcSoftmaxLen, false, lastWorkStart, lastQueryTaskLen);
            if (!lastIsLastKvTile) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                SetNextCore();
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ResetPrevCore();
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
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> sv[PINGPONG_BUF_NUM];
    GlobalTensor<float> max[PINGPONG_BUF_NUM];
    GlobalTensor<float> sum[PINGPONG_BUF_NUM];
    GlobalTensor<float> lastMax;
    GlobalTensor<float> lastSum;
    GlobalTensor<Dtype> output;
    __gm__ int32_t *setNextSync;
    __gm__ int32_t *waitPrevSync;

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;

    uint32_t nHeads;
    uint32_t ropeHeadDim;
    uint32_t nopeHeadDim;
    uint32_t nopeRopeHeadDim;
    uint32_t vHeadDim;
    uint32_t kvLoraRank;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlocks;
    uint32_t maxSeqLen;
    float scale;
    uint32_t nz;
    uint32_t tileSizeOfCachedKV;
    int blockIdx;
    int subBlockIdx;
    int nextBlockIdx;
    int prevBlockIdx;
    uint32_t setNextGeneration;
    uint32_t waitPrevGeneration;
    uint32_t headTileSize;
    uint32_t nHeadTiles;

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
    LocalTensor<Dtype> aqcrl1aBuf;                    // event 0
    LocalTensor<Dtype> awukl1bBuf[PINGPONG_BUF_NUM];  // event 6/7
    LocalTensor<Dtype> akrl1bBuf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<Dtype> akl1bBuf[PINGPONG_BUF_NUM];    // event 4/5

    // SV (absorb)
    LocalTensor<Dtype> aqkl1aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> aktl1bBuf[PINGPONG_BUF_NUM];   // event 4/5
    LocalTensor<Dtype> awuvl1bBuf[PINGPONG_BUF_NUM];  // event 6/7
};

#define FLASH_MLA_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void flash_mla_##dtype(                                      \
        GM_ADDR qWithQr, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR wkvb, GM_ADDR qk, GM_ADDR sv,    \
        GM_ADDR max, GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output, \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,        \
        uint32_t nHeads, uint32_t ropeHeadDim, uint32_t nopeHeadDim, uint32_t vHeadDim,           \
        uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlocks,           \
        float scale, uint32_t nz)                                                                 \
    {                                                                                             \
        FlashMLA<dtype> op;                                                                       \
        op.Init(qWithQr, kCache, vCache, wkvb, qk, sv, max, sum, lastMax, lastSum, sync, output,  \
                queryStartLoc, queryLens, cachedLens, blockTables, nHeads, ropeHeadDim,           \
                nopeHeadDim, vHeadDim, kvLoraRank, blockSize, batch, maxNumBlocks, scale, nz);    \
        op.Run();                                                                                 \
    }
