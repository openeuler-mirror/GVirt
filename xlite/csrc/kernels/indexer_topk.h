/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"
#include "debug.h"

#pragma once
#define INDEXER_KV_TILE_LEN 4096
#define AIV_TO_AIC 2u  // should be 2 for `KERNEL_TYPE_MIX_AIC_1_2`

static_assert(INDEXER_KV_TILE_LEN <= MAX_INDEXER_KV_TILE_LEN,
              "INDEXER_KV_TILE_LEN must not exceed MAX_INDEXER_KV_TILE_LEN");
static_assert(INDEXER_KV_TILE_LEN > MAX_TOPK_NUM, "KV tile must stay wider than MAX_TOPK_NUM");

// WeightDtype is shared by head weights and converted QK values.
template <typename Dtype, typename MatDtype, typename WeightDtype, typename ScoreDtype>
class IndexerTopK
{
public:
    __aicore__ inline IndexerTopK()
    {
    }

    // Hybrid partitioning, decided before launch by the host (auto_tuner's
    // PickIndexerKvSplit) and passed down through the `kvSplit` launch parameter:
    // `kvSplit=0` - one task = one (batch, query tile); the owning block walks all KV
    // chunks sequentially, `lastTopk` stays core-local (MTE3->MTE2 fence, no sync flags).
    // `kvSplit=1` - the sparse (tile, chunk) tasks cannot fill all blocks, so tasks split
    // to (tile, chunk) granularity again and blocks relay the running topK across cores
    // through the `sync` GM flags (software ring).
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc,
                                GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,
                                GM_ADDR scores, GM_ADDR lastTopk, GM_ADDR seqPositions,
                                GM_ADDR topkIndices, GM_ADDR sync, uint32_t kCacheStride,
                                uint32_t nHeads, uint32_t headDim, uint32_t blockSize,
                                uint32_t batch, uint32_t blockTableStride, uint32_t maxNumBlock,
                                uint32_t topK, uint8_t skipDenseTopk, GM_ADDR kScaleCache = nullptr,
                                uint32_t kScaleCacheStride = 0, uint32_t compressRatio = 1,
                                uint32_t weightCols = 0, uint8_t kvSplit = 0,
                                uint8_t qTileOverride = 0, uint8_t packTail = 0, uint8_t reluQk = 1,
                                uint32_t outStride = 0)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->q.SetGlobalBuffer((__gm__ Dtype *)q);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->weight.SetGlobalBuffer((__gm__ WeightDtype *)weight);
        this->kScaleCache = (__gm__ half *)kScaleCache;
        // Packed pages: scale pool's own block stride; dense (stride 0/blockSize)
        // → flat addressing via MAX.
        this->kScaleCacheStride = MAX(kScaleCacheStride, blockSize);
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;
        this->lastTopk = (__gm__ uint32_t *)lastTopk;
        this->seqPositions = (__gm__ uint32_t *)seqPositions;
        this->topkIndices = (__gm__ uint32_t *)topkIndices;
        this->nHeads = nHeads;
        this->headDim = headDim;
        // weight row pitch; head weights tail-align at (weightCols - nHeads), so guard
        // against uint32 underflow for callers that bypass host validation.
        assert(weightCols >= nHeads);
        this->weightCols = weightCols;
        this->blockSize = blockSize;
        this->kCacheStride = MAX(kCacheStride, blockSize * headDim);
        this->batch = batch;
        this->blockTableStride = blockTableStride;
        this->maxNumBlock = maxNumBlock;
        this->topK = topK;
        // 0 = compile-time default; clamped to the macro because L1 staging
        // (ql1Buf/wl1Buf) is macro-sized at build time.
        this->queryTileSize =
            qTileOverride > 0 ? MIN(qTileOverride, MAX_INDEXER_Q_TILE_LEN) : MAX_INDEXER_Q_TILE_LEN;
        this->skipDenseTopk = skipDenseTopk > 0;
        this->compressRatio = compressRatio > 0 ? compressRatio : 1;
        // packTail expands each selected pool index into kpool token indices + causal tail.
        this->packTail = packTail > 0;
        this->kpool = this->packTail ? MAX(this->compressRatio, 2u) : 0;
        this->outWidth = this->packTail ? this->topK * this->kpool + this->kpool - 1 : this->topK;
        // outStride (host-owned) pads rows to TOPK_RUN_ALIGN slots so every row base
        // is 512B-aligned and the AIV softmax sees a full number of 128-slot blocks.
        this->outStride = outStride;
        // glm5 indexer scores need no per-head ReLU; upstream fusion applies one.
        this->reluQk = reluQk > 0;
        this->tileSizeOfCachedKV = ROUND_UP(INDEXER_KV_TILE_LEN, SORT_BLOCK_SIZE);
        this->blockIdx = block_idx;
        this->subBlockIdx = get_subblockid();
        this->nextBlockIdx = (blockIdx + 1) % block_num;
        this->prevBlockIdx = blockIdx == 0 ? (block_num - 1) : (blockIdx - 1);
        this->setNextGeneration = 1;
        this->waitPrevGeneration = 1;
        this->resetPrevCore = 0;

        // Per-block ping-pong score slots are qTile rows wide (the FIX stage
        // writes one run of scores per query row); match the host allocation.
        this->scores[0].SetGlobalBuffer(((__gm__ ScoreDtype *)scores) +
                                        block_idx * MAX_INDEXER_Q_TILE_LEN * tileSizeOfCachedKV);
        this->scores[1].SetGlobalBuffer(((__gm__ ScoreDtype *)scores) +
                                        block_idx * MAX_INDEXER_Q_TILE_LEN * tileSizeOfCachedKV +
                                        block_num * MAX_INDEXER_Q_TILE_LEN * tileSizeOfCachedKV);
        this->setNextSync = (__gm__ int32_t *)sync + blockIdx * AIV_TO_AIC + subBlockIdx;
        this->waitPrevSync = (__gm__ int32_t *)sync + prevBlockIdx * AIV_TO_AIC + subBlockIdx;
        this->kvSplit = kvSplit > 0;

#ifdef __DAV_C220_CUBE__
        /*
         * scores = k * q
         *     m: cachedTokens, n: queryLen * nHeads, k: headDim
         *     m0: blockSize, n0: nHeads, k0: headDim
         * index_scores = weights * scores
         *     m: queryLen, n: cachedTokens, k: nHeads
         *     m0: 16, n0: blockSize, k0: nHeads
         */
        // Consider L0C capacity, limit heads and blockSize
        assert(nHeads <= 64 && blockSize <= 128);
        assert(headDim <= 256 / sizeof(Dtype));
        constexpr uint32_t wsm0 = MBLOCKSIZE;
        uint64_t off = 0;
        uint64_t kl1Size = blockSize * headDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kl1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kl1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kl1Size;
        }

        uint64_t kql1Size = blockSize * nHeads * sizeof(WeightDtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kql1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kql1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kql1Size;
        }

        uint64_t ql1Size = MAX_INDEXER_Q_TILE_LEN * nHeads * headDim * sizeof(Dtype);
        ql1Buf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        ql1Buf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += ql1Size;

        uint64_t wl1Size =
            ROUND_UP(MAX_INDEXER_Q_TILE_LEN, MBLOCKSIZE) * nHeads * sizeof(WeightDtype);
        wl1Buf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        wl1Buf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += wl1Size;

        off = 0;
        // QK and weighted scores reuse the same L0A/L0B storage in sequential phases.
        uint64_t kql0aSize = blockSize * headDim * sizeof(Dtype);
        l0aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += kql0aSize;

        uint64_t wsl0aSize = wsm0 * nHeads * sizeof(WeightDtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            wl0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            wl0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += wsl0aSize;
        }

        off = 0;
        uint64_t kql0bSize = nHeads * headDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kql0bSize;
        }

        uint64_t wsl0bSize = blockSize * nHeads * sizeof(WeightDtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kql0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            kql0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += wsl0bSize;
        }

        off = 0;
        uint64_t qkl0cSize = blockSize * nHeads * sizeof(MatDtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl0cBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
            qkl0cBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qkl0cSize;
        }
        uint64_t scoresSize = wsm0 * blockSize * sizeof(float);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0cBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
            l0cBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += scoresSize;
        }
#endif
#ifdef __DAV_C220_VEC__
        uint64_t maxSeqLen = MAX((uint64_t)blockSize * maxNumBlock, INIT_MIN_SEQ_POS);
        this->sortIndicesStart = (blockIdx * AIV_TO_AIC + subBlockIdx) %
                                 MAX(1, maxSeqLen / tileSizeOfCachedKV) * tileSizeOfCachedKV;
        this->sortIndicesBytes = MIN(tileSizeOfCachedKV, maxSeqLen) * sizeof(uint32_t);
        this->sortIndicesRepeats = DIV_ROUND_UP(this->sortIndicesBytes, VECTOR_MAX_BYTESIZE);
        // totalSort use
        uint64_t off = 0;
        this->totalSort = reinterpret_cast<__ubuf__ float *>(off);
        // MrgSort's dst alternates mrgSortBuf0/1; totalSort only receives
        // isFirst copies, chunk merges and the final vmuls (4*MAX_TOPK_NUM floats).
        off += ROUND_UP(MAX_TOPK_NUM * 4 * sizeof(float) + UB_BANK_CONFLICT_OFFSET,
                        UB_BANKGROUP_ROW_SIZE) -
               UB_BANK_CONFLICT_OFFSET;
        // see `mrgSortBuf0`; separating the two mrgSort buffers to avoid bank conflict in A2/A3
        this->mrgSortBuf1 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        this->sortIndices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        // packTail rows are wider than MAX_TOPK_NUM (topK*kpool + kpool-1) and are
        // copied at the 128-slot-aligned outStride, so the template must cover the
        // full stride, not just outWidth.
        this->defaultTopkIndices = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(outStride * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);

        // in
        this->in[0] = reinterpret_cast<__ubuf__ WeightDtype *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(WeightDtype), VECTOR_MAX_BYTESIZE);
        this->in[1] = reinterpret_cast<__ubuf__ WeightDtype *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(WeightDtype), VECTOR_MAX_BYTESIZE);
        this->lastSort[0] = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        this->lastSort[1] = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);

        // out
        // Non-packTail rows stay MAX_TOPK_NUM-wide (the legacy output buffer contract).
        uint64_t outElems = MAX((uint64_t)outStride, (uint64_t)MAX_TOPK_NUM);
        this->out[0] = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(outElems * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        this->out[1] = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(outElems * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);

        // calc
        uint64_t invoff =
            UB_SIZE - ROUND_UP(MAX_INDEXER_KV_TILE_LEN * 2 * sizeof(float), UB_BANKGROUP_ROW_SIZE);
        this->mrgSortBuf0 = reinterpret_cast<__ubuf__ float *>(invoff);
        assert(off <= invoff);
#endif
    }

    // ---- cross-core relay (kvSplit=1 only): software ring over the `sync` GM flags ----
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

#ifdef __DAV_C220_CUBE__
    __aicore__ inline void RunAicIndexerScores(int queryLen, __gm__ uint32_t *blockTable,
                                               int kvOffset, int kvLen,
                                               GlobalTensor<ScoreDtype> scores)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mIdxStart = kvOffset / blockSize;
        int mLoop = DIV_ROUND_UP(kvLen, blockSize);
        int mSize = blockSize;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = queryLen * nHeads;
        int nTotalBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
        int nBlockPad = ROUND_UP(nHeads, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int kBlockNum = headDim / kBlockSize;

        int wmSize = queryLen;
        int wmBlockPad = ROUND_UP(wmSize, MBLOCKSIZE);
        int wnSize = blockSize;
        int wnBlockPad = ROUND_UP(wnSize, NBLOCKSIZE);
        int wnBlockNum = wnBlockPad / NBLOCKSIZE;
        constexpr int weightBlockSize = 32 / sizeof(WeightDtype);
        int wkBlockNum = nHeads / weightBlockSize;

        int kPingpong = 0;
        int curr = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID2);
        SetFlag<HardEvent::FIX_M>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID3);
        // do queryTileSize's QK^T * weight
        for (int mIdx = 0; mIdx < mLoop; mIdx++) {  // kvLen
            int mOffset = mIdx * blockSize;
            if (mOffset + mSize > kvLen) {
                mSize = kvLen - mOffset;
                mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
                mBlockNum = mBlockPad / MBLOCKSIZE;
                wnSize = mSize;
                wnBlockPad = ROUND_UP(wnSize, NBLOCKSIZE);
                wnBlockNum = wnBlockPad / NBLOCKSIZE;
            }

            // copy k (blockSize, headDim) to L1
            uint32_t block = blockTable[mIdx + mIdxStart];
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + kPingpong);
            CopyGmToL1Nd2Nz(kl1Buf[kPingpong], kCache[(uint64_t)block * kCacheStride], mSize,
                            headDim, headDim, mBlockPad);

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + kPingpong);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + kPingpong);

            // copy k (blockSize, headDim) to L0A
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
            CopyToL0ACol(l0aBuf, kl1Buf[kPingpong], mBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + kPingpong);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID4);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID4);

            if (queryLen != 0) {
                // copy q (heads, headDim) to L0B
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0BCol(l0bBuf[curr], ql1Buf, nBlockNum, 0, kBlockNum, nTotalBlockPad);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // copy weight (1, nHeads) to L0A
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + curr);
                CopyToL0ACol(wl0aBuf[curr], wl1Buf, 1, 0, wkBlockNum, wmBlockPad);

                // mmad qk (blockSize, nHeads)
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0 + curr);
                CalMmad(qkl0cBuf[curr], l0aBuf, l0bBuf[curr], mBlockPad, nBlockPad, headDim, true);
                if (mBlockNum * nBlockNum < 10) {
                    PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);

                SetFlag<HardEvent::M_FIX>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0 + curr);

                // copy qk (blockSize, nHeads) from L0C to L1 with ReLU filter.
                WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID2 + curr);
                if constexpr (std::is_same<Dtype, int8_t>::value) {
                    // LI rounding: FP16(relu(INT32 dot) / 1024).
                    CopyL0CToL1(kql1Buf[curr], qkl0cBuf[curr], mBlockPad, nBlockPad, mBlockPad,
                                mBlockPad, 1.0f / 1024.0f, this->reluQk);
                } else {
                    CopyL0CToL1(kql1Buf[curr], qkl0cBuf[curr], mBlockPad, nBlockPad, mBlockPad,
                                mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE,
                                /*reluEn=*/this->reluQk);
                }
                SetFlag<HardEvent::FIX_M>(EVENT_ID0 + curr);

                SetFlag<HardEvent::FIX_MTE1>(EVENT_ID2 + curr);
            }
            for (int q = 0; q < queryLen; q++) {
                if (q + 1 < queryLen) {
                    int next = 1 - curr;
                    // copy q (heads, headDim) to L0B
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + next);
                    CopyToL0BCol(l0bBuf[next], ql1Buf[(q + 1) * nHeads * kBlockSize], nBlockNum, 0,
                                 kBlockNum, nTotalBlockPad);
                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + next);

                    // copy weight (1, nHeads) to L0A
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + next);
                    CopyToL0ACol(wl0aBuf[next], wl1Buf[(q + 1) * weightBlockSize], 1, 0, wkBlockNum,
                                 wmBlockPad);

                    // mmad qk (blockSize, nHeads)
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + next);
                    WaitFlag<HardEvent::FIX_M>(EVENT_ID0 + next);
                    CalMmad(qkl0cBuf[next], l0aBuf, l0bBuf[next], mBlockPad, nBlockPad, headDim,
                            true);
                    if (mBlockNum * nBlockNum < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + next);

                    SetFlag<HardEvent::M_FIX>(EVENT_ID0 + next);
                    WaitFlag<HardEvent::M_FIX>(EVENT_ID0 + next);

                    // copy qk (blockSize, nHeads) from L0C to L1 with ReLU filter.
                    WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID2 + next);
                    if constexpr (std::is_same<Dtype, int8_t>::value) {
                        // LI rounding: FP16(relu(INT32 dot) / 1024).
                        CopyL0CToL1(kql1Buf[next], qkl0cBuf[next], mBlockPad, nBlockPad, mBlockPad,
                                    mBlockPad, 1.0f / 1024.0f, this->reluQk);
                    } else {
                        CopyL0CToL1(kql1Buf[next], qkl0cBuf[next], mBlockPad, nBlockPad, mBlockPad,
                                    mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE,
                                    /*reluEn=*/this->reluQk);
                    }
                    SetFlag<HardEvent::FIX_M>(EVENT_ID0 + next);

                    SetFlag<HardEvent::FIX_MTE1>(EVENT_ID2 + next);
                }

                WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID2 + curr);

                // copy qk (blockSize, nHeads) to L0B
                CopyToL0BCol(kql0bBuf[curr], kql1Buf[curr], wnBlockNum, 0, wkBlockNum);
                SetFlag<HardEvent::MTE1_FIX>(EVENT_ID2 + curr);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID2 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID2 + curr);

                // mmad scores (1, blockSize)
                WaitFlag<HardEvent::FIX_M>(EVENT_ID2 + curr);
                CalMmad(l0cBuf[curr], wl0aBuf[curr], kql0bBuf[curr], MBLOCKSIZE, wnBlockPad, nHeads,
                        true);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID2 + curr);
                if (wnBlockNum < 10) {
                    PipeBarrier<PIPE_M>();
                }

                SetFlag<HardEvent::M_FIX>(EVENT_ID2 + curr);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID2 + curr);

                // copy scores (1, blockSize) from L0C to GM
                CopyToGm(scores[q * tileSizeOfCachedKV + mOffset], l0cBuf[curr], 1, mSize,
                         MBLOCKSIZE, tileSizeOfCachedKV);
                SetFlag<HardEvent::FIX_M>(EVENT_ID2 + curr);
                curr = 1 - curr;
            }
            SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
            kPingpong = 1 - kPingpong;
        }
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
#endif

#ifdef __DAV_C220_VEC__
    __aicore__ inline void RunAivTopk(__gm__ ScoreDtype *scores, __gm__ uint32_t *lastTopk,
                                      int queryLen, int kvOffset, int kvLen, uint32_t topK,
                                      __gm__ uint32_t *topkIndices, int queryPosBase,
                                      __gm__ uint32_t *blockTable)
    {
        assert(kvLen <= MAX_INDEXER_KV_TILE_LEN && topK <= MAX_TOPK_NUM);
        constexpr float min = FLOAT_MIN;

        constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
        int topKSortRepeat = DIV_ROUND_UP(topK, SORT_BLOCK_SIZE);
        uint32_t topKCalcRepeat = DIV_ROUND_UP(topK, calcPad);
        uint32_t topKCalcRepeatX2 = topKCalcRepeat * 2;

        int curr = 0;
        bool totalSortOnHold = false;
        bool lastChunkFenceWaited = false;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // release `in[curr]`
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);  // release `lastSort[curr]`
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // release `out[curr]`
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        for (int idx = 0; idx < queryLen; idx++) {
            // p0 = KV-space causal boundary (token when ratio==1, compressed block when >1).
            int p0 = MapPosToKv(queryPosBase + idx);
            int validKvLen = MIN(p0 - kvOffset + 1, kvLen);  // per position valid kvLen
            if (p0 < (int32_t)topK ||
                validKvLen <= 0) {  // either dense, or the last chunk was final
                continue;
            }
            bool isFirst = kvOffset <= 0;
            bool isFinal = kvOffset + validKvLen > p0;
            int repeat = DIV_ROUND_UP(validKvLen, calcPad);
            int fullRepeat = validKvLen / calcPad;
            // vbitsort drops the index payload at repeat 1; keep repeat >= 2.
            int sortRepeat = DIV_ROUND_UP(validKvLen, SORT_BLOCK_SIZE);
            if (sortRepeat == 1) {
                sortRepeat = 2;
            }
            pipe_barrier(PIPE_V);

            // prepare sortIndices for the current query position
            int32_t idxDiff = kvOffset - sortIndicesStart;
            if (idxDiff != 0) {
                sortIndicesStart = kvOffset;
                vadds((__ubuf__ int32_t *)sortIndices, (__ubuf__ int32_t *)sortIndices, idxDiff,
                      sortIndicesRepeats, 1, 1, 8, 8);
            }

            // pad the tail of `mrgSortBuf0` (incoming scores) with `FLOAT_MIN`
            int sortLen = sortRepeat * SORT_BLOCK_SIZE;
            if (fullRepeat < repeat || sortLen > repeat * calcPad) {
                uint16_t repeatNum = sortLen / calcPad - fullRepeat + 1;
                vector_dup(mrgSortBuf0 + fullRepeat * calcPad, float(min), repeatNum, 1, 1, 8, 1);
                pipe_barrier(PIPE_V);
            }

            // copy scores to in
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // acquire `in[curr]`
            if constexpr (std::is_same<Dtype, int8_t>::value) {
                for (int start = 0; start < validKvLen; start += blockSize) {
                    int count = MIN(int(blockSize), validKvLen - start);
                    uint32_t block = blockTable[(kvOffset + start) / blockSize];
                    CopyGmToUbufAligned(in[curr] + start, kScaleCache + block * kScaleCacheStride,
                                        count * sizeof(half));
                }
                // in[curr] is released before sorting; mrgSortBuf0 needs its own fence.
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
                CopyGmToUbufAligned(mrgSortBuf0, scores + idx * tileSizeOfCachedKV,
                                    validKvLen * sizeof(float));
            } else {
                CopyGmToUbufAligned(in[curr], scores + idx * tileSizeOfCachedKV,
                                    validKvLen * sizeof(Dtype));
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            // Prepare FP32 scores.
            if constexpr (std::is_same<Dtype, int8_t>::value) {
                vconv_f162f32(mrgSortBuf1, in[curr], repeat, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                vmul(mrgSortBuf0, mrgSortBuf0, mrgSortBuf1, repeat, 1, 1, 1, 8, 8, 8);
                // Restore the sort padding after DMA and rescaling.
                int remain = validKvLen % calcPad;
                if (remain != 0) {
                    pipe_barrier(PIPE_V);
                    SetMaskFromHighBit(calcPad, calcPad - remain);
                    vector_dup(mrgSortBuf0 + fullRepeat * calcPad, float(min), 1, 1, 1, 8, 0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
            } else {
                convert_input(mrgSortBuf0, in[curr], fullRepeat, validKvLen % calcPad);
            }
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // release `in[curr]`

            // sort local
            pipe_barrier(PIPE_V);
            vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices, sortRepeat);
            pipe_barrier(PIPE_V);

            // sort local & last
            uint64_t totalSortLen = MIN(topK, validKvLen);
            __ubuf__ float *localSort;
            if (isFirst) {
                if (totalSortOnHold) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);  // acquire `totalSort`
                    totalSortOnHold = false;
                }
                MrgSort(mrgSortBuf1, mrgSortBuf0, sortRepeat, &localSort, topK, totalSort);
                // MrgSort can still return sort0 for a single repeat; downstream
                // unconditionally reads totalSort, so keep the fallback for that path.
                if (localSort != totalSort) {
                    copy_ubuf_to_ubuf(totalSort, localSort, 0, 1,
                                      DIV_ROUND_UP(topK * 2 * sizeof(float), BLOCK_SIZE), 0, 0);
                    pipe_barrier(PIPE_V);
                }
            } else {
                MrgSort(mrgSortBuf1, mrgSortBuf0, sortRepeat, &localSort, topK);
                pipe_barrier(PIPE_V);
                if (!lastChunkFenceWaited) {
                    if (kvSplit) {
                        WaitPrevCore();
                        resetPrevCore = 1;
                    } else {
                        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
                    }
                    lastChunkFenceWaited = true;
                }
                // copy last intermediate sort results (score + index) to `lastSort[curr]`
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);  // acquire `lastSort[curr]`
                uint64_t lastSortLen = MIN(topK, kvOffset);
                CopyGmToUbufAligned(lastSort[curr], lastTopk + idx * 2 * topK,
                                    lastSortLen * 2 * sizeof(uint32_t));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
                __ubuf__ float *addrs[4] = {localSort, lastSort[curr]};
                if (totalSortOnHold) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);  // acquire `totalSort`
                    totalSortOnHold = false;
                }
                vmrgsort4(totalSort, addrs, totalSortLen | (lastSortLen << 16),
                          1ull | (0x3ull << MGR_SORT_VALID_BITS_OFFSET));
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);  // release `lastSort[curr]`
                totalSortLen = MIN(topK, totalSortLen + lastSortLen);
            }
            pipe_barrier(PIPE_V);

            // `totalSort`: merged topK results (score + index) for the current query position
            if (!isFinal) {
                // copy `totalSort` (intermediate score+index results) to GM; the next chunk
                // call of the same tile (same core) merges it back via `lastSort[curr]`
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                totalSortOnHold = true;
                CopyUbufToGmAligned(lastTopk + idx * topK * 2, totalSort,
                                    MIN(topK, totalSortLen) * 2 * sizeof(uint32_t));
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);  // release `totalSort`
                if (idx == queryLen - 1) {
                    if (kvSplit) {
                        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        SetNextCore();
                    } else {
                        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
                    }
                }
            } else {
                // aggregate topK indices from totalSort, sort them from smallest to largest
                __ubuf__ uint32_t *index0 = (__ubuf__ uint32_t *)mrgSortBuf0;
                vreducev2(index0, (__ubuf__ uint32_t *)totalSort, nullptr, topKCalcRepeatX2, 1, 2,
                          8, 0);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // acquire `out[curr]`
                if (topKSortRepeat <= 1) {
                    // topK <= 32: vbitsort's tag lane is unreliable at repeat 1;
                    // MrgSort just aliases through, so use insertion sort.
                    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    for (uint32_t a = 1; a < topK; a++) {
                        uint32_t v = index0[a];
                        int32_t b = a - 1;
                        while (b >= 0 && index0[b] > v) {
                            index0[b + 1] = index0[b];
                            b--;
                        }
                        index0[b + 1] = v;
                    }
                    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                    copy_ubuf_to_ubuf(out[curr], index0, 0, 1, topK * sizeof(uint32_t), 0, 0);
                } else {
                    // mrgSortBuf0: top k indices; totalSort: negative of the index values as scores
                    vmuls(totalSort, (__ubuf__ float *)index0, -1.0f, topKCalcRepeat, 1, 1, 8, 8);
                    // Score MrgSort returns ROUND_UP(topK, 32) entries; sink the
                    // extras so the index sort keeps the score-selected topK.
                    uint32_t topKAligned = topKSortRepeat * SORT_BLOCK_SIZE;
                    if (topK < topKAligned) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        for (uint32_t j = topK; j < topKAligned; j++) {
                            totalSort[j] = min;
                        }
                        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                    }
                    pipe_barrier(PIPE_V);
                    vbitsort(mrgSortBuf1, totalSort, index0, topKSortRepeat);
                    pipe_barrier(PIPE_V);
                    __ubuf__ float *indexSorted;
                    MrgSort(mrgSortBuf1, mrgSortBuf0, topKSortRepeat, &indexSorted, topK);
                    pipe_barrier(PIPE_V);
                    vreducev2(out[curr], (__ubuf__ uint32_t *)indexSorted, nullptr,
                              topKCalcRepeatX2, 1, 2, 8, 0);
                }
                pipe_barrier(PIPE_V);
                if (packTail) {
                    // Expand block indices to token indices in place, back to front so
                    // source slot c/kpool is still untouched when slot c is written.
                    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    for (int32_t c = (int32_t)topK * (int32_t)kpool - 1; c >= 0; c--) {
                        out[curr][c] = out[curr][c / kpool] * kpool + (uint32_t)(c % kpool);
                    }
                    for (int32_t toff = 0; toff < (int32_t)kpool - 1; toff++) {
                        out[curr][topK * kpool + toff] = 0xFFFFFFFFu;
                    }
                    BuildTail(out[curr], queryPosBase + idx);
                    // Re-fill the 128-slot row padding per row; one-time init is
                    // overwritten by later aligned vector writes.
                    for (uint32_t c = outWidth; c < outStride; c++) {
                        out[curr][c] = 0xFFFFFFFFu;
                    }
                    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                }
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
                CopyUbufToGmAligned(topkIndices + idx * outStride, out[curr],
                                    outStride * sizeof(uint32_t));
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);  // release `out[curr]`
            }
            curr = 1 - curr;
        }
        if (totalSortOnHold) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
#endif

    // Map token pos P to the KV-space causal boundary p0 (inclusive last valid index).
    //   ratio==1: includes own position -> p0 = P.
    //   ratio>1: excludes own incomplete block -> p0 = (P+1)/ratio - 1, so a token whose
    //   block is incomplete (e.g. first block, clen==0) yields -1 ("no valid KV").
    //   Compare against a SIGNED topK: -1 < uint32_t(topK) promotes to a huge value and
    //   wrongly reads as "not dense", skipping the template write and leaving rows unset.
    __aicore__ inline int32_t MapPosToKv(int32_t pos) const
    {
        if (compressRatio <= 1) {
            return pos;
        }
        return (pos + 1) / static_cast<int32_t>(compressRatio) - 1;
    }

#ifdef __DAV_C220_VEC__
    // packTail row epilogue: patch tail lanes into the template UB so each row
    // goes out in one MTE3 copy; scalar S-pipe GM stores race the next row copy
    // even across set/wait fences.
    __aicore__ inline void BuildTail(__ubuf__ uint32_t *tmpl, int32_t pos)
    {
        int32_t tailStart = ((pos + 1) / (int32_t)kpool) * (int32_t)kpool;
        int32_t tailCol = MIN(tailStart, (int32_t)(topK * kpool));
        int32_t tailCount = pos + 1 - tailStart;
        for (int32_t toff = 0; toff < (int32_t)kpool - 1; toff++) {
            tmpl[tailCol + toff] = toff < tailCount ? (uint32_t)(tailStart + toff) : 0xFFFFFFFFu;
        }
        lastTailCol = tailCol;
    }

    // Restore the template lanes clobbered by the previous BuildTail.
    __aicore__ inline void RestorePrevTail(__ubuf__ uint32_t *tmpl)
    {
        if (lastTailCol < 0) {
            return;
        }
        for (int32_t toff = 0; toff < (int32_t)kpool - 1; toff++) {
            uint32_t col = (uint32_t)(lastTailCol + toff);
            tmpl[col] = col < topK * kpool ? col : 0xFFFFFFFFu;
        }
        lastTailCol = -1;
    }

    // Dense default-topk rows from the identity template (in-row fence pairs only).
    __aicore__ inline void StoreDefaultTopkRows(__gm__ uint32_t *dst, int queryPosBase,
                                                int queryTaskLen)
    {
        __ubuf__ uint32_t *tmpl = this->defaultTopkIndices;
        for (int i = 0; i < queryTaskLen; i++) {
            if (MapPosToKv(queryPosBase + i) >= (int32_t)topK) {
                break;
            }
            if (packTail) {
                RestorePrevTail(tmpl);
                BuildTail(tmpl, queryPosBase + i);
                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            }
            CopyUbufToGmAligned(dst + i * outStride, tmpl, outStride * sizeof(uint32_t));
            if (packTail) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            }
        }
    }
#endif

    // Execute one (query tile, KV chunk) work item on this block: AIC computes one
    // chunk's scores and notifies the block's AIVs via the ffts ping-pong; each AIV
    // subblock merges its half of the query rows. Shared by both partitioning modes.
    __aicore__ inline void RunTaskChunk(int batchIdx, int queryStart, int queryOffset,
                                        int queryTaskOffset, int queryTaskLen, int queryPosBase,
                                        __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                        int curr, const uint64_t *a2vSyncFlag,
                                        const uint64_t *v2aSyncFlag)
    {
#ifdef __DAV_C220_CUBE__
        (void)queryPosBase;
        wait_flag_dev(v2aSyncFlag[curr]);
        // kvSplit deals a tile's chunks to different blocks, so every work item reloads
        // q/w into L1; the serial mode loads them once per tile, at its first chunk.
        if (kvSplit || kvOffset == 0) {
            uint32_t qOffset = queryTaskOffset * nHeads * headDim;
            // head weights tail-aligned at (weightCols - nHeads).
            uint32_t wOffset = queryTaskOffset * weightCols + (weightCols - nHeads);
            int nSize = queryTaskLen * nHeads;
            int nTotalBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
            int wmSize = queryTaskLen;
            int wmBlockPad = ROUND_UP(wmSize, MBLOCKSIZE);

            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
            // copy q (queryTaskLen, nHeads, headDim) to L1
            CopyGmToL1Nd2Nz(ql1Buf, q[qOffset], nSize, headDim, headDim, nTotalBlockPad);
            // copy weight [queryTaskLen, nHeads] to L1; weightCols is the GM row pitch.
            CopyGmToL1Nd2Nz(wl1Buf, weight[wOffset], wmSize, nHeads, weightCols, wmBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
        }
        dbg_printf("block%d: {batch %d, query start loc %u, query [%u - %u), index k "
                   "[%u - %u)} use %d buf\n",
                   blockIdx, batchIdx, queryStart, queryOffset, queryOffset + queryTaskLen,
                   kvOffset, kvOffset + kvLen, curr);
        RunAicIndexerScores(queryTaskLen, blockTable, kvOffset, kvLen, scores[curr]);
        ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr]);
#else
        int nWork = queryTaskLen;
        int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
        int nWorkCurCore = nWorkPerCore;
        int nWorkStart = subBlockIdx * nWorkCurCore;
        if (nWorkStart + nWorkCurCore > nWork) {
            nWorkCurCore = nWork - nWorkStart;
        }
        // lastTopk rows are 2*topK wide, but packed output rows use outStride
        // (ROUND_UP(topK*kpool+kpool-1, 128) with packTail); the two strides differ.
        uint32_t lastTopkRow = queryTaskOffset + nWorkStart;
        wait_flag_dev(a2vSyncFlag[curr]);
        dbg_printf("block%d subblock%u: {batch %d, query start loc %u, query [%u - "
                   "%u), index k [%u - %u)} use %d buf\n",
                   blockIdx, subBlockIdx, batchIdx, queryStart, queryOffset + nWorkStart,
                   queryOffset + nWorkStart + nWorkCurCore, kvOffset, kvOffset + kvLen, curr);
        RunAivTopk((__gm__ ScoreDtype *)scores[curr][nWorkStart * tileSizeOfCachedKV].GetPhyAddr(),
                   lastTopk + lastTopkRow * 2 * topK, nWorkCurCore, kvOffset, kvLen, topK,
                   topkIndices + lastTopkRow * outStride, queryPosBase + nWorkStart, blockTable);
        ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[curr]);
#endif
    }

    __aicore__ inline void Run()
    {
        uint64_t flagIdx0 = 0;
        uint64_t flagIdx1 = 1;
        uint64_t flagIdx2 = 2;
        uint64_t flagIdx3 = 3;
        uint64_t mode = 2;  // inner-group aic/aiv sync
#ifdef __DAV_C220_CUBE__
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);
        uint64_t sync0 = 1 | (mode << 4) | (flagIdx0 << 8);
        uint64_t sync1 = 1 | (mode << 4) | (flagIdx1 << 8);
        uint64_t a2vSyncFlag[PINGPONG_BUF_NUM] = {sync0, sync1};
        uint64_t v2aSyncFlag[PINGPONG_BUF_NUM] = {flagIdx2, flagIdx3};
#elif __DAV_C220_VEC__
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t sync2 = 1 | (mode << 4) | (flagIdx2 << 8);
        uint64_t sync3 = 1 | (mode << 4) | (flagIdx3 << 8);
        uint64_t a2vSyncFlag[PINGPONG_BUF_NUM] = {flagIdx0, flagIdx1};
        uint64_t v2aSyncFlag[PINGPONG_BUF_NUM] = {sync2, sync3};

        bool assignedDefaultTopk =
            !skipDenseTopk && blockIdx == block_num - 1 && subBlockIdx == AIV_TO_AIC - 1;
        uint64_t defaultTopkBytes = outStride * sizeof(uint32_t);
        if (assignedDefaultTopk) {
            CopyGmToUbufAligned(this->defaultTopkIndices, this->seqPositions, defaultTopkBytes);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // The template is patched by PIPE_S scalar writes below and in BuildTail;
            // wait for the MTE2 load to land before those writes, otherwise the load
            // can overwrite the first dense row's tail patch.
            if (packTail) {
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                // Cover the causal-tail placeholder and the 128-slot row padding;
                // BuildTail only patches the kpool-1 tail lanes.
                for (uint32_t c = topK * kpool; c < outStride; c++) {
                    this->defaultTopkIndices[c] = 0xFFFFFFFFu;
                }
            }
        }
        CopyGmToUbufAligned(this->sortIndices, this->seqPositions + this->sortIndicesStart,
                            this->sortIndicesBytes);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
#endif

        int queryTileSize = this->queryTileSize;

        int totalIdx = 0;
        int curr = 0;
#ifdef __DAV_C220_VEC__
        ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[0]);
        ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[1]);
#endif
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {  // each loop handles one sequence
            int queryLen = queryLens[batchIdx];
            int cachedLen = cachedLens[batchIdx];
            int queryStart = queryStartLoc[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * blockTableStride * sizeof(uint32_t));

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            if (kvSplit) {
                // Chunk-level tasks: (query tile, KV chunk) pairs are dealt round-robin so
                // idle blocks join in; consecutive chunks chain through the cross-core
                // relay (WaitPrevCore/SetNextCore in RunAivTopk).
                // nKv = completed KV blocks visible to the last query token; kCache holds
                // one row per block (compressed-block space when ratio>1, token space
                // when ==1).
                int nKv = MapPosToKv(cachedLen + queryLen - 1) + 1;
                int kvNumMax = DIV_ROUND_UP(nKv, tileSizeOfCachedKV);
                int taskNum = queryNum * kvNumMax;
                for (int idx = 0; idx < taskNum;) {
                    int queryIdx = idx / kvNumMax;
                    int queryTaskLen = queryTileSize;
                    int queryOffset = queryIdx * queryTileSize;
                    if (queryOffset + queryTaskLen > queryLen) {
                        queryTaskLen = queryLen - queryOffset;
                    }
                    int queryTaskOffset = queryStart + queryOffset;
                    int queryPosBase = cachedLen + queryOffset;
                    int queryPosEnd = queryPosBase + queryTaskLen - 1;  // inclusive (token space)
                    int queryPosEndKvBound = MapPosToKv(queryPosEnd);
                    int kvIdx = idx % kvNumMax;
                    int kvOffset = kvIdx * tileSizeOfCachedKV;
#ifdef __DAV_C220_VEC__
                    if (assignedDefaultTopk && kvOffset == 0 &&
                        MapPosToKv(queryPosBase) < (int32_t)topK) {
                        StoreDefaultTopkRows(this->topkIndices + queryTaskOffset * outStride,
                                             queryPosBase, queryTaskLen);
                    }
#endif
                    // skip dense tiles entirely or chunks past the tile's causal bound
                    if (queryPosEndKvBound < (int32_t)topK || queryPosEndKvBound < kvOffset) {
                        idx = NEXT_MULTIPLE(idx, kvNumMax);  // jump to the next query tile
                        continue;
                    }
                    int kvLen = MIN(tileSizeOfCachedKV, queryPosEndKvBound - kvOffset + 1);
                    idx++;
                    totalIdx++;  // only increase the global task index when the task is valid
                    if (totalIdx % block_num != block_idx) {
                        continue;
                    }
                    RunTaskChunk(batchIdx, queryStart, queryOffset, queryTaskOffset, queryTaskLen,
                                 queryPosBase, blockTable, kvOffset, kvLen, curr, a2vSyncFlag,
                                 v2aSyncFlag);
                    curr = 1 - curr;
                }
            } else {
                for (int queryIdx = 0; queryIdx < queryNum; queryIdx++) {
                    // each loop handles one (batch, query tile) task
                    int queryTaskLen = queryTileSize;
                    int queryOffset = queryIdx * queryTileSize;
                    if (queryOffset + queryTaskLen > queryLen) {
                        queryTaskLen = queryLen - queryOffset;
                    }
                    int queryTaskOffset = queryStart + queryOffset;
                    int queryPosBase = cachedLen + queryOffset;
                    int queryPosEnd = queryPosBase + queryTaskLen - 1;  // inclusive (token space)
                    // KV-space causal boundary for the tile's last token; gates the dense/skip
                    // check and kvLen so the tile never reads KV past the valid range. kCache
                    // holds one row per block (compressed-block space when ratio>1, token
                    // space when ==1), so the bound is in KV rows.
                    int queryPosEndKvBound = MapPosToKv(queryPosEnd);
#ifdef __DAV_C220_VEC__
                    if (assignedDefaultTopk && MapPosToKv(queryPosBase) < (int32_t)topK) {
                        StoreDefaultTopkRows(this->topkIndices + queryTaskOffset * outStride,
                                             queryPosBase, queryTaskLen);
                    }
#endif
                    // skip dense tiles entirely (every token's causal boundary < topK)
                    if (queryPosEndKvBound < (int32_t)topK) {
                        continue;
                    }
                    totalIdx++;  // only increase the global task index when the task is valid
                    if (totalIdx % block_num != block_idx) {
                        continue;
                    }
                    // the owning block serially walks every KV chunk of this query tile,
                    // bounded by the tile's causal boundary; AIC produces one chunk's
                    // scores and notifies the block's AIVs (ffts ping-pong), AIVs merge
                    // the running topK via `lastTopk` locally — cores are fully decoupled
                    for (int kvOffset = 0; kvOffset <= queryPosEndKvBound;
                         kvOffset += tileSizeOfCachedKV) {
                        int kvLen = MIN(tileSizeOfCachedKV, queryPosEndKvBound - kvOffset + 1);
                        RunTaskChunk(batchIdx, queryStart, queryOffset, queryTaskOffset,
                                     queryTaskLen, queryPosBase, blockTable, kvOffset, kvLen, curr,
                                     a2vSyncFlag, v2aSyncFlag);
                        curr = 1 - curr;
                    }
                }
            }
        }
#ifdef __DAV_C220_CUBE__
        wait_flag_dev(v2aSyncFlag[0]);
        wait_flag_dev(v2aSyncFlag[1]);
#else
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ResetPrevCore();
        }
#endif
        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<Dtype> q;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<WeightDtype> weight;
    GlobalTensor<ScoreDtype> scores[PINGPONG_BUF_NUM];
    __gm__ half *kScaleCache;
    uint32_t kScaleCacheStride;
    __gm__ int32_t *setNextSync;
    __gm__ int32_t *waitPrevSync;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;
    __gm__ uint32_t *lastTopk;
    __gm__ uint32_t *seqPositions;
    __gm__ uint32_t *topkIndices;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t weightCols;
    uint32_t blockSize;
    uint32_t kCacheStride;
    uint32_t batch;
    uint32_t blockTableStride;
    uint32_t maxNumBlock;
    uint32_t topK;
    uint32_t tileSizeOfCachedKV;
    bool skipDenseTopk;
    uint32_t compressRatio;  // 1 = token-level KV; >1 = compressed-block-level KV
    int blockIdx;
    int subBlockIdx;
    int nextBlockIdx;
    int prevBlockIdx;
    uint32_t setNextGeneration;
    uint32_t waitPrevGeneration;
    int resetPrevCore;
    bool kvSplit;           // true: split to (tile, chunk) tasks with cross-core relay
    int queryTileSize;      // runtime tile override (Init); see Init's qTileOverride
    bool packTail = false;  // expand block indices to tokens + causal tail (see Init)
    bool reluQk = true;     // per-head ReLU in the AIC score fusion (see Init)
    uint32_t kpool = 0;
    uint32_t outWidth = 0;   // topK, or topK*kpool + kpool-1 when packTail
    uint32_t outStride = 0;  // GM row pitch in words; 128-slot-aligned when packTail

#ifdef __DAV_C220_CUBE__
    LocalTensor<Dtype> ql1Buf;                            // event 4
    LocalTensor<WeightDtype> wl1Buf;                      // event 4
    LocalTensor<Dtype> kl1Buf[PINGPONG_BUF_NUM];          // event 0/1
    LocalTensor<WeightDtype> kql1Buf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<Dtype> l0aBuf;                            // event 4
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];          // event 0/1
    LocalTensor<WeightDtype> wl0aBuf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<WeightDtype> kql0bBuf[PINGPONG_BUF_NUM];  // event 2/3
    LocalTensor<MatDtype> qkl0cBuf[PINGPONG_BUF_NUM];     // event 0/1
    LocalTensor<float> l0cBuf[PINGPONG_BUF_NUM];          // event 2/3
#elif __DAV_C220_VEC__
    __ubuf__ WeightDtype *in[PINGPONG_BUF_NUM];
    __ubuf__ float *lastSort[PINGPONG_BUF_NUM];
    __ubuf__ uint32_t *out[PINGPONG_BUF_NUM];
    __ubuf__ float *mrgSortBuf0;
    __ubuf__ float *mrgSortBuf1;
    __ubuf__ float *totalSort;
    __ubuf__ uint32_t *sortIndices;
    __ubuf__ uint32_t *defaultTopkIndices;
    // template lanes clobbered by BuildTail, restored by RestorePrevTail
    int32_t lastTailCol = -1;
    uint32_t sortIndicesStart = 0;
    uint32_t sortIndicesBytes = 32;
    uint32_t sortIndicesRepeats = 1;
#endif
};

#define INDEXER_TOPK_FUNC_DEFINE(dtype)                                                           \
    extern "C" __global__ __aicore__ void indexer_topk_##dtype(                                   \
        GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc, GM_ADDR queryLens,      \
        GM_ADDR cachedLens, GM_ADDR blockTables, GM_ADDR scores, GM_ADDR lastTopk,                \
        GM_ADDR seqPositions, GM_ADDR topkIndices, GM_ADDR sync, uint32_t kCacheStride,           \
        uint32_t nHeads, uint32_t headDim, uint32_t blockSize, uint32_t batch,                    \
        uint32_t blockTableStride, uint32_t maxNumBlock, uint32_t topK, uint8_t skipDenseTopk,    \
        uint32_t compressRatio, uint32_t weightCols, uint8_t kvSplit, uint8_t qTileOverride,      \
        uint8_t packTail, uint8_t reluQk, uint32_t outStride)                                     \
    {                                                                                             \
        IndexerTopK<dtype, float, dtype, dtype> op;                                               \
        op.Init(q, kCache, weight, queryStartLoc, queryLens, cachedLens, blockTables, scores,     \
                lastTopk, seqPositions, topkIndices, sync, kCacheStride, nHeads, headDim,         \
                blockSize, batch, blockTableStride, maxNumBlock, topK, skipDenseTopk, nullptr, 0, \
                compressRatio, weightCols, kvSplit, qTileOverride, packTail, reluQk, outStride);  \
        op.Run();                                                                                 \
    }

#define INDEXER_TOPK_C8_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void indexer_topk_##dtype(                                   \
        GM_ADDR q, GM_ADDR k_cache, GM_ADDR weight, GM_ADDR query_start_loc, GM_ADDR query_lens,  \
        GM_ADDR cached_lens, GM_ADDR block_tables, GM_ADDR scores, GM_ADDR last_topk,             \
        GM_ADDR seq_positions, GM_ADDR topk_indices, GM_ADDR sync, uint32_t k_cache_stride,       \
        uint32_t heads, uint32_t dim, uint32_t block_size, uint32_t batch,                        \
        uint32_t block_table_stride, uint32_t max_blocks, uint32_t topk, uint8_t skip_dense_topk, \
        GM_ADDR k_scale_cache, uint32_t k_scale_cache_stride, uint32_t compress_ratio,            \
        uint32_t weight_cols, uint8_t kvSplit, uint8_t q_tile_override, uint8_t pack_tail,        \
        uint8_t relu_qk, uint32_t out_stride)                                                     \
    {                                                                                             \
        IndexerTopK<dtype, int32_t, half, float> op;                                              \
        op.Init(q, k_cache, weight, query_start_loc, query_lens, cached_lens, block_tables,       \
                scores, last_topk, seq_positions, topk_indices, sync, k_cache_stride, heads, dim, \
                block_size, batch, block_table_stride, max_blocks, topk, skip_dense_topk,         \
                k_scale_cache, k_scale_cache_stride, compress_ratio, weight_cols, kvSplit,        \
                q_tile_override, pack_tail, relu_qk, out_stride);                                 \
        op.Run();                                                                                 \
    }
