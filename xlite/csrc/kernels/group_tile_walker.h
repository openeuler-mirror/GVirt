/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_GROUP_TILE_WALKER_H_
#define _XLITE_GROUP_TILE_WALKER_H_

#include "kernel_macro.h"

// ---- Grouped-matmul shared tile-walk skeleton and cross-core hand-off ----
// Dedicated fused-op utility: the producer (GroupMatmul walk) and the consumer
// (the AIV mirror walk of fused matmul+dequant shells) live in different
// headers, so the traversal skeleton they must agree on — and the flag id +
// hand-off calls that order their per-tile hand-off — live here, next to each
// other. The skeleton itself is PURE TRAVERSAL: the fused shells issue the
// cross-core calls explicitly around each tile, so a reader of the shell sees
// the synchronization. The dense fused shell and the legacy pipeline kernel
// keep their local identical flag definitions and explicit calls.
//
// Cross-core flag id for the per-tile AIC -> AIV hand-off of the fused
// matmul+dequant family (MIX_AIC_1_2). The id space has 16 entries and is
// independent of the intra-core set_flag/wait_flag event ids used by
// matmul/dequant/quant.
#define C2V_CROSS_CORE_FLAG 8

// Reverse-direction ack of the same family: the paired AIVs raise it once
// per V2C_SYNC_INTERVAL tiles and the AIC waits on it between tile batches,
// capping how far its C2V sets may run ahead of the AIVs' waits — without
// this backpressure the AIC's set stream outruns the single ffts flag slot
// on large walks, the trailing AIV waits never fire and the kernel dies with
// an AI Core "timeout or trap" error (the !609 dense-side failure mode,
// fixed there by !654 with the same ack scheme).
#define V2C_CROSS_CORE_FLAG 9

// Per-tile C2V hand-off calls of the fused group shells. Producer side (AIC):
// one call per executed tile, issued right after the tile's work is enqueued —
// the set rides the FIX pipe queue, so it fires after the tile's CopyToGm
// retires and stays ahead of the next tile's work. Consumer side (AIV): one
// call per handed-out tile, BEFORE the tile body and before any empty-tail
// skip, so the per-pair drain count equals the AIC's set count by construction.
__aicore__ inline void SetC2VFlag()
{
    CrossCoreSetFlag<0x2, PIPE_FIX>(C2V_CROSS_CORE_FLAG);
}

__aicore__ inline void WaitC2VFlag()
{
    CrossCoreWaitFlag<2>(C2V_CROSS_CORE_FLAG);
}

// Reverse backpressure hand-off, the counterpart of the per-tile pair above.
// Consumer side (AIV): set rides the MTE2 queue, so it fires only after this
// batch's GM loads of the in-place FP16 view have retired — the tile data the
// AIC wrote is safely consumed. Producer side (AIC): scalar wait between
// tile batches; the shells own the batching interval, the pipeline depth and
// the end-of-walk drain that keeps the two counts balanced.
__aicore__ inline void SetV2CFlag()
{
    CrossCoreSetFlag<0x2, PIPE_MTE2>(V2C_CROSS_CORE_FLAG);
}

__aicore__ inline void WaitV2CFlag()
{
    CrossCoreWaitFlag<2>(V2C_CROSS_CORE_FLAG);
}

// Shared tile-traversal swizzle (lifted from Matmul::GetMNBlockIdx, matmul.h).
// Reorders the linear tile index within one expert's [0, mLoop*nLoop) grid to
// raise L2 weight reuse: swizzle>>8 = block count, swizzle&0xFF = direction
// (0 = Zn: blocks of `count` M-rows, m fastest, odd blocks flip nIdx; 1 = Nz:
// blocks of `count` N-cols, n fastest, odd blocks flip mIdx). Pure bijection,
// so it never changes which tiles run - only their order - and the matmul
// result is swizzle-invariant. Both the AIC Matmul (RunTileByIdx) and the AIV
// mirror (GroupTileWalker) MUST apply the SAME swizzle so the per-tile C2V flag
// pairs stay aligned; that is why the decode lives here, with the walk
// skeleton, rather than in Matmul.
__aicore__ inline void GetSwizzledMNBlockIdx(int32_t loopIdx, int32_t mLoop, int32_t nLoop,
                                             int32_t swizzlDirection, int32_t swizzlCount,
                                             int64_t &mIdx, int64_t &nIdx)
{
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

// Both sides MUST seed Begin() with the same m0 (the AIC tile m — the AIV
// passes parentM0), n0, kN and the same int4Packed expression; n0 must be > 0.
// GetBlockNum() returns the AIC core count on both sides of a MIX launch.
//
// NextTile() is pure traversal: it hands out this core's tiles and never
// touches the cross-core flag — the fused shells bracket each tile with the
// explicit SetC2VFlag()/WaitC2VFlag() calls above.
//
// swizzle (traversal order, Matmul encoding): default 0 keeps the historical
// linear (row-major, n-fastest) walk. When non-zero, NextTile() remaps
// midx/nidx through GetSwizzledMNBlockIdx so the AIV mirror visits tiles in
// the same order the AIC Matmul (configured with the same swizzle) computes
// them — see the free function's contract above. Both arms of a fused shell
// MUST seed the same value.
struct GroupTileWalker {
    // Caller-readable state: i/kM/off/tiles/nLoop/m0 are valid from the moment
    // NextExpert() returns true until the next NextExpert() call; idx/midx/nidx
    // describe the tile handed out by the most recent NextTile().
    uint32_t i = 0;     // current expert index
    int64_t kM = 0;     // rows of expert i (already doubled when int4Packed)
    int64_t off = 0;    // global row base of expert i (sum of prior kM)
    int64_t tiles = 0;  // mLoop * nLoop tiles of expert i
    int64_t nLoop = 1;  // column tiles: DIV_ROUND_UP(kN, n0)
    int64_t mLoop = 1;  // row tiles: DIV_ROUND_UP(kM, m0); required for swizzle
    int64_t m0 = 1;     // AIC m-tile rows (the AIV's parent m0)
    int64_t idx = 0;    // flat tile index within expert i (row-major)
    int64_t midx = 0;   // swizzle==0: idx / nLoop; else the swizzled M-tile
    int64_t nidx = 0;   // swizzle==0: idx % nLoop; else the swizzled N-tile

    __aicore__ inline void Begin(GM_ADDR counts, uint32_t n, uint32_t startIdx, uint32_t endIdx,
                                 int walkBlockIdx, int64_t m0, int64_t n0, int64_t kN,
                                 bool int4Packed, uint64_t swizzle = 0)
    {
        this->counts = counts;
        this->n = n;
        this->endIdx = endIdx;
        this->i = startIdx;
        this->walkBlockIdx = walkBlockIdx;
        this->blockNum = GetBlockNum();
        this->m0 = m0;
        this->n0 = n0;
        this->kN = kN;
        this->int4Packed = int4Packed;
        this->swizzle = swizzle;
        this->swizzlCount = static_cast<int32_t>(swizzle >> 8);
        this->swizzleDirection = static_cast<int32_t>(swizzle & 0xFF);
        this->off = 0;
        this->coreOffset = 0;
        this->active = false;
    }

    // Advance to the next expert with kM > 0 in [startIdx, min(endIdx, n));
    // closes the previous expert (phase + row-base update + index step)
    // exactly once. Returns false when the walk is done.
    __aicore__ inline bool NextExpert()
    {
        if (this->active) {
            this->coreOffset = (this->coreOffset + this->tiles) % this->blockNum;
            this->off += this->kM;
            this->active = false;
            this->i++;
        }
        for (; this->i < this->endIdx && this->i < this->n; this->i++) {
            int64_t kM = static_cast<int64_t>(
                *((__gm__ uint32_t *)(this->counts + this->i * sizeof(uint32_t))));
            if (kM <= 0) {
                continue;  // zero-count experts produce no tiles on any core
            }
            if (this->int4Packed) {
                kM *= 2;
            }
            this->kM = kM;
            this->nLoop = DIV_ROUND_UP(this->kN, this->n0);
            this->mLoop = DIV_ROUND_UP(kM, this->m0);
            this->tiles = this->mLoop * this->nLoop;
            this->cursor =
                (this->walkBlockIdx + this->blockNum - this->coreOffset) % this->blockNum;
            this->active = true;
            return true;
        }
        return false;
    }

    // Hand out this core's next tile of the current expert (pure traversal —
    // the fused shells issue the per-tile C2V calls around the tile body).
    __aicore__ inline bool NextTile()
    {
        if (this->cursor >= this->tiles) {
            return false;
        }
        this->idx = this->cursor;
        this->midx = this->idx / this->nLoop;
        this->nidx = this->idx % this->nLoop;
        if (this->swizzle != 0) {
            // Match the AIC Matmul's swizzled RunTileByIdx decode exactly, so
            // the AIV dequantizes the tile the AIC just produced (flag-paired).
            GetSwizzledMNBlockIdx(static_cast<int32_t>(this->idx),
                                  static_cast<int32_t>(this->mLoop),
                                  static_cast<int32_t>(this->nLoop), this->swizzleDirection,
                                  this->swizzlCount, this->midx, this->nidx);
        }
        this->cursor += this->blockNum;
        return true;
    }

private:
    GM_ADDR counts = nullptr;
    uint32_t n = 0;  // expert-count bound
    uint32_t endIdx = 0;
    int walkBlockIdx = 0;
    int blockNum = 1;
    int64_t n0 = 1;
    int64_t kN = 0;
    bool int4Packed = false;
    int64_t coreOffset = 0;        // phase carry across experts
    int64_t cursor = 0;            // next candidate flat tile index
    bool active = false;           // positioned on an expert awaiting close
    uint64_t swizzle = 0;          // 0 = linear walk; else Matmul-encoding traversal
    int32_t swizzlCount = 0;       // swizzle >> 8
    int32_t swizzleDirection = 0;  // swizzle & 0xFF
};

#endif  // _XLITE_GROUP_TILE_WALKER_H_
