/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#define UBA(T) __ubuf__ T *
#define GMA(T) __gm__ T *
#define PINGPONG 2
#define AIC_AIV_RATIO 2
// Elements per column tile: the PINGPONG buffer sets (in dtype + f32 tmp + bf16
// out = 8B/elem each) exactly fill the 192KB UB.
#define DEQUANT_N_TILE 12288

#ifdef __DAV_C220_VEC__

template <typename dtype>
class Dequant
{
public:
    __aicore__ inline Dequant()
    {
    }

    __aicore__ inline void Init(bool hasScale, uint64_t m0 = 1, uint64_t n0 = 0)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        this->nTile = DEQUANT_N_TILE;
        this->nPad = ROUND_UP(this->nTile, (VECTOR_MAX_BYTESIZE / sizeof(dtype)));
        this->m0 = m0;
        this->n0 = n0;
        this->hasScale = hasScale;

        this->inUbBuf[0] = reinterpret_cast<UBA(dtype)>((uintptr_t)0);
        this->inUbBuf[1] = reinterpret_cast<UBA(dtype)>((uintptr_t)(this->inUbBuf[0] + this->nPad));
        this->tmpUbBuf[0] =
            reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->inUbBuf[1] + this->nPad));
        this->tmpUbBuf[1] =
            reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->tmpUbBuf[0] + this->nPad));
        this->outUbBuf[0] =
            reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(this->tmpUbBuf[1] + this->nPad));
        this->outUbBuf[1] =
            reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(this->outUbBuf[0] + this->nPad));
    }

    __aicore__ inline void SetFlags()
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

    __aicore__ inline int64_t TaskTilesInit(GM_ADDR in, GM_ADDR scale, GM_ADDR out,
                                            GM_ADDR pnumTokens, uint64_t m, uint64_t n,
                                            bool isInFused)
    {
        this->inGmBuf = reinterpret_cast<GMA(dtype)>(in);
        this->scaleGmBuf = reinterpret_cast<GMA(float32_t)>(scale);
        this->outGmBuf = reinterpret_cast<GMA(dtype)>(out);
        this->m = m;
        this->n = n;
        this->eventId = 0;
        if (pnumTokens != nullptr) {
            uint32_t pnumTokensValue = *((__gm__ uint32_t *)pnumTokens);
            this->m = pnumTokensValue < m ? pnumTokensValue : m;
        }
        // fusion operator need AIC_tilecount:AIV_tilecount=1:2
        this->isInFused = isInFused;
        if (isInFused) {
            this->mTiles = DIV_ROUND_UP(this->m, this->m0 * AIC_AIV_RATIO) * AIC_AIV_RATIO;
        } else {
            this->mTiles = DIV_ROUND_UP(this->m, this->m0);
        }

        this->nTiles = this->n0 > 0 ? DIV_ROUND_UP(this->n, this->n0) : 1;
        return static_cast<int64_t>(this->mTiles * this->nTiles);
    }

    // Linear tile index → 2D (mIdx, nIdx): tileIdx = mIdx * nTiles + nIdx
    __aicore__ inline void RunTileByIdx(int64_t idx)
    {
        int64_t mIdx;
        int64_t nIdx;
        if (this->isInFused) {
            int64_t aicIdx = idx >> 1;  // AIC tile linear index (mIdx_aic * nTiles + nIdx)
            int64_t sub = idx & 1;      // which row-half of the AIC m-tile
            mIdx = (aicIdx / this->nTiles) * 2 + sub;
            nIdx = aicIdx % this->nTiles;
        } else {
            mIdx = idx / this->nTiles;
            nIdx = idx % this->nTiles;
        }
        uint32_t rowOffset = static_cast<uint64_t>(mIdx) * this->m0;
        uint32_t colOffset = this->n0 > 0 ? static_cast<uint64_t>(nIdx) * this->n0 : 0;

        if (rowOffset >= this->m || colOffset >= this->n) {  // skip when tile out of range
            return;
        }
        uint32_t actualRows = MIN((this->m - rowOffset), this->m0);
        uint32_t localCols = this->n0 > 0 ? MIN((this->n - colOffset), this->n0) : this->n;

        // Row stride is always this->n (full matrix width in row-major GM layout)
        RunTile(this->inGmBuf + rowOffset * this->n + colOffset,
                this->scaleGmBuf == nullptr ? nullptr : this->scaleGmBuf + rowOffset,
                this->outGmBuf + rowOffset * this->n + colOffset, actualRows, localCols);
    }

    // Row-batched tile processing (any tile: strided or full-width). Column
    // chunks of at most nTile elements; inside a chunk, rows are batched: one
    // 2D DMA per group moves [r, colChunk] — r bursts of colChunk elements,
    // gaps skipping the GM row stride (n - colChunk) and the UB row pad
    // (colsPad - colChunk). Full 256B-aligned chunks have no pad and
    // back-to-back bursts, i.e. one long contiguous transfer. The pad aligns
    // every per-row vector operand to 256B and lets repeats round up over it
    // (pad garbage is converted and scaled but never stored: the store gap
    // skips it). Burst lengths and gaps are byte-granular, so colChunk and
    // (n - colChunk) must be 32B-aligned (all producer shapes are).
    __aicore__ inline void RunTile(GMA(dtype) tileInGm, GMA(float32_t) tileScaleGm,
                                   GMA(dtype) tileOutGm, uint32_t localRows, uint32_t localCols)
    {
        if (localRows == 0 || localCols == 0) {
            return;
        }
        assert(localCols % (BLOCK_SIZE / sizeof(dtype)) == 0 &&
               (this->n - localCols) % (BLOCK_SIZE / sizeof(dtype)) == 0);
        constexpr uint32_t colAlign = VECTOR_MAX_BYTESIZE / sizeof(dtype);
        float scaleValue;
        uint32_t colOff = 0;
        while (colOff < localCols) {
            uint32_t colRemain = localCols - colOff;
            uint32_t colChunk = colRemain < colAlign
                                    ? colRemain
                                    : MIN(colRemain, this->nTile) / colAlign * colAlign;
            uint32_t colsPad = ROUND_UP(colChunk, colAlign);
            uint32_t rows = MIN(localRows, this->nTile / colsPad);
            uint32_t kRepeats = colsPad / VECTOR_MAX_NUM_OF_FP32;
            uint32_t numGroups = DIV_ROUND_UP(localRows, rows);
            // __set_dmi_config expresses lenBurst and both gaps in 32B units, so
            // colChunk and (n - colChunk) must be 32B-expressible — the %16
            // contract above (align_b16's expanded-arg form was tried and its
            // gap units match neither bytes nor 16/32B; config form is the one
            // quant.h/rope_and_cache.h rely on)
            uint32_t strideGap = (this->n - colChunk) * sizeof(dtype) / BLOCK_SIZE;  // GM row gap
            uint32_t padGap = (colsPad - colChunk) * sizeof(dtype) / BLOCK_SIZE;     // UB row pad
            for (uint32_t g = 0; g < numGroups; g++) {
                uint32_t base = g * rows;
                uint32_t r = MIN(rows, localRows - base);
                uint32_t rRepeats = r * kRepeats;
                int eventId = this->eventId;
                GMA(dtype) chunkInGm = tileInGm + (uint64_t)base * this->n + colOff;
                GMA(dtype) chunkOutGm = tileOutGm + (uint64_t)base * this->n + colOff;

                // in: GM rows strided (srcGap) -> UB rows padded (dstGap);
                // out: UB rows padded (srcGap) -> GM rows strided (dstGap)
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + eventId);
                copy_gm_to_ubuf(this->inUbBuf[eventId], chunkInGm,
                                __set_dmi_config(0, r, colChunk * sizeof(dtype) / BLOCK_SIZE,
                                                 strideGap, padGap));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eventId);

                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eventId);
                vconv_f162f32(this->tmpUbBuf[eventId], this->inUbBuf[eventId], rRepeats, 1, 1, 8,
                              4);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + eventId);

                if (this->hasScale) {
                    for (uint32_t row = 0; row < r; row++) {
                        scaleValue = *(tileScaleGm + base + row);
                        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        vmuls(this->tmpUbBuf[eventId] + row * colsPad,
                              this->tmpUbBuf[eventId] + row * colsPad, scaleValue, kRepeats, 1, 1,
                              8, 8);
                    }
                    pipe_barrier(PIPE_V);
                }

                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eventId);
                vconv_f322bf16r(this->outUbBuf[eventId], this->tmpUbBuf[eventId], rRepeats, 1, 1, 4,
                                8);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eventId);

                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eventId);
                copy_ubuf_to_gm(chunkOutGm, this->outUbBuf[eventId],
                                __set_dmi_config(0, r, colChunk * sizeof(bfloat16_t) / BLOCK_SIZE,
                                                 padGap, strideGap));
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eventId);
                this->eventId = 1 - eventId;
            }
            colOff += colChunk;
        }
    }

    __aicore__ inline void WaitFlags()
    {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

    __aicore__ inline void Run(GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnumTokens,
                               uint64_t m, uint64_t n)
    {
        SetFlags();
        int64_t tiles = TaskTilesInit(in, scale, out, pnumTokens, m, n, false);
        for (int64_t idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum()) {
            RunTileByIdx(idx);
        }
        WaitFlags();
    }

    // Rows per tile. Aim for ~4 tiles per block (grid-stride balance), capped
    // at groupRows — the rows one full 2D batch holds (nTile / colsPad) — so a
    // full tile is exactly one full batch. TP-sharded widths (128..768) give
    // groupRows 16..96, far above what any fixed cap could serve.
    static __aicore__ inline uint64_t PickM0(uint64_t m, uint64_t n)
    {
        constexpr uint64_t colAlign = VECTOR_MAX_BYTESIZE / sizeof(dtype);
        uint64_t colChunk = ROUND_UP(MIN(n, DEQUANT_N_TILE), colAlign);
        uint64_t groupRows = DEQUANT_N_TILE / colChunk;
        uint64_t m0 = DIV_ROUND_UP(m, (uint64_t)GetBlockNum() * 4);
        return MAX(MIN(m0, groupRows), 1);
    }

private:
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t m0 = 1;
    uint64_t n0 = 0;
    uint32_t mTiles = 0;
    uint32_t nTiles = 1;
    uint32_t nTile = 0;
    uint32_t nPad = 0;
    bool hasScale = false;
    bool isInFused = false;
    int eventId = 0;
    GMA(dtype) inGmBuf = nullptr;
    GMA(float32_t) scaleGmBuf = nullptr;
    GMA(dtype) outGmBuf = nullptr;
    UBA(dtype) inUbBuf[PINGPONG] = { nullptr, nullptr };
    UBA(float32_t) tmpUbBuf[PINGPONG] = { nullptr, nullptr };
    UBA(bfloat16_t) outUbBuf[PINGPONG] = { nullptr, nullptr };
};

#define DEQUANT_FUNC_DEFINE(dtype)                                                           \
    extern "C" __global__ __aicore__ void dequant_##dtype(                                   \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t n) \
    {                                                                                        \
        Dequant<dtype> op;                                                                   \
        op.Init(scale != nullptr, Dequant<dtype>::PickM0(m, n));                             \
        op.Run(in, scale, out, pnum_tokens, m, n);                                           \
    }
#else
#define DEQUANT_FUNC_DEFINE(dtype)                                                           \
    extern "C" __global__ __aicore__ void dequant_##dtype(                                   \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t n) \
    {                                                                                        \
    }
#endif
