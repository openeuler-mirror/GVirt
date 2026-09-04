/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#define UBA(T) __ubuf__ T *
#define GMA(T) __gm__ T *
#define PINGPONG 2

#ifdef __DAV_C220_VEC__

template <typename dtype>
class Dequant
{
public:
    __aicore__ inline Dequant()
    {
    }

    __aicore__ inline void Init(bool hasScale)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        this->nTile = 7168;
        this->nPad = ROUND_UP(this->nTile, (256 / sizeof(dtype)));
        this->hasScale = hasScale;

        this->inUbBuf[0] = reinterpret_cast<UBA(dtype)>((uintptr_t)0);
        this->inUbBuf[1] = reinterpret_cast<UBA(dtype)>((uintptr_t)(this->inUbBuf[0] + this->nPad));
        this->tmpUbBuf[0] =
            reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->inUbBuf[1] + this->nPad));
        this->tmpUbBuf[1] =
            reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->tmpUbBuf[0] + this->nPad));
        this->mulUbBuf[0] =
            reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->tmpUbBuf[1] + this->nPad));
        this->mulUbBuf[1] =
            reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->mulUbBuf[0] + this->nPad));
        this->outUbBuf[0] =
            reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(this->mulUbBuf[1] + this->nPad));
        this->outUbBuf[1] =
            reinterpret_cast<UBA(bfloat16_t)>((uintptr_t)(this->outUbBuf[0] + this->nPad));
        this->scaleUbBuf =
            reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->outUbBuf[1] + this->nPad));

        UBA(float32_t)
        endAddr = reinterpret_cast<UBA(float32_t)>((uintptr_t)(this->scaleUbBuf + 1));
    }

    __aicore__ inline void SetFlags()
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        this->eventId = 0;
    }

    __aicore__ inline int64_t TaskTilesInit(GM_ADDR in, GM_ADDR scale, GM_ADDR out,
                                            GM_ADDR pnumTokens, uint32_t m, uint32_t n)
    {
        this->inGmBuf = reinterpret_cast<GMA(dtype)>(in);
        this->scaleGmBuf = reinterpret_cast<GMA(float32_t)>(scale);
        this->outGmBuf = reinterpret_cast<GMA(dtype)>(out);
        this->m = m;
        this->n = n;
        if (pnumTokens != nullptr) {
            uint32_t pnumTokensValue = *((__gm__ uint32_t *)pnumTokens);
            this->m = pnumTokensValue < m ? pnumTokensValue : m;
        }
        this->nLoop = DIV_ROUND_UP(this->n, this->nTile);
        return static_cast<int64_t>(this->m);
    }

    __aicore__ inline void RunTileByIdx(int64_t idx)
    {
        RunTile(this->inGmBuf + idx * this->n,
                this->scaleGmBuf == nullptr ? nullptr : this->scaleGmBuf + idx,
                this->outGmBuf + idx * this->n, 1, this->n);
    }

    __aicore__ inline void RunTile(GMA(dtype) tileInGm, GMA(float32_t) tileScaleGm,
                                   GMA(dtype) tileOutGm, uint32_t localRows, uint32_t nActual)
    {
        if (localRows == 0 || nActual == 0) {
            return;
        }
        uint32_t nLoop = DIV_ROUND_UP(nActual, this->nTile);
        for (uint32_t row = 0; row < localRows; row++) {
            for (uint32_t loop = 0; loop < nLoop; loop++) {
                uint32_t nOffset = loop * this->nTile;
                uint32_t nSize = (loop == nLoop - 1) ? (nActual - nOffset) : this->nTile;
                uint32_t nSizePad = ROUND_UP(nSize, (256 / sizeof(dtype)));
                uint32_t nRepeats = DIV_ROUND_UP(nSizePad, VECTOR_MAX_NUM_OF_FP32);
                int eventId = this->eventId;

                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + eventId);
                copy_gm_to_ubuf_align_b16(this->inUbBuf[eventId],
                                          tileInGm + row * this->n + nOffset, 0, 1,
                                          nSize * sizeof(dtype), 0, 0, 0, 0);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eventId);

                if (this->hasScale) {
                    copy_gm_to_ubuf_align_b16(this->scaleUbBuf, tileScaleGm + row, 0, 1,
                                              sizeof(float), 0, 0, 0, 0);
                    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                }

                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + eventId);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eventId);
                vconv_f162f32(this->tmpUbBuf[eventId], this->inUbBuf[eventId], nRepeats, 1, 1, 8,
                              4);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + eventId);

                UBA(float32_t) tmpPtr = this->tmpUbBuf[eventId];
                if (this->hasScale) {
                    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                    vmuls(this->mulUbBuf[eventId], this->tmpUbBuf[eventId],
                          float(*this->scaleUbBuf), nRepeats, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    tmpPtr = this->mulUbBuf[eventId];
                }

                vconv_f322bf16r(this->outUbBuf[eventId], tmpPtr, nRepeats, 1, 1, 4, 8);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eventId);

                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eventId);
                copy_ubuf_to_gm_align_b16(tileOutGm + row * this->n + nOffset,
                                          this->outUbBuf[eventId], 0, 1, nSize * sizeof(bfloat16_t),
                                          0, 0, 0, 0);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eventId);
                this->eventId = 1 - eventId;
            }
        }
    }

    __aicore__ inline void WaitFlags()
    {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Run(GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnumTokens,
                               uint32_t m, uint32_t n)
    {
        SetFlags();
        int64_t tiles = TaskTilesInit(in, scale, out, pnumTokens, m, n);
        for (int64_t idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum()) {
            RunTileByIdx(idx);
        }
        WaitFlags();
    }

private:
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t nTile = 0;
    uint32_t nPad = 0;
    uint32_t nLoop = 0;
    bool hasScale = false;
    int eventId = 0;
    GMA(dtype) inGmBuf = nullptr;
    GMA(float32_t) scaleGmBuf = nullptr;
    GMA(dtype) outGmBuf = nullptr;
    UBA(dtype) inUbBuf[PINGPONG] = { nullptr, nullptr };
    UBA(float32_t) tmpUbBuf[PINGPONG] = { nullptr, nullptr };
    UBA(float32_t) mulUbBuf[PINGPONG] = { nullptr, nullptr };
    UBA(bfloat16_t) outUbBuf[PINGPONG] = { nullptr, nullptr };
    UBA(float32_t) scaleUbBuf = nullptr;
};

#define DEQUANT_FUNC_DEFINE(dtype)                                                           \
    extern "C" __global__ __aicore__ void dequant_##dtype(                                   \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t n) \
    {                                                                                        \
        Dequant<dtype> op;                                                                   \
        op.Init(scale != nullptr);                                                           \
        op.Run(in, scale, out, pnum_tokens, m, n);                                           \
    }
#else
#define DEQUANT_FUNC_DEFINE(dtype)                                                           \
    extern "C" __global__ __aicore__ void dequant_##dtype(                                   \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t n) \
    {                                                                                        \
    }
#endif
