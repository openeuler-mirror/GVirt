/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

// out = int8(in * scale_reciprocal + offset)
// per-channel (static) quantization, scale_reciprocal/offset: [k]
// [m, k] activation; dtype = float16_t / bfloat16_t
template <typename dtype>
class Quant
{
public:
    __aicore__ inline Quant()
    {
    }

    __aicore__ inline void Init()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    // Since the quant operator needs to process entire rows and its tiling
    // strategy does not align well with that of matmul, it is preferable to
    // complete the fusion operator in a single submission, with each AIV
    // processing multiple consecutive rows.
    // Notice: In the Mix operator, GetBlockNum is determined by the number
    // of AICs. Therefore, taskRatio is required to supplement the actual
    // number of AIVs. This is not required for the AIV operator.
    __aicore__ inline int64_t TaskTilesInit(GM_ADDR in, GM_ADDR scaleReciprocal, GM_ADDR offset,
                                            GM_ADDR out, uint64_t m, uint64_t k,
                                            uint64_t taskRatio = 1)
    {
        this->inGm = reinterpret_cast<__gm__ dtype *>(in);
        this->scaleReciprocalGm = reinterpret_cast<__gm__ dtype *>(scaleReciprocal);
        this->offsetGm = reinterpret_cast<__gm__ dtype *>(offset);
        this->outGm = reinterpret_cast<__gm__ int8_t *>(out);
        // TODO: reduce num to pnumTokens
        this->m = m;
        this->k = k;
        this->tileRows = DIV_ROUND_UP(m, GetBlockNum() * taskRatio);

        constexpr uint32_t align = VECTOR_REPEAT_BYTESIZE / sizeof(dtype);
        // rowMax = (UB_SIZE - 12*kPad) / (12*kPad)
        constexpr uint32_t maxOneRowKTile =
            ROUND_DOWN(UB_SIZE / 24, VECTOR_REPEAT_BYTESIZE / sizeof(dtype));
        this->kLoop = DIV_ROUND_UP((uint32_t)k, maxOneRowKTile);
        this->kPad = ROUND_UP(DIV_ROUND_UP((uint32_t)k, this->kLoop), align);
        uint32_t rMax = MAX(1, (UB_SIZE - 12 * this->kPad) / (12 * this->kPad));
        this->rows = MIN(rMax, (uint32_t)this->tileRows);

        // ** UB layout **
        // all buffers sized by the *actual* kPad (not a worst-case constant),
        // so small k gets more rows.
        // Per-row items: 12 * Rows * kPad
        //   x1,x2 [rows,kPad] dtype, xf32 [rows,kPad] float,
        //   z1,z2 [rows,kPad] int8, xf16 [rows,kPad] half   →
        // Fixed (one-shot) items: 12 * kPad
        //   scale,offset [kPad] dtype
        //   scale_f32,offset_f32 [kPad] float
        // Total = 12*(R+1)*kPad, and rMax guarantees this <= UB_SIZE.
        this->x1 = reinterpret_cast<__ubuf__ dtype *>((uintptr_t)0);
        this->x2 = this->x1 + this->rows * this->kPad;
        this->xf32 = reinterpret_cast<__ubuf__ float *>(this->x2 + this->rows * this->kPad);
        this->z1 = reinterpret_cast<__ubuf__ int8_t *>(this->xf32 + this->rows * this->kPad);
        this->z2 = this->z1 + this->rows * this->kPad;
        this->scale = reinterpret_cast<__ubuf__ dtype *>(this->z2 + this->rows * this->kPad);
        this->offset = this->scale + this->kPad;
        this->scale_f32 = reinterpret_cast<__ubuf__ float *>(this->offset + this->kPad);
        this->offset_f32 = this->scale_f32 + this->kPad;
        this->xf16 = reinterpret_cast<__ubuf__ half *>(this->offset_f32 + this->kPad);
        assert((uint64_t)(this->xf16 + this->rows * this->kPad) <= UB_SIZE);

        return static_cast<int64_t>(DIV_ROUND_UP(m, this->tileRows));
    }

    // After TaskTilesInit initializes a batch of loads, each idx processes one tile.
    // The caller is responsible for calling SetFlags and WaitFlags.
    // Processing method: If k is large, the data is processed in kLoop times.
    // If k is small, multiple rows can be processed at a time.
    // The stride of MTE2/MTE3 requires that k be 32-byte aligned, so that data
    // can be efficiently transferred.
    __aicore__ inline void RunTileByIdx(int64_t idx)
    {
        uint64_t rowStart = static_cast<uint64_t>(idx) * this->tileRows;
        if (rowStart >= this->m) {  // skip when tile out of range
            return;
        }
        const uint32_t start = static_cast<uint32_t>(rowStart);
        const uint32_t rowCount = static_cast<uint32_t>(MIN(this->m - rowStart, this->tileRows));

        // Host rejects ragged k. The stride of MTE2/MTE3 requires that k
        // be 32-byte aligned, so that data can be efficiently transferred.
        assert(this->k % 32 == 0);
        const uint32_t k = (uint32_t)this->k;
        const uint32_t rows = this->rows;
        const uint32_t kLoop = this->kLoop;
        const uint32_t kPad = this->kPad;

        __ubuf__ dtype *xBufs[2] = {this->x1, this->x2};
        __ubuf__ int8_t *zBufs[2] = {this->z1, this->z2};
        auto *xf32 = this->xf32;
        auto *scale = this->scale;
        auto *offset = this->offset;
        auto *scale_f32 = this->scale_f32;
        auto *offset_f32 = this->offset_f32;
        auto *xf16 = this->xf16;

        const uint32_t end = start + rowCount;
        int eventId = 0;
        int xId = 0;
        // When k is pretty large, k will be divided into kLoop segments
        for (uint32_t loop = 0; loop < kLoop; loop++) {
            uint32_t kOffset = loop * kPad;
            uint32_t kSize = MIN(kPad, k - kOffset);
            uint32_t kRepeats = kPad / VECTOR_MAX_NUM_OF_FP32;

            // All rows share the same scale and offset, so we assume these buffers
            // don't need ping-pong buffer to overlap datacopy and calculation
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            copy_gm_to_ubuf_align_b16(scale, this->scaleReciprocalGm + kOffset, 0, 1,
                                      kSize * sizeof(dtype), 0, 0, 0, 0);
            // Notice: offset of most models is all zeros, maybe we can skip
            copy_gm_to_ubuf_align_b16(offset, this->offsetGm + kOffset, 0, 1, kSize * sizeof(dtype),
                                      0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            convert_input<dtype>(scale_f32, scale, kRepeats);
            convert_input<dtype>(offset_f32, offset, kRepeats);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

            // Due to the limitation of the UB size, only rows of data
            // can be loaded at a time.
            for (uint32_t base = start; base < end; base += rows) {
                uint32_t r = MIN(rows, end - base);
                uint32_t rRepeats = r * kRepeats;

                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + xId);
                copy_gm_to_ubuf(xBufs[xId], this->inGm + (uint64_t)base * k + kOffset, 0, r,
                                kSize * sizeof(dtype) / 32, (k - kSize) * sizeof(dtype) / 32,
                                (kPad - kSize) * sizeof(dtype) / 32);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + xId);

                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + xId);
                convert_input<dtype>(xf32, xBufs[xId], rRepeats);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + xId);  // slot dead, its next load may issue
                xId = 1 - xId;

                for (uint32_t i = 0; i < r; i++) {
                    __ubuf__ float *row = xf32 + i * kPad;
                    vmul(row, row, scale_f32, kRepeats, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    vadd(row, row, offset_f32, kRepeats, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }

                // Three-step fp32→int32→fp16→int8 chain, matching ascend_quant_v2
                // CastOut on 910C: RINT for both fp32→int32 and fp16→int8, with
                // SetDeqScale(1.0) for the int32→fp16 dequantize step. The int32
                // intermediate reuses xf32 in-place (same 4-byte element width),
                // so the UB layout is unchanged.
                vconv_f322s32r(reinterpret_cast<__ubuf__ int32_t *>(xf32), xf32, rRepeats, 1, 1, 8,
                               8);
                pipe_barrier(PIPE_V);
                set_deqscale(static_cast<half>(1.0f));
                pipe_barrier(PIPE_V);
                vconv_deq(xf16, reinterpret_cast<__ubuf__ int32_t *>(xf32), rRepeats, 1, 1, 4, 8);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE3, PIPE_V,
                          EVENT_ID0 + eventId);  // zBufs[eventId] free from prev MTE3
                vconv_f162s8r(zBufs[eventId], xf16, r * kPad / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eventId);  // zBufs[eventId] ready for MTE3

                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eventId);
                copy_ubuf_to_gm(this->outGm + (uint64_t)base * k + kOffset, zBufs[eventId], 0, r,
                                kSize / 32, (kPad - kSize) / 32, (k - kSize) / 32);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eventId);
                eventId = 1 - eventId;
            }
        }
    }

    __aicore__ inline void Run(GM_ADDR in, GM_ADDR scaleReciprocal, GM_ADDR offset, GM_ADDR out,
                               uint32_t m, uint32_t k)
    {
        SetFlags();
        int64_t tiles = TaskTilesInit(in, scaleReciprocal, offset, out, m, k);
        for (int64_t idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum()) {
            RunTileByIdx(idx);
        }
        WaitFlags();
    }

    __aicore__ inline void SetFlags()
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    }

    __aicore__ inline void WaitFlags()
    {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

private:
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t tileRows = 1;
    // Per-launch group plan + UB layout, derived once in TaskTilesInit.
    uint32_t rows = 1;
    uint32_t kLoop = 1;
    uint32_t kPad = 1;
    __gm__ dtype *inGm = nullptr;
    __gm__ dtype *scaleReciprocalGm = nullptr;
    __gm__ dtype *offsetGm = nullptr;
    __gm__ int8_t *outGm = nullptr;
    __ubuf__ dtype *x1 = nullptr;
    __ubuf__ dtype *x2 = nullptr;
    __ubuf__ float *xf32 = nullptr;
    __ubuf__ int8_t *z1 = nullptr;
    __ubuf__ int8_t *z2 = nullptr;
    __ubuf__ dtype *scale = nullptr;
    __ubuf__ dtype *offset = nullptr;
    __ubuf__ float *scale_f32 = nullptr;
    __ubuf__ float *offset_f32 = nullptr;
    __ubuf__ half *xf16 = nullptr;
};

#define QUANT_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void quant_static_##dtype(                                    \
        GM_ADDR in, GM_ADDR scale_reciprocal, GM_ADDR offset, GM_ADDR out, uint32_t m, uint32_t k) \
    {                                                                                              \
        Quant<dtype> op;                                                                           \
        op.Init();                                                                                 \
        op.Run(in, scale_reciprocal, offset, out, m, k);                                           \
    }
#else
#define QUANT_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void quant_static_##dtype(                                    \
        GM_ADDR in, GM_ADDR scale_reciprocal, GM_ADDR offset, GM_ADDR out, uint32_t m, uint32_t k) \
    {                                                                                              \
    }

#endif
