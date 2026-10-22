/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "quant.h"
#include "kernel_param.h"

#ifdef __DAV_C220_VEC__

// out = int8(x * 127 / absmax(x_row)), scale = absmax(x_row) / 127
// dynamic per-token quantization, [m, k] activation; dtype = float16_t / bfloat16_t
template <typename dtype>
class QuantDyn
{
public:
    __aicore__ inline QuantDyn()
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
    __aicore__ inline int64_t TaskTilesInit(GM_ADDR x, GM_ADDR scales, GM_ADDR z,
                                            GM_ADDR pnumTokens, uint64_t m, uint64_t k,
                                            uint64_t taskRatio = 1)
    {
        this->xGm = reinterpret_cast<__gm__ dtype *>(x);
        this->zGm = reinterpret_cast<__gm__ int8_t *>(z);
        this->scalesGm = reinterpret_cast<__gm__ float *>(scales);
        if (pnumTokens != nullptr) {
            uint32_t pnumTokensValue = *((__gm__ uint32_t *)pnumTokens);
            this->m = pnumTokensValue < m ? pnumTokensValue : m;
        } else {
            this->m = m;
        }
        this->k = k;
        this->tileRows = DIV_ROUND_UP(this->m, GetBlockNum() * taskRatio);

        constexpr uint32_t align = VECTOR_REPEAT_BYTESIZE / sizeof(dtype);
        uint32_t maxOneRowKTile = ROUND_DOWN((UB_SIZE - 32) / 16, align);
        this->kLoop = DIV_ROUND_UP((uint32_t)k, maxOneRowKTile);
        this->kPad = ROUND_UP(DIV_ROUND_UP((uint32_t)k, this->kLoop), align);
        // rowMax must cover rowAbs (rows * 8 floats = 32*R bytes) and
        // scaleStage (2*R floats = 8*R bytes), totalling 40*R bytes.
        uint32_t rowMax = MAX(1, (UB_SIZE - 4 * this->kPad) / (12 * this->kPad + 40));
        // repeatMax means the max rows AIV can process in one repeat
        uint32_t repeatMax = VECTOR_MAX_REPEAT * VECTOR_MAX_NUM_OF_FP32 / this->kPad;
        this->rows = MIN(MIN(rowMax, repeatMax), (uint32_t)this->tileRows);

        this->x1 = reinterpret_cast<__ubuf__ dtype *>((uintptr_t)0);
        this->x2 = this->x1 + this->rows * this->kPad;
        this->xf32 = reinterpret_cast<__ubuf__ float *>(this->x2 + this->rows * this->kPad);
        this->z1 = reinterpret_cast<__ubuf__ int8_t *>(this->xf32 + this->rows * this->kPad);
        this->z2 = this->z1 + this->rows * this->kPad;
        this->xAbs = reinterpret_cast<__ubuf__ float *>(this->z2 + this->rows * this->kPad);
        this->xf16 = reinterpret_cast<__ubuf__ half *>(this->xAbs + this->kPad);
        // vmax instruction requires rowAbs buffer 32B align
        this->rowAbs = reinterpret_cast<__ubuf__ float *>(this->xf16 + this->rows * this->kPad);
        this->scaleStage = reinterpret_cast<__ubuf__ float *>(this->rowAbs + this->rows * 8);
        assert((uint64_t)(this->scaleStage + 2 * this->rows) <= UB_SIZE);

        return static_cast<int64_t>(DIV_ROUND_UP(this->m, this->tileRows));
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
        auto *xAbs = this->xAbs;
        auto *xf16 = this->xf16;
        auto *rowAbs = this->rowAbs;

        const uint32_t end = start + rowCount;
        int eId = 0;
        int xId = 0;

        for (uint32_t base = start; base < end; base += rows) {
            uint32_t r = MIN(rows, end - base);

            // ---- phase 1: walk the k chunks, merging each row's chunk
            // absmax into its 32B slot
            for (uint32_t loop = 0; loop < kLoop; loop++) {
                uint32_t kOffset = loop * kPad;
                uint32_t kSize = MIN(kPad, k - kOffset);
                // compute width is kPad, not ROUND_UP(kSize): the whole-group
                // converts below fold [r, kPad] into one contiguous run, which
                // only matches the row-strided layout when every chunk uses
                // the same kPad width (a kSizePad != kPad tail chunk shifts
                // row 1+ by the difference and reads stale tails). The stale
                // lanes beyond kSize never reach ReduceMax (real kSize) or
                // the stores (real kSize bytes).
                uint32_t kRepeats = kPad / VECTOR_MAX_NUM_OF_FP32;

                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + xId);  // xBufs[xId] free from prev convert
                // one burst per row: GM rows stride k, UB rows padded to kPad
                copy_gm_to_ubuf(xBufs[xId], this->xGm + (uint64_t)base * k + kOffset, 0, r,
                                kSize * sizeof(dtype) / 32, (k - kSize) * sizeof(dtype) / 32,
                                (kPad - kSize) * sizeof(dtype) / 32);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + xId);

                // whole-group FP16/BF16 -> FP32; the slot is dead right after
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + xId);
                convert_input<dtype>(xf32, xBufs[xId], r * kRepeats);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + xId);
                xId = 1 - xId;

                for (uint32_t i = 0; i < r; i++) {
                    vabs(xAbs, xf32 + i * kPad, kRepeats, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    ReduceMax(xAbs, xAbs, kSize);
                    // move the scalar result to rowAbs[i]: ReduceMax's dst needs a k-sized
                    // scratch, so reduce in place; chunk 0 copies lane 0 into the slot with
                    // a masked vmax, later chunks merge into it
                    SetMask(1);
                    if (loop == 0) {
                        vmax(rowAbs + i * 8, xAbs, xAbs, 1, 1, 1, 1, 8, 8, 8);
                    } else {
                        vmax(rowAbs + i * 8, rowAbs + i * 8, xAbs, 1, 1, 1, 1, 8, 8, 8);
                    }
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    pipe_barrier(PIPE_V);
                }
            }
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);  // absmax table visible to S
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            // S: all scales at once; stage [0,r) = scale (absmax/127),
            // [rows, rows+r) = 127/absmax for phase 2 re-scale
            for (uint32_t i = 0; i < r; i++) {
                float absmax = float(*(rowAbs + i * 8));
                *(scaleStage + i) = absmax / float(127);
                *(scaleStage + this->rows + i) = float(127) / absmax;
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm_align_b32(this->scalesGm + base, scaleStage, 0, 1, r * sizeof(float), 0,
                                      0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);  // also gates V issue: scaleStage consumed

            // ---- phase 2: rescale and emit, chunk by chunk. KLoop == 1
            // keeps xf32 resident from phase 1 — no re-read; wider k
            // reloads each chunk.
            for (uint32_t loop = 0; loop < kLoop; loop++) {
                uint32_t kOffset = loop * kPad;
                uint32_t kSize = MIN(kPad, k - kOffset);
                // kPad compute width, like phase 1 (same layout invariant)
                uint32_t kRepeats = kPad / VECTOR_MAX_NUM_OF_FP32;

                if (kLoop > 1) {
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + xId);
                    // one burst per row: GM rows stride k, UB rows padded to kPad
                    copy_gm_to_ubuf(xBufs[xId], this->xGm + (uint64_t)base * k + kOffset, 0, r,
                                    kSize * sizeof(dtype) / 32, (k - kSize) * sizeof(dtype) / 32,
                                    (kPad - kSize) * sizeof(dtype) / 32);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + xId);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + xId);
                    convert_input<dtype>(xf32, xBufs[xId], r * kRepeats);
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + xId);
                    xId = 1 - xId;
                }

                // per-row re-scale from the (resident or reloaded) xf32 chunk, then
                // three-step fp32→int16→fp16→int8 chain matching dynamic_quant's
                // symmetric path: RINT for fp32→int16, ROUND(away) for int16→fp16,
                // TRUNC for fp16→int8. The int16 intermediate reuses xf16 in-place
                // (int16 and half are both 2 bytes), so the UB layout is unchanged.
                for (uint32_t i = 0; i < r; i++) {
                    float scaleRec = float(*(scaleStage + this->rows + i));
                    vmuls(xf32 + i * kPad, xf32 + i * kPad, scaleRec, kRepeats, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                vconv_f322s16r(reinterpret_cast<__ubuf__ int16_t *>(xf16), xf32, r * kRepeats, 1, 1,
                               4, 8);
                pipe_barrier(PIPE_V);
                vconv_s162f16a(xf16, reinterpret_cast<__ubuf__ int16_t *>(xf16), r * kRepeats, 1, 1,
                               4, 4);
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eId);  // zBufs[eId] free from prev MTE3
                vconv_f162s8z(zBufs[eId], xf16, r * kPad / VECTOR_MAX_NUM_OF_FP16, 1, 1, 4, 8);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eId);  // zBufs[eId] ready for MTE3

                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + eId);
                // z chunk [r, kSize] UB -> GM (2D burst, mirrored gaps)
                copy_ubuf_to_gm(this->zGm + (uint64_t)base * k + kOffset, zBufs[eId], 0, r,
                                kSize / 32, (kPad - kSize) / 32, (k - kSize) / 32);

                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + eId);
                eId = 1 - eId;
            }
        }
    }

    // Thin dispatcher: bind + derive tile geometry (pnumTokens is read once
    // inside TaskTilesInit), then grid-stride the tiles. The host launches
    // min(m, aivNum) blocks, so each block runs at most one tile.
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR scale, GM_ADDR out, GM_ADDR pnumTokens,
                               uint32_t m, uint32_t k)
    {
        int64_t tiles = TaskTilesInit(x, scale, out, pnumTokens, m, k);
        SetFlags();
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
    }

    __aicore__ inline void WaitFlags()
    {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
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
    __gm__ dtype *xGm = nullptr;
    __gm__ int8_t *zGm = nullptr;
    __gm__ float *scalesGm = nullptr;
    __ubuf__ dtype *x1 = nullptr;
    __ubuf__ dtype *x2 = nullptr;
    __ubuf__ float *xf32 = nullptr;
    __ubuf__ int8_t *z1 = nullptr;
    __ubuf__ int8_t *z2 = nullptr;
    __ubuf__ float *xAbs = nullptr;
    __ubuf__ half *xf16 = nullptr;
    __ubuf__ float *rowAbs = nullptr;
    __ubuf__ float *scaleStage = nullptr;
};

#define QUANT_DYN_FUNC_DEFINE(dtype)                                                         \
    extern "C" __global__ __aicore__ void quant_dynamic_##dtype(                             \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t k) \
    {                                                                                        \
        QuantDyn<dtype> op;                                                                  \
        op.Init();                                                                           \
        op.Run(in, scale, out, pnum_tokens, m, k);                                           \
    }
#else
#define QUANT_DYN_FUNC_DEFINE(dtype)                                                         \
    extern "C" __global__ __aicore__ void quant_dynamic_##dtype(                             \
        GM_ADDR in, GM_ADDR scale, GM_ADDR out, GM_ADDR pnum_tokens, uint32_t m, uint32_t k) \
    {                                                                                        \
    }

#endif
