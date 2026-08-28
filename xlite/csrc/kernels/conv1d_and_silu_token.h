/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Token-row-parallel causal conv1d + SiLU for uniform prefill (P3-A v2).
 *
 * Why a separate kernel: the channel-parallel packed path reads a channel
 * across tokens with a stride-C gather (2B DMA per element into a 32B UB
 * slot, ~6% DMA efficiency) which costs more than the two
 * XliteOpTranspose_1_2 round-trips it replaces at S=512 (measured TTFT
 * +4.4%). The v1 token kernel failed (9.08ms vs 1.07ms for the 3D path)
 * because it reloaded + scalar-transposed the weight block per (row, cb)
 * (~410k scalar ops/core/layer), scalar-collected every 64-lane output
 * (~100k), and strided rows across cores (4x input DMA amplification).
 *
 * v2 design (see .xlite-opt/plans/p3a-token-parallel-rewrite-20260818.md):
 *   - cb-outer / row-inner loop: weight block is loaded and tap-transposed
 *     ONCE per (core, cb) with vgather (lane p byte offset p*K*4), zero
 *     scalar loops.
 *   - contiguous row segment per core + sliding window: a chunk of R rows
 *     is DMA'd in one contiguous burst and converted to a combined
 *     [K-1+R][CB] fp32 buffer; each output row's K taps are UB reads.
 *   - state update is a SEPARATE kernel (conv1d_state_update_*) launched
 *     right after this one by the host wrapper: a cross-core GM race is
 *     unavoidable otherwise (row-parallel cores read state for the window
 *     context while the last-row core rewrites it). The launch boundary is
 *     the only safe inter-core barrier.
 *   - hard V-pipe fences (S round-trip through set_flag/wait_flag) after
 *     every vconv/vgather producer: on this part a bare pipe_barrier(PIPE_V)
 *     does NOT order vconv UB writeback before dependent vector reads
 *     (observed: scattered stale elements in the chunk tail). The proven
 *     channel-parallel kernel never consumes vconv output without an event
 *     fence either.
 *
 * Layouts (uniform prefill, seqLen >= K, K in {1,2,4}, channels % 1024 == 0):
 *   input  [T, C] token-major (T = batch * seqLen)
 *   output [T, C] token-major
 *   state  [B, C, K] (read-only here; updated by the state-update kernel)
 *   weight [C, K]
 *
 * Window semantics (match causal conv1d of concat(state, input)):
 *   concat = [state[K], input[0..S)]
 *   out[i] = dot(concat[i+1 : i+1+K], weight)
 *   => tap k source for batch-relative row i:
 *        i+1+k <  K -> state[i+1+k]
 *        i+1+k >= K -> input row i-K+1+k (same batch)
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

// Hard V-pipe fence: vconv/vgather UB writeback is not covered by a bare
// pipe_barrier(PIPE_V) on this part; route through the scalar pipe with an
// event flag (the pattern the channel-parallel kernel uses around ReadFloat).
#define XLITE_V_FENCE()                       \
    do {                                      \
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);  \
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1); \
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);  \
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1); \
    } while (0)

template <typename Dtype>
class XliteCausalConv1dSiLUToken
{
public:
    static constexpr int kMaxKernel = 4;                      // K in {1,2,4}
    static constexpr int kChanBlock = 1024;                   // channels per block
    static constexpr int kBlock = VECTOR_MAX_NUM_OF_FP32;     // 64 lanes
    static constexpr int kRows = 4;                           // R rows per chunk
    static constexpr int kCombRows = kMaxKernel - 1 + kRows;  // 7
    static constexpr int kGroups = kChanBlock / kBlock;       // 16

    __aicore__ inline XliteCausalConv1dSiLUToken()
    {
    }

    // This kernel only computes conv+SiLU outputs; the conv state update is a
    // separate kernel (conv1d_state_update_*) launched afterwards by the host
    // wrapper (see file header for why an in-kernel update would race).
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                uint32_t batch, uint32_t channels, uint32_t seqLen,
                                uint32_t kernelDim)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        this->state = (__gm__ Dtype *)state;
        this->input = (__gm__ Dtype *)input;
        this->weight = (__gm__ Dtype *)weight;
        this->output = (__gm__ Dtype *)output;
        this->batch = batch;
        this->channels = channels;
        this->seqLen = seqLen;
        this->K = (int)kernelDim;
        this->numChanBlocks = (int)(channels / kChanBlock);

        // UB layout (32B-aligned offsets). Budget for K<=4, CB=1024, R=4:
        //   combined 7*CB*4 = 28KB | raw R*CB*2 = 8KB | w_raw CB*4*2 = 8KB
        //   w_f CB*4*4 = 16KB | w_t 4*CB*4 = 16KB | out R*CB*2 = 8KB
        //   s_raw CB*4*2 = 8KB | s_f CB*4*4 = 16KB | misc ~2KB => ~110KB < 192KB
        uint64_t off = 0;
        combined = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kCombRows * kChanBlock * sizeof(float);
        raw_chunk = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kRows * kChanBlock * sizeof(Dtype);
        w_raw = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kChanBlock * kMaxKernel * sizeof(Dtype);
        w_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kChanBlock * kMaxKernel * sizeof(float);
        w_t = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxKernel * kChanBlock * sizeof(float);
        out_stage = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kRows * kChanBlock * sizeof(Dtype);
        s_raw = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kChanBlock * kMaxKernel * sizeof(Dtype);
        s_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kChanBlock * kMaxKernel * sizeof(float);
        acc_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * sizeof(float);
        calc_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * sizeof(float);
        out_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * sizeof(float);
        off_ramp_k = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += kBlock * sizeof(uint32_t);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
    }

    // Convert nElem contiguous UB Dtype src to float dst (repeat cap 64 = 4096
    // lanes per vconv, same limit as the channel-parallel kernel).
    __aicore__ inline void ConvUbToFloat(__ubuf__ Dtype *src, int nElem, __ubuf__ float *dstF)
    {
        constexpr int kChunk = 64 * kBlock;  // 4096
        int done = 0;
        while (done < nElem) {
            int take = nElem - done;
            if (take > kChunk) {
                take = kChunk;
            }
            if (take >= 2 * kBlock) {
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            } else {
                SetMask(take);
            }
            int repeat = DIV_ROUND_UP(take, kBlock);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(dstF + done, src + done, repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(dstF + done, src + done, repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            done += take;
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    // silu(acc) -> 64 Dtype lanes at dst (acc/calc are 64-lane fp32).
    __aicore__ inline void SiLU64(__ubuf__ Dtype *dst)
    {
        vmuls(calc_buf, acc_buf, (float)-1.0, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(calc_buf, calc_buf, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(calc_buf, calc_buf, (float)1.0, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        if constexpr (std::is_same<Dtype, float>::value) {
            vdiv((__ubuf__ float *)dst, acc_buf, calc_buf, 1, 1, 1, 1, 8, 8, 8);
        } else {
            vdiv(out_f, acc_buf, calc_buf, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(dst, out_f, 1, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(dst, out_f, 1, 1, 1, 4, 8);
            }
        }
        pipe_barrier(PIPE_V);
    }

    // DMA a contiguous GM slice into a UB staging buffer with the proven
    // MTE2 -> V event fence. `drainV` guards staging-buffer reuse.
    template <typename UbT>
    __aicore__ inline void DmaIn(__gm__ Dtype *src, __ubuf__ UbT *dst, uint32_t bytes)
    {
        pipe_barrier(PIPE_V);  // prior V reads of the staging buffer must finish
        CopyGmToUbufAligned(dst, src, bytes);
        pipe_barrier(PIPE_MTE2);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    }

    // Strided row gather: rTake rows of kChanBlock elements each, rows are
    // channels apart in GM (token-major [T,C]), dst bursts packed in UB.
    template <typename UbT>
    __aicore__ inline void DmaInRows(__gm__ Dtype *src, __ubuf__ UbT *dst, int rTake)
    {
        uint16_t lenBurst = (uint16_t)(kChanBlock * (int)sizeof(Dtype) / 32);
        uint16_t srcStride = (uint16_t)((channels - kChanBlock) * (int)sizeof(Dtype) / 32);
        pipe_barrier(PIPE_V);  // prior V reads of the staging buffer must finish
        copy_gm_to_ubuf(dst, src, 0, (uint16_t)rTake, lenBurst, srcStride, 0);
        pipe_barrier(PIPE_MTE2);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    }

    // ---- Weight: load [CB][K] block and tap-transpose to w_t[K][CB] --------
    __aicore__ inline void LoadWeightBlock(int cb)
    {
        __gm__ Dtype *wSrc = weight + (uint64_t)cb * kChanBlock * K;
        if constexpr (std::is_same<Dtype, float>::value) {
            DmaIn(wSrc, w_f, (uint32_t)(kChanBlock * K * sizeof(float)));
        } else {
            DmaIn(wSrc, w_raw, (uint32_t)(kChanBlock * K * sizeof(Dtype)));
            ConvUbToFloat(w_raw, kChanBlock * K, w_f);
            XLITE_V_FENCE();  // vgather must not read w_f until vconv lands
        }
        // vgather lane p reads byte base + p*K*4: gathers w_f[c*K+k] for 64
        // consecutive c starting at g*64, tap k.
        for (int k = 0; k < K; ++k) {
            for (int g = 0; g < kGroups; ++g) {
                uint32_t baseAddr =
                    (uint32_t)((uint64_t)w_f + (uint64_t)(g * kBlock * K + k) * sizeof(float));
                vgather((__ubuf__ uint32_t *)(w_t + k * kChanBlock + g * kBlock), off_ramp_k,
                        baseAddr, 8, 1);
            }
        }
        XLITE_V_FENCE();  // w_t must be materialized before compute vmuls
    }

    // ---- Window context prefill: combined[0..K-2] for sub-segment start s0 -
    __aicore__ inline void PrefillCtx(int b, int s0, int cb)
    {
        bool stateLoaded = false;
        for (int j = 0; j < K - 1; ++j) {
            int v = s0 - (K - 1) + j;  // batch-relative virtual row
            if (v >= 0) {
                __gm__ Dtype *rowPtr =
                    input + ((uint64_t)b * seqLen + v) * channels + (uint64_t)cb * kChanBlock;
                if constexpr (std::is_same<Dtype, float>::value) {
                    DmaIn(rowPtr, combined + j * kChanBlock,
                          (uint32_t)(kChanBlock * sizeof(float)));
                } else {
                    DmaIn(rowPtr, raw_chunk, (uint32_t)(kChanBlock * sizeof(Dtype)));
                    ConvUbToFloat(raw_chunk, kChanBlock, combined + j * kChanBlock);
                }
            } else {
                // state tap: combined[j][c] = state[b][c][K+v]
                if (!stateLoaded) {
                    __gm__ Dtype *sSrc =
                        state + ((uint64_t)b * channels + (uint64_t)cb * kChanBlock) * K;
                    if constexpr (std::is_same<Dtype, float>::value) {
                        DmaIn(sSrc, s_f, (uint32_t)(kChanBlock * K * sizeof(float)));
                    } else {
                        DmaIn(sSrc, s_raw, (uint32_t)(kChanBlock * K * sizeof(Dtype)));
                        ConvUbToFloat(s_raw, kChanBlock * K, s_f);
                        XLITE_V_FENCE();  // vgather must not read s_f until vconv lands
                    }
                    stateLoaded = true;
                }
                int col = K + v;  // in [1, K-1]
                for (int g = 0; g < kGroups; ++g) {
                    uint32_t baseAddr =
                        (uint32_t)((uint64_t)s_f +
                                   (uint64_t)(g * kBlock * K + col) * sizeof(float));
                    vgather((__ubuf__ uint32_t *)(combined + j * kChanBlock + g * kBlock),
                            off_ramp_k, baseAddr, 8, 1);
                }
            }
        }
        XLITE_V_FENCE();  // ctx rows feed compute vmuls
    }

    // ---- Chunk load: rows [chunkStart, chunkStart+rTake) of batch b --------
    __aicore__ inline void LoadChunk(int b, int chunkStart, int rTake, int cb)
    {
        __gm__ Dtype *src =
            input + ((uint64_t)b * seqLen + chunkStart) * channels + (uint64_t)cb * kChanBlock;
        if constexpr (std::is_same<Dtype, float>::value) {
            DmaInRows(src, combined + (K - 1) * kChanBlock, rTake);
        } else {
            DmaInRows(src, raw_chunk, rTake);
            ConvUbToFloat(raw_chunk, rTake * kChanBlock, combined + (K - 1) * kChanBlock);
        }
        XLITE_V_FENCE();  // chunk rows must be materialized before compute
    }

    // ---- Compute rTake rows into out_stage ---------------------------------
    __aicore__ inline void ComputeChunk(int rTake)
    {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // out_stage free (DMA-out drained)
        for (int j = 0; j < rTake; ++j) {
            __ubuf__ float *winBase = combined + j * kChanBlock;
            __ubuf__ Dtype *outRow = out_stage + j * kChanBlock;
            for (int g = 0; g < kGroups; ++g) {
                int sc = g * kBlock;
                // first tap: acc = win[0] * w_t[0]
                vmul(acc_buf, winBase + sc, w_t + sc, 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                for (int k = 1; k < K; ++k) {
                    vmul(calc_buf, winBase + k * kChanBlock + sc, w_t + k * kChanBlock + sc, 1, 1,
                         1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    vadd(acc_buf, acc_buf, calc_buf, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                SiLU64(outRow + sc);
            }
        }
    }

    // ---- Store rTake rows to GM --------------------------------------------
    __aicore__ inline void StoreChunk(int b, int chunkStart, int rTake, int cb)
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        for (int j = 0; j < rTake; ++j) {
            __gm__ Dtype *dst = output + ((uint64_t)b * seqLen + chunkStart + j) * channels +
                                (uint64_t)cb * kChanBlock;
            CopyUbufToGmAligned(dst, out_stage + j * kChanBlock,
                                (uint32_t)(kChanBlock * sizeof(Dtype)));
        }
        pipe_barrier(PIPE_MTE3);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    // ---- Slide context: last K-1 rows -> combined[0..K-2] ------------------
    __aicore__ inline void SlideCtx(int rTake)
    {
        // combined rows [rTake .. rTake+K-2] -> [0 .. K-2]; non-overlapping
        // because a slide only runs when another chunk follows, which implies
        // rTake == kRows = 4 > K-1. (K-1)*CB floats <= 3*1024 = 3072 -> 48
        // repeats.
        int nElem = (K - 1) * kChanBlock;
        __ubuf__ float *src = combined + rTake * kChanBlock;
        int done = 0;
        while (done < nElem) {
            int take = nElem - done;
            if (take > 64 * kBlock) {
                take = 64 * kBlock;
            }
            int repeat = DIV_ROUND_UP(take, kBlock);
            vadds(combined + done, src + done, 0.0f, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            done += take;
        }
        XLITE_V_FENCE();  // slid rows feed the next chunk's compute
    }

    __aicore__ inline void Process()
    {
        if (K < 1 || K > kMaxKernel || (kBlock % K) != 0) {
            return;  // host guarantees K in {1,2,4}
        }
        if (seqLen == 0 || (int)seqLen < K) {
            return;  // token kernel is uniform-prefill only, seqLen >= K
        }
        uint32_t totalRows = batch * seqLen;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        // vgather offset ramps (scalar-written once, then handed to V pipe).
        for (int p = 0; p < kBlock; ++p) {
            off_ramp_k[p] = (uint32_t)(p * K * (int)sizeof(float));
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // prime: no DMA-out in flight yet

        // ---- main row compute: contiguous segment per core ----------------
        uint32_t rowsPerCore = DIV_ROUND_UP(totalRows, blockNum);
        uint32_t rowStart = blockIdx * rowsPerCore;
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > totalRows) {
            rowEnd = totalRows;
        }
        uint32_t row = rowStart;
        while (row < rowEnd) {
            int b = (int)(row / seqLen);
            int s0 = (int)(row % seqLen);
            uint32_t segEnd = (uint32_t)(b + 1) * seqLen;  // global row index
            if (segEnd > rowEnd) {
                segEnd = rowEnd;
            }
            int segRows = (int)segEnd - (int)row;  // rows in this sub-segment
            for (int cb = 0; cb < numChanBlocks; ++cb) {
                LoadWeightBlock(cb);
                PrefillCtx(b, s0, cb);
                int done = 0;
                while (done < segRows) {
                    int rTake = segRows - done;
                    if (rTake > kRows) {
                        rTake = kRows;
                    }
                    LoadChunk(b, s0 + done, rTake, cb);
                    ComputeChunk(rTake);
                    StoreChunk(b, s0 + done, rTake, cb);
                    done += rTake;
                    if (done < segRows) {
                        SlideCtx(rTake);
                    }
                }
            }
            row = segEnd;
        }
        // keep the MTE3->V event balanced: consume the flag set by the last
        // StoreChunk so the event queue drains cleanly at kernel exit.
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

private:
    __gm__ Dtype *state;
    __gm__ Dtype *input;
    __gm__ Dtype *weight;
    __gm__ Dtype *output;

    __ubuf__ float *combined;   // [K-1+R][CB] fp32 sliding window
    __ubuf__ Dtype *raw_chunk;  // [R][CB] Dtype DMA staging
    __ubuf__ Dtype *w_raw;      // [CB][K] Dtype weight staging
    __ubuf__ float *w_f;        // [CB][K] fp32 weight (channel-major)
    __ubuf__ float *w_t;        // [K][CB] fp32 weight (tap-major)
    __ubuf__ Dtype *out_stage;  // [R][CB] Dtype output staging
    __ubuf__ Dtype *s_raw;      // [CB][K] Dtype state staging
    __ubuf__ float *s_f;        // [CB][K] fp32 state
    __ubuf__ float *acc_buf;
    __ubuf__ float *calc_buf;
    __ubuf__ float *out_f;
    __ubuf__ uint32_t *off_ramp_k;  // lane p -> p*K*4 (tap gather)

    int K;
    int numChanBlocks;
    uint32_t batch;
    uint32_t channels;
    uint32_t seqLen;
};

// ---------------------------------------------------------------------------
// Separate kernel: conv state update (state[b][c][:] = last K input rows of
// batch b). Must be its own launch: row-parallel conv cores read state GM for
// the window context, so an in-kernel update would race them; the launch
// boundary is the inter-core barrier. Units (b, cb) are round-robin over cores.
// ---------------------------------------------------------------------------
template <typename Dtype>
class XliteConvStateUpdate
{
public:
    static constexpr int kMaxKernel = 4;
    static constexpr int kChanBlock = 1024;
    static constexpr int kBlock = VECTOR_MAX_NUM_OF_FP32;

    __aicore__ inline XliteConvStateUpdate()
    {
    }

    __aicore__ inline void Init(GM_ADDR state, GM_ADDR input, uint32_t batch, uint32_t channels,
                                uint32_t seqLen, uint32_t kernelDim)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        this->state = (__gm__ Dtype *)state;
        this->input = (__gm__ Dtype *)input;
        this->batch = batch;
        this->channels = channels;
        this->seqLen = seqLen;
        this->K = (int)kernelDim;
        this->numChanBlocks = (int)(channels / kChanBlock);

        uint64_t off = 0;
        rows_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxKernel * kChanBlock * sizeof(float);  // 16KB
        raw = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kChanBlock * sizeof(Dtype);  // 2KB (per-row staging)
        tile_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kChanBlock * kMaxKernel * sizeof(float);  // 16KB
        tile_raw = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kChanBlock * kMaxKernel * sizeof(Dtype);  // 8KB
        off_tile = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += kBlock * sizeof(uint32_t);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
    }

    __aicore__ inline void Process()
    {
        if (K < 1 || K > kMaxKernel || (kBlock % K) != 0 || (int)seqLen < K) {
            return;
        }
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        // lane p (group g) reads rows_f[p%K][g*(64/K) + p/K]
        for (int p = 0; p < kBlock; ++p) {
            off_tile[p] = (uint32_t)((p % K) * kChanBlock * (int)sizeof(float) +
                                     (p / K) * (int)sizeof(float));
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);

        int units = (int)batch * numChanBlocks;
        for (int u = (int)blockIdx; u < units; u += (int)blockNum) {
            int b = u / numChanBlocks;
            int cb = u % numChanBlocks;
            // Load K tail rows into rows_f[j].
            for (int j = 0; j < K; ++j) {
                __gm__ Dtype *rowPtr = input + ((uint64_t)b * seqLen + seqLen - K + j) * channels +
                                       (uint64_t)cb * kChanBlock;
                pipe_barrier(PIPE_V);
                CopyGmToUbufAligned(raw, rowPtr, (uint32_t)(kChanBlock * sizeof(Dtype)));
                pipe_barrier(PIPE_MTE2);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
                if constexpr (std::is_same<Dtype, float>::value) {
                    // rows_f[j] = raw (already float): plain vector copy
                    int repeat = DIV_ROUND_UP(kChanBlock, kBlock);
                    vadds(rows_f + j * kChanBlock, (__ubuf__ float *)raw, 0.0f, repeat, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                } else {
                    int repeat = DIV_ROUND_UP(kChanBlock, kBlock);
                    if constexpr (std::is_same<Dtype, float16_t>::value) {
                        vconv_f162f32(rows_f + j * kChanBlock, raw, repeat, 1, 1, 8, 4);
                    } else {
                        vconv_bf162f32(rows_f + j * kChanBlock, raw, repeat, 1, 1, 8, 4);
                    }
                    pipe_barrier(PIPE_V);
                }
            }
            XLITE_V_FENCE();  // vconv writeback must land before the vgather reads
            // Pack [K][CB] rows into the [CB][K] GM tile with a reverse vgather.
            int chPerGather = kBlock / K;
            int nGather = kChanBlock / chPerGather;
            for (int g = 0; g < nGather; ++g) {
                uint32_t baseAddr =
                    (uint32_t)((uint64_t)rows_f + (uint64_t)(g * chPerGather) * sizeof(float));
                vgather((__ubuf__ uint32_t *)(tile_f + g * kBlock), off_tile, baseAddr, 8, 1);
            }
            XLITE_V_FENCE();
            __gm__ Dtype *sDst = state + ((uint64_t)b * channels + (uint64_t)cb * kChanBlock) * K;
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            if constexpr (std::is_same<Dtype, float>::value) {
                CopyUbufToGmAligned(sDst, tile_f, (uint32_t)(kChanBlock * K * sizeof(float)));
            } else {
                // tile_raw = bf16(tile_f)
                int total = kChanBlock * K;
                int done = 0;
                while (done < total) {
                    int take = total - done;
                    if (take > 64 * kBlock) {
                        take = 64 * kBlock;
                    }
                    if (take >= 2 * kBlock) {
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    } else {
                        SetMask(take);
                    }
                    int repeat = DIV_ROUND_UP(take, kBlock);
                    if constexpr (std::is_same<Dtype, float16_t>::value) {
                        vconv_f322f16(tile_raw + done, tile_f + done, repeat, 1, 1, 4, 8);
                    } else {
                        vconv_f322bf16r(tile_raw + done, tile_f + done, repeat, 1, 1, 4, 8);
                    }
                    pipe_barrier(PIPE_V);
                    done += take;
                }
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                CopyUbufToGmAligned(sDst, tile_raw, (uint32_t)(kChanBlock * K * sizeof(Dtype)));
            }
            pipe_barrier(PIPE_MTE3);
        }
    }

private:
    __gm__ Dtype *state;
    __gm__ Dtype *input;
    int K;
    int numChanBlocks;
    uint32_t batch;
    uint32_t channels;
    uint32_t seqLen;
    __ubuf__ float *rows_f;    // [K][CB] fp32 tail rows
    __ubuf__ Dtype *raw;       // [CB] Dtype staging
    __ubuf__ float *tile_f;    // [CB][K] fp32 packed tile
    __ubuf__ Dtype *tile_raw;  // [CB][K] Dtype packed tile
    __ubuf__ uint32_t *off_tile;
};

#define CONV1D_AND_SILU_TOKEN_FUNC_DEFINE(dtype)                                          \
    extern "C" __global__ __aicore__ void conv1d_and_silu_token_##dtype(                  \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch,     \
        uint32_t channels, uint32_t seqLen, uint32_t kernelDim)                           \
    {                                                                                     \
        XliteCausalConv1dSiLUToken<dtype> op;                                             \
        op.Init(state, input, weight, output, batch, channels, seqLen, kernelDim);        \
        op.Process();                                                                     \
    }                                                                                     \
    extern "C" __global__ __aicore__ void conv1d_state_update_##dtype(                    \
        GM_ADDR state, GM_ADDR input, uint32_t batch, uint32_t channels, uint32_t seqLen, \
        uint32_t kernelDim)                                                               \
    {                                                                                     \
        XliteConvStateUpdate<dtype> op;                                                   \
        op.Init(state, input, batch, channels, seqLen, kernelDim);                        \
        op.Process();                                                                     \
    }
#else
#define CONV1D_AND_SILU_TOKEN_FUNC_DEFINE(dtype)                                          \
    extern "C" __global__ __aicore__ void conv1d_and_silu_token_##dtype(                  \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch,     \
        uint32_t channels, uint32_t seqLen, uint32_t kernelDim)                           \
    {                                                                                     \
        (void)state;                                                                      \
        (void)input;                                                                      \
        (void)weight;                                                                     \
        (void)output;                                                                     \
        (void)batch;                                                                      \
        (void)channels;                                                                   \
        (void)seqLen;                                                                     \
        (void)kernelDim;                                                                  \
    }                                                                                     \
    extern "C" __global__ __aicore__ void conv1d_state_update_##dtype(                    \
        GM_ADDR state, GM_ADDR input, uint32_t batch, uint32_t channels, uint32_t seqLen, \
        uint32_t kernelDim)                                                               \
    {                                                                                     \
        (void)state;                                                                      \
        (void)input;                                                                      \
        (void)batch;                                                                      \
        (void)channels;                                                                   \
        (void)seqLen;                                                                     \
        (void)kernelDim;                                                                  \
    }
#endif
