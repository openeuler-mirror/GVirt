/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Fused causal conv1d + SiLU + optional conv_state update.
 * Semantics (match ConcatCol + Conv1dAndSiLU / Python):
 *   concat = cat(state[K], input[S])
 *   out[i] = SiLU(dot(concat[i+1 : i+1+K], weight))
 *   if updateState: state = concat[..., -K:]
 *
 * No GM workspace concat. State/input are loaded into aligned UB float
 * buffers.
 *
 * Layouts:
 *   Uniform (seqLen != 0): input/output [B, S, C] (channel is the inner,
 *     contiguous dim), state [B, C, K], weight [C, 1, K].
 *   Packed (seqLen == 0): input/output [T, C] token-major, state still [B, C, K].
 *
 * The 64-lane (vectorized) dimension is the CHANNEL, not the sequence. Each tap
 * of the K-tap convolution is a contiguous, 32B-aligned 64-channel vector load
 * (vmul + vadd), instead of the per-tap vgather the [B,C,S] layout required.
 * Consuming/producing [B,S,C] directly also removes the two [B,S,C]<->[B,C,S]
 * Transpose passes around the kernel.
 *
 * Limits: kernelDim <= 16, seqLen (or per-request lens) <= 4096.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#ifdef __DAV_C220_VEC__

template <typename Dtype>
class XliteCausalConv1dSiLU
{
public:
    static constexpr int kMaxKernel = 16;
    static constexpr int kMaxInputF = 4096;
    static constexpr int kMaxBatchMeta = 256;
    static constexpr int kBlock = VECTOR_MAX_NUM_OF_FP32;  // 64 lanes
    static constexpr int kTile = 128;
    static constexpr int kWin = kMaxKernel - 1 + kTile;

    __aicore__ inline XliteCausalConv1dSiLU()
    {
    }

    __aicore__ inline void Init(GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                uint32_t batch, uint32_t channels, uint32_t seqLen,
                                uint32_t kernelDim, uint32_t updateState, GM_ADDR queryStartLoc,
                                GM_ADDR queryLens)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        this->state = (__gm__ Dtype *)state;
        this->input = (__gm__ Dtype *)input;
        this->weight = (__gm__ Dtype *)weight;
        this->output = (__gm__ Dtype *)output;
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->batch = batch;
        this->channels = channels;
        this->seqLen = seqLen;
        this->kernelDim = kernelDim;
        this->updateState = updateState;

        uint64_t off = 0;
        // Aligned staging for GM<->UB dtype conversion round trips.
        stage_buf = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kMaxInputF * sizeof(Dtype);

        w_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * kMaxKernel * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        w_reorg = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * kMaxKernel * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        state_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * kMaxKernel * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        state_reorg = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * kMaxKernel * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }

        if constexpr (std::is_same<Dtype, float>::value) {
            win_raw = nullptr;
        } else {
            win_raw = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
            off += kWin * kBlock * sizeof(Dtype);
            if (off % 32 != 0) {
                off = (off + 31) / 32 * 32;
            }
        }
        window_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kWin * kBlock * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }

        state_win = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxKernel * kBlock * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        new_state_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxKernel * kBlock * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }

        acc_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        calc_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        out_tile = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kTile * kBlock * sizeof(Dtype);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        meta_start = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
        off += kMaxBatchMeta * sizeof(int32_t);
        meta_lens = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)off);
    }

    __aicore__ inline void SetLaneMask(int w)
    {
        if (w >= kBlock) {
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        } else {
            SetMask(w);
        }
    }

    // pipe_barrier(PIPE_X) alone does NOT order UB data produced/consumed by
    // *different* pipes (MTE2/V/MTE3/S); explicit event flags are required.
    // EVENT_ID0: MTE2 <-> V and V <-> MTE3 for the window/out_tile pipeline.
    // EVENT_ID1: scalar (S) pipe fences.
    // EVENT_ID2: stage_buf MTE2 <-> V inside LoadGmToFloat/StoreFloatToGm.
    __aicore__ inline void FlagMTE2V()
    {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }
    __aicore__ inline void FlagVMTE2()
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    __aicore__ inline void FlagVMTE3()
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }
    __aicore__ inline void FlagMTE3V()
    {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    __aicore__ inline void FlagMTE2S()
    {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
    }
    __aicore__ inline void FlagVS()
    {
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    }
    __aicore__ inline void FlagSV()
    {
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    }
    __aicore__ inline void FlagSMTE3()
    {
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    }

    __aicore__ inline void SiLU(__ubuf__ Dtype *dst)
    {
        vmuls(calc_buf, acc_buf, (float)-1.0, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(calc_buf, calc_buf, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(calc_buf, calc_buf, (float)1.0, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        if constexpr (std::is_same<Dtype, float>::value) {
            vdiv(dst, acc_buf, calc_buf, 1, 1, 1, 1, 8, 8, 8);
        } else {
            vdiv(calc_buf, acc_buf, calc_buf, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(dst, calc_buf, 1, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(dst, calc_buf, 1, 1, 1, 4, 8);
            }
        }
        pipe_barrier(PIPE_V);
    }

    // Load nElem dtype values from GM into dstF (float). One round trip per
    // chunk; chunks up to kMaxInputF keep the float dst 32B-aligned.
    __aicore__ inline void LoadGmToFloat(__gm__ Dtype *src, int nElem, __ubuf__ float *dstF)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(dstF, src, static_cast<uint32_t>(nElem * sizeof(float)));
            pipe_barrier(PIPE_MTE2);
            return;
        }
        constexpr int kChunk = kMaxInputF;  // 8KB stage_buf: full input per round trip
        int done = 0;
        while (done < nElem) {
            int take = nElem - done;
            if (take > kChunk) {
                take = kChunk;
            }
            uint32_t bytes = static_cast<uint32_t>(take * (int)sizeof(Dtype));
            CopyGmToUbufAligned(stage_buf, src + done, bytes);
            pipe_barrier(PIPE_MTE2);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            if (take >= 2 * VECTOR_MAX_NUM_OF_FP32) {
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            } else {
                SetMask(take);
            }
            int repeat = DIV_ROUND_UP(take, VECTOR_MAX_NUM_OF_FP32);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(dstF + done, stage_buf, repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(dstF + done, stage_buf, repeat, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            done += take;
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void StoreFloatToGm(__ubuf__ float *srcF, __gm__ Dtype *dst, int nElem)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyUbufToGmAligned(dst, srcF, static_cast<uint32_t>(nElem * sizeof(float)));
            pipe_barrier(PIPE_MTE3);
            return;
        }
        constexpr int kChunk = kMaxInputF;
        int done = 0;
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        while (done < nElem) {
            int take = nElem - done;
            if (take > kChunk) {
                take = kChunk;
            }
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            if (take >= 2 * VECTOR_MAX_NUM_OF_FP32) {
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            } else {
                SetMask(take);
            }
            int repeat = DIV_ROUND_UP(take, VECTOR_MAX_NUM_OF_FP32);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(stage_buf, srcF + done, repeat, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(stage_buf, srcF + done, repeat, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            CopyUbufToGmAligned(dst + done, stage_buf,
                                static_cast<uint32_t>(take * (int)sizeof(Dtype)));
            pipe_barrier(PIPE_MTE3);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            done += take;
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    // Load a [nPos, w] tile of channel-block c0 (w channels starting at c0) from
    // GM positions [tokenBase+loPos, tokenBase+loPos+nPos) into UB at row dstIdx.
    // Each UB row is kBlock elements wide. When both the row and the channel
    // width are 32B-aligned, use one multi-burst DMA; otherwise fall back to a
    // per-position copy.
    __aicore__ inline void LoadTileInto(__ubuf__ Dtype *dst, int tokenBase, int c0, int loPos,
                                        int nPos, int dstIdx)
    {
        int C = (int)channels;
        int w = MIN(kBlock, C - c0);
        __gm__ Dtype *src = input + ((tokenBase + loPos) * C + c0);
        int burstBytes = w * (int)sizeof(Dtype);
        int burstBlocks = burstBytes / BLOCK_SIZE;
        int rowBytes = C * (int)sizeof(Dtype);
        if (w == kBlock && burstBlocks * BLOCK_SIZE == burstBytes && rowBytes % BLOCK_SIZE == 0) {
            uint64_t cfg =
                __set_dmi_config(0, nPos, burstBlocks, rowBytes / BLOCK_SIZE - burstBlocks, 0);
            copy_gm_to_ubuf(dst + dstIdx * kBlock, src, cfg);
        } else {
            for (int i = 0; i < nPos; i++) {
                CopyGmToUbufAligned(dst + (dstIdx + i) * kBlock, src + i * C,
                                    static_cast<uint32_t>(burstBytes));
            }
        }
        pipe_barrier(PIPE_MTE2);
    }

    __aicore__ inline void ConvertTile(int nPos, int dstIdx)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            return;
        }
        int off = dstIdx * kBlock;
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        if constexpr (std::is_same<Dtype, float16_t>::value) {
            vconv_f162f32(window_f + off, win_raw + off, nPos, 1, 1, 8, 4);
        } else {
            vconv_bf162f32(window_f + off, win_raw + off, nPos, 1, 1, 8, 4);
        }
        pipe_barrier(PIPE_V);
    }

    // Load [w, K] weights and re-layout them to [K, kBlock] so tap j lives at a
    // contiguous 64-lane vector (w_reorg + j*kBlock).
    __aicore__ inline void LoadWeights(int c0, int w)
    {
        int K = (int)kernelDim;
        LoadGmToFloat(weight + c0 * K, w * K, w_f);
        if constexpr (std::is_same<Dtype, float>::value) {
            FlagMTE2S();
        } else {
            FlagVS();
        }
        for (int l = 0; l < w; l++) {
            for (int j = 0; j < K; j++) {
                w_reorg[j * kBlock + l] = w_f[l * K + j];
            }
        }
        FlagSV();
    }

    // Load [w, K] state and re-layout it to [K, kBlock].
    __aicore__ inline void LoadState(int b, int c0, int w)
    {
        int K = (int)kernelDim;
        int stateBase = (b * (int)channels + c0) * K;
        LoadGmToFloat(state + stateBase, w * K, state_f);
        if constexpr (std::is_same<Dtype, float>::value) {
            FlagMTE2S();
        } else {
            FlagVS();
        }
        for (int l = 0; l < w; l++) {
            for (int p = 0; p < K; p++) {
                state_reorg[p * kBlock + l] = state_f[l * K + p];
            }
        }
        FlagSV();
    }

    // Compute one output position (channel block c0, w lanes) and store the
    // SiLU result into out_tile row (s - s0).
    __aicore__ inline void ComputePosition(int s, int s0, int w)
    {
        int K = (int)kernelDim;
        SetLaneMask(w);
        vector_dup(acc_buf, 0.0f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        for (int j = 0; j < K; j++) {
            int p = s + 1 + j;
            __ubuf__ float *tap =
                (p < K) ? (state_reorg + p * kBlock) : (window_f + (p - s0 - 1) * kBlock);
            vmul(calc_buf, tap, w_reorg + j * kBlock, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(acc_buf, acc_buf, calc_buf, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        SiLU(out_tile + (s - s0) * kBlock);
    }

    // Store out_tile rows [0, nPos) to GM positions [tokenBase+s0, tokenBase+s0+nPos)
    // with a channel-stride C. Bulk multi-burst DMA when the full block is used.
    __aicore__ inline void StoreTile(int tokenBase, int c0, int s0, int nPos, int w)
    {
        int C = (int)channels;
        __gm__ Dtype *dst = output + (tokenBase + s0) * C + c0;
        int burstBytes = w * (int)sizeof(Dtype);
        int burstBlocks = burstBytes / BLOCK_SIZE;
        int rowBlocks = C * (int)sizeof(Dtype) / BLOCK_SIZE;
        if (w == kBlock && burstBlocks * BLOCK_SIZE == burstBytes &&
            rowBlocks * BLOCK_SIZE == C * (int)sizeof(Dtype)) {
            uint64_t cfg = __set_dmi_config(0, nPos, burstBlocks, 0, rowBlocks - burstBlocks);
            copy_ubuf_to_gm(dst, out_tile, cfg);
        } else {
            for (int s = 0; s < nPos; s++) {
                CopyUbufToGmAligned(dst + s * C, out_tile + s * kBlock,
                                    static_cast<uint32_t>(burstBytes));
            }
        }
        pipe_barrier(PIPE_MTE3);
    }

    __aicore__ inline void ProcessSequence(int tokenBase, int c0, int w, int S)
    {
        int K = (int)kernelDim;
        for (int s0 = 0; s0 < S; s0 += kTile) {
            int loPos = (s0 == 0) ? 0 : (s0 - K + 1);
            int hiPos = MIN(s0 + kTile - 1, S - 1);
            if (hiPos < loPos) {
                break;
            }
            int nPos = hiPos - loPos + 1;
            int dstIdx = (s0 == 0) ? (K - 1) : 0;

            if constexpr (std::is_same<Dtype, float>::value) {
                LoadTileInto(window_f, tokenBase, c0, loPos, nPos, dstIdx);
            } else {
                LoadTileInto(win_raw, tokenBase, c0, loPos, nPos, dstIdx);
            }
            FlagMTE2V();
            ConvertTile(nPos, dstIdx);
            FlagVMTE2();

            int sEnd = MIN(s0 + kTile, S);
            for (int s = s0; s < sEnd; s++) {
                ComputePosition(s, s0, w);
            }
            FlagVMTE3();
            StoreTile(tokenBase, c0, s0, sEnd - s0, w);
            FlagMTE3V();
        }
    }

    // Load the trailing input window used to derive the new state.
    __aicore__ inline void LoadStateWin(int tokenBase, int c0, int S)
    {
        int K = (int)kernelDim;
        int firstPos = MAX(0, S - K);
        int nPos = S - firstPos;
        if (nPos <= 0) {
            return;
        }
        if constexpr (std::is_same<Dtype, float>::value) {
            LoadTileInto(state_win, tokenBase, c0, firstPos, nPos, 0);
        } else {
            LoadTileInto(win_raw, tokenBase, c0, firstPos, nPos, 0);
            FlagMTE2V();
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(state_win, win_raw, nPos, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(state_win, win_raw, nPos, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            FlagVMTE2();
        }
    }

    __aicore__ inline void WriteBackState(int b, int tokenBase, int c0, int w, int S)
    {
        int K = (int)kernelDim;
        int C = (int)channels;
        int firstPos = MAX(0, S - K);

        LoadStateWin(tokenBase, c0, S);
        if constexpr (std::is_same<Dtype, float>::value) {
            FlagMTE2S();
        } else {
            FlagVS();
        }
        for (int t = 0; t < K; t++) {
            int concatIdx = S + t;
            for (int l = 0; l < w; l++) {
                float v;
                if (concatIdx < K) {
                    v = state_reorg[concatIdx * kBlock + l];
                } else {
                    v = state_win[(concatIdx - K - firstPos) * kBlock + l];
                }
                new_state_f[l * K + t] = v;
            }
        }
        if constexpr (std::is_same<Dtype, float>::value) {
            FlagSMTE3();
        } else {
            FlagSV();
        }
        StoreFloatToGm(new_state_f, state + (b * C + c0) * K, w * K);
    }

    __aicore__ inline bool Packed()
    {
        // Host sets seqLen=0 for packed mixed-length. GM pointer != nullptr is
        // not reliable on device (CANN may pass a non-null dummy).
        return seqLen == 0;
    }

    __aicore__ inline void Process()
    {
        int K = (int)kernelDim;
        if (K > kMaxKernel) {
            return;
        }
        bool packed = Packed();
        if (!packed && (int)seqLen > kMaxInputF) {
            return;
        }
        if (packed) {
            uint32_t n = batch < kMaxBatchMeta ? batch : kMaxBatchMeta;
            CopyGmToUbufAligned(meta_start, queryStartLoc, n * sizeof(int32_t));
            pipe_barrier(PIPE_MTE2);
            CopyGmToUbufAligned(meta_lens, queryLens, n * sizeof(int32_t));
            pipe_barrier(PIPE_MTE2);
            FlagMTE2S();
        }

        int C = (int)channels;
        int nBlocks = DIV_ROUND_UP(C, kBlock);
        for (int cb = 0; cb < nBlocks; cb++) {
            if (cb % GetBlockNum() != GetBlockIdx()) {
                continue;
            }
            int c0 = cb * kBlock;
            int w = MIN(kBlock, C - c0);

            LoadWeights(c0, w);
            for (int b = 0; b < (int)batch; b++) {
                int S;
                int tokenBase;
                if (packed) {
                    S = (int)meta_lens[b];
                    tokenBase = (int)meta_start[b];
                } else {
                    S = (int)seqLen;
                    tokenBase = b * S;
                }
                if (S <= 0 || S > kMaxInputF) {
                    continue;
                }
                LoadState(b, c0, w);
                ProcessSequence(tokenBase, c0, w, S);
                if (updateState) {
                    WriteBackState(b, tokenBase, c0, w, S);
                }
                pipe_barrier(PIPE_ALL);
            }
        }
    }

private:
    __gm__ Dtype *state;
    __gm__ Dtype *input;
    __gm__ Dtype *weight;
    __gm__ Dtype *output;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;

    __ubuf__ Dtype *stage_buf;
    __ubuf__ float *w_f;
    __ubuf__ float *w_reorg;
    __ubuf__ float *state_f;
    __ubuf__ float *state_reorg;
    __ubuf__ Dtype *win_raw;
    __ubuf__ float *window_f;
    __ubuf__ float *state_win;
    __ubuf__ float *new_state_f;
    __ubuf__ float *acc_buf;
    __ubuf__ float *calc_buf;
    __ubuf__ Dtype *out_tile;
    __ubuf__ int32_t *meta_start;
    __ubuf__ int32_t *meta_lens;

    uint32_t batch;
    uint32_t channels;
    uint32_t seqLen;
    uint32_t kernelDim;
    uint32_t updateState;
};

#define CONV1D_AND_SILU_FUNC_DEFINE(dtype)                                                     \
    extern "C" __global__ __aicore__ void conv1d_and_silu_##dtype(                             \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch,          \
        uint32_t channels, uint32_t seqLen, uint32_t kernelDim, uint32_t updateState,          \
        GM_ADDR queryStartLoc, GM_ADDR queryLens)                                              \
    {                                                                                          \
        XliteCausalConv1dSiLU<dtype> op;                                                       \
        op.Init(state, input, weight, output, batch, channels, seqLen, kernelDim, updateState, \
                queryStartLoc, queryLens);                                                     \
        op.Process();                                                                          \
    }
#else
#define CONV1D_AND_SILU_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void conv1d_and_silu_##dtype(                    \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch, \
        uint32_t channels, uint32_t seqLen, uint32_t kernelDim, uint32_t updateState, \
        GM_ADDR queryStartLoc, GM_ADDR queryLens)                                     \
    {                                                                                 \
        (void)state;                                                                  \
        (void)input;                                                                  \
        (void)weight;                                                                 \
        (void)output;                                                                 \
        (void)batch;                                                                  \
        (void)channels;                                                               \
        (void)seqLen;                                                                 \
        (void)kernelDim;                                                              \
        (void)updateState;                                                            \
        (void)queryStartLoc;                                                          \
        (void)queryLens;                                                              \
    }
#endif
