/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Native page-indexed SD causal conv1d + SiLU for single-token decode.
 * State is [pages, kernel_width-1, channels], with native page stride.
 * Read the full history and current token, compute SiLU(weighted sum), then
 * shift the history in place. Independent 64-channel tiles use vector ops.
 * Page IDs <=0 produce zero output and never modify cache storage.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "../gdn_metadata.h"

#ifdef __DAV_C220_VEC__

// Fence vector writes before scalar reads and scalar writes before vector use.
#ifndef XLITE_V_FENCE
#define XLITE_V_FENCE()                       \
    do {                                      \
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);  \
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1); \
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);  \
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1); \
    } while (0)
#endif

template <typename Dtype>
class XliteCausalConv1dSiLUDecode
{
public:
    static constexpr int kBlock = VECTOR_MAX_NUM_OF_FP32;  // 64 fp32 lanes
    static constexpr int kChanTile = 64;                   // channels per tile
    static constexpr int kMaxK = 4;
    static constexpr int kMaxWin = kChanTile * kMaxK;    // 256 elems
    static constexpr int kRawLen = kMaxWin + kChanTile;  // state tile + input row

    __aicore__ inline XliteCausalConv1dSiLUDecode()
    {
    }

    __aicore__ inline void Init(GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                uint32_t batch, uint32_t channels, uint32_t kernelDim,
                                uint32_t updateState, GM_ADDR stateIndices,
                                uint64_t statePageStride, uint32_t stateLayout, uint32_t numPages,
                                GM_ADDR cachedLens, uint64_t tableRowStride, uint32_t tableColumns,
                                uint32_t stateBlockSize, uint32_t aligned)
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
        this->kernelDim = kernelDim;
        this->updateState = updateState;
        this->stateIndices = (__gm__ int32_t *)stateIndices;
        this->statePageStride = statePageStride;
        this->stateLayout = stateLayout;
        this->numPages = numPages;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->tableRowStride = tableRowStride;
        this->tableColumns = tableColumns;
        this->stateBlockSize = stateBlockSize;
        this->aligned = aligned;

        uint64_t off = 0;
        w_raw = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kMaxWin * sizeof(Dtype);
        s_raw = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kRawLen * sizeof(Dtype);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        w_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxWin * sizeof(float);
        s_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kRawLen * sizeof(float);
        win_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxWin * sizeof(float);
        acc_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxWin * sizeof(float);
        r_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * sizeof(float);
        calc_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * sizeof(float);
        out_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kBlock * sizeof(float);
        out_buf = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kChanTile * sizeof(Dtype);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        st_stage = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += kMaxWin * sizeof(Dtype);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        off_win = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += kMaxWin * sizeof(uint32_t);
        off_k = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += kBlock * sizeof(uint32_t);
    }

    __aicore__ inline void BuildRamps()
    {
        int K = (int)kernelDim;
        int winLen = kChanTile * K;
        // Shifted-window gather table: lane p = c*K+j reads
        //   j < K-1 -> s_f[p+1]      (state tap, shift by one lane in group)
        //   j = K-1 -> s_f[winLen+c] (appended input row)
        for (int p = 0; p < winLen; ++p) {
            int c = p / K;
            int j = p % K;
            off_win[p] = (j < K - 1) ? (uint32_t)(p + 1) * sizeof(float)
                                     : (uint32_t)(winLen + c) * sizeof(float);
        }
        // Tap-reduction table: lane c reads acc_f[c*K] (base advanced by j*4).
        for (int p = 0; p < kBlock; ++p) {
            off_k[p] = (uint32_t)(p * K) * sizeof(float);
        }
        // Hand the scalar-written tables to the V pipe.
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    }

    // DMA a contiguous GM slice into a UB staging buffer with the proven
    // Make DMA input visible to vector instructions before use.
    template <typename UbT>
    __aicore__ inline void DmaIn(__gm__ Dtype *src, __ubuf__ UbT *dst, uint32_t bytes)
    {
        pipe_barrier(PIPE_V);  // prior V reads of the staging buffer must finish
        CopyGmToUbufAligned(dst, src, bytes);
        pipe_barrier(PIPE_MTE2);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    }

    // Convert nElem contiguous UB Dtype src to float dst (repeat cap 4096
    // lanes per vconv, same limit as the other conv kernels).
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
                SetMask(take);  // kernel_macro.h: set_vector_mask(high, low)
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

    __aicore__ inline void FinishStore()
    {
        // The next task can reuse the state staging buffer through either
        // MTE2 (FP32 history loads) or V (BF16/FP16 state conversion).
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    // silu(out_f) -> 64 Dtype lanes at out_buf (out_f/calc_buf are 64-lane fp32).
    __aicore__ inline void SiLU64(bool roundConv = false)
    {
        if constexpr (!std::is_same<Dtype, float>::value) {
            if (roundConv) {
                // Native Prefill uses conv1d followed by SiLU: the conv output
                // is an activation-dtype tensor before evaluating SiLU.
                convert_output(out_buf, out_f, 1);
                XLITE_V_FENCE();
                convert_input(out_f, out_buf, 1);
                XLITE_V_FENCE();
            }
        }
        vmuls(calc_buf, out_f, (float)-1.0, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(calc_buf, calc_buf, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(calc_buf, calc_buf, (float)1.0, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        if constexpr (std::is_same<Dtype, float>::value) {
            vdiv((__ubuf__ float *)out_buf, out_f, calc_buf, 1, 1, 1, 1, 8, 8, 8);
        } else {
            vdiv(calc_buf, out_f, calc_buf, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(out_buf, calc_buf, 1, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(out_buf, calc_buf, 1, 1, 1, 4, 8);
            }
        }
        pipe_barrier(PIPE_V);
        // vconv UB writeback is not covered by pipe_barrier alone (this part's
        // errata); the DMA-out below reads out_buf on MTE3 immediately, so use
        // the proven V->MTE3 event fence (generic conv kernel's SiLU pattern).
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }

    // Paged path: vLLM-Ascend GDN conv pages. Per page layout is
    // [stateRows, channels] (stateRows = K-1) with channels contiguous, which
    // is the SD orientation used on Ascend. window = [row0..row(K-2), input];
    // new state = [row1..row(K-2), input]. All rows/inputs are contiguous
    // 64-channel slices, so no vgather-based window assembly is needed.
    __aicore__ inline void ProcessPaged()
    {
        int K = (int)kernelDim;
        int rows = K - 1;
        if (K < 2 || K > kMaxK || (channels % kChanTile) != 0 || batch == 0) {
            return;
        }
        int totalTiles = (int)(channels / kChanTile);
        // Weight column-gather table: column j lives at w_f[c*K + j].
        for (int c = 0; c < kBlock; ++c) {
            off_k[c] = (uint32_t)(c * K) * sizeof(float);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);

        for (int t = (int)GetBlockIdx(); t < totalTiles; t += (int)GetBlockNum()) {
            uint64_t c0 = (uint64_t)t * kChanTile;
            // Weight tile [64][K] -> w_f, loaded once per tile, reused across batch.
            if constexpr (std::is_same<Dtype, float>::value) {
                DmaIn(weight + c0 * K, w_f, (uint32_t)(kChanTile * K * sizeof(float)));
            } else {
                DmaIn(weight + c0 * K, w_raw, (uint32_t)(kChanTile * K * sizeof(Dtype)));
                ConvUbToFloat(w_raw, kChanTile * K, w_f);
                XLITE_V_FENCE();
            }

            for (int b = 0; b < (int)batch; ++b) {
                int32_t page = GdnStatePage(stateIndices, cachedLens, b, tableRowStride,
                                            tableColumns, stateBlockSize, aligned != 0);
                if (page <= 0) {
                    vector_dup(out_f, 0.0f, 1, 1, 0, 8, 0);
                    pipe_barrier(PIPE_V);
                    SiLU64();
                    CopyUbufToGmAligned(output + (uint64_t)b * channels + c0, out_buf,
                                        kChanTile * sizeof(Dtype));
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    continue;
                }
                assert((uint32_t)page < numPages);
                if ((uint32_t)page >= numPages) {
                    return;
                }
                __gm__ Dtype *base = state + (uint64_t)page * statePageStride;
                // Stage (K-1) history rows + input row: s_f[0 .. K*64).
                if constexpr (std::is_same<Dtype, float>::value) {
                    for (int r = 0; r < rows; ++r) {
                        DmaIn(base + (uint64_t)r * channels + c0, s_f + r * kChanTile,
                              (uint32_t)(kChanTile * sizeof(float)));
                    }
                    DmaIn(input + (uint64_t)b * channels + c0, s_f + rows * kChanTile,
                          (uint32_t)(kChanTile * sizeof(float)));
                } else {
                    for (int r = 0; r < rows; ++r) {
                        DmaIn(base + (uint64_t)r * channels + c0, s_raw + r * kChanTile,
                              (uint32_t)(kChanTile * sizeof(Dtype)));
                    }
                    DmaIn(input + (uint64_t)b * channels + c0, s_raw + rows * kChanTile,
                          (uint32_t)(kChanTile * sizeof(Dtype)));
                    ConvUbToFloat(s_raw, kChanTile * K, s_f);
                    XLITE_V_FENCE();
                }

                // acc[c] = sum_j s_f[j*64 + c] * w_f[c*K + j]
                for (int j = 0; j < K; ++j) {
                    uint64_t wBase = (uint64_t)w_f + (uint64_t)j * sizeof(float);
                    vgather((__ubuf__ uint32_t *)r_f, off_k, (uint32_t)wBase, 8, 1);
                    XLITE_V_FENCE();
                    if (j == 0) {
                        vmul(out_f, s_f, r_f, 1, 1, 1, 1, 8, 8, 8);
                    } else {
                        vmul(calc_buf, s_f + j * kBlock, r_f, 1, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                        vadd(out_f, out_f, calc_buf, 1, 1, 1, 1, 8, 8, 8);
                    }
                    pipe_barrier(PIPE_V);
                }

                SiLU64();
                CopyUbufToGmAligned(output + (uint64_t)b * channels + c0, out_buf,
                                    (uint32_t)(kChanTile * sizeof(Dtype)));

                if (updateState) {
                    // New page rows are s_f[kChanTile .. K*kChanTile) = [row1..,input].
                    if constexpr (std::is_same<Dtype, float>::value) {
                        for (int r = 0; r < rows; ++r) {
                            CopyUbufToGmAligned(base + (uint64_t)r * channels + c0,
                                                s_f + (r + 1) * kChanTile,
                                                (uint32_t)(kChanTile * sizeof(float)));
                        }
                    } else {
                        if constexpr (std::is_same<Dtype, float16_t>::value) {
                            vconv_f322f16(st_stage, s_f + kChanTile, rows, 1, 1, 4, 8);
                        } else {
                            vconv_f322bf16r(st_stage, s_f + kChanTile, rows, 1, 1, 4, 8);
                        }
                        XLITE_V_FENCE();
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        for (int r = 0; r < rows; ++r) {
                            CopyUbufToGmAligned(base + (uint64_t)r * channels + c0,
                                                st_stage + r * kChanTile,
                                                (uint32_t)(kChanTile * sizeof(Dtype)));
                        }
                    }
                }
                FinishStore();
            }
        }
    }

    // Each channel tile owns its page rows until every token is computed.
    // This avoids racing a final-state writer against readers of initial history.
    __aicore__ inline void ProcessPrefill(GM_ADDR starts, GM_ADDR initial, uint32_t tokens)
    {
        auto bounds = (__gm__ int32_t *)starts;
        auto hasInitial = (__gm__ int32_t *)initial;
        uint32_t K = kernelDim, rows = K - 1, tiles = channels / kChanTile;
        for (int c = 0; c < kBlock; ++c) {
            off_k[c] = c * K * sizeof(float);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        for (uint32_t task = GetBlockIdx(); task < batch * tiles; task += GetBlockNum()) {
            uint32_t b = task / tiles;
            uint64_t c0 = (task % tiles) * kChanTile;
            int32_t start = bounds[b], end = bounds[b + 1], page = stateIndices[b];
            assert(start >= 0 && end >= start && (uint32_t)end <= tokens);
            if (start < 0 || end < start || (uint32_t)end > tokens || start == end) {
                continue;
            }
            if (page <= 0 || (uint32_t)page >= numPages) {
                assert(page <= 0 || (uint32_t)page < numPages);
                // Null rows are exactly zero, independent of prior tile contents.
                vector_dup(out_f, 0.0f, 1, 1, 0, 8, 0);
                XLITE_V_FENCE();
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f322f16(out_buf, out_f, 1, 1, 1, 4, 8);
                } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
                    vconv_f322bf16r(out_buf, out_f, 1, 1, 1, 4, 8);
                }
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                for (int32_t t = start; t < end; ++t) {
                    if constexpr (std::is_same<Dtype, float>::value) {
                        CopyUbufToGmAligned(output + (uint64_t)t * channels + c0, out_f,
                                            kChanTile * sizeof(float));
                    } else {
                        CopyUbufToGmAligned(output + (uint64_t)t * channels + c0, out_buf,
                                            kChanTile * sizeof(Dtype));
                    }
                }
                // All stores read the same unchanged zero tile. Wait once,
                // before a following task can overwrite out_f/out_buf on V.
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                continue;
            }
            if constexpr (std::is_same<Dtype, float>::value) {
                DmaIn(weight + c0 * K, w_f, kChanTile * K * sizeof(float));
            } else {
                DmaIn(weight + c0 * K, w_raw, kChanTile * K * sizeof(Dtype));
                ConvUbToFloat(w_raw, kChanTile * K, w_f);
                XLITE_V_FENCE();
            }
            __gm__ Dtype *base = state + (uint64_t)page * statePageStride;
            if (hasInitial[b]) {
                for (uint32_t r = 0; r < rows; ++r) {
                    if constexpr (std::is_same<Dtype, float>::value) {
                        DmaIn(base + r * channels + c0, s_f + r * kChanTile,
                              kChanTile * sizeof(float));
                    } else {
                        DmaIn(base + r * channels + c0, s_raw + r * kChanTile,
                              kChanTile * sizeof(Dtype));
                    }
                }
                if constexpr (!std::is_same<Dtype, float>::value) {
                    ConvUbToFloat(s_raw, rows * kChanTile, s_f);
                    XLITE_V_FENCE();
                }
            } else {
                vector_dup(s_f, 0.0f, rows, 1, 0, 8, 0);
                pipe_barrier(PIPE_V);
            }
            for (int32_t t = start; t < end; ++t) {
                if constexpr (std::is_same<Dtype, float>::value) {
                    DmaIn(input + (uint64_t)t * channels + c0, s_f + rows * kChanTile,
                          kChanTile * sizeof(float));
                } else {
                    DmaIn(input + (uint64_t)t * channels + c0, s_raw, kChanTile * sizeof(Dtype));
                    ConvUbToFloat(s_raw, kChanTile, s_f + rows * kChanTile);
                    XLITE_V_FENCE();
                }
                for (uint32_t j = 0; j < K; ++j) {
                    vgather((__ubuf__ uint32_t *)r_f, off_k,
                            (uint32_t)((uint64_t)w_f + j * sizeof(float)), 8, 1);
                    XLITE_V_FENCE();
                    vmul(calc_buf, s_f + j * kChanTile, r_f, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    if (j == 0) {
                        vadds(out_f, calc_buf, 0.0f, 1, 1, 1, 8, 8);
                    } else {
                        vadd(out_f, out_f, calc_buf, 1, 1, 1, 1, 8, 8, 8);
                    }
                    pipe_barrier(PIPE_V);
                }
                SiLU64(true);
                CopyUbufToGmAligned(output + (uint64_t)t * channels + c0, out_buf,
                                    kChanTile * sizeof(Dtype));
                // The token loop reuses s_f/out_buf across MTE2, V and MTE3.
                // Keep the full fence until per-pipeline dependencies are proven
                // for the FP32 path as well as the converted BF16/FP16 paths.
                pipe_barrier(PIPE_ALL);
                for (uint32_t r = 0; r < rows; ++r) {
                    vadds(s_f + r * kChanTile, s_f + (r + 1) * kChanTile, 0.0f, 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                }
            }
            if constexpr (!std::is_same<Dtype, float>::value) {
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f322f16(st_stage, s_f, rows, 1, 1, 4, 8);
                } else {
                    vconv_f322bf16r(st_stage, s_f, rows, 1, 1, 4, 8);
                }
                XLITE_V_FENCE();
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            for (uint32_t r = 0; r < rows; ++r) {
                if constexpr (std::is_same<Dtype, float>::value) {
                    CopyUbufToGmAligned(base + r * channels + c0, s_f + r * kChanTile,
                                        kChanTile * sizeof(float));
                } else {
                    CopyUbufToGmAligned(base + r * channels + c0, st_stage + r * kChanTile,
                                        kChanTile * sizeof(Dtype));
                }
            }
            FinishStore();
        }
    }

    __aicore__ inline void Process()
    {
        ProcessPaged();
    }

private:
    __gm__ Dtype *state;
    __gm__ Dtype *input;
    __gm__ Dtype *weight;
    __gm__ Dtype *output;
    // Paged state (vLLM GDN pages): when stateLayout==1, the per-page layout is
    // [stateRows, channels] (stateRows = kernelDim-1) and request b lives at
    // state + stateIndices[b] * statePageStride. Indices are mandatory.
    __gm__ int32_t *stateIndices;
    __gm__ int32_t *cachedLens;
    uint64_t tableRowStride;
    uint32_t tableColumns, stateBlockSize, aligned;

    __ubuf__ Dtype *w_raw;
    __ubuf__ Dtype *s_raw;
    __ubuf__ float *w_f;
    __ubuf__ float *s_f;
    __ubuf__ float *win_f;
    __ubuf__ float *acc_f;
    __ubuf__ float *r_f;
    __ubuf__ float *calc_buf;
    __ubuf__ float *out_f;
    __ubuf__ Dtype *out_buf;
    __ubuf__ Dtype *st_stage;
    __ubuf__ uint32_t *off_win;
    __ubuf__ uint32_t *off_k;

    uint32_t batch;
    uint32_t channels;
    uint32_t kernelDim;
    uint32_t updateState;
    uint64_t statePageStride;
    uint32_t numPages;
    uint32_t stateLayout;
};

#define CONV1D_AND_SILU_PAGED_FUNC_DEFINE(dtype)                                                   \
    extern "C" __global__ __aicore__ void conv1d_and_silu_paged_##dtype(                           \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch,              \
        uint32_t channels, uint32_t kernelDim, uint32_t updateState, GM_ADDR stateIndices,         \
        uint64_t statePageStride, uint32_t stateLayout, uint32_t numPages, GM_ADDR cachedLens,     \
        uint64_t tableRowStride, uint32_t tableColumns, uint32_t stateBlockSize, uint32_t aligned) \
    {                                                                                              \
        XliteCausalConv1dSiLUDecode<dtype> op;                                                     \
        op.Init(state, input, weight, output, batch, channels, kernelDim, updateState,             \
                stateIndices, statePageStride, stateLayout, numPages, cachedLens, tableRowStride,  \
                tableColumns, stateBlockSize, aligned);                                            \
        op.Process();                                                                              \
    }
#else
#define CONV1D_AND_SILU_PAGED_FUNC_DEFINE(dtype)                                                   \
    extern "C" __global__ __aicore__ void conv1d_and_silu_paged_##dtype(                           \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch,              \
        uint32_t channels, uint32_t kernelDim, uint32_t updateState, GM_ADDR stateIndices,         \
        uint64_t statePageStride, uint32_t stateLayout, uint32_t numPages, GM_ADDR cachedLens,     \
        uint64_t tableRowStride, uint32_t tableColumns, uint32_t stateBlockSize, uint32_t aligned) \
    {                                                                                              \
        (void)state;                                                                               \
        (void)input;                                                                               \
        (void)weight;                                                                              \
        (void)output;                                                                              \
        (void)batch;                                                                               \
        (void)channels;                                                                            \
        (void)kernelDim;                                                                           \
        (void)updateState;                                                                         \
        (void)stateIndices;                                                                        \
        (void)statePageStride;                                                                     \
        (void)stateLayout;                                                                         \
        (void)numPages;                                                                            \
        (void)cachedLens;                                                                          \
        (void)tableRowStride;                                                                      \
        (void)tableColumns;                                                                        \
        (void)stateBlockSize;                                                                      \
        (void)aligned;                                                                             \
    }
#endif

#ifdef __DAV_C220_VEC__
#define CONV1D_AND_SILU_PREFILL_PAGED_BODY(dtype)                                                  \
    XliteCausalConv1dSiLUDecode<dtype> op;                                                         \
    op.Init(state, input, weight, output, batch, channels, kernelDim, 1u, indices, pageStride, 1u, \
            pages, nullptr, 0, 0, 0, 0);                                                           \
    op.ProcessPrefill(starts, initial, tokens);
#else
#define CONV1D_AND_SILU_PREFILL_PAGED_BODY(dtype)
#endif
#define CONV1D_AND_SILU_PREFILL_PAGED_FUNC_DEFINE(dtype)                              \
    extern "C" __global__ __aicore__ void conv1d_and_silu_prefill_paged_##dtype(      \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch, \
        uint32_t channels, uint32_t kernelDim, GM_ADDR indices, uint64_t pageStride,  \
        uint32_t pages, GM_ADDR starts, GM_ADDR initial, uint32_t tokens)             \
    {                                                                                 \
        CONV1D_AND_SILU_PREFILL_PAGED_BODY(dtype)                                     \
    }
