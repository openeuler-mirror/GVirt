/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Fused causal conv1d + SiLU + optional conv_state update (scheme A).
 * Semantics (match ConcatCol + Conv1dAndSiLU / Python):
 *   concat = cat(state[K], input[S])
 *   out[i] = SiLU(dot(concat[i+1 : i+1+K], weight))
 *   if updateState: state = concat[..., -K:]
 *
 * No GM workspace concat. State/input are loaded into aligned UB float buffers;
 * windows are gathered via scalar float moves (PIPE_S). GM copies use
 * CopyGmToUbufAligned / CopyUbufToGmAligned so lengths need not be 32B-aligned.
 *
 * Limits (host falls back if exceeded): kernelDim <= 16, seqLen <= 4096.
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

    __aicore__ inline XliteCausalConv1dSiLU()
    {
    }

    __aicore__ inline void Init(GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                uint32_t batch, uint32_t channels, uint32_t seqLen,
                                uint32_t kernelDim, uint32_t updateState)
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
        this->kernelDim = kernelDim;
        this->updateState = updateState;

        uint64_t off = 0;
        // Aligned staging for GM<->UB dtype copies (must stay at offset 0 of a 32B region).
        stage_buf = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += 8 * 32;  // 256B
        tmp_kernel_buf = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += 8 * 32;

        state_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxKernel * sizeof(float);  // 64B, keep 32B-aligned
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        input_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxInputF * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }
        new_state_f = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += kMaxKernel * sizeof(float);
        if (off % 32 != 0) {
            off = (off + 31) / 32 * 32;
        }

        qkv_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        kernel_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        mul_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        calc_buf = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 8 * 32;
        out_buf = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
    }

    __aicore__ inline float ReadFloat(__ubuf__ float *buf, int idx)
    {
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        float val = buf[idx];
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        return val;
    }

    __aicore__ inline void WriteFloat(__ubuf__ float *buf, int idx, float val)
    {
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        buf[idx] = val;
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    }

    __aicore__ inline void SiLU()
    {
        vmuls(calc_buf, mul_buf, (float)-1.0, 1, 0, 0, 0, 0);
        pipe_barrier(PIPE_V);
        vexp(calc_buf, calc_buf, 1, 0, 0, 0, 0);
        pipe_barrier(PIPE_V);
        vadds(calc_buf, calc_buf, (float)1.0, 1, 0, 0, 0, 0);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if constexpr (std::is_same<Dtype, float>::value) {
            vdiv(out_buf, mul_buf, calc_buf, 1, 0, 0, 0, 0, 0, 0);
        } else {
            vdiv(calc_buf, mul_buf, calc_buf, 1, 0, 0, 0, 0, 0, 0);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(out_buf, calc_buf, 1, 0, 0, 0, 0);
            } else {
                vconv_f322bf16r(out_buf, calc_buf, 1, 0, 0, 0, 0);
            }
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }

    // Load nElem dtype values from GM into dstF (float). Chunks of 8 keep float dst 32B-aligned.
    __aicore__ inline void LoadGmToFloat(__gm__ Dtype *src, int nElem, __ubuf__ float *dstF)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(dstF, src, static_cast<uint32_t>(nElem * sizeof(float)));
            pipe_barrier(PIPE_MTE2);
            return;
        }
        constexpr int kChunk = 8;
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

            uint64_t mask = (take >= 64) ? (uint64_t)-1 : ((1ull << take) - 1ull);
            set_vector_mask(0, mask);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(dstF + done, stage_buf, 1, 0, 0, 0, 0);
            } else {
                vconv_bf162f32(dstF + done, stage_buf, 1, 0, 0, 0, 0);
            }
            pipe_barrier(PIPE_V);
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
        constexpr int kChunk = 8;
        int done = 0;
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        while (done < nElem) {
            int take = nElem - done;
            if (take > kChunk) {
                take = kChunk;
            }
            uint64_t mask = (take >= 64) ? (uint64_t)-1 : ((1ull << take) - 1ull);
            set_vector_mask(0, mask);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(stage_buf, srcF + done, 1, 0, 0, 0, 0);
            } else {
                vconv_f322bf16r(stage_buf, srcF + done, 1, 0, 0, 0, 0);
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

    __aicore__ inline void GatherWindow(int i)
    {
        int K = (int)kernelDim;
        for (int j = 0; j < K; ++j) {
            int absIdx = i + 1 + j;
            float v = (absIdx < K) ? ReadFloat(state_f, absIdx) : ReadFloat(input_f, absIdx - K);
            WriteFloat(qkv_buf, j, v);
        }
    }

    __aicore__ inline void WriteBackState(int batchIdx, int channel)
    {
        int K = (int)kernelDim;
        int S = (int)seqLen;
        for (int j = 0; j < K; ++j) {
            int absIdx = S + j;
            float v = (absIdx < K) ? ReadFloat(state_f, absIdx) : ReadFloat(input_f, absIdx - K);
            WriteFloat(new_state_f, j, v);
        }
        int stateBase = (batchIdx * (int)channels + channel) * K;
        StoreFloatToGm(new_state_f, state + stateBase, K);
    }

    __aicore__ inline void Process()
    {
        if ((int)kernelDim > kMaxKernel || (int)seqLen > kMaxInputF) {
            return;
        }

        set_vector_mask(0, 15);
        for (int channel = 0; channel < (int)channels; channel++) {
            if (channel % GetBlockNum() != GetBlockIdx())
                continue;

            int wOffset = channel * (int)kernelDim;
            if constexpr (std::is_same<Dtype, float>::value) {
                CopyGmToUbufAligned(kernel_buf, weight + wOffset,
                                    static_cast<uint32_t>(kernelDim * sizeof(Dtype)));
                pipe_barrier(PIPE_MTE2);
            } else {
                CopyGmToUbufAligned(tmp_kernel_buf, weight + wOffset,
                                    static_cast<uint32_t>(kernelDim * sizeof(Dtype)));
                pipe_barrier(PIPE_MTE2);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                uint64_t wmask = (kernelDim >= 64) ? (uint64_t)-1 : ((1ull << kernelDim) - 1ull);
                set_vector_mask(0, wmask);
                if constexpr (std::is_same<Dtype, float16_t>::value) {
                    vconv_f162f32(kernel_buf, tmp_kernel_buf, 1, 0, 0, 0, 0);
                } else {
                    vconv_bf162f32(kernel_buf, tmp_kernel_buf, 1, 0, 0, 0, 0);
                }
                pipe_barrier(PIPE_V);
                set_vector_mask(0, 15);
            }

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            for (int batchIdx = 0; batchIdx < (int)batch; ++batchIdx) {
                int stateBase = (batchIdx * (int)channels + channel) * (int)kernelDim;
                int inputBase = (batchIdx * (int)channels + channel) * (int)seqLen;

                LoadGmToFloat(state + stateBase, (int)kernelDim, state_f);
                LoadGmToFloat(input + inputBase, (int)seqLen, input_f);

                for (int i = 0; i < (int)seqLen; ++i) {
                    GatherWindow(i);

                    uint64_t kmask =
                        (kernelDim >= 64) ? (uint64_t)-1 : ((1ull << kernelDim) - 1ull);
                    set_vector_mask(0, kmask);
                    vmul(mul_buf, qkv_buf, kernel_buf, 1, 0, 0, 0, 0, 0, 0);
                    pipe_barrier(PIPE_V);
                    vcadd(mul_buf, mul_buf, 1, 0, 0, 0, 0);
                    pipe_barrier(PIPE_V);
                    set_vector_mask(0, 15);

                    SiLU();

                    DumpBuffer(kernel_buf, "kernel", 4);
                    DumpBuffer(qkv_buf, "qkv", 4);
                    DumpBuffer(mul_buf, "mul", 4);
                    DumpBuffer(out_buf, "out", 1);

                    int outOffset = (batchIdx * (int)channels + channel) * (int)seqLen + i;
                    // Per-element dst is often not 32B-aligned (bf16); use aligned helper.
                    CopyUbufToGmAligned(output + outOffset, out_buf, sizeof(Dtype));
                    pipe_barrier(PIPE_MTE3);
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                }

                if (updateState) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    WriteBackState(batchIdx, channel);
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                }
            }
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }

private:
    __gm__ Dtype *state;
    __gm__ Dtype *input;
    __gm__ Dtype *weight;
    __gm__ Dtype *output;

    __ubuf__ Dtype *stage_buf;
    __ubuf__ Dtype *tmp_kernel_buf;
    __ubuf__ float *state_f;
    __ubuf__ float *input_f;
    __ubuf__ float *new_state_f;
    __ubuf__ float *qkv_buf;
    __ubuf__ float *kernel_buf;
    __ubuf__ float *mul_buf;
    __ubuf__ float *calc_buf;
    __ubuf__ Dtype *out_buf;

    uint32_t batch;
    uint32_t channels;
    uint32_t seqLen;
    uint32_t kernelDim;
    uint32_t updateState;
};

#define CONV1D_AND_SILU_FUNC_DEFINE(dtype)                                                      \
    extern "C" __global__ __aicore__ void conv1d_and_silu_##dtype(                              \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch,           \
        uint32_t channels, uint32_t seqLen, uint32_t kernelDim, uint32_t updateState)           \
    {                                                                                           \
        XliteCausalConv1dSiLU<dtype> op;                                                        \
        op.Init(state, input, weight, output, batch, channels, seqLen, kernelDim, updateState); \
        op.Process();                                                                           \
    }
#else
#define CONV1D_AND_SILU_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void conv1d_and_silu_##dtype(                    \
        GM_ADDR state, GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t batch, \
        uint32_t channels, uint32_t seqLen, uint32_t kernelDim, uint32_t updateState) \
    {                                                                                 \
    }
#endif
