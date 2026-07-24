/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Recurrent gated delta rule (GDN Step 7).
 * Matches tests/models/qwen3_5.py::_torch_recurrent_gated_delta_rule with
 * use_qk_l2norm=False (L2Norm already applied upstream).
 *
 * Layouts:
 *   query/key: [B*S, H*kDim]
 *   value/out: [B*S, H*vDim]
 *   beta/g:    [B*S, H]   (g is log-space; kernel applies exp(g))
 *   state:     [B, H, kDim, vDim]  (updated in-place)
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

// Max dims that fit full state in UB (float): kDim*vDim*4 + temps < ~180KB
#define GDR_MAX_K_DIM 128
#define GDR_MAX_V_DIM 128

template <typename Dtype>
class RecurrentGatedDeltaRule
{
public:
    __aicore__ inline RecurrentGatedDeltaRule()
    {
    }

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR g,
                                GM_ADDR state, GM_ADDR out, uint32_t batch, uint32_t seqlen,
                                uint32_t numHeads, uint32_t kDim, uint32_t vDim, float scale)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        this->query = (__gm__ Dtype *)query;
        this->key = (__gm__ Dtype *)key;
        this->value = (__gm__ Dtype *)value;
        this->beta = (__gm__ Dtype *)beta;
        this->g = (__gm__ Dtype *)g;
        this->state = (__gm__ Dtype *)state;
        this->out = (__gm__ Dtype *)out;
        this->batch = batch;
        this->seqlen = seqlen;
        this->numHeads = numHeads;
        this->kDim = kDim;
        this->vDim = vDim;
        this->scale = scale;

        qkBytes = kDim * sizeof(Dtype);
        vBytes = vDim * sizeof(Dtype);
        stateBytes = kDim * vDim * sizeof(Dtype);
        qkBlocks = DIV_ROUND_UP(qkBytes, 32);
        vBlocks = DIV_ROUND_UP(vBytes, 32);
        stateBlocks = DIV_ROUND_UP(stateBytes, 32);
        vRepeat = DIV_ROUND_UP(vDim, VECTOR_MAX_NUM_OF_FP32);
        kRepeat = DIV_ROUND_UP(kDim, VECTOR_MAX_NUM_OF_FP32);
        stateF32Repeat = DIV_ROUND_UP(kDim * vDim, VECTOR_MAX_NUM_OF_FP32);

        uint64_t off = 0;
        qIn = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += ROUND_UP(qkBytes, 256);
        kIn = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += ROUND_UP(qkBytes, 256);
        vIn = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += ROUND_UP(vBytes, 256);
        stateIn = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += ROUND_UP(stateBytes, 256);
        outTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += ROUND_UP(MAX(vBytes, 32u), 256);
        qF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(kDim * sizeof(float), 256);
        kF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(kDim * sizeof(float), 256);
        vF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(vDim * sizeof(float), 256);
        stateF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(kDim * vDim * sizeof(float), 256);
        kvMem = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(vDim * sizeof(float), 256);
        delta = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(vDim * sizeof(float), 256);
        outF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(vDim * sizeof(float), 256);
        tmpRow = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(vDim * sizeof(float), 256);
        // Dedicated 1-element scratch for scalar<->vector handoff (g/beta/exp).
        scalarF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 256;
        assert(off <= UB_SIZE);
    }

    __aicore__ inline void ConvHalfToFloat(__ubuf__ float *dst, __ubuf__ Dtype *src, int repeat)
    {
        // vconv repeat is uint8 (max 255). K=V=128 state needs 256 repeats.
        uint32_t done = 0;
        int remain = repeat;
        while (remain > 0) {
            int rep = remain > VECTOR_MAX_REPEAT ? VECTOR_MAX_REPEAT : remain;
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(dst + done, src + done, rep, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(dst + done, src + done, rep, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            done += static_cast<uint32_t>(rep) * VECTOR_MAX_NUM_OF_FP32;
            remain -= rep;
        }
    }

    __aicore__ inline void ConvFloatToHalf(__ubuf__ Dtype *dst, __ubuf__ float *src, int repeat)
    {
        uint32_t done = 0;
        int remain = repeat;
        while (remain > 0) {
            int rep = remain > VECTOR_MAX_REPEAT ? VECTOR_MAX_REPEAT : remain;
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f322f16(dst + done, src + done, rep, 1, 1, 4, 8);
            } else {
                vconv_f322bf16r(dst + done, src + done, rep, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);
            done += static_cast<uint32_t>(rep) * VECTOR_MAX_NUM_OF_FP32;
            remain -= rep;
        }
    }

    __aicore__ inline void LoadVec(__ubuf__ float *dst, __gm__ Dtype *src, __ubuf__ Dtype *tmp,
                                   uint32_t nElem, uint32_t nBlocks, int repeat)
    {
        (void)nElem;
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_gm_to_ubuf(dst, src, 0, 1, nBlocks, 0, 0);
            pipe_barrier(PIPE_MTE2);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        } else {
            copy_gm_to_ubuf(tmp, src, 0, 1, nBlocks, 0, 0);
            pipe_barrier(PIPE_MTE2);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            ConvHalfToFloat(dst, tmp, repeat);
        }
    }

    __aicore__ inline void StoreVec(__gm__ Dtype *dst, __ubuf__ float *src, __ubuf__ Dtype *tmp,
                                    uint32_t nBlocks, int repeat)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm(dst, src, 0, 1, nBlocks, 0, 0);
            pipe_barrier(PIPE_MTE3);
        } else {
            ConvFloatToHalf(tmp, src, repeat);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm(dst, tmp, 0, 1, nBlocks, 0, 0);
            pipe_barrier(PIPE_MTE3);
        }
    }

    __aicore__ inline void VMulsLarge(__ubuf__ float *dst, __ubuf__ float *src, float scale,
                                      uint32_t nElem)
    {
        // scale comes from scalar pipe; publish to V before vmuls.
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        uint32_t done = 0;
        while (done < nElem) {
            uint32_t remain = nElem - done;
            int rep = static_cast<int>(DIV_ROUND_UP(remain, VECTOR_MAX_NUM_OF_FP32));
            if (rep > VECTOR_MAX_REPEAT) {
                rep = VECTOR_MAX_REPEAT;
            }
            vmuls(dst + done, src + done, scale, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            done += static_cast<uint32_t>(rep) * VECTOR_MAX_NUM_OF_FP32;
            if (done > nElem) {
                done = nElem;
            }
        }
    }

    __aicore__ inline float LoadScalar(__gm__ Dtype *ptr)
    {
        copy_gm_to_ubuf(outTmp, ptr, 0, 1, 1, 0, 0);
        pipe_barrier(PIPE_MTE2);
        if constexpr (std::is_same<Dtype, float>::value) {
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            return reinterpret_cast<__ubuf__ float *>(outTmp)[0];
        } else {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            if constexpr (std::is_same<Dtype, float16_t>::value) {
                vconv_f162f32(scalarF, outTmp, 1, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(scalarF, outTmp, 1, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            return scalarF[0];
        }
    }

    __aicore__ inline float ExpScalar(float x)
    {
        scalarF[0] = x;
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        vexp(scalarF, scalarF, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        return scalarF[0];
    }

    __aicore__ inline float ReadFloat(__ubuf__ float *buf, uint32_t idx)
    {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float val = buf[idx];
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        return val;
    }

    __aicore__ inline void ProcessOneHead(uint32_t b, uint32_t h)
    {
        uint32_t stateOff = ((b * numHeads) + h) * kDim * vDim;
        LoadVec(stateF, state + stateOff, stateIn, kDim * vDim, stateBlocks, stateF32Repeat);

        float scale = this->scale;

        for (uint32_t s = 0; s < seqlen; ++s) {
            uint32_t t = b * seqlen + s;
            uint32_t qkOff = t * numHeads * kDim + h * kDim;
            uint32_t vOff = t * numHeads * vDim + h * vDim;
            uint32_t bgOff = t * numHeads + h;

            LoadVec(qF, query + qkOff, qIn, kDim, qkBlocks, kRepeat);
            LoadVec(kF, key + qkOff, kIn, kDim, qkBlocks, kRepeat);
            LoadVec(vF, value + vOff, vIn, vDim, vBlocks, vRepeat);

            // q *= scale
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            vmuls(qF, qF, scale, kRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            float gExp = ExpScalar(LoadScalar(g + bgOff));
            float betaVal = LoadScalar(beta + bgOff);

            // state *= exp(g)
            VMulsLarge(stateF, stateF, gExp, kDim * vDim);

            // kv_mem[v] = sum_k state[k,v] * k[k]
            vector_dup(kvMem, float(0), vRepeat, 1, 0, 8, 0);
            pipe_barrier(PIPE_V);
            for (uint32_t ki = 0; ki < kDim; ++ki) {
                float kk = ReadFloat(kF, ki);
                vmuls(tmpRow, stateF + ki * vDim, kk, vRepeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vadd(kvMem, kvMem, tmpRow, vRepeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }

            // delta = (v - kv_mem) * beta
            vsub(delta, vF, kvMem, vRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            vmuls(delta, delta, betaVal, vRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // state[k,v] += k[k] * delta[v]
            for (uint32_t ki = 0; ki < kDim; ++ki) {
                float kk = ReadFloat(kF, ki);
                vmuls(tmpRow, delta, kk, vRepeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vadd(stateF + ki * vDim, stateF + ki * vDim, tmpRow, vRepeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }

            // out[v] = sum_k state[k,v] * q[k]
            vector_dup(outF, float(0), vRepeat, 1, 0, 8, 0);
            pipe_barrier(PIPE_V);
            for (uint32_t ki = 0; ki < kDim; ++ki) {
                float qq = ReadFloat(qF, ki);
                vmuls(tmpRow, stateF + ki * vDim, qq, vRepeat, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vadd(outF, outF, tmpRow, vRepeat, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }

            StoreVec(out + vOff, outF, outTmp, vBlocks, vRepeat);
            // StoreVec ends on MTE3; next LoadVec starts on MTE2.
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        StoreVec(state + stateOff, stateF, stateIn, stateBlocks, stateF32Repeat);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    __aicore__ inline void Process()
    {
        if (kDim > GDR_MAX_K_DIM || vDim > GDR_MAX_V_DIM || kDim == 0 || vDim == 0) {
            return;
        }
        uint32_t total = batch * numHeads;
        for (uint32_t idx = GetBlockIdx(); idx < total; idx += GetBlockNum()) {
            uint32_t b = idx / numHeads;
            uint32_t h = idx % numHeads;
            ProcessOneHead(b, h);
        }
    }

private:
    __gm__ Dtype *query;
    __gm__ Dtype *key;
    __gm__ Dtype *value;
    __gm__ Dtype *beta;
    __gm__ Dtype *g;
    __gm__ Dtype *state;
    __gm__ Dtype *out;

    __ubuf__ Dtype *qIn;
    __ubuf__ Dtype *kIn;
    __ubuf__ Dtype *vIn;
    __ubuf__ Dtype *stateIn;
    __ubuf__ Dtype *outTmp;

    __ubuf__ float *qF;
    __ubuf__ float *kF;
    __ubuf__ float *vF;
    __ubuf__ float *stateF;
    __ubuf__ float *kvMem;
    __ubuf__ float *delta;
    __ubuf__ float *outF;
    __ubuf__ float *tmpRow;
    __ubuf__ float *scalarF;

    uint32_t batch;
    uint32_t seqlen;
    uint32_t numHeads;
    uint32_t kDim;
    uint32_t vDim;
    float scale;
    uint32_t qkBytes;
    uint32_t vBytes;
    uint32_t stateBytes;
    uint32_t qkBlocks;
    uint32_t vBlocks;
    uint32_t stateBlocks;
    int vRepeat;
    int kRepeat;
    int stateF32Repeat;
};

#define RECURRENT_GATED_DELTA_RULE_FUNC_DEFINE(dtype)                                        \
    extern "C" __global__ __aicore__ void recurrent_gated_delta_rule_##dtype(                \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR g, GM_ADDR state,   \
        GM_ADDR out, uint32_t batch, uint32_t seqlen, uint32_t numHeads, uint32_t kDim,      \
        uint32_t vDim, float scale)                                                          \
    {                                                                                        \
        RecurrentGatedDeltaRule<dtype> op;                                                   \
        op.Init(query, key, value, beta, g, state, out, batch, seqlen, numHeads, kDim, vDim, \
                scale);                                                                      \
        op.Process();                                                                        \
    }
#else
#define RECURRENT_GATED_DELTA_RULE_FUNC_DEFINE(dtype)                                      \
    extern "C" __global__ __aicore__ void recurrent_gated_delta_rule_##dtype(              \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR g, GM_ADDR state, \
        GM_ADDR out, uint32_t batch, uint32_t seqlen, uint32_t numHeads, uint32_t kDim,    \
        uint32_t vDim, float scale)                                                        \
    {                                                                                      \
        (void)query;                                                                       \
        (void)key;                                                                         \
        (void)value;                                                                       \
        (void)beta;                                                                        \
        (void)g;                                                                           \
        (void)state;                                                                       \
        (void)out;                                                                         \
        (void)batch;                                                                       \
        (void)seqlen;                                                                      \
        (void)numHeads;                                                                    \
        (void)kDim;                                                                        \
        (void)vDim;                                                                        \
        (void)scale;                                                                       \
    }
#endif
