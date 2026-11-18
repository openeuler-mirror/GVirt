/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Single-token KDA safe-gate recurrence using vLLM's native [P, Hv, V, K] state
 * (state_v_first, K contiguous). Skeleton shared with
 * recurrent_gated_delta_rule_paged.h (page addressing, MatVec reduction,
 * 64-V-row tiles); differences are the native run_recurrent_kda contract:
 *   - beta is raw: sigmoid(b) in fp32 in-kernel (beta_is_preprocessed=False).
 *   - gate is raw [T, Hk*K]: gate = lowerBound * sigmoid(exp(A_log[h]) *
 *     (g1 + dt_bias)) in fp32 in-kernel (safe_gate); applied as log-decay
 *     per k row: state[k,:] *= exp(gate[k]) -one broadcast vmul per tile.
 *   - A_log [Hk] / dt_bias [Hk*K] are fp32 parameters.
 *   - q/k L2-normalized in-kernel; q additionally scaled by `scale`.
 */
#pragma once
#include "kernel_macro.h"
#include "../gdn_metadata.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
class RecurrentGatedDeltaRuleKdaPaged
{
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                GM_ADDR gate, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR state,
                                GM_ADDR out, GM_ADDR indices, uint32_t batch, uint32_t kHeads,
                                uint32_t vHeads, uint32_t kDim, uint32_t vDim, uint64_t pageStride,
                                uint32_t numPages, uint32_t stateFp32, float scale,
                                float lowerBound, float normEps, GM_ADDR cachedLens,
                                uint64_t tableRowStride, uint32_t tableColumns,
                                uint32_t stateBlockSize, uint32_t aligned)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        qGm = (__gm__ Dtype *)query;
        kGm = (__gm__ Dtype *)key;
        vGm = (__gm__ Dtype *)value;
        betaGm = (__gm__ Dtype *)beta;
        gateGm = (__gm__ Dtype *)gate;
        aLogGm = (__gm__ float *)aLog;
        dtBiasGm = (__gm__ float *)dtBias;
        stateGm = state;
        outGm = (__gm__ Dtype *)out;
        indicesGm = (__gm__ int32_t *)indices;
        this->batch = batch;
        this->kHeads = kHeads;
        this->vHeads = vHeads;
        this->kDim = kDim;
        this->vDim = vDim;
        this->pageStride = pageStride;
        this->numPages = numPages;
        this->stateFp32 = stateFp32;
        this->scale = scale;
        this->lowerBound = lowerBound;
        this->normEps = normEps;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->tableRowStride = tableRowStride;
        this->tableColumns = tableColumns;
        this->stateBlockSize = stateBlockSize;
        this->aligned = aligned;

        uint64_t off = 0;
        ioTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += 512;
        stateTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += std::is_same<Dtype, float>::value ? 0 : 64 * 128 * sizeof(Dtype);
        qF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        kF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        vF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        gF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        dtBiasF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        onesF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        stateF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 64 * 128 * 4;
        product = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 64 * 128 * 4;
        delta = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 256;
        outF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        reduceTail = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 256;
        deltaBlocks = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 64 * 32;
        scalarF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 256;
        assert(off <= UB_SIZE);
        vector_dup(onesF, 1.0f, DIV_ROUND_UP(kDim, 64), 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void LoadVector(__ubuf__ float *dst, __gm__ Dtype *src, uint32_t count)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(dst, src, count * sizeof(Dtype));
        } else {
            CopyGmToUbufAligned(ioTmp, src, count * sizeof(Dtype));
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (!std::is_same<Dtype, float>::value) {
            convert_input(dst, ioTmp, DIV_ROUND_UP(count, 64));
            pipe_barrier(PIPE_V);
            // The next load reuses ioTmp; a V barrier alone does not prevent
            // its MTE2 write from overtaking this conversion's UB read.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
    }

    __aicore__ inline void StoreVector(__gm__ Dtype *dst, __ubuf__ float *src, uint32_t count)
    {
        if constexpr (!std::is_same<Dtype, float>::value) {
            convert_output(ioTmp, src, DIV_ROUND_UP(count, 64));
            pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyUbufToGmAligned(dst, src, count * sizeof(Dtype));
        } else {
            CopyUbufToGmAligned(dst, ioTmp, count * sizeof(Dtype));
        }
        FinishStore();
    }

    __aicore__ inline void FinishStore()
    {
        // Reuse of UB staging and subsequent GM reads must wait for DMA writes.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    // FP32 parameter scalar (A_log); `exponent` applies exp in-vector.
    __aicore__ inline float LoadScalarF32(__gm__ float *src, bool exponent)
    {
        CopyGmToUbufAligned(scalarF, src, sizeof(float));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if (exponent) {
            vexp(scalarF, scalarF, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float result = scalarF[0];
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        return result;
    }

    // Sigmoid in fp32 on a contiguous lane buffer: 1 / (1 + exp(-x)).
    __aicore__ inline void SigmoidLane(__ubuf__ float *lane, uint32_t repeats)
    {
        vmuls(lane, lane, -1.0f, repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(lane, lane, repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(lane, lane, 1.0f, repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(lane, onesF, lane, repeats, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    // beta_raw[t*H + h] -> sigmoid fp32.
    __aicore__ inline float LoadBeta(uint64_t index)
    {
        LoadVector(scalarF, betaGm + index, 1);
        SigmoidLane(scalarF, 1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float beta = scalarF[0];
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        return beta;
    }

    // Per-head dt_bias slice [kDim], fp32 parameter.
    __aicore__ inline void LoadDtBias(uint32_t kh)
    {
        CopyGmToUbufAligned(dtBiasF, dtBiasGm + (uint64_t)kh * kDim, kDim * sizeof(float));
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    // Raw gate row [kDim] -> fp32 log-decay:
    //   log_g = lowerBound * sigmoid(exp(A_log[h]) * (g1 + dt_bias))
    //   gF = exp(log_g)   (decay applied per k row below)
    __aicore__ inline void LoadDecayVec(uint64_t index, float decayHead)
    {
        LoadVector(gF, gateGm + index, kDim);
        uint32_t repeats = DIV_ROUND_UP(kDim, 64);
        vadd(gF, gF, dtBiasF, repeats, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(gF, gF, decayHead, repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        SigmoidLane(gF, repeats);
        vmuls(gF, gF, lowerBound, repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(gF, gF, repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    // Per-k decay on the v-first tile: state[r, k] *= gF[k] for 64 rows.
    __aicore__ inline void ApplyDecay()
    {
        uint32_t rowStride = kDim / 8;
        for (uint32_t col = 0; col < kDim; col += 64) {
            SetMask(MIN(64u, kDim - col));
            vmul(stateF + col, stateF + col, gF + col, 64, 1, 1, 1, rowStride, rowStride, 0);
            pipe_barrier(PIPE_V);
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    // In-kernel L2 norm: rstd = rsqrt(sum(x^2) + eps); q additionally scaled.
    __aicore__ inline void L2Norm(__ubuf__ float *buf, float scaleMul)
    {
        for (uint32_t col = 0; col < kDim; col += 64) {
            SetMask(MIN(64u, kDim - col));
            vmul(product + col, buf + col, buf + col, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        ReduceSum<float>(scalarF, product, kDim);
        vadds(scalarF, scalarF, normEps, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // vrsqrt is a fast approximate instruction (~5e-4 relative error); use the
        // exact sqrt+div pair instead so fp32 state matches the reference bit-closely.
        vsqrt(scalarF, scalarF, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(scalarF, onesF, scalarF, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float rstd = scalarF[0] * scaleMul;
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        vmuls(buf, buf, rstd, DIV_ROUND_UP(kDim, 64), 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void LoadState(uint64_t offset)
    {
        uint32_t count = 64 * kDim;
        if (stateFp32 || std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(stateF, (__gm__ float *)stateGm + offset, count * 4);
        } else {
            CopyGmToUbufAligned(stateTmp, (__gm__ Dtype *)stateGm + offset, count * sizeof(Dtype));
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if (!stateFp32) {
            if constexpr (!std::is_same<Dtype, float>::value) {
                convert_input(stateF, stateTmp, count / 64);
                pipe_barrier(PIPE_V);
            }
        }
    }

    __aicore__ inline void StoreState(uint64_t offset)
    {
        uint32_t count = 64 * kDim;
        if (!stateFp32) {
            if constexpr (!std::is_same<Dtype, float>::value) {
                convert_output(stateTmp, stateF, count / 64);
                pipe_barrier(PIPE_V);
            }
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (stateFp32 || std::is_same<Dtype, float>::value) {
            CopyUbufToGmAligned((__gm__ float *)stateGm + offset, stateF, count * 4);
        } else {
            CopyUbufToGmAligned((__gm__ Dtype *)stateGm + offset, stateTmp, count * sizeof(Dtype));
        }
        FinishStore();
    }

    __aicore__ inline void MatVec(__ubuf__ float *vec, __ubuf__ float *dst)
    {
        uint32_t rowStride = kDim / 8;
        for (uint32_t col = 0; col < kDim; col += 64) {
            SetMask(MIN(64u, kDim - col));
            vmul(product + col, stateF + col, vec + col, 64, 1, 1, 1, rowStride, rowStride, 0);
            pipe_barrier(PIPE_V);
        }
        if (kDim == 128) {
            // Fold into the upper half to avoid multi-repeat src0/dst overlap.
            vadd(product + 64, product, product + 64, 64, 1, 1, 1, 16, 16, 16);
            pipe_barrier(PIPE_V);
            vcadd(dst, product + 64, 64, 1, 1, 16, 0);
        } else {
            SetMask(MIN(64u, kDim));
            vcadd(dst, product, 64, 1, 1, rowStride, 0);
            if (kDim > 64) {
                pipe_barrier(PIPE_V);
                SetMask(kDim - 64);
                vcadd(reduceTail, product + 64, 64, 1, 1, rowStride, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
                vadd(dst, dst, reduceTail, 1, 1, 1, 1, 8, 8, 8);
            }
        }
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }

    __aicore__ inline void ProcessHead(uint32_t b, uint32_t h)
    {
        int32_t page = GdnStatePage(indicesGm, cachedLens, b, tableRowStride, tableColumns,
                                    stateBlockSize, aligned != 0);
        uint64_t vo = ((uint64_t)b * vHeads + h) * vDim;
        if (page <= 0 || (uint32_t)page >= numPages) {
            // Page zero is vLLM's null block; never dereference or update it.
            assert(page <= 0 || (uint32_t)page < numPages);
            vector_dup(outF, 0.0f, vDim / 64, 1, 0, 8, 0);
            pipe_barrier(PIPE_V);
            StoreVector(outGm + vo, outF, vDim);
            return;
        }
        uint32_t kh = h / (vHeads / kHeads);
        uint64_t qo = ((uint64_t)b * kHeads + kh) * kDim;
        float decayHead = LoadScalarF32(aLogGm + kh, true);
        LoadDtBias(kh);
        LoadVector(qF, qGm + qo, kDim);
        LoadVector(kF, kGm + qo, kDim);
        LoadVector(vF, vGm + vo, vDim);
        L2Norm(qF, scale);
        L2Norm(kF, 1.0f);
        float beta = LoadBeta((uint64_t)b * vHeads + h);
        LoadDecayVec(((uint64_t)b * kHeads + kh) * kDim, decayHead);
        uint64_t headOffset = (uint64_t)page * pageStride + (uint64_t)h * vDim * kDim;
        for (uint32_t row = 0; row < vDim; row += 64) {
            uint64_t offset = headOffset + (uint64_t)row * kDim;
            LoadState(offset);
            ApplyDecay();
            MatVec(kF, delta);
            vsub(delta, vF + row, delta, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vmuls(delta, delta, beta, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            // Each delta scalar becomes one 32B block, reused across K blocks.
            vbrcb((__ubuf__ uint32_t *)deltaBlocks, (__ubuf__ uint32_t *)delta, 1, 8, 8);
            pipe_barrier(PIPE_V);
            uint32_t rowStride = kDim / 8;
            for (uint32_t col = 0; col < kDim; col += 64) {
                SetMask(MIN(64u, kDim - col));
                // Separate multiply/add avoids the A2 vmadd broadcast hazard.
                vmul(product + col, deltaBlocks, kF + col, 64, 1, 0, 1, rowStride, 1, 0);
                pipe_barrier(PIPE_V);
                vadd(stateF + col, stateF + col, product + col, 64, 1, 1, 1, rowStride, rowStride,
                     rowStride);
                pipe_barrier(PIPE_V);
            }
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            MatVec(qF, outF + row);
            StoreState(offset);
        }
        StoreVector(outGm + vo, outF, vDim);
    }

    // One task owns a sequence/head/V tile throughout the query. FP32 scratch
    // remains in UB; only the final state is committed to the native page.
    __aicore__ inline void ProcessPrefill(GM_ADDR starts, GM_ADDR initial, uint32_t tokens)
    {
        auto bounds = (__gm__ int32_t *)starts;
        auto hasInitial = (__gm__ int32_t *)initial;
        uint32_t tiles = vDim / 64;
        for (uint32_t task = GetBlockIdx(); task < batch * vHeads * tiles; task += GetBlockNum()) {
            uint32_t row = (task % tiles) * 64;
            uint32_t h = (task / tiles) % vHeads;
            uint32_t b = task / (tiles * vHeads);
            int32_t start = bounds[b], end = bounds[b + 1];
            assert(start >= 0 && end >= start && (uint32_t)end <= tokens);
            if (start < 0 || end < start || (uint32_t)end > tokens || start == end) {
                continue;
            }
            int32_t page = indicesGm[b];
            if (page <= 0 || (uint32_t)page >= numPages) {
                assert(page <= 0 || (uint32_t)page < numPages);
                vector_dup(outF, 0.0f, 1, 1, 0, 8, 0);
                pipe_barrier(PIPE_V);
                for (int32_t t = start; t < end; ++t) {
                    StoreVector(outGm + ((uint64_t)t * vHeads + h) * vDim + row, outF, 64);
                }
                continue;
            }
            uint32_t kh = h / (vHeads / kHeads);
            float decayHead = LoadScalarF32(aLogGm + kh, true);
            LoadDtBias(kh);
            uint64_t offset = (uint64_t)page * pageStride + ((uint64_t)h * vDim + row) * kDim;
            if (hasInitial[b]) {
                LoadState(offset);
            } else {
                vector_dup(stateF, 0.0f, kDim, 1, 0, 8, 0);
                pipe_barrier(PIPE_V);
            }
            for (int32_t t = start; t < end; ++t) {
                uint64_t qo = ((uint64_t)t * kHeads + kh) * kDim;
                uint64_t vo = ((uint64_t)t * vHeads + h) * vDim + row;
                LoadVector(qF, qGm + qo, kDim);
                LoadVector(kF, kGm + qo, kDim);
                LoadVector(vF, vGm + vo, 64);
                L2Norm(qF, scale);
                L2Norm(kF, 1.0f);
                float beta = LoadBeta((uint64_t)t * vHeads + h);
                LoadDecayVec(((uint64_t)t * kHeads + kh) * kDim, decayHead);
                ApplyDecay();
                MatVec(kF, delta);
                vsub(delta, vF, delta, 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vmuls(delta, delta, beta, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vbrcb((__ubuf__ uint32_t *)deltaBlocks, (__ubuf__ uint32_t *)delta, 1, 8, 8);
                pipe_barrier(PIPE_V);
                uint32_t rowStride = kDim / 8;
                for (uint32_t col = 0; col < kDim; col += 64) {
                    SetMask(MIN(64u, kDim - col));
                    vmul(product + col, deltaBlocks, kF + col, 64, 1, 0, 1, rowStride, 1, 0);
                    pipe_barrier(PIPE_V);
                    vadd(stateF + col, stateF + col, product + col, 64, 1, 1, 1, rowStride,
                         rowStride, rowStride);
                    pipe_barrier(PIPE_V);
                }
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
                MatVec(qF, outF);
                StoreVector(outGm + vo, outF, 64);
            }
            StoreState(offset);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t idx = GetBlockIdx(); idx < batch * vHeads; idx += GetBlockNum()) {
            ProcessHead(idx / vHeads, idx % vHeads);
        }
    }

private:
    __gm__ Dtype *qGm, *kGm, *vGm, *betaGm, *gateGm, *outGm;
    __gm__ float *aLogGm, *dtBiasGm;
    GM_ADDR stateGm;
    __gm__ int32_t *indicesGm, *cachedLens;
    uint64_t tableRowStride;
    uint32_t tableColumns, stateBlockSize, aligned;
    __ubuf__ Dtype *ioTmp, *stateTmp;
    __ubuf__ float *qF, *kF, *vF, *gF, *dtBiasF, *onesF, *stateF, *product, *delta, *outF;
    __ubuf__ float *reduceTail, *deltaBlocks, *scalarF;
    uint32_t batch, kHeads, vHeads, kDim, vDim, numPages, stateFp32;
    uint64_t pageStride;
    float scale, lowerBound, normEps;
};
#define RECURRENT_GATED_DELTA_RULE_KDA_PAGED_BODY(dtype)                                     \
    RecurrentGatedDeltaRuleKdaPaged<dtype> op;                                               \
    op.Init(query, key, value, beta, gate, aLog, dtBias, state, out, indices, batch, kHeads, \
            vHeads, kDim, vDim, pageStride, numPages, stateFp32, scale, lowerBound, normEps, \
            cachedLens, tableRowStride, tableColumns, stateBlockSize, aligned);              \
    op.Process();
#else
#define RECURRENT_GATED_DELTA_RULE_KDA_PAGED_BODY(dtype)
#endif
#define RECURRENT_GATED_DELTA_RULE_KDA_PAGED_FUNC_DEFINE(dtype)                              \
    extern "C" __global__ __aicore__ void recurrent_gated_delta_rule_kda_paged_##dtype(      \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR gate, GM_ADDR aLog, \
        GM_ADDR dtBias, GM_ADDR state, GM_ADDR out, GM_ADDR indices, uint32_t batch,         \
        uint32_t kHeads, uint32_t vHeads, uint32_t kDim, uint32_t vDim, uint64_t pageStride, \
        uint32_t numPages, uint32_t stateFp32, float scale, float lowerBound, float normEps, \
        GM_ADDR cachedLens, uint64_t tableRowStride, uint32_t tableColumns,                  \
        uint32_t stateBlockSize, uint32_t aligned)                                           \
    {                                                                                        \
        RECURRENT_GATED_DELTA_RULE_KDA_PAGED_BODY(dtype)                                     \
    }

#ifdef __DAV_C220_VEC__
#define RECURRENT_GATED_DELTA_RULE_KDA_PREFILL_PAGED_BODY(dtype)                             \
    RecurrentGatedDeltaRuleKdaPaged<dtype> op;                                               \
    op.Init(query, key, value, beta, gate, aLog, dtBias, state, out, indices, batch, kHeads, \
            vHeads, kDim, vDim, pageStride, numPages, stateFp32, scale, lowerBound, normEps, \
            nullptr, 0, 0, 0, 0);                                                            \
    op.ProcessPrefill(starts, initial, tokens);
#else
#define RECURRENT_GATED_DELTA_RULE_KDA_PREFILL_PAGED_BODY(dtype)
#endif
#define RECURRENT_GATED_DELTA_RULE_KDA_PREFILL_PAGED_FUNC_DEFINE(dtype)                         \
    extern "C" __global__ __aicore__ void recurrent_gated_delta_rule_kda_prefill_paged_##dtype( \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR gate, GM_ADDR aLog,    \
        GM_ADDR dtBias, GM_ADDR state, GM_ADDR out, GM_ADDR indices, uint32_t batch,            \
        uint32_t kHeads, uint32_t vHeads, uint32_t kDim, uint32_t vDim, uint64_t pageStride,    \
        uint32_t numPages, uint32_t stateFp32, float scale, float lowerBound, float normEps,    \
        GM_ADDR starts, GM_ADDR initial, uint32_t tokens)                                       \
    {                                                                                           \
        RECURRENT_GATED_DELTA_RULE_KDA_PREFILL_PAGED_BODY(dtype)                                \
    }
