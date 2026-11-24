/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * DeepSeek-V4.1 engram gate: fused post-wkv gate + residual inject on the
 * residual stream (AIV). wkv (a matmul) runs outside; this op consumes its
 * [T, (HC+1)*DIM] output (HC*DIM key rows ++ DIM value row) and writes hidden
 * in place.
 *
 * Per token t, per hc branch b (HC branches):
 *   rstd = rsqrt(Σh²/DIM+eps) * rsqrt(Σkey²/DIM+eps)
 *   dot  = Σ(h · (qw*kw) · key) · rstd · DIM^-0.5
 *   mag  = sqrt(max(|dot|, 1e-6))
 *   gate = sigmoid(copysign(mag, dot))   (dot≥0 -> +mag, dot<0 -> -mag)
 *   gate = mask[t] ? gate : 0
 *   out[t,b] = hResid[t,b] + gate · value[t]
 *
 * Rotated checkpoints have two hidden bases: the gate-math (dot/rstd) runs on
 * the un-rotated hidden (hidden @ R.T), the inject adds onto the rotated
 * residual stream (hiddenResidual). On unrotated checkpoints the caller may
 * omit hiddenResidual (host default, ptr==nullptr) or alias hidden's storage;
 * nullptr aliases hidden (no extra GM load). resIn is always loaded and the
 * FMA always injects onto resIn.
 *
 * Layout:
 *   hidden, hiddenResidual, out  [T, HC, DIM] bf16, stride-1 on DIM
 *                               (hiddenResidual may == hidden)
 *   kv     [T, (HC+1)*DIM] bf16  — [t] = key[t, 0..HC*DIM) ++ value[t, DIM)
 *   qw, kw [HC, DIM] bf16
 *   mask   [T] int32 (nonzero = active)
 *
 * One block strides over (token*HC + branch) pairs by block_num: one branch's
 * hidden[DIM] per work item. DIM (V4.1 = 5120) fits a single UB row, loaded
 * whole and not tiled. Single-pass per work item (load -> reduce -> FMA ->
 * store) with inter-work-item ping-pong.
 *
 * Flag protocol (in/out buffers separated):
 *   V<->MTE2 ID0/ID1   h/key ping-pong
 *   V<->MTE3 ID2/ID3   out ping-pong
 *   V<->MTE2 ID4       qw/kw (shared slot, counting-semaphore)
 *   V<->MTE2 ID5/ID6   val/res ping-pong
 * V-internal ordering is pipe_barrier(PIPE_V) only. Scalar readback
 * (hh/kk/hk/rstd/mag/gate) uses V->S EVENT_ID0; each read is acked S->V
 * EVENT_ID0 before V reuses the landing word (calcH/calcKey/scalarRstdH).
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
__aicore__ void engram_gate(__gm__ Dtype *hidden, __gm__ Dtype *hiddenResidual, __gm__ Dtype *kv,
                            __gm__ Dtype *qw, __gm__ Dtype *kw, __gm__ int32_t *mask,
                            __gm__ Dtype *out, uint32_t numTokens, uint32_t hcMult, uint32_t dim,
                            float eps)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_NUM_OF_FP32;  // fp32 elems per full vector repeat
    const int repeat = DIV_ROUND_UP(static_cast<int>(dim), calcPad);
    const uint64_t dtypeLen = ROUND_UP(dim * sizeof(Dtype), UB_BUF_ALIGN_SIZE);
    const uint64_t f32Len = ROUND_UP(dim * sizeof(float), UB_BUF_ALIGN_SIZE);

    // UB layout (DIM=5120: bf16 row=10KB, fp32=20KB; ~190KB < 196KB A3).
    // Ping-pong in/out buffers separated so MTE2 load and MTE3 store never
    // target the same slot:
    //   hIn{0,1}    bf16 h row (gate-math basis)
    //   keyIn{0,1}  bf16 key row
    //   valIn{0,1}  bf16 value row (read once, in the FMA)
    //   resIn{0,1}  bf16 residual row (FMA inject base — always loaded)
    //   outIn{0,1}  bf16 out staging
    //   qwIn, kwIn  bf16 per-branch constants (ID4, shared slot)
    // Single full-dim buffers:
    //   qwF            fp32 (qw·kw precomputed)
    //   calcH/calcKey  fp32 scratch (reused across gate-math and FMA)
    //   scalar*        scalar landing (rstdH/K/mag/gate)
    // kwF is folded into calcKey (kw converts into calcKey, then qwF=qwF·kw
    // before calcKey is needed for the key convert).
    uint64_t off = 0;
    __ubuf__ Dtype *hIn0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *hIn1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *keyIn0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *keyIn1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *valIn0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *valIn1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *resIn0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *resIn1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *outIn0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *outIn1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *qwIn = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ Dtype *kwIn = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += dtypeLen;
    __ubuf__ float *qwF = reinterpret_cast<__ubuf__ float *>(off);
    off += f32Len;
    __ubuf__ float *calcH = reinterpret_cast<__ubuf__ float *>(off);
    off += f32Len;
    __ubuf__ float *calcKey = reinterpret_cast<__ubuf__ float *>(off);
    off += f32Len;
    // Scalar landing buffers: distinct UB words so V's write to one cannot race
    // S's pending read of another.
    __ubuf__ float *scalarRstdH = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(sizeof(float), UB_BUF_ALIGN_SIZE);
    __ubuf__ float *scalarRstdK = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(sizeof(float), UB_BUF_ALIGN_SIZE);
    __ubuf__ float *scalarMag = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(sizeof(float), UB_BUF_ALIGN_SIZE);
    __ubuf__ float *scalarGate = reinterpret_cast<__ubuf__ float *>(off);
    off += ROUND_UP(sizeof(float), UB_BUF_ALIGN_SIZE);
    assert(off <= UB_SIZE);

    __ubuf__ Dtype *hIn[2] = {hIn0, hIn1};
    __ubuf__ Dtype *keyIn[2] = {keyIn0, keyIn1};
    __ubuf__ Dtype *valIn[2] = {valIn0, valIn1};
    __ubuf__ Dtype *resIn[2] = {resIn0, resIn1};
    __ubuf__ Dtype *outIn[2] = {outIn0, outIn1};

    pipe_barrier(PIPE_V);

    // total work = numTokens * hcMult, strided across blocks.
    const uint32_t totalWork = numTokens * hcMult;
    const uint32_t kvStrideT = (hcMult + 1) * dim;  // kv row stride in elements
    const float invDim = (float)1.0 / dim;
    // Precompute dim^-0.5 via vsqrt (AIV has no scalar sqrtf).
    SetMask(1);
    vector_dup(scalarRstdH, 1.0f / dim, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    vsqrt(scalarRstdH, scalarRstdH, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const float invSqrtDim = *scalarRstdH;
    // Ack the scalar read before V rewrites scalarRstdH (vector_dup in the loop).
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    // Prime: ID0/ID1 -> h/key slots; ID2/ID3 -> out slots; ID4 -> qw/kw slot;
    // ID5/ID6 -> val/res slots.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);

    int curr = 0;
    for (uint32_t work = block_idx; work < totalWork; work += uint32_t(block_num)) {
        const uint32_t token = work / hcMult;
        const uint32_t branch = work % hcMult;
        const uint32_t branchDimOff = branch * dim;
        const __gm__ Dtype *gmH = hidden + (uint64_t)token * hcMult * dim + branchDimOff;
        // gmHResid: residual stream (rotated basis). On unrotated checkpoints the
        // caller may pass nullptr (host default) or alias hidden's storage; both
        // mean "inject onto hidden". nullptr aliases gmH (no extra GM load).
        const __gm__ Dtype *gmHResid =
            (hiddenResidual != nullptr)
                ? hiddenResidual + (uint64_t)token * hcMult * dim + branchDimOff
                : gmH;
        const __gm__ Dtype *gmKey = kv + (uint64_t)token * kvStrideT + branchDimOff;
        const __gm__ Dtype *gmVal = kv + (uint64_t)token * kvStrideT + hcMult * dim;
        const __gm__ Dtype *gmOut = out + (uint64_t)token * hcMult * dim + branchDimOff;
        const uint32_t dBytes = dim * sizeof(Dtype);

        // --- Load qw/kw (ID4) and precompute qw·kw into qwF. ---
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        CopyGmToUbufAligned(qwIn, qw + branchDimOff, dBytes);
        CopyGmToUbufAligned(kwIn, kw + branchDimOff, dBytes);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        convert_input(qwF, qwIn, repeat);
        convert_input(calcKey, kwIn, repeat);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);

        vmul(qwF, qwF, calcKey, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // --- Load h/key (ID0+curr) and val/res (ID5+curr). ---
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(hIn[curr], gmH, dBytes);
        CopyGmToUbufAligned(keyIn[curr], gmKey, dBytes);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5 + curr);
        CopyGmToUbufAligned(valIn[curr], gmVal, dBytes);
        CopyGmToUbufAligned(resIn[curr], gmHResid, dBytes);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5 + curr);

        // hh = Σ h², kk = Σ key².
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        convert_input(calcH, hIn[curr], repeat);
        convert_input(calcKey, keyIn[curr], repeat);
        pipe_barrier(PIPE_V);
        vmul(calcH, calcH, calcH, repeat, 1, 1, 1, 8, 8, 8);
        vmul(calcKey, calcKey, calcKey, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        ReduceSum(calcH, calcH, dim);
        ReduceSum(calcKey, calcKey, dim);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float hh = *calcH;
        float kk = *calcKey;
        // Ack the scalar reads before V reuses calcH/calcKey (convert_input below).
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

        // hk = Σ (h · (qw·kw) · key). Reconvert h/key from UB (overwrite the
        // reduce scalars). h/key slot curr's last V read.
        convert_input(calcH, hIn[curr], repeat);
        convert_input(calcKey, keyIn[curr], repeat);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);  // release h/key slot curr
        vmul(calcH, calcH, qwF, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(calcH, calcH, calcKey, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        ReduceSum(calcH, calcH, dim);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float hk = *calcH;
        // Ack the scalar read before V reuses calcH/calcKey (convert_input below).
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

        // Convert res/value into calcH/calcKey (reused scratch — holding res/value).
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5 + curr);
        convert_input(calcH, resIn[curr], repeat);
        convert_input(calcKey, valIn[curr], repeat);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5 + curr);  // release val/res slot curr

        // --- Derive gate (scalar math). ---
        SetMask(1);
        vector_dup(scalarRstdH, hh * invDim + eps, 1, 1, 1, 8, 0);
        vector_dup(scalarRstdK, kk * invDim + eps, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vsqrt(scalarRstdH, scalarRstdH, 1, 1, 1, 8, 8);
        vsqrt(scalarRstdK, scalarRstdK, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float rstdH = 1.0f / *scalarRstdH;
        float rstdK = 1.0f / *scalarRstdK;
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        float rstd = rstdH * rstdK;

        // dot = hk * rstd * DIM^-0.5; mag = sqrt(max(|dot|, 1e-6)).
        float dot = hk * rstd * invSqrtDim;
        float absDot = dot < 0.0f ? -dot : dot;
        float magSrc = absDot > 1e-6f ? absDot : 1e-6f;
        SetMask(1);
        vector_dup(scalarMag, magSrc, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vsqrt(scalarMag, scalarMag, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float mag = *scalarMag;
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        // copysign(mag, dot): negate mag when dot's sign bit is set.
        union {
            float f;
            int32_t i;
        } u;
        u.f = dot;
        float signedMag = (u.i & 0x80000000) ? -mag : mag;
        // gate = sigmoid(signedMag) = 1/(1+exp(-signedMag)). Reciprocal via
        // scalar divide (not vdiv/vrec).
        SetMask(1);
        vector_dup(scalarGate, -signedMag, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vexp(scalarGate, scalarGate, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(scalarGate, scalarGate, 1.0f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float gate = 1.0f / *scalarGate;

        // mask: gate = mask[t] ? gate : 0
        if (!mask[token]) {
            gate = 0.0f;
        }

        // --- FMA + store: out = res + gate·value. ---
        vmuls(calcKey, calcKey, gate, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(calcH, calcH, calcKey, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        convert_output(outIn[curr], calcH, repeat);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
        CopyUbufToGmAligned(gmOut, outIn[curr], dBytes);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        curr = 1 - curr;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
}

#define ENGRAM_GATE_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void engram_gate_##dtype(                                    \
        GM_ADDR hidden, GM_ADDR hiddenResidual, GM_ADDR kv, GM_ADDR qw, GM_ADDR kw, GM_ADDR mask, \
        GM_ADDR out, uint32_t numTokens, uint32_t hcMult, uint32_t dim, float eps)                \
    {                                                                                             \
        engram_gate((__gm__ dtype *)hidden, (__gm__ dtype *)hiddenResidual, (__gm__ dtype *)kv,   \
                    (__gm__ dtype *)qw, (__gm__ dtype *)kw, (__gm__ int32_t *)mask,               \
                    (__gm__ dtype *)out, numTokens, hcMult, dim, eps);                            \
    }
#else
#define ENGRAM_GATE_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void engram_gate_##dtype(                                    \
        GM_ADDR hidden, GM_ADDR hiddenResidual, GM_ADDR kv, GM_ADDR qw, GM_ADDR kw, GM_ADDR mask, \
        GM_ADDR out, uint32_t numTokens, uint32_t hcMult, uint32_t dim, float eps)                \
    {                                                                                             \
    }
#endif
