/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"

// hc_act: per-token activation for Hyper-Connection pre/post/comb (fp32 only).
//   pre  = sigmoid(mixes[:, :hcMult]   * scalePre  + basePre)  + eps
//   post = 2*sigmoid(mixes[:, hcMult:2*hcMult] * scalePost + basePost)
//   comb = sinkhorn(softmax(mixes[:, 2*hcMult:] * scaleComb + baseComb) + eps)
// headOnly: hc_head path — mixes is [m, hcMult], base is [hcMult], scale is [1]; only pre
// runs (no post/comb/sinkhorn); post/comb GM ptrs may be null (never dereferenced).

#ifdef __DAV_C220_VEC__
__aicore__ inline void hc_col_normalize(__ubuf__ float *comb, __ubuf__ float *colBuf,
                                        uint32_t hcMult, uint32_t combRowStride, float eps)
{
    // colSum = Σ_r row_r (cross-row same-column add), +eps, then comb[r][c] /= colSum[c].
    copy_ubuf_to_ubuf(colBuf, comb, 0, 1, 1, 0, 0);  // colSum = row0
    pipe_barrier(PIPE_V);
    set_mask_norm();
    SetMask(hcMult);
    for (uint32_t r = 1; r < hcMult; r++) {
        vadd(colBuf, colBuf, comb + r * combRowStride, 1, 1, 1, 1, combRowStride, combRowStride,
             combRowStride);
        pipe_barrier(PIPE_V);
    }
    vadds(colBuf, colBuf, eps, 1, 1, 1, combRowStride, combRowStride);  // denom = colSum + eps
    pipe_barrier(PIPE_V);
    vdiv(comb, comb, colBuf, hcMult, 1, 1, 0, 1, 1, 0);
    pipe_barrier(PIPE_V);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

__aicore__ inline void hc_row_normalize(__ubuf__ float *comb, __ubuf__ float *scalarBuf,
                                        __ubuf__ float *brcbBuf, uint32_t hcMult,
                                        uint32_t combRowStride, int combRep, float eps)
{
    // rowSum = Σ_c comb[r][c], +eps, broadcast per row, then comb /= rowSum.
    set_mask_norm();
    SetMask(hcMult);
    vcadd(scalarBuf, comb, hcMult, 1, 1, 1, 0);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_V);
    vadds(scalarBuf, scalarBuf, eps, combRep, 1, 1, 8, 8);  // denom = rowSum + eps
    pipe_barrier(PIPE_V);
    vbrcb((__ubuf__ uint32_t *)brcbBuf, (__ubuf__ uint32_t *)scalarBuf, 1, 8, 1);
    pipe_barrier(PIPE_V);
    vdiv(comb, comb, brcbBuf, combRep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
}

__aicore__ inline void hc_softmax_rows(__ubuf__ float *comb, __ubuf__ float *scalarBuf,
                                       __ubuf__ float *brcbBuf, uint32_t hcMult,
                                       uint32_t combRowStride, int combRep, float eps)
{
    // softmax over rows: rowMax → -= max → exp → rowSum → /= sum, +eps.
    set_mask_norm();
    SetMask(hcMult);
    vcmax(scalarBuf, comb, hcMult, 1, 1, 1, Order_t::ONLY_VALUE);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_V);
    vbrcb((__ubuf__ uint32_t *)brcbBuf, (__ubuf__ uint32_t *)scalarBuf, 1, 8, 1);
    pipe_barrier(PIPE_V);
    vsub(comb, comb, brcbBuf, combRep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vexp(comb, comb, combRep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    SetMask(hcMult);
    vcadd(scalarBuf, comb, hcMult, 1, 1, 1, 0);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    pipe_barrier(PIPE_V);
    vbrcb((__ubuf__ uint32_t *)brcbBuf, (__ubuf__ uint32_t *)scalarBuf, 1, 8, 1);
    pipe_barrier(PIPE_V);
    vdiv(comb, comb, brcbBuf, combRep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(comb, comb, eps, combRep, 1, 1, 8, 8);  // softmax + eps
    pipe_barrier(PIPE_V);
}

template <typename Dtype>
__aicore__ inline void hc_act(__gm__ float *mixes, __gm__ float *hcBase, __gm__ float *post,
                              __gm__ float *comb, __gm__ float *hcScale, uint32_t m,
                              uint32_t hcMult, float eps, uint32_t sinkhornIters, uint32_t headOnly,
                              __gm__ Dtype *xResid, __gm__ Dtype *yOut, uint32_t hidden)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);  // fp32 elems per full vector
    const uint32_t mixHc = headOnly ? hcMult : (2 + hcMult) * hcMult;
    // Bit-masks below assume all segments fit in one fp32 vector (64 lanes): pre/post/comb
    // together occupy at most 2*hcMult + hcMult*hcMult bits, and (1ULL << (hcMult*hcMult)) is
    // UB at hcMult>=8. Enforce hcMult in [1,7].
    assert(hcMult >= 1 && hcMult <= 7);
    const uint32_t combElems = hcMult * hcMult;
    const uint32_t prePostBytes = hcMult * sizeof(float);
    const uint64_t lenPrePost = ROUND_UP(prePostBytes, VECTOR_MAX_BYTESIZE);
    int repeat = DIV_ROUND_UP(hcMult, calcPad);

    const uint64_t maskPre = (hcMult >= 64) ? (uint64_t)-1 : ((1ULL << hcMult) - 1);
    const uint64_t maskPost = maskPre << hcMult;
    const uint64_t maskPrePost = maskPre | (headOnly ? 0 : maskPost);
    const uint64_t maskComb = headOnly ? 0 : (((1ULL << (hcMult * hcMult)) - 1) << (2 * hcMult));
    const uint64_t maskAll = maskPrePost | maskComb;

    const uint64_t baseUbBytes = mixHc * sizeof(float);
    const uint64_t lenBaseUb = ROUND_UP(baseUbBytes, VECTOR_MAX_BYTESIZE);

    uint64_t off = 0;
    __ubuf__ float *mixesUb0 = (__ubuf__ float *)off;
    off += lenBaseUb;
    __ubuf__ float *mixesUb1 = (__ubuf__ float *)off;
    off += lenBaseUb;
    __ubuf__ float *mixesUb[2] = {mixesUb0, mixesUb1};

    __ubuf__ float *calcUb = (__ubuf__ float *)off;
    off += lenBaseUb;

    __ubuf__ float *postOut0 = (__ubuf__ float *)off;
    off += lenPrePost;
    __ubuf__ float *postOut1 = (__ubuf__ float *)off;
    off += lenPrePost;
    __ubuf__ float *combOut0 = (__ubuf__ float *)off;
    off += lenBaseUb;
    __ubuf__ float *combOut1 = (__ubuf__ float *)off;
    off += lenBaseUb;
    __ubuf__ float *postOut[2] = {postOut0, postOut1};
    __ubuf__ float *combOut[2] = {combOut0, combOut1};

    const uint32_t hcMultAlign = ROUND_UP(hcMult, BLOCK_SIZE / sizeof(float));    // hcMult=4 → 8
    const uint64_t combAlignBytes = hcMult * hcMultAlign * sizeof(float);         // 4*8*4 = 128B
    const uint64_t lenCombAlign = ROUND_UP(combAlignBytes, VECTOR_MAX_BYTESIZE);  // 256B
    const int combRep =
        DIV_ROUND_UP(hcMult * hcMultAlign, calcPad);  // whole-block vector repeats over combAlign
    __ubuf__ float *combAlign = (__ubuf__ float *)off;
    off += lenCombAlign;

    __ubuf__ float *baseUb = (__ubuf__ float *)off;
    off += lenBaseUb;

    __ubuf__ float *colBuf = (__ubuf__ float *)off;  // colSum scratch (col-norm broadcast source)
    off += lenPrePost;

    __ubuf__ float *scalarBuf = (__ubuf__ float *)off;
    off += lenCombAlign;
    __ubuf__ float *brcbBuf = (__ubuf__ float *)off;
    off += lenCombAlign;

    __ubuf__ float *onesUb = (__ubuf__ float *)off;  // 1.0 for vdiv-based 1/x (avoids vrec)
    off += lenPrePost;
    __ubuf__ float *scaleUb = (__ubuf__ float *)off;  // {scalePre[,scalePost,scaleComb]} from GM
    off += lenPrePost;
    __ubuf__ uint32_t *offRamp = (__ubuf__ uint32_t *)off;
    off += lenPrePost;

    const uint64_t residDtypeLen = ROUND_UP(hcMult * hidden * sizeof(Dtype), UB_BUF_ALIGN_SIZE);
    const uint64_t yDtypeLen = ROUND_UP(hidden * sizeof(Dtype), UB_BUF_ALIGN_SIZE);
    const uint64_t residFp32Len = ROUND_UP(hcMult * hidden * sizeof(float), UB_BUF_ALIGN_SIZE);
    const uint64_t mergeFp32Len = ROUND_UP(hidden * sizeof(float), UB_BUF_ALIGN_SIZE);
    // IO slot 0: residual [hcMult, hidden] bf16 in, y [hidden] bf16 out.
    __ubuf__ Dtype *inDtype0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += residDtypeLen;
    __ubuf__ Dtype *outDtype0 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += yDtypeLen;
    // IO slot 1
    __ubuf__ Dtype *inDtype1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += residDtypeLen;
    __ubuf__ Dtype *outDtype1 = reinterpret_cast<__ubuf__ Dtype *>(off);
    off += yDtypeLen;
    __ubuf__ float *xFp32 = (__ubuf__ float *)off;  // whole [hcMult, hidden] residual as fp32
    off += residFp32Len;
    __ubuf__ float *yCalc = (__ubuf__ float *)off;  // accumulator: sum_h pre[h]*x (V reused)
    off += mergeFp32Len;
    __ubuf__ Dtype *inDtypeArr[2] = {inDtype0, inDtype1};
    __ubuf__ Dtype *outDtypeArr[2] = {outDtype0, outDtype1};
    assert(off <= UB_SIZE);

    CopyGmToUbufAligned(baseUb, hcBase, baseUbBytes);
    CopyGmToUbufAligned(scaleUb, hcScale, headOnly ? sizeof(float) : 3 * sizeof(float));
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    float scalePre = scaleUb[0];
    float scalePost = headOnly ? 0.0f : scaleUb[1];
    float scaleComb = headOnly ? 0.0f : scaleUb[2];
    vector_dup(onesUb, 1.0f, repeat, 1, 1, 8, 0);
    if (!headOnly) {
        for (int p = 0; p < hcMult; p++) {
            offRamp[p] = static_cast<uint32_t>(p * sizeof(float));
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID4);
    }

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    if (!headOnly) {
        // comb writeback ping-pong uses EVENT_ID2/3; head mode skips it entirely.
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    // Merge loop-invariant constants (hidden is the per-row merge dim, fixed across tokens).
    const int vecRep =
        DIV_ROUND_UP(hidden, calcPad);     // vector repeats over hidden (calcPad=64 fp32/repeat)
    const int totalRep = hcMult * vecRep;  // fp32 repeats over the whole [hcMult, hidden] residual
    const uint32_t dBytes = hidden * sizeof(Dtype);
    int curr = 0;
    for (uint32_t process = block_idx; process < m; process += uint32_t(block_num)) {
        __gm__ float *mixPtr = mixes + process * mixHc;
        __gm__ Dtype *xBase = xResid + (uint64_t)process * hcMult * hidden;
        __gm__ Dtype *yRow = yOut + (uint64_t)process * hidden;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        CopyGmToUbufAligned(mixesUb[curr], mixPtr, baseUbBytes);
        for (uint32_t h = 0; h < hcMult; h++) {
            CopyGmToUbufAligned(inDtypeArr[curr] + h * hidden, xBase + h * hidden, dBytes);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

        SetMask(hcMult);
        vmuls(calcUb, mixesUb[curr], scalePre, repeat, 1, 1, 8, 8);
        if (!headOnly) {
            set_vector_mask(0x0, maskPost);
            vmuls(calcUb, mixesUb[curr], scalePost, repeat, 1, 1, 8, 8);
            set_vector_mask(0x0, maskComb);
            vmuls(calcUb, mixesUb[curr], scaleComb, repeat, 1, 1, 8, 8);
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        // Cast residual bf16 [hcMult, hidden] -> fp32 in VECTOR_MAX_REPEAT chunks.
        for (int chunk = 0; chunk < totalRep; chunk += VECTOR_MAX_REPEAT) {
            const int rep =
                (totalRep - chunk) < VECTOR_MAX_REPEAT ? (totalRep - chunk) : VECTOR_MAX_REPEAT;
            convert_input(xFp32 + chunk * calcPad, inDtypeArr[curr] + chunk * calcPad, rep);
        }
        vector_dup(yCalc, 0.0f, vecRep, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

        // `mixes*scale + base` is shared by pre/post/comb.
        set_vector_mask(0x0, maskAll);
        vadd(calcUb, calcUb, baseUb, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // sigmoid chain runs only on pre+post lanes (comb is not sigmoided).
        set_vector_mask(0x0, maskPrePost);
        vmuls(calcUb, calcUb, -1.0f, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(calcUb, calcUb, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(calcUb, calcUb, 1.0f, repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(calcUb, onesUb, calcUb, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // pre = sigmoid + eps (maskPre, always); post = 2*sigmoid, no eps (maskPost,
        // headOnly skips it).
        set_vector_mask(0x0, maskPre);
        vadds(calcUb, calcUb, eps, repeat, 1, 1, 8, 8);
        if (!headOnly) {
            set_vector_mask(0x0, maskPost);
            vmuls(calcUb, calcUb, 2.0f, repeat, 1, 1, 8, 8);
            set_mask_norm();
            SetMask(hcMult);
            for (uint32_t r = 0; r < hcMult; r++) {
                uint32_t rowBase = static_cast<uint32_t>(
                    reinterpret_cast<uint64_t>(calcUb + 2 * hcMult + r * hcMult));
                vgather((__ubuf__ uint32_t *)(combAlign + r * hcMultAlign), offRamp, rowBase, 8, 1);
            }
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID4);
        for (uint32_t h = 0; h < hcMult; h++) {
            float preH = calcUb[h];
            vaxpy(yCalc, xFp32 + h * hidden, preH, vecRep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6 + curr);
        convert_output(outDtypeArr[curr], yCalc, vecRep);
        if (!headOnly) {
            if (process == block_idx) {
                wait_flag(PIPE_S, PIPE_V, EVENT_ID4);
            }
            SetMask(hcMult);
            uint32_t baseAddr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(calcUb + hcMult));
            vgather((__ubuf__ uint32_t *)postOut[curr], offRamp, baseAddr, 8, 1);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6 + curr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6 + curr);
        CopyUbufToGmAligned(yRow, outDtypeArr[curr], dBytes);
        if (!headOnly) {
            __gm__ float *postOutPtr = post + process * hcMult;
            CopyUbufToGmAligned(postOutPtr, postOut[curr], prePostBytes);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6 + curr);

        // for comb
        if (!headOnly) {
            // Softmax over rows of combAlign + eps.
            hc_softmax_rows(combAlign, scalarBuf, brcbBuf, hcMult, hcMultAlign, combRep, eps);

            // Sinkhorn: col-norm once, then (row-norm, col-norm)*(iters-1).
            hc_col_normalize(combAlign, colBuf, hcMult, hcMultAlign, eps);
            for (uint32_t it = 0; it + 1 < sinkhornIters; it++) {
                hc_row_normalize(combAlign, scalarBuf, brcbBuf, hcMult, hcMultAlign, combRep, eps);
                hc_col_normalize(combAlign, colBuf, hcMult, hcMultAlign, eps);
            }

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            copy_ubuf_to_ubuf(combOut[curr], combAlign, 0, hcMult, 1, 0, 0);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
            for (uint32_t r = 0; r < hcMult; r++) {
                CopyUbufToGmAligned(comb + process * combElems + r * hcMult,
                                    combOut[curr] + r * hcMultAlign, hcMult * sizeof(float));
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        }
        curr = 1 - curr;
    }
    // EVENT_ID2/3 are comb-only (head skips them); trailing waits must match what was set.
    if (!headOnly) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

#define HC_ACT_FUNC_DEFINE                                                                      \
    extern "C" __global__ __aicore__ void hc_act_float(                                         \
        GM_ADDR mixes, GM_ADDR hcBase, GM_ADDR post, GM_ADDR comb, GM_ADDR hcScale, uint32_t m, \
        uint32_t hcMult, float eps, uint32_t sinkhornIters, uint32_t headOnly, GM_ADDR xResid,  \
        GM_ADDR yOut, uint32_t hidden)                                                          \
    {                                                                                           \
        hc_act<bfloat16_t>((__gm__ float *)mixes, (__gm__ float *)hcBase, (__gm__ float *)post, \
                           (__gm__ float *)comb, (__gm__ float *)hcScale, m, hcMult, eps,       \
                           sinkhornIters, headOnly, (__gm__ bfloat16_t *)xResid,                \
                           (__gm__ bfloat16_t *)yOut, hidden);                                  \
    }
#else
#define HC_ACT_FUNC_DEFINE                                                                      \
    extern "C" __global__ __aicore__ void hc_act_float(                                         \
        GM_ADDR mixes, GM_ADDR hcBase, GM_ADDR post, GM_ADDR comb, GM_ADDR hcScale, uint32_t m, \
        uint32_t hcMult, float eps, uint32_t sinkhornIters, uint32_t headOnly, GM_ADDR xResid,  \
        GM_ADDR yOut, uint32_t hidden)                                                          \
    {                                                                                           \
    }
#endif
