/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#ifdef __DAV_C220_VEC__
constexpr uint64_t SORT_RESULT_BLOCK_SIZE = SORT_BLOCK_SIZE * 2;
constexpr uint32_t BIT_SIZE_OF_U32 = 32;
// Max input_ids prefetched into UB per batch. The S-pipe reads inputIdsArr directly; a fresh
// batch is DMA'd when the window slides past baseInputIdsIdx + HASH_INPUT_IDS_BATCH. Caps UB.
constexpr uint32_t HASH_INPUT_IDS_BATCH = 256;

// V4 MoE gate, sqrtsoftplus only (the part below the matmul):
//   scores = sqrt(softplus(scores));  original = scores;  if bias: scores += bias
//   indices = hash ? tid2eid[input_ids] : scores.topk(K)[1]   # bias steers topk only
//   weights = original.gather(indices);  weights /= sum;  weights *= route_scale
// Outputs: w = sparse [M,N] (K nonzero slots, rest 0); r = [M,N] BIT1 bitmap (N bits -> N/32).
// indicesTopK is INTERNAL — drives the sparse write + bitmap, never DMA'd to GM.
template <typename Dtype>
class SqrtsoftplusHashTopK
{
public:
    __aicore__ inline SqrtsoftplusHashTopK() = default;

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR indices, GM_ADDR bias, GM_ADDR inputIds,
                                GM_ADDR tid2eid, GM_ADDR outWeights, GM_ADDR routingMap,
                                float scale, uint32_t numTokens, uint32_t numRoutedExperts,
                                uint32_t topK, uint8_t hash)
    {
        set_mask_norm();
        this->scoresGm = (__gm__ Dtype *)scores;
        this->indicesGm = (__gm__ uint32_t *)indices;
        this->biasGm = (__gm__ float *)bias;
        this->inputIdsGm = (__gm__ uint32_t *)inputIds;
        this->tid2eidGm = (__gm__ uint32_t *)tid2eid;
        this->outWeightsGm = (__gm__ Dtype *)outWeights;
        this->routingMapGm = (__gm__ uint32_t *)routingMap;

        this->nTokens = numTokens;
        this->nRoutedExperts = numRoutedExperts;
        this->topK = topK;
        this->hash = hash;
        this->scale = scale;
        this->calcRepeat = DIV_ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32);

        uint32_t padN = ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32) * sizeof(float);
        uint32_t padNDtype =
            ROUND_UP(nRoutedExperts, VECTOR_MAX_BYTESIZE / sizeof(Dtype)) * sizeof(Dtype);
        // topK bufs padded to a full 256B block (ReduceSum/vbrcb/vreducev2 assume block-aligned).
        uint32_t padK = ROUND_UP(topK, VECTOR_MAX_NUM_OF_FP32) * sizeof(float);
        uint64_t off = 0;
        // INPUT: MTE2 fills scoresInTmp[curr] (Dtype, ping-pong); V relays into the SINGLE fp32
        // scoresIn (V-private, no MTE touch). bias + identity indices copied ONCE at Run start.
        scoresInTmp[0] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += padNDtype;
        scoresInTmp[1] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += padNDtype;
        scoresIn = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += padN;
        biasIn = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);  // shared across tokens
        off += padN;
        indicesIn = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);  // identity 0..N-1
        off += padN;
        tid2eidRow[0] =
            reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);  // hash: [topK] row, ping-pong
        off += padK;
        tid2eidRow[1] = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += padK;
        inputIdsArr =
            reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);  // hash: prefetched window
        off += ROUND_UP(HASH_INPUT_IDS_BATCH * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        calc = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += padN;
        calc_unbiased = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += padN;
        reduceTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        sortTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 2 * padN;
        sortMrgTmp = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 2 * padN;
        // COMPUTE (single, V/S-private; slot-reuse hazard handled at OUTPUT DMA stage).
        weightsTopK = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        indicesTopK = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += VECTOR_MAX_BYTESIZE;
        weightsSparse = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += padN;
        // this-> — Init param `routingMap` (GM_ADDR) shadows this UB member.
        this->routingMap = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += padN;
        // OUTPUT DMA staging (ping-pong): V relays compute -> [curr], MTE3 drains to GM.
        weightsOut[0] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += padNDtype;
        weightsOut[1] = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += padNDtype;
        routingMapOut[0] = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += padN;
        routingMapOut[1] = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)off);
        off += padN;
    }

    __aicore__ inline void Run()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        // Copy identity indices + bias ONCE (shared across tokens); fence into V before the loop.
        copy_gm_to_ubuf_align_b32(indicesIn, indicesGm, 0, 1, nRoutedExperts * sizeof(uint32_t), 0,
                                  0, 0, 0);
        if (biasGm != nullptr) {
            copy_gm_to_ubuf_align_b32(biasIn, biasGm, 0, 1, nRoutedExperts * sizeof(float), 0, 0, 0,
                                      0);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);

        // Pre-arm the ping-pong slot-free flags so the first token's wait doesn't hang.
        //   scores : V<->MTE2 ID0+curr.   output : V<->MTE3 ID0+curr.
        //   tid2eid (hash) : V<->MTE2 ID2+curr (distinct track from scores).
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        if (hash) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        }

        int curr = 0;
        for (int tokenIdx = block_idx; tokenIdx < (int)nTokens; tokenIdx += block_num) {
            FetchInputs(tokenIdx, curr);
            CalcScores();
            if (hash) {
                GatherWeights();
            } else {
                SelectTopK();
            }
            NormalizeAndScale();
            FillRoutingMap();
            StageOut(curr);
            CopyOut(tokenIdx, curr);
            curr = 1 - curr;
        }

        // Drain the last token's slot-release sets (no next waiter) so counting flags don't leak
        // +1 into the next launch.
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if (hash) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        }
        pipe_barrier(PIPE_ALL);
    }

    // MTE2 fills scoresInTmp[curr] (Dtype) + relays into fp32 scoresIn (V). hash: also reads
    // inputId (S) -> DMAs the tid2eid row into tid2eidRow[curr] (ping-pong) -> relays into
    // indicesTopK (V). Flag tracks (independent, one counting-semaphore each):
    //   scores:   V->MTE2 ID0+curr (slot free) / MTE2->V ID0+curr (filled).
    //   tid2eid:  V->MTE2 ID2+curr (slot free) / MTE2->V ID2+curr (landed).
    //   addr:     S->MTE2 ID1 — MTE2 must see the S-computed GM row base before the DMA.
    __aicore__ inline void CopyIn(int tokenIdx, int curr)
    {
        // --- scores ---
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        copy_gm_to_ubuf_align_b32(scoresInTmp[curr], scoresGm + tokenIdx * nRoutedExperts, 0, 1,
                                  nRoutedExperts * sizeof(Dtype), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        // --- tid2eid (hash) ---
        __gm__ uint32_t *tid2eidRowGm = nullptr;
        if (hash) {
            uint32_t inputId = inputIdsArr[tokenIdx - baseInputIdsIdx];
            tid2eidRowGm = tid2eidGm + inputId * topK;
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);  // addr ready
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);  // tid2eidRow[curr] free
            // topK (e.g. 6 -> 24B) not 32B-aligned -> CopyGmToUbufAligned (b16/b8 fallback).
            CopyGmToUbufAligned(tid2eidRow[curr], tid2eidRowGm, topK * sizeof(uint32_t));
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
        }
        // --- scores relay (V) ---
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        if constexpr (std::is_same<Dtype, float>::value) {
            copy_ubuf_to_ubuf(scoresIn, scoresInTmp[curr], 0, 1,
                              DIV_ROUND_UP(nRoutedExperts * sizeof(float), BLOCK_SIZE), 0, 0);
        } else {
            uint64_t cfg = set_vector_1src_xt(8, 4, 1, 1, calcRepeat);
            vconv_bf162f32(scoresIn, scoresInTmp[curr], cfg);
        }
        // --- tid2eid relay (V, hash) ---
        if (hash) {
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
            copy_ubuf_to_ubuf(indicesTopK, tid2eidRow[curr], 0, 1,
                              DIV_ROUND_UP(topK * sizeof(uint32_t), BLOCK_SIZE), 0, 0);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
    }

    // calc = sqrt(softplus(scoresIn));  calc_unbiased = sqrt(softplus) (pre-bias, for gather);
    // if bias: calc = calc_unbiased + bias. Numerically-stable softplus = max(x,0) +
    // log(1+exp(-|x|)).
    __aicore__ inline void CalcScores()
    {
        vabs(calc, scoresIn, calcRepeat, 1, 1, 8, 8);  // |x|
        pipe_barrier(PIPE_V);
        vmuls(calc, calc, float(-1), calcRepeat, 1, 1, 8, 8);  // -|x|
        pipe_barrier(PIPE_V);
        vexp(calc, calc, calcRepeat, 1, 1, 8, 8);  // exp(-|x|)
        pipe_barrier(PIPE_V);
        vadds(calc, calc, float(1), calcRepeat, 1, 1, 8, 8);  // 1 + exp(-|x|)
        pipe_barrier(PIPE_V);
        vln(calc, calc, calcRepeat, 1, 1, 8, 8);  // log(1+exp(-|x|))
        pipe_barrier(PIPE_V);
        vrelu((__ubuf__ float *)sortTmp, scoresIn, calcRepeat, 1, 1, 8, 8);  // max(x,0)
        pipe_barrier(PIPE_V);
        vadd(calc, calc, (__ubuf__ float *)sortTmp, calcRepeat, 1, 1, 1, 8, 8, 8);  // softplus
        pipe_barrier(PIPE_V);
        vsqrt(calc_unbiased, calc, calcRepeat, 1, 1, 8, 8);  // original = sqrt(softplus)
        pipe_barrier(PIPE_V);
        if (biasGm != nullptr) {
            vadd(calc, calc_unbiased, biasIn, calcRepeat, 1, 1, 1, 8, 8, 8);  // biased
            pipe_barrier(PIPE_V);
        }
    }

    // All copy/DMA for one token. PrefetchInputIds must run before CopyIn (CopyIn's hash half reads
    // inputIdsArr via S, gated by PrefetchInputIds's MTE2->S).
    __aicore__ inline void FetchInputs(int tokenIdx, int curr)
    {
        if (hash) {
            PrefetchInputIds(tokenIdx);
        }
        CopyIn(tokenIdx, curr);
    }

    // hash: batch-prefetch input_ids into UB (rope window pattern). No-op inside the loaded window;
    // slides past baseInputIdsIdx + BATCH -> next batch (tail-clamped) DMA'd in one shot.
    // MTE2 batch DMA <-> S scalar read, on S<->MTE2 ID0 (distinct from CopyIn's MTE2<->V).
    __aicore__ inline void PrefetchInputIds(int tokenIdx)
    {
        if (baseInputIdsIdx != -1 && tokenIdx < baseInputIdsIdx + (int)HASH_INPUT_IDS_BATCH) {
            return;
        }
        int remain = HASH_INPUT_IDS_BATCH;
        if (tokenIdx + remain > (int)nTokens) {
            remain = nTokens - tokenIdx;
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        CopyGmToUbufAligned(inputIdsArr, inputIdsGm + tokenIdx, remain * sizeof(uint32_t));
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        baseInputIdsIdx = tokenIdx;
    }

    // indicesTopK = biased.topk(K); gather original (calc_unbiased) weights. Multi-block merge
    // via vmrgsort4 (arbitrary N; N=256 -> tailLen=4).
    __aicore__ inline void SelectTopK()
    {
        vbitsort(sortTmp, calc, indicesIn, nRoutedExperts / SORT_BLOCK_SIZE);
        pipe_barrier(PIPE_V);
        uint64_t mrgRepeat = 1;
        uint64_t validBits = 0xF;
        uint64_t ifExhaustedSuspension = 0;
        uint64_t config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
                          ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
        uint64_t lengths =
            SORT_BLOCK_SIZE | SORT_BLOCK_SIZE << 16 | SORT_BLOCK_SIZE << 32 | SORT_BLOCK_SIZE << 48;
        __ubuf__ float *addrArray[4] = {sortTmp, sortTmp + SORT_RESULT_BLOCK_SIZE,
                                        sortTmp + 2 * SORT_RESULT_BLOCK_SIZE,
                                        sortTmp + 3 * SORT_RESULT_BLOCK_SIZE};
        vmrgsort4(sortMrgTmp, addrArray, lengths, config);
        pipe_barrier(PIPE_V);

        uint64_t tailLen = nRoutedExperts / SORT_BLOCK_SIZE - 4;
        copy_ubuf_to_ubuf(sortTmp, sortMrgTmp, 0, 1,
                          DIV_ROUND_UP(4 * SORT_RESULT_BLOCK_SIZE * sizeof(uint32_t), BLOCK_SIZE),
                          0, 0);
        pipe_barrier(PIPE_V);
        for (int i = 4; i < 4 + tailLen; i++) {
            validBits = 3;  // queue 0 (running merge) + queue 1 (next block) valid
            addrArray[1] = sortTmp + i * SORT_RESULT_BLOCK_SIZE;
            lengths = SORT_BLOCK_SIZE * i | SORT_BLOCK_SIZE << 16;
            config = mrgRepeat | validBits << MGR_SORT_VALID_BITS_OFFSET |
                     ifExhaustedSuspension << MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET;
            vmrgsort4(sortMrgTmp, addrArray, lengths, config);
            pipe_barrier(PIPE_V);
            copy_ubuf_to_ubuf(
                sortTmp, sortMrgTmp, 0, 1,
                DIV_ROUND_UP((i + 1) * SORT_RESULT_BLOCK_SIZE * sizeof(uint32_t), BLOCK_SIZE), 0,
                0);
            pipe_barrier(PIPE_V);
        }

        // top-K indices (vreducev2 mode=2 = index payload), then gather original weights.
        vreducev2(indicesTopK, (__ubuf__ uint32_t *)sortMrgTmp, (__ubuf__ uint32_t *)sortMrgTmp, 1,
                  1, 2, 8, 0);
        pipe_barrier(PIPE_V);
        GatherWeights();
    }

    // vmuls(idx * 4) -> byte offset; vgather original weights (calc_unbiased) into weightsTopK.
    // NB: this vmuls takes an INT scalar (byte-offset multiply), so pass bare `4` — NOT 4.0f
    // (vmuls rejects a float: "need type 'int'"), NOT (float)sizeof(uint32_t) (aicore forbids the
    // runtime unsigned->float cast).
    __aicore__ inline void GatherWeights()
    {
        vmuls((__ubuf__ int32_t *)sortMrgTmp, (__ubuf__ int32_t *)indicesTopK, 4, 1, 1, 0, 0, 0);
        pipe_barrier(PIPE_V);
        vgather((__ubuf__ uint32_t *)weightsTopK, (__ubuf__ uint32_t *)sortMrgTmp,
                (uint64_t)calc_unbiased, 0, 1);
        pipe_barrier(PIPE_V);
    }

    // weightsTopK /= sum; *= scale. Ends V->S so FillRoutingMap's S-pipe read of indicesTopK
    // (V-written) is fenced — S-pipe scalar needs a V->S flag; pipe_barrier(PIPE_V) is V-only.
    __aicore__ inline void NormalizeAndScale()
    {
        ReduceSum(reduceTmp, weightsTopK, topK);
        vbrcb((__ubuf__ uint32_t *)reduceTmp, (__ubuf__ uint32_t *)reduceTmp, 0, 0, 1);
        pipe_barrier(PIPE_V);
        vdiv(weightsTopK, weightsTopK, reduceTmp, 1, 1, 1, 0, 8, 8, 0);
        pipe_barrier(PIPE_V);
        vmuls(weightsTopK, weightsTopK, scale, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }

    // Build sparse weightsSparse[N] + bitmap routingMap. Zeroed per-token (K sparse writes only
    // touch selected slots), then S-pipe fills from indicesTopK/weightsTopK. wait V->S before the
    // S read; set S->V after so StageOut's V relay sees the S writes.
    __aicore__ inline void FillRoutingMap()
    {
        vector_dup(weightsSparse, float(0), calcRepeat, 1, 0, 8, 0);
        vector_dup(routingMap, uint32_t(0), 1, 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int i = 0; i < (int)topK; ++i) {
            uint32_t idx = *(indicesTopK + i);
            bitmapSet((__ubuf__ uint64_t *)routingMap, idx);
            *(weightsSparse + idx) = *(weightsTopK + i);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
    }

    // Relay compute -> [curr] ping-pong slot: bf16 via vconv, float/bitmap via copy_ubuf.
    // wait MTE3->V (slot drained), relay, set V->MTE3 (slot full).
    __aicore__ inline void StageOut(int curr)
    {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
        if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
            uint64_t cfg = set_vector_1src_xt(4, 8, 1, 1, calcRepeat);
            vconv_f322bf16r(weightsOut[curr], weightsSparse, cfg);
        } else {
            copy_ubuf_to_ubuf(weightsOut[curr], weightsSparse, 0, 1,
                              DIV_ROUND_UP(nRoutedExperts * sizeof(float), BLOCK_SIZE), 0, 0);
        }
        copy_ubuf_to_ubuf(
            routingMapOut[curr], routingMap, 0, 1,
            DIV_ROUND_UP(nRoutedExperts * sizeof(uint32_t) / BIT_SIZE_OF_U32, BLOCK_SIZE), 0, 0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
    }

    // DMA [curr] slot -> GM: w -> outWeightsGm + t*N (sparse [M,N]); r -> routingMapGm + t*N/32
    // (BIT1 packed). wait V->MTE3 (full), DMA, set MTE3->V (release).
    __aicore__ inline void CopyOut(int tokenIdx, int curr)
    {
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
        CopyUbufToGmAligned(outWeightsGm + tokenIdx * nRoutedExperts, weightsOut[curr],
                            nRoutedExperts * sizeof(Dtype));
        CopyUbufToGmAligned(routingMapGm + tokenIdx * nRoutedExperts / BIT_SIZE_OF_U32,
                            routingMapOut[curr],
                            nRoutedExperts * sizeof(uint32_t) / BIT_SIZE_OF_U32);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
    }

private:
    uint32_t nTokens;
    uint32_t nRoutedExperts;
    uint32_t topK;
    uint8_t hash;
    float scale;
    int calcRepeat;
    int baseInputIdsIdx = -1;  // token idx where the loaded inputIds batch starts (-1=none)

    __gm__ Dtype *scoresGm;
    __gm__ uint32_t *indicesGm;
    __gm__ float *biasGm;
    __gm__ uint32_t *inputIdsGm;
    __gm__ uint32_t *tid2eidGm;
    __gm__ Dtype *outWeightsGm;
    __gm__ uint32_t *routingMapGm;

    __ubuf__ float *scoresIn;        // V's fp32 compute buf, single (CopyIn relays into it)
    __ubuf__ Dtype *scoresInTmp[2];  // MTE2 DMA-in, ping-pong (Dtype, matches GM)
    __ubuf__ float *biasIn;
    __ubuf__ uint32_t *indicesIn;
    __ubuf__ uint32_t *tid2eidRow[2];  // hash: [topK] row, ping-pong
    __ubuf__ uint32_t *inputIdsArr;    // hash: prefetched input_ids window
    __ubuf__ float *calc;
    __ubuf__ float *calc_unbiased;
    __ubuf__ float *reduceTmp;
    __ubuf__ float *sortTmp;
    __ubuf__ float *sortMrgTmp;
    __ubuf__ float *weightsTopK;          // compute, single ([topK] fp32, normalized+scaled)
    __ubuf__ uint32_t *indicesTopK;       // compute, single, INTERNAL (not a GM output)
    __ubuf__ float *weightsSparse;        // compute, single (sparse [N] -> w)
    __ubuf__ uint32_t *routingMap;        // compute, single ([N/32] bitmap -> r)
    __ubuf__ Dtype *weightsOut[2];        // MTE3 DMA-out, ping-pong ([N] Dtype)
    __ubuf__ uint32_t *routingMapOut[2];  // MTE3 DMA-out, ping-pong ([N/32] bitmap)
};

#define SQRTSOFTPLUS_HASH_TOPK_FUNC_DEFINE(dtype)                                         \
    extern "C" __global__ __aicore__ void sqrtsoftplus_hash_topk_##dtype(                 \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid, \
        GM_ADDR outWeights, GM_ADDR routingMap, float scale, uint32_t numTokens,          \
        uint32_t numRoutedExperts, uint32_t topK, uint8_t hash)                           \
    {                                                                                     \
        SqrtsoftplusHashTopK<dtype> op;                                                   \
        op.Init(scores, indices, bias, inputIds, tid2eid, outWeights, routingMap, scale,  \
                numTokens, numRoutedExperts, topK, hash);                                 \
        op.Run();                                                                         \
    }
#else
#define SQRTSOFTPLUS_HASH_TOPK_FUNC_DEFINE(dtype)                                         \
    extern "C" __global__ __aicore__ void sqrtsoftplus_hash_topk_##dtype(                 \
        GM_ADDR scores, GM_ADDR indices, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid, \
        GM_ADDR outWeights, GM_ADDR routingMap, float scale, uint32_t numTokens,          \
        uint32_t numRoutedExperts, uint32_t topK, uint8_t hash)                           \
    {                                                                                     \
    }
#endif
