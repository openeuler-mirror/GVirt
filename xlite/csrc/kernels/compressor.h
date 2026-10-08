/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.
 * ===============================================================================
 *
 * DeepSeek-V4 KV Compressor fused kernel (multi-batch). Mirrors deepseek_v4.py
 * Compressor.forward after wkv/wgate GEMMs: overlap_transform + softmax-gated
 * weighted sum + RMSNorm + rotary + compressed KV cache write. Cores stride the
 * global compressed-block index (totalBlockIdx % block_num == block_idx); each
 * core also runs norm/rope on its rows — no cross-core sync.
 *
 * Args (see signature for GM_ADDR types):
 *   kv/score    : [batchedTokens, coff*head_dim] fp32
 *   ape         : [ratio, coff*head_dim]  (null => raw score, no +ape)
 *   norm        : [head_dim]              (null => skip norm)
 *   freqs       : [max_seq_len, rope_head_dim/2] complex64 (null => skip rope+cache)
 *   weightedSum : [nTotalBlocks, head_dim] scratch GM
 *   compressPositions/compressSlots : [nTotalBlocks] — COMPRESSED-token space
 *   state       : [blocks, block_size, 1, 2*coff*head_dim] fp32 (decode state)
 *   stateSlotMapping/stateBlockTable : paged state-cache addressing (spill/decode)
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"
#include "norm.h"
#include "rope_complex_and_cache.h"

using namespace AscendC;

#ifdef __DAV_C220_VEC__

#define SPILL_ROWS 4  // max rows per batched run
// Spill prefill tokens into `state`: state[slot] = [kv | score+ape]. Per request
// spill the tail [cutoff,absEnd) (never compressed); under overlap also the last
// full window [cutoff-ratio,cutoff). Score stored WITH ape (decode reads as-is).
// Cores stride over requests. Runs of consecutive slots (capped at SPILL_ROWS):
// MTE2 loads, V computes score+ape, MTE3 stores; split at state-block boundaries
// (2D copies need continuity); PING-PONG halves (fill k+1 while computing k).
//
// State slot = FLAT id = blockId*stateBlockSize + rowInBlock via stateBlockTable.
// Multi-row copies split at block boundaries (rows may span non-adjacent blocks).
// Table entry 0 = UNALLOCATED sentinel: writes skipped, reads unguarded (stale).
__aicore__ inline uint32_t state_block_id(__gm__ const uint32_t *stateBlockTable,
                                          uint32_t maxStateBlocks, uint32_t stateBlockSize,
                                          uint32_t b, uint32_t slot)
{
    return stateBlockTable[(uint64_t)b * maxStateBlocks + slot / stateBlockSize];
}

__aicore__ inline __gm__ float *state_row_addr(__gm__ float *state,
                                               __gm__ const uint32_t *stateBlockTable,
                                               uint32_t maxStateBlocks, uint32_t stateBlockSize,
                                               uint64_t stateCacheStrideDim0,
                                               uint32_t stateRowFloats, uint32_t b, uint32_t slot)
{
    uint32_t blockId = state_block_id(stateBlockTable, maxStateBlocks, stateBlockSize, b, slot);
    uint32_t rowInBlock = slot % stateBlockSize;
    return state + (uint64_t)blockId * stateCacheStrideDim0 + (uint64_t)rowInBlock * stateRowFloats;
}

// Max consecutive rows from `slot` within one block and `limit` (2D copies need continuity).
__aicore__ inline uint32_t state_rows_in_block(uint32_t slot, uint32_t stateBlockSize,
                                               uint32_t limit)
{
    uint32_t inBlock = stateBlockSize - slot % stateBlockSize;
    return inBlock < limit ? inBlock : limit;
}

// Spill a prefill request's tail window to streaming state (kv raw, score+ape) so the
// next chunk's load_window (abs<cached) finds it. Spill range [spillBeginAbs, absEnd):
//   * overlap & cutoff>=ratio+cached: last full window [cutoff-ratio, cutoff) + tail
//     remainder [cutoff, absEnd) (window-row scores already +ape at spill time).
//   * non-overlap: tail remainder [cutoff, absEnd) (== whole qlen when cutoff==cached,
//     a short prefill segment with no completed ratio window, nTotalBlocks=0).
// Rows addressed by stateSlotMapping (host-built, prefill only). Decode (qLen==1) is
// skipped: its spill range is empty and the decode token's state write goes through
// compressor_decode_token_state (by absolute pos).
__aicore__ inline void compressor_spill_prefill_to_state(
    __gm__ float *gmKv, __gm__ float *gmScore, __gm__ const float *ape, __gm__ float *state,
    __gm__ const uint32_t *stateSlotMapping, __gm__ const int32_t *queryStartLoc,
    __gm__ const int32_t *queryLens, __gm__ const int32_t *cachedLens, uint32_t batch,
    uint32_t ratio, uint32_t overlap, uint32_t head_dim, uint32_t full_dim,
    __gm__ const uint32_t *stateBlockTable, uint32_t maxStateBlocks, uint32_t stateBlockSize,
    uint64_t stateCacheStrideDim0)
{
    const uint32_t blockNum = get_block_num();
    const uint32_t blockIdx = get_block_idx();
    const uint32_t coff = 1 + overlap;
    const uint32_t rowFloats = coff * head_dim;     // one half-row (kv or score)
    const uint32_t stateRowFloats = 2 * rowFloats;  // [kv | score] per state slot
    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);

    const uint32_t rowStride = ROUND_UP(rowFloats, calcPad);
    uint64_t off = 0;
    __ubuf__ float *kvUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *kvUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *kvUb[2] = {kvUb0, kvUb1};
    __ubuf__ float *scoreUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *scoreUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *scoreUb[2] = {scoreUb0, scoreUb1};
    __ubuf__ float *apeUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *apeUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *apeUb[2] = {apeUb0, apeUb1};
    __ubuf__ float *scoreOutUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *scoreOutUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    __ubuf__ float *scoreOutUb[2] = {scoreOutUb0, scoreOutUb1};
    __ubuf__ float *calcUb = reinterpret_cast<__ubuf__ float *>(off);
    off += SPILL_ROWS * rowStride * sizeof(float);
    assert(off <= UB_SIZE);

    const uint32_t kvBlk = rowFloats * sizeof(float) / BLOCK_SIZE;  // 32B units per half-row
    const uint32_t repeat = DIV_ROUND_UP(rowFloats, calcPad);       // vadd repeats per half-row
    const uint32_t gmTokStride = full_dim * sizeof(float) / BLOCK_SIZE - kvBlk;
    const uint32_t ubHalfStride = rowStride * sizeof(float) / BLOCK_SIZE - kvBlk;
    const uint32_t gmStateGap = stateRowFloats * sizeof(float) / BLOCK_SIZE - kvBlk;

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    int curr = 0;  // ping-pong half: 0/1
    for (uint32_t batchIdx = blockIdx; batchIdx < batch; batchIdx += blockNum) {
        uint32_t qLen = (uint32_t)queryLens[batchIdx];
        uint32_t qStart = (uint32_t)queryStartLoc[batchIdx];
        uint32_t cachedLen = (uint32_t)cachedLens[batchIdx];
        uint32_t absEnd = cachedLen + qLen;        // absolute end position
        uint32_t cutoff = absEnd / ratio * ratio;  // last full-window boundary
        uint32_t spillBeginAbs = cutoff;           // tail start
        if (overlap && cutoff >= ratio + cachedLen) {
            spillBeginAbs = cutoff - ratio;  // include last window
        }
        if (spillBeginAbs < cachedLen) {
            spillBeginAbs = cachedLen;  // clip to this request
        }
        // Skip decode (qLen==1): spill range is empty on a compress step (cutoff==absEnd),
        // and the host zeros stateSlotMapping for decode rows (0 = unallocated sentinel) —
        // touching it here is UB. The decode token's state write is done by
        // compressor_decode_token_state (by ABSOLUTE pos, see below).
        if (qLen == 1) {
            continue;
        }
        uint32_t spillBegin = qStart + (spillBeginAbs - cachedLen);
        uint32_t spillEnd = qStart + qLen;
        // Runs of consecutive slots capped at SPILL_ROWS and block boundaries.
        for (uint32_t tok = spillBegin; tok < spillEnd;) {
            uint32_t firstSlot = stateSlotMapping[tok];
            uint32_t rows = 1;
            while (rows < SPILL_ROWS && tok + rows < spillEnd &&
                   stateSlotMapping[tok + rows] == firstSlot + rows) {
                rows++;
            }
            rows = state_rows_in_block(firstSlot, stateBlockSize, rows);
            __gm__ float *dstRow =
                state_row_addr(state, stateBlockTable, maxStateBlocks, stateBlockSize,
                               stateCacheStrideDim0, stateRowFloats, batchIdx, firstSlot);
            const bool blockAllocated = state_block_id(stateBlockTable, maxStateBlocks,
                                                       stateBlockSize, batchIdx, firstSlot) != 0;
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);
            copy_gm_to_ubuf(kvUb[curr], gmKv + (uint64_t)tok * full_dim, 0, rows, kvBlk,
                            gmTokStride, ubHalfStride);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);
            if (blockAllocated) {
                copy_ubuf_to_gm(dstRow, kvUb[curr], 0, rows, kvBlk, ubHalfStride, gmStateGap);
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);

            // score half: load score + ape, vadd into calc half, store from it.
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            copy_gm_to_ubuf(scoreUb[curr], gmScore + (uint64_t)tok * full_dim, 0, rows, kvBlk,
                            gmTokStride, ubHalfStride);
            // ape row = abs pos % ratio; run splits at ratio wrap. Skipped when null.
            if (ape != nullptr) {
                uint32_t apeRow0 = (cachedLen + (tok - qStart)) % ratio;
                if (apeRow0 + rows <= ratio) {
                    copy_gm_to_ubuf(apeUb[curr], ape + (uint64_t)apeRow0 * full_dim, 0, rows, kvBlk,
                                    gmTokStride, ubHalfStride);
                } else {
                    uint32_t wrap = ratio - apeRow0;
                    copy_gm_to_ubuf(apeUb[curr], ape + (uint64_t)apeRow0 * full_dim, 0, wrap, kvBlk,
                                    gmTokStride, ubHalfStride);
                    copy_gm_to_ubuf(apeUb[curr] + wrap * rowStride, ape, 0, rows - wrap, kvBlk,
                                    gmTokStride, ubHalfStride);
                }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            if (ape != nullptr) {
                vadd(calcUb, scoreUb[curr], apeUb[curr], rows * repeat, 1, 1, 1, 8, 8, 8);
            } else {
                copy_ubuf_to_ubuf(calcUb, scoreUb[curr], 0, rows, kvBlk, ubHalfStride,
                                  ubHalfStride);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            copy_ubuf_to_ubuf(scoreOutUb[curr], calcUb, 0, rows, kvBlk, ubHalfStride, ubHalfStride);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
            if (blockAllocated) {
                copy_ubuf_to_gm(dstRow + rowFloats, scoreOutUb[curr], 0, rows, kvBlk, ubHalfStride,
                                gmStateGap);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
            curr = 1 - curr;
            tok += rows;
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// Writes each decode request's single token (qLen==1) to state at its ABSOLUTE
// pos (cachedLens[b]) — kv raw, score+ape — so the next step's window read
// (load_window, abs<cached) finds it. Addressed by pos (not stateSlotMapping),
// matching the host's per-token convention. Skips prefill (qLen>1) — spill_remainder
// handles those.
__aicore__ inline void compressor_decode_token_state(
    __gm__ float *gmKv, __gm__ float *gmScore, __gm__ const float *ape, __gm__ float *state,
    __gm__ const int32_t *queryStartLoc, __gm__ const int32_t *queryLens,
    __gm__ const int32_t *cachedLens, uint32_t batch, uint32_t ratio, uint32_t overlap,
    uint32_t head_dim, uint32_t full_dim, __gm__ const uint32_t *stateBlockTable,
    uint32_t maxStateBlocks, uint32_t stateBlockSize, uint64_t stateCacheStrideDim0)
{
    const uint32_t blockNum = get_block_num();
    const uint32_t blockIdx = get_block_idx();
    const uint32_t stateRowFloats = 2 * full_dim;
    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    const uint32_t repeat = DIV_ROUND_UP(full_dim, calcPad);
    const uint32_t tokBlk = full_dim * sizeof(float) / BLOCK_SIZE;

    uint64_t off = 0;
    __ubuf__ float *kvUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *kvUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *kvUb[2] = {kvUb0, kvUb1};
    __ubuf__ float *scoreUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *scoreUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *scoreUb[2] = {scoreUb0, scoreUb1};
    __ubuf__ float *apeUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *apeUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *apeUb[2] = {apeUb0, apeUb1};
    __ubuf__ float *calcUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *calcUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *calcUb[2] = {calcUb0, calcUb1};
    __ubuf__ float *outUb0 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *outUb1 = reinterpret_cast<__ubuf__ float *>(off);
    off += full_dim * sizeof(float);
    __ubuf__ float *outUb[2] = {outUb0, outUb1};
    assert(off <= UB_SIZE);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    int curr = 0;
    for (uint32_t b = blockIdx; b < batch; b += blockNum) {
        uint32_t qLen = (uint32_t)queryLens[b];
        if (qLen > 1) {
            continue;  // prefill request: handled by spill_remainder
        }
        uint32_t pos = (uint32_t)cachedLens[b];
        uint32_t row = pos % ratio;
        // mixed-batch: decode token is packed at its absolute gmKv row queryStartLoc[b];
        // pure decode batch has queryStartLoc[b]==b, unchanged.
        uint32_t tokOff = (uint32_t)queryStartLoc[b];
        __gm__ float *dstRow =
            state_row_addr(state, stateBlockTable, maxStateBlocks, stateBlockSize,
                           stateCacheStrideDim0, stateRowFloats, b, pos);
        const bool blockAllocated =
            state_block_id(stateBlockTable, maxStateBlocks, stateBlockSize, b, pos) != 0;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
        copy_gm_to_ubuf(kvUb[curr], gmKv + (uint64_t)tokOff * full_dim, 0, 1, tokBlk, 0, 0);
        copy_gm_to_ubuf(scoreUb[curr], gmScore + (uint64_t)tokOff * full_dim, 0, 1, tokBlk, 0, 0);
        if (ape != nullptr) {
            copy_gm_to_ubuf(apeUb[curr], ape + (uint64_t)row * full_dim, 0, 1, tokBlk, 0, 0);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
        if (ape != nullptr) {
            vadd(calcUb[curr], scoreUb[curr], apeUb[curr], repeat, 1, 1, 1, 8, 8, 8);
        } else {
            copy_ubuf_to_ubuf(calcUb[curr], scoreUb[curr], 0, 1, tokBlk, 0, 0);
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        copy_ubuf_to_ubuf(outUb[curr], kvUb[curr], 0, 1, tokBlk, 0, 0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
        if (blockAllocated) {
            copy_ubuf_to_gm(dstRow, outUb[curr], 0, 1, tokBlk, 0, 0);
            copy_ubuf_to_gm(dstRow + full_dim, calcUb[curr], 0, 1, tokBlk, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
        curr = 1 - curr;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// out = sum_k softmax(score[k] (+ ape[k])) * kv[k] over mergeSize rows of
// rowStride. May alias kv (vmul consumes kv before out is written). dCur > 0
// (tiled): only the first dCur channels are live (masked, restored after); 0 = all.
__aicore__ inline void softmax_weighted_sum(__ubuf__ float *kv, __ubuf__ float *score,
                                            __ubuf__ float *ape, __ubuf__ float *out,
                                            __ubuf__ float *tmp, __ubuf__ float *mx,
                                            __ubuf__ float *sm, uint32_t mergeSize,
                                            uint32_t rowStride, uint32_t repeat, uint32_t dCur)
{
    float negInf = -3.4028235e+38f;
    if (dCur > 0) {
        SetMask(dCur);  // only the low dCur lanes compute; UB stride unchanged
    }
    const uint32_t totalRepeat = repeat * mergeSize;                                // <=255
    const uint32_t unitStride = (rowStride / repeat) * sizeof(float) / BLOCK_SIZE;  // blocks
    // 1) score += ape
    if (ape != nullptr) {
        vadd(score, score, ape, totalRepeat, 1, 1, 1, unitStride, unitStride, unitStride);
        pipe_barrier(PIPE_V);
    }
    // 2) mx = max_k score[k]
    vector_dup(mx, negInf, repeat, 1, 1, 8, 0);
    vector_dup(sm, 0.0f, repeat, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    for (uint32_t k = 0; k < mergeSize; k++) {
        vmax(mx, mx, score + k * rowStride, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    // 3) tmp = score - mx
    for (uint32_t k = 0; k < mergeSize; k++) {
        vsub(tmp + k * rowStride, score + k * rowStride, mx, repeat, 1, 1, 1, 8, 8, 8);
    }
    pipe_barrier(PIPE_V);
    // 4) score = exp(tmp)
    vexp(score, tmp, totalRepeat, 1, 1, unitStride, unitStride);
    pipe_barrier(PIPE_V);
    // 5) sm = sum_k score[k]
    for (uint32_t k = 0; k < mergeSize; k++) {
        vadd(sm, sm, score + k * rowStride, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    // 6) score /= sm
    for (uint32_t k = 0; k < mergeSize; k++) {
        vdiv(score + k * rowStride, score + k * rowStride, sm, repeat, 1, 1, 1, 8, 8, 8);
    }
    pipe_barrier(PIPE_V);
    // 7) tmp = score * kv
    vmul(tmp, score, kv, totalRepeat, 1, 1, 1, unitStride, unitStride, unitStride);
    pipe_barrier(PIPE_V);
    // 8) out = sum_k tmp[k]
    vector_dup(out, 0.0f, repeat, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    for (uint32_t k = 0; k < mergeSize; k++) {
        vadd(out, out, tmp + k * rowStride, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    if (dCur > 0) {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);  // restore full lanes
    }
}

// Unified window loader. Loads `ratio` rows of a `rowWidth`-float column slice into
// kvUb/scoreUb at [ubStartOff, ubStartOff + ratio*ubRowStride). Per row, source is by
// abs vs cached: abs<cached -> paged state row (score already +ape), abs>=cached ->
// gmKv row (raw score). Contiguous same-source rows batch into one 2D DMA, split at
// state/gmKv boundaries. Two call shapes:
//   * non-tiled half-window: rowWidth=head_dim, ubRowStride=paddedHeadDim, gmColOff =
//     0 front / head_dim back — loads one half (front [:head_dim] / back [head_dim:]).
//   * tiled column-slice (ratio=128, overlap=0): rowWidth=dCur, ubRowStride=slice,
//     gmColOff=c0 — loads column slice [c0, c0+dCur) of [:head_dim].
// noPredecessor front padding is the caller's job (skip the rows, vector_dup them).
__aicore__ inline void load_window(__ubuf__ float *kvUb, __ubuf__ float *scoreUb,
                                   uint32_t ubStartOff, __gm__ float *gmKv, __gm__ float *gmScore,
                                   __gm__ float *state, __gm__ const uint32_t *stateBlockTable,
                                   uint32_t maxStateBlocks, uint32_t stateBlockSize,
                                   uint64_t stateCacheStrideDim0, uint32_t stateRowFloats,
                                   uint32_t full_dim, uint32_t rowWidth, uint32_t ubRowStride,
                                   uint32_t ratio, uint32_t baseAbs, uint32_t cached,
                                   uint32_t qStart, uint32_t batchIdx, uint32_t gmColOff)
{
    const uint32_t blkPerRow = rowWidth * sizeof(float) / BLOCK_SIZE;
    const uint32_t gmSrcGap = (full_dim - rowWidth) * sizeof(float) / BLOCK_SIZE;
    const uint32_t stateSrcGap = (stateRowFloats - rowWidth) * sizeof(float) / BLOCK_SIZE;
    const uint32_t dstGap = (ubRowStride - rowWidth) * sizeof(float) / BLOCK_SIZE;

    uint32_t k = 0;
    while (k < ratio) {
        uint32_t absPos = baseAbs + k;
        if (absPos < cached) {
            // state run: contiguous rows within ONE paged block (row stride
            // stateRowFloats). Cap at the block boundary (state_rows_in_block) and at
            // the state/gmKv boundary (cached) so the 2D copy stays contiguous+state.
            uint32_t run = state_rows_in_block(absPos, stateBlockSize, ratio - k);
            if (absPos + run > cached) {
                run = cached - absPos;
            }
            __gm__ float *srow =
                state_row_addr(state, stateBlockTable, maxStateBlocks, stateBlockSize,
                               stateCacheStrideDim0, stateRowFloats, batchIdx, absPos);
            // kv at +gmColOff, score at +full_dim+gmColOff; both span rowWidth, same row stride.
            copy_gm_to_ubuf(kvUb + ubStartOff + k * ubRowStride, srow + gmColOff, 0, run, blkPerRow,
                            stateSrcGap, dstGap);
            copy_gm_to_ubuf(scoreUb + ubStartOff + k * ubRowStride, srow + full_dim + gmColOff, 0,
                            run, blkPerRow, stateSrcGap, dstGap);
            k += run;
        } else {
            // gmKv run: contiguous rows [k, k+run) all with abs >= cached.
            uint32_t run = 0;
            while (k + run < ratio && (baseAbs + k + run) >= cached) {
                run++;
            }
            uint32_t gmRow = qStart + ((baseAbs + k) - cached);
            copy_gm_to_ubuf(kvUb + ubStartOff + k * ubRowStride,
                            gmKv + (uint64_t)gmRow * full_dim + gmColOff, 0, run, blkPerRow,
                            gmSrcGap, dstGap);
            copy_gm_to_ubuf(scoreUb + ubStartOff + k * ubRowStride,
                            gmScore + (uint64_t)gmRow * full_dim + gmColOff, 0, run, blkPerRow,
                            gmSrcGap, dstGap);
            k += run;
        }
    }
}

// Per (batchIdx, localBlockIdx) -> totalBlockIdx: assemble a mergeSize-row
// kv/score block, compute (kv * score.softmax(dim=2)).sum(dim=2) ->
// weightedSum[totalBlockIdx]. Cores stride totalBlockIdx (disjoint rows).
//
// Window alignment is ABSOLUTE per request: head = (ratio - cachedLen%ratio) % ratio
// leading tokens are skipped as block starts; overlap reads their rows from `state`
// (prev chunk's spill) via load_window. Chunked prefill (cachedLen>0): the first
// overlap block's front half assembles from state (prev) + gmKv (this chunk's head).
template <typename Dtype>
__aicore__ inline void compressor_blocks_weighted_sum(
    __gm__ float *gmKv, __gm__ float *gmScore, __gm__ const float *ape, __gm__ Dtype *weightedSum,
    __gm__ const int32_t *queryStartLoc, __gm__ const int32_t *queryLens,
    __gm__ const int32_t *cachedLens, uint32_t batch, uint32_t nTotalBlocks, uint32_t ratio,
    uint32_t overlap, uint32_t head_dim, __gm__ float *state,
    __gm__ const uint32_t *stateSlotMapping, __gm__ const uint32_t *stateBlockTable,
    uint32_t maxStateBlocks, uint32_t stateBlockSize, uint64_t stateCacheStrideDim0)
{
    uint32_t full_dim = (1 + overlap) * head_dim;
    uint32_t mergeSize = (1 + overlap) * ratio;  // 2*ratio or ratio
    constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
    const uint32_t paddedHeadDim = ROUND_UP(head_dim, calcPad);
    const uint32_t blockNum = get_block_num();
    const uint32_t blockIdx = get_block_idx();
    // Non-tiled (ratio=2/4) vs tiled (ratio=128) UB layout, chosen by size.
    // UB footprint of the non-tiled branch below, matched 1:1 to its `off` bumps:
    //   8 × (mergeSize * paddedHeadDim * sizeof(float))  -- kvUb[0/1], scoreUb[0/1],
    //                                                    kvCalcUb, scoreCalcUb, tmpUb, apeUb
    //   4 × (paddedHeadDim * sizeof(float))             -- kvOutUb[0/1] (Dtype; rounded up
    //                                                    to float as a safe upper bound),
    //                                                    mxUb, smUb
    bool useTiled = (8 * mergeSize + 4) * paddedHeadDim * sizeof(float) > (uint64_t)UB_SIZE;

    if (!useTiled) {
        uint64_t off = 0;
        __ubuf__ float *kvUb0 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        __ubuf__ float *kvUb1 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        __ubuf__ float *scoreUb0 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        __ubuf__ float *scoreUb1 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        __ubuf__ Dtype *kvOutUb0 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += paddedHeadDim * sizeof(Dtype);
        __ubuf__ Dtype *kvOutUb1 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += paddedHeadDim * sizeof(Dtype);
        __ubuf__ float *kvUb[2] = {kvUb0, kvUb1};
        __ubuf__ float *scoreUb[2] = {scoreUb0, scoreUb1};
        __ubuf__ Dtype *kvOutUb[2] = {kvOutUb0, kvOutUb1};
        __ubuf__ float *kvCalcUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        __ubuf__ float *scoreCalcUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        __ubuf__ float *mxUb = reinterpret_cast<__ubuf__ float *>(off);
        off += paddedHeadDim * sizeof(float);
        __ubuf__ float *smUb = reinterpret_cast<__ubuf__ float *>(off);
        off += paddedHeadDim * sizeof(float);
        __ubuf__ float *tmpUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        __ubuf__ float *apeUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * paddedHeadDim * sizeof(float);
        assert(off <= UB_SIZE);

        const uint32_t repeat = DIV_ROUND_UP(head_dim, calcPad);
        float negInf = -3.4028235e+38f;
        const uint32_t blkPerRow = head_dim * sizeof(float) / BLOCK_SIZE;            // 32B units
        const uint32_t srcGap = (full_dim - head_dim) * sizeof(float) / BLOCK_SIZE;  // overlap only
        const uint32_t dstGap = (paddedHeadDim - head_dim) * sizeof(float) / BLOCK_SIZE;

        // Load ape once, rearranged to match scoreUb's row order.
        if (ape != nullptr) {
            if (overlap) {
                copy_gm_to_ubuf(apeUb, ape, 0, ratio, blkPerRow, srcGap, dstGap);
                copy_gm_to_ubuf(apeUb + ratio * paddedHeadDim, ape + head_dim, 0, ratio, blkPerRow,
                                srcGap, dstGap);
            } else {
                copy_gm_to_ubuf(apeUb, ape, 0, mergeSize, blkPerRow, 0, dstGap);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }

        uint32_t totalBlockIdx = 0;
        for (uint32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
            uint32_t qLen = (uint32_t)queryLens[batchIdx];
            uint32_t qStart = (uint32_t)queryStartLoc[batchIdx];
            uint32_t cachedLen = (uint32_t)cachedLens[batchIdx];
            uint32_t nBlocks = ((cachedLen + qLen) / ratio) - (cachedLen / ratio);
            if (nBlocks == 0) {
                continue;
            }
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            int curr = 0;
            for (uint32_t localBlockIdx = 0; localBlockIdx < nBlocks;
                 localBlockIdx++, totalBlockIdx++) {
                if (totalBlockIdx % blockNum != blockIdx) {
                    continue;  // not this core's block
                }
                // bStart = absolute pos of block j's first token (ratio-aligned).
                uint32_t bStart = (cachedLen / ratio + localBlockIdx) * ratio;
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
                // hasStateRows: window holds any state row (abs<cached). With ape!=null those
                // rows already carry ape (spill baked it in), so gmKv rows are pre-added and
                // softmax runs with ape=null (no double-add). overlap's front window extends the
                // check to bStart<cachedLen+ratio.
                bool hasStateRows =
                    overlap ? (bStart < (uint64_t)cachedLen + ratio) : (bStart < cachedLen);
                // noPredecessor: first-ever window (overlap only, bStart==0). Front half
                // [bStart-ratio,bStart) underflows, so load skips it and pad fills 0/-inf — load
                // and pad MUST share this predicate (bStart==0 occurs with cachedLen in
                // (0,ratio), not just cachedLen==0).
                const bool noPredecessor = overlap && (bStart < ratio);
                if (overlap) {
                    // back [ratio,2*ratio) = cur [head_dim:]; front [0,ratio) = prev [:head_dim].
                    const uint32_t frontBase = bStart - ratio;  // abs of front row 0
                    if (noPredecessor) {
                        // pad front 0/-inf (filled after copy_to_calc): load back only.
                        load_window(kvUb[curr], scoreUb[curr], ratio * paddedHeadDim, gmKv, gmScore,
                                    state, stateBlockTable, maxStateBlocks, stateBlockSize,
                                    stateCacheStrideDim0, 2 * full_dim, full_dim, head_dim,
                                    paddedHeadDim, ratio, bStart, cachedLen, qStart, batchIdx,
                                    head_dim);
                    } else {
                        // front: prev window [:head_dim], abs [bStart-ratio, bStart).
                        load_window(kvUb[curr], scoreUb[curr], 0, gmKv, gmScore, state,
                                    stateBlockTable, maxStateBlocks, stateBlockSize,
                                    stateCacheStrideDim0, 2 * full_dim, full_dim, head_dim,
                                    paddedHeadDim, ratio, frontBase, cachedLen, qStart, batchIdx,
                                    0);
                        // back: cur window [head_dim:], abs [bStart, bStart+ratio).
                        load_window(kvUb[curr], scoreUb[curr], ratio * paddedHeadDim, gmKv, gmScore,
                                    state, stateBlockTable, maxStateBlocks, stateBlockSize,
                                    stateCacheStrideDim0, 2 * full_dim, full_dim, head_dim,
                                    paddedHeadDim, ratio, bStart, cachedLen, qStart, batchIdx,
                                    head_dim);
                    }
                } else {
                    // non-overlap: ratio rows [:head_dim], abs [bStart, bStart+ratio).
                    load_window(kvUb[curr], scoreUb[curr], 0, gmKv, gmScore, state, stateBlockTable,
                                maxStateBlocks, stateBlockSize, stateCacheStrideDim0, 2 * full_dim,
                                full_dim, head_dim, paddedHeadDim, ratio, bStart, cachedLen, qStart,
                                batchIdx, 0);
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
                copy_ubuf_to_ubuf(kvCalcUb, kvUb[curr], 0, mergeSize, blkPerRow, 0, 0);
                copy_ubuf_to_ubuf(scoreCalcUb, scoreUb[curr], 0, mergeSize, blkPerRow, 0, 0);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
                if (noPredecessor) {
                    // Pad front half 0 (kv) / -inf (score) — load skipped it (same predicate).
                    vector_dup(kvCalcUb, 0.0f, ratio * repeat, 1, 1, 8, 0);
                    vector_dup(scoreCalcUb, negInf, ratio * repeat, 1, 1, 8, 0);
                    pipe_barrier(PIPE_V);
                }
                if (hasStateRows && ape != nullptr) {
                    // Pre-add ape to raw gmKv rows (abs>=cached) so all rows carry score+ape,
                    // then softmax with ape=null (state rows already carry ape). gmKv rows form a
                    // CONTIGUOUS SUFFIX within each half-window (abs increases with k, crossing
                    // cachedLen at most once per half), so each half's gmKv rows are one vadd over
                    // [firstGm, halfEnd) instead of a per-row loop. repeat*rows <= 8*4=32 <= 255.
                    if (overlap) {
                        // Back half [ratio, 2*ratio): row k -> abs = bStart + (k - ratio).
                        uint32_t firstGmBack = cachedLen > bStart ? (cachedLen - bStart) : 0;
                        if (firstGmBack < ratio) {  // has gmKv rows in back half
                            uint32_t off = (ratio + firstGmBack) * paddedHeadDim;
                            uint32_t rows = ratio - firstGmBack;
                            vadd(scoreCalcUb + off, scoreCalcUb + off, apeUb + off, repeat * rows,
                                 1, 1, 1, 8, 8, 8);
                            pipe_barrier(PIPE_V);
                        }
                        // Front half [0, ratio): row k -> abs = frontBase + k. noPredecessor skips
                        // it entirely (padded -inf; also avoids frontBase uint underflow).
                        if (!noPredecessor) {
                            uint32_t frontBase = bStart - ratio;
                            uint32_t firstGmFront =
                                cachedLen > frontBase ? (cachedLen - frontBase) : 0;
                            if (firstGmFront < ratio) {  // has gmKv rows in front half
                                uint32_t off = firstGmFront * paddedHeadDim;
                                uint32_t rows = ratio - firstGmFront;
                                vadd(scoreCalcUb + off, scoreCalcUb + off, apeUb + off,
                                     repeat * rows, 1, 1, 1, 8, 8, 8);
                                pipe_barrier(PIPE_V);
                            }
                        }
                    } else {
                        // Non-overlap [0, ratio): row k -> abs = bStart + k.
                        uint32_t firstGm = cachedLen > bStart ? (cachedLen - bStart) : 0;
                        if (firstGm < ratio) {  // has gmKv rows (suffix)
                            uint32_t off = firstGm * paddedHeadDim;
                            uint32_t rows = ratio - firstGm;
                            vadd(scoreCalcUb + off, scoreCalcUb + off, apeUb + off, repeat * rows,
                                 1, 1, 1, 8, 8, 8);
                            pipe_barrier(PIPE_V);
                        }
                    }
                }
                __ubuf__ float *apeArg = (hasStateRows || ape == nullptr) ? nullptr : apeUb;
                softmax_weighted_sum(kvCalcUb, scoreCalcUb, apeArg, kvCalcUb, tmpUb, mxUb, smUb,
                                     mergeSize, paddedHeadDim, repeat, 0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
                convert_output(kvOutUb[curr], kvCalcUb, DIV_ROUND_UP(head_dim, calcPad));
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                CopyUbufToGmAligned(weightedSum + totalBlockIdx * head_dim, kvOutUb[curr],
                                    head_dim * sizeof(Dtype));
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
                curr = 1 - curr;
            }
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
    } else {
        const uint32_t slice = calcPad / 2;  // tiled compress column (32 floats)
        uint32_t nSlice = DIV_ROUND_UP(head_dim, slice);
        uint64_t off = 0;
        __ubuf__ float *kvUb0 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        __ubuf__ float *kvUb1 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        __ubuf__ float *kvUb[2] = {kvUb0, kvUb1};
        __ubuf__ float *scoreUb0 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        __ubuf__ float *scoreUb1 = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        __ubuf__ float *scoreUb[2] = {scoreUb0, scoreUb1};
        __ubuf__ float *kvCalcUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        __ubuf__ float *scoreCalcUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        __ubuf__ float *kvOutF32Ub = reinterpret_cast<__ubuf__ float *>(off);
        off += slice * sizeof(float);
        __ubuf__ Dtype *kvOutUb0 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += slice * sizeof(Dtype);
        __ubuf__ Dtype *kvOutUb1 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += slice * sizeof(Dtype);
        __ubuf__ Dtype *kvOutUb[2] = {kvOutUb0, kvOutUb1};
        __ubuf__ float *mxUb = reinterpret_cast<__ubuf__ float *>(off);
        off += slice * sizeof(float);
        __ubuf__ float *smUb = reinterpret_cast<__ubuf__ float *>(off);
        off += slice * sizeof(float);
        __ubuf__ float *tmpUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        __ubuf__ float *apeUb = reinterpret_cast<__ubuf__ float *>(off);
        off += mergeSize * slice * sizeof(float);
        assert(off <= UB_SIZE);  // head_dim=512, ratio=128: ~128KB <= 192KB

        for (uint32_t sIdx = 0; sIdx < nSlice; sIdx++) {
            uint32_t c0 = sIdx * slice;
            uint32_t dCur = (c0 + slice) > head_dim ? (head_dim - c0) : slice;
            uint32_t blkPerRowSlice = dCur * sizeof(float) / BLOCK_SIZE;
            uint32_t srcGapSlice = (full_dim - dCur) * sizeof(float) / BLOCK_SIZE;
            uint32_t dstGapSlice = (slice - dCur) * sizeof(float) / BLOCK_SIZE;
            if (ape != nullptr) {
                copy_gm_to_ubuf(apeUb, ape + c0, 0, mergeSize, blkPerRowSlice, srcGapSlice,
                                dstGapSlice);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
            }
            uint32_t sliceBlockIdx = 0;
            for (uint32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
                uint32_t qLen = (uint32_t)queryLens[batchIdx];
                uint32_t qStart = (uint32_t)queryStartLoc[batchIdx];
                uint32_t cachedLen = (uint32_t)cachedLens[batchIdx];
                uint32_t nBlocks = ((cachedLen + qLen) / ratio) - (cachedLen / ratio);
                if (nBlocks == 0) {
                    continue;
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                int curr = 0;
                for (uint32_t localBlockIdx = 0; localBlockIdx < nBlocks;
                     localBlockIdx++, sliceBlockIdx++) {
                    if (sliceBlockIdx % blockNum != blockIdx) {
                        continue;  // not this core's block
                    }
                    uint32_t bStart = (cachedLen / ratio + localBlockIdx) * ratio;
                    bool hasStateRows = (bStart < cachedLen);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
                    // Unified source: state rows (abs<cached) + gmKv rows (abs>=cached).
                    load_window(kvUb[curr], scoreUb[curr], /*ubStartOff=*/0, gmKv, gmScore, state,
                                stateBlockTable, maxStateBlocks, stateBlockSize,
                                stateCacheStrideDim0, 2 * full_dim, full_dim,
                                /*rowWidth=*/dCur, /*ubRowStride=*/slice, ratio,
                                /*baseAbs=*/bStart, cachedLen, qStart, batchIdx,
                                /*gmColOff=*/c0);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
                    copy_ubuf_to_ubuf(kvCalcUb, kvUb[curr], 0, mergeSize, blkPerRowSlice,
                                      dstGapSlice, dstGapSlice);
                    copy_ubuf_to_ubuf(scoreCalcUb, scoreUb[curr], 0, mergeSize, blkPerRowSlice,
                                      dstGapSlice, dstGapSlice);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
                    if (hasStateRows && ape != nullptr) {
                        // Pre-add ape to the gmKv rows (abs>=cached; state rows already carry
                        // ape). gmKv rows form a contiguous suffix (abs rises with k), so one vadd
                        // over [firstGm, mergeSize) replaces the per-row loop. SetMask(dCur) masks
                        // each repeat unit; stride=slice/8 blocks (=slice floats) jumps the
                        // dstGapSlice between rows without clobbering.
                        uint32_t firstGm = cachedLen > bStart ? (cachedLen - bStart) : 0;
                        if (firstGm < mergeSize) {  // has gmKv rows (suffix)
                            SetMask(dCur);
                            uint32_t off = firstGm * slice;
                            uint32_t nGm = mergeSize - firstGm;
                            vadd(scoreCalcUb + off, scoreCalcUb + off, apeUb + off, nGm, 1, 1, 1,
                                 slice / (BLOCK_SIZE / sizeof(float)),
                                 slice / (BLOCK_SIZE / sizeof(float)),
                                 slice / (BLOCK_SIZE / sizeof(float)));
                            pipe_barrier(PIPE_V);
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        }
                    }
                    __ubuf__ float *apeArg = (hasStateRows || ape == nullptr) ? nullptr : apeUb;
                    softmax_weighted_sum(kvCalcUb, scoreCalcUb, apeArg, kvOutF32Ub, tmpUb, mxUb,
                                         smUb, mergeSize, slice, 1, dCur);
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
                    SetMask(dCur);
                    convert_output(kvOutUb[curr], kvOutF32Ub, 1);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2 + curr);
                    CopyUbufToGmAligned(weightedSum + (uint64_t)sliceBlockIdx * head_dim + c0,
                                        kvOutUb[curr], dCur * sizeof(Dtype));
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2 + curr);
                    curr = 1 - curr;
                }
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }
        }
    }
    if (state != nullptr && stateSlotMapping != nullptr) {
        compressor_spill_prefill_to_state(gmKv, gmScore, ape, state, stateSlotMapping,
                                          queryStartLoc, queryLens, cachedLens, batch, ratio,
                                          overlap, head_dim, full_dim, stateBlockTable,
                                          maxStateBlocks, stateBlockSize, stateCacheStrideDim0);
    }
    // Decode-token state writes: spill skips decode (qLen==1), so this pass writes each
    // decode token to state at its ABSOLUTE pos (cachedLens[b], not stateSlotMapping) —
    // matching the host's per-token state convention. Runs every step (compress or not)
    // so the next window read finds the token.
    if (state != nullptr) {
        compressor_decode_token_state(gmKv, gmScore, ape, state, queryStartLoc, queryLens,
                                      cachedLens, batch, ratio, overlap, head_dim, full_dim,
                                      stateBlockTable, maxStateBlocks, stateBlockSize,
                                      stateCacheStrideDim0);
    }
    pipe_barrier(PIPE_ALL);
}

// Top-level dispatch — UNIFIED. One compressor_blocks_weighted_sum pass handles
// prefill, chunked-prefill, AND decode: block count ((cached+qLen)/ratio -
// cached/ratio) is 1 for a decode compress step, 0 for a non-compress step, so
// decode needs no separate path. Blocks dispatch via totalBlockIdx % blockNum ==
// blockIdx, and the norm/rope tail strides the same range, so each core reads the
// weightedSum[R] it wrote — no cross-core GM barrier. Decode-token state writes
// (skipped by spill, which only does prefill tails) go via compressor_decode_token_state.
template <typename Dtype>
__aicore__ void compressor(GM_ADDR kv, GM_ADDR score, GM_ADDR ape, GM_ADDR normWeight,
                           GM_ADDR freqs, GM_ADDR weightedSum, GM_ADDR queryStartLoc,
                           GM_ADDR queryLens, GM_ADDR cachedLens, uint32_t batch,
                           uint32_t nTotalBlocks, uint32_t ratio, uint32_t overlap,
                           uint32_t head_dim, uint32_t rope_head_dim, float norm_eps,
                           GM_ADDR compressKv, GM_ADDR compressPositions, GM_ADDR compressSlots,
                           uint32_t compressBlockSize, GM_ADDR state, GM_ADDR stateSlotMapping,
                           GM_ADDR stateBlockTable, uint32_t stateBlockSize,
                           uint32_t maxStateBlocks, uint64_t stateCacheStrideDim0, bool hasNorm,
                           bool hasRope, bool hasRotate = false, float rotateScale = 1.0f)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ float *gmKv = (__gm__ float *)kv;
    __gm__ float *gmScore = (__gm__ float *)score;
    __gm__ const float *gmApe = (__gm__ const float *)ape;
    __gm__ Dtype *gmWeightedSum = (__gm__ Dtype *)weightedSum;
    __gm__ const int32_t *gmQStartLoc = (__gm__ const int32_t *)queryStartLoc;
    __gm__ const int32_t *gmQLens = (__gm__ const int32_t *)queryLens;
    __gm__ float *gmState = (__gm__ float *)state;
    __gm__ const uint32_t *gmStateSlotMapping = (__gm__ const uint32_t *)stateSlotMapping;
    __gm__ const uint32_t *gmStateBlockTable = (__gm__ const uint32_t *)stateBlockTable;

    // 1) unified compress pass — prefill/chunked/decode all flow through here.
    //    Writes weightedSum[0..nTotalBlocks) and spills tail + decode tokens to state.
    compressor_blocks_weighted_sum<Dtype>(gmKv, gmScore, gmApe, gmWeightedSum, gmQStartLoc, gmQLens,
                                          (__gm__ const int32_t *)cachedLens, batch, nTotalBlocks,
                                          ratio, overlap, head_dim, gmState, gmStateSlotMapping,
                                          gmStateBlockTable, maxStateBlocks, stateBlockSize,
                                          stateCacheStrideDim0);
    // nTotalBlocks==0 on a decode NON-compress step (qlen=1, cached%ratio!=ratio-1):
    // the pass above still wrote the decode token + spill to state, but no compress
    // block was produced, so the norm+rope tail below (over [0,nTotalBlocks)) has an
    // empty range — skip it. This is the hot path in decode streams (ratio-1 of every
    // ratio launches here); only the compress step (cached%ratio==ratio-1) yields 1
    // block. Also occurs for a short prefill segment with no completed ratio window.
    if (nTotalBlocks == 0) {
        return;
    }
    // 2) single norm + rope tail over the full [0, nTotalBlocks) range. Cores
    //    stride the same totalBlockIdx space as the compress pass, so each row
    //    is read by the core that wrote it (no cross-core sync needed).
    if (hasNorm) {
        auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::Rms);
        norm<Dtype>(weightedSum, nullptr, normWeight, nullptr, weightedSum, nTotalBlocks, head_dim,
                    norm_eps, kind, 1, head_dim, head_dim, 0, 0, true, nullptr, 1, nullptr, nullptr,
                    0, 0, nullptr, false);
    }
    if (hasRope) {
        rope_complex_and_cache<Dtype>(
            nTotalBlocks, 1, head_dim, rope_head_dim, head_dim - rope_head_dim, head_dim,
            weightedSum, nullptr, 0, 0, freqs, compressPositions, compressBlockSize, compressKv,
            compressSlots, false, true, 0, nullptr, hasRotate, rotateScale);
    }
}

#define COMPRESSOR_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void compressor_##dtype(                                      \
        GM_ADDR kv, GM_ADDR score, GM_ADDR ape, GM_ADDR normWeight, GM_ADDR freqs,                 \
        GM_ADDR weightedSum, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,         \
        uint32_t batch, uint32_t nTotalBlocks, uint32_t ratio, uint32_t overlap,                   \
        uint32_t head_dim, uint32_t rope_head_dim, float norm_eps, GM_ADDR compressKv,             \
        GM_ADDR compressPositions, GM_ADDR compressSlots, uint32_t compressBlockSize,              \
        GM_ADDR state, GM_ADDR stateSlotMapping, GM_ADDR stateBlockTable, uint32_t stateBlockSize, \
        uint32_t maxStateBlocks, uint64_t stateCacheStrideDim0, uint32_t doRotate,                 \
        float rotateScale)                                                                         \
    {                                                                                              \
        compressor<dtype>(kv, score, ape, normWeight, freqs, weightedSum, queryStartLoc,           \
                          queryLens, cachedLens, batch, nTotalBlocks, ratio, overlap, head_dim,    \
                          rope_head_dim, norm_eps, compressKv, compressPositions, compressSlots,   \
                          compressBlockSize, state, stateSlotMapping, stateBlockTable,             \
                          stateBlockSize, maxStateBlocks, stateCacheStrideDim0,                    \
                          normWeight != nullptr, freqs != nullptr, doRotate != 0, rotateScale);    \
    }
#else
#define COMPRESSOR_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void compressor_##dtype(                                      \
        GM_ADDR kv, GM_ADDR score, GM_ADDR ape, GM_ADDR normWeight, GM_ADDR freqs,                 \
        GM_ADDR weightedSum, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,         \
        uint32_t batch, uint32_t nTotalBlocks, uint32_t ratio, uint32_t overlap,                   \
        uint32_t head_dim, uint32_t rope_head_dim, float norm_eps, GM_ADDR compressKv,             \
        GM_ADDR compressPositions, GM_ADDR compressSlots, uint32_t compressBlockSize,              \
        GM_ADDR state, GM_ADDR stateSlotMapping, GM_ADDR stateBlockTable, uint32_t stateBlockSize, \
        uint32_t maxStateBlocks, uint64_t stateCacheStrideDim0, uint32_t doRotate,                 \
        float rotateScale)                                                                         \
    {                                                                                              \
    }
#endif