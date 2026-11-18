/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * GLM-5.3-Flash indexer kpool tail compress + spill (native-semantics full chain,
 * semantics = vllm-ascend native triton _glm5_next_kpool_tail_compress_kernel +
 * _store_kpool_tail_kernel).
 *
 * Compress kernel (one program per token, strided over cores):
 *   valid = slot in range & token < last_query_end & pos >= 0 & (pos+1) % kpool == 0;
 *   window = positions pos-kpool+1 .. pos; in-batch rows read from the k/gate
 *   inputs (src_row = prev_query_end + eff_pos - request_query_start), older rows
 *   from the request-owned fp32 tail ring (physical = tail_block_table[req_id],
 *   offset = eff_pos % cap, gate adjacent to k in the same row), rows before
 *   position 0 read as 0;
 *   scores = gate + ape -> per-dim softmax over the pool axis -> weighted sum of
 *   raw k -> stored BF16/FP16/FP32 into the paged indexer cache.
 *
 * Dtype only selects the indexerCache store width; k/gate are always FP32,
 * matching the native contract (vllm-ascend sparse_attn_indexer_kpool.py: keys,
 * gates and the compressor tail must be float32; the compressed cache is bf16).
 *
 * Spill kernel (one program per retained row, num_reqs * cap programs):
 *   row = query_end - 1 - retained_offset (only rows of the current batch);
 *   writes the (k, gate) row into tail_cache at physical*tailStride + (pos%cap)*2*headDim (fp32).
 *
 * The compress launch completes before the spill launch (host orders them on one
 * stream) so historical rows are not overwritten while still being read.
 *
 * Layouts (dense inner dims; pool caches carry a dim-0 page pitch via tailStrideT/idxStrideT):
 *   tailCache:    [tailBlocks, cap, 2*headDim] FP32 — row-major [k | gate] per
 *                 pool slot, matching the runner's paged tail pool view
 *   indexerCache: [idxBlocks, indexerBlockSize, 1, headDim] Dtype
 *   k / gate:     [numTokens, headDim] FP32 (row stride passed for safety)
 *   ape:          [kpool, headDim] FP32
 *   positions:    [numTokens] INT64
 *   slotMapping / tailBlockTable / queryStartLoc / lens / cachedLens: INT32 (same
 *   convention as indexer_topk/compressor: queryStartLoc[r] is request r's start,
 *   lens[r] its query length). Derived in-kernel: pooled write slot = mainSlot/kpool;
 *   spill row = physical*tailStride + (pos%cap)*2*headDim; seqLen = cachedLens[req] + lens[req].
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#ifdef __DAV_C220_VEC__

template <typename Dtype>
class KpoolTailCompress
{
public:
    __aicore__ inline KpoolTailCompress()
    {
    }

    __aicore__ inline void Init(GM_ADDR tailCache, GM_ADDR indexerCache, GM_ADDR k, GM_ADDR gate,
                                GM_ADDR ape, GM_ADDR positions, GM_ADDR queryStartLoc, GM_ADDR lens,
                                GM_ADDR cachedLens, GM_ADDR slotMapping, GM_ADDR tailBlockTable,
                                uint32_t numReqs, uint32_t numTokens, uint32_t headDim,
                                uint32_t kpool, uint32_t cap, uint32_t tailBlocks,
                                uint32_t indexerBlockSize, uint32_t idxBlocks,
                                uint32_t mainBlockSize, uint32_t kStrideT, uint32_t gateStrideT,
                                uint32_t tailStrideT, uint32_t idxStrideT)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        this->tailCache = (__gm__ float *)tailCache;
        this->indexerCache = (__gm__ Dtype *)indexerCache;
        this->k = (__gm__ float *)k;
        this->gate = (__gm__ float *)gate;
        this->ape = (__gm__ float *)ape;
        this->positions = (__gm__ int64_t *)positions;
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->lens = (__gm__ int32_t *)lens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->slotMapping = (__gm__ int32_t *)slotMapping;
        this->tailBlockTable = (__gm__ int32_t *)tailBlockTable;
        this->numReqs = numReqs;
        this->numTokens = numTokens;
        this->headDim = headDim;
        this->kpool = kpool;
        this->cap = cap;
        this->tailBlocks = tailBlocks;
        this->indexerBlockSize = indexerBlockSize;
        this->idxBlocks = idxBlocks;
        this->mainBlockSize = mainBlockSize;
        this->kStrideT = kStrideT;
        this->gateStrideT = gateStrideT;
        this->tailStrideT = tailStrideT;
        this->idxStrideT = idxStrideT;

        headBytesF = headDim * sizeof(float);
        headBytesD = headDim * sizeof(Dtype);
        headBlocksF = DIV_ROUND_UP(headBytesF, 32);
        headBlocksD = DIV_ROUND_UP(headBytesD, 32);
        poolBytesF = kpool * headDim * sizeof(float);
        poolBlocksF = DIV_ROUND_UP(poolBytesF, 32);
        headRepeat = DIV_ROUND_UP(headDim, VECTOR_MAX_NUM_OF_FP32);
        numIndexerSlots = idxBlocks * indexerBlockSize;

        uint64_t off = 0;
        apeF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(poolBytesF, 256);
        kF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(poolBytesF, 256);
        gF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(poolBytesF, 256);
        maxF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(headBytesF, 256);
        sumF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(headBytesF, 256);
        accF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(headBytesF, 256);
        tmpF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(headBytesF, 256);
        gTmpD = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += ROUND_UP(headBytesD, 256);
        assert(off <= UB_SIZE);
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

    __aicore__ inline void LoadFloatVec(__ubuf__ float *dst, __gm__ float *src, uint32_t nBlocks)
    {
        copy_gm_to_ubuf(dst, src, 0, 1, nBlocks, 0, 0);
        pipe_barrier(PIPE_MTE2);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    // Load one headDim row (FP32) from GM into a given FP32 UB slot.
    __aicore__ inline void LoadRowF32(__ubuf__ float *dstF, __gm__ float *src)
    {
        copy_gm_to_ubuf(dstF, src, 0, 1, headBlocksF, 0, 0);
        pipe_barrier(PIPE_MTE2);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void GatherWindow(uint32_t ti, int32_t pos, int32_t reqStart, int32_t seqLen,
                                        int32_t prevEnd, uint32_t reqId)
    {
        int32_t physical = tailBlockTable[reqId];
        for (uint32_t p = 0; p < kpool; ++p) {
            int32_t poolPos = pos - static_cast<int32_t>(kpool - 1 - p);
            int32_t eff = poolPos > 0 ? poolPos : 0;
            bool inWindow = (poolPos >= reqStart) && (poolPos < seqLen);
            if (inWindow) {
                // Current-batch row: read straight from the k/gate inputs.
                int32_t srcRow = prevEnd + eff - reqStart;
                srcRow = srcRow < 0 ? 0 : srcRow;
                srcRow = srcRow > static_cast<int32_t>(numTokens - 1)
                             ? static_cast<int32_t>(numTokens - 1)
                             : srcRow;
                LoadRowF32(kF + p * headDim, k + static_cast<size_t>(srcRow) * kStrideT);
                LoadRowF32(gF + p * headDim, gate + static_cast<size_t>(srcRow) * gateStrideT);
            } else if (poolPos >= 0) {
                // Older row: read from the request-owned fp32 tail ring.
                int32_t pageOffset = eff % static_cast<int32_t>(cap);
                size_t histBase = static_cast<size_t>(physical) * tailStrideT +
                                  static_cast<size_t>(pageOffset) * 2 * headDim;
                bool histValid = (physical >= 0) && (static_cast<uint32_t>(physical) < tailBlocks);
                if (histValid) {
                    LoadRowF32(kF + p * headDim, tailCache + histBase);
                    LoadRowF32(gF + p * headDim, tailCache + histBase + headDim);
                } else {
                    vector_dup(kF + p * headDim, float(0), headRepeat, 1, 0, 8, 0);
                    pipe_barrier(PIPE_V);
                    vector_dup(gF + p * headDim, float(0), headRepeat, 1, 0, 8, 0);
                    pipe_barrier(PIPE_V);
                }
            } else {
                // Before position 0: masked load -> 0.
                vector_dup(kF + p * headDim, float(0), headRepeat, 1, 0, 8, 0);
                pipe_barrier(PIPE_V);
                vector_dup(gF + p * headDim, float(0), headRepeat, 1, 0, 8, 0);
                pipe_barrier(PIPE_V);
            }
        }
    }

    // Softmax(gate + ape) over the pool axis (per-dim) + weighted sum of k.
    // Expects kF/gF already gathered ([kpool, headDim] FP32 each) and apeF loaded.
    __aicore__ inline void CompressAndStore(uint32_t pi, int64_t slot)
    {
        for (uint32_t p = 0; p < kpool; ++p) {
            vadd(gF + p * headDim, gF + p * headDim, apeF + p * headDim, headRepeat, 1, 1, 1, 8, 8,
                 8);
            pipe_barrier(PIPE_V);
        }
        vmax(maxF, gF, gF, headRepeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        for (uint32_t p = 1; p < kpool; ++p) {
            vmax(maxF, maxF, gF + p * headDim, headRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        for (uint32_t p = 0; p < kpool; ++p) {
            vsub(gF + p * headDim, gF + p * headDim, maxF, headRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vexp(gF + p * headDim, gF + p * headDim, headRepeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        vector_dup(sumF, float(0), headRepeat, 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
        for (uint32_t p = 0; p < kpool; ++p) {
            vadd(sumF, sumF, gF + p * headDim, headRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        vector_dup(accF, float(0), headRepeat, 1, 0, 8, 0);
        pipe_barrier(PIPE_V);
        for (uint32_t p = 0; p < kpool; ++p) {
            vmul(tmpF, kF + p * headDim, gF + p * headDim, headRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(accF, accF, tmpF, headRepeat, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        vdiv(accF, accF, sumF, headRepeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Store into the paged indexer cache: slot -> (block, offset).
        uint32_t indexerBlock = static_cast<uint32_t>(slot / indexerBlockSize);
        uint32_t indexerOffset = static_cast<uint32_t>(slot % indexerBlockSize);
        if constexpr (std::is_same<Dtype, float>::value) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm(indexerCache + static_cast<size_t>(indexerBlock) * idxStrideT +
                                indexerOffset * headDim,
                            accF, 0, 1, headBlocksF, 0, 0);
            pipe_barrier(PIPE_MTE3);
        } else {
            ConvFloatToHalf(gTmpD, accF, headRepeat);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm(indexerCache + static_cast<size_t>(indexerBlock) * idxStrideT +
                                indexerOffset * headDim,
                            gTmpD, 0, 1, headBlocksD, 0, 0);
            pipe_barrier(PIPE_MTE3);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    __aicore__ inline void ProcessOneToken(uint32_t ti)
    {
        // Pooled write slot derived from the main-KV token slot: pooled = mainSlot/kpool
        // (mainSlot = block*mainBlockSize + r -> block*(mainBlockSize/kpool) + r/kpool).
        int32_t mainSlot = slotMapping[ti];
        int32_t pos = static_cast<int32_t>(positions[ti]);
        int64_t slot = static_cast<int64_t>(mainSlot) >= 0
                           ? static_cast<int64_t>(mainSlot) / static_cast<int32_t>(kpool)
                           : -1;
        // ti < numTokens == queryStartLoc[numReqs]: padding tokens (slot -1) are filtered
        // by the slot check.
        bool valid = (slot >= 0) && (slot < static_cast<int64_t>(numIndexerSlots)) && (pos >= 0) &&
                     ((pos + 1) % static_cast<int32_t>(kpool) == 0);
        if (!valid) {
            return;
        }
        // Request lookup, same convention as the other kernels (indexer_topk/compressor):
        // queryStartLoc[r] is request r's absolute start ([batch+1] prefix-sum); the last
        // r with start <= ti owns the token.
        uint32_t reqId = 0;
        for (uint32_t r = 0; r < numReqs; ++r) {
            if (ti >= static_cast<uint32_t>(queryStartLoc[r])) {
                reqId = r;
            }
        }
        int32_t queryStart = queryStartLoc[reqId];
        int32_t queryLen = lens[reqId];
        // Full sequence length = cached prefix + current batch query tokens.
        int32_t seqLen = cachedLens[reqId] + queryLen;
        int32_t reqStart = seqLen - queryLen;

        GatherWindow(ti, pos, reqStart, seqLen, queryStart, reqId);
        CompressAndStore(ti, slot);
    }

    __aicore__ inline void Process()
    {
        if (headDim > KPOOL_TAIL_MAX_HEAD_DIM || kpool > KPOOL_TAIL_MAX_KPOOL || headDim == 0 ||
            kpool == 0 || numTokens == 0 || numReqs == 0 || cap == 0) {
            return;
        }
        LoadFloatVec(apeF, ape, poolBlocksF);
        for (uint32_t ti = GetBlockIdx(); ti < numTokens; ti += GetBlockNum()) {
            ProcessOneToken(ti);
        }
    }

private:
    __gm__ float *tailCache;
    __gm__ Dtype *indexerCache;
    __gm__ float *k;
    __gm__ float *gate;
    __gm__ float *ape;
    __gm__ int64_t *positions;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *lens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *slotMapping;
    __gm__ int32_t *tailBlockTable;

    __ubuf__ float *apeF;
    __ubuf__ float *kF;
    __ubuf__ float *gF;
    __ubuf__ float *maxF;
    __ubuf__ float *sumF;
    __ubuf__ float *accF;
    __ubuf__ float *tmpF;
    __ubuf__ Dtype *gTmpD;

    uint32_t numReqs;
    uint32_t numTokens;
    uint32_t headDim;
    uint32_t kpool;
    uint32_t cap;
    uint32_t tailBlocks;
    uint32_t indexerBlockSize;
    uint32_t idxBlocks;
    uint32_t mainBlockSize;
    uint32_t kStrideT;
    uint32_t gateStrideT;
    uint32_t tailStrideT;
    uint32_t idxStrideT;
    uint32_t headBytesF;
    uint32_t headBytesD;
    uint32_t headBlocksF;
    uint32_t headBlocksD;
    uint32_t poolBytesF;
    uint32_t poolBlocksF;
    uint32_t numIndexerSlots;
    int headRepeat;
};

// Spill has no Dtype-dependent storage (tailCache is FP32, k/gate are FP32); the
// exported symbols keep one instance per dtype name so host dispatch is unchanged.
class KpoolTailSpill
{
public:
    __aicore__ inline KpoolTailSpill()
    {
    }

    __aicore__ inline void Init(GM_ADDR tailCache, GM_ADDR k, GM_ADDR gate, GM_ADDR queryStartLoc,
                                GM_ADDR lens, GM_ADDR tailBlockTable, GM_ADDR positions,
                                uint32_t numReqs, uint32_t numTokens, uint32_t headDim,
                                uint32_t cap, uint32_t tailBlocks, uint32_t kStrideT,
                                uint32_t gateStrideT, uint32_t tailStrideT)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        this->tailCache = (__gm__ float *)tailCache;
        this->k = (__gm__ float *)k;
        this->gate = (__gm__ float *)gate;
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->lens = (__gm__ int32_t *)lens;
        this->tailBlockTable = (__gm__ int32_t *)tailBlockTable;
        this->positions = (__gm__ int64_t *)positions;
        this->numReqs = numReqs;
        this->numTokens = numTokens;
        this->headDim = headDim;
        this->cap = cap;
        this->tailBlocks = tailBlocks;
        this->kStrideT = kStrideT;
        this->gateStrideT = gateStrideT;
        this->tailStrideT = tailStrideT;

        headBytesF = headDim * sizeof(float);
        headBlocksF = DIV_ROUND_UP(headBytesF, 32);
        headRepeat = DIV_ROUND_UP(headDim, VECTOR_MAX_NUM_OF_FP32);

        uint64_t off = 0;
        rowF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += ROUND_UP(headBytesF, 256);
        assert(off <= UB_SIZE);
    }

    // Load one headDim row (FP32) into rowF.
    __aicore__ inline void LoadRowF(__gm__ float *src)
    {
        copy_gm_to_ubuf(rowF, src, 0, 1, headBlocksF, 0, 0);
        pipe_barrier(PIPE_MTE2);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void ProcessOneRow(uint32_t pid)
    {
        uint32_t reqId = pid / cap;
        uint32_t retainedOffset = pid % cap;
        if (reqId >= numReqs) {
            return;
        }
        int32_t start = queryStartLoc[reqId];
        int32_t end = start + lens[reqId];
        int32_t row = end - 1 - static_cast<int32_t>(retainedOffset);
        bool valid = (row >= start) && (row >= 0) && (row < static_cast<int32_t>(numTokens));
        if (!valid) {
            return;
        }
        // Absolute ring slot derived from the per-request tail block + row position
        // (native tail builder slot formula: block*cap + pos%cap).
        int64_t pos = positions[row];
        int32_t physical = tailBlockTable[reqId];
        valid = (pos >= 0) && (physical >= 0) && (static_cast<uint32_t>(physical) < tailBlocks);
        if (!valid) {
            return;
        }
        uint32_t block = static_cast<uint32_t>(physical);
        uint32_t offset = static_cast<uint32_t>(pos) % cap;
        size_t addr =
            static_cast<size_t>(block) * tailStrideT + static_cast<size_t>(offset) * 2 * headDim;
        LoadRowF(k + static_cast<size_t>(row) * kStrideT);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(tailCache + addr, rowF, 0, 1, headBlocksF, 0, 0);
        pipe_barrier(PIPE_MTE3);
        // rowF is rewritten by the gate load/convert below; make sure the k
        // store above has drained first (MTE3 read vs V write / MTE2 write hazard).
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        LoadRowF(gate + static_cast<size_t>(row) * gateStrideT);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(tailCache + addr + headDim, rowF, 0, 1, headBlocksF, 0, 0);
        pipe_barrier(PIPE_MTE3);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    __aicore__ inline void Process()
    {
        if (headDim == 0 || numTokens == 0 || numReqs == 0 || cap == 0) {
            return;
        }
        uint32_t total = numReqs * cap;
        for (uint32_t pid = GetBlockIdx(); pid < total; pid += GetBlockNum()) {
            ProcessOneRow(pid);
        }
    }

private:
    __gm__ float *tailCache;
    __gm__ float *k;
    __gm__ float *gate;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *lens;
    __gm__ int32_t *tailBlockTable;
    __gm__ int64_t *positions;

    __ubuf__ float *rowF;

    uint32_t numReqs;
    uint32_t numTokens;
    uint32_t headDim;
    uint32_t cap;
    uint32_t tailBlocks;
    uint32_t kStrideT;
    uint32_t gateStrideT;
    uint32_t tailStrideT;
    uint32_t headBytesF;
    uint32_t headBlocksF;
    int headRepeat;
};

#define KPOOL_TAIL_COMPRESS_FUNC_DEFINE(dtype)                                                     \
    extern "C" __global__ __aicore__ void kpool_tail_compress_##dtype(                             \
        GM_ADDR tailCache, GM_ADDR indexerCache, GM_ADDR k, GM_ADDR gate, GM_ADDR ape,             \
        GM_ADDR positions, GM_ADDR queryStartLoc, GM_ADDR lens, GM_ADDR cachedLens,                \
        GM_ADDR slotMapping, GM_ADDR tailBlockTable, uint32_t numReqs, uint32_t numTokens,         \
        uint32_t headDim, uint32_t kpool, uint32_t cap, uint32_t tailBlocks,                       \
        uint32_t indexerBlockSize, uint32_t idxBlocks, uint32_t mainBlockSize, uint32_t kStrideT,  \
        uint32_t gateStrideT, uint32_t tailStrideT, uint32_t idxStrideT)                           \
    {                                                                                              \
        KpoolTailCompress<dtype> op;                                                               \
        op.Init(tailCache, indexerCache, k, gate, ape, positions, queryStartLoc, lens, cachedLens, \
                slotMapping, tailBlockTable, numReqs, numTokens, headDim, kpool, cap, tailBlocks,  \
                indexerBlockSize, idxBlocks, mainBlockSize, kStrideT, gateStrideT, tailStrideT,    \
                idxStrideT);                                                                       \
        op.Process();                                                                              \
    }

#define KPOOL_TAIL_SPILL_FUNC_DEFINE(dtype)                                                  \
    extern "C" __global__ __aicore__ void kpool_tail_spill_##dtype(                          \
        GM_ADDR tailCache, GM_ADDR k, GM_ADDR gate, GM_ADDR queryStartLoc, GM_ADDR lens,     \
        GM_ADDR tailBlockTable, GM_ADDR positions, uint32_t numReqs, uint32_t numTokens,     \
        uint32_t headDim, uint32_t cap, uint32_t tailBlocks, uint32_t kStrideT,              \
        uint32_t gateStrideT, uint32_t tailStrideT)                                          \
    {                                                                                        \
        KpoolTailSpill op;                                                                   \
        op.Init(tailCache, k, gate, queryStartLoc, lens, tailBlockTable, positions, numReqs, \
                numTokens, headDim, cap, tailBlocks, kStrideT, gateStrideT, tailStrideT);    \
        op.Process();                                                                        \
    }
#else
#define KPOOL_TAIL_COMPRESS_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void kpool_tail_compress_##dtype(                            \
        GM_ADDR tailCache, GM_ADDR indexerCache, GM_ADDR k, GM_ADDR gate, GM_ADDR ape,            \
        GM_ADDR positions, GM_ADDR queryStartLoc, GM_ADDR lens, GM_ADDR cachedLens,               \
        GM_ADDR slotMapping, GM_ADDR tailBlockTable, uint32_t numReqs, uint32_t numTokens,        \
        uint32_t headDim, uint32_t kpool, uint32_t cap, uint32_t tailBlocks,                      \
        uint32_t indexerBlockSize, uint32_t idxBlocks, uint32_t mainBlockSize, uint32_t kStrideT, \
        uint32_t gateStrideT, uint32_t tailStrideT, uint32_t idxStrideT)                          \
    {                                                                                             \
        (void)tailCache;                                                                          \
        (void)indexerCache;                                                                       \
        (void)k;                                                                                  \
        (void)gate;                                                                               \
        (void)ape;                                                                                \
        (void)positions;                                                                          \
        (void)queryStartLoc;                                                                      \
        (void)lens;                                                                               \
        (void)cachedLens;                                                                         \
        (void)slotMapping;                                                                        \
        (void)tailBlockTable;                                                                     \
        (void)numReqs;                                                                            \
        (void)numTokens;                                                                          \
        (void)headDim;                                                                            \
        (void)kpool;                                                                              \
        (void)cap;                                                                                \
        (void)tailBlocks;                                                                         \
        (void)indexerBlockSize;                                                                   \
        (void)idxBlocks;                                                                          \
        (void)mainBlockSize;                                                                      \
        (void)kStrideT;                                                                           \
        (void)gateStrideT;                                                                        \
        (void)tailStrideT;                                                                        \
        (void)idxStrideT;                                                                         \
    }

#define KPOOL_TAIL_SPILL_FUNC_DEFINE(dtype)                                              \
    extern "C" __global__ __aicore__ void kpool_tail_spill_##dtype(                      \
        GM_ADDR tailCache, GM_ADDR k, GM_ADDR gate, GM_ADDR queryStartLoc, GM_ADDR lens, \
        GM_ADDR tailBlockTable, GM_ADDR positions, uint32_t numReqs, uint32_t numTokens, \
        uint32_t headDim, uint32_t cap, uint32_t tailBlocks, uint32_t kStrideT,          \
        uint32_t gateStrideT, uint32_t tailStrideT)                                      \
    {                                                                                    \
        (void)tailCache;                                                                 \
        (void)k;                                                                         \
        (void)gate;                                                                      \
        (void)queryStartLoc;                                                             \
        (void)lens;                                                                      \
        (void)tailBlockTable;                                                            \
        (void)positions;                                                                 \
        (void)numReqs;                                                                   \
        (void)numTokens;                                                                 \
        (void)headDim;                                                                   \
        (void)cap;                                                                       \
        (void)tailBlocks;                                                                \
        (void)kStrideT;                                                                  \
        (void)gateStrideT;                                                               \
        (void)tailStrideT;                                                               \
    }
#endif
