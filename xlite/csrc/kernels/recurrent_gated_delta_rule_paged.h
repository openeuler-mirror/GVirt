/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Single-token GDN using vLLM's native [P, Hv, V, K] state (K contiguous).
 * Uses the row-wise matvec / K reduction approach of FLA-NPU's recurrent GDN.
 * No state transpose, private cache, workspace allocation, or host synchronization.
 */
#pragma once
#include "kernel_macro.h"
#include "../gdn_metadata.h"

#ifdef __DAV_C220_VEC__
template <typename Dtype>
class RecurrentGatedDeltaRulePaged
{
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR g,
                                GM_ADDR state, GM_ADDR out, GM_ADDR indices, uint32_t batch,
                                uint32_t kHeads, uint32_t vHeads, uint32_t kDim, uint32_t vDim,
                                uint64_t pageStride, uint32_t numPages, uint32_t stateFp32,
                                float scale, GM_ADDR cachedLens, uint64_t tableRowStride,
                                uint32_t tableColumns, uint32_t stateBlockSize, uint32_t aligned,
                                uint32_t gateFp32 = 0)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        qGm = (__gm__ Dtype *)query;
        kGm = (__gm__ Dtype *)key;
        vGm = (__gm__ Dtype *)value;
        betaGm = (__gm__ Dtype *)beta;
        gGm = (__gm__ Dtype *)g;
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
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->tableRowStride = tableRowStride;
        this->tableColumns = tableColumns;
        this->stateBlockSize = stateBlockSize;
        this->aligned = aligned;
        this->gateFp32 = gateFp32;

        // 64 V rows keep even K=128 well below the A2 192 KiB UB limit.
        uint64_t off = 0;
        ioTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += 512;
        stateTmp = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += std::is_same<Dtype, float>::value ? 0 : 64 * 128 * sizeof(Dtype);
        stateTmp2 = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += std::is_same<Dtype, float>::value ? 0 : 64 * 128 * sizeof(Dtype);
        qF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        kF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        vF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        stateF = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 64 * 128 * 4;
        stateF2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
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
        qF2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        kF2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        vF2 = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += 512;
        qStage = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += std::is_same<Dtype, float>::value ? 0 : 128 * sizeof(Dtype);
        kStage = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += std::is_same<Dtype, float>::value ? 0 : 128 * sizeof(Dtype);
        vStage = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += std::is_same<Dtype, float>::value ? 0 : 64 * sizeof(Dtype);
        decayRow = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += GATE_CAP * 4;
        betaRow = reinterpret_cast<__ubuf__ float *>((uintptr_t)off);
        off += GATE_CAP * 4;
        assert(off <= UB_SIZE);
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

    // Load g and beta for one decode head into decayRow[0]/betaRow[0]
    __aicore__ inline void LoadDecayBetaStage(uint64_t index)
    {
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(decayRow, gGm + index, sizeof(float));
            CopyGmToUbufAligned(betaRow, betaGm + index, sizeof(float));
        } else if (gateFp32) {
            CopyGmToUbufAligned(decayRow, (__gm__ float *)gGm + index, sizeof(float));
            CopyGmToUbufAligned(ioTmp, betaGm + index, sizeof(Dtype));
        } else {
            CopyGmToUbufAligned(ioTmp, gGm + index, sizeof(Dtype));
            CopyGmToUbufAligned(ioTmp + 64, betaGm + index, sizeof(Dtype));
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (!std::is_same<Dtype, float>::value) {
            if (gateFp32) {
                convert_input(betaRow, ioTmp, 1);
            } else {
                convert_input(decayRow, ioTmp, 1);
                convert_input(betaRow, ioTmp + 64, 1);
            }
            pipe_barrier(PIPE_V);
        }
    }

    // exp(g) and finish decay, beta load
    __aicore__ inline void LoadDecayBetaFinish(float &decay, float &beta)
    {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        vexp(decayRow, decayRow, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        decay = decayRow[0];
        beta = betaRow[0];
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }

    // Batch-load exp(g) and beta for a run of `n` tokens of one head into UB.
    __aicore__ inline void LoadGates(int32_t start, uint32_t n, uint32_t h)
    {
        uint32_t count = n * vHeads;
        uint64_t rowBase = (uint64_t)start * vHeads;

        // decay = exp(g)
        if (gateFp32) {
            CopyGmToUbufAligned(decayRow, (__gm__ float *)gGm + rowBase, count * 4);
        } else {
            CopyGmToUbufAligned(stateTmp, gGm + rowBase, count * sizeof(Dtype));
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if (!gateFp32) {
            convert_input(decayRow, stateTmp, DIV_ROUND_UP(count, 64));
            pipe_barrier(PIPE_V);
            // The beta gather reuses stateTmp below; protect this conversion's UB read.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        vexp(decayRow, decayRow, DIV_ROUND_UP(count, 64), 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // beta
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(betaRow, (__gm__ float *)betaGm + rowBase, count * 4);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        } else {
            CopyGmToUbufAligned(stateTmp, betaGm + rowBase, count * sizeof(Dtype));
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            convert_input(betaRow, stateTmp, DIV_ROUND_UP(count, 64));
            pipe_barrier(PIPE_V);
            // The next chunk's gather reuses stateTmp; protect this conversion's UB read.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }

    // Issue the q/k/v DMA for token `t` into ping-pong slot `buf`
    __aicore__ inline void PrefetchIssue(uint32_t buf, int32_t t, uint32_t kh, uint32_t h,
                                         uint32_t row)
    {
        uint64_t qo = ((uint64_t)t * kHeads + kh) * kDim;
        uint64_t vo = ((uint64_t)t * vHeads + h) * vDim + row;
        if constexpr (std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(buf == 0 ? qF : qF2, qGm + qo, kDim * 4);
            CopyGmToUbufAligned(buf == 0 ? kF : kF2, kGm + qo, kDim * 4);
            CopyGmToUbufAligned(buf == 0 ? vF : vF2, vGm + vo, 64 * 4);
        } else {
            CopyGmToUbufAligned(qStage, qGm + qo, kDim * sizeof(Dtype));
            CopyGmToUbufAligned(kStage, kGm + qo, kDim * sizeof(Dtype));
            CopyGmToUbufAligned(vStage, vGm + vo, 64 * sizeof(Dtype));
        }
    }

    // Wait for the prefetch DMA and (for fp16/bf16) convert the staging into slot `buf`
    __aicore__ inline void PrefetchFinish(uint32_t buf)
    {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (!std::is_same<Dtype, float>::value) {
            convert_input(buf == 0 ? qF : qF2, qStage, DIV_ROUND_UP(kDim, 64));
            convert_input(buf == 0 ? kF : kF2, kStage, DIV_ROUND_UP(kDim, 64));
            convert_input(buf == 0 ? vF : vF2, vStage, 1);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
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

    // Issue the state DMA for `buf`
    __aicore__ inline void LoadStateIssue(uint64_t offset, uint32_t buf)
    {
        uint32_t count = 64 * kDim;
        uint32_t id = buf == 0 ? EVENT_ID3 : EVENT_ID4;
        __ubuf__ float *sf = buf == 0 ? stateF : stateF2;
        __ubuf__ Dtype *st = buf == 0 ? stateTmp : stateTmp2;
        wait_flag(PIPE_MTE3, PIPE_MTE2, id);
        if (stateFp32 || std::is_same<Dtype, float>::value) {
            CopyGmToUbufAligned(sf, (__gm__ float *)stateGm + offset, count * 4);
        } else {
            CopyGmToUbufAligned(st, (__gm__ Dtype *)stateGm + offset, count * sizeof(Dtype));
        }
        set_flag(PIPE_MTE2, PIPE_V, id);
    }

    // Wait for the staged state DMA and (fp16/bf16) convert into slot `buf`
    __aicore__ inline void LoadStateFinish(uint32_t buf)
    {
        uint32_t count = 64 * kDim;
        uint32_t id = buf == 0 ? EVENT_ID3 : EVENT_ID4;
        __ubuf__ float *sf = buf == 0 ? stateF : stateF2;
        __ubuf__ Dtype *st = buf == 0 ? stateTmp : stateTmp2;
        wait_flag(PIPE_MTE2, PIPE_V, id);
        if (!stateFp32) {
            if constexpr (!std::is_same<Dtype, float>::value) {
                convert_input(sf, st, count / 64);
                pipe_barrier(PIPE_V);
            }
        }
    }

    // Convert + DMA the computed state out of `buf`, then signal it free
    __aicore__ inline void StoreStateIssue(uint64_t offset, uint32_t buf)
    {
        uint32_t count = 64 * kDim;
        uint32_t id = buf == 0 ? EVENT_ID3 : EVENT_ID4;
        __ubuf__ float *sf = buf == 0 ? stateF : stateF2;
        __ubuf__ Dtype *st = buf == 0 ? stateTmp : stateTmp2;
        if (!stateFp32) {
            if constexpr (!std::is_same<Dtype, float>::value) {
                convert_output(st, sf, count / 64);
                pipe_barrier(PIPE_V);
            }
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (stateFp32 || std::is_same<Dtype, float>::value) {
            CopyUbufToGmAligned((__gm__ float *)stateGm + offset, sf, count * 4);
        } else {
            CopyUbufToGmAligned((__gm__ Dtype *)stateGm + offset, st, count * sizeof(Dtype));
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, id);
    }

    __aicore__ inline void MatVec(__ubuf__ float *state, __ubuf__ float *vec, __ubuf__ float *dst)
    {
        uint32_t rowStride = kDim / 8;
        for (uint32_t col = 0; col < kDim; col += 64) {
            SetMask(MIN(64u, kDim - col));
            vmul(product + col, state + col, vec + col, 64, 1, 1, 1, rowStride, rowStride, 0);
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
        uint64_t headOffset = (uint64_t)page * pageStride + (uint64_t)h * vDim * kDim;
        uint32_t tiles = vDim / 64;
        LoadVector(qF, qGm + qo, kDim);
        LoadVector(kF, kGm + qo, kDim);
        LoadVector(vF, vGm + vo, vDim);
        vmuls(qF, qF, scale, DIV_ROUND_UP(kDim, 64), 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        LoadDecayBetaStage((uint64_t)b * vHeads + h);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        LoadStateIssue(headOffset, 0);
        float decay, beta;
        LoadDecayBetaFinish(decay, beta);
        for (uint32_t t = 0; t < tiles; ++t) {
            uint32_t cur = t & 1;
            uint32_t row = t * 64;
            __ubuf__ float *sf = cur == 0 ? stateF : stateF2;
            if (t + 1 < tiles) {
                LoadStateIssue(headOffset + (uint64_t)(row + 64) * kDim, cur ^ 1);
            }
            LoadStateFinish(cur);
            vmuls(sf, sf, decay, kDim, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            MatVec(sf, kF, delta);
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
                vadd(sf + col, sf + col, product + col, 64, 1, 1, 1, rowStride, rowStride,
                     rowStride);
                pipe_barrier(PIPE_V);
            }
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            MatVec(sf, qF, outF + row);
            StoreStateIssue(headOffset + (uint64_t)row * kDim, cur);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
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
            uint64_t offset = (uint64_t)page * pageStride + ((uint64_t)h * vDim + row) * kDim;
            if (hasInitial[b]) {
                LoadState(offset);
                if (!stateFp32) {
                    // stateTmp is reused by LoadGates right after
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                }
            } else {
                vector_dup(stateF, 0.0f, kDim, 1, 0, 8, 0);
                pipe_barrier(PIPE_V);
            }
            uint32_t kh = h / (vHeads / kHeads);
            uint32_t rowStride = kDim / 8;
            uint32_t gateChunk = MIN(256u, GATE_CAP / vHeads);
            for (int32_t cs = start; cs < end; cs += gateChunk) {
                uint32_t n = MIN((uint32_t)(end - cs), gateChunk);
                LoadGates(cs, n, h);
                uint32_t cur = 0;
                PrefetchIssue(0, cs, kh, h, row);
                PrefetchFinish(0);
                for (uint32_t i = 0; i < n; ++i) {
                    int32_t t = cs + i;
                    if (i + 1 < n) {
                        PrefetchIssue(1 - cur, t + 1, kh, h, row);
                    }
                    __ubuf__ float *q = cur == 0 ? qF : qF2;
                    __ubuf__ float *k = cur == 0 ? kF : kF2;
                    __ubuf__ float *v = cur == 0 ? vF : vF2;
                    uint64_t vo = ((uint64_t)t * vHeads + h) * vDim + row;
                    vmuls(q, q, scale, DIV_ROUND_UP(kDim, 64), 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    float decay = decayRow[i * vHeads + h];
                    float beta = betaRow[i * vHeads + h];
                    vmuls(stateF, stateF, decay, kDim, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    MatVec(stateF, k, delta);
                    vsub(delta, v, delta, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    vmuls(delta, delta, beta, 1, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    vbrcb((__ubuf__ uint32_t *)deltaBlocks, (__ubuf__ uint32_t *)delta, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    for (uint32_t col = 0; col < kDim; col += 64) {
                        SetMask(MIN(64u, kDim - col));
                        vmul(product + col, deltaBlocks, k + col, 64, 1, 0, 1, rowStride, 1, 0);
                        pipe_barrier(PIPE_V);
                        vadd(stateF + col, stateF + col, product + col, 64, 1, 1, 1, rowStride,
                             rowStride, rowStride);
                        pipe_barrier(PIPE_V);
                    }
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    MatVec(stateF, q, outF);
                    StoreVector(outGm + vo, outF, 64);
                    if (i + 1 < n) {
                        PrefetchFinish(1 - cur);
                    }
                    cur = 1 - cur;
                }
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
    static constexpr uint32_t GATE_CAP = XLITE_GDN_GATE_CAP;  // fp32 gate-row elements per chunk
    __gm__ Dtype *qGm, *kGm, *vGm, *betaGm, *gGm, *outGm;
    GM_ADDR stateGm;
    __gm__ int32_t *indicesGm, *cachedLens;
    uint64_t tableRowStride;
    uint32_t tableColumns, stateBlockSize, aligned;
    __ubuf__ Dtype *ioTmp, *stateTmp, *stateTmp2, *qStage, *kStage, *vStage;
    __ubuf__ float *qF, *kF, *vF, *qF2, *kF2, *vF2, *stateF, *stateF2, *product, *delta, *outF;
    __ubuf__ float *reduceTail, *deltaBlocks, *decayRow, *betaRow;
    uint32_t batch, kHeads, vHeads, kDim, vDim, numPages, stateFp32, gateFp32;
    uint64_t pageStride;
    float scale;
};
#define RECURRENT_GATED_DELTA_RULE_PAGED_BODY(dtype)                                            \
    RecurrentGatedDeltaRulePaged<dtype> op;                                                     \
    op.Init(query, key, value, beta, g, state, out, indices, batch, kHeads, vHeads, kDim, vDim, \
            pageStride, numPages, stateFp32, scale, cachedLens, tableRowStride, tableColumns,   \
            stateBlockSize, aligned, gateFp32);                                                 \
    op.Process();
#else
#define RECURRENT_GATED_DELTA_RULE_PAGED_BODY(dtype)
#endif
#define RECURRENT_GATED_DELTA_RULE_PAGED_FUNC_DEFINE(dtype)                                       \
    extern "C" __global__ __aicore__ void recurrent_gated_delta_rule_paged_##dtype(               \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR g, GM_ADDR state,        \
        GM_ADDR out, GM_ADDR indices, uint32_t batch, uint32_t kHeads, uint32_t vHeads,           \
        uint32_t kDim, uint32_t vDim, uint64_t pageStride, uint32_t numPages, uint32_t stateFp32, \
        float scale, GM_ADDR cachedLens, uint64_t tableRowStride, uint32_t tableColumns,          \
        uint32_t stateBlockSize, uint32_t aligned, uint32_t gateFp32)                             \
    {                                                                                             \
        RECURRENT_GATED_DELTA_RULE_PAGED_BODY(dtype)                                              \
    }

#ifdef __DAV_C220_VEC__
#define RECURRENT_GATED_DELTA_RULE_PREFILL_PAGED_BODY(dtype)                                    \
    RecurrentGatedDeltaRulePaged<dtype> op;                                                     \
    op.Init(query, key, value, beta, g, state, out, indices, batch, kHeads, vHeads, kDim, vDim, \
            pageStride, numPages, stateFp32, scale, nullptr, 0, 0, 0, 0, gateFp32);             \
    op.ProcessPrefill(starts, initial, tokens);
#else
#define RECURRENT_GATED_DELTA_RULE_PREFILL_PAGED_BODY(dtype)
#endif
#define RECURRENT_GATED_DELTA_RULE_PREFILL_PAGED_FUNC_DEFINE(dtype)                               \
    extern "C" __global__ __aicore__ void recurrent_gated_delta_rule_prefill_paged_##dtype(       \
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR g, GM_ADDR state,        \
        GM_ADDR out, GM_ADDR indices, uint32_t batch, uint32_t kHeads, uint32_t vHeads,           \
        uint32_t kDim, uint32_t vDim, uint64_t pageStride, uint32_t numPages, uint32_t stateFp32, \
        float scale, uint32_t gateFp32, GM_ADDR starts, GM_ADDR initial, uint32_t tokens)         \
    {                                                                                             \
        RECURRENT_GATED_DELTA_RULE_PREFILL_PAGED_BODY(dtype)                                      \
    }
