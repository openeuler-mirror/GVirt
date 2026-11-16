/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"

// UbAlloc: a monotonic (bump) allocator over the unified buffer (UB) for AIV
// cores.
//
// The unified buffer (UB, `__ubuf__ T *`) exists only on AIV cores, where it is
// a single 0-based byte address space. A `__ubuf__ T *` is just a byte offset
// reinterpreted as a typed pointer (subsequent pointer arithmetic then scales by
// sizeof(T)). AIC cores have no UB: they hold the cube A/B/C tiles in L1 and the
// per-matrix accumulators in L0A/L0B/L0C (`__cc__ T *` / `__l1__ T *`), which
// are allocated separately (see mla_aic_helper.h) and must not use this class.
// Kernels historically carved AIV buffers out of UB by hand:
//
//     uint64_t off = 0;
//     __ubuf__ Dtype *in0 = reinterpret_cast<__ubuf__ Dtype *>(off);
//     off += ROUND_UP(total_dim * sizeof(Dtype), BLOCK_SIZE);
//     ...
//     assert(off <= UB_SIZE);
//
// That spread the offset bookkeeping, the alignment, and the overflow check
// across every kernel. UbAlloc folds all three into named, typed allocations:
//
//     UbAlloc ub;
//     __ubuf__ Dtype *in[2] = {ub.Alloc<Dtype>(total_dim), ub.Alloc<Dtype>(total_dim)};
//     __ubuf__ float *calc0 = ub.Alloc<float>(total_dim);
//     ...
//
// Each allocation rounds its byte size up to UB_BUF_ALIGN_SIZE (32B) and
// asserts that the running offset stays within capacity (UB_SIZE), so an
// oversized layout fails at the offending allocation instead of relying on a
// manual final assert that is easy to forget or place too late.
class UbAlloc
{
public:
    __aicore__ inline explicit UbAlloc(uint64_t capacity = UB_SIZE) : m_off(0), m_capacity(capacity)
    {
    }

    // Allocate `nElements` of T, with size aligned to `align` (32B by default), returning
    // a typed UB pointer. If a buffer must start at an `align`-aligned offset, call
    // Align(align) first.
    template <typename T>
    __aicore__ inline __ubuf__ T *Alloc(uint64_t nElements, uint64_t align = UB_BUF_ALIGN_SIZE)
    {
        return reinterpret_cast<__ubuf__ T *>(AllocBytes(nElements * sizeof(T), align));
    }

    // Allocate `nBytes` (size aligned to `align`, 32B by default) and return a typed
    // UB pointer. Use this when the element type is not a plain multiple of the
    // stored dtype (e.g. an fp32 payload staged through a Dtype-sized buffer).
    // If a buffer must start at an `align`-aligned offset, call Align(align) first.
    template <typename T>
    __aicore__ inline __ubuf__ T *AllocBytesAs(uint64_t nBytes, uint64_t align = UB_BUF_ALIGN_SIZE)
    {
        return reinterpret_cast<__ubuf__ T *>(AllocBytes(nBytes, align));
    }

    // Allocate `nBytes` (size aligned to `align`, 32B by default) and return the
    // byte offset of the buffer. If a buffer must start at an `align`-aligned
    // offset, call Align(align) first.
    __aicore__ inline uint64_t AllocBytes(uint64_t nBytes, uint64_t align = UB_BUF_ALIGN_SIZE)
    {
        nBytes = ROUND_UP(nBytes, align);
        const uint64_t base = m_off;
        m_off += nBytes;
        assert(m_off <= m_capacity);
        return base;
    }

    // Round the current offset up to `align` (e.g. UB_BANKGROUP_ROW_SIZE to put
    // the next buffer at the start of a fresh bank group, avoiding bank conflicts).
    __aicore__ inline void Align(uint64_t align)
    {
        m_off = ROUND_UP(m_off, align);
        assert(m_off <= m_capacity);
    }

    // Total bytes allocated so far.
    __aicore__ inline uint64_t Offset() const
    {
        return m_off;
    }

private:
    uint64_t m_off;
    uint64_t m_capacity;
};
