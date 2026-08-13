/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */

#pragma once

#include <cstdint>
#include "base.h"
#include "matmul_tracer.h"

class XRuntime;

namespace xlite_trace
{

// Facade over the MatmulTracer singleton.

// Record one matmul shape (m x n x k) into the histogram and flush it
// crash-safely to disk. rankId is used to name the per-rank output file.
static inline void RecordMatmul(uint32_t rankId, enum XDtype dtype, bool transpose, bool nz,
                                uint64_t m, uint64_t n, uint64_t k)
{
#ifdef XLITE_TRACE_MATMUL
    MatmulTracer::Instance().Record(rankId, dtype, transpose, nz, m, n, k);
#endif
};

// Record one shape per expert (m = counts[i]) for a grouped matmul with a shared
// n (outDim) and k (inDim). Reads the per-expert token counts for experts in
// [start, end) from device memory via a blocking D2H.
static inline void RecordGroupMatmul(XRuntime &rt, XTensor &counts, uint32_t start, uint32_t end,
                                     enum XDtype dtype, bool transpose, bool nz, uint64_t n,
                                     uint64_t k)
{
#ifdef XLITE_TRACE_MATMUL
    MatmulTracer::Instance().RecordGroup(rt, counts, start, end, dtype, transpose, nz, n, k);
#endif
};

}  // namespace xlite_trace
