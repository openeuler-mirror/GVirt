/* Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved. */
#pragma once
#include <cstdint>

#ifdef __DAV_C220_VEC__
#define XLITE_GDN_INLINE __aicore__ inline
#else
#define XLITE_GDN_INLINE inline
#endif

// Logical column of the last scheduled token in a raw Mamba block table.
// none-mode has one live state slot; align-mode retains token-aligned columns.
// This selects a page only: vLLM still owns allocation, migration and checkpoints.
XLITE_GDN_INLINE uint32_t GdnStateColumn(uint32_t cached, uint32_t query, uint32_t blockSize,
                                         bool aligned)
{
    uint64_t total = static_cast<uint64_t>(cached) + query;
    return aligned && total != 0 ? static_cast<uint32_t>((total - 1) / blockSize) : 0;
}

XLITE_GDN_INLINE bool GdnHasInitialState(uint32_t cached)
{
    return cached != 0;
}

#ifdef __DAV_C220_VEC__
// Legacy callers supply resolved 1D indices. V2 supplies raw tables and cached lens.
XLITE_GDN_INLINE int32_t GdnStatePage(__gm__ int32_t *table, __gm__ int32_t *cachedLens,
                                      uint32_t row, uint64_t rowStride, uint32_t columns,
                                      uint32_t blockSize, bool aligned)
{
    if (cachedLens == nullptr) {
        return table[row];
    }
    int32_t cached = cachedLens[row];
    if (cached < 0 || !GdnHasInitialState(static_cast<uint32_t>(cached))) {
        return 0;
    }
    uint32_t column = GdnStateColumn(static_cast<uint32_t>(cached), 1, blockSize, aligned);
    return column < columns ? table[static_cast<uint64_t>(row) * rowStride + column] : 0;
}
#endif
#undef XLITE_GDN_INLINE
