/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_DSA_CP_WEIGHT_H_
#define _XLITE_DSA_CP_WEIGHT_H_

#include "base.h"

// FRACTAL_NZ stores each matrix as [column blocks, row blocks, 16, C0].
// C0 occupies 32 bytes for INT8/FP16/BF16 weights.
constexpr size_t DSA_CP_NZ_ROWS = 16;
constexpr size_t DSA_CP_NZ_COLUMN_BYTES = 32;

inline std::vector<size_t> DsaCpWeightStorageShape(const XTensor &weight, bool nz)
{
    if (!nz) {
        return weight.shape;
    }
    auto shape = weight.shape;
    const size_t cols = shape.back();
    shape.pop_back();
    const size_t rows = shape.back();
    shape.pop_back();
    const size_t columnBlock = DSA_CP_NZ_COLUMN_BYTES * 8 / XDtypeBit(weight.dtype);
    shape.insert(shape.end(), {DIV_ROUND_UP(cols, columnBlock), DIV_ROUND_UP(rows, DSA_CP_NZ_ROWS),
                               DSA_CP_NZ_ROWS, columnBlock});
    return shape;
}

#endif
