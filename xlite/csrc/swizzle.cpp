/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <map>
#include "base.h"

using Key = std::tuple<uint64_t, uint64_t>;

// Values were picked by running matmul_swizzle_perf.py script on Ascend 910B3 NPU.
static const std::map<Key, uint64_t> BestSwizzles = {
    {{6144, 2048}, 0xe01},
    {{2048, 6144}, 0xe01},
};

void XlitePickSwizzle(uint64_t n, uint64_t k, uint64_t *swizzle)
{
    const auto best = BestSwizzles.find({n, k});
    if (best != BestSwizzles.end()) {
        *swizzle = best->second;
    }
}
