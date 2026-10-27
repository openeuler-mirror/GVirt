/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "microbench.h"

// Merged MIX micro-benchmark kernel (one kernel per TU - see microbench.h
// "Translation unit layout"). MIX routes cube+vector in one binary (hardware
// schedules 1 AIC : 2 AIV). Only the empty launch-floor probe has a MIX variant:
// pure scalar work has no cube/vector data-path component, so a mix launch
// would only add scheduling noise on top of the same per-core measurement.
extern "C" __global__ __aicore__ void bench_mix(GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR p3,
                                                GM_ADDR ticks, uint32_t seed, uint64_t sz0,
                                                uint64_t sz1, uint64_t sz2, uint32_t iters,
                                                uint32_t op)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    switch (op) {
        case BENCH_OP_EMPTY:
            BENCH_EMPTY_BODY();
            break;
        default:
            break;
    }
}
