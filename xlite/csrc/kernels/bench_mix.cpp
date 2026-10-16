/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "microbench.h"

// MIX-routed micro-benchmark test item: cube+vector in one binary, hardware
// schedules 1 AIC : 2 AIV. Only the empty kernel has a MIX variant - pure
// scalar work has no cube/vector data-path component, so a mix launch would
// only add scheduling noise on top of the same per-core measurement (see
// microbench.h and doc/kernels/microbench.md).

BENCH_EMPTY(mix);
