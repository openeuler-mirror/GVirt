/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "microbench.h"

// AIC micro-benchmark item list - implementation lives entirely in
// microbench.h. BENCH_TIMED_ITEMS (invoked without a trailing semicolon: its
// default-build expansion is empty) adds the *_timed in-kernel-sampling
// twins ONLY in builds configured with CMake XLITE_BENCH_TIMED=ON; the
// default build's 7-kernel TU stays token-identical to the calibrated
// layout (frozen TU, see microbench.h).

BENCH_EMPTY(aic);
BENCH_SCALAR_ADD(aic);
BENCH_SCALAR_XOR(aic);
BENCH_SCALAR_MUL(aic);
BENCH_SCALAR_DIV(aic);
BENCH_SCALAR_FADD(aic);
BENCH_SCALAR_ADD_UNROLL4(aic);
BENCH_TIMED_ITEMS(aic)
BENCH_CUBE_ITEMS(aic)
