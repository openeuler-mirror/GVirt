/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "microbench.h"

// AIV micro-benchmark item list - implementation lives entirely in
// microbench.h. The three no-semicolon macros below expand to NOTHING in
// the default build, so the 7-kernel TU stays token-identical to the
// calibrated layout (frozen TU, see microbench.h). A build configured with
// CMake XLITE_BENCH_TIMED=ON additionally gets, in this same TU:
//   BENCH_TIMED_ITEMS  the *_timed scalar twins (experiment 2)
//   BENCH_MTE_ITEMS    the MTE memory probes (experiment 3; AIV only - the
//                      MTE pipes live on the vector core)
//   BENCH_VEC_ITEMS    the vector-op probes (experiment 4; AIV only - the
//                      vector pipe lives on the vector core)

BENCH_EMPTY(aiv);
BENCH_SCALAR_ADD(aiv);
BENCH_SCALAR_XOR(aiv);
BENCH_SCALAR_MUL(aiv);
BENCH_SCALAR_DIV(aiv);
BENCH_SCALAR_FADD(aiv);
BENCH_SCALAR_ADD_UNROLL4(aiv);
BENCH_TIMED_ITEMS(aiv)
BENCH_MTE_ITEMS(aiv)
BENCH_VEC_ITEMS(aiv)
