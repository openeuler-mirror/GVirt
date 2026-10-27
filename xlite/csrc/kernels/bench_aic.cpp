/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "microbench.h"

// Merged AIC micro-benchmark kernel (one kernel per TU - see microbench.h
// "Translation unit layout"). KERNEL_TASK_TYPE_DEFAULT is emitted once at the
// top; the switch dispatches on op (kernel_param.h BenchOp) to the per-op BODY macro.
// Valid ops: EMPTY + six scalar chains (both builds); MICROBENCH_TIMED adds the
// scalar timed twins and CUBE_MMAD.
extern "C" __global__ __aicore__ void bench_aic(GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR p3,
                                                GM_ADDR ticks, uint32_t seed, uint64_t sz0,
                                                uint64_t sz1, uint64_t sz2, uint32_t iters,
                                                uint32_t op)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    switch (op) {
        case BENCH_OP_EMPTY:
            BENCH_EMPTY_BODY();
            break;
        case BENCH_OP_SCALAR_ADD:
            BENCH_SCALAR_ADD_BODY();
            break;
        case BENCH_OP_SCALAR_XOR:
            BENCH_SCALAR_XOR_BODY();
            break;
        case BENCH_OP_SCALAR_MUL:
            BENCH_SCALAR_MUL_BODY();
            break;
        case BENCH_OP_SCALAR_DIV:
            BENCH_SCALAR_DIV_BODY();
            break;
        case BENCH_OP_SCALAR_FADD:
            BENCH_SCALAR_FADD_BODY();
            break;
        case BENCH_OP_SCALAR_ADD_UNROLL4:
            BENCH_SCALAR_ADD_UNROLL4_BODY();
            break;
#if MICROBENCH_TIMED
        case BENCH_OP_SCALAR_ADD_TIMED:
            BENCH_SCALAR_ADD_TIMED_BODY();
            break;
        case BENCH_OP_SCALAR_XOR_TIMED:
            BENCH_SCALAR_XOR_TIMED_BODY();
            break;
        case BENCH_OP_SCALAR_MUL_TIMED:
            BENCH_SCALAR_MUL_TIMED_BODY();
            break;
        case BENCH_OP_SCALAR_DIV_TIMED:
            BENCH_SCALAR_DIV_TIMED_BODY();
            break;
        case BENCH_OP_SCALAR_FADD_TIMED:
            BENCH_SCALAR_FADD_TIMED_BODY();
            break;
        case BENCH_OP_SCALAR_ADD_UNROLL4_TIMED:
            BENCH_SCALAR_ADD_UNROLL4_TIMED_BODY();
            break;
        case BENCH_OP_CUBE_MMAD:
            BENCH_CUBE_MMAD_BODY();
            break;
#endif
        default:
            break;
    }
}
