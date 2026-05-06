/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

template <typename T>
static __aicore__ inline void DumpBuffer(__ubuf__ T *buf, const __gm__ char *name, int size,
                                         int step = 1, int offset = 0, bool toInt = false)
{
#ifdef XLITE_KERNEL_DEBUG
    pipe_barrier(PIPE_ALL);
    printf("%s (%lx): \n[", name, (unsigned long)buf);
    for (int i = 0; i < size; i++) {
        if (i % 10 == 0) {
            printf("\n");
        }
        if (toInt) {
            printf("%u ", buf[i * step + offset]);
        } else {
            printf("%f ", buf[i * step + offset]);
        }
    }
    printf("]\n");
    pipe_barrier(PIPE_ALL);
#endif
}

#ifdef XLITE_KERNEL_DEBUG
#define dbg_printf(args...) printf(args)
#else
#define dbg_printf(args...)
#endif