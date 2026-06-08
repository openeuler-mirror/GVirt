/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_KERNEL_PARAM_H_
#define _XLITE_KERNEL_PARAM_H_

// ccl kernel param
#define XLITE_CCL_MAX_RANK_SIZE 32
#define XLITE_IPC_MEM_FLAG_OFFSET 4096
#define COPY_SIZE 32768
#define MAX_TOTAL_COPY_SIZE 12800
#define DOUBLE_AIVNUM_SIZE_BOUND 327680

// ipc mem layout
struct XcclIpcMemData {
    uint64_t inputOffset;
    uint64_t outputOffset;
};

struct XcclParam {
    uint64_t ipcMems[XLITE_CCL_MAX_RANK_SIZE];
    uint64_t ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE];
};
#endif