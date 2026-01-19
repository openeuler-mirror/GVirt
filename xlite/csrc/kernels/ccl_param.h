#ifndef _XLITE_CCL_PARAM_H_
#define _XLITE_CCL_PARAM_H_

#define XLITE_CCL_MAX_RANK_SIZE 32
#define XLITE_IPC_MEM_FLAG_OFFSET 4096
#define COPY_SIZE 5120

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