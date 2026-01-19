/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <cstdint>
#include "ccl.h"

#define XLITE_DEFAULT_DEVS_PER_NODE 8
#define XLITE_DEFAULT_PORT 10266
#define XLITE_DEFAULT_COMM_OPTIMIZE_LEN 6144

typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *HcclComm;
class XTensorPool;
class XcclComm;

enum commType {
    TP,
    DP,
    MAX_COMM_TYPE,
};

class XRuntime {
public:
    XRuntime(uint32_t devid, size_t sizeMB = 0, uint32_t rankId = 0,
             uint32_t tpSize = 1, uint32_t dpSize = 1);
    ~XRuntime(void);
    void Synchronize(void);
    void EventWaitCurrStream(aclrtStream currStream);
    void EventRecordCurrStream(aclrtStream currStream);
    void MemcpyH2D(void *dst, void *src, size_t size);
    void UpdateCoreNum(float blockDimUtilization);
    int InitTensorPool(size_t sizeMB);
    uint32_t rankId(void) { return _rankId; };
    uint32_t tpSize(void) { return _tpSize; };
    uint32_t dpSize(void) { return _dpSize; };
    aclrtStream stream;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t originAicNum;
    uint32_t originAivNum;
    XTensorPool *pool = nullptr;
    HcclComm _tpComm = nullptr;
    HcclComm _dpComm = nullptr;
    uint32_t commOptimizeLen = XLITE_DEFAULT_COMM_OPTIMIZE_LEN;
    bool enableCommOptimize;
    XTensor hiddenStateSlice;

    XcclComm *_tpXcclComm = nullptr;
    XcclComm *_dpXcclComm = nullptr;

private:
    int GetNodeIps(void);
    int InitHcclComm(void);
    int InitXcclComm(void);
    void FiniXcclComm(void);
    uint32_t _devid;
    aclrtEvent _event;
    bool _init_outside = false;
    uint32_t _rankId;
    uint32_t _tpSize;
    uint32_t _dpSize;
    uint32_t _rankSize;
    uint32_t _nDevPerNode = XLITE_DEFAULT_DEVS_PER_NODE;
    uint32_t _port = XLITE_DEFAULT_PORT;
    std::vector<std::string> _ips;

    char _ipcXTensorKeys[XLITE_CCL_MAX_RANK_SIZE][EXPORT_KEY_LEN];
    void *_ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE] = {nullptr};
};

#endif
