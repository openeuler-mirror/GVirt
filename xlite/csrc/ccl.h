
/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_CCL_H_
#define _XLITE_CCL_H_

#include <cstring>

#define XLITE_CCL_MAX_RANK_SIZE 32
#define EXPORT_KEY_LEN 65

class XcclComm {
public:
    XcclComm(uint32_t rankId, uint32_t rankSize) : _rankId(rankId), _rankSize(rankSize) {};
    ~XcclComm(void);
    int Init(const std::string &ip, uint32_t port);
    void *ipcMems[XLITE_CCL_MAX_RANK_SIZE];
    void *ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE];
private:
    uint32_t _rankId;
    uint32_t _rankSize;
    char _ipcKeys[XLITE_CCL_MAX_RANK_SIZE][EXPORT_KEY_LEN];
};

#endif