
/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_CCL_H_
#define _XLITE_CCL_H_

#include <cstdint>
#include <cstring>
#include <string>

#define XLITE_CCL_MAX_RANK_SIZE 32
#define EXPORT_KEY_LEN 65

class XcclComm
{
public:
    XcclComm(uint32_t rankId, uint32_t rankSize) : _rankId(rankId), _rankSize(rankSize) {};
    ~XcclComm(void);
    int Init(const std::string &ip, uint32_t port, void *ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE]);
    struct XcclParam *dParam;  // device param pointer
    uint64_t generation = 1;

private:
    uint32_t _rankId;
    uint32_t _rankSize;
    void *_ipcMems[XLITE_CCL_MAX_RANK_SIZE];
    char _ipcKeys[XLITE_CCL_MAX_RANK_SIZE][EXPORT_KEY_LEN];
};

#endif