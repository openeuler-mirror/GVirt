/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_ASCEND_H_
#define _XLITE_ASCEND_H_

#include "acl/acl.h"
#include "hccl/hccl.h"

#define CHECK_ACL(x)                                                                          \
    do {                                                                                      \
        aclError __ret = x;                                                                   \
        if (__ret != ACL_ERROR_NONE) {                                                        \
            throw std::runtime_error(std::string(__func__) + ":" + std::to_string(__LINE__) + \
                                     " aclError:" + std::to_string(__ret));                   \
        }                                                                                     \
    } while (0);

#define CHECK_HCCL(x)                                                                         \
    do {                                                                                      \
        HcclResult __ret = x;                                                                 \
        if (__ret != HCCL_SUCCESS) {                                                          \
            throw std::runtime_error(std::string(__func__) + ":" + std::to_string(__LINE__) + \
                                     " HcclResult:" + std::to_string(__ret));                 \
        }                                                                                     \
    } while (0);

#endif