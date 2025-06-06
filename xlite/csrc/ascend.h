/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_ASCEND_H_
#define _XLITE_ASCEND_H_

#include "acl/acl.h"
#include "hccl/hccl.h"

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __func__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
            return;                                                                         \
        }                                                                                   \
    } while (0);

#define CHECK_ACL_RET(x, _err)                                                              \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __func__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
            return _err;                                                                    \
        }                                                                                   \
    } while (0);

#define CHECK_HCCL(x)                                                                         \
    do {                                                                                      \
        HcclResult __ret = x;                                                                 \
        if (__ret != HCCL_SUCCESS) {                                                          \
            std::cerr << __func__ << ":" << __LINE__ << " HcclResult:" << __ret << std::endl; \
            return;                                                                           \
        }                                                                                     \
    } while (0);

#define CHECK_HCCL_RET(x, _err)                                                               \
    do {                                                                                      \
        HcclResult __ret = x;                                                                 \
        if (__ret != HCCL_SUCCESS) {                                                          \
            std::cerr << __func__ << ":" << __LINE__ << " HcclResult:" << __ret << std::endl; \
            return _err;                                                                      \
        }                                                                                     \
    } while (0);

#endif