/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_ASCEND_H_
#define _XLITE_ASCEND_H_

#include "acl/acl.h"
#include "hccl/hccl.h"

/* ACL_SUCCESS and ACL_ERROR_NONE are both 0; newer toolkits deprecate the
 * latter with a compiler-level attribute, which -Werror rejects. CMake probes
 * the header and passes XLITE_ACL_OK; the fallback below keeps manual or
 * standalone compiles working when the probe result is absent. */
#ifndef XLITE_ACL_OK
#define XLITE_ACL_OK ACL_SUCCESS
#endif

#define CHECK_ACL(x)                                                                          \
    do {                                                                                      \
        aclError __ret = x;                                                                   \
        if (__ret != XLITE_ACL_OK) {                                                          \
            throw std::runtime_error(std::string(__func__) + ":" + std::to_string(__LINE__) + \
                                     " aclError:" + std::to_string(__ret));                   \
        }                                                                                     \
    } while (0);

#define TRACECHECK_ACL(x, loc)                                                   \
    do {                                                                         \
        aclError __ret = x;                                                      \
        if (__ret != XLITE_ACL_OK) {                                             \
            throw std::runtime_error(loc.ToStr() + std::string(__func__) + ":" + \
                                     std::to_string(__LINE__) +                  \
                                     " aclError:" + std::to_string(__ret));      \
        }                                                                        \
    } while (0);

#define CHECK_HCCL(x)                                                                         \
    do {                                                                                      \
        HcclResult __ret = x;                                                                 \
        if (__ret != HCCL_SUCCESS) {                                                          \
            throw std::runtime_error(std::string(__func__) + ":" + std::to_string(__LINE__) + \
                                     " HcclResult:" + std::to_string(__ret));                 \
        }                                                                                     \
    } while (0);

#define TRACECHECK_HCCL(x, loc)                                                  \
    do {                                                                         \
        HcclResult __ret = x;                                                    \
        if (__ret != HCCL_SUCCESS) {                                             \
            throw std::runtime_error(loc.ToStr() + std::string(__func__) + ":" + \
                                     std::to_string(__LINE__) +                  \
                                     " HcclResult:" + std::to_string(__ret));    \
        }                                                                        \
    } while (0);

#endif
