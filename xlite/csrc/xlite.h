/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_H_
#define _XLITE_H_

// Umbrella header exposing the public C++ API.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "base.h"
#include "runtime.h"
#include "model.h"

inline enum XDtype XDtypeOf(const torch::Tensor &t)
{
    switch (t.scalar_type()) {
        case at::ScalarType::Char:
            return INT8;
        case at::ScalarType::Int:
            return INT32;
        case at::ScalarType::Long:
            return INT64;
        case at::ScalarType::Half:
            return FP16;
        case at::ScalarType::BFloat16:
            return BF16;
        case at::ScalarType::Float:
            return FP32;
        case at::ScalarType::ComplexFloat:
            return CPLXF;
        default:
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": unknown torch dtype for xlite: " +
                                     std::to_string(static_cast<int>(t.scalar_type())));
    }
}

inline void *TensorPtr(const torch::Tensor &t)
{
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(t.storage().data_ptr().get()) +
                                    t.storage_offset() * t.dtype().itemsize());
}

inline void InitXTensor(XTensor &out, const torch::Tensor &in)
{
    auto sizesVec = in.sizes().vec();
    std::vector<size_t> sizes;
    sizes.reserve(sizesVec.size());

    for (auto s : sizesVec) {
        if (s < 0) {
            throw std::runtime_error("Negative size detected: " + std::to_string(s));
        }
        sizes.push_back(static_cast<size_t>(s));
    }

    out.Init(sizes, XDtypeOf(in), TensorPtr(in));
}

#endif  // _XLITE_H_
