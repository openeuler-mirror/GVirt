/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include "xlite.h"

namespace py = pybind11;

static inline enum XDtype XDtype(at::Tensor &t)
{
    switch (t.scalar_type()) {
        case at::kHalf:
            return FP16;
        default:
            return MAX_XDTYPE;
    }
}

static inline void *TensorPtr(at::Tensor &t)
{
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(t.storage().data_ptr().get()) +
        t.storage_offset() * t.dtype().itemsize());
}

void Add(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z)
{
    XTensor _x(x.sizes().vec(), XDtype(x), TensorPtr(x));
    XTensor _y(y.sizes().vec(), XDtype(y), TensorPtr(y));
    XTensor _z(z.sizes().vec(), XDtype(z), TensorPtr(z));
    XliteOpAdd(rt, &_x, &_y, &_z);
}

PYBIND11_MODULE(_C, m) {
    py::class_<XRuntime>(m, "runtime")
        .def(py::init<uint32_t, size_t>());
    m.def("add", &Add);
}