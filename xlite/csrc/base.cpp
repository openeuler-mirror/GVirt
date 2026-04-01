/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <cstring>
#include <iomanip>
#include "base.h"
#include "ascend.h"

void XTensor::Init(std::vector<size_t> shape, enum XDtype dtype, void *ptr, enum XTensorType type)
{
    size_t numel = 1;

    if (shape.empty()) {
        numel = 0;
    }

    for (uint32_t i = 0; i < shape.size(); i++) {
        numel *= shape[i];
    }

    this->numel = numel;
    this->shape = shape;
    this->dtype = dtype;
    this->ptr = ptr;
    this->type = type;
}

XTensor::XTensor(std::vector<size_t> shape, enum XDtype dtype, void *ptr)
{
    Init(std::move(shape), dtype, ptr, XTENSOR_STATIC);
}

void XTensor::Init(std::vector<size_t> shape, enum XDtype dtype, void *ptr)
{
    Init(std::move(shape), dtype, ptr, XTENSOR_STATIC);
}

void XTensor::PrintMemoryVal(void *p, uint64_t off, XDtype dtype)
{
    switch (dtype) {
        case BIT1: {
            uint64_t *raw = static_cast<uint64_t *>(p) + off / 64;
            uint32_t val = ((*raw) & (1ull << (off % 64))) ? 1 : 0;
            std::cout << val;
            break;
        }
        case INT8: {
            int32_t val = (static_cast<int8_t *>(p))[off];
            if (val >= 0) {
                std::cout << " ";
            }
            std::cout << val;
            break;
        }
        case INT32: {
            int32_t val = (static_cast<int32_t *>(p))[off];
            if (val >= 0) {
                std::cout << " ";
            }
            std::cout << val;
            break;
        }
        case INT64: {
            int64_t val = (static_cast<int64_t *>(p))[off];
            if (val >= 0) {
                std::cout << " ";
            }
            std::cout << val;
            break;
        }
        case FP16: {
            __fp16 val = (static_cast<__fp16 *>(p))[off];
            if (val >= 0) {
                std::cout << " ";
            }
            std::cout << std::scientific << std::setprecision(4);
            std::cout << val;
            std::cout.unsetf(std::ios::scientific);
            break;
        }
        case BF16: {
            uint16_t data = (static_cast<uint16_t *>(p))[off];
            uint32_t float32Data = (static_cast<uint32_t>(data) << 16);
            float val;
            std::memcpy(&val, &float32Data, sizeof(float));
            if (val >= 0) {
                std::cout << " ";
            }
            std::cout << std::scientific << std::setprecision(4);
            std::cout << val;
            std::cout.unsetf(std::ios::scientific);
            break;
        }
        case FP32: {
            float val = (static_cast<float *>(p))[off];
            if (val >= 0) {
                std::cout << " ";
            }
            std::cout << std::scientific << std::setprecision(4);
            std::cout << val;
            std::cout.unsetf(std::ios::scientific);
            break;
        }
        case CPLXF: {
            std::complex<float> val = (static_cast<std::complex<float> *>(p))[off];
            if (val.real() >= 0 && val.imag() >= 0) {
                std::cout << " ";
            }
            std::cout << std::scientific << std::setprecision(4);
            std::cout << val.real();
            if (val.imag() >= 0) {
                std::cout << " + ";
            } else {
                std::cout << " - ";
            }
            std::cout << std::abs(val.imag()) << "j";
            std::cout.unsetf(std::ios::scientific);
            break;
        }
        default:
            break;
    }
}

void XTensor::Print(const char *name, uint32_t nRow, uint32_t nCol)
{
    uint32_t i, j;
    uint32_t hRow = DIV_ROUND_UP(nRow, 2);
    uint32_t hCol = DIV_ROUND_UP(nCol, 2);
    size_t size = numel * XDtypeBit(dtype) / 8;
    aclError err;

    if (size == 0) {
        return;
    }

    void *p = malloc(size);
    if (!p) {
        return;
    }

    err = aclrtMemcpy(p, size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (err != ACL_ERROR_NONE) {
        free(p);
        return;
    }

    std::cout << name << ": XTensor(";
    for (uint32_t i = 0; i < shape.size(); i++) {
        std::cout << "[";
    }

    size_t col = shape[shape.size() - 1];
    size_t row = numel / col;
    for (j = 0; j < row && j < hRow; j++) {
        for (i = 0; i < col && i < hCol; i++) {
            PrintMemoryVal(p, j * col + i, dtype);
            if (i != col - 1) {
                std::cout << ", ";
            }
        }

        if (col > hCol && i < col - hCol) {
            std::cout << " ..., ";
            i = col - hCol;
        }

        for (; i < col; i++) {
            PrintMemoryVal(p, j * col + i, dtype);
            if (i != col - 1) {
                std::cout << ", ";
            }
        }
        if (j != row - 1) {
            std::cout << "]," << std::endl << "        ";
            for (uint32_t i = 0; i < shape.size() - 1; i++) {
                std::cout << " ";
            }
            if (j != hRow - 1 || j >= row - hRow - 1) {
                std::cout << "[";
            }
        }
    }

    if (row > hRow && j < row - hRow) {
        std::cout << "...," << std::endl << "        ";
        for (uint32_t i = 0; i < shape.size() - 1; i++) {
            std::cout << " ";
        }
        std::cout << "[";
        j = row - hRow;
    }

    for (; j < row; j++) {
        for (i = 0; i < col && i < hCol; i++) {
            PrintMemoryVal(p, j * col + i, dtype);
            if (i != col - 1) {
                std::cout << ", ";
            }
        }

        if (col > hCol && i < col - hCol) {
            std::cout << " ..., ";
            i = col - hCol;
        }

        for (; i < col; i++) {
            PrintMemoryVal(p, j * col + i, dtype);
            if (i != col - 1) {
                std::cout << ", ";
            }
        }
        if (j != row - 1) {
            std::cout << "]," << std::endl << "        ";
            for (uint32_t i = 0; i < shape.size() - 1; i++) {
                std::cout << " ";
            }
            std::cout << "[";
        }
    }

    for (uint32_t i = 0; i < shape.size(); i++) {
        std::cout << "]";
    }
    std::cout << ", shape=(";
    for (uint32_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i != shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "), dtype=" << XDtypeStr(dtype) << ")" << std::endl;
    free(p);
}

void XTensor::Memset(int value)
{
    size_t size = numel * XDtypeBit(dtype) / 8;
    if (size == 0) {
        return;
    }
    CHECK_ACL(aclrtMemset(ptr, size, value, size));
}

std::ostream &operator<<(std::ostream &os, const XTensor &p)
{
    os << "[(";
    for (size_t i = 0; i < p.shape.size(); i++) {
        os << p.shape[i];
        if (i != p.shape.size() - 1) {
            os << ", ";
        }
    }
    os << ") " << XDtypeStr(p.dtype) << " (" << XTensorTypeStr(p.type) << ")]";
    return os;
}

int XTensorPool::Init(void)
{
    CHECK_ACL(aclrtMalloc(&_ptr, _size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (int i = 0; i < XLITE_MAX_NUM_DYNAMIC_TENSOR; i++) {
        _free.push_back(this->_t[i]);
    }
    return 0;
}

XTensorPool::~XTensorPool(void)
{
    if (_ptr != nullptr) {
        (void)aclrtFree(_ptr);
    }
}

XTensor &XTensorPool::GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc)
{
    size_t numel = 1, size, free;
    void *ptr = _ptr;

    if (shape.empty()) {
        std::cerr << loc.file << ":" << loc.line << ": size is 0" << std::endl;
        throw std::invalid_argument("get tensor shape size is 0");
    }

    if (_free.empty()) {
        std::cerr << loc.file << ":" << loc.line
                  << ": dynamic tensor too many, please put after use" << std::endl;
        throw std::runtime_error("dynamic tensor too many, please put after use");
    }
    XTensor &t = _free.front();

    for (uint64_t i = 0; i < shape.size(); i++) {
        numel *= shape[i];
    }
    size = ROUND_UP(numel * XDtypeBit(dtype) / 8, XLITE_TENSOR_ALIGN);

    for (auto it = _used.begin(); it != _used.end(); it++) {
        XTensor &use = it->get();
        free = reinterpret_cast<uintptr_t>(use.ptr) - reinterpret_cast<uintptr_t>(ptr);
        if (free >= size) {
            t.Init(shape, dtype, ptr, XTENSOR_DYNAMIC);
            _free.pop_front();
            _used.insert(it, t);
            return t;
        }
        ptr = reinterpret_cast<void *>(
            reinterpret_cast<uint64_t>(use.ptr) +
            ROUND_UP(use.numel * XDtypeBit(use.dtype) / 8, XLITE_TENSOR_ALIGN));
    }
    if (reinterpret_cast<uint64_t>(_ptr) + _size - reinterpret_cast<uint64_t>(ptr) >= size) {
        t.Init(shape, dtype, ptr, XTENSOR_DYNAMIC);
        _free.pop_front();
        _used.push_back(t);
        return t;
    }

    std::cerr << loc.file << ":" << loc.line << ": get " << size << " B failed, no free tensor";
    std::cerr << ", shape=(";
    for (uint32_t i = 0; i < shape.size(); i++) {
        std::cerr << shape[i];
        if (i != shape.size() - 1) {
            std::cerr << ", ";
        }
    }
    std::cerr << "), dtype=" << XDtypeStr(dtype) << std::endl;
    throw std::runtime_error("no free tensor");
}

void XTensorPool::PutTensor(XTensor &t)
{
    if (t.type != XTENSOR_DYNAMIC) {
        return;
    }
    for (auto it = _used.begin(); it != _used.end(); ++it) {
        if (&it->get() == &t) {
            _used.erase(it);
            _free.push_back(t);
            break;
        }
    }
}

bool XTensorPool::TensorInPool(XTensor &t)
{
    return t.ptr >= _ptr &&
           t.ptr < reinterpret_cast<void *>(reinterpret_cast<uint64_t>(_ptr) + _size);
}

int XDummyTensorPool::Init(void)
{
    for (int i = 0; i < XLITE_MAX_NUM_DYNAMIC_TENSOR; i++) {
        _free.push_back(this->_t[i]);
    }
    return 0;
}

XTensor &XDummyTensorPool::GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc)
{
    size_t numel = 1, size;

    if (shape.empty()) {
        std::cerr << loc.file << ":" << loc.line << ": size is 0" << std::endl;
        throw std::invalid_argument("get tensor shape size is 0");
    }

    if (_free.empty()) {
        std::cerr << loc.file << ":" << loc.line
                  << ": dynamic tensor too many, please put after use" << std::endl;
        throw std::runtime_error("dynamic tensor too many, please put after use");
    }

    for (uint64_t i = 0; i < shape.size(); i++) {
        numel *= shape[i];
    }
    size = ROUND_UP(numel * XDtypeBit(dtype) / 8, XLITE_TENSOR_ALIGN);
    currUsedSize += size;
    if (currUsedSize > maxUsedSize) {
        maxUsedSize = currUsedSize;
    }
    XTensor &t = _free.front();
    t.Init(std::move(shape), dtype, nullptr, XTENSOR_DYNAMIC);
    _free.pop_front();
    _used.push_back(t);
    return t;
}

void XDummyTensorPool::PutTensor(XTensor &t)
{
    if (t.type != XTENSOR_DYNAMIC) {
        return;
    }
    size_t numel = 1, size;
    for (uint64_t i = 0; i < t.shape.size(); i++) {
        numel *= t.shape[i];
    }
    size = ROUND_UP(numel * XDtypeBit(t.dtype) / 8, XLITE_TENSOR_ALIGN);
    currUsedSize -= size;
    XTensorPool::PutTensor(t);
}

bool XDummyTensorPool::TensorInPool(XTensor &t)
{
    return t.type == XTENSOR_DYNAMIC;
}