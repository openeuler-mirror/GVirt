/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "xlite_base.h"
#include "xlite_acl.h"

void XTensor::Init(std::vector<long> shape, enum XDtype dtype, void *ptr, enum XTensorType type)
{
    size_t numel = 1;

    if (shape.size() == 0) {
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

XTensor::XTensor(std::vector<long> shape, enum XDtype dtype, void *ptr)
{
    Init(shape, dtype, ptr, TORCH_NPU);
}

void XTensor::Init(std::vector<long> shape, enum XDtype dtype, void *ptr)
{
    Init(shape, dtype, ptr, TORCH_NPU);
}

int XTensorPool::Init(void)
{
    CHECK_ACL_RET(aclrtMalloc(&_ptr, _size, ACL_MEM_MALLOC_HUGE_FIRST), -ENOMEM);
    for (int i = 0; i < XLITE_MAX_NUM_DYNAMIC_TENSOR; i++) {
        _free.push_back(&this->_t[i]);
    }
    return 0;
}

XTensorPool::~XTensorPool(void)
{
    CHECK_ACL(aclrtFree(_ptr));
}

XTensor *XTensorPool::GetTensor(std::vector<long> shape, enum XDtype dtype)
{
    size_t numel = 1, size, free = _size;
    void *ptr = _ptr;
    XTensor *t, *use;

    if (shape.size() == 0) {
        std::cerr << __FILE__ << ":" << __LINE__ << "size is 0" << std::endl;
        return nullptr;
    }

    if (_free.empty()) {
        std::cerr << __FILE__ << ":" << __LINE__ << "dynamic tensor too many, please put after use" << std::endl;
        return nullptr;
    }
    t = _free.front();

    for (int i = 0; i < shape.size(); i++) {
        numel *= shape[i];
    }
    size = ROUND_UP(numel * XDtypeSize(dtype), XLITE_TENSOR_ALIGN);

    for (auto it = _used.begin(); it != _used.end(); it++) {
        use = (*it);
        free = reinterpret_cast<uintptr_t>(use->ptr) - reinterpret_cast<uintptr_t>(ptr);
        if (free >= size) {
            t->Init(shape, dtype, ptr, XLITE_DYNAMIC);
            _free.pop_front();
            _used.insert(it, t);
            return t;
        }
        ptr = (void *)((uint64_t)use->ptr + ROUND_UP(use->numel * XDtypeSize(use->dtype), XLITE_TENSOR_ALIGN));
    }
    if ((uint64_t)_ptr + _size - (uint64_t)ptr >= size) {
        t->Init(shape, dtype, ptr, XLITE_DYNAMIC);
        _free.pop_front();
        _used.push_back(t);
        return t;
    }

    std::cerr << __FILE__ << ":" << __LINE__ << "get " << size << " B failed, no free tensor" << std::endl;
    return nullptr;
}

void XTensorPool::PutTensor(XTensor *t)
{
    if (!t || t->type != XLITE_DYNAMIC) {
        return;
    }
    _used.remove(t);
    _free.push_back(t);
}