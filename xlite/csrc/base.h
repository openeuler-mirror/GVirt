/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_BASE_H_
#define _XLITE_BASE_H_

#include <cstdio>
#include <iostream>
#include <vector>
#include <functional>
#include <list>
#include <cstdint>
#include <complex>
#include <algorithm>

#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define DIV_ROUND_UP(x, y) (((x) + ((y) - 1)) / (y))
#define ROUND_DOWN(x, y) (((x) / (y)) * (y))

#define MB_BIT 20
#define XLITE_MAX_NUM_DYNAMIC_TENSOR 128
#define XLITE_TENSOR_ALIGN 1024

#define DBG_LOC DebugSrcLoc{__FILE__, __LINE__}

enum XDtype {
    BIT1,
    INT8,
    INT32,
    INT64,
    FP16,
    BF16,
    FP32,
    CPLXF,
    MAX_XDTYPE,
};

enum XTensorType {
    XTENSOR_STATIC,
    XTENSOR_DYNAMIC,
    MAX_XTENSOR_TYPE,
};

enum XRopeType {
    NORMAL,
    INPLACE,
    MIX,
};

typedef struct DebugSrcLoc {
    const char* file;
    int line;
} DebugSrcLoc;

inline const char * XDtypeStr(enum XDtype dtype)
{
    switch (dtype) {
        case BIT1:
            return "BIT1";
        case INT8:
            return "INT8";
        case INT32:
            return "INT32";
        case INT64:
            return "INT64";
        case FP16:
            return "FP16";
        case BF16:
            return "BF16";
        case FP32:
            return "FP32";
        case CPLXF:
            return "CPLXF";
        default:
            return "unknown";
    }
}

inline const char * XTensorTypeStr(enum XTensorType type)
{
    switch (type) {
        case XTENSOR_STATIC:
            return "static";
        case XTENSOR_DYNAMIC:
            return "dynamic";
        default:
            return "unknown";
    }
}

size_t inline XDtypeBit(enum XDtype dtype)
{
    switch (dtype) {
        case BIT1:
            return 1;
        case INT8:
            return 8;
        case FP16:
        case BF16:
            return 16;
        case INT32:
        case FP32:
            return 32;
        case INT64:
        case CPLXF:
            return 64;
        default:
            std::cerr << __FILE__ << ":" << __LINE__ << "unknown data type " << dtype << std::endl;
            return 0;
    }
}

class XTensorPool;
class XTensor {
public:
    XTensor() {};
    XTensor(std::vector<long> shape, enum XDtype dtype, void *ptr);
    void Init(std::vector<long> shape, enum XDtype dtype, void *ptr);
    void Print(const char *name = "", uint32_t nRow = 6, uint32_t nCol = 6);
    friend std::ostream& operator<<(std::ostream& os, const XTensor& p);
    enum XTensorType GetType() { return type; };
    std::vector<long> shape;
    size_t numel;
    enum XDtype dtype;
    void *ptr;

private:
    void Init(std::vector<long> shape, enum XDtype dtype, void *ptr, enum XTensorType type);
    void PrintMemoryVal(void *p, uint64_t off, XDtype dtype);
    enum XTensorType type;
    friend class XTensorPool;
};

class XTensorPool {
public:
    XTensorPool(size_t size) : _size(size) {};
    ~XTensorPool(void);
    int Init(void);
    XTensor &GetTensor(std::vector<long> shape, enum XDtype dtype, DebugSrcLoc loc);
    void PutTensor(XTensor &t);
    void *Ptr() { return _ptr; };
    size_t Size() { return _size; };

private:
    void *_ptr;
    size_t _size;
    XTensor _t[XLITE_MAX_NUM_DYNAMIC_TENSOR];
    std::list<std::reference_wrapper<XTensor>> _free;
    std::list<std::reference_wrapper<XTensor>> _used;
};

#endif