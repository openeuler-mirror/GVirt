/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_BASE_H_
#define _XLITE_BASE_H_

#ifdef XLITE_DEBUG_ON
#include <torch/torch.h>
#endif
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <numeric>
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

#define DBG_LOC            \
    DebugSrcLoc            \
    {                      \
        __func__, __LINE__ \
    }

#define DBG_PREFIX (std::string(__func__) + ": ")
#define XT_STR(x) ((x).ToStr(#x) + ", ")

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

enum QuantType {
    UNKONOWN_QUANT,
    NO_QUANT,
    STATIC_QUANT,
    DYNAMIC_QUANT,
};

typedef struct DebugSrcLoc {
    const char *func;
    int line;
} DebugSrcLoc;

inline const char *XDtypeStr(enum XDtype dtype)
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

inline const char *XTensorTypeStr(enum XTensorType type)
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
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": unknown data type " + std::to_string(dtype));
    }
}

inline std::string ToSizeStr(size_t size)
{
    if (size < (1 << 10)) {
        return std::to_string(size) + "B";
    } else if (size < (1 << 20)) {
        return std::to_string(size >> 10) + "KB";
    } else {
        return std::to_string(size >> 20) + "MB";
    }
}

#ifdef XLITE_DEBUG_ON
at::ScalarType inline ToScalarType(enum XDtype dtype)
{
    switch (dtype) {
        case BIT1:
            return at::ScalarType::Bool;
        case INT8:
            return at::ScalarType::Char;
        case FP16:
            return at::ScalarType::Half;
        case BF16:
            return at::ScalarType::BFloat16;
        case FP32:
            return at::ScalarType::Float;
        case INT32:
            return at::ScalarType::Int;
        case INT64:
        case CPLXF:
            return at::ScalarType::Long;
        default:
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": unknown data type " + std::to_string(dtype));
    }
}
#endif

bool inline isEnvironmentVariableTrue(const char *env_value_cstr)
{
    if (env_value_cstr == nullptr) {
        return false;
    }

    std::string env_value = env_value_cstr;
    for (size_t i = 0; i < env_value.size(); ++i) {
        env_value[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(env_value[i])));
    }

    return env_value == "true" || env_value == "1" || env_value == "yes" || env_value == "on";
}

class XTensorPool;
class XTensor
{
public:
    XTensor() {};
    XTensor(std::vector<size_t> shape, enum XDtype dtype, void *ptr);
    void Init(std::vector<size_t> shape, enum XDtype dtype, void *ptr);
    // The os parameter lets callers (e.g. the debug module) capture the output into a
    // stream so it can be flushed atomically with a rank/color prefix. Defaults to
    // std::cout to keep legacy direct callers unchanged.
    void Print(const char *name = "", uint32_t nRow = 6, uint32_t nCol = 6,
               std::ostream &os = std::cout);
    void PrintPtr(const char *name, std::vector<size_t> &subShape, enum XDtype subDtype,
                  uint32_t nRow = 6, uint32_t nCol = 6, std::ostream &os = std::cout);
    void Memset(int value);
    std::string ToStr(const char *name = "") const;
    void View(std::vector<size_t> shape);
    void View(enum XDtype type);
    void Save(const std::string &path);
    bool CheckNanInf(const char *name = "", float threshold = -1.0f, std::ostream &os = std::cout);
    friend std::ostream &operator<<(std::ostream &os, const XTensor &p);
    enum XTensorType GetType()
    {
        return type;
    };
    std::vector<size_t> shape;
    size_t numel;
    enum XDtype dtype;
    void *ptr = nullptr;

private:
    void Init(std::vector<size_t> shape, enum XDtype dtype, void *ptr, enum XTensorType type);
    void PrintMemoryVal(void *p, uint64_t off, XDtype dtype, std::ostream &os = std::cout);
    enum XTensorType type = XTENSOR_STATIC;
    size_t bytes;
    friend class XTensorPool;
    friend class XDummyTensorPool;
};

class XTensorPool
{
public:
    XTensorPool(size_t size, uint32_t rankId) : _size(size), _rankId(rankId) {};
    virtual ~XTensorPool(void);
    virtual int Init(void);
    virtual XTensor &GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc);
    void PutTensor(XTensor &t);
    virtual bool TensorInPool(XTensor &t);
    void *Ptr()
    {
        return _ptr;
    };
    size_t Size()
    {
        return _size;
    };

protected:
    void *_ptr = nullptr;
    size_t _size;
    uint32_t _rankId;
    XTensor _t[XLITE_MAX_NUM_DYNAMIC_TENSOR];
    std::list<std::reference_wrapper<XTensor>> _free;
    std::list<std::reference_wrapper<XTensor>> _used;
};

class XDummyTensorPool : public XTensorPool
{
public:
    using XTensorPool::XTensorPool;

    int Init(void) override;
    XTensor &GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc) override;
    bool TensorInPool(XTensor &t) override;
    size_t maxUsedSize = 0;
    size_t currUsedSize = 0;
};

class MatmulWeight
{
public:
    std::string name;
    XTensor weight;
    XTensor inputScale;   // for static quantization
    XTensor inputOffset;  // for static quantization
    XTensor quantBias;    // for static and dynamic quantization
    XTensor deqScale;     // for static and dynamic quantization

    enum QuantType GetQuantType();
    bool IsQuanted();

private:
    enum QuantType quantType = UNKONOWN_QUANT;
};

#endif
