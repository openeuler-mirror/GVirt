/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */

#include "core_assigner.h"

float XCoreAssigner::AssignCore(bool isDecode)
{
    if (isDecode) {
        return AssignCoreForDecode();
    }
    return AssignCoreForPrefill();
}

float XCoreAssigner::AssignCoreForPrefill()
{
    std::unique_lock<std::mutex> lock(_mtx);
    _isPrefillRunning = true;
    _prefillOccupied = _isDecodeRunning ? _prefillConfigRatio : XLITE_MAX_CORE_RATIO;

    // 保证分配核数不超过总核数
    _cv.wait(lock, [this]()-> bool
    {
        if (!_isDecodeRunning) {
            return true;
        }

        return std::abs(XLITE_MAX_CORE_RATIO - _prefillOccupied - _decodeOccupied) < XLITE_UTILIZATION_EPSILON;
    });

    return _prefillOccupied;
}


float XCoreAssigner::AssignCoreForDecode()
{
    std::unique_lock<std::mutex> lock(_mtx);
    _isDecodeRunning = true;
    _decodeOccupied = _isPrefillRunning ? _decodeConfigRatio : XLITE_MAX_CORE_RATIO;

    // 保证分配核数不超过总核数
    _cv.wait(lock, [this]()-> bool
    {
        if (!_isPrefillRunning) {
            return true;
        }

        return std::abs(XLITE_MAX_CORE_RATIO - _prefillOccupied - _decodeOccupied) < XLITE_UTILIZATION_EPSILON;
    });

    return _decodeOccupied;
}

void XCoreAssigner::ReleaseCore(bool isDecode)
{
    std::unique_lock<std::mutex> lock(_mtx);

    if (isDecode) {
        _isDecodeRunning = false;
        _decodeOccupied = 0;
    } else {
        _isPrefillRunning = false;
        _prefillOccupied = 0;
    }

    _cv.notify_one();
}
