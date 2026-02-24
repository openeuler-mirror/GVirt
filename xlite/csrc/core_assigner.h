/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */

#ifndef _XLITE_CORE_ASSIGNER_H
#define _XLITE_CORE_ASSIGNER_H

#include <condition_variable>

const float XLITE_MAX_CORE_RATIO = 1.0;
const float XLITE_UTILIZATION_EPSILON = 1e-5f;

class XCoreAssigner
{
public:
    XCoreAssigner(float prefillRatio) : _prefillConfigRatio(prefillRatio)
    {
        _decodeConfigRatio = XLITE_MAX_CORE_RATIO - _prefillConfigRatio;
    }

    float AssignCore(bool isDecode);
    void ReleaseCore(bool isDecode);

private:
    float AssignCoreForPrefill();
    float AssignCoreForDecode();

private:
    float _prefillConfigRatio{XLITE_MAX_CORE_RATIO};
    float _decodeConfigRatio{XLITE_MAX_CORE_RATIO};
    float _prefillOccupied{0};
    float _decodeOccupied{0};
    bool _isPrefillRunning{false};
    bool _isDecodeRunning{false};
    std::mutex _mtx;
    std::condition_variable _cv;
};

#endif  //_XLITE_CORE_ASSIGNER_H
