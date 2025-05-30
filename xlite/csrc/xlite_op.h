/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_OP_H_
#define _XLITE_OP_H_

#include "xlite_base.h"

class XRuntime;
void XliteOpAdd(XRuntime &rt, XTensor *x, XTensor *y, XTensor *z);

#endif