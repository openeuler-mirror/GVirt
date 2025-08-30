/*
 * @file kernel_macro.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef _XLITE_KERNEL_MACRO_H_
#define _XLITE_KERNEL_MACRO_H_

#define ROUND_DOWN(x, y) (((x) / (y)) * (y))
#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define DIV_ROUND_UP(x, y) (((x) + ((y) - 1)) / (y))

#endif