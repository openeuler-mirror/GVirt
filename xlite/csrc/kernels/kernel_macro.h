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
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BLOCK_SIZE 32
#define VECTOR_MAX_BYTESIZE 256               // The maximum byte size of one repeat in vector
#define VECTOR_MAX_NUM_OF_FP32 64             // The maximum num of float32 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_FP16 128            // The maximum num of float16 dtype in one vector repeat
#endif