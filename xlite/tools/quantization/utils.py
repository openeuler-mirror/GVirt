#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import gc
try:
    import torch_npu
    NPU_AVAILABLE = torch_npu.npu.is_available()
except Exception:
    NPU_AVAILABLE = False


def cleanup_memory(device=None) -> None:
    """Run GC and clear GPU memory."""
    gc.collect()

    if NPU_AVAILABLE:
        if device is not None:
            with torch_npu.npu.device(device):
                torch_npu.npu.empty_cache()
        else:
            torch_npu.npu.empty_cache()
