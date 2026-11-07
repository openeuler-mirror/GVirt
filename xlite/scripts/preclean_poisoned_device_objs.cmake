# coding=utf-8
# =============================================================================
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# =============================================================================
# Delete device objects poisoned by CANN's merge_aic/aiv_obj_text steps.
#
# Those steps relink every device object in place (`ld.lld -m aicorelinux
# -Ttext=0 <f>.o -o <f>.o`) to merge template text sections into `.text`,
# which flips the object from ET_REL to ET_EXEC. The merge is a custom ALL
# target, so it re-runs on every build and the already-merged objects fail
# the next incremental build with `ld.lld: error: ...: unknown file type`.
#
# Run via `cmake -DCLR_DIR=<binary_dir> -P <this script>` before each kernel
# preprocess ExternalProject (see CMakeLists.txt); deleting them lets the
# object libraries rebuild them fresh. Fresh builds match nothing (no-op).
#
# Only the aic/aiv object dirs of the ascendc kernel preprocess projects are
# scanned, and an object is removed only when its ELF type is not ET_REL, so
# valid outputs are never touched.

if(NOT CLR_DIR)
    message(FATAL_ERROR "CLR_DIR (the cmake binary dir) is required")
endif()

set(_poisoned 0)
foreach(_glob IN ITEMS
        "${CLR_DIR}/xlite_kernels_*_preprocess-prefix/src/*-build/CMakeFiles/aic_obj.dir/**/*.o"
        "${CLR_DIR}/xlite_kernels_*_preprocess-prefix/src/*-build/CMakeFiles/aiv_obj.dir/**/*.o")
    file(GLOB_RECURSE _objs "${_glob}")
    foreach(_obj IN LISTS _objs)
        # First 18 bytes: 4-byte magic + 2-byte e_type at offset 16 (little
        # endian, so ET_REL reads back as the hex pair "0100").
        file(READ "${_obj}" _elf LIMIT 18 HEX)
        string(LENGTH "${_elf}" _len)
        if(_len LESS 36 OR NOT _elf MATCHES "^7f454c46")
            # Truncated or not ELF at all: rebuild it.
            file(REMOVE "${_obj}")
            math(EXPR _poisoned "${_poisoned} + 1")
            continue()
        endif()
        string(SUBSTRING "${_elf}" 32 4 _etype)
        if(NOT _etype STREQUAL "0100")
            file(REMOVE "${_obj}")
            math(EXPR _poisoned "${_poisoned} + 1")
        endif()
    endforeach()
endforeach()

if(_poisoned GREATER 0)
    message(STATUS "preclean: removed ${_poisoned} device object(s) poisoned by the in-place obj-text merge")
endif()
