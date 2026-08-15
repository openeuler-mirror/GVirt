# Imported target xlite (libxlite.so + headers)

# Python interpreter for locating xlite
find_program(_XLITE_PYTHON_EXECUTABLE NAMES python3 python)

# --- 1. Locate xlite -------------------------------------------------------
# Installed at <pkgroot>/xlite/lib/cmake/xlite/xliteConfig.cmake.
get_filename_component(_XLITE_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_XLITE_CMAKE_DIR  "${_XLITE_CONFIG_DIR}" PATH)
get_filename_component(_XLITE_LIB_DIR    "${_XLITE_CMAKE_DIR}" PATH)
get_filename_component(_XLITE_PKG_ROOT   "${_XLITE_LIB_DIR}" PATH)

set(xlite_INCLUDE_DIR "${_XLITE_PKG_ROOT}/include")
set(xlite_LIBRARY     "${_XLITE_LIB_DIR}/libxlite.so")

# Fall back to Python import if the install-layout path doesn't exist.
if(NOT EXISTS "${xlite_LIBRARY}" AND _XLITE_PYTHON_EXECUTABLE)
    execute_process(
        COMMAND ${_XLITE_PYTHON_EXECUTABLE} -c "import xlite,os;print(os.path.dirname(xlite.__file__))"
        OUTPUT_VARIABLE _XLITE_PKG_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _XLITE_IMPORT_RESULT)
    if(_XLITE_IMPORT_RESULT EQUAL 0 AND _XLITE_PKG_ROOT)
        set(xlite_INCLUDE_DIR "${_XLITE_PKG_ROOT}/include")
        set(xlite_LIBRARY     "${_XLITE_PKG_ROOT}/lib/libxlite.so")
        set(_XLITE_LIB_DIR    "${_XLITE_PKG_ROOT}/lib")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(xlite DEFAULT_MSG xlite_LIBRARY xlite_INCLUDE_DIR)

if(NOT TARGET xlite::xlite)
    add_library(xlite::xlite SHARED IMPORTED)
    set_target_properties(xlite::xlite PROPERTIES
        IMPORTED_LOCATION "${xlite_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${xlite_INCLUDE_DIR}")
endif()

# --- 2. Propagate -rpath-link for torch (linked PRIVATE by libxlite.so) ----
set(_XLITE_LINK_DIRS  "${_XLITE_LIB_DIR}")
set(_XLITE_RPATH_LINK "-Wl,-rpath-link,${_XLITE_LIB_DIR}")

if(_XLITE_PYTHON_EXECUTABLE)
    execute_process(
        COMMAND ${_XLITE_PYTHON_EXECUTABLE} -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"
        OUTPUT_VARIABLE _XLITE_TORCH_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _XLITE_TORCH_RESULT)
    if(_XLITE_TORCH_RESULT EQUAL 0 AND EXISTS "${_XLITE_TORCH_LIB_DIR}")
        list(APPEND _XLITE_LINK_DIRS "${_XLITE_TORCH_LIB_DIR}")
        string(APPEND _XLITE_RPATH_LINK ";-Wl,-rpath-link,${_XLITE_TORCH_LIB_DIR}")
    endif()
endif()

set_target_properties(xlite::xlite PROPERTIES
    INTERFACE_LINK_DIRECTORIES "${_XLITE_LINK_DIRS}"
    INTERFACE_LINK_OPTIONS "${_XLITE_RPATH_LINK}")

unset(_XLITE_PYTHON_EXECUTABLE)
unset(_XLITE_CONFIG_DIR)
unset(_XLITE_CMAKE_DIR)
unset(_XLITE_LIB_DIR)
unset(_XLITE_PKG_ROOT)
unset(_XLITE_LINK_DIRS)
unset(_XLITE_RPATH_LINK)
unset(_XLITE_IMPORT_RESULT)
unset(_XLITE_TORCH_LIB_DIR)
unset(_XLITE_TORCH_RESULT)
