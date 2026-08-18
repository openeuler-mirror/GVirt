"""xlite Python package.

This package has two integration surfaces:
- Python: native extension module :mod:`xlite._C`, with runtime, model, and
kernel bindings implemented in ``csrc/_C.cpp``.
- C++: link libxlite.so and include public headers, ``cmake_prefix_path``
exposes the cmake package config for find_package(xlite).

Supported Python versions: 3.9 through 3.12.
"""

import os.path

__version__ = "0.2.0rc0"

# CMake package config dir for find_package(xlite)
cmake_prefix_path = os.path.join(os.path.dirname(__file__), "lib", "cmake")
