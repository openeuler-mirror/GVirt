"""xlite Python package.

This package has two integration surfaces:
- Python: native extension module :mod:`xlite._C`, with runtime, model, and
kernel bindings implemented in ``csrc/_C.cpp``.
- C++: link libxlite.so and include public headers, ``cmake_prefix_path``
exposes the cmake package config for find_package(xlite).

Supported Python versions: 3.9 through 3.12.
"""

import os.path


def _resolve_version() -> str:
    """Best-effort version that works from a git checkout, an sdist build,
    and an installed wheel. Tries the file the setuptools_scm plugin writes
    (xlite/_version.py) first, then the installed distribution metadata, then
    a dev sentinel."""
    try:
        from ._version import __version__ as _v

        if _v:
            return _v
    except ImportError:
        pass
    try:
        from importlib.metadata import version, PackageNotFoundError

        return version("xlite")
    except (ImportError, PackageNotFoundError):
        return "0.0.0"


__version__: str = _resolve_version()
__version_tuple__: tuple[int | str, ...] = tuple(
    int(p) if p.isdigit() else p for p in __version__.replace("-", ".").split(".") if p
)

# CMake package config dir for find_package(xlite)
cmake_prefix_path = os.path.join(os.path.dirname(__file__), "lib", "cmake")
