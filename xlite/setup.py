#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext


ROOT_DIR = Path(__file__).resolve().parent


class CleanCommand(Command):
    description = "Remove local build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for dirname in ("build", "cmake_build", "xlite.egg-info"):
            path = ROOT_DIR / dirname
            if path.exists():
                shutil.rmtree(path)
                print(f"Removed {path}")


def _get_torch_cmake_dir() -> str:
    import torch

    torch_site_path = (
        Path(torch.__file__).resolve().parent / "share" / "cmake" / "Torch"
    )
    if not torch_site_path.exists():
        raise RuntimeError(
            f"Torch CMake package was not found at {torch_site_path}. "
            "Install torch in the active environment first."
        )
    return str(torch_site_path)


def _get_pybind11_cmake_dir() -> str:
    return subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"], text=True
    ).strip()


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        build_temp = Path(self.build_temp or (ROOT_DIR / "cmake_build")).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        is_editable = self.inplace or any(x in sys.argv for x in ("develop", "editable_wheel"))
        install_prefix = Path(self.build_lib).resolve()
        print(f"CMakeBuild: Building in {'editable' if is_editable else 'standard'} mode; installing to "
              f"{install_prefix}")
        cmake_prefix_paths = ";".join(
            [_get_pybind11_cmake_dir(), _get_torch_cmake_dir()]
        )

        configure_cmd = [
            "cmake",
            "-S",
            str(ROOT_DIR),
            "-B",
            str(build_temp),
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_paths}",
            f"-DXLITE_EDITABLE_BUILD={'ON' if is_editable else 'OFF'}",
        ]
        build_cmd = ["cmake", "--build", str(build_temp), "-j"]
        install_cmd = ["cmake", "--install", str(build_temp)]

        subprocess.check_call(configure_cmd)
        subprocess.check_call(build_cmd)
        subprocess.check_call(install_cmd)

        # Remove headers and lib64 staging outputs from wheel contents.
        for artifact_dir_sub in ("include", "lib", "lib64", "csrc"):
            artifact_dir = install_prefix / artifact_dir_sub
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)

        # Manually sync auxiliary libraries for editable installs
        if is_editable:
            cmake_lib_dir = install_prefix / "xlite" / "lib"
            source_tree_lib_dir = ROOT_DIR / "xlite" / "lib"

            if not cmake_lib_dir.exists():
                raise RuntimeError(f"Expected CMake install to produce {cmake_lib_dir}, but it was not found.")

            if cmake_lib_dir.resolve() == source_tree_lib_dir.resolve():
                return

            # move the .so files to the source tree for editable installs
            source_tree_lib_dir.mkdir(parents=True, exist_ok=True)
            for so_file in cmake_lib_dir.glob("*.so"):
                shutil.copy2(so_file, source_tree_lib_dir / so_file.name)
                print(f"Editable install detected: Copied {so_file} to {source_tree_lib_dir}")


setup(
    ext_modules=[Extension(name="xlite._C", sources=[])],
    cmdclass={"build_ext": CMakeBuild, "clean": CleanCommand},
)
