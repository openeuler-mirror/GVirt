#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

from __future__ import absolute_import
import os
import subprocess
import sys
import re
import shutil
from typing import List
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

def find_version(filepath: str):
    """Extract version information from the given filepath."""
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            version = version_match.group(1)
            return version
        raise RuntimeError("Unable to find version string.")

class CleanCommand(Command):
    description = "Remove build artifacts and egg-info directory"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for dirname in ("build", "cmake_build", "xlite.egg-info"):
            dirname = os.path.abspath(dirname)
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
                print(f"Removed {dirname}")

functions_module=Extension(
    name='xlite',
    sources=[],
)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        install_prefix = self.build_lib
        python_command = "python3 -m pip show torch | grep '^Location:' | awk '{print $2}'"
        try:
            pybind11_cmake_path = (subprocess.check_output(
                                   [sys.executable, "-m", "pybind11",
                                    "--cmakedir"]).decode().strip())
            abs_python_path = subprocess.check_output(
                              python_command, shell=True).decode().strip()
            torch_cmake_path = os.path.join(abs_python_path, "torch", "share", "cmake", "Torch")
            cmake_prefix_paths = ";".join([pybind11_cmake_path, torch_cmake_path])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake configuration failed: {e}")
        cmake_args = [f"-DCMAKE_PREFIX_PATH={cmake_prefix_paths}"]

        subprocess.check_call(['rm', '-rf', 'cmake_build'])
        if not os.path.exists('cmake_build'):
            os.makedirs('cmake_build')
        subprocess.check_call(['cmake',
                               '-B',
                               'cmake_build',
                               f"-DCMAKE_ABS_PYTHON_PATH={abs_python_path}",
                               f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
                               *cmake_args])
        subprocess.check_call(['cmake', '--build', 'cmake_build', '-j'])
        subprocess.check_call(['cmake', '--install', 'cmake_build'])

setup(
    name='xlite',
    version=find_version(get_path("xlite", "__init__.py")),
    author='Huawei Euler OS Team',
    license='MulanPSL2',
    description='A lightweight, effective and easy-to-extend inference runtime',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://gitee.com/openeuler/GVirt',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=get_requirements(),
    packages=find_packages(exclude=["*.tools", "*.tools.*"]),
    ext_modules=[functions_module],
    python_requires='>=3.9',
    cmdclass=dict(build_ext=CMakeBuild, clean=CleanCommand),
    package_data={"xlite": ['*.so', 'lib/*.so']},
)