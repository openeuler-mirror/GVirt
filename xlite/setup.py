#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext

try:
    # setuptools >=70.1 ships bdist_wheel; fall back to the wheel package on
    # older setuptools, and finally to None (no retag) if neither is present.
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        _bdist_wheel = None


ROOT_DIR = Path(__file__).resolve().parent


def _read_version() -> str:
    """Read the version the setuptools_scm plugin wrote to xlite/_version.py.

    The plugin (configured in pyproject.toml) owns inference and also ships
    scm_version.json in the sdist, so the version survives a no-git rebuild.
    Calling get_version() here directly would bypass that recovery and fall
    back to 0.0.0 when .git is absent. Read lazily so CMakeBuild sees the file
    the plugin writes during the build; on first configure fall back to
    SETUPTOOLS_SCM_PRETEND_VERSION or 0.0.0 (CMake's own default applies).
    """
    version_file = ROOT_DIR / "xlite" / "_version.py"
    if version_file.exists():
        namespace: dict[str, str] = {}
        exec(version_file.read_text(encoding="utf-8"), namespace)
        if namespace.get("__version__"):
            return namespace["__version__"]
    return os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")


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

    torch_site_path = Path(torch.__file__).resolve().parent / "share" / "cmake" / "Torch"
    if not torch_site_path.exists():
        raise RuntimeError(
            f"Torch CMake package was not found at {torch_site_path}. Install torch in the active environment first."
        )
    return str(torch_site_path)


def _get_pybind11_cmake_dir() -> str:
    return subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()



if _bdist_wheel is not None:

    class _ManylinuxTagBdistWheel(_bdist_wheel):
        """Retag built wheel `linux_<arch>` -> `manylinux2014_<arch>` (filename
        + `.dist-info/WHEEL` Tag) for PyPI upload. Arch-agnostic."""

        @staticmethod
        def _retag_wheel(whl: Path) -> Path:
            name = whl.name
            if "-linux_" not in name:
                return whl
            target = whl.with_name(name.replace("-linux_", "-manylinux2014_"))
            with zipfile.ZipFile(whl, "r") as zin:
                infos = zin.infolist()
                entries = {it.filename: zin.read(it.filename) for it in infos}

            # Rewrite WHEEL's Tag, then refresh its sha256/size in RECORD.
            # Two-pass read-then-write so WHEEL/RECORD order doesn't matter.
            wheel_path = next((p for p in entries if p.endswith(".dist-info/WHEEL")), None)
            record_path = next((p for p in entries if p.endswith(".dist-info/RECORD")), None)
            if wheel_path is not None:
                entries[wheel_path] = entries[wheel_path].replace(b"linux_", b"manylinux2014_")
                if record_path is not None:
                    new_hash = hashlib.sha256(entries[wheel_path]).hexdigest()
                    new_size = len(entries[wheel_path])
                    wheel_basename = wheel_path.split("/")[-1]
                    lines = entries[record_path].decode("utf-8").splitlines(keepends=True)
                    for i, line in enumerate(lines):
                        if line.split(",", 1)[0].split("/")[-1] == wheel_basename:
                            lines[i] = f"{wheel_path},sha256={new_hash},{new_size}\n"
                            break
                    entries[record_path] = "".join(lines).encode("utf-8")

            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
                for it in infos:
                    zout.writestr(it, entries[it.filename])
            whl.unlink()
            return target

        def run(self):
            super().run()
            dist_dir = Path(self.dist_dir)
            for whl in dist_dir.glob("*.whl"):
                if "-linux_" not in whl.name:
                    continue
                target = self._retag_wheel(whl)
                print(f"bdist_wheel: retagged {whl.name} -> {target.name}")
else:  # pragma: no cover
    _ManylinuxTagBdistWheel = None


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        build_temp = Path(self.build_temp or (ROOT_DIR / "cmake_build")).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        is_editable = self.inplace or any(x in sys.argv for x in ("develop", "editable_wheel"))
        install_prefix = Path(self.build_lib).resolve()
        print(
            f"CMakeBuild: Building in {'editable' if is_editable else 'standard'} mode; installing to {install_prefix}"
        )
        cmake_prefix_paths = ";".join([_get_pybind11_cmake_dir(), _get_torch_cmake_dir()])

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
        configure_cmd.append(f"-DXLITE_VERSION={_read_version()}")

        # XLITE_BENCH_TIMED env var -> cmake -DXLITE_BENCH_TIMED=ON, enabling the
        # timed micro-benchmark kernels (bench_timed/bench_mte/bench_vec/bench_cube).
        # The CMake option defaults to OFF; only flip it when the env var is set.
        if os.environ.get("XLITE_BENCH_TIMED", "").upper() in ("ON", "1", "TRUE", "YES"):
            configure_cmd.append("-DXLITE_BENCH_TIMED=ON")

        build_cmd = ["cmake", "--build", str(build_temp), "-j"]
        install_cmd = ["cmake", "--install", str(build_temp)]

        # Wheel (non-editable) builds must never ship debug code. Debug is driven purely by the XLITE_DEBUG_ON env var.
        # CMakeLists.txt parses it into macros. Editable builds can set XLITE_DEBUG_ON=forward,tuner...
        cmake_env = None
        if not is_editable and not os.environ.get("XLITE_DEBUG_ON", "").lower().startswith("force"):
            cmake_env = os.environ.copy()
            cmake_env.pop("XLITE_DEBUG_ON", None)
            print("CMakeBuild: stripped XLITE_DEBUG_ON from env for non-editable build")

        # Overflow-status buffer caching is applied by
        # scripts/patch_host_stub.py inside the CMake build (always on), so
        # the CANN installation is no longer touched from here.
        subprocess.check_call(configure_cmd, env=cmake_env)
        subprocess.check_call(build_cmd, env=cmake_env)
        subprocess.check_call(install_cmd, env=cmake_env)

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

            # move headers and cmake config to source tree for editable installs
            cmake_include_dir = install_prefix / "xlite" / "include"
            if cmake_include_dir.exists():
                source_tree_include_dir = ROOT_DIR / "xlite" / "include"
                if source_tree_include_dir.exists():
                    shutil.rmtree(source_tree_include_dir)
                shutil.copytree(cmake_include_dir, source_tree_include_dir)
                print(f"Editable install detected: Synced headers to {source_tree_include_dir}")

            cmake_config_dir = cmake_lib_dir / "cmake"
            if cmake_config_dir.exists():
                source_tree_config_dir = source_tree_lib_dir / "cmake"
                if source_tree_config_dir.exists():
                    shutil.rmtree(source_tree_config_dir)
                shutil.copytree(cmake_config_dir, source_tree_config_dir)
                print(f"Editable install detected: Synced cmake config to {source_tree_config_dir}")


setup(
    # version is dynamic in pyproject.toml; not passed here so the
    # setuptools_scm plugin infers it (and recovers from scm_version.json).
    ext_modules=[Extension(name="xlite._C", sources=[])],
    cmdclass={
        "build_ext": CMakeBuild,
        "clean": CleanCommand,
        "bdist_wheel": _ManylinuxTagBdistWheel,
    },
)
