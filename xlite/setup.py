#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

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


if _bdist_wheel is not None:

    class _ManylinuxTagBdistWheel(_bdist_wheel):
        """Retag built wheels: platform `linux_<arch>` -> `manylinux2014_<arch>`
        for PyPI upload, and rewrite the CPython/ABI tag to `cp311-abi3` so a
        single wheel installs on any CPython 3.11+.

        The abi3 retag is content-driven: if the wheel ships an `.abi3.so`
        extension (built via `Python3_add_library(... USE_SABI ...)`), the tag
        is rewritten; detection is automatic so the daily bot's
        `bdist_wheel` needs no flag.
        """

        # The abi3 floor. A module built with `USE_SABI 3.11` (Py_LIMITED_API
        # 0x030B0000) loads on any CPython >= 3.11. Bump here only if the
        # CMake USE_SABI version is raised.
        _ABI3_PYTAG = "cp311"
        _ABI3_ABITAG = "abi3"

        @staticmethod
        def _wheel_has_abi3_so(whl: Path) -> bool:
            """True iff the wheel ships a `.abi3.so` extension module."""
            with zipfile.ZipFile(whl, "r") as zin:
                return any(name.endswith(".abi3.so") or ".abi3.so." in name for name in zin.namelist())

        @staticmethod
        def _retag_wheel(whl: Path) -> Path:
            name = whl.name
            # Filename stem rewrite:
            #   <pkg>-<ver>-<pytag>-<abitag>-linux_<arch>.whl
            # -> <pkg>-<ver>-cp311-abi3-manylinux2014_<arch>.whl
            parts = name[: -len(".whl")].split("-")
            # expected: [pkg, ver, pytag, abitag, plat]
            if len(parts) >= 5:
                parts[2] = _ManylinuxTagBdistWheel._ABI3_PYTAG
                parts[3] = _ManylinuxTagBdistWheel._ABI3_ABITAG
            stem = "-".join(parts)
            stem = stem.replace("-linux_", "-manylinux2014_")
            target = whl.with_name(stem + ".whl")

            with zipfile.ZipFile(whl, "r") as zin:
                infos = zin.infolist()
                entries = {it.filename: zin.read(it.filename) for it in infos}

            # Rewrite WHEEL's Tag, then refresh its sha256/size in RECORD.
            # Two-pass read-then-write so WHEEL/RECORD order doesn't matter.
            wheel_path = next((p for p in entries if p.endswith(".dist-info/WHEEL")), None)
            record_path = next((p for p in entries if p.endswith(".dist-info/RECORD")), None)
            if wheel_path is not None:
                tag_bytes = entries[wheel_path]
                # WHEEL's Tag line is "<pytag>-<abitag>-<plat>". Replace the
                # leading <pytag>-<abitag> with the abi3 tag and linux_ -> manylinux2014_.
                lines = tag_bytes.decode("utf-8").splitlines(keepends=True)
                for i, line in enumerate(lines):
                    if line.startswith("Tag:"):
                        # "Tag: cp312-cp312-linux_aarch64" -> "Tag: cp311-abi3-manylinux2014_aarch64"
                        rest = line[len("Tag:") :].lstrip()
                        plat = rest.split("-", 2)[-1] if rest.count("-") >= 2 else rest
                        plat = plat.replace("linux_", "manylinux2014_")
                        lines[i] = (
                            f"Tag: {_ManylinuxTagBdistWheel._ABI3_PYTAG}-{_ManylinuxTagBdistWheel._ABI3_ABITAG}-{plat}"
                        )
                        if not lines[i].endswith("\n"):
                            lines[i] += "\n"
                        break
                tag_bytes = "".join(lines).encode("utf-8")
                entries[wheel_path] = tag_bytes
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
            # An already-retagged wheel retags to itself; don't unlink the
            # freshly written file in that case.
            if target != whl:
                whl.unlink()
            return target

        def run(self):
            super().run()
            dist_dir = Path(self.dist_dir)
            for whl in dist_dir.glob("*.whl"):
                # Only retag this run's freshly built wheels (still tagged
                # -linux_<arch>); skip already-retagged and unrelated wheels.
                if "-linux_" not in whl.name:
                    continue
                if not self._wheel_has_abi3_so(whl):
                    raise SystemExit(
                        f"bdist_wheel: {whl.name} has no .abi3.so — the SABI build did not "
                        "produce a stable-ABI module; check that CMake built xlite_sabi.abi3.so."
                    )
                target = self._retag_wheel(whl)
                print(f"bdist_wheel: retagged {whl.name} -> {target.name}")
else:  # pragma: no cover
    _ManylinuxTagBdistWheel = None


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        is_editable = self.inplace or any(x in sys.argv for x in ("develop", "editable_wheel"))
        # Editable builds may reuse a stable cmake tree across runs (XLITE_BUILD_TEMP,
        # e.g. ~/.cache/xlite/build) so pip install -e . stays incremental; wheel
        # builds always get a fresh tree.
        reuse_tree = os.environ.get("XLITE_BUILD_TEMP") if is_editable else None
        build_temp = Path(
            os.path.expanduser(reuse_tree) if reuse_tree else (self.build_temp or (ROOT_DIR / "cmake_build"))
        ).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        install_prefix = Path(self.build_lib).resolve()
        print(
            f"CMakeBuild: Building in {'editable' if is_editable else 'standard'} mode; installing to {install_prefix}"
        )
        cmake_prefix_paths = _get_torch_cmake_dir()

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

        # Numbered -j (not bare) so sub-makes get a jobserver instead of unbounded MAKEFLAGS,
        # which they run verbatim and fail with EMFILE on shared hosts.
        max_jobs = os.environ.get("MAX_JOBS") or os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        build_cmd = ["cmake", "--build", str(build_temp), "-j", max_jobs or str(min(os.cpu_count() or 4, 128))]
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
    # Stub extension that only triggers the CMake build; the name must match the
    # artifact CMake installs (xlite/xlite_sabi.abi3.so) so setuptools' inplace
    # copy step finds it. py_limited_api makes the expected suffix .abi3.so.
    ext_modules=[Extension(name="xlite.xlite_sabi", sources=[], py_limited_api=True)],
    cmdclass={
        "build_ext": CMakeBuild,
        "clean": CleanCommand,
        "bdist_wheel": _ManylinuxTagBdistWheel,
    },
)
