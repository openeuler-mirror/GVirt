#!/usr/bin/env bash
# Run xlite kernels on the CANN simulator (no NPU).
#
# Usage:
#   tests/run_sim.sh setup              # create simulator shim + venv with deps
#   tests/run_sim.sh build              # build xlite and stage artifacts
#   tests/run_sim.sh run [test.py ...]  # run a test under `msprof op simulator`
#   tests/run_sim.sh all                # setup + build + run tests/sim_smoke.py
#
# Environment overrides:
#   XLITE_SIM_SOC        target SoC (default Ascend910B3)
#   XLITE_SIM_WORK       work dir for shim/venv/build/out (default ~/.cache/xlite-sim)
#   XLITE_SIM_VENV       venv dir (default $XLITE_SIM_WORK/venv)
#   XLITE_SIM_PYTHON     interpreter for uv (default 3.11)
#   XLITE_SIM_TORCH      torch version (default 2.9.1; xlite requires <=2.10.0)
#   XLITE_SIM_TORCH_NPU  torch-npu version (default 2.9.1)
#   XLITE_SIM_OUT        msprof output dir (default $XLITE_SIM_WORK/out)
#   XLITE_SIM_BUILD      CMake build dir (default $XLITE_SIM_WORK/build)
#   XLITE_SIM_JOBS       parallel build jobs (default: nproc)
#   XLITE_SIM_CLEAN=1    force a clean build (default: clean build anyway)
#   XLITE_SIM_INCREMENTAL=1  reuse $XLITE_SIM_BUILD instead of wiping it
#   CANN_SET_ENV         path to set_env.sh
#   CC / CXX             host compilers (default clang/clang++)
#
# Why the shim: on CANN 9.2.0-beta.1 `msprof op simulator` injects the hardware
# runtime and then TSD/HDC fails against the stub driver (aclError 507033). The
# shim makes the injection resolve `libacl_rt_impl.so` -> `libruntime_camodel.so`
# next to it, so camodel keeps using the `no TsdOpen` path.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOC=${XLITE_SIM_SOC:-Ascend910B3}
WORK=${XLITE_SIM_WORK:-$HOME/.cache/xlite-sim}
VENV=${XLITE_SIM_VENV:-$WORK/venv}
PYTHON_BIN=${XLITE_SIM_PYTHON:-3.11}
TORCH_VERSION=${XLITE_SIM_TORCH:-2.9.1}
TORCH_NPU_VERSION=${XLITE_SIM_TORCH_NPU:-2.9.1}
OUT=${XLITE_SIM_OUT:-$WORK/out}
BUILD_DIR=${XLITE_SIM_BUILD:-$WORK/build}
JOBS=${XLITE_SIM_JOBS:-$(nproc)}
CANN_SET_ENV=${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
CC=${CC:-clang}
CXX=${CXX:-clang++}

die() { echo "error: $*" >&2; exit 1; }

source_cann() {
    [[ -f "$CANN_SET_ENV" ]] || die "CANN set_env.sh not found: $CANN_SET_ENV (set CANN_SET_ENV)"
    # shellcheck disable=SC1090
    set +u
    source "$CANN_SET_ENV"
    set -u
    CANN_HOME=${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}
    [[ -n "$CANN_HOME" ]] || die "ASCEND_HOME_PATH is not set after sourcing $CANN_SET_ENV"
}

shim_dir() { echo "$WORK/shim-$SOC"; }

cmd_setup() {
    source_cann
    local simlib="$CANN_HOME/tools/simulator/$SOC/lib"
    [[ -f "$simlib/libruntime_camodel.so" ]] ||
        die "no simulator for $SOC: $simlib/libruntime_camodel.so"
    local acl_impl
    acl_impl=$(ls "$CANN_HOME"/*-linux/lib64/libacl_rt_impl.so 2>/dev/null | head -n1 || true)
    [[ -n "$acl_impl" ]] || die "libacl_rt_impl.so not found under $CANN_HOME/*-linux/lib64"

    local shim; shim=$(shim_dir)
    mkdir -p "$shim" "$WORK"
    cp -f "$acl_impl" "$shim/"
    ln -sfn "$simlib/libruntime_camodel.so" "$shim/libruntime.so"
    echo "shim:  $shim"
    echo "       libruntime.so -> $simlib/libruntime_camodel.so"

    if [[ -x "$VENV/bin/python" ]]; then
        echo "venv:  $VENV (exists, skipping install)"
        return
    fi
    command -v uv >/dev/null || die "uv is required (https://docs.astral.sh/uv/)"
    echo "venv:  $VENV (creating)"
    uv venv "$VENV" --python "$PYTHON_BIN"
    uv pip install --python "$VENV/bin/python" "torch==$TORCH_VERSION" \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python "$VENV/bin/python" \
        "setuptools>=64,<82" "setuptools-scm>=8" "cmake>=3.22" \
        "numpy<2" pyyaml "torch-npu==$TORCH_NPU_VERSION"
}

cmd_build() {
    source_cann
    [[ -x "$VENV/bin/python" ]] || die "run 'setup' first (missing $VENV)"
    local py="$VENV/bin/python"
    local torch_cmake torch_lib
    torch_cmake=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$py" -c \
        'import torch,pathlib;print(pathlib.Path(torch.__file__).parent/"share"/"cmake"/"Torch")')
    torch_lib=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$py" -c \
        'import torch,pathlib;print(pathlib.Path(torch.__file__).parent/"lib")')

    command -v "$CC" >/dev/null || die "host compiler '$CC' not found (clang is needed on x86 for __fp16)"
    command -v "$CXX" >/dev/null || die "host compiler '$CXX' not found"

    # Build out of tree ($XLITE_SIM_BUILD, default $XLITE_SIM_WORK/build) so
    # the default clean build never wipes a regular NPU build in $REPO_ROOT/build.
    # CMake caches the imported Torch package dir (Torch_DIR) from the
    # first configure, and the AscendC kernel preprocess step is not reliable
    # on incremental rebuilds (`ld.lld: unknown file type`). Default to a clean
    # build; set XLITE_SIM_INCREMENTAL=1 to reuse an existing build dir.
    local sig stamp
    sig="$SOC|$CC|$CXX|$CANN_HOME|$VENV|$REPO_ROOT"
    stamp="$BUILD_DIR/.xlite-sim-config"
    if [[ -d "$BUILD_DIR" ]]; then
        if [[ "${XLITE_SIM_INCREMENTAL:-0}" != "1" ]] ||
            [[ "${XLITE_SIM_CLEAN:-0}" == "1" ]] ||
            [[ ! -f "$stamp" ]] ||
            [[ "$(cat "$stamp" 2>/dev/null)" != "$sig" ]]; then
            echo "build: removing $BUILD_DIR for a clean rebuild"
            rm -rf "$BUILD_DIR"
        fi
    fi

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DSOC_VERSION="$SOC" \
        -DRUN_MODE=npu \
        -DASCEND_CANN_PACKAGE_PATH="$CANN_HOME" \
        -DCMAKE_INSTALL_PREFIX="$REPO_ROOT/out" \
        -DXLITE_VERSION="${XLITE_VERSION:-0.0.0}" \
        -DCMAKE_PREFIX_PATH="$torch_cmake" \
        -DTorch_DIR="$torch_cmake" \
        -DPython3_EXECUTABLE="$py"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 cmake --build "$BUILD_DIR" -j "$JOBS"
    printf '%s\n' "$sig" > "$stamp"

    mkdir -p "$REPO_ROOT/xlite/lib"
    cp -f "$BUILD_DIR/libxlite.so" "$REPO_ROOT/xlite/lib/"
    cp -f "$BUILD_DIR"/lib/libxlite_kernels_*_npu.so "$REPO_ROOT/xlite/lib/"
    cp -f "$BUILD_DIR"/xlite_sabi*.so "$REPO_ROOT/xlite/"
    echo "build: artifacts staged in $REPO_ROOT/xlite"
    echo "       torch lib: $torch_lib"
}

cmd_run() {
    source_cann
    local test=${1:-tests/sim_smoke.py}
    shift || true
    # The wrapper cd's to the repo root, so relative paths resolve there.
    if [[ "$test" == /* ]]; then
        [[ -f "$test" ]] || die "test script not found: $test"
    else
        [[ -f "$REPO_ROOT/$test" ]] || die "test script not found: $test (under $REPO_ROOT)"$'\n'"\
subcommands are setup|build|run|all; to prepare the environment run '$0 setup'"
    fi
    [[ -x "$VENV/bin/python" ]] || die "run 'setup' first (missing $VENV)"
    local shim; shim=$(shim_dir)
    [[ -L "$shim/libruntime.so" ]] || die "run 'setup' first (missing $shim)"

    local torch_lib wrapper launcher
    torch_lib=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV/bin/python" -c \
        'import torch,pathlib;print(pathlib.Path(torch.__file__).parent/"lib")')
    mkdir -p "$OUT" "$WORK"
    launcher="$REPO_ROOT/tests/sim_launch.py"
    [[ -f "$launcher" ]] || die "missing $launcher"
    # The launcher skips interpreter finalization: torch-npu/camodel teardown
    # segfaults on the stub driver after the test has finished ("Child process
    # killed by signal 11"), which makes msprof report a failed run even though
    # every kernel ran and the profile was written. See tests/sim_launch.py.
    #
    # msprof does not forward the child's exit status: a failing test only
    # prints "[WARN] Child process exited with status 256" and msprof itself
    # exits 0. The wrapper records the status in $rc_file so `run` can return
    # it and CI sees test failures.
    local rc_file="$WORK/run_wrapper.rc"
    rm -f "$rc_file"
    wrapper="$WORK/run_wrapper.sh"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -uo pipefail'
        printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "$torch_lib"
        printf 'export PYTHONPATH=%q:${PYTHONPATH:-}\n' "$REPO_ROOT"
        printf 'cd %q\n' "$REPO_ROOT"
        printf '%q -u %q %q' "$VENV/bin/python" "$launcher" "$test"
        local a
        for a in "$@"; do printf ' %q' "$a"; done
        printf '\n'
        echo 'rc=$?'
        printf 'printf "%%s\\n" "$rc" > %q 2>/dev/null || true\n' "$rc_file"
        echo 'exit "$rc"'
    } > "$wrapper"
    chmod +x "$wrapper"

    echo "test:  $test"
    echo "soc:   $SOC"
    echo "out:   $OUT"
    # `source_cann` already set LD_LIBRARY_PATH; put the shim in front of CANN lib64.
    export LD_LIBRARY_PATH="$shim:${LD_LIBRARY_PATH:-}"
    set +e
    msprof op simulator --soc-version="$SOC" --output="$OUT" "$wrapper"
    local rc=$?
    set -e
    local latest child_rc=""
    latest=$(ls -dt "$OUT"/OPPROF_* 2>/dev/null | head -n1 || true)
    if [[ -f "$rc_file" ]]; then
        child_rc=$(<"$rc_file")
    fi
    if [[ -n "$latest" ]]; then
        echo "profiling output: $latest"
    else
        echo "warning: no OPPROF_* produced (msprof rc=$rc)" >&2
    fi
    # The test's own status wins: msprof reports a failed child as a warning.
    if [[ -n "$child_rc" && "$child_rc" != "0" ]]; then
        echo "error: test exited with status $child_rc (msprof rc=$rc)" >&2
        return "$child_rc"
    fi
    if [[ -z "$child_rc" ]]; then
        echo "warning: test did not report an exit status (msprof rc=$rc); check the output above" >&2
    fi
    return "$rc"
}

usage() {
    awk 'NR > 1 && /^set -euo/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
}

case ${1:-} in
    setup) cmd_setup ;;
    build) cmd_build ;;
    run) shift; cmd_run "$@" ;;
    all) cmd_setup; cmd_build; cmd_run ;;
    ""|-h|--help|help) usage ;;
    *) die "unknown command '$1' (use setup|build|run|all)" ;;
esac
