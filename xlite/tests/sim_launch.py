"""Run a test script on the simulator without the crashing interpreter teardown.

`msprof op simulator` runs the test in a child process. On the CANN stub driver
the torch-npu/camodel interpreter finalization segfaults *after* the test has
finished ("Child process killed by signal 11"), so msprof reports a failed run
even though every kernel ran and the profile was written.

This launcher runs the script via ``runpy`` (so ``__main__`` semantics and
``sys.argv`` are preserved), then ``os._exit()``s from an ``atexit`` hook
registered last (thus executed first). Finalization — including the C++
destructors that crash — is skipped, while the script's exit status is kept.

Used by ``tests/run_sim.sh run``:

    python -u tests/sim_launch.py <test.py> [args...]
"""

from __future__ import annotations

import atexit
import os
import runpy
import sys
import traceback


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: sim_launch.py <test.py> [args...]", file=sys.stderr)
        return 2
    script, *args = sys.argv[1:]
    sys.argv = [script, *args]
    code = 1
    try:
        runpy.run_path(script, run_name="__main__")
        code = 0
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code
        if not isinstance(code, int):
            print(code, file=sys.stderr)
            code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        atexit.register(os._exit, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
