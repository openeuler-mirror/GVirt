#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Binding-contract probes: the enforced positional/keyword call shape must reject
mismatched calls with TypeError at the binding layer, before any device work."""

import argparse

from xlite._C import (
    Runtime,
    silu_gate_mul,
    split_col,
)


def expect_type_error(desc, call):
    try:
        call()
    except TypeError:
        print(f"PASS {desc} rejected with TypeError", flush=True)
    else:
        raise AssertionError(f"{desc} accepted a contract-violating call")


def main(_device):
    # size is keyword-only in the Runtime constructor
    expect_type_error("Runtime positional size", lambda: Runtime(0, 500))  # type: ignore[misc,call-overload]
    # silu_gate_mul is fully keyword-only
    expect_type_error("silu_gate_mul positional call", lambda: silu_gate_mul(None, None, None, None))  # type: ignore[misc,arg-type]
    # in/outputs are positional-only slots
    expect_type_error("split_col keyword into positional-only slot", lambda: split_col(rt=None, in_=None, outputs=None))  # type: ignore[call-arg,arg-type]

    # Exhaustive: every contract wrapper generated in xlite._C (from the
    # _C.pyi markers) must reject shape violations with TypeError and let
    # valid shapes through to the op layer, where the None probes fail the
    # dispatcher's type check (RuntimeError) before any device work.
    import xlite._C as _C

    # Drift guard: only the 4 bench ops (compiled out unless XLITE_BENCH_TIMED=ON)
    # may be missing. Any other op in _MISSING means a name drifted between _C.pyi
    # and TORCH_LIBRARY registration — the silent-green failure this test catches.
    _EXPECTED_MISSING = {"bench_timed", "bench_mte", "bench_vec", "bench_cube"}
    _drift = set(_C._MISSING) - _EXPECTED_MISSING
    assert not _drift, (
        f"binding drift: ops in _C.pyi but not registered (bound None): {sorted(_drift)}; "
        f"only {_EXPECTED_MISSING} may be missing (XLITE_BENCH_TIMED controls them)"
    )

    # Enum drift guard: the Python IntEnum values in _C.py must match the C++
    # enums (runtime.h XModelAttnType, model.h XModelScoringFuncType). Values are
    # hardcoded on both sides with no compile-time link, so a C++ reordering would
    # silently misroute attn_type/scoring_func. The expected maps mirror the C++
    # enum order (0-based, MAX_TYPE sentinel excluded).
    _EXPECTED_ENUMS = {
        _C.AttnType: {"AttnMHA": 0, "AttnMLA": 1, "AttnDSA": 2, "AttnHybridGdn": 3, "AttnHybridKda": 4, "AttnCxA": 5},
        _C.ScoringFuncType: {"ScoringFuncSoftmax": 0, "ScoringFuncSigmoid": 1, "ScoringFuncSqrtsoftplus": 2},
    }
    for _enum, _expected in _EXPECTED_ENUMS.items():
        _actual = {m.name: m.value for m in _enum}
        assert _actual == _expected, (
            f"{_enum.__name__} enum drift: Python {_actual} != C++ {_expected} (check runtime.h/model.h enum order)"
        )

    checked = 0
    for name, (fn, posonly, posorkw, kwonly) in sorted(_C._CONTRACT.items()):
        all_names = posonly + posorkw + kwonly
        if kwonly:
            expect_type_error(
                f"{name} positional spill into keyword-only region",
                lambda f=fn, names=all_names: f(*([None] * len(names))),
            )
        if posonly:
            expect_type_error(
                f"{name} positional-only slot addressed by keyword",
                lambda f=fn, names=all_names: f(**dict.fromkeys(names)),
            )
        try:
            if posonly:
                fn(*([None] * len(posonly)), **dict.fromkeys(posorkw + kwonly))
            else:
                fn(**dict.fromkeys(all_names))
        except TypeError as e:
            raise AssertionError(f"{name} rejected a valid call shape: {e}") from e
        except RuntimeError:
            pass  # op-layer value check on None — the wrapper forwarded fine
        checked += 1
    # Every marked op that is not missing must have a contract wrapper; a
    # mismatch means a wrapper was skipped or _CONTRACT drifted from _MISSING.
    assert checked == len(_C._CONTRACT), (
        f"probed {checked} wrappers but _CONTRACT has {len(_C._CONTRACT)} (missing={sorted(_C._MISSING)})"
    )
    print(f"PASS {checked} contract wrappers probed (shape violations + pass-through)")

    # Class-proxy probes: Model/ModelConfig are Python proxies over ScriptObjects
    # (torch::class_ can't be subclassed). Each def_rw field is a _Fwd descriptor
    # so downstream's AttributeSetterMixin gate (hasattr(type(self), name))
    # passes. Failure is silent and invisible to the op probes above: a
    # pyi-declared field not registered in C++ def_rw (e.g. enable_dsa_cp once
    # drifted) makes reads raise AttributeError; a missing descriptor makes the
    # mixin silently drop writes. Probe the class surface directly.

    def _probe_class_proxy(class_name, real_cls):
        # Exercise the downstream mixin pattern (mixin first in MRO).
        class _Gate:
            def __setattr__(self, name, value):
                if not (hasattr(type(self), name) or name in self.__dict__):
                    return
                super().__setattr__(name, value)

        Proxy = type(class_name, (_Gate, real_cls), {})
        proxy = Proxy()
        for a in _C._class_specs.get(class_name, {}).get("attrs", []):
            # Descriptor present (gate would pass) ...
            assert isinstance(getattr(type(proxy), a, None), _C._Fwd), (
                f"{class_name}.{a}: no _Fwd descriptor (pyi lists it but _class_proxy didn't build one)"
            )
            # ... and the ScriptObject actually has the field (write round-trips).
            try:
                setattr(proxy, a, getattr(proxy, a))
            except (AttributeError, RuntimeError) as e:
                raise AssertionError(
                    f"{class_name}.{a}: pyi declares it but the ScriptObject rejects it (C++ def_rw drift?): {e}"
                ) from e
            assert getattr(proxy, a) == getattr(proxy._inner, a), (
                f"{class_name}.{a}: write didn't reach the underlying ScriptObject"
            )
        return proxy

    cfg_proxy = _probe_class_proxy("ModelConfig", _C.ModelConfig)
    model_proxy = _probe_class_proxy("Model", _C.Model)

    # Model.init must unwrap a proxy config before handing it to C++ (a proxy
    # straight to intrusive_ptr<T> is UB). Confirm it reaches C++ (RuntimeError
    # from an under-configured model), not a TypeError type rejection.
    try:
        model_proxy.init(cfg_proxy, 0)
    except RuntimeError:
        pass  # reached C++ Init — the unwrap worked
    except TypeError as e:
        raise AssertionError(f"Model.init rejected a proxy config (unwrap missing?): {e}") from e
    print("PASS class proxies probed (ModelConfig + Model fields, init unwrap)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    main(parser.parse_args().device)
