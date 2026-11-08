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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    main(parser.parse_args().device)
