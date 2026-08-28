#!/usr/bin/env python3
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
"""Render ``xlite.spec`` from ``xlite.spec.in`` using the current version.

The version is sourced from ``setuptools_scm`` against the same monorepo root
that ``setup.py`` uses, so the wheel (``setup.py`` -> ``xlite/_version.py``)
and the RPM (this script -> ``xlite.spec`` ``Version:``) always read the
identical git-tag-derived string. This removes the need to hand-bump the spec
version in a ``[Version] chore: bump`` commit.

Usage::

    python scripts/make_spec.py            # write xlite.spec
    python scripts/make_spec.py --check   # exit 1 if xlite.spec is stale
"""
import argparse
import datetime
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SPEC_IN = ROOT_DIR / "xlite.spec.in"
SPEC_OUT = ROOT_DIR / "xlite.spec"


def _resolve_version() -> str:
    """Return the setuptools_scm version for this checkout.

    Mirrors ``setup.py``: same ``root`` (monorepo root = ROOT_DIR.parent) and
    same ``fallback_version`` so the two consumers can never diverge.
    """
    from setuptools_scm import get_version

    return get_version(root=ROOT_DIR.parent, fallback_version="0.0.0")


def _changelog_entry(version: str) -> str:
    """Render a ``%changelog`` entry for ``version``.

    ``setuptools_scm`` emits PEP 440 local-version segments (``+g52790bf``)
    which are invalid in an RPM Release; the entry records the raw version
    verbatim in the body so a reviewer can see what tag produced the spec.
    """
    today = datetime.date.today().strftime("%a %b %d %Y")
    return f"* {today} xlite <noreply@huawei.com> - {version}-1\n- Auto-generated spec for {version}.\n"


def render(version: str) -> str:
    """Render the spec template with ``version`` and a changelog entry."""
    template = SPEC_IN.read_text(encoding="utf-8")
    return template.replace("@VERSION@", version).replace(
        "@CHANGELOG@", _changelog_entry(version)
    )


def write_spec(version: str) -> Path:
    """Overwrite ``xlite.spec`` with the rendered template; return its path."""
    SPEC_OUT.write_text(render(version), encoding="utf-8")
    return SPEC_OUT


def is_stale(version: str) -> bool:
    """True if ``xlite.spec`` does not match a fresh render of ``version``."""
    if not SPEC_OUT.exists():
        return True
    return SPEC_OUT.read_text(encoding="utf-8") != render(version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if xlite.spec is stale vs the current version",
    )
    parser.add_argument(
        "--version",
        help="override the resolved version (e.g. for testing); defaults to setuptools_scm",
    )
    args = parser.parse_args(argv)

    version = args.version if args.version else _resolve_version()
    if args.check:
        if is_stale(version):
            print(
                f"xlite.spec is stale; regenerate with `python {Path(__file__).name}` "
                f"(current version: {version})",
                file=sys.stderr,
            )
            return 1
        print(f"xlite.spec is up to date ({version})")
        return 0

    out = write_spec(version)
    print(f"wrote {out} (version: {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
