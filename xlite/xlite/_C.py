"""xlite._C — Python shim over the abi3 SABI binding module.

Loads ``xlite/xlite_sabi.abi3.so`` through torch's dispatcher and re-exports the
registered classes/ops so existing ``from xlite._C import Runtime, add, ...``
call sites work unchanged after the pybind11 → SABI/stable-ABI migration. The
native surface lives under the ``xlite_sabi`` torch namespace (classes under
``torch.classes.xlite_sabi``, ops under ``torch.ops.xlite_sabi``); this module
re-exports both so callers use plain ``xlite._C.<name>``.

Three things the shim supplies that torch can't expose natively:

- **Enums** (AttnType, ScoringFuncType) are reproduced as ``IntEnum`` —
  torch::class_ has no enum primitive — with int values matching the C++ enum
  underlying values. Aliases (AttnMHA, ScoringFuncSoftmax, ...) mirror the flat
  names the pybind11 module exported.
- **Forward overloads**: torch ops don't overload by arg count, so each forward
  variant takes its optional arguments (aux_outputs, deepstack_input,
  input_ids) as defaulted parameters — omitting them selects the regular path.
  See the TORCH_LIBRARY block in csrc/_C.cpp for the full variant mapping.
- **Enforced binding contract**: callables whose ``_C.pyi`` signature carries
  ``/`` or ``*`` markers are re-exported as thin Python wrappers with the same
  native markers, so contract violations fail with TypeError at this layer (the
  pybind11 behavior). torch can't provide this natively — schemas have no
  positional-only args, and ``torch.ops`` shape violations raise RuntimeError.
  ``_C.pyi`` is the single source of truth for names, order, markers, defaults.
  Model methods are the one unenforced gap: custom-class methods can't be
  intercepted from Python, so their ``_C.pyi`` markers are a typing contract
  only.

Ops not registered in a given build (e.g. bench_timed/mte/vec/cube unless
``XLITE_BENCH_TIMED=ON``) bind ``None`` here, so ``getattr(_C, name, None)``
and ``hasattr`` availability checks behave as in the pybind11 era; ``_MISSING``
lists those names for callers that need the compiled-out set.
"""

from __future__ import annotations

import ast
import enum
import os

import torch

# ---------------------------------------------------------------------------
# Load the SABI .so through torch's dispatcher (idempotent).
# ---------------------------------------------------------------------------
_SO = os.path.join(os.path.dirname(__file__), "xlite_sabi.abi3.so")
_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    if not os.path.exists(_SO):
        raise ImportError(
            f"xlite SABI module not found at {_SO}; build xlite_sabi.abi3.so "
            "(cmake --build build && cmake --install build, or "
            "pip install --no-build-isolation -v -e .) before importing xlite._C"
        )
    torch.ops.load_library(_SO)
    _LOADED = True


_ensure_loaded()

# ---------------------------------------------------------------------------
# Classes (torch custom classes registered in namespace "xlite_sabi").
# ---------------------------------------------------------------------------
_ops_ns = torch.ops
_classes_ns = torch.classes

Runtime = _classes_ns.xlite_sabi.Runtime
AttnMeta = _classes_ns.xlite_sabi.AttnMeta
AttnMetaV2 = _classes_ns.xlite_sabi.AttnMetaV2
CoreAssigner = _classes_ns.xlite_sabi.CoreAssigner
# Model / ModelConfig wrap the raw torch.classes objects in a subclassable
# proxy (defined below, after _parse_pyi supplies the field lists).

# ---------------------------------------------------------------------------
# Enums — IntEnum with the same int values the C++ enums use, and the same
# member names the pybind11 block exported (.export_values + module-level
# aliases). ModelConfig.attn_type / scoring_func carry these as int64.
# ---------------------------------------------------------------------------
AttnType = enum.IntEnum(
    "AttnType",
    {
        "AttnMHA": 0,
        "AttnMLA": 1,
        "AttnDSA": 2,
        "AttnHybridGdn": 3,
        "AttnHybridKda": 4,
        "AttnCxA": 5,
    },
)
ScoringFuncType = enum.IntEnum(
    "ScoringFuncType",
    {
        "ScoringFuncSoftmax": 0,
        "ScoringFuncSigmoid": 1,
        "ScoringFuncSqrtsoftplus": 2,
    },
)

# Flat aliases mirroring the pybind11 .export_values() module-level names.
AttnMHA = AttnType.AttnMHA
AttnMLA = AttnType.AttnMLA
AttnDSA = AttnType.AttnDSA
AttnHybridGdn = AttnType.AttnHybridGdn
AttnHybridKda = AttnType.AttnHybridKda
AttnCxA = AttnType.AttnCxA
ScoringFuncSoftmax = ScoringFuncType.ScoringFuncSoftmax
ScoringFuncSigmoid = ScoringFuncType.ScoringFuncSigmoid
ScoringFuncSqrtsoftplus = ScoringFuncType.ScoringFuncSqrtsoftplus

# ---------------------------------------------------------------------------
# Ops. Bound from torch.ops.xlite_sabi.<name> into this module's namespace so
# `from xlite._C import add` / `xlite._C.add(...)` both work. Not-registered ops
# (compiled out unless XLITE_BENCH_TIMED=ON) bind None — see _MISSING below.
# The op-name list is derived from _C.pyi (single source of truth) by
# _parse_pyi() below, so adding an op needs only _C.cpp + _C.pyi.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Enforced binding contract. _C.pyi marks every callable whose call shape is
# restricted (`/` positional-only, `*` keyword-only). torch's dispatcher can't
# enforce those markers (see module docstring), so generate Python wrappers
# with native `/`/`*` syntax from the stub and re-export them under the same
# names. tests/funcs/binding_contract.py probes every generated wrapper.
# ---------------------------------------------------------------------------
_UNSET = object()


def _pyi_default(node):
    """Literal default from a .pyi AST node (the forms the stub uses)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    if isinstance(node, ast.List) and not node.elts:
        return []
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _pyi_default(node.left), _pyi_default(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
    raise ImportError(f"xlite._C: unsupported default literal in _C.pyi: {ast.dump(node)}")


def _pyi_params(args):
    """(posonly, posorkw, kwonly) lists of (name, default-or-_UNSET)."""
    combined = [(a.arg, _UNSET) for a in args.posonlyargs + args.args]
    first_default = len(combined) - len(args.defaults)
    for i, d in enumerate(args.defaults):
        combined[first_default + i] = (combined[first_default + i][0], _pyi_default(d))
    split = len(args.posonlyargs)
    posonly, posorkw = combined[:split], combined[split:]
    kwonly = [(a.arg, _UNSET if d is None else _pyi_default(d)) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    return posonly, posorkw, kwonly


def _parse_pyi():
    """Parse _C.pyi once. Return (op_names, contract_specs, class_specs).

    op_names: tuple of every module-level function name in pyi source order
        (the full op surface, including compiled-out bench ops). Replaces the
        former hand-maintained _OP_NAMES tuple — _C.pyi is now the sole source
        for op names, so adding an op needs only _C.cpp + _C.pyi.
    contract_specs: {key: (params, doc)} where key is an op name (str) for
        MARKED module-level functions (those carrying ``/`` or ``*``), or
        ("__init__", ClassName) for marked constructors. Unmarked module-level
        funcs appear in op_names but NOT here — they have no call-shape contract
        to enforce, so they bind the raw op handle.
    class_specs: {ClassName: {"attrs": [...], "methods": [...]}} for the
        torch::class_-registered classes. attrs (AnnAssign field names) drive
        the _Fwd descriptors on the proxy classes (see _class_proxy) so that
        downstream ``class X(Mixin, Model)`` survives the mixin's
        ``hasattr(type(self), name)`` gate. methods (non-__init__ FunctionDef
        names) are forwarded via __getattr__ / explicit overrides.
    """
    pyi = os.path.join(os.path.dirname(__file__), "_C.pyi")
    with open(pyi, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=pyi)
    op_names = []
    specs = {}
    class_specs = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            attrs = []
            methods = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attrs.append(item.target.id)
                elif isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        if item.args.posonlyargs or item.args.kwonlyargs:
                            # Drop `self`; the wrapper is a factory function.
                            item.args.args = item.args.args[1:]
                            specs[("__init__", node.name)] = (
                                _pyi_params(item.args),
                                ast.get_docstring(item),
                            )
                    else:
                        methods.append(item.name)
            class_specs[node.name] = {"attrs": attrs, "methods": methods}
        elif isinstance(node, ast.FunctionDef):
            if node.name not in op_names:  # dedup: tolerate future @overload
                op_names.append(node.name)
            if node.args.posonlyargs or node.args.kwonlyargs:
                specs[node.name] = (_pyi_params(node.args), ast.get_docstring(node))
    return tuple(op_names), specs, class_specs


def _contract_wrapper(name, params, doc, target):
    posonly, posorkw, kwonly = params

    def fmt(entries):
        return [n if d is _UNSET else f"{n}={d!r}" for n, d in entries]

    sig = []
    if posonly:
        sig.append(", ".join(n for n, _ in posonly) + ", /")
    if posorkw:
        sig.append(", ".join(fmt(posorkw)))
    if kwonly:
        sig.append("*, " + ", ".join(fmt(kwonly)))
    call = [n for n, _ in posonly + posorkw] + [f"{n}={n}" for n, _ in kwonly]
    src = f"def {name}({', '.join(sig)}):\n    return _target({', '.join(call)})\n"
    ns = {"_target": target}
    exec(src, ns)
    fn = ns[name]
    fn.__doc__ = doc
    return fn


# ---------------------------------------------------------------------------
# Single _C.pyi parse drives both phases: op-name list (every module-level
# function) + contract specs (only marked functions / ctors) + class specs
# (Model/ModelConfig field+method names for the proxy classes below).
# ---------------------------------------------------------------------------
_OP_NAMES, _specs, _class_specs = _parse_pyi()


# ---------------------------------------------------------------------------
# Class proxies for Model / ModelConfig.
#
# torch::class_ registers these as torch ScriptObjects whose metaclass is not a
# subclass of `type`, so Python cannot subclass them — `class X(Mixin, Model)`
# raises metaclass conflict. Downstream relies on that subclassing
# (`class XModel(AttributeSetterMixin, Model)`), so the proxy is a plain
# Python class holding the real ScriptObject as `_inner`.
#
# Two subtleties (both verified):
# 1. AttributeSetterMixin's __setattr__ gates on `hasattr(type(self), name) or
#    name in self.__dict__`. ScriptObject does not expose def_rw fields at
#    class level, so `hasattr(Model, 'embed')` is False; a plain
#    __getattr__/__setattr__ proxy is silently blocked by the mixin (every
#    field write, including `_inner`, is dropped, then reads recurse). The fix
#    is a class-level forwarding descriptor per field, so the gate passes.
# 2. A proxy passed straight to a C++ method expecting `intrusive_ptr<T>` is
#    undefined behaviour (toCustomClass<T> casts out a garbage pointer). So
#    Model.init unwraps its `config` arg to the underlying ScriptObject before
#    forwarding. Runtime / AttnMeta stay raw ScriptObjects downstream, so
#    forward*'s rt/attn_meta need no unwrap.
# ---------------------------------------------------------------------------
class _Fwd:
    """Class-level forwarding descriptor: routes get/set to ``obj._inner``
    (via object.__getattribute__ to bypass overrides)."""

    def __init__(self, name):
        self._n = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(object.__getattribute__(obj, "_inner"), self._n)

    def __set__(self, obj, value):
        setattr(object.__getattribute__(obj, "_inner"), self._n, value)


def _class_proxy(real_cls, class_name):
    """Build a Python-subclassable proxy for a torch ScriptObject class.
    Each def_rw field (from _class_specs[class_name]) becomes a _Fwd; methods
    forward via __getattr__. Model.init unwraps its config arg (see above)."""
    spec = _class_specs.get(class_name, {})
    attrs = spec.get("attrs", [])

    ns = {"_inner": None}  # class attr: lets the mixin gate pass for _inner
    for a in attrs:
        ns[a] = _Fwd(a)

    def __init__(self):
        object.__setattr__(self, "_inner", real_cls())

    ns["__init__"] = __init__

    def __getattr__(self, name):
        # Reaching here for `_inner` means it was never set — raise to avoid
        # recursion. Everything else (fields not in pyi, all methods) forwards.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_inner"), name)

    ns["__getattr__"] = __getattr__

    if class_name == "Model":

        def init(self, config, rank=0):
            real = getattr(config, "_inner", config)
            object.__getattribute__(self, "_inner").init(real, rank)

        ns["init"] = init

    return type(class_name, (), ns)


ModelConfig = _class_proxy(_classes_ns.xlite_sabi.ModelConfig, "ModelConfig")
Model = _class_proxy(_classes_ns.xlite_sabi.Model, "Model")

_sabi_ops = torch.ops.xlite_sabi
_MISSING = []

# Phase 1: bind every op name. Registered ops get the torch op handle;
# compiled-out ops (e.g. bench_timed/mte/vec/cube unless XLITE_BENCH_TIMED=ON)
# bind None so downstream `getattr(_C, name, None) is None` / `hasattr`
# availability checks behave as in the pybind11 era. _MISSING lists the
# compiled-out names for callers that need the definitive set.
for _name in _OP_NAMES:
    _op = getattr(_sabi_ops, _name, None)
    if _op is None:
        globals()[_name] = None
        _MISSING.append(_name)
    else:
        globals()[_name] = _op
del _name

# Phase 2: upgrade marked ops + marked constructors to contract wrappers that
# enforce the `/`/`*` call shape from _C.pyi (torch can't express this natively
# — see module docstring). Unmarked ops keep the raw handle from Phase 1.
# name -> (wrapper, posonly names, pos-or-kw names, kwonly names)
_CONTRACT = {}
for _key, (_params, _doc) in _specs.items():
    if isinstance(_key, tuple):
        # Marked constructor: replace the class binding with a factory
        # function carrying the enforced shape (only Runtime today).
        _name = _key[1]
    else:
        _name = _key
        if _name in _MISSING:
            continue  # compiled-out op: leave the None binding from Phase 1
    _target = globals()[_name]
    _fn = _contract_wrapper(_name, _params, _doc, _target)
    globals()[_name] = _fn
    _CONTRACT[_name] = (
        _fn,
        [n for n, _ in _params[0]],
        [n for n, _ in _params[1]],
        [n for n, _ in _params[2]],
    )
del _key, _params, _doc, _name, _target, _fn

__all__ = [
    "Runtime",
    "ModelConfig",
    "AttnMeta",
    "AttnMetaV2",
    "Model",
    "CoreAssigner",
    "AttnType",
    "ScoringFuncType",
    "AttnMHA",
    "AttnMLA",
    "AttnDSA",
    "AttnHybridGdn",
    "AttnHybridKda",
    "AttnCxA",
    "ScoringFuncSoftmax",
    "ScoringFuncSigmoid",
    "ScoringFuncSqrtsoftplus",
    *_OP_NAMES,
]
