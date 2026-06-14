"""gaia_simpleeval — high-performance drop-in replacement for simpleeval.

This module exposes the public Python API. Supported expressions are routed to
the native Rust evaluator (``_gaia_simpleeval.fast_eval``); anything outside the
fast-path guard falls back to the original ``simpleeval`` implementation.

The fast-path guard conditions are inlined directly in ``simple_eval`` and
``SimpleEval.eval`` — no helper function is called on the hot path, per the
project's wrapper-overhead budget (<0.1 µs).
"""
# @implements: G001, G005, G006, G007
# @spec: §2.2, §2.3, §2.6, §6.11

import sys as _sys

# --- Re-export the public API surface from the original simpleeval ----------
#
# Constants, exception classes, default operator/function tables, and helper
# callables are re-exported unchanged so that ``gaia_simpleeval`` is a true
# drop-in. The DEFAULT_OPERATORS / DEFAULT_FUNCTIONS objects must be the *same*
# objects as in simpleeval so the identity checks in the fast-path guard
# (``operators is DEFAULT_OPERATORS``) succeed.
from simpleeval import (  # noqa: F401
    DEFAULT_OPERATORS,
    DEFAULT_FUNCTIONS,
    DEFAULT_NAMES,
    InvalidExpression,
    FunctionNotDefined,
    NameNotDefined,
    AttributeDoesNotExist,
    OperatorNotDefined,
    FeatureNotAvailable,
    NumberTooHigh,
    IterableTooLong,
    AssignmentAttempted,
)

# Optional re-exports — present in most simpleeval versions but guarded so the
# import does not fail on older releases.
try:  # pragma: no cover - version dependent
    from simpleeval import (  # noqa: F401
        EvalWithCompoundTypes as _OrigEvalWithCompoundTypes,
        SimpleEval as _OrigSimpleEval,
        simple_eval as _orig_simple_eval,
    )
except ImportError:  # pragma: no cover
    import simpleeval as _se_mod
    _OrigEvalWithCompoundTypes = getattr(_se_mod, "EvalWithCompoundTypes", None)
    _OrigSimpleEval = _se_mod.SimpleEval
    _orig_simple_eval = _se_mod.simple_eval

for _optname in (
    "MultipleExpressions",
    "TypeNotSpecified",
    "BASIC_ALLOWED_ATTRS",
    "ATTR_INDEX_FALLBACK",
    "ModuleWrapper",
    "random_int",
    "safe_power",
    "safe_mult",
    "safe_add",
    "safe_rshift",
    "safe_lshift",
):
    try:
        globals()[_optname] = getattr(__import__("simpleeval"), _optname)
    except (ImportError, AttributeError):  # pragma: no cover
        pass

# --- Immutable safety constants ---------------------------------------------
#
# These mirror the hardcoded Rust constants. They are protected from runtime
# modification by the module-level ``__setattr__`` guard defined below. We set
# them via the underlying ``__dict__`` so the guard does not block our own
# initialization.
_IMMUTABLE_CONSTANTS = frozenset(
    {
        "MAX_STRING_LENGTH",
        "MAX_COMPREHENSION_LENGTH",
        "MAX_POWER",
        "MAX_SHIFT",
        "MAX_SHIFT_BASE",
    }
)

_self = _sys.modules[__name__]
_self.__dict__["MAX_STRING_LENGTH"] = 100000
_self.__dict__["MAX_COMPREHENSION_LENGTH"] = 10000
_self.__dict__["MAX_POWER"] = 4000000
_self.__dict__["MAX_SHIFT"] = 10000
_self.__dict__["MAX_SHIFT_BASE"] = int(_sys.float_info.max)

# Re-export the disallow tables for API compatibility (best-effort).
for _name in ("DISALLOW_PREFIXES", "DISALLOW_METHODS", "DISALLOW_FUNCTIONS"):
    try:
        _self.__dict__[_name] = getattr(__import__("simpleeval"), _name)
    except (ImportError, AttributeError):  # pragma: no cover
        pass


from types import ModuleType as _ModuleType


class _GuardedModule(_ModuleType):
    """Module subclass that makes the ``MAX_*`` constants immutable.

    A plain module-level ``__setattr__`` function is not honored by the
    interpreter, so we re-class this module object (assigning to
    ``self.__class__``) to install a real ``__setattr__`` hook.
    """

    def __setattr__(self, name, value):
        if name in _IMMUTABLE_CONSTANTS:
            raise AttributeError(
                "gaia-simpleeval: {0} is immutable. gaia-simpleeval uses "
                "hardcoded safety limits that cannot be modified at runtime. "
                "See LIMITATIONS.md.".format(name)
            )
        super().__setattr__(name, value)


# Re-class the live module object so the guard takes effect for all subsequent
# attribute assignments (e.g. ``gaia_simpleeval.MAX_POWER = 1``).
_self.__class__ = _GuardedModule


# --- Native extension --------------------------------------------------------
#
# When no compiled wheel is available for the current platform, fall back to the
# original simpleeval for every call (pure-Python fallback).
try:
    from . import _gaia_simpleeval as _rust
except ImportError:  # pragma: no cover - platform without a wheel
    _rust = None

_PRIMITIVE_TYPES = (int, float, bool, str, type(None))


def simple_eval(expr, operators=None, functions=None, names=None, allowed_attrs=None):
    """Evaluate a string expression and return the result.

    Routes to the native Rust evaluator when the fast-path guard holds:
    default operators/functions, no ``allowed_attrs``, and ``names`` is either
    ``None`` or a plain ``dict`` of primitive values. Otherwise delegates to the
    original Python ``simpleeval``.
    """
    if (
        _rust is not None
        and (operators is None or operators is DEFAULT_OPERATORS)
        and (functions is None or functions is DEFAULT_FUNCTIONS)
        and allowed_attrs is None
        and (
            names is None
            or (
                isinstance(names, dict)
                and all(type(v) in _PRIMITIVE_TYPES for v in names.values())
            )
        )
    ):
        return _rust.fast_eval(expr, names if names is not None else {})
    return _OrigSimpleEval(
        operators=operators,
        functions=functions,
        names=names,
        allowed_attrs=allowed_attrs,
    ).eval(expr)


class SimpleEval(_OrigSimpleEval):
    """Accelerated ``SimpleEval``.

    ``eval()`` takes the Rust fast path when the same guard as
    :func:`simple_eval` holds and no pre-parsed AST is supplied; otherwise it
    delegates to the original Python implementation.
    """

    def eval(self, expr, previously_parsed=None):
        if (
            _rust is not None
            and previously_parsed is None
            and (self.operators is DEFAULT_OPERATORS)
            and (self.functions is DEFAULT_FUNCTIONS)
            and getattr(self, "allowed_attrs", None) is None
            and isinstance(self.names, dict)
            and all(type(v) in _PRIMITIVE_TYPES for v in self.names.values())
        ):
            return _rust.fast_eval(expr, self.names)
        return super().eval(expr, previously_parsed=previously_parsed)


if _OrigEvalWithCompoundTypes is not None:

    class EvalWithCompoundTypes(_OrigEvalWithCompoundTypes):
        """Always delegates to the original Python implementation.

        Comprehensions and compound-type handling are not accelerated; this
        subclass exists purely for API compatibility / re-export.
        """

        pass

else:  # pragma: no cover

    EvalWithCompoundTypes = None


__all__ = [
    "simple_eval",
    "SimpleEval",
    "EvalWithCompoundTypes",
    "MAX_STRING_LENGTH",
    "MAX_COMPREHENSION_LENGTH",
    "MAX_POWER",
    "MAX_SHIFT",
    "MAX_SHIFT_BASE",
    "DEFAULT_OPERATORS",
    "DEFAULT_FUNCTIONS",
    "DEFAULT_NAMES",
    "InvalidExpression",
    "FunctionNotDefined",
    "NameNotDefined",
    "AttributeDoesNotExist",
    "OperatorNotDefined",
    "FeatureNotAvailable",
    "NumberTooHigh",
    "IterableTooLong",
    "AssignmentAttempted",
]
