"""test_security.py — Security / sandbox guarantees.

The sandbox model (rules.security: sandbox_no_fallback) requires that unsupported
*dangerous* constructs raise explicit errors rather than silently falling back,
and that the fast-path guard refuses non-primitive names so that arbitrary Python
objects never reach the native evaluator.

Covers SPEC.md section 10.1 (Security) and decisions G004/G005/G008.
Requires the compiled extension to run.
"""

import pytest

pytestmark = pytest.mark.usefixtures("gse")


# ---------------------------------------------------------------------------
# Fast-path guard rejects non-primitive names (must NOT reach Rust).
# We assert behaviour rather than introspecting internals: a module/callable
# value must not crash the evaluator and must follow the Python path.
# ---------------------------------------------------------------------------
def test_names_with_module_value_falls_back(gse):
    import math as math_mod

    # With default SimpleEval, attribute access on the module is blocked, but the
    # guard must route to Python (not hand a module to Rust). Using the bare name
    # in a no-op position should evaluate via the fallback without a Rust crash.
    funcs = dict(gse.DEFAULT_FUNCTIONS)
    funcs["is_module"] = lambda m: hasattr(m, "pi")
    # module objects are blocked by Python fallback (FeatureNotAvailable)
    import pytest
    with pytest.raises(gse.FeatureNotAvailable):
        gse.simple_eval("is_module(m)", names={"m": math_mod}, functions=funcs)


def test_names_with_callable_value_falls_back(gse):
    def make():
        return 5

    funcs = dict(gse.DEFAULT_FUNCTIONS)
    funcs["call_it"] = lambda f: f()
    assert gse.simple_eval("call_it(fn)", names={"fn": make}, functions=funcs) == 5


# ---------------------------------------------------------------------------
# Dangerous constructs -> explicit error, never silent fallback.
# ---------------------------------------------------------------------------
def test_attribute_access_is_explicit_feature_error(gse):
    # Must be FeatureNotAvailable (not a silent fallback / not NameNotDefined).
    with pytest.raises(gse.FeatureNotAvailable):
        gse.simple_eval("'x'.__class__")


def test_dunder_import_is_blocked(gse):
    # __import__ is not a defined function/name -> NameNotDefined or
    # FeatureNotAvailable, but never executed.
    with pytest.raises((gse.NameNotDefined, gse.FeatureNotAvailable, gse.FunctionNotDefined)):
        gse.simple_eval("__import__('os')")


@pytest.mark.parametrize("expr", ["eval('1')", "exec('x=1')", "open('/etc/passwd')"])
def test_forbidden_builtins_raise_function_not_defined(gse, expr):
    with pytest.raises(gse.FunctionNotDefined):
        gse.simple_eval(expr)


def test_attribute_chain_does_not_leak(gse):
    # A classic sandbox-escape probe must be refused explicitly.
    with pytest.raises(gse.InvalidExpression):
        gse.simple_eval("().__class__.__bases__")
