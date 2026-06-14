"""test_fallback.py — Python fallback path.

The fast-path guard (decisions G005/G008) only routes to Rust when:
  * operators is None or DEFAULT_OPERATORS, AND
  * functions is None or DEFAULT_FUNCTIONS, AND
  * allowed_attrs is None, AND
  * names is None or a plain dict whose values are all primitives
    (int, float, bool, str, None).

Anything else must fall back to the original Python ``simpleeval`` evaluator and
still produce a correct result. These tests assert *correctness* of the fallback
result; whether a given input actually fell back is asserted more directly in
``test_security.py`` where it matters for the sandbox guarantee.

Requires the compiled extension to run.
"""

import pytest

pytestmark = pytest.mark.usefixtures("gse")


def test_custom_operator_falls_back(gse):
    """Custom operators must not use the Rust fast path, but still evaluate."""
    import ast
    import operator

    ops = dict(gse.DEFAULT_OPERATORS)
    # Redefine subtraction as addition to make the fallback observable.
    ops[ast.Sub] = operator.add
    result = gse.simple_eval("10-3", operators=ops)
    assert result == 13  # 10 + 3 via the overridden Python operator


def test_custom_function_falls_back(gse):
    funcs = dict(gse.DEFAULT_FUNCTIONS)
    funcs["double"] = lambda x: x * 2
    assert gse.simple_eval("double(21)", functions=funcs) == 42


def test_allowed_attrs_falls_back(gse):
    """Setting allowed_attrs forces the Python path (attrs need real objects)."""
    # An expression that needs the Python attribute machinery.
    result = gse.simple_eval(
        "s.upper()",
        names={"s": "abc"},
        allowed_attrs={str: ["upper"]},
    )
    assert result == "ABC"


def test_names_non_primitive_value_falls_back(gse):
    """A names dict containing a non-primitive value must fall back."""

    class Custom:
        value = 99

    # Accessing the attribute requires the Python evaluator; with the default
    # SimpleEval (no allowed_attrs) attribute access is blocked, so we instead
    # verify a non-primitive value simply round-trips through the fallback when
    # used by identity, e.g. passed to a custom function.
    funcs = dict(gse.DEFAULT_FUNCTIONS)
    funcs["get_value"] = lambda o: o.value
    assert gse.simple_eval("get_value(obj)", names={"obj": Custom()}, functions=funcs) == 99


def test_names_callable_falls_back(gse):
    """names given as a callable (not a dict) must fall back to Python."""

    def name_lookup(node):
        # simpleeval calls names(node) with the ast.Name node.
        return {"x": 7}[node.id]

    assert gse.simple_eval("x+1", names=name_lookup) == 8


def test_names_dict_with_list_value_falls_back(gse):
    """A dict value that is a list is non-primitive -> Python path."""
    funcs = dict(gse.DEFAULT_FUNCTIONS)
    funcs["first"] = lambda xs: xs[0]
    assert gse.simple_eval("first(items)", names={"items": [10, 20]}, functions=funcs) == 10


def test_previously_parsed_uses_python_path(gse):
    """SimpleEval.eval(expr, previously_parsed=...) exercises the Python path."""
    se = gse.SimpleEval()
    # First parse to obtain an AST, then re-evaluate with previously_parsed set.
    se.eval("2+3")
    parsed = getattr(se, "expr_ast", None)
    if parsed is None:
        pytest.skip("SimpleEval does not expose expr_ast for previously_parsed reuse")
    assert se.eval("2+3", previously_parsed=parsed) == 5
