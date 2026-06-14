"""test_parity.py — Parity with the original ``simpleeval``.

For every expression supported by BOTH gaia-simpleeval and the original
simpleeval, the result must be identical. Per the CODEGEN_PROMPT, this file only
exercises expressions supported by both implementations — it deliberately avoids:

  * gaia-only built-ins (abs/round/min/max/len/sum/bool are NOT in the original
    ``DEFAULT_FUNCTIONS``, which only provides float/int/rand/randint/str),
  * container literals and membership over containers via the plain evaluator
    (the original needs ``EvalWithCompoundTypes`` for those — tested separately
    below against the same class on the gaia side),
  * the known gaia limitations (chained comparisons, f-strings, subscript,
    attribute access).

Non-deterministic builtins (rand/randint) are excluded from value parity.

Requires BOTH the compiled gaia extension and the original simpleeval.
"""

import math

import pytest

pytestmark = [pytest.mark.usefixtures("gse"), pytest.mark.usefixtures("orig")]


# 50+ representative expressions supported by the *plain* simple_eval on both
# sides (default operators, default functions limited to float/int/str).
SHARED_EXPRESSIONS = [
    # arithmetic
    "2+2", "2-3", "2*3", "10//3", "10%3", "2**10", "1+2*3", "(1+2)*3",
    "-5", "+5", "2-3-4", "100-1", "3*4+5", "(2+3)*(4-1)",
    # negative floor div / modulo
    "-7//2", "7//-2", "-7//-2", "-7%3", "7%-3", "-7%-3",
    # big powers within limits
    "2**32", "2**62", "10**18",
    # bigint
    "2**100", "2**200",
    # true division
    "10/3", "7/2", "1/4", "9/3",
    # strings
    '"a"+"b"', '"ab"*3', '3*"ab"', '"hello"+"world"', '"x"*0',
    # comparison
    "1<2", "2<1", "1==1", "1!=2", "2>=2", "2<=1", "3>1", "5==5",
    # boolean
    "True and False", "True or False", "not True", "not False",
    "1 and 2", "0 or 3", "True and True", "False or False",
    # bitwise
    "5&3", "5|3", "5^3", "~5", "0&0", "7|8",
    # shift
    "1<<3", "8>>2", "1<<0", "256>>4",
    # ternary
    "1 if True else 2", "1 if False else 2", "10 if 3>2 else 20",
    # string membership / identity (no container needed)
    '"ab" in "abc"', '"xy" in "abc"', '"x" not in "abc"',
    "None is None", "True is True", "None is not None",
    # builtins present in BOTH default_functions
    'int("42")', "int(3.9)", "int(-3.9)", 'float("3.14")', "float(2)",
    "str(42)", 'str(3.5)',
]


@pytest.mark.parametrize("expr", SHARED_EXPRESSIONS)
def test_value_parity(gse, orig, expr):
    g = gse.simple_eval(expr)
    o = orig.simple_eval(expr)
    if isinstance(g, float) or isinstance(o, float):
        assert math.isclose(g, o, rel_tol=1e-12, abs_tol=0.0)
    else:
        assert g == o
        # Type parity matters too (int vs bool, etc.).
        assert type(g) is type(o)


def test_parity_count_is_substantial():
    # Guard against accidental shrinkage of the parity matrix.
    assert len(SHARED_EXPRESSIONS) >= 50


# ---------------------------------------------------------------------------
# Container / membership parity via EvalWithCompoundTypes on BOTH sides.
# ---------------------------------------------------------------------------
COMPOUND_EXPRESSIONS = [
    "[1,2,3]",
    "(1,2)",
    '{"a":1}',
    "{1,2,3}",
    "1 in [1,2]",
    "3 in [1,2]",
    '"a" in {"a":1}',
    "[1,2]+[3,4]",
]


@pytest.mark.parametrize("expr", COMPOUND_EXPRESSIONS)
def test_compound_type_parity(gse, orig, expr):
    g = gse.EvalWithCompoundTypes().eval(expr)
    o = orig.EvalWithCompoundTypes().eval(expr)
    assert g == o
    assert type(g) is type(o)


# ---------------------------------------------------------------------------
# Exception parity: same expression -> same exception type (and key message
# fragments) on both implementations.
# ---------------------------------------------------------------------------
EXCEPTION_EXPRESSIONS = [
    ("missing + 1", "NameNotDefined"),
    ("nope(1)", "FunctionNotDefined"),
    ("", "InvalidExpression"),
]


@pytest.mark.parametrize("expr,exc_name", EXCEPTION_EXPRESSIONS)
def test_exception_type_parity(gse, orig, expr, exc_name):
    gse_exc = getattr(gse, exc_name)
    orig_exc = getattr(orig, exc_name)

    with pytest.raises(orig_exc) as oi:
        orig.simple_eval(expr)
    with pytest.raises(gse_exc) as gi:
        gse.simple_eval(expr)

    # Messages should match exactly per decision G002 (reproduce exact formats).
    assert str(gi.value) == str(oi.value)
