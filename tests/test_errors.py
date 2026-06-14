"""test_errors.py — Error cases.

Verifies that unsupported or invalid expressions raise the correct exception
types (and, where specified, the exact message). The exception hierarchy mirrors
the original simpleeval: all custom exceptions derive from ``InvalidExpression``.

Covers SPEC.md section 10.1 (Error cases) and decisions
G001 (chained comparisons), G003 (f-strings), G004 (attribute access),
G006 (MAX_* immutability), G011 (subscript).

Requires the compiled extension to run.
"""

import pytest

pytestmark = pytest.mark.usefixtures("gse")


# ---------------------------------------------------------------------------
# Known limitations -> explicit errors (no silent fallback)
# ---------------------------------------------------------------------------
def test_chained_comparison_raises_invalid_expression(gse):
    # G001: chained comparisons are not implemented and must error, not fall back.
    with pytest.raises(gse.InvalidExpression):
        gse.simple_eval("1 < 2 < 3")


def test_attribute_access_raises_feature_not_available(gse):
    # G004: attribute access is explicitly blocked for the sandbox guarantee.
    with pytest.raises(gse.FeatureNotAvailable):
        gse.simple_eval("'abc'.upper")


def test_subscript_raises_invalid_expression(gse):
    # G011: subscript / indexing is not supported.
    with pytest.raises(gse.InvalidExpression):
        gse.simple_eval("[1,2,3][0]")


def test_fstring_raises_invalid_expression(gse):
    # G003: f-strings are not supported.
    with pytest.raises(gse.InvalidExpression):
        gse.simple_eval('f"{1+1}"')


# ---------------------------------------------------------------------------
# Undefined names / functions (exact message match)
# ---------------------------------------------------------------------------
def test_undefined_name_raises_with_message(gse):
    expr = "missing + 1"
    with pytest.raises(gse.NameNotDefined) as ei:
        gse.simple_eval(expr)
    # Original simpleeval message format:
    # "'{name}' is not defined for expression '{expression}'"
    assert "missing" in str(ei.value)
    assert "is not defined" in str(ei.value)
    assert expr in str(ei.value)


def test_undefined_function_raises_with_message(gse):
    expr = "nope(1)"
    with pytest.raises(gse.FunctionNotDefined) as ei:
        gse.simple_eval(expr)
    # "Function '{func_name}' not defined, for expression '{expression}'."
    assert "nope" in str(ei.value)
    assert "not defined" in str(ei.value)
    assert expr in str(ei.value)


# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------
def test_power_too_high_raises_number_too_high(gse):
    with pytest.raises(gse.NumberTooHigh):
        gse.simple_eval("10 ** 9999999")


def test_shift_too_high_raises_number_too_high(gse):
    with pytest.raises(gse.NumberTooHigh):
        gse.simple_eval("1 << 999999")


def test_string_too_long_raises_iterable_too_long(gse):
    # "x" * huge would exceed MAX_STRING_LENGTH (100000).
    with pytest.raises(gse.IterableTooLong):
        gse.simple_eval('"x" * 999999')


# ---------------------------------------------------------------------------
# Arithmetic / parse errors
# ---------------------------------------------------------------------------
def test_division_by_zero_raises_invalid_expression(gse):
    with pytest.raises(gse.InvalidExpression):
        gse.simple_eval("1/0")


def test_empty_string_raises_invalid_expression(gse):
    with pytest.raises(gse.InvalidExpression):
        gse.simple_eval("")


# ---------------------------------------------------------------------------
# MAX_* immutability (G006) — exact AttributeError message
# ---------------------------------------------------------------------------
EXPECTED_MAX_POWER_MSG = (
    "gaia-simpleeval: MAX_POWER is immutable. gaia-simpleeval uses hardcoded "
    "safety limits that cannot be modified at runtime. See LIMITATIONS.md."
)


def test_setattr_max_power_raises_attribute_error(gse):
    with pytest.raises(AttributeError) as ei:
        setattr(gse, "MAX_POWER", 1)
    assert str(ei.value) == EXPECTED_MAX_POWER_MSG


@pytest.mark.parametrize(
    "const_name",
    ["MAX_STRING_LENGTH", "MAX_COMPREHENSION_LENGTH", "MAX_POWER", "MAX_SHIFT"],
)
def test_setattr_max_constants_raise_attribute_error(gse, const_name):
    with pytest.raises(AttributeError):
        setattr(gse, const_name, 1)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
def test_exception_hierarchy(gse):
    assert issubclass(gse.FunctionNotDefined, gse.InvalidExpression)
    assert issubclass(gse.NameNotDefined, gse.InvalidExpression)
    assert issubclass(gse.FeatureNotAvailable, gse.InvalidExpression)
    assert issubclass(gse.NumberTooHigh, gse.InvalidExpression)
    assert issubclass(gse.IterableTooLong, gse.InvalidExpression)
