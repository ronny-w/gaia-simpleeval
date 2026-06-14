"""test_fast_path.py — Fast path correctness.

These tests verify that expressions handled natively by the Rust evaluator
(default operators + default functions + primitive names) produce exactly the
same results Python would. They require the compiled extension to run.

Covers SPEC.md section 10.1 (Fast path correctness).
"""

import math

import pytest

pytestmark = pytest.mark.usefixtures("gse")


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2+2", 4),
        ("2*3", 6),
        ("10//3", 3),
        ("10%3", 1),
        ("2**10", 1024),
        ("1+2*3", 7),
        ("(1+2)*3", 9),
        ("-5", -5),
        ("+5", 5),
        ("2-3-4", -5),
    ],
)
def test_arithmetic(ev, expr, expected):
    assert ev(expr) == expected


def test_true_division(ev):
    assert math.isclose(ev("10/3"), 10 / 3, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Negative floor division (Python semantics: result rounds toward -inf)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("-7//2", -4),
        ("7//-2", -4),
        ("-7//-2", 3),
        ("7//2", 3),
    ],
)
def test_negative_floor_div(ev, expr, expected):
    assert ev(expr) == expected


# ---------------------------------------------------------------------------
# Negative modulo (Python: sign follows divisor)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("-7%3", 2),
        ("7%-3", -2),
        ("-7%-3", -1),
        ("7%3", 1),
    ],
)
def test_negative_modulo(ev, expr, expected):
    assert ev(expr) == expected


# ---------------------------------------------------------------------------
# String operations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ('"a"+"b"', "ab"),
        ('"ab"*3', "ababab"),
        ('3*"ab"', "ababab"),
        ('""+"x"', "x"),
    ],
)
def test_string_ops(ev, expr, expected):
    assert ev(expr) == expected


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1<2", True),
        ("2<1", False),
        ("1==1", True),
        ("1!=2", True),
        ("2>=2", True),
        ("2<=1", False),
        ("3>1", True),
    ],
)
def test_comparison(ev, expr, expected):
    assert ev(expr) is expected


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("True and False", False),
        ("True or False", True),
        ("not True", False),
        ("not False", True),
        ("True and True", True),
        ("False or False", False),
    ],
)
def test_boolean(ev, expr, expected):
    assert ev(expr) is expected


def test_boolean_short_circuit_values(ev):
    # Python `and`/`or` return operands, not coerced bools.
    assert ev("1 and 2") == 2
    assert ev("0 or 3") == 3


# ---------------------------------------------------------------------------
# Bitwise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("5&3", 1),
        ("5|3", 7),
        ("5^3", 6),
        ("~5", -6),
    ],
)
def test_bitwise(ev, expr, expected):
    assert ev(expr) == expected


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1<<3", 8),
        ("8>>2", 2),
        ("1<<0", 1),
    ],
)
def test_shift(ev, expr, expected):
    assert ev(expr) == expected


# ---------------------------------------------------------------------------
# Ternary / conditional
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1 if True else 2", 1),
        ("1 if False else 2", 2),
        ("'y' if 1<2 else 'n'", "y"),
    ],
)
def test_ternary(ev, expr, expected):
    assert ev(expr) == expected


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def test_names_basic(ev):
    assert ev("x+1", names={"x": 5}) == 6


def test_names_multiple(ev):
    assert ev("x+y*z", names={"x": 1, "y": 2, "z": 3}) == 7


def test_names_string(ev):
    assert ev("s+'!'", names={"s": "hi"}) == "hi!"


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------
def test_int_function(ev):
    assert ev("int('42')") == 42
    assert ev("int(3.9)") == 3


def test_float_function(ev):
    assert math.isclose(ev("float('3.14')"), 3.14, rel_tol=1e-12)
    assert ev("float(2)") == 2.0


def test_str_function(ev):
    assert ev("str(42)") == "42"


def test_abs_function(ev):
    assert ev("abs(-5)") == 5
    assert ev("abs(5)") == 5


def test_round_function(ev):
    # Banker's rounding (round-half-to-even).
    assert ev("round(2.5)") == 2
    assert ev("round(3.5)") == 4


def test_min_max(ev):
    assert ev("min(1,2)") == 1
    assert ev("max(1,2)") == 2
    assert ev("min(3,1,2)") == 1
    assert ev("max(3,1,2)") == 3


def test_len(ev):
    assert ev('len("abc")') == 3


def test_sum(ev):
    assert ev("sum([1,2,3])") == 6


def test_bool_function(ev):
    assert ev("bool(0)") is False
    assert ev("bool(1)") is True


def test_rand_in_range(ev):
    for _ in range(50):
        r = ev("rand()")
        assert isinstance(r, float)
        assert 0.0 <= r < 1.0


def test_randint_in_range(ev):
    for _ in range(50):
        r = ev("randint(10)")
        assert isinstance(r, int)
        assert 0 <= r < 10


# ---------------------------------------------------------------------------
# Container literals
# ---------------------------------------------------------------------------
def test_list_literal(ev):
    assert ev("[1,2,3]") == [1, 2, 3]


def test_tuple_literal(ev):
    assert ev("(1,2)") == (1, 2)


def test_dict_literal(ev):
    assert ev('{"a":1}') == {"a": 1}


def test_set_literal(ev):
    assert ev("{1,2,3}") == {1, 2, 3}


# ---------------------------------------------------------------------------
# Membership / identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ('"ab" in "abc"', True),
        ('"xy" in "abc"', False),
        ("1 in [1,2]", True),
        ("3 in [1,2]", False),
        ('"a" in {"a":1}', True),
        ('"b" in {"a":1}', False),
        ('"x" not in "abc"', True),
    ],
)
def test_in_operator(ev, expr, expected):
    assert ev(expr) is expected


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("None is None", True),
        ("True is True", True),
        ("None is not None", False),
    ],
)
def test_is_operator(ev, expr, expected):
    assert ev(expr) is expected
