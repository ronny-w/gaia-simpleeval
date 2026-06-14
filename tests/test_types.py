"""test_types.py — Type preservation.

The Rust evaluator must reproduce Python's numeric type semantics exactly:
  * int op int  -> int   (never float)
  * int op float -> float
  * bool is a subtype of int and must be preserved
  * comparisons yield real bool
  * arbitrary-precision results return Python int (BigInt path)

Covers SPEC.md section 10.1 (Type preservation) and rules type_semantics.
Requires the compiled extension to run.
"""

import pytest

pytestmark = pytest.mark.usefixtures("gse")


def test_int_op_int_is_int_not_bool(ev):
    r = ev("2+2")
    assert isinstance(r, int)
    assert not isinstance(r, bool)


def test_int_op_float_is_float(ev):
    r = ev("2+2.0")
    assert isinstance(r, float)


def test_bool_op_bool_is_int(ev):
    # True + True == 2, and the result type is int (bool's arithmetic widens).
    r = ev("True+True")
    assert r == 2
    assert isinstance(r, int)
    assert not isinstance(r, bool)


def test_comparison_result_is_bool(ev):
    r = ev("1<2")
    assert isinstance(r, bool)
    assert r is True


def test_true_literal_is_bool(ev):
    assert type(ev("True")) is bool
    assert type(ev("False")) is bool


def test_bigint_result_is_python_int(ev):
    r = ev("2**100")
    assert isinstance(r, int)
    assert r == 2 ** 100


def test_division_always_float(ev):
    r = ev("4/2")
    assert isinstance(r, float)
    assert r == 2.0


def test_floor_division_int_is_int(ev):
    r = ev("7//2")
    assert isinstance(r, int)
    assert not isinstance(r, bool)


def test_string_result_is_str(ev):
    assert type(ev('"a"+"b"')) is str


def test_none_literal_is_none(ev):
    assert ev("None") is None


def test_negative_bigint_round_trip(ev):
    r = ev("-(2**100)")
    assert isinstance(r, int)
    assert r == -(2 ** 100)
