//! Unit tests for the native built-in functions (`call_builtin`).
//!
//! Calls `call_builtin(func_name, args, original_expr)` directly with
//! constructed `Value` arguments and asserts on results — covering int/float/
//! str/bool conversions, abs, banker's rounding, min/max (variadic and single
//! list form), len, sum (int vs float result), rand/randint ranges, and the
//! `FunctionNotDefined` path for unknown functions.

use super::super::*;

fn call(name: &str, args: Vec<Value>) -> Value {
    call_builtin(name, args, name)
        .unwrap_or_else(|e| panic!("call_builtin({name:?}) failed: {e:?}"))
}

fn call_res(name: &str, args: Vec<Value>) -> EvalResult<Value> {
    call_builtin(name, args, name)
}

// ----- int() -----

#[test]
fn test_int_from_float_truncates() {
    match call("int", vec![Value::Float(3.9)]) {
        Value::Int(3) => {}
        other => panic!("int(3.9) => {other:?}"),
    }
    match call("int", vec![Value::Float(-3.9)]) {
        Value::Int(-3) => {}
        other => panic!("int(-3.9) => {other:?}"),
    }
}

#[test]
fn test_int_from_string() {
    match call("int", vec![Value::Str("42".to_string())]) {
        Value::Int(42) => {}
        other => panic!("int('42') => {other:?}"),
    }
}

#[test]
fn test_int_from_string_with_whitespace() {
    match call("int", vec![Value::Str("  7  ".to_string())]) {
        Value::Int(7) => {}
        other => panic!("int('  7  ') => {other:?}"),
    }
}

#[test]
fn test_int_from_bool() {
    assert!(matches!(call("int", vec![Value::Bool(true)]), Value::Int(1)));
    assert!(matches!(call("int", vec![Value::Bool(false)]), Value::Int(0)));
}

#[test]
fn test_int_from_invalid_string_errors() {
    match call_res("int", vec![Value::Str("notanint".to_string())]) {
        Err(EvalError::InvalidExpression(_)) => {}
        other => panic!("int('notanint') should error, got {other:?}"),
    }
}

// ----- float() -----

#[test]
fn test_float_from_int() {
    match call("float", vec![Value::Int(5)]) {
        Value::Float(f) => assert_eq!(f, 5.0),
        other => panic!("float(5) => {other:?}"),
    }
}

#[test]
fn test_float_from_string() {
    match call("float", vec![Value::Str("3.14".to_string())]) {
        Value::Float(f) => assert!((f - 3.14).abs() < 1e-9),
        other => panic!("float('3.14') => {other:?}"),
    }
}

#[test]
fn test_float_from_invalid_string_errors() {
    match call_res("float", vec![Value::Str("xyz".to_string())]) {
        Err(EvalError::InvalidExpression(_)) => {}
        other => panic!("float('xyz') should error, got {other:?}"),
    }
}

// ----- str() -----

#[test]
fn test_str_of_int() {
    match call("str", vec![Value::Int(42)]) {
        Value::Str(s) => assert_eq!(s, "42"),
        other => panic!("str(42) => {other:?}"),
    }
}

#[test]
fn test_str_of_float_appends_dot_zero() {
    match call("str", vec![Value::Float(1.0)]) {
        Value::Str(s) => assert_eq!(s, "1.0"),
        other => panic!("str(1.0) => {other:?}"),
    }
}

#[test]
fn test_str_of_float_with_decimal_unchanged() {
    match call("str", vec![Value::Float(3.5)]) {
        Value::Str(s) => assert_eq!(s, "3.5"),
        other => panic!("str(3.5) => {other:?}"),
    }
}

#[test]
fn test_str_of_bool_and_none() {
    match call("str", vec![Value::Bool(true)]) {
        Value::Str(s) => assert_eq!(s, "True"),
        other => panic!("str(True) => {other:?}"),
    }
    match call("str", vec![Value::Bool(false)]) {
        Value::Str(s) => assert_eq!(s, "False"),
        other => panic!("str(False) => {other:?}"),
    }
    match call("str", vec![Value::None]) {
        Value::Str(s) => assert_eq!(s, "None"),
        other => panic!("str(None) => {other:?}"),
    }
}

// ----- bool() -----

#[test]
fn test_bool_truthiness() {
    assert!(matches!(call("bool", vec![Value::Int(0)]), Value::Bool(false)));
    assert!(matches!(call("bool", vec![Value::Int(5)]), Value::Bool(true)));
    assert!(matches!(
        call("bool", vec![Value::Str(String::new())]),
        Value::Bool(false)
    ));
    assert!(matches!(
        call("bool", vec![Value::Str("x".to_string())]),
        Value::Bool(true)
    ));
}

// ----- abs() -----

#[test]
fn test_abs_int_and_float() {
    assert!(matches!(call("abs", vec![Value::Int(-5)]), Value::Int(5)));
    assert!(matches!(call("abs", vec![Value::Int(5)]), Value::Int(5)));
    match call("abs", vec![Value::Float(-2.5)]) {
        Value::Float(f) => assert_eq!(f, 2.5),
        other => panic!("abs(-2.5) => {other:?}"),
    }
}

// ----- round() with banker's rounding -----

#[test]
fn test_round_half_to_even() {
    // round(0.5) == 0, round(1.5) == 2, round(2.5) == 2, round(3.5) == 4
    for (input, expected) in [(0.5, 0i64), (1.5, 2), (2.5, 2), (3.5, 4)] {
        match call("round", vec![Value::Float(input)]) {
            Value::Int(n) => assert_eq!(n, expected, "round({input}) => {n}"),
            other => panic!("round({input}) => {other:?}"),
        }
    }
}

#[test]
fn test_round_to_decimal_places() {
    match call("round", vec![Value::Float(3.14159), Value::Int(2)]) {
        Value::Float(f) => assert!((f - 3.14).abs() < 1e-9, "round(3.14159, 2) => {f}"),
        other => panic!("round(3.14159, 2) => {other:?}"),
    }
}

// ----- min / max -----

#[test]
fn test_min_max_variadic() {
    assert!(matches!(
        call("min", vec![Value::Int(3), Value::Int(1), Value::Int(2)]),
        Value::Int(1)
    ));
    assert!(matches!(
        call("max", vec![Value::Int(3), Value::Int(1), Value::Int(2)]),
        Value::Int(3)
    ));
}

#[test]
fn test_min_max_single_list_argument() {
    let lst = Value::List(vec![Value::Int(5), Value::Int(2), Value::Int(9)]);
    assert!(matches!(call("min", vec![lst.clone()]), Value::Int(2)));
    assert!(matches!(call("max", vec![lst]), Value::Int(9)));
}

// ----- len -----

#[test]
fn test_len_of_string() {
    assert!(matches!(
        call("len", vec![Value::Str("abc".to_string())]),
        Value::Int(3)
    ));
}

#[test]
fn test_len_of_list() {
    let lst = Value::List(vec![Value::Int(1), Value::Int(2)]);
    assert!(matches!(call("len", vec![lst]), Value::Int(2)));
}

#[test]
fn test_len_of_dict() {
    let d = Value::Dict(vec![(Value::Str("a".to_string()), Value::Int(1))]);
    assert!(matches!(call("len", vec![d]), Value::Int(1)));
}

// ----- sum -----

#[test]
fn test_sum_all_int_returns_int() {
    let lst = Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert!(matches!(call("sum", vec![lst]), Value::Int(6)));
}

#[test]
fn test_sum_with_float_returns_float() {
    let lst = Value::List(vec![Value::Int(1), Value::Float(2.5)]);
    match call("sum", vec![lst]) {
        Value::Float(f) => assert!((f - 3.5).abs() < 1e-9, "sum => {f}"),
        other => panic!("sum([1, 2.5]) => {other:?}"),
    }
}

// ----- rand / randint -----

#[test]
fn test_rand_in_unit_interval() {
    for _ in 0..50 {
        match call("rand", vec![]) {
            Value::Float(f) => assert!((0.0..1.0).contains(&f), "rand() out of range: {f}"),
            other => panic!("rand() => {other:?}"),
        }
    }
}

#[test]
fn test_randint_in_range() {
    for _ in 0..50 {
        match call("randint", vec![Value::Int(10)]) {
            Value::Int(n) => assert!((0..10).contains(&n), "randint(10) out of range: {n}"),
            other => panic!("randint(10) => {other:?}"),
        }
    }
}

// ----- unknown function -----

#[test]
fn test_unknown_function_errors() {
    match call_res("frobnicate", vec![Value::Int(1)]) {
        Err(EvalError::FunctionNotDefined { func_name, .. }) => {
            assert_eq!(func_name, "frobnicate");
        }
        other => panic!("expected FunctionNotDefined, got {other:?}"),
    }
}
