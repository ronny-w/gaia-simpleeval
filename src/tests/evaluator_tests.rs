//! Unit tests for the evaluator (`eval_expr`, `value_is_truthy`).
//!
//! Each test tokenizes -> parses -> evaluates a source string against a
//! `names` map and asserts on the resulting `Value`. Covers arithmetic
//! (incl. Python floor-div / modulo semantics and integer tier promotion),
//! comparisons, boolean short-circuit, bitwise/shift, membership, identity,
//! ternary, names, truthiness, and the documented error cases.

use super::super::*;
use std::collections::HashMap;

/// Tokenize -> parse -> eval against an empty names map.
fn ev(src: &str) -> Value {
    let names: HashMap<String, Value> = HashMap::new();
    ev_with(src, &names)
}

/// Tokenize -> parse -> eval against a supplied names map.
fn ev_with(src: &str, names: &HashMap<String, Value>) -> Value {
    let toks = tokenize(src).unwrap_or_else(|e| panic!("tokenize({src:?}): {e:?}"));
    let ast = parse(&toks, src).unwrap_or_else(|e| panic!("parse({src:?}): {e:?}"));
    eval_expr(&ast, names, src).unwrap_or_else(|e| panic!("eval({src:?}): {e:?}"))
}

/// Tokenize -> parse -> eval, returning the Result (for error tests).
fn ev_res(src: &str) -> EvalResult<Value> {
    let names: HashMap<String, Value> = HashMap::new();
    let toks = tokenize(src)?;
    let ast = parse(&toks, src)?;
    eval_expr(&ast, &names, src)
}

fn assert_int(src: &str, expected: i64) {
    match ev(src) {
        Value::Int(n) => assert_eq!(n, expected, "{src} => {n}, expected {expected}"),
        other => panic!("{src}: expected Int({expected}), got {other:?}"),
    }
}

fn assert_float(src: &str, expected: f64) {
    match ev(src) {
        Value::Float(f) => assert!(
            (f - expected).abs() < 1e-9,
            "{src} => {f}, expected {expected}"
        ),
        other => panic!("{src}: expected Float({expected}), got {other:?}"),
    }
}

fn assert_bool(src: &str, expected: bool) {
    match ev(src) {
        Value::Bool(b) => assert_eq!(b, expected, "{src} => {b}, expected {expected}"),
        other => panic!("{src}: expected Bool({expected}), got {other:?}"),
    }
}

fn assert_str(src: &str, expected: &str) {
    match ev(src) {
        Value::Str(s) => assert_eq!(s, expected, "{src} => {s:?}"),
        other => panic!("{src}: expected Str({expected:?}), got {other:?}"),
    }
}

// ----- basic arithmetic -----

#[test]
fn test_addition() {
    assert_int("2 + 2", 4);
}

#[test]
fn test_subtraction() {
    assert_int("10 - 3", 7);
}

#[test]
fn test_multiplication() {
    assert_int("2 * 3", 6);
}

#[test]
fn test_true_division_returns_float() {
    assert_float("10 / 4", 2.5);
    // Even when evenly divisible, `/` returns float.
    assert_float("8 / 2", 4.0);
}

#[test]
fn test_floor_division_positive() {
    assert_int("10 // 3", 3);
}

#[test]
fn test_power() {
    assert_int("2 ** 10", 1024);
}

#[test]
fn test_modulo_positive() {
    assert_int("10 % 3", 1);
}

// ----- Python-compatible floor div / modulo with negatives -----

#[test]
fn test_floor_div_negative_dividend() {
    assert_int("-7 // 2", -4);
}

#[test]
fn test_floor_div_negative_divisor() {
    assert_int("7 // -2", -4);
}

#[test]
fn test_floor_div_both_negative() {
    assert_int("-7 // -2", 3);
}

#[test]
fn test_modulo_negative_dividend() {
    assert_int("-7 % 3", 2);
}

#[test]
fn test_modulo_negative_divisor() {
    assert_int("7 % -3", -2);
}

#[test]
fn test_modulo_both_negative() {
    assert_int("-7 % -3", -1);
}

// ----- type preservation / coercion -----

#[test]
fn test_int_op_float_is_float() {
    assert_float("2 + 2.0", 4.0);
}

#[test]
fn test_bool_op_bool_is_int() {
    // True + True == 2, and must be Int, not Bool.
    match ev("True + True") {
        Value::Int(2) => {}
        other => panic!("expected Int(2), got {other:?}"),
    }
}

#[test]
fn test_integer_tier_promotion_to_bigint() {
    // 2 ** 100 overflows i64 and i128 — must become a BigInt, never wrap.
    match ev("2 ** 100") {
        Value::BigInt(_) => {}
        other => panic!("expected BigInt for 2**100, got {other:?}"),
    }
}

#[test]
fn test_integer_tier_promotion_to_i128() {
    // i64::MAX + 1 stays exact via Int128 (or BigInt), never wraps negative.
    match ev("9223372036854775807 + 1") {
        Value::Int128(n) => assert_eq!(n, 9_223_372_036_854_775_808i128),
        Value::BigInt(_) => {}
        other => panic!("expected Int128/BigInt, got {other:?}"),
    }
}

// ----- string operations -----

#[test]
fn test_string_concat() {
    assert_str("\"a\" + \"b\"", "ab");
}

#[test]
fn test_string_repeat() {
    assert_str("\"ab\" * 3", "ababab");
}

#[test]
fn test_string_repeat_int_times_str() {
    assert_str("3 * \"x\"", "xxx");
}

// ----- comparison -----

#[test]
fn test_comparisons() {
    assert_bool("1 < 2", true);
    assert_bool("2 < 1", false);
    assert_bool("1 == 1", true);
    assert_bool("1 != 2", true);
    assert_bool("2 >= 2", true);
    assert_bool("3 <= 2", false);
}

// ----- boolean / short-circuit -----

#[test]
fn test_and_returns_first_falsy() {
    assert_int("0 and 1", 0);
}

#[test]
fn test_and_returns_last_if_all_truthy() {
    assert_int("1 and 2", 2);
}

#[test]
fn test_or_returns_first_truthy() {
    assert_int("1 or 3", 1);
}

#[test]
fn test_or_returns_last_if_all_falsy() {
    assert_int("0 or 3", 3);
}

#[test]
fn test_not_returns_bool() {
    assert_bool("not 0", true);
    assert_bool("not 1", false);
}

// ----- bitwise / shift -----

#[test]
fn test_bitwise_ops() {
    assert_int("5 & 3", 1);
    assert_int("5 | 3", 7);
    assert_int("5 ^ 3", 6);
    assert_int("~5", -6);
}

#[test]
fn test_shift_ops() {
    assert_int("1 << 3", 8);
    assert_int("8 >> 2", 2);
}

// ----- unary -----

#[test]
fn test_unary_neg_and_pos() {
    assert_int("-5", -5);
    assert_int("+5", 5);
}

// ----- ternary -----

#[test]
fn test_ternary_true_branch() {
    assert_int("1 if True else 2", 1);
}

#[test]
fn test_ternary_false_branch() {
    assert_int("1 if False else 2", 2);
}

// ----- membership -----

#[test]
fn test_in_substring() {
    assert_bool("\"ab\" in \"abc\"", true);
    assert_bool("\"z\" in \"abc\"", false);
}

#[test]
fn test_in_list() {
    assert_bool("1 in [1, 2, 3]", true);
    assert_bool("9 in [1, 2, 3]", false);
}

#[test]
fn test_not_in_list() {
    assert_bool("9 not in [1, 2, 3]", true);
}

#[test]
fn test_in_dict_key() {
    assert_bool("\"a\" in {\"a\": 1}", true);
    assert_bool("\"b\" in {\"a\": 1}", false);
}

// ----- identity -----

#[test]
fn test_identity_ops() {
    assert_bool("None is None", true);
    assert_bool("True is True", true);
    assert_bool("1 is 1", true);
    assert_bool("1 is not 2", true);
}

// ----- names -----

#[test]
fn test_name_lookup() {
    let mut names = HashMap::new();
    names.insert("x".to_string(), Value::Int(5));
    match ev_with("x + 1", &names) {
        Value::Int(6) => {}
        other => panic!("expected Int(6), got {other:?}"),
    }
}

#[test]
fn test_builtin_names_always_available() {
    let names: HashMap<String, Value> = HashMap::new();
    assert!(matches!(
        ev_with("True", &names),
        Value::Bool(true)
    ));
    assert!(matches!(ev_with("None", &names), Value::None));
}

#[test]
fn test_undefined_name_errors() {
    match ev_res("nope") {
        Err(EvalError::NameNotDefined { name, .. }) => assert_eq!(name, "nope"),
        other => panic!("expected NameNotDefined, got {other:?}"),
    }
}

// ----- truthiness -----

#[test]
fn test_value_is_truthy_primitives() {
    assert!(value_is_truthy(&Value::Bool(true)));
    assert!(!value_is_truthy(&Value::Bool(false)));
    assert!(!value_is_truthy(&Value::Int(0)));
    assert!(value_is_truthy(&Value::Int(1)));
    assert!(!value_is_truthy(&Value::Float(0.0)));
    assert!(value_is_truthy(&Value::Float(1.5)));
    assert!(!value_is_truthy(&Value::Str(String::new())));
    assert!(value_is_truthy(&Value::Str("x".to_string())));
    assert!(!value_is_truthy(&Value::None));
}

#[test]
fn test_value_is_truthy_collections() {
    assert!(!value_is_truthy(&Value::List(vec![])));
    assert!(value_is_truthy(&Value::List(vec![Value::Int(1)])));
    assert!(!value_is_truthy(&Value::Tuple(vec![])));
    assert!(!value_is_truthy(&Value::Dict(vec![])));
    assert!(!value_is_truthy(&Value::Set(vec![])));
}

// ----- error cases -----

#[test]
fn test_floor_div_by_zero_errors() {
    match ev_res("1 // 0") {
        Err(EvalError::InvalidExpression(_)) => {}
        other => panic!("expected InvalidExpression for //0, got {other:?}"),
    }
}

#[test]
fn test_modulo_by_zero_errors() {
    match ev_res("1 % 0") {
        Err(EvalError::InvalidExpression(_)) => {}
        other => panic!("expected InvalidExpression for %0, got {other:?}"),
    }
}

#[test]
fn test_true_division_by_zero_errors() {
    match ev_res("1 / 0") {
        Err(EvalError::InvalidExpression(_)) => {}
        other => panic!("expected InvalidExpression for /0, got {other:?}"),
    }
}

#[test]
fn test_power_too_high_errors() {
    // exponent beyond MAX_POWER must raise NumberTooHigh.
    let src = format!("2 ** {}", MAX_POWER + 1);
    let names: HashMap<String, Value> = HashMap::new();
    let toks = tokenize(&src).unwrap();
    let ast = parse(&toks, &src).unwrap();
    match eval_expr(&ast, &names, &src) {
        Err(EvalError::NumberTooHigh(_)) => {}
        other => panic!("expected NumberTooHigh, got {other:?}"),
    }
}

#[test]
fn test_shift_too_high_errors() {
    let src = format!("1 << {}", MAX_SHIFT + 1);
    let names: HashMap<String, Value> = HashMap::new();
    let toks = tokenize(&src).unwrap();
    let ast = parse(&toks, &src).unwrap();
    match eval_expr(&ast, &names, &src) {
        Err(EvalError::NumberTooHigh(_)) => {}
        other => panic!("expected NumberTooHigh, got {other:?}"),
    }
}

// ----- collection literals -----

#[test]
fn test_list_literal_evaluates() {
    match ev("[1, 2, 3]") {
        Value::List(items) => {
            assert_eq!(items.len(), 3);
            assert!(matches!(items[0], Value::Int(1)));
        }
        other => panic!("expected List, got {other:?}"),
    }
}

#[test]
fn test_dict_literal_evaluates() {
    match ev("{\"a\": 1}") {
        Value::Dict(pairs) => assert_eq!(pairs.len(), 1),
        other => panic!("expected Dict, got {other:?}"),
    }
}

#[test]
fn test_tuple_literal_evaluates() {
    match ev("(1, 2)") {
        Value::Tuple(items) => assert_eq!(items.len(), 2),
        other => panic!("expected Tuple, got {other:?}"),
    }
}
