//! End-to-end integration tests for the Rust core.
//!
//! These drive the full pipeline `tokenize -> parse -> eval_expr` the same way
//! the PyO3 `fast_eval` boundary will, verifying that the units compose
//! correctly for representative expressions, error propagation across stages,
//! and the documented limitations (chained comparison, subscript, f-string,
//! attribute access).

use super::super::*;
use std::collections::HashMap;

/// Full pipeline against an empty names map.
fn run(src: &str) -> EvalResult<Value> {
    run_with(src, &HashMap::new())
}

/// Full pipeline against a supplied names map.
fn run_with(src: &str, names: &HashMap<String, Value>) -> EvalResult<Value> {
    let toks = tokenize(src)?;
    let ast = parse(&toks, src)?;
    eval_expr(&ast, names, src)
}

fn ok(src: &str) -> Value {
    run(src).unwrap_or_else(|e| panic!("pipeline({src:?}) failed: {e:?}"))
}

#[test]
fn test_arithmetic_pipeline() {
    assert!(matches!(ok("2 + 2"), Value::Int(4)));
    assert!(matches!(ok("2 + 2 * 3"), Value::Int(8)));
    assert!(matches!(ok("(2 + 2) * 3"), Value::Int(12)));
    assert!(matches!(ok("2 ** 10"), Value::Int(1024)));
}

#[test]
fn test_mixed_numeric_pipeline() {
    match ok("10 / 3") {
        Value::Float(f) => assert!((f - 3.333_333_333_3).abs() < 1e-6),
        other => panic!("10/3 => {other:?}"),
    }
}

#[test]
fn test_comparison_and_bool_pipeline() {
    assert!(matches!(ok("1 < 2"), Value::Bool(true)));
    assert!(matches!(ok("True and False"), Value::Bool(false)));
    assert!(matches!(ok("True or False"), Value::Bool(true)));
    assert!(matches!(ok("not True"), Value::Bool(false)));
}

#[test]
fn test_builtin_call_pipeline() {
    assert!(matches!(ok("abs(-5)"), Value::Int(5)));
    assert!(matches!(ok("min(3, 1, 2)"), Value::Int(1)));
    assert!(matches!(ok("max(3, 1, 2)"), Value::Int(3)));
    assert!(matches!(ok("len(\"abc\")"), Value::Int(3)));
    assert!(matches!(ok("sum([1, 2, 3])"), Value::Int(6)));
}

#[test]
fn test_builtin_round_pipeline() {
    assert!(matches!(ok("round(2.5)"), Value::Int(2)));
    assert!(matches!(ok("round(3.5)"), Value::Int(4)));
}

#[test]
fn test_conversion_builtins_pipeline() {
    assert!(matches!(ok("int(\"42\")"), Value::Int(42)));
    match ok("float(\"3.14\")") {
        Value::Float(f) => assert!((f - 3.14).abs() < 1e-9),
        other => panic!("float('3.14') => {other:?}"),
    }
    match ok("str(42)") {
        Value::Str(s) => assert_eq!(s, "42"),
        other => panic!("str(42) => {other:?}"),
    }
}

#[test]
fn test_names_pipeline() {
    let mut names = HashMap::new();
    names.insert("x".to_string(), Value::Int(5));
    names.insert("y".to_string(), Value::Int(2));
    assert!(matches!(run_with("x + y", &names).unwrap(), Value::Int(7)));
    assert!(matches!(run_with("x * y", &names).unwrap(), Value::Int(10)));
}

#[test]
fn test_ternary_pipeline() {
    assert!(matches!(ok("1 if True else 2"), Value::Int(1)));
    assert!(matches!(ok("1 if False else 2"), Value::Int(2)));
}

#[test]
fn test_membership_pipeline() {
    assert!(matches!(ok("\"ab\" in \"abc\""), Value::Bool(true)));
    assert!(matches!(ok("1 in [1, 2, 3]"), Value::Bool(true)));
    assert!(matches!(ok("\"a\" in {\"a\": 1}"), Value::Bool(true)));
}

#[test]
fn test_negative_floor_div_and_mod_pipeline() {
    assert!(matches!(ok("-7 // 2"), Value::Int(-4)));
    assert!(matches!(ok("-7 % 3"), Value::Int(2)));
}

#[test]
fn test_bigint_pipeline_no_panic() {
    // Large powers must produce a BigInt and never panic / wrap.
    match ok("2 ** 200") {
        Value::BigInt(_) => {}
        other => panic!("2**200 => {other:?}"),
    }
}

// ----- documented limitations: explicit errors, no silent fallback -----

#[test]
fn test_chained_comparison_is_error() {
    match run("1 < 2 < 3") {
        Err(EvalError::InvalidExpression(msg)) => assert!(msg.contains("chained")),
        other => panic!("chained comparison should error, got {other:?}"),
    }
}

#[test]
fn test_subscript_is_error() {
    assert!(run("[1, 2, 3][0]").is_err(), "subscript must error");
}

#[test]
fn test_undefined_name_propagates() {
    match run("missing + 1") {
        Err(EvalError::NameNotDefined { name, .. }) => assert_eq!(name, "missing"),
        other => panic!("expected NameNotDefined, got {other:?}"),
    }
}

#[test]
fn test_undefined_function_propagates() {
    match run("nope(1)") {
        Err(EvalError::FunctionNotDefined { func_name, .. }) => {
            assert_eq!(func_name, "nope");
        }
        other => panic!("expected FunctionNotDefined, got {other:?}"),
    }
}

#[test]
fn test_division_by_zero_propagates() {
    assert!(matches!(
        run("1 / 0"),
        Err(EvalError::InvalidExpression(_))
    ));
    assert!(matches!(
        run("1 // 0"),
        Err(EvalError::InvalidExpression(_))
    ));
    assert!(matches!(
        run("1 % 0"),
        Err(EvalError::InvalidExpression(_))
    ));
}

#[test]
fn test_representative_expression_matrix() {
    // A small parity-style matrix exercising every operator category through
    // the full pipeline at once.
    let cases: &[(&str, Value)] = &[
        ("1 + 2 - 3", Value::Int(0)),
        ("2 * 3 + 1", Value::Int(7)),
        ("10 // 3", Value::Int(3)),
        ("10 % 3", Value::Int(1)),
        ("5 & 3", Value::Int(1)),
        ("5 | 2", Value::Int(7)),
        ("1 << 4", Value::Int(16)),
        ("32 >> 2", Value::Int(8)),
        ("~0", Value::Int(-1)),
        ("1 == 1", Value::Bool(true)),
        ("1 != 1", Value::Bool(false)),
        ("3 > 2", Value::Bool(true)),
        ("0 or 5", Value::Int(5)),
        ("7 and 0", Value::Int(0)),
    ];
    for (src, expected) in cases {
        let got = ok(src);
        let matches = match (&got, expected) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            _ => false,
        };
        assert!(matches, "{src} => {got:?}, expected {expected:?}");
    }
}
