//! Unit tests for the parser (`parse`).
//!
//! Exercises `parse(tokens: &[Token], expr: &str) -> EvalResult<Expr>` by
//! tokenizing source, parsing it, and asserting on the produced `Expr` AST
//! shape — including operator precedence, associativity, literals, calls,
//! and the explicit rejection of unsupported constructs (chained comparison,
//! subscript). Types come from `unit_01_types`.

use super::super::*;

/// Tokenize then parse, panicking on lexer/parser error.
fn parse_ok(src: &str) -> Expr {
    let toks = tokenize(src).unwrap_or_else(|e| panic!("tokenize({src:?}) failed: {e:?}"));
    parse(&toks, src).unwrap_or_else(|e| panic!("parse({src:?}) failed: {e:?}"))
}

/// Tokenize then parse, returning the Result (errors expected in some tests).
fn parse_res(src: &str) -> EvalResult<Expr> {
    let toks = tokenize(src)?;
    parse(&toks, src)
}

#[test]
fn test_parse_integer_literal() {
    match parse_ok("42") {
        Expr::Literal(Value::Int(42)) => {}
        other => panic!("expected Literal(Int(42)), got {other:?}"),
    }
}

#[test]
fn test_parse_float_literal() {
    match parse_ok("3.5") {
        Expr::Literal(Value::Float(f)) => assert_eq!(f, 3.5),
        other => panic!("expected Literal(Float), got {other:?}"),
    }
}

#[test]
fn test_parse_string_literal() {
    match parse_ok("'hi'") {
        Expr::Literal(Value::Str(s)) => assert_eq!(s, "hi"),
        other => panic!("expected Literal(Str), got {other:?}"),
    }
}

#[test]
fn test_parse_bool_and_none_literals() {
    assert!(matches!(parse_ok("True"), Expr::Literal(Value::Bool(true))));
    assert!(matches!(parse_ok("False"), Expr::Literal(Value::Bool(false))));
    assert!(matches!(parse_ok("None"), Expr::Literal(Value::None)));
}

#[test]
fn test_parse_name() {
    match parse_ok("x") {
        Expr::Name(n) => assert_eq!(n, "x"),
        other => panic!("expected Name, got {other:?}"),
    }
}

#[test]
fn test_parse_simple_binop_add() {
    match parse_ok("1 + 2") {
        Expr::BinOp { op: BinOpKind::Add, left, right } => {
            assert!(matches!(*left, Expr::Literal(Value::Int(1))));
            assert!(matches!(*right, Expr::Literal(Value::Int(2))));
        }
        other => panic!("expected BinOp Add, got {other:?}"),
    }
}

#[test]
fn test_precedence_mul_binds_tighter_than_add() {
    // 2 + 2 * 3  =>  Add(2, Mul(2, 3))
    match parse_ok("2 + 2 * 3") {
        Expr::BinOp { op: BinOpKind::Add, left, right } => {
            assert!(matches!(*left, Expr::Literal(Value::Int(2))));
            match *right {
                Expr::BinOp { op: BinOpKind::Mul, .. } => {}
                other => panic!("expected right side Mul, got {other:?}"),
            }
        }
        other => panic!("expected top-level Add, got {other:?}"),
    }
}

#[test]
fn test_precedence_pow_binds_tighter_than_unary_and_is_right_assoc() {
    // 2 ** 3 ** 2 => Pow(2, Pow(3, 2)) (right-associative)
    match parse_ok("2 ** 3 ** 2") {
        Expr::BinOp { op: BinOpKind::Pow, left, right } => {
            assert!(matches!(*left, Expr::Literal(Value::Int(2))));
            match *right {
                Expr::BinOp { op: BinOpKind::Pow, .. } => {}
                other => panic!("expected right-assoc Pow nest, got {other:?}"),
            }
        }
        other => panic!("expected top-level Pow, got {other:?}"),
    }
}

#[test]
fn test_parens_override_precedence() {
    // (2 + 2) * 3 => Mul(Add(2,2), 3)
    match parse_ok("(2 + 2) * 3") {
        Expr::BinOp { op: BinOpKind::Mul, left, right } => {
            assert!(matches!(*left, Expr::BinOp { op: BinOpKind::Add, .. }));
            assert!(matches!(*right, Expr::Literal(Value::Int(3))));
        }
        other => panic!("expected top-level Mul, got {other:?}"),
    }
}

#[test]
fn test_parse_unary_neg() {
    match parse_ok("-5") {
        Expr::UnaryOp { op: UnaryOpKind::Neg, operand } => {
            assert!(matches!(*operand, Expr::Literal(Value::Int(5))));
        }
        other => panic!("expected UnaryOp Neg, got {other:?}"),
    }
}

#[test]
fn test_parse_unary_invert() {
    match parse_ok("~5") {
        Expr::UnaryOp { op: UnaryOpKind::BitNot, .. } => {}
        other => panic!("expected UnaryOp BitNot, got {other:?}"),
    }
}

#[test]
fn test_parse_comparison() {
    match parse_ok("1 < 2") {
        Expr::Compare { op: CmpOpKind::Lt, left, right } => {
            assert!(matches!(*left, Expr::Literal(Value::Int(1))));
            assert!(matches!(*right, Expr::Literal(Value::Int(2))));
        }
        other => panic!("expected Compare Lt, got {other:?}"),
    }
}

#[test]
fn test_parse_boolop_and() {
    match parse_ok("True and False") {
        Expr::BoolOp { op: BoolOpKind::And, values } => {
            assert_eq!(values.len(), 2);
        }
        other => panic!("expected BoolOp And, got {other:?}"),
    }
}

#[test]
fn test_parse_boolop_or() {
    match parse_ok("1 or 0") {
        Expr::BoolOp { op: BoolOpKind::Or, values } => assert_eq!(values.len(), 2),
        other => panic!("expected BoolOp Or, got {other:?}"),
    }
}

#[test]
fn test_parse_ternary_ifexp() {
    match parse_ok("1 if True else 2") {
        Expr::IfExp { test, body, orelse } => {
            assert!(matches!(*test, Expr::Literal(Value::Bool(true))));
            assert!(matches!(*body, Expr::Literal(Value::Int(1))));
            assert!(matches!(*orelse, Expr::Literal(Value::Int(2))));
        }
        other => panic!("expected IfExp, got {other:?}"),
    }
}

#[test]
fn test_parse_function_call() {
    match parse_ok("abs(-5)") {
        Expr::Call { func, args } => {
            assert_eq!(func, "abs");
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected Call, got {other:?}"),
    }
}

#[test]
fn test_parse_call_multiple_args() {
    match parse_ok("min(1, 2, 3)") {
        Expr::Call { func, args } => {
            assert_eq!(func, "min");
            assert_eq!(args.len(), 3);
        }
        other => panic!("expected Call, got {other:?}"),
    }
}

#[test]
fn test_parse_list_literal() {
    match parse_ok("[1, 2, 3]") {
        Expr::List(items) => assert_eq!(items.len(), 3),
        other => panic!("expected List, got {other:?}"),
    }
}

#[test]
fn test_parse_empty_list() {
    match parse_ok("[]") {
        Expr::List(items) => assert!(items.is_empty()),
        other => panic!("expected empty List, got {other:?}"),
    }
}

#[test]
fn test_parse_tuple_literal() {
    match parse_ok("(1, 2)") {
        Expr::Tuple(items) => assert_eq!(items.len(), 2),
        other => panic!("expected Tuple, got {other:?}"),
    }
}

#[test]
fn test_parse_single_element_tuple() {
    match parse_ok("(1,)") {
        Expr::Tuple(items) => assert_eq!(items.len(), 1),
        other => panic!("expected 1-tuple, got {other:?}"),
    }
}

#[test]
fn test_parse_dict_literal() {
    match parse_ok("{'a': 1, 'b': 2}") {
        Expr::Dict { keys, values } => {
            assert_eq!(keys.len(), 2);
            assert_eq!(values.len(), 2);
        }
        other => panic!("expected Dict, got {other:?}"),
    }
}

#[test]
fn test_parse_set_literal() {
    match parse_ok("{1, 2, 3}") {
        Expr::Set(items) => assert_eq!(items.len(), 3),
        other => panic!("expected Set, got {other:?}"),
    }
}

#[test]
fn test_parse_membership_in() {
    match parse_ok("1 in [1, 2]") {
        Expr::BinOp { op: BinOpKind::In, .. } => {}
        other => panic!("expected BinOp In, got {other:?}"),
    }
}

#[test]
fn test_parse_not_in() {
    match parse_ok("3 not in [1, 2]") {
        Expr::BinOp { op: BinOpKind::NotIn, .. } => {}
        other => panic!("expected BinOp NotIn, got {other:?}"),
    }
}

#[test]
fn test_parse_is_and_is_not() {
    assert!(matches!(
        parse_ok("None is None"),
        Expr::BinOp { op: BinOpKind::Is, .. }
    ));
    assert!(matches!(
        parse_ok("1 is not 2"),
        Expr::BinOp { op: BinOpKind::IsNot, .. }
    ));
}

#[test]
fn test_chained_comparison_rejected() {
    let res = parse_res("1 < 2 < 3");
    match res {
        Err(EvalError::InvalidExpression(msg)) => {
            assert!(
                msg.contains("chained"),
                "message should mention chained comparisons, got {msg:?}"
            );
        }
        other => panic!("expected InvalidExpression for chained comparison, got {other:?}"),
    }
}

#[test]
fn test_subscript_rejected() {
    // x[0] is not supported and must error at parse time.
    let res = parse_res("x[0]");
    assert!(
        res.is_err(),
        "subscript/indexing must not parse successfully"
    );
}

#[test]
fn test_shift_lower_precedence_than_add() {
    // 1 << 2 + 3 => LShift(1, Add(2,3))
    match parse_ok("1 << 2 + 3") {
        Expr::BinOp { op: BinOpKind::LShift, left, right } => {
            assert!(matches!(*left, Expr::Literal(Value::Int(1))));
            assert!(matches!(*right, Expr::BinOp { op: BinOpKind::Add, .. }));
        }
        other => panic!("expected top-level LShift, got {other:?}"),
    }
}
