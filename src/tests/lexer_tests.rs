//! Unit tests for the lexer (`tokenize`).
//!
//! These tests exercise `tokenize(input: &str) -> EvalResult<Vec<Token>>`
//! and assert on the produced `Token` stream. They assume the types from
//! `unit_01_types` (`Token`, `EvalError`, `EvalResult`) and the lexer from
//! `unit_02_lexer` are in scope via the parent module (`use super::super::*;`).

use super::super::*;

/// Helper: tokenize and unwrap, panicking with the error on failure.
fn lex(src: &str) -> Vec<Token> {
    tokenize(src).unwrap_or_else(|e| panic!("tokenize({src:?}) failed: {e:?}"))
}

/// Helper: tokenize, dropping a trailing `Eof` if present, for easier matching.
fn lex_no_eof(src: &str) -> Vec<Token> {
    let mut toks = lex(src);
    if matches!(toks.last(), Some(Token::Eof)) {
        toks.pop();
    }
    toks
}

#[test]
fn test_integer_literal_i64() {
    assert_eq!(lex_no_eof("42"), vec![Token::IntLit(42)]);
    assert_eq!(lex_no_eof("0"), vec![Token::IntLit(0)]);
    assert_eq!(lex_no_eof("1000000"), vec![Token::IntLit(1_000_000)]);
}

#[test]
fn test_hex_octal_binary_literals() {
    assert_eq!(lex_no_eof("0x1F"), vec![Token::IntLit(0x1F)]);
    assert_eq!(lex_no_eof("0o17"), vec![Token::IntLit(0o17)]);
    assert_eq!(lex_no_eof("0b1010"), vec![Token::IntLit(0b1010)]);
}

#[test]
fn test_float_literal() {
    assert_eq!(lex_no_eof("3.14"), vec![Token::FloatLit(3.14)]);
    assert_eq!(lex_no_eof("1e10"), vec![Token::FloatLit(1e10)]);
    assert_eq!(lex_no_eof("1.5e-3"), vec![Token::FloatLit(1.5e-3)]);
}

#[test]
fn test_big_integer_literal_promotes() {
    // A value far beyond i64 range must not be lexed as IntLit.
    let toks = lex_no_eof("99999999999999999999999999999999");
    assert_eq!(toks.len(), 1);
    match &toks[0] {
        Token::BigIntLit(_) => {}
        other => panic!("expected BigIntLit, got {other:?}"),
    }
}

#[test]
fn test_i128_literal_promotes() {
    // Beyond i64 (>9.2e18) but within i128.
    let toks = lex_no_eof("9223372036854775808"); // i64::MAX + 1
    assert_eq!(toks.len(), 1);
    match &toks[0] {
        Token::Int128Lit(_) | Token::BigIntLit(_) => {}
        other => panic!("expected Int128Lit/BigIntLit, got {other:?}"),
    }
}

#[test]
fn test_string_literal_double_quote() {
    assert_eq!(lex_no_eof("\"hello\""), vec![Token::StrLit("hello".to_string())]);
}

#[test]
fn test_string_literal_single_quote() {
    assert_eq!(lex_no_eof("'world'"), vec![Token::StrLit("world".to_string())]);
}

#[test]
fn test_string_escape_sequences() {
    assert_eq!(
        lex_no_eof("\"a\\nb\\tc\""),
        vec![Token::StrLit("a\nb\tc".to_string())]
    );
    assert_eq!(
        lex_no_eof("'it\\'s'"),
        vec![Token::StrLit("it's".to_string())]
    );
    assert_eq!(
        lex_no_eof("\"back\\\\slash\""),
        vec![Token::StrLit("back\\slash".to_string())]
    );
}

#[test]
fn test_keywords_no_identifier() {
    assert_eq!(lex_no_eof("True"), vec![Token::BoolLit(true)]);
    assert_eq!(lex_no_eof("False"), vec![Token::BoolLit(false)]);
    assert_eq!(lex_no_eof("None"), vec![Token::NoneLit]);
    assert_eq!(lex_no_eof("and"), vec![Token::And]);
    assert_eq!(lex_no_eof("or"), vec![Token::Or]);
    assert_eq!(lex_no_eof("not"), vec![Token::Not]);
    assert_eq!(lex_no_eof("in"), vec![Token::In]);
    assert_eq!(lex_no_eof("is"), vec![Token::Is]);
    assert_eq!(lex_no_eof("if"), vec![Token::If]);
    assert_eq!(lex_no_eof("else"), vec![Token::Else]);
}

#[test]
fn test_identifier() {
    assert_eq!(lex_no_eof("x"), vec![Token::Ident("x".to_string())]);
    assert_eq!(
        lex_no_eof("my_var"),
        vec![Token::Ident("my_var".to_string())]
    );
    // A name that merely contains a keyword substring is still an identifier.
    assert_eq!(
        lex_no_eof("android"),
        vec![Token::Ident("android".to_string())]
    );
}

#[test]
fn test_arithmetic_operators() {
    assert_eq!(
        lex_no_eof("+ - * / // % **"),
        vec![
            Token::Plus,
            Token::Minus,
            Token::Star,
            Token::Slash,
            Token::SlashSlash,
            Token::Percent,
            Token::StarStar,
        ]
    );
}

#[test]
fn test_bitwise_and_shift_operators() {
    assert_eq!(
        lex_no_eof("& | ^ ~ << >>"),
        vec![
            Token::Amp,
            Token::Pipe,
            Token::Caret,
            Token::Tilde,
            Token::LtLt,
            Token::GtGt,
        ]
    );
}

#[test]
fn test_comparison_operators() {
    assert_eq!(
        lex_no_eof("== != < > <= >="),
        vec![
            Token::EqEq,
            Token::BangEq,
            Token::Lt,
            Token::Gt,
            Token::LtEq,
            Token::GtEq,
        ]
    );
}

#[test]
fn test_punctuation() {
    assert_eq!(
        lex_no_eof("( ) [ ] { } , :"),
        vec![
            Token::LParen,
            Token::RParen,
            Token::LBracket,
            Token::RBracket,
            Token::LBrace,
            Token::RBrace,
            Token::Comma,
            Token::Colon,
        ]
    );
}

#[test]
fn test_whitespace_is_skipped() {
    assert_eq!(
        lex_no_eof("  1   +    2  "),
        vec![Token::IntLit(1), Token::Plus, Token::IntLit(2)]
    );
}

#[test]
fn test_compound_expression_token_stream() {
    assert_eq!(
        lex_no_eof("2 + 2 * 3"),
        vec![
            Token::IntLit(2),
            Token::Plus,
            Token::IntLit(2),
            Token::Star,
            Token::IntLit(3),
        ]
    );
}

#[test]
fn test_tokenize_appends_eof() {
    let toks = lex("1");
    assert_eq!(toks.last(), Some(&Token::Eof));
}

#[test]
fn test_empty_input_tokenizes_to_eof_only() {
    let toks = tokenize("").unwrap();
    // Either just Eof, or empty — accept either, but no value tokens.
    assert!(toks.iter().all(|t| matches!(t, Token::Eof)));
}

#[test]
fn test_unrecognized_character_errors() {
    let res = tokenize("1 $ 2");
    match res {
        Err(EvalError::InvalidExpression(_)) => {}
        other => panic!("expected InvalidExpression for '$', got {other:?}"),
    }
}

#[test]
fn test_distinguish_slash_and_slashslash() {
    assert_eq!(lex_no_eof("/"), vec![Token::Slash]);
    assert_eq!(lex_no_eof("//"), vec![Token::SlashSlash]);
}

#[test]
fn test_distinguish_star_and_starstar() {
    assert_eq!(lex_no_eof("*"), vec![Token::Star]);
    assert_eq!(lex_no_eof("**"), vec![Token::StarStar]);
}

#[test]
fn test_distinguish_lt_lteq_ltlt() {
    assert_eq!(lex_no_eof("<"), vec![Token::Lt]);
    assert_eq!(lex_no_eof("<="), vec![Token::LtEq]);
    assert_eq!(lex_no_eof("<<"), vec![Token::LtLt]);
}
