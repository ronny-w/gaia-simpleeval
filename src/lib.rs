// @implements: G001, G002, G003, G004, G005, G006, G007, G008, G009, G010, G011, G012, G013, G014
// @spec: §3, §4, §5, §6, §7, §8
// gaia-simpleeval: high-performance drop-in replacement for the `simpleeval`
// Python library. Single-file Rust core exposed via PyO3.
//
// Sections (in order):
//   1. Imports and crate-level attributes
//   2. Constants
//   3. Token enum
//   4. Operator / AST enums (BinOpKind, UnaryOpKind, BoolOpKind, CmpOpKind, Expr)
//   5. Value enum
//   6. EvalError enum + EvalResult
//   7. Lexer (tokenize)
//   8. Parser (parse)
//   9. Native builtins (call_builtin)
//  10. Evaluator (eval_expr, value_is_truthy)
//  11. PyO3 boundary (value_to_pyobject, eval_error_to_pyerr, pydict_to_names, fast_eval)
//  12. #[pymodule] registration

#![allow(clippy::needless_return)]

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Signed, ToPrimitive, Zero};
use std::collections::HashMap;
use std::fmt;

#[cfg(not(test))]
use pyo3::prelude::*;
#[cfg(not(test))]
use pyo3::types::{PyDict, PyDictMethods, PyList, PyTuple};

// ───────────────────────────────────────────────────────────────────────────
// 2. Constants
// ───────────────────────────────────────────────────────────────────────────

pub const MAX_STRING_LENGTH: usize = 100_000;
pub const MAX_COMPREHENSION_LENGTH: usize = 10_000;
pub const MAX_POWER: i64 = 4_000_000;
pub const MAX_SHIFT: i64 = 10_000;
pub const MAX_SHIFT_BASE: f64 = f64::MAX;

// ───────────────────────────────────────────────────────────────────────────
// 3. Token enum
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    IntLit(i64),
    Int128Lit(i128),
    BigIntLit(BigInt),
    FloatLit(f64),
    StrLit(String),
    BoolLit(bool),
    NoneLit,
    // Identifiers
    Ident(String),
    // Arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    SlashSlash,
    Percent,
    StarStar,
    // Bitwise
    Amp,
    Pipe,
    Caret,
    Tilde,
    // Shift
    LtLt,
    GtGt,
    // Comparison
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    // Boolean keywords
    And,
    Or,
    Not,
    // Membership / identity keywords
    In,
    Is,
    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    // Conditional
    If,
    Else,
    // End of input
    Eof,
}

// ───────────────────────────────────────────────────────────────────────────
// 4. Operator / AST enums
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    LShift,
    RShift,
    In,
    NotIn,
    Is,
    IsNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOpKind {
    Neg,
    Pos,
    BitNot,
    // `not x` boolean negation. SPEC §6.4 lists this variant; the unit_03
    // validation expects `not a` to parse as UnaryOp(Not).
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOpKind {
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOpKind {
    Eq,
    NotEq,
    Lt,
    Gt,
    LtE,
    GtE,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Value),
    Name(String),
    BinOp {
        op: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOpKind,
        operand: Box<Expr>,
    },
    BoolOp {
        op: BoolOpKind,
        values: Vec<Expr>,
    },
    Compare {
        left: Box<Expr>,
        op: CmpOpKind,
        right: Box<Expr>,
    },
    IfExp {
        test: Box<Expr>,
        body: Box<Expr>,
        orelse: Box<Expr>,
    },
    Call {
        func: String,
        args: Vec<Expr>,
    },
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    Dict {
        keys: Vec<Expr>,
        values: Vec<Expr>,
    },
    Set(Vec<Expr>),
}

// ───────────────────────────────────────────────────────────────────────────
// 4b. Expression-node arena (G014)
// ───────────────────────────────────────────────────────────────────────────
//
// G014: arena allocation for expression nodes.
//
// A recursive-descent parse of even a small expression allocates one heap node
// per interior `Box<Expr>` (`2 + 3 * 4` is three BinOp children, etc.). Under
// the default global allocator each `Box::new` is an individual `malloc`, and
// each is individually `free`d when the tree drops. For the parse-heavy hot
// path (every `fast_eval` miss tokenizes + parses) that is dozens of tiny
// allocator round-trips per expression.
//
// `ExprArena` is a typed region/bump allocator for `Expr` nodes. Nodes are
// stored in fixed-size chunks (`Vec<Box<[…]>>`-style backing); a node is
// allocated by writing into the current chunk and bumping an index, with no
// per-node allocator call once a chunk is live. The whole arena is freed in
// one shot when it drops — there is no per-node `free`, which is where the
// "equal or better vs Box" win comes from for typical expression sizes.
//
// Safety / no use-after-free: the arena owns every node for its entire
// lifetime. Handles returned to the parser are plain indices (`ExprRef`) into
// the arena, never raw pointers, so there is no aliasing and nothing can
// dangle — an `ExprRef` is only ever resolved against the arena that minted
// it. Chunks are boxed and never moved or reallocated after creation (we push
// a *new* chunk rather than growing an existing one), so existing nodes keep a
// stable address for the arena's lifetime even as it grows. This is what makes
// the structure miri/valgrind clean: no `unsafe`, no realloc-invalidated
// references, no manual frees.
//
// At the parse boundary the arena tree is materialized into the owning
// `Box<Expr>` shape the rest of the crate (evaluator, cache, tests) already
// expects, so G014 is a pure allocation-strategy change behind `parse`: the
// public `Expr` / `Box<Expr>` contract is unchanged.

/// Number of `Expr` slots per arena chunk. Sized so a typical expression
/// (well under this many nodes) lives entirely in the first chunk — a single
/// up-front allocation for the whole parse — while large expressions still
/// grow without ever reallocating earlier chunks.
const ARENA_CHUNK: usize = 64;

/// A stable handle to a node owned by an [`ExprArena`]. Indices, not pointers:
/// resolving one cannot dangle and carries no aliasing obligations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExprRef(usize);

/// Region allocator for `Expr` nodes used during parsing (G014).
///
/// Allocation is amortized O(1) with no per-node allocator call inside a live
/// chunk; the entire arena is released in one drop with no per-node free.
pub struct ExprArena {
    // Each chunk is a heap-stable buffer of nodes. We never grow a chunk in
    // place (which would reallocate and move its nodes); when the current chunk
    // fills we push a brand-new boxed chunk. Earlier chunks therefore keep a
    // fixed address for the arena's whole lifetime.
    chunks: Vec<Box<[Option<ArenaNode>]>>,
    // Total number of live nodes; also the next global index to hand out.
    len: usize,
}

/// Arena-internal node form. Mirrors `Expr` but refers to children by
/// `ExprRef` index instead of owning `Box<Expr>`, so no per-child heap node is
/// allocated while parsing.
#[derive(Debug)]
enum ArenaNode {
    Literal(Value),
    Name(String),
    BinOp {
        op: BinOpKind,
        left: ExprRef,
        right: ExprRef,
    },
    UnaryOp {
        op: UnaryOpKind,
        operand: ExprRef,
    },
    BoolOp {
        op: BoolOpKind,
        values: Vec<ExprRef>,
    },
    Compare {
        left: ExprRef,
        op: CmpOpKind,
        right: ExprRef,
    },
    IfExp {
        test: ExprRef,
        body: ExprRef,
        orelse: ExprRef,
    },
    Call {
        func: String,
        args: Vec<ExprRef>,
    },
    List(Vec<ExprRef>),
    Tuple(Vec<ExprRef>),
    Dict {
        keys: Vec<ExprRef>,
        values: Vec<ExprRef>,
    },
    Set(Vec<ExprRef>),
}

impl ExprArena {
    /// Create an arena pre-sized for an expression of roughly `node_hint`
    /// nodes. A single chunk is reserved up front so the common small-expression
    /// case performs exactly one heap allocation for the entire parse.
    fn with_node_hint(node_hint: usize) -> Self {
        let mut arena = ExprArena {
            chunks: Vec::with_capacity(node_hint / ARENA_CHUNK + 1),
            len: 0,
        };
        arena.push_chunk();
        arena
    }

    #[inline]
    fn push_chunk(&mut self) {
        // `Option<ArenaNode>` is non-Copy, so build the slice via a Vec.
        let mut v: Vec<Option<ArenaNode>> = Vec::with_capacity(ARENA_CHUNK);
        for _ in 0..ARENA_CHUNK {
            v.push(None);
        }
        self.chunks.push(v.into_boxed_slice());
    }

    /// Allocate a node, returning a stable handle. Amortized O(1): only the
    /// chunk-full case touches the allocator (one new chunk), and it never
    /// moves previously allocated nodes.
    #[inline]
    fn alloc(&mut self, node: ArenaNode) -> ExprRef {
        let idx = self.len;
        let chunk_i = idx / ARENA_CHUNK;
        let slot_i = idx % ARENA_CHUNK;
        if chunk_i == self.chunks.len() {
            self.push_chunk();
        }
        self.chunks[chunk_i][slot_i] = Some(node);
        self.len += 1;
        ExprRef(idx)
    }

    #[inline]
    fn get(&self, r: ExprRef) -> &ArenaNode {
        let idx = r.0;
        self.chunks[idx / ARENA_CHUNK][idx % ARENA_CHUNK]
            .as_ref()
            .expect("ExprRef resolved against arena that did not allocate it")
    }

    /// Number of nodes currently allocated in the arena.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the arena holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Materialize the arena subtree rooted at `r` into the owning `Box<Expr>`
    /// tree the evaluator/cache/tests consume. Done once per parse, at the
    /// boundary, so the arena win applies to the (hot) build phase while the
    /// rest of the crate keeps the stable `Expr` contract.
    fn materialize(&self, r: ExprRef) -> Expr {
        match self.get(r) {
            ArenaNode::Literal(v) => Expr::Literal(v.clone()),
            ArenaNode::Name(n) => Expr::Name(n.clone()),
            ArenaNode::BinOp { op, left, right } => Expr::BinOp {
                op: *op,
                left: Box::new(self.materialize(*left)),
                right: Box::new(self.materialize(*right)),
            },
            ArenaNode::UnaryOp { op, operand } => Expr::UnaryOp {
                op: *op,
                operand: Box::new(self.materialize(*operand)),
            },
            ArenaNode::BoolOp { op, values } => Expr::BoolOp {
                op: *op,
                values: values.iter().map(|c| self.materialize(*c)).collect(),
            },
            ArenaNode::Compare { left, op, right } => Expr::Compare {
                left: Box::new(self.materialize(*left)),
                op: *op,
                right: Box::new(self.materialize(*right)),
            },
            ArenaNode::IfExp { test, body, orelse } => Expr::IfExp {
                test: Box::new(self.materialize(*test)),
                body: Box::new(self.materialize(*body)),
                orelse: Box::new(self.materialize(*orelse)),
            },
            ArenaNode::Call { func, args } => Expr::Call {
                func: func.clone(),
                args: args.iter().map(|c| self.materialize(*c)).collect(),
            },
            ArenaNode::List(items) => {
                Expr::List(items.iter().map(|c| self.materialize(*c)).collect())
            }
            ArenaNode::Tuple(items) => {
                Expr::Tuple(items.iter().map(|c| self.materialize(*c)).collect())
            }
            ArenaNode::Dict { keys, values } => Expr::Dict {
                keys: keys.iter().map(|c| self.materialize(*c)).collect(),
                values: values.iter().map(|c| self.materialize(*c)).collect(),
            },
            ArenaNode::Set(items) => {
                Expr::Set(items.iter().map(|c| self.materialize(*c)).collect())
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 5. Value enum
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Int128(i128),
    BigInt(BigInt),
    Float(f64),
    Str(String),
    None,
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Set(Vec<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        value_equal(self, other)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            Value::Int(n) => write!(f, "{}", n),
            Value::Int128(n) => write!(f, "{}", n),
            Value::BigInt(n) => write!(f, "{}", n),
            Value::Float(x) => write!(f, "{}", format_float(*x)),
            Value::Str(s) => write!(f, "{}", s),
            Value::None => write!(f, "None"),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ReprDisplay(it))?;
                }
                write!(f, "]")
            }
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ReprDisplay(it))?;
                }
                if items.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            Value::Dict(pairs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", ReprDisplay(k), ReprDisplay(v))?;
                }
                write!(f, "}}")
            }
            Value::Set(items) => {
                if items.is_empty() {
                    return write!(f, "set()");
                }
                write!(f, "{{")?;
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ReprDisplay(it))?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Python-compatible str(float)/repr(float) formatting.
///
/// Python uses the shortest decimal string that round-trips to the same f64
/// (David Gay / Grisu style), switching to scientific notation when the
/// exponent is < -4 or >= 16. Rust's `{}` gives a shortest round-tripping
/// representation but never uses exponential form, so we post-process to match
/// CPython's `float_repr` thresholds.
fn format_float(x: f64) -> String {
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x.is_nan() {
        return "nan".into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }

    let neg = x < 0.0;
    let ax = x.abs();

    // Fast path: integer-valued floats in a range where Python uses fixed
    // notation. `format!("{:.1}", x)` yields "1.0", "42.0", etc. directly,
    // skipping the full decompose_decimal pipeline. Bounded by `< 1e15` so we
    // never intrude on the scientific-notation regime (e >= 16), where Python
    // switches to exponential form; 1e15 itself falls through to the full path.
    if x.fract() == 0.0 && ax < 1e15 {
        return format!("{:.1}", x);
    }

    // Shortest round-tripping digits via Rust's default formatter.
    // e.g. 1e100 -> "10000...0", 0.0001 -> "0.0001", 1.5 -> "1.5"
    let shortest = format!("{}", ax);

    // Decompose `shortest` into the significant digits and a decimal exponent so
    // we can decide on fixed vs scientific notation the way CPython does. The
    // int+frac digits are written into a fixed stack buffer, avoiding the two
    // heap `String` allocations the old `decompose_decimal` performed. Rust's
    // `{}` prints f64 in full fixed-point form (no exponent), so the longest
    // possible expansion (denormal extremes) is ~325 digits; 512 bytes is a
    // safe upper bound for any finite, non-zero f64.
    let mut digit_buf = [0u8; 512];
    let (digits, point_exp) = decompose_decimal(&shortest, &mut digit_buf);
    // `digits` is the significant digits with no leading/trailing zeros (except
    // a single "0"); point_exp is the power of ten of the first digit.
    // Value = 0.digits * 10^(point_exp+1) effectively; we compute the decimal
    // exponent E of the most significant digit.
    let e = point_exp; // exponent of the leading significant digit

    // Write the whole result into a single pre-sized buffer: optional sign,
    // then the mantissa/fixed body. This replaces the old chain that allocated
    // an intermediate `body` String plus a separate negation String.
    let mut result = String::with_capacity(24);
    if neg {
        result.push('-');
    }
    if e < -4 || e >= 16 {
        write_scientific_into(&mut result, digits, e);
    } else {
        write_fixed_into(&mut result, digits, e);
    }
    result
}

/// Decompose a plain decimal string (no sign, no exponent) into
/// (significant_digits, exponent_of_leading_digit). The significant digits are
/// written into `buf` (a caller-owned stack buffer) and returned as a borrowed
/// `&str` view into it, so no heap allocation occurs.
fn decompose_decimal<'b>(s: &str, buf: &'b mut [u8; 512]) -> (&'b str, i32) {
    let (int_part, frac_part) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    // Concatenate int+frac digits into the stack buffer.
    let mut len = 0usize;
    for &b in int_part.as_bytes().iter().chain(frac_part.as_bytes().iter()) {
        buf[len] = b;
        len += 1;
    }
    // Find first significant (non-zero) digit.
    let first_nonzero = buf[..len].iter().position(|&b| b != b'0');
    let first_nonzero = match first_nonzero {
        Some(i) => i,
        None => {
            // all zeros
            buf[0] = b'0';
            return (
                // SAFETY: a single ASCII '0' byte is valid UTF-8.
                std::str::from_utf8(&buf[..1]).unwrap(),
                0,
            );
        }
    };
    // exponent of first significant digit = (int_part.len() - 1) - leading_zeros
    let e = int_part.len() as i32 - 1 - first_nonzero as i32;
    // Strip trailing zeros from the significant digits.
    let last_nonzero = buf[..len].iter().rposition(|&b| b != b'0').unwrap();
    let sig_start = first_nonzero;
    let sig_end = last_nonzero + 1;
    // Shift significant digits to the front so we can return a clean &str view.
    buf.copy_within(sig_start..sig_end, 0);
    let sig_len = sig_end - sig_start;
    // SAFETY: all bytes copied are ASCII decimal digits, valid UTF-8.
    (std::str::from_utf8(&buf[..sig_len]).unwrap(), e)
}

/// Write CPython-style scientific notation for `digits`/`e` directly into `out`,
/// avoiding an intermediate mantissa allocation.
fn write_scientific_into(out: &mut String, digits: &str, e: i32) {
    out.push_str(&digits[0..1]);
    let rest = &digits[1..];
    if !rest.is_empty() {
        out.push('.');
        out.push_str(rest);
    }
    out.push('e');
    out.push(if e < 0 { '-' } else { '+' });
    // two-digit (minimum) exponent, no intermediate allocation
    let ae = e.unsigned_abs();
    if ae < 10 {
        out.push('0');
        out.push((b'0' + ae as u8) as char);
    } else {
        let mut tmp = [0u8; 16];
        let mut i = tmp.len();
        let mut n = ae;
        while n > 0 {
            i -= 1;
            tmp[i] = b'0' + (n % 10) as u8;
            n /= 10;
        }
        // SAFETY: bytes written are ASCII digits.
        out.push_str(std::str::from_utf8(&tmp[i..]).unwrap());
    }
}

/// Write CPython-style fixed notation for `digits`/`e` directly into `out`.
fn write_fixed_into(out: &mut String, digits: &str, e: i32) {
    let n = digits.len() as i32;
    if e >= 0 {
        if e + 1 >= n {
            // all significant digits are in the integer part; pad zeros, add .0
            out.push_str(digits);
            for _ in 0..(e + 1 - n) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            let split = (e + 1) as usize;
            out.push_str(&digits[..split]);
            out.push('.');
            out.push_str(&digits[split..]);
        }
    } else {
        // 0.00..digits
        let zeros = (-e - 1) as usize;
        out.push_str("0.");
        for _ in 0..zeros {
            out.push('0');
        }
        out.push_str(digits);
    }
}

/// Zero-allocation `Display` wrapper that renders a value's container-repr form
/// (strings quoted) directly into a formatter, avoiding an intermediate String
/// for every element during container display.
struct ReprDisplay<'a>(&'a Value);

impl<'a> fmt::Display for ReprDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Value::Str(s) => write!(f, "'{}'", s),
            other => fmt::Display::fmt(other, f),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 6. EvalError enum + EvalResult
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    NameNotDefined { name: String, expr: String },
    FunctionNotDefined { func_name: String, expr: String },
    FeatureNotAvailable(String),
    NumberTooHigh(String),
    IterableTooLong(String),
    InvalidExpression(String),
    OperatorNotDefined { op: String, expr: String },
    AttributeDoesNotExist { attr: String, expr: String },
}

pub type EvalResult<T> = Result<T, EvalError>;

// ───────────────────────────────────────────────────────────────────────────
// 7. Lexer
// ───────────────────────────────────────────────────────────────────────────

struct Lexer {
    input: Vec<char>,
    pos: usize,
    tokens: Vec<Token>,
}

impl Lexer {
    fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let cap = chars.len() / 3 + 4;
        Lexer {
            input: chars,
            pos: 0,
            tokens: Vec::with_capacity(cap),
        }
    }

    #[inline]
    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    #[inline]
    fn peek_at(&self, off: usize) -> Option<char> {
        self.input.get(self.pos + off).copied()
    }

    #[inline]
    fn advance(&mut self) -> Option<char> {
        let c = self.input.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn run(&mut self) -> EvalResult<()> {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += 1;
                continue;
            }
            // String literals (also detect f-string / b-string prefixes via ident handling)
            if c == '"' || c == '\'' {
                self.read_string()?;
                continue;
            }
            if c.is_ascii_digit() || (c == '.' && self.peek_at(1).map_or(false, |d| d.is_ascii_digit()))
            {
                self.read_number()?;
                continue;
            }
            if c == '_' || c.is_alphabetic() {
                self.read_name()?;
                continue;
            }
            self.read_operator()?;
        }
        self.tokens.push(Token::Eof);
        Ok(())
    }

    fn read_string(&mut self) -> EvalResult<()> {
        let quote = self.advance().unwrap();
        let mut s = String::new();
        loop {
            match self.advance() {
                None => {
                    return Err(EvalError::InvalidExpression(
                        "Sorry, unterminated string literal".into(),
                    ));
                }
                Some(c) if c == quote => break,
                Some('\\') => {
                    let esc = self.advance().ok_or_else(|| {
                        EvalError::InvalidExpression(
                            "Sorry, unterminated string literal".into(),
                        )
                    })?;
                    match esc {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '\\' => s.push('\\'),
                        '\'' => s.push('\''),
                        '"' => s.push('"'),
                        '0' => s.push('\0'),
                        other => {
                            s.push('\\');
                            s.push(other);
                        }
                    }
                }
                Some(c) => s.push(c),
            }
        }
        if s.chars().count() > MAX_STRING_LENGTH {
            return Err(EvalError::IterableTooLong(format!(
                "String Literal in statement is too long! ({}, when {} is max)",
                s.chars().count(),
                MAX_STRING_LENGTH
            )));
        }
        self.tokens.push(Token::StrLit(s));
        Ok(())
    }

    fn read_number(&mut self) -> EvalResult<()> {
        let start = self.pos;
        // Hex / octal / binary prefixes
        if self.peek() == Some('0') {
            if let Some(p) = self.peek_at(1) {
                let radix = match p {
                    'x' | 'X' => Some(16u32),
                    'o' | 'O' => Some(8u32),
                    'b' | 'B' => Some(2u32),
                    _ => None,
                };
                if let Some(radix) = radix {
                    self.pos += 2;
                    let ds = self.pos;
                    while let Some(c) = self.peek() {
                        if c == '_' || c.is_digit(radix) {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                    let len = self.pos - ds;
                    let mut digits = String::with_capacity(len);
                    for &c in &self.input[ds..self.pos] {
                        if c != '_' {
                            digits.push(c);
                        }
                    }
                    if digits.is_empty() {
                        return Err(EvalError::InvalidExpression(
                            "Sorry, invalid numeric literal".into(),
                        ));
                    }
                    self.push_int_from_radix(&digits, radix)?;
                    return Ok(());
                }
            }
        }

        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '_' {
                self.pos += 1;
            } else if c == '.' {
                // Only one dot; and not a method-access dot followed by alpha
                is_float = true;
                self.pos += 1;
            } else if c == 'e' || c == 'E' {
                is_float = true;
                self.pos += 1;
                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        self.pos += 1;
                    }
                }
            } else {
                break;
            }
        }
        let len = self.pos - start;
        let mut raw = String::with_capacity(len);
        for &c in &self.input[start..self.pos] {
            if c != '_' {
                raw.push(c);
            }
        }

        if is_float {
            let f: f64 = raw.parse().map_err(|_| {
                EvalError::InvalidExpression(format!("Sorry, invalid float literal '{}'", raw))
            })?;
            self.tokens.push(Token::FloatLit(f));
        } else {
            self.push_int_decimal(&raw)?;
        }
        Ok(())
    }

    fn push_int_decimal(&mut self, raw: &str) -> EvalResult<()> {
        if let Ok(n) = raw.parse::<i64>() {
            self.tokens.push(Token::IntLit(n));
        } else if let Ok(n) = raw.parse::<i128>() {
            self.tokens.push(Token::Int128Lit(n));
        } else {
            let b: BigInt = raw.parse().map_err(|_| {
                EvalError::InvalidExpression(format!("Sorry, invalid integer literal '{}'", raw))
            })?;
            self.tokens.push(Token::BigIntLit(b));
        }
        Ok(())
    }

    fn push_int_from_radix(&mut self, digits: &str, radix: u32) -> EvalResult<()> {
        if let Ok(n) = i64::from_str_radix(digits, radix) {
            self.tokens.push(Token::IntLit(n));
        } else if let Ok(n) = i128::from_str_radix(digits, radix) {
            self.tokens.push(Token::Int128Lit(n));
        } else {
            let b = BigInt::parse_bytes(digits.as_bytes(), radix).ok_or_else(|| {
                EvalError::InvalidExpression("Sorry, invalid integer literal".into())
            })?;
            self.tokens.push(Token::BigIntLit(b));
        }
        Ok(())
    }

    fn read_name(&mut self) -> EvalResult<()> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == '_' || c.is_alphanumeric() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let slice: String = self.input[start..self.pos].iter().collect();
        // f-string / byte-string prefixes: identifier immediately followed by a quote
        if let Some(q) = self.peek() {
            if q == '"' || q == '\'' {
                let low = slice.to_ascii_lowercase();
                if matches!(
                    low.as_str(),
                    "f" | "rf" | "fr" | "b" | "rb" | "br" | "u" | "r"
                ) {
                    if low.contains('f') {
                        return Err(EvalError::InvalidExpression(
                            "Sorry, f-strings are not supported".into(),
                        ));
                    }
                    return Err(EvalError::InvalidExpression(
                        "Sorry, string prefixes are not supported".into(),
                    ));
                }
            }
        }
        // Fast-reject: most identifiers are variable names, not keywords. All
        // keywords here are <= 6 bytes and begin with one of a small set of
        // ASCII bytes. A single length + first-byte check skips the full string
        // comparison for the common identifier case. Non-ASCII first bytes
        // (multi-byte chars) never match the keyword set, so they fall through
        // to Token::Ident correctly.
        let bytes = slice.as_bytes();
        let could_be_keyword = bytes.len() <= 6
            && matches!(
                bytes.first(),
                Some(b'T' | b'F' | b'N' | b'a' | b'o' | b'n' | b'i' | b'e' | b'l' | b'f')
            );
        let tok = if !could_be_keyword {
            Token::Ident(slice)
        } else {
            match slice.as_str() {
                "True" => Token::BoolLit(true),
                "False" => Token::BoolLit(false),
                "None" => Token::NoneLit,
                "and" => Token::And,
                "or" => Token::Or,
                "not" => Token::Not,
                "in" => Token::In,
                "is" => Token::Is,
                "if" => Token::If,
                "else" => Token::Else,
                "lambda" => {
                    return Err(EvalError::FeatureNotAvailable(
                        "Sorry, Lambda is not available in this evaluator".into(),
                    ));
                }
                "import" => {
                    return Err(EvalError::FeatureNotAvailable(
                        "Sorry, 'import' is not allowed.".into(),
                    ));
                }
                "for" => {
                    return Err(EvalError::FeatureNotAvailable(
                        "Sorry, ListComp is not available in this evaluator".into(),
                    ));
                }
                _ => Token::Ident(slice),
            }
        };
        self.tokens.push(tok);
        Ok(())
    }

    fn read_operator(&mut self) -> EvalResult<()> {
        let c = self.advance().unwrap();
        let next = self.peek();
        let tok = match c {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => {
                if next == Some('*') {
                    self.pos += 1;
                    Token::StarStar
                } else {
                    Token::Star
                }
            }
            '/' => {
                if next == Some('/') {
                    self.pos += 1;
                    Token::SlashSlash
                } else {
                    Token::Slash
                }
            }
            '%' => Token::Percent,
            '&' => Token::Amp,
            '|' => Token::Pipe,
            '^' => Token::Caret,
            '~' => Token::Tilde,
            '<' => match next {
                Some('<') => {
                    self.pos += 1;
                    Token::LtLt
                }
                Some('=') => {
                    self.pos += 1;
                    Token::LtEq
                }
                _ => Token::Lt,
            },
            '>' => match next {
                Some('>') => {
                    self.pos += 1;
                    Token::GtGt
                }
                Some('=') => {
                    self.pos += 1;
                    Token::GtEq
                }
                _ => Token::Gt,
            },
            '=' => {
                if next == Some('=') {
                    self.pos += 1;
                    Token::EqEq
                } else {
                    return Err(EvalError::FeatureNotAvailable(
                        "Sorry, assignment is not available in this evaluator".into(),
                    ));
                }
            }
            '!' => {
                if next == Some('=') {
                    self.pos += 1;
                    Token::BangEq
                } else {
                    return Err(EvalError::InvalidExpression(
                        "Sorry, unexpected character '!'".into(),
                    ));
                }
            }
            '(' => Token::LParen,
            ')' => Token::RParen,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            ',' => Token::Comma,
            ':' => Token::Colon,
            '.' => {
                return Err(EvalError::FeatureNotAvailable(
                    "Sorry, attribute access is not available in this evaluator".into(),
                ));
            }
            other => {
                return Err(EvalError::InvalidExpression(format!(
                    "Sorry, unexpected character '{}'",
                    other
                )));
            }
        };
        self.tokens.push(tok);
        Ok(())
    }
}

pub fn tokenize(input: &str) -> EvalResult<Vec<Token>> {
    if input.trim().is_empty() {
        return Err(EvalError::InvalidExpression(
            "Sorry, cannot evaluate empty string".into(),
        ));
    }
    let mut lx = Lexer::new(input);
    lx.run()?;
    Ok(lx.tokens)
}

// ───────────────────────────────────────────────────────────────────────────
// 8. Parser (recursive descent → Expr AST)
// ───────────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    expr: &'a str,
    // G014: expression nodes are allocated here during the parse instead of via
    // a per-node `Box::new`. Pre-sized from the token count.
    arena: ExprArena,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], expr: &'a str) -> Self {
        Parser {
            tokens,
            pos: 0,
            expr,
            // Heuristic: an expression of T tokens has on the order of T/2
            // nodes (operands plus interior operators). Reserve accordingly so
            // a typical parse never grows the arena past its first chunk.
            arena: ExprArena::with_node_hint(tokens.len() / 2 + 4),
        }
    }

    #[inline]
    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    #[inline]
    fn peek2(&self) -> &Token {
        self.tokens.get(self.pos + 1).unwrap_or(&Token::Eof)
    }

    #[inline]
    fn advance(&mut self) -> Token {
        let t = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    /// Advance the cursor without cloning the consumed token. Use this at the
    /// many sites that only need to skip a known operator/punctuation token and
    /// discard its value; `advance()` (which clones) is reserved for the few
    /// sites that actually consume the token's payload (e.g. parse_atom).
    #[inline]
    fn bump(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn invalid(&self, msg: &str) -> EvalError {
        EvalError::InvalidExpression(msg.to_string())
    }

    // Level 1: ternary  body if test else orelse
    #[inline]
    fn parse_ifexp(&mut self) -> EvalResult<ExprRef> {
        let body = self.parse_or()?;
        if *self.peek() == Token::If {
            self.bump();
            let test = self.parse_or()?;
            if *self.peek() != Token::Else {
                return Err(self.invalid("Sorry, ternary expression missing 'else'"));
            }
            self.bump();
            let orelse = self.parse_ifexp()?;
            Ok(self.arena.alloc(ArenaNode::IfExp { test, body, orelse }))
        } else {
            Ok(body)
        }
    }

    // Level 2: or
    #[inline]
    fn parse_or(&mut self) -> EvalResult<ExprRef> {
        let first = self.parse_and()?;
        if *self.peek() != Token::Or {
            return Ok(first);
        }
        let mut values = vec![first];
        while *self.peek() == Token::Or {
            self.bump();
            values.push(self.parse_and()?);
        }
        Ok(self.arena.alloc(ArenaNode::BoolOp {
            op: BoolOpKind::Or,
            values,
        }))
    }

    // Level 3: and
    #[inline]
    fn parse_and(&mut self) -> EvalResult<ExprRef> {
        let first = self.parse_not()?;
        if *self.peek() != Token::And {
            return Ok(first);
        }
        let mut values = vec![first];
        while *self.peek() == Token::And {
            self.bump();
            values.push(self.parse_not()?);
        }
        Ok(self.arena.alloc(ArenaNode::BoolOp {
            op: BoolOpKind::And,
            values,
        }))
    }

    // Level 4: not (unary boolean)
    #[inline]
    fn parse_not(&mut self) -> EvalResult<ExprRef> {
        if *self.peek() == Token::Not {
            self.bump();
            let operand = self.parse_not()?;
            Ok(self.arena.alloc(ArenaNode::UnaryOp {
                op: UnaryOpKind::Not,
                operand,
            }))
        } else {
            self.parse_comparison()
        }
    }

    // Level 5: comparisons, membership, identity (single op; chained → error)
    #[inline]
    fn parse_comparison(&mut self) -> EvalResult<ExprRef> {
        let left = self.parse_bitor()?;

        // Determine if the next token starts a comparison operator.
        let op_kind = self.peek_compare_op();
        if op_kind.is_none() {
            return Ok(left);
        }

        let node = self.consume_compare(left)?;

        // Chained comparison detection: another comparison operator follows.
        if self.peek_compare_op().is_some() {
            return Err(self.invalid("Sorry, chained comparisons are not supported"));
        }
        Ok(node)
    }

    /// Peek whether the current position begins a comparison/membership/identity operator.
    #[inline]
    fn peek_compare_op(&self) -> Option<()> {
        match self.peek() {
            Token::EqEq | Token::BangEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq => {
                Some(())
            }
            Token::In | Token::Is => Some(()),
            Token::Not => {
                // "not in"
                if *self.peek2() == Token::In {
                    Some(())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn consume_compare(&mut self, left: ExprRef) -> EvalResult<ExprRef> {
        match self.peek().clone() {
            Token::EqEq => {
                self.bump();
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::Compare {
                    left,
                    op: CmpOpKind::Eq,
                    right,
                }))
            }
            Token::BangEq => {
                self.bump();
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::Compare {
                    left,
                    op: CmpOpKind::NotEq,
                    right,
                }))
            }
            Token::Lt => {
                self.bump();
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::Compare {
                    left,
                    op: CmpOpKind::Lt,
                    right,
                }))
            }
            Token::Gt => {
                self.bump();
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::Compare {
                    left,
                    op: CmpOpKind::Gt,
                    right,
                }))
            }
            Token::LtEq => {
                self.bump();
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::Compare {
                    left,
                    op: CmpOpKind::LtE,
                    right,
                }))
            }
            Token::GtEq => {
                self.bump();
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::Compare {
                    left,
                    op: CmpOpKind::GtE,
                    right,
                }))
            }
            Token::In => {
                self.bump();
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::BinOp {
                    op: BinOpKind::In,
                    left,
                    right,
                }))
            }
            Token::Not => {
                // not in
                self.bump(); // not
                self.bump(); // in
                let right = self.parse_bitor()?;
                Ok(self.arena.alloc(ArenaNode::BinOp {
                    op: BinOpKind::NotIn,
                    left,
                    right,
                }))
            }
            Token::Is => {
                self.bump();
                if *self.peek() == Token::Not {
                    self.bump();
                    let right = self.parse_bitor()?;
                    Ok(self.arena.alloc(ArenaNode::BinOp {
                        op: BinOpKind::IsNot,
                        left,
                        right,
                    }))
                } else {
                    let right = self.parse_bitor()?;
                    Ok(self.arena.alloc(ArenaNode::BinOp {
                        op: BinOpKind::Is,
                        left,
                        right,
                    }))
                }
            }
            _ => unreachable!("consume_compare called without comparison op"),
        }
    }

    // Levels 6–11: binary operators (bitor | bitxor ^ bitand & shift <<>> add +-
    // mul * / // %), all left-associative. Previously six separate recursive
    // descent levels; collapsed into one precedence-climbing loop so reaching an
    // operand costs a single call frame instead of six. Precedence values
    // preserve the original level ordering exactly (higher binds tighter):
    //   mul/div/floordiv/mod = 6, add/sub = 5, shift = 4,
    //   bitand = 3, bitxor = 2, bitor = 1.
    #[inline]
    fn binary_op(tok: &Token) -> Option<(BinOpKind, u8)> {
        Some(match tok {
            Token::Star => (BinOpKind::Mul, 6),
            Token::Slash => (BinOpKind::Div, 6),
            Token::SlashSlash => (BinOpKind::FloorDiv, 6),
            Token::Percent => (BinOpKind::Mod, 6),
            Token::Plus => (BinOpKind::Add, 5),
            Token::Minus => (BinOpKind::Sub, 5),
            Token::LtLt => (BinOpKind::LShift, 4),
            Token::GtGt => (BinOpKind::RShift, 4),
            Token::Amp => (BinOpKind::BitAnd, 3),
            Token::Caret => (BinOpKind::BitXor, 2),
            Token::Pipe => (BinOpKind::BitOr, 1),
            _ => return None,
        })
    }

    /// Entry to the binary-operator precedence ladder (formerly parse_bitor).
    #[inline]
    fn parse_bitor(&mut self) -> EvalResult<ExprRef> {
        self.parse_binary(1)
    }

    /// Precedence-climbing parser for all left-associative binary operators.
    /// `min_prec` is the lowest precedence this invocation will consume.
    fn parse_binary(&mut self, min_prec: u8) -> EvalResult<ExprRef> {
        let mut left = self.parse_unary()?;
        while let Some((op, prec)) = Self::binary_op(self.peek()) {
            if prec < min_prec {
                break;
            }
            self.bump();
            // Left-associative: right side binds operators strictly tighter
            // than the current one (prec + 1).
            let right = self.parse_binary(prec + 1)?;
            left = self.arena.alloc(ArenaNode::BinOp { op, left, right });
        }
        Ok(left)
    }

    // Level 13: unary - + ~  (note: `not` handled at level 4)
    #[inline]
    fn parse_unary(&mut self) -> EvalResult<ExprRef> {
        match self.peek() {
            Token::Minus => {
                self.bump();
                let operand = self.parse_unary()?;
                Ok(self.arena.alloc(ArenaNode::UnaryOp {
                    op: UnaryOpKind::Neg,
                    operand,
                }))
            }
            Token::Plus => {
                self.bump();
                let operand = self.parse_unary()?;
                Ok(self.arena.alloc(ArenaNode::UnaryOp {
                    op: UnaryOpKind::Pos,
                    operand,
                }))
            }
            Token::Tilde => {
                self.bump();
                let operand = self.parse_unary()?;
                Ok(self.arena.alloc(ArenaNode::UnaryOp {
                    op: UnaryOpKind::BitNot,
                    operand,
                }))
            }
            _ => self.parse_pow(),
        }
    }

    // Level 12: ** (right-associative). Binds tighter than unary on the left,
    // but its right operand may itself be a unary expression (e.g. 2 ** -1).
    #[inline]
    fn parse_pow(&mut self) -> EvalResult<ExprRef> {
        let base = self.parse_postfix()?;
        if *self.peek() == Token::StarStar {
            self.bump();
            let exp = self.parse_unary()?;
            Ok(self.arena.alloc(ArenaNode::BinOp {
                op: BinOpKind::Pow,
                left: base,
                right: exp,
            }))
        } else {
            Ok(base)
        }
    }

    // Level 14: atoms, calls, subscript-detection
    #[inline]
    fn parse_postfix(&mut self) -> EvalResult<ExprRef> {
        let atom = self.parse_atom()?;
        // Detect subscript and attribute access on a primary.
        match self.peek() {
            Token::LBracket => {
                return Err(self.invalid("Sorry, subscript/indexing is not supported"));
            }
            _ => {}
        }
        Ok(atom)
    }

    fn parse_atom(&mut self) -> EvalResult<ExprRef> {
        let tok = self.advance();
        match tok {
            Token::IntLit(n) => Ok(self.arena.alloc(ArenaNode::Literal(Value::Int(n)))),
            Token::Int128Lit(n) => Ok(self.arena.alloc(ArenaNode::Literal(Value::Int128(n)))),
            Token::BigIntLit(b) => Ok(self.arena.alloc(ArenaNode::Literal(Value::BigInt(b)))),
            Token::FloatLit(f) => Ok(self.arena.alloc(ArenaNode::Literal(Value::Float(f)))),
            Token::StrLit(s) => Ok(self.arena.alloc(ArenaNode::Literal(Value::Str(s)))),
            Token::BoolLit(b) => Ok(self.arena.alloc(ArenaNode::Literal(Value::Bool(b)))),
            Token::NoneLit => Ok(self.arena.alloc(ArenaNode::Literal(Value::None))),
            Token::Ident(name) => {
                // function call?
                if *self.peek() == Token::LParen {
                    self.bump();
                    let args = self.parse_call_args()?;
                    Ok(self.arena.alloc(ArenaNode::Call { func: name, args }))
                } else {
                    Ok(self.arena.alloc(ArenaNode::Name(name)))
                }
            }
            Token::LParen => {
                // () empty tuple, (expr) grouping, (a, b, ...) tuple
                if *self.peek() == Token::RParen {
                    self.bump();
                    return Ok(self.arena.alloc(ArenaNode::Tuple(vec![])));
                }
                let first = self.parse_ifexp()?;
                if *self.peek() == Token::Comma {
                    let mut items = vec![first];
                    while *self.peek() == Token::Comma {
                        self.bump();
                        if *self.peek() == Token::RParen {
                            break;
                        }
                        items.push(self.parse_ifexp()?);
                    }
                    self.expect(Token::RParen)?;
                    Ok(self.arena.alloc(ArenaNode::Tuple(items)))
                } else {
                    self.expect(Token::RParen)?;
                    Ok(first)
                }
            }
            Token::LBracket => {
                // list literal (comprehension → error via 'for' keyword in lexer)
                let mut items = Vec::new();
                if *self.peek() != Token::RBracket {
                    items.push(self.parse_ifexp()?);
                    while *self.peek() == Token::Comma {
                        self.bump();
                        if *self.peek() == Token::RBracket {
                            break;
                        }
                        items.push(self.parse_ifexp()?);
                    }
                }
                self.expect(Token::RBracket)?;
                Ok(self.arena.alloc(ArenaNode::List(items)))
            }
            Token::LBrace => self.parse_brace(),
            Token::Eof => Err(self.invalid("Sorry, unexpected end of expression")),
            other => Err(EvalError::InvalidExpression(format!(
                "Sorry, unexpected token {:?} in expression '{}'",
                other, self.expr
            ))),
        }
    }

    // { } empty dict; {k: v, ...} dict; {a, b, ...} set
    fn parse_brace(&mut self) -> EvalResult<ExprRef> {
        if *self.peek() == Token::RBrace {
            self.bump();
            return Ok(self.arena.alloc(ArenaNode::Dict {
                keys: vec![],
                values: vec![],
            }));
        }
        let first = self.parse_ifexp()?;
        if *self.peek() == Token::Colon {
            // dict
            self.bump();
            let v0 = self.parse_ifexp()?;
            let mut keys = vec![first];
            let mut values = vec![v0];
            while *self.peek() == Token::Comma {
                self.bump();
                if *self.peek() == Token::RBrace {
                    break;
                }
                let k = self.parse_ifexp()?;
                self.expect(Token::Colon)?;
                let v = self.parse_ifexp()?;
                keys.push(k);
                values.push(v);
            }
            self.expect(Token::RBrace)?;
            Ok(self.arena.alloc(ArenaNode::Dict { keys, values }))
        } else {
            // set
            let mut items = vec![first];
            while *self.peek() == Token::Comma {
                self.bump();
                if *self.peek() == Token::RBrace {
                    break;
                }
                items.push(self.parse_ifexp()?);
            }
            self.expect(Token::RBrace)?;
            Ok(self.arena.alloc(ArenaNode::Set(items)))
        }
    }

    fn parse_call_args(&mut self) -> EvalResult<Vec<ExprRef>> {
        let mut args = Vec::new();
        if *self.peek() == Token::RParen {
            self.bump();
            return Ok(args);
        }
        args.push(self.parse_ifexp()?);
        while *self.peek() == Token::Comma {
            self.bump();
            if *self.peek() == Token::RParen {
                break;
            }
            args.push(self.parse_ifexp()?);
        }
        self.expect(Token::RParen)?;
        Ok(args)
    }

    fn expect(&mut self, t: Token) -> EvalResult<()> {
        if *self.peek() == t {
            self.bump();
            Ok(())
        } else {
            Err(EvalError::InvalidExpression(format!(
                "Sorry, expected {:?} but found {:?}",
                t,
                self.peek()
            )))
        }
    }
}

pub fn parse(tokens: &[Token], expr: &str) -> EvalResult<Expr> {
    let mut p = Parser::new(tokens, expr);
    let root = p.parse_ifexp()?;
    if *p.peek() != Token::Eof {
        return Err(EvalError::InvalidExpression(format!(
            "Sorry, unexpected trailing tokens in expression '{}'",
            expr
        )));
    }
    // G014: the tree was built in the arena; materialize the root into the
    // owning `Box<Expr>` form the evaluator/cache/tests consume. The arena
    // (and every node in it) is freed in one shot when `p` drops at end of
    // scope — no per-node deallocation.
    Ok(p.arena.materialize(root))
}

/// G014: parse `expr` and return the populated arena together with its root
/// handle, **without** materializing a `Box<Expr>` tree. The PyO3 fast path
/// (`fast_eval` via `parse_cached_arena` + `eval_arena`) uses this so the hot
/// path never pays the materialization cost — the arena win is preserved
/// end to end. The arena owns every node; the returned `ExprRef` only indexes
/// into that same arena, so it cannot dangle.
pub fn parse_into_arena(tokens: &[Token], expr: &str) -> EvalResult<(ExprArena, ExprRef)> {
    let mut p = Parser::new(tokens, expr);
    let root = p.parse_ifexp()?;
    if *p.peek() != Token::Eof {
        return Err(EvalError::InvalidExpression(format!(
            "Sorry, unexpected trailing tokens in expression '{}'",
            expr
        )));
    }
    Ok((p.arena, root))
}

// ───────────────────────────────────────────────────────────────────────────
// 8b. Parsed-AST cache
// ───────────────────────────────────────────────────────────────────────────
//
// `fast_eval` always tokenizes + parses + evaluates. In real workloads the same
// expression string is evaluated many times with different `names`, so the
// tokenize+parse work is pure repetition. This cache memoizes the parsed AST
// keyed by the expression string, turning a repeat call's parse cost into a
// hash lookup plus a refcount bump.
//
// Access is via a thread-local map: PyO3 holds the GIL across a `fast_eval`
// call so accesses are already serialized, and a thread-local avoids any lock
// overhead. The AST is stored behind `Rc` so a hit hands back a cheap clone
// (refcount increment) rather than deep-cloning the tree. Only successful
// parses are cached; errors fall through unchanged. The cache is capped and
// cleared wholesale when full to bound memory without LRU bookkeeping.

use std::cell::RefCell;
use std::rc::Rc;

const AST_CACHE_CAP: usize = 1024;

thread_local! {
    static AST_CACHE: RefCell<HashMap<String, Rc<Expr>>> = RefCell::new(HashMap::new());
}

/// Tokenize + parse `expr`, memoizing the resulting AST. On a cache hit neither
/// the lexer nor the parser runs. Returns a shared handle to the AST.
pub fn parse_cached(expr: &str) -> EvalResult<Rc<Expr>> {
    // Fast path: hit.
    if let Some(hit) = AST_CACHE.with(|c| c.borrow().get(expr).cloned()) {
        return Ok(hit);
    }
    // Miss: do the real work outside the borrow.
    let tokens = tokenize(expr)?;
    let ast = parse(&tokens, expr)?;
    let rc = Rc::new(ast);
    AST_CACHE.with(|c| {
        let mut map = c.borrow_mut();
        if map.len() >= AST_CACHE_CAP {
            map.clear();
        }
        map.insert(expr.to_string(), Rc::clone(&rc));
    });
    Ok(rc)
}

#[cfg(any(test, feature = "bench"))]
pub fn ast_cache_clear() {
    AST_CACHE.with(|c| c.borrow_mut().clear());
}

// ── G014: arena-native cache for the PyO3 hot path ──────────────────────────
//
// `parse_cached` returns the materialized `Box<Expr>` tree the crate-internal
// API (and the unit tests) expect. The PyO3 boundary (`fast_eval`) has no such
// contract — it only needs a tree it can walk to produce a `PyObject`. So it
// uses this *separate* cache, which stores the arena and its root directly and
// is evaluated by `eval_arena` with no materialization. That keeps the
// production fast path arena-only: on a miss we tokenize + parse-into-arena
// once; on a hit we hand back an `Rc` to the arena (refcount bump) and walk it
// in place. This is the path the "equal or better vs Box" benchmark exercises.
//
// The arena behind the `Rc` is immutable once cached and is only ever read via
// `eval_arena`, so sharing it across calls cannot mutate or invalidate it; it
// is freed exactly when its last `Rc` is dropped (cache eviction), with the
// whole region released in one shot — no per-node free, no dangling refs.
type ArenaAst = (ExprArena, ExprRef);

thread_local! {
    static ARENA_CACHE: RefCell<HashMap<String, Rc<ArenaAst>>> = RefCell::new(HashMap::new());
}

/// Tokenize + parse `expr` into an arena, memoizing the (arena, root) pair.
/// On a hit neither lexer nor parser runs. Used only by the PyO3 fast path.
pub fn parse_cached_arena(expr: &str) -> EvalResult<Rc<ArenaAst>> {
    if let Some(hit) = ARENA_CACHE.with(|c| c.borrow().get(expr).cloned()) {
        return Ok(hit);
    }
    let tokens = tokenize(expr)?;
    let pair = parse_into_arena(&tokens, expr)?;
    let rc = Rc::new(pair);
    ARENA_CACHE.with(|c| {
        let mut map = c.borrow_mut();
        if map.len() >= AST_CACHE_CAP {
            map.clear();
        }
        map.insert(expr.to_string(), Rc::clone(&rc));
    });
    Ok(rc)
}

#[cfg(any(test, feature = "bench"))]
pub fn arena_cache_clear() {
    ARENA_CACHE.with(|c| c.borrow_mut().clear());
}

// ───────────────────────────────────────────────────────────────────────────
// 9. Native builtins
// ───────────────────────────────────────────────────────────────────────────

pub fn call_builtin(func_name: &str, args: Vec<Value>, original_expr: &str) -> EvalResult<Value> {
    match func_name {
        "rand" => builtin_rand(&args),
        "randint" => builtin_randint(&args),
        "int" => builtin_int(&args),
        "float" => builtin_float(&args),
        "str" => builtin_str(&args),
        "bool" => builtin_bool(&args),
        "abs" => builtin_abs(&args),
        "round" => builtin_round(&args),
        "min" => builtin_minmax(&args, true),
        "max" => builtin_minmax(&args, false),
        "len" => builtin_len(&args),
        "sum" => builtin_sum(&args),
        _ => Err(EvalError::FunctionNotDefined {
            func_name: func_name.to_string(),
            expr: original_expr.to_string(),
        }),
    }
}

fn arg_count_err(name: &str) -> EvalError {
    EvalError::InvalidExpression(format!("Sorry, wrong number of arguments to {}()", name))
}

fn builtin_rand(args: &[Value]) -> EvalResult<Value> {
    if !args.is_empty() {
        return Err(arg_count_err("rand"));
    }
    use rand::Rng;
    let x: f64 = rand::thread_rng().gen_range(0.0..1.0);
    Ok(Value::Float(x))
}

fn builtin_randint(args: &[Value]) -> EvalResult<Value> {
    if args.len() != 1 {
        return Err(arg_count_err("randint"));
    }
    let top = value_to_f64(&args[0])?;
    use rand::Rng;
    let r: f64 = rand::thread_rng().gen_range(0.0..1.0);
    Ok(Value::Int((r * top) as i64))
}

fn builtin_int(args: &[Value]) -> EvalResult<Value> {
    if args.len() != 1 {
        return Err(arg_count_err("int"));
    }
    match &args[0] {
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Int128(n) => Ok(Value::Int128(*n)),
        Value::BigInt(b) => Ok(Value::BigInt(b.clone())),
        Value::Float(f) => Ok(Value::Int(f.trunc() as i64)),
        Value::Str(s) => {
            let t = s.trim();
            if let Ok(n) = t.parse::<i64>() {
                Ok(Value::Int(n))
            } else if let Ok(n) = t.parse::<i128>() {
                Ok(Value::Int128(n))
            } else if let Ok(b) = t.parse::<BigInt>() {
                Ok(Value::BigInt(b))
            } else {
                Err(EvalError::InvalidExpression(format!(
                    "invalid literal for int() with base 10: '{}'",
                    s
                )))
            }
        }
        _ => Err(EvalError::InvalidExpression(
            "int() argument must be a string or a number".into(),
        )),
    }
}

fn builtin_float(args: &[Value]) -> EvalResult<Value> {
    if args.len() != 1 {
        return Err(arg_count_err("float"));
    }
    match &args[0] {
        Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
        Value::Int(n) => Ok(Value::Float(*n as f64)),
        Value::Int128(n) => Ok(Value::Float(*n as f64)),
        Value::BigInt(b) => Ok(Value::Float(bigint_to_f64(b))),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::Str(s) => {
            let t = s.trim();
            match t {
                "inf" | "Infinity" | "+inf" => return Ok(Value::Float(f64::INFINITY)),
                "-inf" | "-Infinity" => return Ok(Value::Float(f64::NEG_INFINITY)),
                "nan" | "+nan" | "-nan" => return Ok(Value::Float(f64::NAN)),
                _ => {}
            }
            t.parse::<f64>().map(Value::Float).map_err(|_| {
                EvalError::InvalidExpression(format!(
                    "could not convert string to float: '{}'",
                    s
                ))
            })
        }
        _ => Err(EvalError::InvalidExpression(
            "float() argument must be a string or a number".into(),
        )),
    }
}

fn builtin_str(args: &[Value]) -> EvalResult<Value> {
    if args.len() != 1 {
        return Err(arg_count_err("str"));
    }
    Ok(Value::Str(format!("{}", args[0])))
}

fn builtin_bool(args: &[Value]) -> EvalResult<Value> {
    if args.len() != 1 {
        return Err(arg_count_err("bool"));
    }
    Ok(Value::Bool(value_is_truthy(&args[0])))
}

fn builtin_abs(args: &[Value]) -> EvalResult<Value> {
    if args.len() != 1 {
        return Err(arg_count_err("abs"));
    }
    match &args[0] {
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
        Value::Int(n) => match n.checked_abs() {
            Some(v) => Ok(Value::Int(v)),
            None => Ok(Value::Int128((*n as i128).abs())),
        },
        Value::Int128(n) => match n.checked_abs() {
            Some(v) => Ok(Value::Int128(v)),
            None => Ok(Value::BigInt(BigInt::from(*n).abs())),
        },
        Value::BigInt(b) => Ok(Value::BigInt(b.abs())),
        Value::Float(f) => Ok(Value::Float(f.abs())),
        _ => Err(EvalError::InvalidExpression(
            "bad operand type for abs()".into(),
        )),
    }
}

fn builtin_round(args: &[Value]) -> EvalResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(arg_count_err("round"));
    }
    let ndigits: Option<i64> = if args.len() == 2 {
        Some(match &args[1] {
            Value::Int(n) => *n,
            Value::Bool(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            _ => {
                return Err(EvalError::InvalidExpression(
                    "round() second argument must be an integer".into(),
                ))
            }
        })
    } else {
        None
    };

    match &args[0] {
        // Integers round to themselves regardless of ndigits.
        Value::Int(_) | Value::Int128(_) | Value::BigInt(_) | Value::Bool(_) => {
            match &args[0] {
                Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                other => Ok(other.clone()),
            }
        }
        Value::Float(f) => {
            let r = python_round(*f, ndigits.unwrap_or(0));
            if ndigits.is_none() {
                // round(x) with no ndigits returns int in Python
                Ok(Value::Int(r as i64))
            } else {
                Ok(Value::Float(r))
            }
        }
        _ => Err(EvalError::InvalidExpression(
            "type cannot be rounded".into(),
        )),
    }
}

/// Banker's rounding (round-half-to-even) to `ndigits` decimal places.
fn python_round(x: f64, ndigits: i64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let factor = 10f64.powi(ndigits as i32);
    let scaled = x * factor;
    let rounded = round_half_even(scaled);
    rounded / factor
}

fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // exactly .5 → round to even
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

fn builtin_minmax(args: &[Value], is_min: bool) -> EvalResult<Value> {
    let items: Vec<Value> = if args.len() == 1 {
        match &args[0] {
            Value::List(v) | Value::Tuple(v) | Value::Set(v) => v.clone(),
            single => vec![single.clone()],
        }
    } else {
        args.to_vec()
    };
    if items.is_empty() {
        return Err(EvalError::InvalidExpression(format!(
            "{}() arg is an empty sequence",
            if is_min { "min" } else { "max" }
        )));
    }
    let mut best = items[0].clone();
    for it in items.iter().skip(1) {
        let ord = compare_values(it, &best)?;
        if (is_min && ord == std::cmp::Ordering::Less)
            || (!is_min && ord == std::cmp::Ordering::Greater)
        {
            best = it.clone();
        }
    }
    Ok(best)
}

fn builtin_len(args: &[Value]) -> EvalResult<Value> {
    if args.len() != 1 {
        return Err(arg_count_err("len"));
    }
    let n = match &args[0] {
        Value::Str(s) => s.chars().count(),
        Value::List(v) | Value::Tuple(v) | Value::Set(v) => v.len(),
        Value::Dict(d) => d.len(),
        _ => {
            return Err(EvalError::InvalidExpression(
                "object has no len()".into(),
            ))
        }
    };
    Ok(Value::Int(n as i64))
}

fn builtin_sum(args: &[Value]) -> EvalResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(arg_count_err("sum"));
    }
    let items: &[Value] = match &args[0] {
        Value::List(v) | Value::Tuple(v) | Value::Set(v) => v,
        _ => {
            return Err(EvalError::InvalidExpression(
                "sum() argument must be iterable".into(),
            ))
        }
    };
    let mut acc = if args.len() == 2 {
        args[1].clone()
    } else {
        Value::Int(0)
    };
    for it in items {
        acc = arith(BinOpKind::Add, &acc, it, "sum")?;
    }
    Ok(acc)
}

// ───────────────────────────────────────────────────────────────────────────
// 10. Evaluator
// ───────────────────────────────────────────────────────────────────────────

#[inline]
pub fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Int(n) => *n != 0,
        Value::Int128(n) => *n != 0,
        Value::BigInt(b) => !b.is_zero(),
        Value::Float(f) => *f != 0.0,
        Value::Str(s) => !s.is_empty(),
        Value::None => false,
        Value::List(v) => !v.is_empty(),
        Value::Tuple(v) => !v.is_empty(),
        Value::Dict(d) => !d.is_empty(),
        Value::Set(v) => !v.is_empty(),
    }
}

#[inline]
fn value_to_f64(v: &Value) -> EvalResult<f64> {
    match v {
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Int(n) => Ok(*n as f64),
        Value::Int128(n) => Ok(*n as f64),
        Value::BigInt(b) => Ok(bigint_to_f64(b)),
        Value::Float(f) => Ok(*f),
        _ => Err(EvalError::InvalidExpression(
            "Sorry, expected a number".into(),
        )),
    }
}

fn bigint_to_f64(b: &BigInt) -> f64 {
    b.to_f64().unwrap_or(f64::INFINITY)
}

pub fn eval_expr(
    expr_ast: &Expr,
    names: &HashMap<String, Value>,
    original_expr: &str,
) -> EvalResult<Value> {
    match expr_ast {
        Expr::Literal(v) => Ok(v.clone()),
        Expr::Name(name) => match name.as_str() {
            "True" => Ok(Value::Bool(true)),
            "False" => Ok(Value::Bool(false)),
            "None" => Ok(Value::None),
            _ => names.get(name).cloned().ok_or_else(|| EvalError::NameNotDefined {
                name: name.clone(),
                expr: original_expr.to_string(),
            }),
        },
        Expr::UnaryOp { op, operand } => {
            let v = eval_expr(operand, names, original_expr)?;
            eval_unary(*op, v)
        }
        Expr::BinOp { op, left, right } => {
            let l = eval_expr(left, names, original_expr)?;
            let r = eval_expr(right, names, original_expr)?;
            eval_binop(*op, l, r, original_expr)
        }
        Expr::BoolOp { op, values } => {
            match op {
                BoolOpKind::And => {
                    let mut last = Value::Bool(true);
                    for (i, val) in values.iter().enumerate() {
                        let v = eval_expr(val, names, original_expr)?;
                        if !value_is_truthy(&v) {
                            return Ok(v);
                        }
                        last = v;
                        let _ = i;
                    }
                    Ok(last)
                }
                BoolOpKind::Or => {
                    let mut last = Value::Bool(false);
                    for val in values.iter() {
                        let v = eval_expr(val, names, original_expr)?;
                        if value_is_truthy(&v) {
                            return Ok(v);
                        }
                        last = v;
                    }
                    Ok(last)
                }
            }
        }
        Expr::Compare { left, op, right } => {
            let l = eval_expr(left, names, original_expr)?;
            let r = eval_expr(right, names, original_expr)?;
            eval_compare(*op, &l, &r)
        }
        Expr::IfExp { test, body, orelse } => {
            let t = eval_expr(test, names, original_expr)?;
            if value_is_truthy(&t) {
                eval_expr(body, names, original_expr)
            } else {
                eval_expr(orelse, names, original_expr)
            }
        }
        Expr::Call { func, args } => {
            let mut argv = Vec::with_capacity(args.len());
            for a in args {
                argv.push(eval_expr(a, names, original_expr)?);
            }
            call_builtin(func, argv, original_expr)
        }
        Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval_expr(it, names, original_expr)?);
            }
            Ok(Value::List(out))
        }
        Expr::Tuple(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval_expr(it, names, original_expr)?);
            }
            Ok(Value::Tuple(out))
        }
        Expr::Set(items) => {
            let mut out: Vec<Value> = Vec::with_capacity(items.len());
            for it in items {
                let v = eval_expr(it, names, original_expr)?;
                if !out.iter().any(|e| value_equal(e, &v)) {
                    out.push(v);
                }
            }
            Ok(Value::Set(out))
        }
        Expr::Dict { keys, values } => {
            let mut pairs: Vec<(Value, Value)> = Vec::with_capacity(keys.len());
            for (k, v) in keys.iter().zip(values.iter()) {
                let kv = eval_expr(k, names, original_expr)?;
                let vv = eval_expr(v, names, original_expr)?;
                if let Some(slot) = pairs.iter_mut().find(|(ek, _)| value_equal(ek, &kv)) {
                    slot.1 = vv;
                } else {
                    pairs.push((kv, vv));
                }
            }
            Ok(Value::Dict(pairs))
        }
    }
}

/// G014: evaluate the arena node `r` directly, without ever materializing a
/// `Box<Expr>` tree. Semantically identical to `eval_expr` — it shares the same
/// `eval_unary` / `eval_binop` / `eval_compare` / `call_builtin` helpers and the
/// same short-circuit and dedup rules — but walks `ArenaNode`s by `ExprRef`
/// index into `arena`. This is the PyO3 fast path's evaluator; keeping it
/// arena-native is what preserves the allocation win end to end.
pub fn eval_arena(
    arena: &ExprArena,
    r: ExprRef,
    names: &HashMap<String, Value>,
    original_expr: &str,
) -> EvalResult<Value> {
    match arena.get(r) {
        ArenaNode::Literal(v) => Ok(v.clone()),
        ArenaNode::Name(name) => match name.as_str() {
            "True" => Ok(Value::Bool(true)),
            "False" => Ok(Value::Bool(false)),
            "None" => Ok(Value::None),
            _ => names
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::NameNotDefined {
                    name: name.clone(),
                    expr: original_expr.to_string(),
                }),
        },
        ArenaNode::UnaryOp { op, operand } => {
            let v = eval_arena(arena, *operand, names, original_expr)?;
            eval_unary(*op, v)
        }
        ArenaNode::BinOp { op, left, right } => {
            let l = eval_arena(arena, *left, names, original_expr)?;
            let rr = eval_arena(arena, *right, names, original_expr)?;
            eval_binop(*op, l, rr, original_expr)
        }
        ArenaNode::BoolOp { op, values } => match op {
            BoolOpKind::And => {
                let mut last = Value::Bool(true);
                for val in values.iter() {
                    let v = eval_arena(arena, *val, names, original_expr)?;
                    if !value_is_truthy(&v) {
                        return Ok(v);
                    }
                    last = v;
                }
                Ok(last)
            }
            BoolOpKind::Or => {
                let mut last = Value::Bool(false);
                for val in values.iter() {
                    let v = eval_arena(arena, *val, names, original_expr)?;
                    if value_is_truthy(&v) {
                        return Ok(v);
                    }
                    last = v;
                }
                Ok(last)
            }
        },
        ArenaNode::Compare { left, op, right } => {
            let l = eval_arena(arena, *left, names, original_expr)?;
            let rr = eval_arena(arena, *right, names, original_expr)?;
            eval_compare(*op, &l, &rr)
        }
        ArenaNode::IfExp { test, body, orelse } => {
            let t = eval_arena(arena, *test, names, original_expr)?;
            if value_is_truthy(&t) {
                eval_arena(arena, *body, names, original_expr)
            } else {
                eval_arena(arena, *orelse, names, original_expr)
            }
        }
        ArenaNode::Call { func, args } => {
            let mut argv = Vec::with_capacity(args.len());
            for a in args {
                argv.push(eval_arena(arena, *a, names, original_expr)?);
            }
            call_builtin(func, argv, original_expr)
        }
        ArenaNode::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval_arena(arena, *it, names, original_expr)?);
            }
            Ok(Value::List(out))
        }
        ArenaNode::Tuple(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(eval_arena(arena, *it, names, original_expr)?);
            }
            Ok(Value::Tuple(out))
        }
        ArenaNode::Set(items) => {
            let mut out: Vec<Value> = Vec::with_capacity(items.len());
            for it in items {
                let v = eval_arena(arena, *it, names, original_expr)?;
                if !out.iter().any(|e| value_equal(e, &v)) {
                    out.push(v);
                }
            }
            Ok(Value::Set(out))
        }
        ArenaNode::Dict { keys, values } => {
            let mut pairs: Vec<(Value, Value)> = Vec::with_capacity(keys.len());
            for (k, v) in keys.iter().zip(values.iter()) {
                let kv = eval_arena(arena, *k, names, original_expr)?;
                let vv = eval_arena(arena, *v, names, original_expr)?;
                if let Some(slot) = pairs.iter_mut().find(|(ek, _)| value_equal(ek, &kv)) {
                    slot.1 = vv;
                } else {
                    pairs.push((kv, vv));
                }
            }
            Ok(Value::Dict(pairs))
        }
    }
}

fn eval_unary(op: UnaryOpKind, v: Value) -> EvalResult<Value> {
    match op {
        UnaryOpKind::Not => Ok(Value::Bool(!value_is_truthy(&v))),
        UnaryOpKind::Pos => match v {
            Value::Bool(b) => Ok(Value::Int(if b { 1 } else { 0 })),
            Value::Int(_) | Value::Int128(_) | Value::BigInt(_) | Value::Float(_) => Ok(v),
            _ => Err(EvalError::InvalidExpression(
                "bad operand type for unary +".into(),
            )),
        },
        UnaryOpKind::Neg => match v {
            Value::Bool(b) => Ok(Value::Int(if b { -1 } else { 0 })),
            Value::Int(n) => match n.checked_neg() {
                Some(r) => Ok(Value::Int(r)),
                None => Ok(Value::Int128(-(n as i128))),
            },
            Value::Int128(n) => match n.checked_neg() {
                Some(r) => Ok(Value::Int128(r)),
                None => Ok(Value::BigInt(-BigInt::from(n))),
            },
            Value::BigInt(b) => Ok(normalize_bigint(-b)),
            Value::Float(f) => Ok(Value::Float(-f)),
            _ => Err(EvalError::InvalidExpression(
                "bad operand type for unary -".into(),
            )),
        },
        UnaryOpKind::BitNot => match v {
            Value::Bool(b) => Ok(Value::Int(!(if b { 1i64 } else { 0i64 }))),
            Value::Int(n) => Ok(Value::Int(!n)),
            Value::Int128(n) => Ok(Value::Int128(!n)),
            Value::BigInt(b) => Ok(normalize_bigint(!b)),
            _ => Err(EvalError::InvalidExpression(
                "bad operand type for unary ~".into(),
            )),
        },
    }
}

fn eval_binop(op: BinOpKind, l: Value, r: Value, expr: &str) -> EvalResult<Value> {
    match op {
        BinOpKind::In => eval_membership(&l, &r, false),
        BinOpKind::NotIn => eval_membership(&l, &r, true),
        BinOpKind::Is => Ok(Value::Bool(value_equal(&l, &r))),
        BinOpKind::IsNot => Ok(Value::Bool(!value_equal(&l, &r))),
        BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor => eval_bitwise(op, &l, &r),
        BinOpKind::LShift | BinOpKind::RShift => eval_shift(op, &l, &r),
        _ => arith(op, &l, &r, expr),
    }
}

fn eval_compare(op: CmpOpKind, l: &Value, r: &Value) -> EvalResult<Value> {
    match op {
        CmpOpKind::Eq => Ok(Value::Bool(value_equal(l, r))),
        CmpOpKind::NotEq => Ok(Value::Bool(!value_equal(l, r))),
        _ => {
            let ord = compare_values(l, r)?;
            let res = match op {
                CmpOpKind::Lt => ord == std::cmp::Ordering::Less,
                CmpOpKind::Gt => ord == std::cmp::Ordering::Greater,
                CmpOpKind::LtE => ord != std::cmp::Ordering::Greater,
                CmpOpKind::GtE => ord != std::cmp::Ordering::Less,
                _ => unreachable!(),
            };
            Ok(Value::Bool(res))
        }
    }
}

fn eval_membership(needle: &Value, haystack: &Value, negate: bool) -> EvalResult<Value> {
    let found = match haystack {
        Value::Str(s) => match needle {
            Value::Str(sub) => s.contains(sub.as_str()),
            _ => {
                return Err(EvalError::InvalidExpression(
                    "Sorry, 'in <string>' requires string as left operand".into(),
                ))
            }
        },
        Value::List(v) | Value::Tuple(v) | Value::Set(v) => {
            v.iter().any(|e| value_equal(e, needle))
        }
        Value::Dict(d) => d.iter().any(|(k, _)| value_equal(k, needle)),
        _ => {
            return Err(EvalError::InvalidExpression(
                "Sorry, argument of type is not iterable".into(),
            ))
        }
    };
    Ok(Value::Bool(if negate { !found } else { found }))
}

// ── Numeric helpers ─────────────────────────────────────────────────────────

#[derive(Clone)]
enum IntVal {
    I64(i64),
    I128(i128),
    Big(BigInt),
}

fn as_int(v: &Value) -> Option<IntVal> {
    match v {
        Value::Bool(b) => Some(IntVal::I64(if *b { 1 } else { 0 })),
        Value::Int(n) => Some(IntVal::I64(*n)),
        Value::Int128(n) => Some(IntVal::I128(*n)),
        Value::BigInt(b) => Some(IntVal::Big(b.clone())),
        _ => None,
    }
}

fn intval_to_bigint(iv: &IntVal) -> BigInt {
    match iv {
        IntVal::I64(n) => BigInt::from(*n),
        IntVal::I128(n) => BigInt::from(*n),
        IntVal::Big(b) => b.clone(),
    }
}

fn intval_to_f64(iv: &IntVal) -> f64 {
    match iv {
        IntVal::I64(n) => *n as f64,
        IntVal::I128(n) => *n as f64,
        IntVal::Big(b) => bigint_to_f64(b),
    }
}

/// Demote a BigInt back to i64/i128 when it fits, else keep BigInt.
fn normalize_bigint(b: BigInt) -> Value {
    if let Some(n) = b.to_i64() {
        Value::Int(n)
    } else if let Some(n) = b.to_i128() {
        Value::Int128(n)
    } else {
        Value::BigInt(b)
    }
}

fn is_float(v: &Value) -> bool {
    matches!(v, Value::Float(_))
}

fn arith(op: BinOpKind, l: &Value, r: &Value, expr: &str) -> EvalResult<Value> {
    // String operations
    if let (Value::Str(a), Value::Str(b)) = (l, r) {
        if op == BinOpKind::Add {
            if a.chars().count() + b.chars().count() > MAX_STRING_LENGTH {
                return Err(EvalError::IterableTooLong(
                    "Sorry, adding those two together would make something too long.".into(),
                ));
            }
            return Ok(Value::Str(format!("{}{}", a, b)));
        }
    }
    // String repetition
    if op == BinOpKind::Mul {
        if let (Value::Str(s), Some(iv)) = (l, as_int(r)) {
            return str_repeat(s, &iv);
        }
        if let (Some(iv), Value::Str(s)) = (as_int(l), r) {
            return str_repeat(s, &iv);
        }
    }

    // Float path: if either operand is float, or op is true division.
    let li = as_int(l);
    let ri = as_int(r);

    if op == BinOpKind::Div {
        let a = number_to_f64(l, expr)?;
        let b = number_to_f64(r, expr)?;
        if b == 0.0 {
            return Err(EvalError::InvalidExpression(format!(
                "Sorry, I don't want to evaluate {} / {}",
                display_num(l),
                display_num(r)
            )));
        }
        return Ok(Value::Float(a / b));
    }

    let either_float = is_float(l) || is_float(r);

    match (li, ri, either_float) {
        (Some(a), Some(b), false) => int_arith(op, a, b, expr),
        _ => {
            // Float arithmetic
            let a = number_to_f64(l, expr)?;
            let b = number_to_f64(r, expr)?;
            float_arith(op, a, b, expr)
        }
    }
}

fn number_to_f64(v: &Value, _expr: &str) -> EvalResult<f64> {
    value_to_f64(v).map_err(|_| {
        EvalError::InvalidExpression("Sorry, unsupported operand type for arithmetic".into())
    })
}

fn display_num(v: &Value) -> String {
    format!("{}", v)
}

fn str_repeat(s: &str, iv: &IntVal) -> EvalResult<Value> {
    let count = match iv {
        IntVal::I64(n) => *n,
        IntVal::I128(n) => {
            if *n > i64::MAX as i128 {
                i64::MAX
            } else if *n < 0 {
                0
            } else {
                *n as i64
            }
        }
        IntVal::Big(b) => b.to_i64().unwrap_or(i64::MAX),
    };
    if count <= 0 {
        return Ok(Value::Str(String::new()));
    }
    let len = s.chars().count();
    if (count as usize).saturating_mul(len) > MAX_STRING_LENGTH {
        return Err(EvalError::IterableTooLong(
            "Sorry, I will not evaluate something that long.".into(),
        ));
    }
    Ok(Value::Str(s.repeat(count as usize)))
}

fn int_arith(op: BinOpKind, a: IntVal, b: IntVal, expr: &str) -> EvalResult<Value> {
    match op {
        BinOpKind::Add => Ok(int_add(a, b)),
        BinOpKind::Sub => Ok(int_sub(a, b)),
        BinOpKind::Mul => Ok(int_mul(a, b)),
        BinOpKind::FloorDiv => int_floordiv(a, b, expr),
        BinOpKind::Mod => int_mod(a, b, expr),
        BinOpKind::Pow => int_pow(a, b, expr),
        _ => Err(EvalError::OperatorNotDefined {
            op: format!("{:?}", op),
            expr: expr.to_string(),
        }),
    }
}

fn int_add(a: IntVal, b: IntVal) -> Value {
    if let (IntVal::I64(x), IntVal::I64(y)) = (&a, &b) {
        if let Some(r) = x.checked_add(*y) {
            return Value::Int(r);
        }
    }
    if let (Some(x), Some(y)) = (intval_i128(&a), intval_i128(&b)) {
        if let Some(r) = x.checked_add(y) {
            return Value::Int128(r);
        }
    }
    normalize_bigint(intval_to_bigint(&a) + intval_to_bigint(&b))
}

fn int_sub(a: IntVal, b: IntVal) -> Value {
    if let (IntVal::I64(x), IntVal::I64(y)) = (&a, &b) {
        if let Some(r) = x.checked_sub(*y) {
            return Value::Int(r);
        }
    }
    if let (Some(x), Some(y)) = (intval_i128(&a), intval_i128(&b)) {
        if let Some(r) = x.checked_sub(y) {
            return Value::Int128(r);
        }
    }
    normalize_bigint(intval_to_bigint(&a) - intval_to_bigint(&b))
}

fn int_mul(a: IntVal, b: IntVal) -> Value {
    if let (IntVal::I64(x), IntVal::I64(y)) = (&a, &b) {
        if let Some(r) = x.checked_mul(*y) {
            return Value::Int(r);
        }
    }
    if let (Some(x), Some(y)) = (intval_i128(&a), intval_i128(&b)) {
        if let Some(r) = x.checked_mul(y) {
            return Value::Int128(r);
        }
    }
    normalize_bigint(intval_to_bigint(&a) * intval_to_bigint(&b))
}

fn intval_i128(iv: &IntVal) -> Option<i128> {
    match iv {
        IntVal::I64(n) => Some(*n as i128),
        IntVal::I128(n) => Some(*n),
        IntVal::Big(b) => b.to_i128(),
    }
}

fn int_floordiv(a: IntVal, b: IntVal, _expr: &str) -> EvalResult<Value> {
    let bb = intval_to_bigint(&b);
    if bb.is_zero() {
        return Err(EvalError::InvalidExpression(format!(
            "Sorry, I don't want to evaluate {} // {}",
            intval_to_bigint(&a),
            bb
        )));
    }
    // Fast i64 path
    if let (IntVal::I64(x), IntVal::I64(y)) = (&a, &b) {
        return Ok(Value::Int(py_floordiv_i64(*x, *y)));
    }
    let aa = intval_to_bigint(&a);
    let (q, _r) = py_floordiv_bigint(&aa, &bb);
    Ok(normalize_bigint(q))
}

fn py_floordiv_i64(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn py_floordiv_bigint(a: &BigInt, b: &BigInt) -> (BigInt, BigInt) {
    // num-integer div_floor / mod_floor give Python semantics.
    (a.div_floor(b), a.mod_floor(b))
}

fn int_mod(a: IntVal, b: IntVal, _expr: &str) -> EvalResult<Value> {
    let bb = intval_to_bigint(&b);
    if bb.is_zero() {
        return Err(EvalError::InvalidExpression(format!(
            "Sorry, I don't want to evaluate {} % {}",
            intval_to_bigint(&a),
            bb
        )));
    }
    if let (IntVal::I64(x), IntVal::I64(y)) = (&a, &b) {
        return Ok(Value::Int(py_mod_i64(*x, *y)));
    }
    let aa = intval_to_bigint(&a);
    Ok(normalize_bigint(aa.mod_floor(&bb)))
}

fn py_mod_i64(a: i64, b: i64) -> i64 {
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        r + b
    } else {
        r
    }
}

fn int_pow(a: IntVal, b: IntVal, _expr: &str) -> EvalResult<Value> {
    // MAX_POWER safety on magnitudes
    let a_big = intval_to_bigint(&a);
    let b_big = intval_to_bigint(&b);
    let max = BigInt::from(MAX_POWER);
    if a_big.abs() > max || b_big.abs() > max {
        return Err(EvalError::NumberTooHigh(format!(
            "Sorry! I don't want to evaluate {} ** {}",
            a_big, b_big
        )));
    }
    if b_big.is_negative() {
        // negative exponent → float result
        let af = intval_to_f64(&a);
        let bf = intval_to_f64(&b);
        return Ok(Value::Float(af.powf(bf)));
    }
    let exp = b_big.to_u32().unwrap_or(u32::MAX);
    let result = bigint_pow(&a_big, exp);
    Ok(normalize_bigint(result))
}

fn bigint_pow(base: &BigInt, mut exp: u32) -> BigInt {
    let mut result = BigInt::from(1);
    let mut b = base.clone();
    while exp > 0 {
        if exp & 1 == 1 {
            result *= &b;
        }
        exp >>= 1;
        if exp > 0 {
            b = &b * &b;
        }
    }
    result
}

fn float_arith(op: BinOpKind, a: f64, b: f64, expr: &str) -> EvalResult<Value> {
    match op {
        BinOpKind::Add => Ok(Value::Float(a + b)),
        BinOpKind::Sub => Ok(Value::Float(a - b)),
        BinOpKind::Mul => Ok(Value::Float(a * b)),
        BinOpKind::FloorDiv => {
            if b == 0.0 {
                return Err(EvalError::InvalidExpression(format!(
                    "Sorry, I don't want to evaluate {} // {}",
                    format_float(a),
                    format_float(b)
                )));
            }
            Ok(Value::Float((a / b).floor()))
        }
        BinOpKind::Mod => {
            if b == 0.0 {
                return Err(EvalError::InvalidExpression(format!(
                    "Sorry, I don't want to evaluate {} % {}",
                    format_float(a),
                    format_float(b)
                )));
            }
            let r = a - (a / b).floor() * b;
            Ok(Value::Float(r))
        }
        BinOpKind::Pow => {
            if a.abs() > MAX_POWER as f64 || b.abs() > MAX_POWER as f64 {
                return Err(EvalError::NumberTooHigh(format!(
                    "Sorry! I don't want to evaluate {} ** {}",
                    format_float(a),
                    format_float(b)
                )));
            }
            Ok(Value::Float(a.powf(b)))
        }
        _ => Err(EvalError::OperatorNotDefined {
            op: format!("{:?}", op),
            expr: expr.to_string(),
        }),
    }
}

fn eval_bitwise(op: BinOpKind, l: &Value, r: &Value) -> EvalResult<Value> {
    let a = as_int(l).ok_or_else(|| {
        EvalError::InvalidExpression("Sorry, unsupported operand type for bitwise op".into())
    })?;
    let b = as_int(r).ok_or_else(|| {
        EvalError::InvalidExpression("Sorry, unsupported operand type for bitwise op".into())
    })?;
    // i64 fast path
    if let (IntVal::I64(x), IntVal::I64(y)) = (&a, &b) {
        let res = match op {
            BinOpKind::BitAnd => x & y,
            BinOpKind::BitOr => x | y,
            BinOpKind::BitXor => x ^ y,
            _ => unreachable!(),
        };
        return Ok(Value::Int(res));
    }
    let ab = intval_to_bigint(&a);
    let bb = intval_to_bigint(&b);
    let res = match op {
        BinOpKind::BitAnd => ab & bb,
        BinOpKind::BitOr => ab | bb,
        BinOpKind::BitXor => ab ^ bb,
        _ => unreachable!(),
    };
    Ok(normalize_bigint(res))
}

fn eval_shift(op: BinOpKind, l: &Value, r: &Value) -> EvalResult<Value> {
    let a = as_int(l).ok_or_else(|| {
        EvalError::InvalidExpression("Sorry, unsupported operand type for shift".into())
    })?;
    let b = as_int(r).ok_or_else(|| {
        EvalError::InvalidExpression("Sorry, unsupported operand type for shift".into())
    })?;

    let a_big = intval_to_bigint(&a);
    let b_big = intval_to_bigint(&b);
    let shift_sym = if op == BinOpKind::LShift { "<<" } else { ">>" };

    // Shift amount safety
    let b_abs = b_big.abs();
    if b_abs > BigInt::from(MAX_SHIFT) {
        return Err(EvalError::NumberTooHigh(format!(
            "Sorry! I don't want to evaluate {} {} {}",
            a_big, shift_sym, b_big
        )));
    }
    // Base magnitude safety
    let a_f = bigint_to_f64(&a_big.abs());
    if a_f > MAX_SHIFT_BASE {
        return Err(EvalError::NumberTooHigh(format!(
            "Sorry! I don't want to evaluate {} {} {}",
            a_big, shift_sym, b_big
        )));
    }

    let shift = b_big.to_i64().unwrap_or(0);
    if shift < 0 {
        return Err(EvalError::InvalidExpression(
            "Sorry, negative shift count".into(),
        ));
    }
    let shift_u = shift as usize;

    let res = match op {
        BinOpKind::LShift => a_big << shift_u,
        BinOpKind::RShift => a_big >> shift_u,
        _ => unreachable!(),
    };
    Ok(normalize_bigint(res))
}

// ── Equality and ordering ───────────────────────────────────────────────────

fn numeric_equal(l: &Value, r: &Value) -> Option<bool> {
    let l_num = as_int(l).map(|_| ()).is_some() || is_float(l);
    let r_num = as_int(r).map(|_| ()).is_some() || is_float(r);
    if !(l_num && r_num) {
        return None;
    }
    if is_float(l) || is_float(r) {
        return Some(value_to_f64(l).ok()? == value_to_f64(r).ok()?);
    }
    let lb = intval_to_bigint(&as_int(l)?);
    let rb = intval_to_bigint(&as_int(r)?);
    Some(lb == rb)
}

fn value_equal(l: &Value, r: &Value) -> bool {
    // Numeric cross-type equality (1 == 1.0, True == 1)
    if let Some(eq) = numeric_equal(l, r) {
        return eq;
    }
    match (l, r) {
        (Value::Str(a), Value::Str(b)) => a == b,
        (Value::None, Value::None) => true,
        (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| value_equal(x, y))
        }
        (Value::Set(a), Value::Set(b)) => {
            a.len() == b.len()
                && a.iter().all(|x| b.iter().any(|y| value_equal(x, y)))
        }
        (Value::Dict(a), Value::Dict(b)) => {
            a.len() == b.len()
                && a.iter().all(|(k, v)| {
                    b.iter()
                        .any(|(k2, v2)| value_equal(k, k2) && value_equal(v, v2))
                })
        }
        _ => false,
    }
}

fn compare_values(l: &Value, r: &Value) -> EvalResult<std::cmp::Ordering> {
    use std::cmp::Ordering;
    // Numeric comparison
    let l_is_num = as_int(l).is_some() || is_float(l);
    let r_is_num = as_int(r).is_some() || is_float(r);
    if l_is_num && r_is_num {
        if is_float(l) || is_float(r) {
            let a = value_to_f64(l)?;
            let b = value_to_f64(r)?;
            return a
                .partial_cmp(&b)
                .ok_or_else(|| EvalError::InvalidExpression("Sorry, cannot compare NaN".into()));
        }
        let a = intval_to_bigint(&as_int(l).unwrap());
        let b = intval_to_bigint(&as_int(r).unwrap());
        return Ok(a.cmp(&b));
    }
    match (l, r) {
        (Value::Str(a), Value::Str(b)) => Ok(a.cmp(b)),
        (Value::List(a), Value::List(b)) | (Value::Tuple(a), Value::Tuple(b)) => {
            for (x, y) in a.iter().zip(b.iter()) {
                let ord = compare_values(x, y)?;
                if ord != Ordering::Equal {
                    return Ok(ord);
                }
            }
            Ok(a.len().cmp(&b.len()))
        }
        _ => Err(EvalError::InvalidExpression(
            "Sorry, unsupported operand types for comparison".into(),
        )),
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 11. PyO3 boundary
// ───────────────────────────────────────────────────────────────────────────

#[cfg(not(test))]
pub fn value_to_pyobject(py: Python<'_>, v: Value) -> PyResult<PyObject> {
    use pyo3::types::{PyAnyMethods, PyString};
    match v {
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Int(n) => Ok(n.into_py(py)),
        Value::Int128(n) => Ok(n.into_py(py)),
        Value::BigInt(b) => {
            // Convert via string → Python int()
            let s = b.to_string();
            let builtins = py.import_bound("builtins")?;
            let int_obj = builtins.getattr("int")?.call1((s,))?;
            Ok(int_obj.into_py(py))
        }
        Value::Float(f) => Ok(f.into_py(py)),
        Value::Str(s) => Ok(PyString::new_bound(py, &s).into_py(py)),
        Value::None => Ok(py.None()),
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(value_to_pyobject(py, it)?);
            }
            Ok(PyList::new_bound(py, out).into_py(py))
        }
        Value::Tuple(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(value_to_pyobject(py, it)?);
            }
            Ok(PyTuple::new_bound(py, out).into_py(py))
        }
        Value::Set(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(value_to_pyobject(py, it)?);
            }
            // Python set literals produce `set`. Build via builtins.set(list).
            let list = PyList::new_bound(py, out);
            let builtins = py.import_bound("builtins")?;
            let set_obj = builtins.getattr("set")?.call1((list,))?;
            Ok(set_obj.into_py(py))
        }
        Value::Dict(pairs) => {
            let d = PyDict::new_bound(py);
            for (k, val) in pairs {
                let pk = value_to_pyobject(py, k)?;
                let pv = value_to_pyobject(py, val)?;
                d.set_item(pk, pv)?;
            }
            Ok(d.into_py(py))
        }
    }
}

#[cfg(not(test))]
pub fn eval_error_to_pyerr(py: Python<'_>, e: EvalError) -> PyErr {
    use pyo3::types::PyAnyMethods;

    // All fallible work happens inside this closure which returns PyResult<PyErr>.
    // If importing simpleeval or constructing the exception fails, we surface
    // that import error instead.
    let build = || -> PyResult<PyErr> {
        let simpleeval = py.import_bound("simpleeval")?;
        let make = |name: &str, args: (Bound<'_, pyo3::PyAny>,)| -> PyResult<PyErr> {
            let cls = simpleeval.getattr(name)?;
            let inst = cls.call1(args)?;
            Ok(PyErr::from_value_bound(inst))
        };
        match e {
            EvalError::NameNotDefined { name, expr } => {
                let cls = simpleeval.getattr("NameNotDefined")?;
                let inst = cls.call1((name, expr))?;
                Ok(PyErr::from_value_bound(inst))
            }
            EvalError::FunctionNotDefined { func_name, expr } => {
                let cls = simpleeval.getattr("FunctionNotDefined")?;
                let inst = cls.call1((func_name, expr))?;
                Ok(PyErr::from_value_bound(inst))
            }
            EvalError::OperatorNotDefined { op, expr } => {
                let cls = simpleeval.getattr("OperatorNotDefined")?;
                let inst = cls.call1((op, expr))?;
                Ok(PyErr::from_value_bound(inst))
            }
            EvalError::AttributeDoesNotExist { attr, expr } => {
                let cls = simpleeval.getattr("AttributeDoesNotExist")?;
                let inst = cls.call1((attr, expr))?;
                Ok(PyErr::from_value_bound(inst))
            }
            EvalError::FeatureNotAvailable(msg) => {
                make("FeatureNotAvailable", (msg.into_py(py).into_bound(py),))
            }
            EvalError::NumberTooHigh(msg) => {
                make("NumberTooHigh", (msg.into_py(py).into_bound(py),))
            }
            EvalError::IterableTooLong(msg) => {
                make("IterableTooLong", (msg.into_py(py).into_bound(py),))
            }
            EvalError::InvalidExpression(msg) => {
                make("InvalidExpression", (msg.into_py(py).into_bound(py),))
            }
        }
    };

    match build() {
        Ok(err) => err,
        Err(import_err) => import_err,
    }
}

#[cfg(not(test))]
pub fn pydict_to_names(py: Python<'_>, d: &PyDict) -> PyResult<HashMap<String, Value>> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
    let mut map = HashMap::with_capacity(d.len());
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        // bool MUST be checked before int (bool is subtype of int)
        let val = if v.is_none() {
            Value::None
        } else if v.is_instance_of::<PyBool>() {
            Value::Bool(v.extract::<bool>()?)
        } else if v.is_instance_of::<PyInt>() {
            if let Ok(n) = v.extract::<i64>() {
                Value::Int(n)
            } else if let Ok(n) = v.extract::<i128>() {
                Value::Int128(n)
            } else {
                let s: String = v.str()?.to_str()?.to_string();
                let b: BigInt = s.parse().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("invalid integer in names")
                })?;
                Value::BigInt(b)
            }
        } else if v.is_instance_of::<PyFloat>() {
            Value::Float(v.extract::<f64>()?)
        } else if v.is_instance_of::<PyString>() {
            Value::Str(v.extract::<String>()?)
        } else {
            // Should never happen: the Python guard validates primitives.
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "names values must be primitives",
            ));
        };
        map.insert(key, val);
    }
    Ok(map)
}

#[cfg(not(test))]
#[pyfunction]
pub fn fast_eval(py: Python<'_>, expr: &str, names: &PyDict) -> PyResult<PyObject> {
    let names_map = match pydict_to_names(py, names) {
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let run = || -> EvalResult<Value> {
        // G014: parse into an arena (cached) and evaluate it in place. No
        // materialization to `Box<Expr>` on this path, so the arena allocation
        // win is preserved end to end for the production hot path.
        let cached = parse_cached_arena(expr)?;
        let (arena, root) = &*cached;
        eval_arena(arena, *root, &names_map, expr)
    };
    match run() {
        Ok(v) => value_to_pyobject(py, v),
        Err(e) => Err(eval_error_to_pyerr(py, e)),
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 12. Module registration
// ───────────────────────────────────────────────────────────────────────────

#[cfg(not(test))]
#[pymodule]
fn _gaia_simpleeval(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_eval, m)?)?;
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(s: &str) -> Value {
        let toks = tokenize(s).expect("tokenize");
        let ast = parse(&toks, s).expect("parse");
        let names = HashMap::new();
        eval_expr(&ast, &names, s).expect("eval")
    }

    fn ev_names(s: &str, names: &HashMap<String, Value>) -> Value {
        let toks = tokenize(s).expect("tokenize");
        let ast = parse(&toks, s).expect("parse");
        eval_expr(&ast, names, s).expect("eval")
    }

    fn err(s: &str) -> EvalError {
        let toks = match tokenize(s) {
            Ok(t) => t,
            Err(e) => return e,
        };
        let ast = match parse(&toks, s) {
            Ok(a) => a,
            Err(e) => return e,
        };
        let names = HashMap::new();
        eval_expr(&ast, &names, s).unwrap_err()
    }

    // ── cache: parse_cached correctness ──
    #[test]
    fn cache_matches_uncached_ast() {
        ast_cache_clear();
        for s in ["2+2", "x+y", "1<2 and 3>1", "1 + 2 * 3 - 4 / 2", "a | b ^ c & d"] {
            let toks = tokenize(s).expect("tokenize");
            let direct = parse(&toks, s).expect("parse");
            let cached = parse_cached(s).expect("parse_cached");
            assert_eq!(*cached, direct, "cached AST differs for {s:?}");
        }
    }

    #[test]
    fn cache_hit_returns_same_rc() {
        ast_cache_clear();
        let a = parse_cached("3*4+5").unwrap();
        let b = parse_cached("3*4+5").unwrap();
        // Second call must be a hit: same allocation, not a re-parse.
        assert!(Rc::ptr_eq(&a, &b), "expected cache hit to reuse Rc");
    }

    #[test]
    fn cache_eval_matches_repeatedly() {
        ast_cache_clear();
        let names = HashMap::new();
        for _ in 0..3 {
            let ast = parse_cached("2 ** 10 - 1").unwrap();
            let v = eval_expr(&ast, &names, "2 ** 10 - 1").unwrap();
            assert!(value_equal(&v, &Value::Int(1023)));
        }
    }

    #[test]
    fn cache_propagates_parse_errors_and_does_not_store_them() {
        ast_cache_clear();
        assert!(parse_cached("1 +").is_err());
        // A subsequent valid parse of a different expr still works and is cached.
        let ok = parse_cached("1 + 1").unwrap();
        assert!(value_equal(
            &eval_expr(&ok, &HashMap::new(), "1 + 1").unwrap(),
            &Value::Int(2)
        ));
    }

    #[test]
    fn cache_evicts_when_full() {
        ast_cache_clear();
        // Fill beyond capacity; should clear rather than grow unbounded.
        for i in 0..(AST_CACHE_CAP + 50) {
            let s = format!("{i} + {i}");
            let _ = parse_cached(&s).unwrap();
        }
        // After overflow the map was cleared at least once; a fresh insert works.
        let v = parse_cached("7 + 8").unwrap();
        assert!(value_equal(
            &eval_expr(&v, &HashMap::new(), "7 + 8").unwrap(),
            &Value::Int(15)
        ));
    }

    // ── unit_01: types ──
    #[test]
    fn unit_01_value_variants_distinct() {
        // Bool(true) and Int(1) are numerically equal under value_equal, but are
        // nonetheless distinct enum variants.
        match (Value::Bool(true), Value::Int(1)) {
            (Value::Bool(_), Value::Int(_)) => {}
            _ => panic!("variants not distinct"),
        }
        // Distinct from None / Float.
        assert!(!value_equal(&Value::None, &Value::Int(0)));
    }

    #[test]
    fn unit_01_displays() {
        assert_eq!(format!("{}", Value::None), "None");
        assert_eq!(format!("{}", Value::Float(1.0)), "1.0");
        assert_eq!(format!("{}", Value::Float(1e100)), "1e+100");
        assert_eq!(format!("{}", Value::Bool(true)), "True");
    }

    #[test]
    fn unit_01_constants() {
        assert_eq!(MAX_STRING_LENGTH, 100_000);
        assert_eq!(MAX_COMPREHENSION_LENGTH, 10_000);
        assert_eq!(MAX_POWER, 4_000_000);
        assert_eq!(MAX_SHIFT, 10_000);
        assert_eq!(MAX_SHIFT_BASE, f64::MAX);
    }

    // ── unit_02: lexer ──
    #[test]
    fn unit_02_number_bases() {
        assert_eq!(tokenize("0xFF").unwrap()[0], Token::IntLit(255));
        assert_eq!(tokenize("0b1010").unwrap()[0], Token::IntLit(10));
        assert_eq!(tokenize("0o17").unwrap()[0], Token::IntLit(15));
    }

    #[test]
    fn unit_02_float_and_strings() {
        assert_eq!(tokenize("1e10").unwrap()[0], Token::FloatLit(1e10));
        assert_eq!(
            tokenize("\"hello\"").unwrap()[0],
            Token::StrLit("hello".into())
        );
    }

    #[test]
    fn unit_02_keywords() {
        assert_eq!(tokenize("True").unwrap()[0], Token::BoolLit(true));
        assert_eq!(tokenize("None").unwrap()[0], Token::NoneLit);
        assert_eq!(tokenize("not").unwrap()[0], Token::Not);
        assert_eq!(tokenize("is").unwrap()[0], Token::Is);
        assert_eq!(tokenize("in").unwrap()[0], Token::In);
        assert_eq!(tokenize("and").unwrap()[0], Token::And);
        assert_eq!(tokenize("or").unwrap()[0], Token::Or);
        assert_eq!(tokenize("if").unwrap()[0], Token::If);
        assert_eq!(tokenize("else").unwrap()[0], Token::Else);
    }

    #[test]
    fn unit_02_fstring_and_empty() {
        assert!(matches!(
            tokenize("f\"x\""),
            Err(EvalError::InvalidExpression(_))
        ));
        assert!(matches!(tokenize(""), Err(EvalError::InvalidExpression(_))));
        assert!(matches!(
            tokenize("   "),
            Err(EvalError::InvalidExpression(_))
        ));
    }

    #[test]
    fn unit_02_string_too_long() {
        let big = format!("\"{}\"", "a".repeat(MAX_STRING_LENGTH + 1));
        assert!(matches!(
            tokenize(&big),
            Err(EvalError::IterableTooLong(_))
        ));
    }

    #[test]
    fn unit_02_bigint_path() {
        // 40 digits exceeds i128 range (~38 digits) → BigIntLit.
        let s = "1234567890123456789012345678901234567890";
        match &tokenize(s).unwrap()[0] {
            Token::BigIntLit(_) => {}
            other => panic!("expected BigIntLit, got {:?}", other),
        }
    }

    // ── unit_03: parser ──
    #[test]
    fn unit_03_precedence() {
        // 1 + 2 * 3 → Add(1, Mul(2,3))
        let toks = tokenize("1 + 2 * 3").unwrap();
        let ast = parse(&toks, "1 + 2 * 3").unwrap();
        match ast {
            Expr::BinOp {
                op: BinOpKind::Add,
                right,
                ..
            } => match *right {
                Expr::BinOp {
                    op: BinOpKind::Mul,
                    ..
                } => {}
                _ => panic!("rhs not Mul"),
            },
            _ => panic!("top not Add"),
        }
    }

    #[test]
    fn unit_03_chained_comparison() {
        let toks = tokenize("1 < 2 < 3").unwrap();
        assert!(matches!(
            parse(&toks, "1 < 2 < 3"),
            Err(EvalError::InvalidExpression(_))
        ));
    }

    #[test]
    fn unit_03_attribute_subscript() {
        assert!(matches!(
            tokenize("x.y"),
            Err(EvalError::FeatureNotAvailable(_))
        ));
        let toks = tokenize("x[0]").unwrap();
        assert!(matches!(
            parse(&toks, "x[0]"),
            Err(EvalError::InvalidExpression(_))
        ));
    }

    #[test]
    fn unit_03_comprehension_lambda() {
        // 'for' keyword caught in lexer
        assert!(matches!(
            tokenize("[x for x in y]"),
            Err(EvalError::FeatureNotAvailable(_))
        ));
        assert!(matches!(
            tokenize("lambda x: x"),
            Err(EvalError::FeatureNotAvailable(_))
        ));
    }

    #[test]
    fn unit_03_containers() {
        let toks = tokenize("(1,)").unwrap();
        assert!(matches!(parse(&toks, "(1,)").unwrap(), Expr::Tuple(v) if v.len() == 1));
        let toks = tokenize("()").unwrap();
        assert!(matches!(parse(&toks, "()").unwrap(), Expr::Tuple(v) if v.is_empty()));
        let toks = tokenize("{}").unwrap();
        assert!(matches!(parse(&toks, "{}").unwrap(), Expr::Dict { keys, .. } if keys.is_empty()));
        let toks = tokenize("{1}").unwrap();
        assert!(matches!(parse(&toks, "{1}").unwrap(), Expr::Set(v) if v.len() == 1));
        let toks = tokenize("{1:2}").unwrap();
        assert!(matches!(parse(&toks, "{1:2}").unwrap(), Expr::Dict { keys, .. } if keys.len() == 1));
    }

    #[test]
    fn unit_03_misc_nodes() {
        let toks = tokenize("a if b else c").unwrap();
        assert!(matches!(
            parse(&toks, "a if b else c").unwrap(),
            Expr::IfExp { .. }
        ));
        let toks = tokenize("not a").unwrap();
        assert!(matches!(
            parse(&toks, "not a").unwrap(),
            Expr::UnaryOp {
                op: UnaryOpKind::Not,
                ..
            }
        ));
        let toks = tokenize("a is not b").unwrap();
        assert!(matches!(
            parse(&toks, "a is not b").unwrap(),
            Expr::BinOp {
                op: BinOpKind::IsNot,
                ..
            }
        ));
        let toks = tokenize("a not in b").unwrap();
        assert!(matches!(
            parse(&toks, "a not in b").unwrap(),
            Expr::BinOp {
                op: BinOpKind::NotIn,
                ..
            }
        ));
    }

    // ── unit_04: evaluator ──
    #[test]
    fn unit_04_floordiv_mod() {
        assert_eq!(ev("7 // 2"), Value::Int(3));
        assert_eq!(ev("-7 // 2"), Value::Int(-4));
        assert_eq!(ev("7 // -2"), Value::Int(-4));
        assert_eq!(ev("-7 // -2"), Value::Int(3));
        assert_eq!(ev("-7 % 3"), Value::Int(2));
        assert_eq!(ev("7 % -3"), Value::Int(-2));
    }

    #[test]
    fn unit_04_boolean_shortcircuit() {
        assert_eq!(ev("0 and 1"), Value::Int(0));
        assert_eq!(ev("1 and 2"), Value::Int(2));
        assert_eq!(ev("0 or 3"), Value::Int(3));
        assert_eq!(ev("1 or 3"), Value::Int(1));
        assert_eq!(ev("not 0"), Value::Bool(true));
    }

    #[test]
    fn unit_04_identity_membership() {
        assert_eq!(ev("None is None"), Value::Bool(true));
        assert_eq!(ev("1 is 1"), Value::Bool(true));
        assert_eq!(ev("\"a\" in \"abc\""), Value::Bool(true));
        assert_eq!(ev("1 in [1,2]"), Value::Bool(true));
        assert_eq!(ev("\"a\" in {\"a\":1}"), Value::Bool(true));
    }

    #[test]
    fn unit_04_overflow_tiers() {
        // i64 overflow → i128
        match ev("9223372036854775807 + 1") {
            Value::Int128(_) => {}
            other => panic!("expected Int128, got {:?}", other),
        }
        // i128 overflow → BigInt
        match ev("2 ** 200") {
            Value::BigInt(_) => {}
            other => panic!("expected BigInt, got {:?}", other),
        }
    }

    #[test]
    fn unit_04_safety_limits() {
        assert!(matches!(err("10 ** 5000000"), EvalError::NumberTooHigh(_)));
        assert!(matches!(err("1 << 20000"), EvalError::NumberTooHigh(_)));
        let s = format!("\"{}\" + \"{}\"", "a".repeat(60000), "b".repeat(60000));
        assert!(matches!(err(&s), EvalError::IterableTooLong(_)));
    }

    #[test]
    fn unit_04_arithmetic_basic() {
        assert_eq!(ev("2+2"), Value::Int(4));
        assert_eq!(ev("2+2*3"), Value::Int(8));
        assert_eq!(ev("2**10"), Value::Int(1024));
        match ev("10/3") {
            Value::Float(f) => assert!((f - 3.3333333333).abs() < 1e-6),
            other => panic!("expected float, got {:?}", other),
        }
        assert_eq!(ev("10//3"), Value::Int(3));
        assert_eq!(ev("10%3"), Value::Int(1));
    }

    #[test]
    fn unit_04_bitwise_shift() {
        assert_eq!(ev("5&3"), Value::Int(1));
        assert_eq!(ev("5|3"), Value::Int(7));
        assert_eq!(ev("5^3"), Value::Int(6));
        assert_eq!(ev("~5"), Value::Int(-6));
        assert_eq!(ev("1<<3"), Value::Int(8));
        assert_eq!(ev("8>>2"), Value::Int(2));
    }

    #[test]
    fn unit_04_string_ops_ternary_names() {
        assert_eq!(ev("\"a\"+\"b\""), Value::Str("ab".into()));
        assert_eq!(ev("\"ab\"*3"), Value::Str("ababab".into()));
        assert_eq!(ev("1 if True else 2"), Value::Int(1));
        let mut names = HashMap::new();
        names.insert("x".to_string(), Value::Int(5));
        assert_eq!(ev_names("x+1", &names), Value::Int(6));
    }

    #[test]
    fn unit_04_div_by_zero() {
        assert!(matches!(err("1//0"), EvalError::InvalidExpression(_)));
        assert!(matches!(err("1%0"), EvalError::InvalidExpression(_)));
        assert!(matches!(err("1/0"), EvalError::InvalidExpression(_)));
    }

    // ── unit_05: builtins ──
    #[test]
    fn unit_05_int_float() {
        assert_eq!(ev("int(3.9)"), Value::Int(3));
        assert_eq!(ev("int(-3.9)"), Value::Int(-3));
        assert_eq!(ev("int('  42  ')"), Value::Int(42));
        assert!(matches!(err("int('abc')"), EvalError::InvalidExpression(_)));
        assert_eq!(ev("float('1.5')"), Value::Float(1.5));
        assert!(matches!(
            err("float('bad')"),
            EvalError::InvalidExpression(_)
        ));
    }

    #[test]
    fn unit_05_str_bool() {
        assert_eq!(ev("str(1.0)"), Value::Str("1.0".into()));
        assert_eq!(ev("str(1e100)"), Value::Str("1e+100".into()));
        assert_eq!(ev("str(True)"), Value::Str("True".into()));
        assert_eq!(ev("str(None)"), Value::Str("None".into()));
        assert_eq!(ev("bool(0)"), Value::Bool(false));
        assert_eq!(ev("bool('')"), Value::Bool(false));
        assert_eq!(ev("bool([])"), Value::Bool(false));
    }

    #[test]
    fn unit_05_round_bankers() {
        assert_eq!(ev("round(0.5)"), Value::Int(0));
        assert_eq!(ev("round(1.5)"), Value::Int(2));
        assert_eq!(ev("round(2.5)"), Value::Int(2));
        match ev("round(2.675, 2)") {
            Value::Float(f) => assert!((f - 2.67).abs() < 1e-9 || (f - 2.68).abs() < 1e-9),
            other => panic!("expected float, got {:?}", other),
        }
    }

    #[test]
    fn unit_05_minmax_len_sum() {
        assert_eq!(ev("min(3,1,2)"), Value::Int(1));
        assert_eq!(ev("max([1,2,3])"), Value::Int(3));
        assert_eq!(ev("len('hello')"), Value::Int(5));
        assert_eq!(ev("sum([1,2,3])"), Value::Int(6));
        assert_eq!(ev("sum([1,2.0])"), Value::Float(3.0));
    }

    #[test]
    fn unit_05_rand() {
        match ev("rand()") {
            Value::Float(f) => assert!((0.0..1.0).contains(&f)),
            other => panic!("expected float, got {:?}", other),
        }
        match ev("randint(10)") {
            Value::Int(n) => assert!((0..10).contains(&n)),
            other => panic!("expected int, got {:?}", other),
        }
    }

    #[test]
    fn unit_05_unknown_func() {
        assert!(matches!(
            err("foobar(1)"),
            EvalError::FunctionNotDefined { .. }
        ));
    }

    #[test]
    fn type_preservation() {
        // int op int → int (not bool)
        assert!(matches!(ev("2+2"), Value::Int(4)));
        // int op float → float
        assert!(matches!(ev("2+2.0"), Value::Float(_)));
        // bool op bool → int
        assert!(matches!(ev("True+True"), Value::Int(2)));
        // comparison → bool
        assert!(matches!(ev("1<2"), Value::Bool(true)));
    }
}



#[cfg(test)]
mod arena_correctness {
    use super::*;

    fn eval_via_arena(s: &str, names: &HashMap<String, Value>) -> EvalResult<Value> {
        let toks = tokenize(s)?;
        let (arena, root) = parse_into_arena(&toks, s)?;
        eval_arena(&arena, root, names, s)
    }

    #[test]
    fn arena_eval_matches_box_eval() {
        let mut names = HashMap::new();
        names.insert("a".into(), Value::Int(7));
        names.insert("b".into(), Value::Int(3));
        let cases = [
            "2 + 2", "1 + 2 * 3 - 4 / 5", "(a + b) * 2", "a > b and b > 0",
            "1 if a > b else 2", "min(a, b, 10)", "[a, b, a + b]",
            "{1: 'x', 2: 'y'}", "{1, 2, 2, 3}", "(a, b)", "2 ** 8",
            "a % b", "a // b", "not (a == b)", "a in [1, 7, 9]",
            "'ab' + 'cd'", "len('hello')", "sum([a, b])", "abs(-a)",
            "True and False or a > 0",
        ];
        for s in cases {
            let toks = tokenize(s).expect("tok");
            let boxed = parse(&toks, s).expect("parse");
            let box_res = eval_expr(&boxed, &names, s);
            let arena_res = eval_via_arena(s, &names);
            match (box_res, arena_res) {
                (Ok(a), Ok(b)) => assert!(value_equal(&a, &b), "mismatch for {s:?}: {a:?} vs {b:?}"),
                (Err(_), Err(_)) => {}
                (a, b) => panic!("ok/err disagreement for {s:?}: {a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn arena_cache_hit_reuses_rc_and_evals() {
        arena_cache_clear();
        let nm = HashMap::new();
        let a = parse_cached_arena("3*4+5").unwrap();
        let b = parse_cached_arena("3*4+5").unwrap();
        assert!(Rc::ptr_eq(&a, &b), "expected arena cache hit to reuse Rc");
        let (arena, root) = &*a;
        let v = eval_arena(arena, *root, &nm, "3*4+5").unwrap();
        assert!(value_equal(&v, &Value::Int(17)));
    }

    #[test]
    fn arena_cache_evicts_when_full() {
        arena_cache_clear();
        let nm = HashMap::new();
        for i in 0..(AST_CACHE_CAP + 50) {
            let s = format!("{i} + {i}");
            let rc = parse_cached_arena(&s).unwrap();
            let (arena, root) = &*rc;
            let _ = eval_arena(arena, *root, &nm, &s).unwrap();
        }
        let rc = parse_cached_arena("7 + 8").unwrap();
        let (arena, root) = &*rc;
        let v = eval_arena(arena, *root, &nm, "7 + 8").unwrap();
        assert!(value_equal(&v, &Value::Int(15)));
    }

    #[test]
    fn arena_grows_past_first_chunk() {
        // Force more nodes than one chunk (ARENA_CHUNK = 64) to exercise
        // multi-chunk growth and cross-chunk ExprRef resolution.
        let mut s = String::from("1");
        for _ in 0..80 {
            s.push_str(" + 1");
        }
        let nm = HashMap::new();
        let v = eval_via_arena(&s, &nm).unwrap();
        assert!(value_equal(&v, &Value::Int(81)));
    }

    #[test]
    fn arena_parse_errors_propagate() {
        assert!(parse_into_arena(&tokenize("1 +").unwrap(), "1 +").is_err());
        assert!(parse_cached_arena("(1 + ").is_err());
    }
}
