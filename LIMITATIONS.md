# Limitations of gaia-simpleeval

## Unsupported expression types

The following expression types are not supported by the Rust evaluator and
raise explicit errors (no silent fallback to Python):

- **Chained comparisons** (`1 < 2 < 3`): raises `InvalidExpression`.
  Use `1 < 2 and 2 < 3` instead.
- **Attribute access** (`x.y`): raises `FeatureNotAvailable`.
- **Subscript/indexing** (`x[0]`, `x['key']`, `x[1:3]`): raises
  `InvalidExpression`. List literals (`[1, 2, 3]`) are supported for use with
  the `in` operator, but they cannot be indexed.
- **F-strings** (`f"hello {name}"`): raises `InvalidExpression`.
  Use string concatenation instead.
- **Comprehensions** (`[x for x in y]`): raises `FeatureNotAvailable`.
- **Lambda functions**: raises `FeatureNotAvailable`.
- **Import statements**: raises `FeatureNotAvailable`.

No silent fallback occurs for these constructs when the fast path is taken.
This is a deliberate security decision: a silent fallback could bypass the
sandbox if a caller relies on one of these features being blocked. The error is
always explicit.

## Immutable safety constants

`MAX_STRING_LENGTH`, `MAX_POWER`, `MAX_SHIFT`, `MAX_SHIFT_BASE`, and
`MAX_COMPREHENSION_LENGTH` are hardcoded in the Rust evaluator and cannot be
modified at runtime. Attempting to set any of them on the `gaia_simpleeval`
module raises `AttributeError`:

```
gaia-simpleeval: MAX_POWER is immutable. gaia-simpleeval uses hardcoded safety
limits that cannot be modified at runtime. See LIMITATIONS.md.
```

Modifying `simpleeval.MAX_POWER` (etc.) on the original module does not affect
the gaia-simpleeval fast path. If configurable limits are required, use
`simpleeval` directly.

## 32-bit platform note

On 32-bit platforms, i128 arithmetic uses software emulation and is
approximately 4–6× slower than on 64-bit platforms. The evaluator remains
correct but may be slower for expressions requiring i128 intermediate values.

## `names` dict restriction

The fast path requires `names` to be `None` or a plain `dict` with all values
being primitive types (`int`, `float`, `bool`, `str`, `None`). Callable names,
custom objects, and `ModuleWrapper` instances cause fallback to the original
Python `simpleeval`.

## DEFAULT_FUNCTIONS only

Only the built-in functions (`rand`, `randint`, `int`, `float`, `str`, `bool`,
`abs`, `round`, `min`, `max`, `len`, `sum`) are supported on the fast path.
Custom functions cause fallback to Python. Functions outside this set that are
referenced in an otherwise fast-path expression raise `FunctionNotDefined`.

## Pure-Python fallback

If no compiled wheel is available for the current platform, gaia-simpleeval
falls back to the original `simpleeval` for all calls.
