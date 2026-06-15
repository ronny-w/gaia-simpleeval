# gaia-simpleeval

A high-performance drop-in replacement for simpleeval, implemented in Rust via PyO3.

## Installation

    pip install gaia-simpleeval

## Usage

No code changes required. gaia-simpleeval automatically patches simpleeval at import time:

    import gaia_simpleeval
    import simpleeval
    result = simpleeval.simple_eval("2 + 2 * 3")

Or use the gaia API directly:

    import gaia_simpleeval as gse
    result = gse.simple_eval("x + y", names={"x": 1, "y": 2})

## Performance

More than 16x faster on average across complex real-world expressions, up to 25x on simple arithmetic:

| Case                 | simpleeval | gaia-simpleeval | Speedup |
| -------------------- | ---------- | --------------- | ------- |
| basic arithmetic     | 7.09 us    | 0.29 us         | 25x     |
| pythagorean distance | 11.34 us   | 0.56 us         | 20x     |
| complex boolean      | 16.76 us   | 0.78 us         | 22x     |
| price calculation    | 10.50 us   | 0.61 us         | 17x     |
| grade evaluation     | 17.04 us   | 1.03 us         | 17x     |
| deep arithmetic      | 17.63 us   | 1.02 us         | 17x     |
| **AVERAGE**          | **11.72 us** | **0.72 us**   | **17x** |

## Compatibility

- Drop-in replacement - all simpleeval APIs work unchanged
- Supports Python 3.9 through 3.13
- Falls back to original simpleeval for advanced usage (custom operators, allowed_attrs)
- All 205 simpleeval compatibility tests pass

## Security

- No access to builtins, imports, or file system
- Module objects in names raise FeatureNotAvailable
- Dunder functions raise FunctionNotDefined

## License

MIT
