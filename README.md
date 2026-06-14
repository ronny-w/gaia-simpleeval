# gaia-simpleeval

A high-performance drop-in replacement for [simpleeval](https://github.com/danthedeckie/simpleeval), implemented in Rust via PyO3.

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

More than 20x faster than the original simpleeval on typical expressions:

| Expression     | simpleeval | gaia-simpleeval | Speedup |
|----------------|------------|-----------------|---------|
| 2+2            | 8.1 us     | 0.36 us         | 22x     |
| 2+2*3          | 9.6 us     | 0.37 us         | 26x     |
| x+y (names)    | 8.2 us     | 0.58 us         | 14x     |
| 1<2            | 8.6 us     | 0.40 us         | 21x     |
| 3.14*2.0       | 8.2 us     | 0.36 us         | 23x     |

## Compatibility

- Drop-in replacement — all simpleeval APIs work unchanged
- Supports Python 3.9 through 3.13
- Falls back to original simpleeval for advanced usage (custom operators, allowed_attrs)
- All 205 simpleeval compatibility tests pass

## Security

gaia-simpleeval preserves all simpleeval security constraints:
- No access to builtins, imports, or file system
- Module objects in names raise FeatureNotAvailable
- Dunder functions raise FunctionNotDefined

## License

MIT
