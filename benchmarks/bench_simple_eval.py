#!/usr/bin/env python3
"""bench_simple_eval.py — Performance benchmark: gaia vs original simpleeval.

Runs each representative expression 100,000 times through both
``gaia_simpleeval.simple_eval`` and the original ``simpleeval.simple_eval``,
reports per-call time (microseconds) for each, and the speedup factor.

Usage:
    python benchmarks/bench_simple_eval.py

Requires the compiled gaia extension to be built and importable, and the
original simpleeval to be installed. If either is missing the script prints a
clear message and exits without error so it can be run in any environment.
"""

import sys
import timeit

ITERATIONS = 100_000

# (expression, names) pairs covering the common fast-path shapes.
CASES = [
    ("2+2", {}),
    ("2+2*3", {}),
    ("x+y", {"x": 1, "y": 2}),
    ('"hello"+"world"', {}),
    ("1<2", {}),
    ("2**10", {}),
    ("10//3", {}),
    ("5&3", {}),
    ("3.14*2.0", {}),
    ("x*y", {"x": 1.5, "y": 2.7}),
]

# Per-call budget asserted by the SPEC (section 10.2): each fast-path call must
# stay comfortably under 5 microseconds.
MAX_US_PER_CALL = 5.0


def _import_both():
    try:
        import gaia_simpleeval as gaia
    except Exception as e:  # pragma: no cover
        print(f"gaia_simpleeval not importable ({e}); build the extension first.")
        return None, None
    try:
        gaia.simple_eval("1+1")
    except Exception as e:  # pragma: no cover
        print(f"gaia_simpleeval native extension not built ({e}).")
        return None, None
    try:
        import simpleeval as orig
    except Exception as e:  # pragma: no cover
        print(f"original simpleeval not installed ({e}).")
        return gaia, None
    return gaia, orig


def _time(fn):
    total = timeit.timeit(fn, number=ITERATIONS)
    return total / ITERATIONS * 1e6  # microseconds per call


def bench():
    gaia, orig = _import_both()
    if gaia is None:
        return None

    results = {}
    print(f"{'expression':<22}{'gaia(µs)':>12}{'orig(µs)':>12}{'speedup':>12}")
    print("-" * 58)
    for expr, names in CASES:
        gaia_us = _time(lambda: gaia.simple_eval(expr, names=names))

        # SPEC performance regression guard for the gaia fast path.
        assert gaia_us < MAX_US_PER_CALL, (
            f"Performance regression: {expr} took {gaia_us:.2f}µs "
            f"(budget {MAX_US_PER_CALL}µs)"
        )

        if orig is not None:
            orig_us = _time(lambda: orig.simple_eval(expr, names=names))
            speedup = orig_us / gaia_us if gaia_us else float("inf")
            print(f"{expr:<22}{gaia_us:>12.3f}{orig_us:>12.3f}{speedup:>11.1f}x")
            results[expr] = {"gaia_us": gaia_us, "orig_us": orig_us, "speedup": speedup}
        else:
            print(f"{expr:<22}{gaia_us:>12.3f}{'n/a':>12}{'n/a':>12}")
            results[expr] = {"gaia_us": gaia_us, "orig_us": None, "speedup": None}

    speedups = [r["speedup"] for r in results.values() if r["speedup"]]
    if speedups:
        avg = sum(speedups) / len(speedups)
        print("-" * 58)
        print(f"average speedup: {avg:.1f}x over {len(speedups)} cases")
    return results


if __name__ == "__main__":
    out = bench()
    if out is None:
        # Nothing was benchmarked (missing build); not a hard failure.
        sys.exit(0)
