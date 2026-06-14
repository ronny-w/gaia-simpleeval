"""Autopatch logic for gaia-simpleeval.

Invoked at interpreter startup by ``gaia_simpleeval.pth``. Replaces the
module-level ``simple_eval`` function and ``SimpleEval`` class on the original
``simpleeval`` module with the accelerated gaia-simpleeval implementations, so
existing ``import simpleeval`` callers transparently use the Rust evaluator for
supported expressions with no code changes.

The patch is silent and best-effort: if ``simpleeval`` (or the accelerated
package) is not importable, it does nothing.
"""


def _install():
    try:
        import simpleeval as _se
        import gaia_simpleeval as _ge
    except ImportError:
        # simpleeval not installed (or gaia_simpleeval unavailable) — nothing
        # to patch.
        return

    try:
        _se.simple_eval = _ge.simple_eval
        _se.SimpleEval = _ge.SimpleEval
    except Exception:
        # Never let a startup patch failure break interpreter import.
        pass


_install()
