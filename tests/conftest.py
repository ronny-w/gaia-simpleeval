"""Shared pytest fixtures and helpers for the gaia-simpleeval test suite.

Many tests in this suite require the compiled Rust extension (`_gaia_simpleeval`)
to be built and importable via the `gaia_simpleeval` package. When the extension
is not available (e.g. the wheel has not been built in the current environment)
those tests are skipped rather than failing, so that the suite can still be
collected and the build-independent tests can run.

Tests that exercise *parity* with the original ``simpleeval`` additionally
require ``simpleeval`` itself to be importable.
"""

import importlib

import pytest


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - import failure path
        return None


# Attempt the imports once at collection time.
gaia = _try_import("gaia_simpleeval")
simpleeval = _try_import("simpleeval")

# The compiled extension is required for almost everything in gaia_simpleeval.
# We detect "built" by checking that simple_eval actually evaluates rather than
# raising ImportError for the missing native module.
_gaia_built = False
if gaia is not None:
    try:
        gaia.simple_eval("1+1")
        _gaia_built = True
    except Exception:
        _gaia_built = False


requires_gaia = pytest.mark.skipif(
    not _gaia_built,
    reason="gaia_simpleeval native extension (_gaia_simpleeval) not built/importable",
)

requires_simpleeval = pytest.mark.skipif(
    simpleeval is None,
    reason="original simpleeval package not installed",
)


@pytest.fixture(scope="session")
def gse():
    """The gaia_simpleeval module (skips if not built)."""
    if not _gaia_built:
        pytest.skip("gaia_simpleeval native extension not built/importable")
    return gaia


@pytest.fixture(scope="session")
def orig():
    """The original simpleeval module (skips if not installed)."""
    if simpleeval is None:
        pytest.skip("original simpleeval package not installed")
    return simpleeval


@pytest.fixture(scope="session")
def ev(gse):
    """Convenience: the gaia simple_eval callable."""
    return gse.simple_eval
