"""Nox automation sessions for lauren-ai."""

from __future__ import annotations

import os

import nox

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True
# Keep envs in a user-writable location so `nox` works regardless of which
# user originally bootstrapped the project (root vs ai-slave, etc.).
nox.options.envdir = os.path.join(
    os.path.expanduser("~"), ".cache", "nox", "lauren-ai"
)

PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
DEFAULT_PYTHON = "3.12"

SRC = "src"
TESTS = "tests"


LAUREN_FRAMEWORK_PATH = "../lauren-framework"


def _install_dev(session: nox.Session) -> None:
    """Install the package in editable mode with dev extras via uv.

    ``[tool.uv.sources]`` is not honoured by ``uv pip install`` (only by
    ``uv sync``).  We pass both the local ``lauren`` path and this package in
    **one** call so that uv's resolver sees the local ``lauren`` editable
    install and accepts it as satisfying ``lauren>=1.0``.
    """
    session.install("-e", LAUREN_FRAMEWORK_PATH, "-e", ".[dev,anthropic]")


# ---------------------------------------------------------------------------
# Linting & formatting
# ---------------------------------------------------------------------------


@nox.session(name="lint", python=DEFAULT_PYTHON)
def lint(session: nox.Session) -> None:
    """Run ruff lint and format checks."""
    session.install("ruff>=0.4")
    session.run("ruff", "check", SRC, TESTS, "noxfile.py")
    session.run("ruff", "format", "--check", SRC, TESTS, "noxfile.py")


@nox.session(name="format", python=DEFAULT_PYTHON)
def format_(session: nox.Session) -> None:
    """Auto-fix lint issues and reformat code."""
    session.install("ruff>=0.4")
    session.run("ruff", "check", "--fix", SRC, TESTS, "noxfile.py")
    session.run("ruff", "format", SRC, TESTS, "noxfile.py")


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------


@nox.session(name="typecheck", python=DEFAULT_PYTHON)
def typecheck(session: nox.Session) -> None:
    """Run mypy strict type checking."""
    _install_dev(session)
    session.run("mypy", SRC)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@nox.session(name="tests", python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run all non-benchmark, non-eval tests with coverage."""
    _install_dev(session)
    session.run(
        "pytest",
        TESTS,
        "-q",
        *session.posargs,
    )
    

@nox.session(name="coverage", python=DEFAULT_PYTHON)
def coverage(session: nox.Session) -> None:
    """Run all non-benchmark, non-eval tests with coverage."""
    _install_dev(session)
    session.run(
        "pytest",
        TESTS,
        "--cov=lauren_ai",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "-q",
        *session.posargs,
    )


@nox.session(name="tests_unit", python=DEFAULT_PYTHON)
def tests_unit(session: nox.Session) -> None:
    """Run unit tests only."""
    _install_dev(session)
    session.run(
        "pytest",
        f"{TESTS}/unit",
        "-v",
        *session.posargs,
    )


@nox.session(name="tests_integration", python=DEFAULT_PYTHON)
def tests_integration(session: nox.Session) -> None:
    """Run integration tests (may require external services / API keys)."""
    _install_dev(session)
    session.run(
        "pytest",
        f"{TESTS}/integration",
        "-v",
        "--no-header",
        *session.posargs,
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@nox.session(name="benchmark", python=DEFAULT_PYTHON)
def benchmark(session: nox.Session) -> None:
    """Run benchmark tests (excluded from default run)."""
    _install_dev(session)
    session.install("pytest-benchmark>=4.0")
    session.run(
        "pytest",
        f"{TESTS}/benchmarks",
        "-m",
        "benchmark",
        "--benchmark-autosave",
        *session.posargs,
    )


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


@nox.session(name="eval", python=DEFAULT_PYTHON)
def eval_(session: nox.Session) -> None:
    """Run evaluation tests (requires ANTHROPIC_API_KEY)."""
    _install_dev(session)
    session.run(
        "pytest",
        "-m",
        "eval",
        f"{TESTS}/eval",
        "-v",
        *session.posargs,
    )


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


@nox.session(name="docs", python=DEFAULT_PYTHON)
def docs(session: nox.Session) -> None:
    """Build the MkDocs documentation."""
    session.install("mkdocs>=1.6", "mkdocs-material>=9.5", "pymdown-extensions>=10.7")
    session.run("mkdocs", "build", "--strict")


@nox.session(name="docs_serve", python=DEFAULT_PYTHON)
def docs_serve(session: nox.Session) -> None:
    """Serve the MkDocs documentation locally."""
    session.install("mkdocs>=1.6", "mkdocs-material>=9.5", "pymdown-extensions>=10.7")
    session.run("mkdocs", "serve")


# ---------------------------------------------------------------------------
# Build & release
# ---------------------------------------------------------------------------


@nox.session(name="build", python=DEFAULT_PYTHON)
def build(session: nox.Session) -> None:
    """Build source distribution and wheel."""
    session.install("build>=1.0")
    session.run("python", "-m", "build")


@nox.session(name="build_check", python=DEFAULT_PYTHON)
def build_check(session: nox.Session) -> None:
    """Check the built distributions with twine."""
    build(session)
    session.install("twine>=5.0")
    session.run("twine", "check", "dist/*")


@nox.session(name="release_test", python=DEFAULT_PYTHON)
def release_test(session: nox.Session) -> None:
    """Publish to TestPyPI."""
    build(session)
    session.install("twine>=5.0")
    session.run(
        "twine",
        "upload",
        "--repository",
        "testpypi",
        "dist/*",
    )


@nox.session(name="release", python=DEFAULT_PYTHON)
def release(session: nox.Session) -> None:
    """Publish to PyPI."""
    build(session)
    session.install("twine>=5.0")
    session.run("twine", "upload", "dist/*")


# ---------------------------------------------------------------------------
# CI composite
# ---------------------------------------------------------------------------


@nox.session(name="ci", python=DEFAULT_PYTHON)
def ci(session: nox.Session) -> None:
    """Run lint + tests + typecheck (full CI pipeline)."""
    lint(session)
    tests(session)
    typecheck(session)


# ---------------------------------------------------------------------------
# Pre-commit / prek
# ---------------------------------------------------------------------------


@nox.session(name="prek", python=DEFAULT_PYTHON, reuse_venv=True)
def prek(session: nox.Session) -> None:
    """Run the prek (pre-commit) hook suite.

    Locally, install prek globally once (``uv tool install prek``) and let
    ``prek install`` wire up the git hook.  This session is for CI and one-off
    runs.

    Pass extra arguments after ``--``::

        nox -s prek -- run --all-files
        nox -s prek -- run ruff --files src/lauren_ai/_tools/__init__.py
    """
    session.install("prek>=0.3")
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.run("prek", *args)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@nox.session(name="clean", python=DEFAULT_PYTHON)
def clean(session: nox.Session) -> None:
    """Remove build artifacts."""
    import shutil

    for path in ["dist", "build", ".coverage", "htmlcov", ".mypy_cache", ".ruff_cache"]:
        shutil.rmtree(path, ignore_errors=True)
    session.log("Cleaned build artifacts.")
