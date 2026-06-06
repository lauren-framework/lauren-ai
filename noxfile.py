"""Nox automation for lauren-ai.

This file is the canonical task runner — every check that runs in CI runs
here.

Discoverability
---------------
List every session::

    nox -l

Run the default session set (everything that gates a PR)::

    nox

Run one session::

    nox -s tests
    nox -s lint
    nox -s docs

Pass extra arguments to the session's tool (after ``--``)::

    nox -s tests -- -k agent -v
    nox -s docs -- --strict

Design principles
-----------------
1. **Idempotent.** Every session is safe to re-run; isolated venvs prevent
   bleed-through.
2. **Reuse-friendly.** Sessions opt into ``reuse_venv=True`` whenever the
   environment is expensive to create and stable across runs (linting,
   docs, type-checking).
3. **CI parity.** A green ``nox`` locally implies green CI; both call the
   same code paths.
4. **No hidden state.** Build / release sessions wipe ``dist/`` first.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import nox

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
SRC_DIR = ROOT / "src"
TESTS_DIR = ROOT / "tests"
DOCS_BUILD_DIR = ROOT / "site"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
DOCS_REQUIREMENTS = ROOT / "docs-requirements.txt"

# ---------------------------------------------------------------------------
# Nox global configuration
# ---------------------------------------------------------------------------
# We pin a single primary Python for most sessions; the ``tests`` session
# parametrises across the supported matrix below.
#
# ``PRIMARY_PYTHON`` is the default interpreter for single-version sessions
# (lint, typecheck, docs, build, …). Honour the ``LAUREN_AI_PRIMARY_PYTHON``
# env var so contributors / CI can pin to whatever interpreter is installed
# without editing this file.
PRIMARY_PYTHON = os.environ.get("LAUREN_AI_PRIMARY_PYTHON", "3.12")
SUPPORTED_PYTHONS = ["3.11", "3.12", "3.13", "3.14"]

# Default sessions when running ``nox`` with no -s argument.
nox.options.sessions = [
    "lint",
    "tests",
    "format",
    "build",
    "build_check",
    "typecheck",
    "prek",
]
nox.options.reuse_existing_virtualenvs = True
# ``error_on_missing_interpreters = False`` lets contributors run only the
# Python versions they have installed locally; CI explicitly installs all.
nox.options.error_on_missing_interpreters = False
nox.options.stop_on_first_error = False


def _install_dev(session: nox.Session) -> None:
    """Install all dev dependencies from pyproject.toml via uv sync.

    ``uv sync`` honours ``[tool.uv.sources]`` (local path overrides), so the
    local ``lauren-framework`` editable install is resolved automatically.
    """
    session.run("uv", "sync", "--extra", "dev", "--active", external=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@nox.session(python=SUPPORTED_PYTHONS)
def tests(session: nox.Session) -> None:
    """Run the full test suite (unit + integration)."""
    _install_dev(session)
    args = session.posargs or ["-q"]
    session.run("pytest", str(TESTS_DIR), *args)


@nox.session(python=PRIMARY_PYTHON, name="tests_unit")
def tests_unit(session: nox.Session) -> None:
    """Run only unit tests under tests/unit/."""
    _install_dev(session)
    args = session.posargs or ["-q"]
    session.run("pytest", str(TESTS_DIR / "unit"), *args)


@nox.session(python=PRIMARY_PYTHON, name="tests_integration")
def tests_integration(session: nox.Session) -> None:
    """Run only integration tests under tests/integration/."""
    _install_dev(session)
    args = session.posargs or ["-q"]
    session.run("pytest", str(TESTS_DIR / "integration"), *args)


@nox.session(python=PRIMARY_PYTHON, name="tests_verbose")
def tests_verbose(session: nox.Session) -> None:
    """Run the full test suite with verbose output."""
    _install_dev(session)
    args = session.posargs or ["-v"]
    session.run("pytest", str(TESTS_DIR), *args)


@nox.session(python=PRIMARY_PYTHON)
def coverage(session: nox.Session) -> None:
    """Run tests under coverage and print a terminal summary."""
    _install_dev(session)
    session.run("uv", "pip", "install", "coverage[toml]", "pytest-cov", external=True)
    args = session.posargs or [
        str(TESTS_DIR / "unit"),
        str(TESTS_DIR / "integration"),
        "--cov-report=term-missing",
        "--cov-report=xml",
    ]
    session.run(
        "pytest",
        "--cov=lauren_ai",
        "--cov-branch",
        *args,
        "-q",
    )


@nox.session(python=PRIMARY_PYTHON)
def benchmark(session: nox.Session) -> None:
    """Run performance benchmarks (excluded from the default test run)."""
    _install_dev(session)
    session.run("uv", "pip", "install", "pytest-benchmark>=4.0", external=True)
    args = session.posargs or ["-v", "-m", "benchmark", str(TESTS_DIR / "benchmarks")]
    session.run("pytest", *args)


@nox.session(python=PRIMARY_PYTHON)
def eval_(session: nox.Session) -> None:
    """Run evaluation tests (requires ANTHROPIC_API_KEY)."""
    _install_dev(session)
    args = session.posargs or ["-m", "eval", "-v", str(TESTS_DIR / "eval")]
    session.run("pytest", *args)


# ---------------------------------------------------------------------------
# Lint / type-check
# ---------------------------------------------------------------------------


@nox.session(python=PRIMARY_PYTHON, reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run ruff against the package and tests.

    Use ``nox -s lint -- --fix`` to auto-fix.
    """
    session.install("ruff>=0.6")
    extra = session.posargs or []
    session.run("ruff", "check", "--fix", "src", "tests", *extra)


@nox.session(python=PRIMARY_PYTHON, reuse_venv=True)
def format(session: nox.Session) -> None:  # noqa: A001
    """Auto-format the codebase with ruff.

    This *writes* changes. Run ``nox -s lint`` afterwards to verify.
    """
    session.install("ruff>=0.6")
    session.run("ruff", "format", "src", "tests")


@nox.session(python=PRIMARY_PYTHON, reuse_venv=True)
def typecheck(session: nox.Session) -> None:
    """Run mypy over the lauren_ai package."""
    _install_dev(session)
    session.run("uv", "pip", "install", "mypy==2.1.0", external=True)
    args = session.posargs or ["src"]
    session.run("mypy", *args)


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


@nox.session(python=PRIMARY_PYTHON, reuse_venv=True)
def docs_install(session: nox.Session) -> None:
    """Install MkDocs + Material requirements."""
    if DOCS_REQUIREMENTS.exists():
        session.install("-r", str(DOCS_REQUIREMENTS))
    else:
        session.install(
            "mkdocs>=1.6",
            "mkdocs-material>=9.5",
            "pymdown-extensions>=10.7",
            "mkdocstrings[python]>=0.27",
            "griffe>=1.0",
        )


@nox.session(python=PRIMARY_PYTHON, reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the documentation site into ./site (strict mode).

    Also regenerates docs/generated-reference/ — the plain-Markdown API
    reference consumed by the lauren-ai-website (Next.js).  The generated
    files are committed to the repo so the website's production build works
    without requiring Python.

    Strict mode treats any warning as an error, matching CI.
    """
    if DOCS_REQUIREMENTS.exists():
        session.install("-r", str(DOCS_REQUIREMENTS))
    else:
        session.install(
            "mkdocs>=1.6",
            "mkdocs-material>=9.5",
            "pymdown-extensions>=10.7",
            "mkdocstrings[python]>=0.27",
            "griffe>=1.0",
        )
    session.run("python", "scripts/generate_api_docs.py")
    args = session.posargs or ["--strict"]
    session.run("mkdocs", "build", *args)


@nox.session(python=PRIMARY_PYTHON, reuse_venv=True, name="docs_serve")
def docs_serve(session: nox.Session) -> None:
    """Serve the docs locally with live reload at http://localhost:8000.

    Also regenerates docs/generated-reference/ before starting the server.
    """
    if DOCS_REQUIREMENTS.exists():
        session.install("-r", str(DOCS_REQUIREMENTS))
    else:
        session.install(
            "mkdocs>=1.6",
            "mkdocs-material>=9.5",
            "pymdown-extensions>=10.7",
            "mkdocstrings[python]>=0.27",
            "griffe>=1.0",
        )
    session.run("python", "scripts/generate_api_docs.py")
    session.run("mkdocs", "serve", *session.posargs)


# ---------------------------------------------------------------------------
# Build & release
# ---------------------------------------------------------------------------


def _clean_build_artifacts() -> None:
    for path in (DIST_DIR, BUILD_DIR):
        if path.exists():
            shutil.rmtree(path)
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg)


@nox.session(python=PRIMARY_PYTHON)
def build(session: nox.Session) -> None:
    """Build wheel + sdist into ./dist."""
    _clean_build_artifacts()
    session.install("build>=1.2")
    session.run("python", "-m", "build")
    if DIST_DIR.exists():
        session.log("Built artefacts:")
        for art in sorted(DIST_DIR.iterdir()):
            session.log(f"  {art.name}  ({art.stat().st_size} bytes)")


@nox.session(python=PRIMARY_PYTHON, name="build_check")
def build_check(session: nox.Session) -> None:
    """Validate the built distributions with ``twine check``."""
    if not DIST_DIR.exists() or not any(DIST_DIR.iterdir()):
        session.error("dist/ is empty; run `nox -s build` first or chain them: `nox -s build build_check`.")
    session.install("twine>=5.1")
    session.run("twine", "check", *[str(p) for p in DIST_DIR.iterdir()])


@nox.session(python=PRIMARY_PYTHON, name="release_test")
def release_test(session: nox.Session) -> None:
    """Upload wheel + sdist to TestPyPI."""
    build(session)  # type: ignore[arg-type]
    build_check(session)  # type: ignore[arg-type]
    session.install("twine>=5.1")
    session.log("Uploading to TestPyPI...")
    session.run(
        "twine",
        "upload",
        "--repository-url",
        "https://test.pypi.org/legacy/",
        *[str(p) for p in DIST_DIR.iterdir()],
    )


@nox.session(python=PRIMARY_PYTHON)
def release(session: nox.Session) -> None:
    """Upload wheel + sdist to the real PyPI.

    This is destructive and irreversible. Refuses to run without an
    explicit ``--yes`` posarg::

        nox -s release -- --yes

    Prefer the GitHub Actions ``release`` workflow + PyPI Trusted
    Publishing; this session is the local-only fallback.
    """
    if "--yes" not in session.posargs:
        session.error(
            "Refusing to release without --yes. "
            "Run: nox -s release -- --yes\n"
            "Better: tag the commit (`git tag vX.Y.Z && git push --tags`) "
            "and let .github/workflows/release.yml publish via OIDC."
        )
    build(session)  # type: ignore[arg-type]
    build_check(session)  # type: ignore[arg-type]
    session.install("twine>=5.1")
    session.log("Publishing to https://pypi.org/project/lauren-ai/ ...")
    session.run("twine", "upload", *[str(p) for p in DIST_DIR.iterdir()])
    session.log("")
    session.log("Released. Verify with: pip install lauren-ai")


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------


@nox.session(python=False)
def clean(session: nox.Session) -> None:
    """Remove build artefacts, caches, and the docs site.

    Uses ``python=False`` so we don't bother creating a virtualenv.
    """
    targets = [
        BUILD_DIR,
        DIST_DIR,
        DOCS_BUILD_DIR,
        ROOT / ".pytest_cache",
        ROOT / ".mypy_cache",
        ROOT / ".ruff_cache",
        ROOT / ".coverage",
        ROOT / "htmlcov",
        ROOT / "coverage.xml",
        ROOT / ".nox",
    ]
    for path in targets:
        if path.exists():
            session.log(f"Removing {path.relative_to(ROOT)}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for egg in ROOT.glob("*.egg-info"):
        session.log(f"Removing {egg.relative_to(ROOT)}")
        shutil.rmtree(egg)
    removed = 0
    for pycache in ROOT.rglob("__pycache__"):
        if any(part in {".venv", "venv", ".nox"} for part in pycache.parts):
            continue
        shutil.rmtree(pycache)
        removed += 1
    if removed:
        session.log(f"Removed {removed} __pycache__ directories")


# ---------------------------------------------------------------------------
# Pre-commit / prek
# ---------------------------------------------------------------------------


@nox.session(python=PRIMARY_PYTHON, reuse_venv=True)
def prek(session: nox.Session) -> None:
    """Run the prek (pre-commit) hook suite across the repository.

    Locally, you almost certainly want to install prek once globally
    (``uv tool install prek``) and let ``prek install`` wire up the
    git hook — this session exists for CI and one-off runs.

    Pass extra arguments after ``--``::

        nox -s prek -- run --all-files
        nox -s prek -- run ruff --files src/lauren_ai/_tools/__init__.py
    """
    session.install("prek>=0.3")
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.run("prek", *args)


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


@nox.session(python=False, name="ci")
def ci(session: nox.Session) -> None:
    """Run the full CI matrix locally (lint + tests + typecheck + docs).

    Equivalent to what GitHub Actions runs on a PR. Use sparingly — the
    full matrix can take several minutes. Most of the time you only need
    the default ``nox`` (which is ``lint`` + ``tests`` + ``typecheck``).
    """
    sessions = ["lint", "tests", "typecheck", "docs"]
    nox_bin = shutil.which("nox") or "nox"
    for s in sessions:
        session.log(f"--- nox -s {s} ---")
        session.run(nox_bin, "-s", s, external=True)


# ---------------------------------------------------------------------------
# Version Management
# ---------------------------------------------------------------------------

_SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _latest_release_tag() -> tuple[str, tuple[int, int, int]]:
    result = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    parsed: list[tuple[tuple[int, int, int], str]] = []
    for tag in tags:
        match = _SEMVER_TAG_RE.fullmatch(tag)
        if match is None:
            continue
        parsed.append(((int(match.group(1)), int(match.group(2)), int(match.group(3))), tag))
    if not parsed:
        raise RuntimeError(
            "No release tags matching v<major>.<minor>.<patch> were found. Create an initial tag such as v0.1.0 first."
        )
    version, tag = max(parsed, key=lambda item: item[0])
    return tag, version


def _version_bump_kind(session: nox.Session) -> str:
    allowed = {"--major": "major", "--minor": "minor", "--patch": "patch"}
    selected = [allowed[arg] for arg in session.posargs if arg in allowed]
    if not selected:
        return "patch"
    if len(selected) > 1:
        session.error("Choose exactly one of --major, --minor, or --patch.")
    return selected[0]


def _adjust_version(version: tuple[int, int, int], kind: str, *, delta: int) -> tuple[int, int, int]:
    major, minor, patch = version
    if kind == "major":
        major += delta
        if major < 0:
            raise ValueError("Cannot decrement major below 0.")
        return major, 0, 0
    if kind == "minor":
        minor += delta
        if minor < 0:
            raise ValueError("Cannot decrement minor below 0.")
        return major, minor, 0
    if kind == "patch":
        patch += delta
        if patch < 0:
            raise ValueError("Cannot decrement patch below 0.")
        return major, minor, patch
    raise ValueError(f"Unsupported version bump kind: {kind}")


def _render_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def _log_version_suggestion(session: nox.Session, *, action: str, delta: int) -> None:
    current_tag, current_version = _latest_release_tag()
    kind = _version_bump_kind(session)
    try:
        next_version = _adjust_version(current_version, kind, delta=delta)
    except ValueError as exc:
        session.error(str(exc))
    next_version_str = _render_version(next_version)
    next_tag = f"v{next_version_str}"
    session.log(f"Latest release tag: {current_tag}")
    session.log(f"{action} {kind}: {current_tag} -> {next_tag}")
    session.log("")
    session.log("Copy/paste:")
    session.log(f'  git tag -a {next_tag} -m "Release {next_tag}"')
    session.log("")
    session.log("Then push it with:")
    session.log(f"  git push origin {next_tag}")
    session.log("")
    session.log("Together:")
    session.log(f"  git tag -a {next_tag} -m 'Release {next_tag}' && git push origin {next_tag}")


@nox.session(python=PRIMARY_PYTHON, name="ver_inc")
def ver_inc(session: nox.Session) -> None:
    """Print the next release tag after incrementing major/minor/patch.

    Examples::

        nox -s ver_inc
        nox -s ver_inc -- --minor
        nox -s ver_inc -- --major
    """
    _log_version_suggestion(session, action="Increment", delta=1)


@nox.session(python=PRIMARY_PYTHON, name="ver_dec")
def ver_dec(session: nox.Session) -> None:
    """Print the previous release tag after decrementing major/minor/patch.

    Examples::

        nox -s ver_dec -- --patch
        nox -s ver_dec -- --minor
        nox -s ver_dec -- --major
    """
    _log_version_suggestion(session, action="Decrement", delta=-1)


# ---------------------------------------------------------------------------
# Backwards-compatible alias for `make help`
# ---------------------------------------------------------------------------


@nox.session(python=False, name="help")
def help_session(session: nox.Session) -> None:
    """Print every available session with its docstring."""
    from inspect import getdoc

    print("Available nox sessions:")
    print()
    _sessions = {
        "benchmark",
        "build",
        "build_check",
        "ci",
        "clean",
        "coverage",
        "docs",
        "docs_install",
        "docs_serve",
        "eval_",
        "format",
        "help_session",
        "lint",
        "prek",
        "release",
        "release_test",
        "tests",
        "tests_integration",
        "tests_unit",
        "tests_verbose",
        "typecheck",
    }
    for name, fn in sorted(globals().items()):
        if name not in _sessions:
            continue
        doc = (getdoc(fn) or "").splitlines()[0] if getdoc(fn) else ""
        display = name.rstrip("_")
        print(f"  nox -s {display:<22}  {doc}")
    print()
    print("Run `nox -l` for nox's own listing.")


__all__ = [
    "benchmark",
    "build",
    "build_check",
    "ci",
    "clean",
    "coverage",
    "docs",
    "docs_install",
    "docs_serve",
    "eval_",
    "format",
    "help_session",
    "lint",
    "prek",
    "release",
    "release_test",
    "tests",
    "tests_integration",
    "tests_unit",
    "tests_verbose",
    "typecheck",
]
