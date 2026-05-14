# Release Guide

This page walks you through the full release cycle for `lauren-ai`:
preparing a release branch, tagging, building, and publishing to PyPI.

---

## Prerequisites

- You have push access to the `main` branch (or can open a PR to it).
- Your local environment has Python 3.11+ and the dev dependencies
  installed (`uv sync --extra dev`).
- The PyPI and TestPyPI repository environments are configured for GitHub OIDC
  Trusted Publishing. No long-lived upload token is required.

---

## 1. Prepare the Release

### 1.1 Branch from `main`

```bash
git checkout main
git pull origin main
git checkout -b release/v1.2.3
```

Use the version you are about to publish as the branch name — it makes
pull-request titles self-documenting.

### 1.2 Update `CHANGELOG.md`

`lauren-ai` follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
format. Move items from `[Unreleased]` to a new `[1.2.3] - YYYY-MM-DD`
section. Example:

```markdown
## [1.2.3] - 2026-05-01

### Added
- Unified `ExtractionMarker.extract` signature with `ExecutionContext`.
- `@public` decorator and `IS_PUBLIC_KEY` for cooperative guard bypass.

### Fixed
- `LaurenFactory.create()` no longer requires `await`.

### Changed
- `_ExtractionMarker` renamed to `ExtractionMarker` (public alias kept).
```

### 1.3 Run the Full CI Matrix Locally

```bash
uv run nox                # repo default gate: lint + tests + format + build + build_check + prek
uv run nox -s typecheck   # strict mypy run
uv run nox -s docs        # strict MkDocs build
uv run nox -s llms_check  # every public symbol is in llms-full.txt
```

All sessions must be green before you open a PR.

### 1.4 Open and Merge the Release PR

Push the branch and open a pull request against `main`. The PR title
should be `release: vX.Y.Z`. After review, merge with a standard merge
commit (no squash — the tag must point to a commit that is already on
`main`).

---

## 2. Tag the Release

After the release PR lands on `main`, tag it:

```bash
git checkout main
git pull origin main

# Annotated tag — the message becomes the release description on GitHub.
git tag -a v1.2.3 -m "Release v1.2.3"
```

### Push the Tag

```bash
git push origin v1.2.3
```

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which:

1. Runs `uv run nox -s build` to produce `dist/`.
2. Publishes to PyPI via OIDC Trusted Publishing (no token required in
   the environment).

!!! warning "Do not push a tag to a non-`main` commit"
    The release workflow publishes whatever commit the tag points to.
    Always tag the merged `main` commit, not the release branch tip.

---

## 3. Verify the Release

Once the GitHub Actions `release` job completes:

```bash
# In a clean environment:
pip install lauren_ai==1.2.3
python -c "import lauren_ai; print(lauren_ai.__version__)"
# Expected: 1.2.3
```

Then create a GitHub Release:

1. Go to **Releases → Draft a new release**.
2. Select the `v1.2.3` tag.
3. Paste the changelog section as the release body.
4. Publish.

---

## 4. Manual / Local Release (Fallback)

Prefer the GitHub Actions workflow. Use the manual path only if CI is
unavailable:

```bash
uv run nox -s build       # produces dist/lauren_ai-X.Y.Z*.whl + .tar.gz
uv run nox -s build_check # validates the artifacts
```

For ad-hoc publishing outside the tag-triggered workflow, use the
`workflow_dispatch` input on `.github/workflows/release.yml` and choose
`testpypi` or `pypi`.

---

## 5. Post-Release Housekeeping

- Add a new `[Unreleased]` section to the top of `CHANGELOG.md`.
- Close the milestone on GitHub (if one was used).
- Announce in the project discussion board or release notes.

---

## Version Format

`lauren` uses **`setuptools-scm`** to derive the package version
from git tags automatically:

| Scenario | Example version |
|----------|-----------------|
| Exactly on a tag | `1.2.3` |
| Commits after a tag | `1.2.3.post4+g1a2b3c4` |
| Untagged / dirty working tree | `0.0.0+unknown` |

You never edit a version number manually. The tag *is* the version.
See [Versioning](versioning.md) for the full details.
