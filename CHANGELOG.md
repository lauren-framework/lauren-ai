# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added maintainer-oriented development guides under `docs/development/`, covering setup, testing, versioning, changelog writing, and release workflows.

### Changed

- Simplified `AgentRunnerBase` construction: the runner no longer accepts `config=...`, so direct construction now uses `AgentRunnerBase(transport=..., signals=..., cache_backend=...)`.
- Moved model fallback into `AgentModule.for_root()`, so agents that omit `model=` now inherit the module's `LLMConfig.model` during wiring.
- Updated `AgentTestClient` and mock-runner helpers to assign `"mock-model"` when needed instead of building synthetic `LLMConfig` objects for tests.
- Changed the default `nox` session set to `lint`, `tests`, `format`, `build`, `build_check`, and `prek`; docs, typechecking, and other workflows remain available as explicit sessions.
- Clarified release automation around the new development docs and current GitHub Actions workflow.

### Removed

- Removed the local `[tool.uv.sources]` path override from `pyproject.toml`, so `lauren` is resolved like a normal dependency instead of via a repository-local source mapping.

### Fixed

- Fixed the failing nox/test helper path by updating runner construction and mock-model fallback logic to match the current `AgentRunnerBase` API.
