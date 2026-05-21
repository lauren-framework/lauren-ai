# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-05-21

### Added

- **`@use_hooks(*hook_classes)` decorator** — attaches injectable ``ToolHook`` classes to a
  ``@tool()``.  Each hook class is a subclass of ``ToolHook`` and may override any combination
  of ``before_tool_call``, ``after_tool_call``, and ``on_tool_error``.  Multiple ``@use_hooks``
  on the same tool stack (merge) in order.  Hook classes are auto-decorated with
  ``@injectable(scope=SINGLETON)`` if not already marked.

- **``ToolHook`` base class** and decision return types — ``BeforeToolHookDecision`` (proceed /
  abort / modify), ``AfterToolHookDecision`` (proceed / replace), ``ErrorToolHookDecision``
  (reraise / suppress_with).  Unoverridden methods are no-ops (proceed / reraise).

- **``ToolCallContext``** — extends ``ToolContext`` with ``tool_name: str`` and
  ``tool_input: dict[str, Any]`` (mutated in place when a ``modify`` decision is returned).

- **``global_tool_hooks``** parameter on ``AgentModule.for_root()`` — registers hook classes
  that fire for every tool call in the module.  Global hook classes are resolved as DI singletons
  and passed to the runner's ``ToolExecutor`` as ``global_hooks``.

- **Hook execution order** (per call): global before → per-tool before → dispatch → per-tool
  after → global after.  Error hooks follow the same after order (per-tool first, then global).
  A ``before_tool_call`` returning ``abort`` short-circuits all remaining hooks and the tool
  itself.  An ``on_tool_error`` returning ``suppress_with`` swallows the exception and skips all
  remaining error hooks.

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

[Unreleased]: https://github.com/lauren-framework/lauren-ai/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/lauren-framework/lauren-ai/releases/tag/v1.1.0
[1.0.0]: https://github.com/lauren-framework/lauren-ai/releases/tag/v1.0.0
