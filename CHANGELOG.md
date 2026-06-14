# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — MCP integration

- **`AgentMcpServer`** (`lauren_ai.mcp.AgentMcpServer`) — wraps any `@agent`-decorated class
  as a first-class MCP server exposing `run`, `stream`, `memory` (resource), and
  `system_prompt` (prompt) endpoints.  Call `.build_module(llm_module)` to get a Lauren
  `@module` ready for `LaurenFactory.create()`.

- **`McpServerConfig`** (re-exported from `lauren_ai.mcp`) — pairs an alias string with an
  `McpClientProtocol` instance; pass a list to `AgentModule.for_root(mcp_servers=[...])`.

- **Dynamic MCP tool discovery** — `AgentModule.for_root` grows a `dynamic_mcp` flag;
  when enabled, a `_DynamicMcpBridge` singleton subscribes to
  `notifications/tools/list_changed` and atomically diffs the tool catalogue at runtime,
  adding and removing namespaced tools from every eligible agent's tool map.

- **`McpAgentTeam`** (`lauren_ai.mcp.McpAgentTeam`) — coordinates multiple independently
  deployed `AgentMcpServer` instances as a multi-agent team.  Accepts a `coordinator`
  `@agent` class and a dict of `alias → McpClientProtocol` workers; returns an
  `McpTeamResult` with the final answer and per-worker token usage.

- **`McpTeamResult`** and **`TeamWorkerResult`** — aggregated and per-worker result
  dataclasses returned by `McpAgentTeam.run()`.

- **`McpConversationStore`** (`lauren_ai.mcp.McpConversationStore`) — `ConversationStore`
  implementation backed by a remote MCP server via `resources/read` and `tools/call`;
  compatible with any MCP server that exposes `save_conversation` / `delete_conversation`.

- **`McpUserMemoryStore`** (`lauren_ai.mcp.McpUserMemoryStore`) — `UserMemoryStore`
  implementation backed by a remote MCP server; expects `save_user_fact`, `get_user_facts`,
  and `delete_user_fact` tools on the server.

- **`McpMemoryFact`** — dataclass returned by `McpUserMemoryStore.get_all()` representing
  a single stored user-memory key/value pair.

- **`McpPromptTemplate`** (`lauren_ai.mcp.McpPromptTemplate`) — adapter that wraps an MCP
  prompt as a callable prompt template; can be passed directly to `@agent(system=...)`.
  Implements `async __call__(**arguments)` and supports `Chain` composition via `|`.

- **`McpSystemPromptBuilder`** (`lauren_ai.mcp.McpSystemPromptBuilder`) — lazily fetches a
  named MCP prompt at agent startup and returns it as a plain `str` system prompt.

- **`list_mcp_prompts`** (`lauren_ai.mcp.list_mcp_prompts`) — convenience coroutine that
  lists all prompts on a connected client and returns a `McpPromptTemplate` per entry.

- **`McpResourceKnowledgeSource`** (`lauren_ai.mcp.McpResourceKnowledgeSource`) — bridges
  MCP resource listings into the lauren-ai knowledge-source duck-typed protocol; implements
  `async search(query, k)` returning `KnowledgeChunk` objects.

- **`KnowledgeChunk`** — dataclass representing a single retrieved knowledge fragment with
  `content`, `source`, `metadata`, and optional `score` fields.

- **`AgentSamplingHandler`** (`lauren_ai.mcp.AgentSamplingHandler`) — routes MCP
  `sampling/createMessage` requests to an `AgentRunnerBase`, allowing MCP server tools to
  leverage the host LLM for sub-tasks without managing their own credentials.

- **`@use_mcp_servers(*aliases)`** — class decorator that restricts which MCP server aliases
  an agent is allowed to call; stored as `AgentMeta.allowed_mcp_aliases` and enforced by the
  MCP bridge at tool-injection time.

### Added — Signals

- **`ToolProgressEvent`** (`lauren_ai.ToolProgressEvent`) — lifecycle signal emitted when an
  MCP tool sends a `notifications/progress` message; carries `tool_name`, `tool_use_id`,
  `progress`, `total`, `message`, and `alias`.

- **`McpToolsRefreshed`** (`lauren_ai.McpToolsRefreshed`) — lifecycle signal emitted by the
  dynamic MCP bridge after atomically updating the tool catalogue; carries `alias`, `added`,
  `removed`, and `total` counts.

- **`serialize(event)`** (`lauren_ai.serialize`) — converts any dataclass-based
  `LifecycleEvent` to a JSON-safe dict with a `signal_type` discriminator key; handles nested
  dataclasses, `type` values, enums, datetimes, and UUIDs.

- **`EventSink` protocol** (`lauren_ai.EventSink`) — runtime-checkable protocol for pluggable
  event sinks; implementations expose `async on_signal(signal)` and are called sequentially
  before the `SignalBus` fan-out.

### Added — HTTP endpoints

- **`AgentHttpModule`** (`lauren_ai._http.AgentHttpModule`) — factory that creates a Lauren
  `@module` with two standard HTTP endpoints for deploying any `@agent` as an HTTP service:
  `POST /chat` (JSON response) and `POST /stream` (NDJSON SSE streaming).

- **`AgentEvent`** union type and wire-serialisable event dataclasses — `AgentHttpTokenEvent`,
  `AgentToolStartEvent`, `AgentToolResultEvent`, `AgentToolProgressEvent`, and `AgentDoneEvent`
  each implement `to_json()` for SSE serialisation.

### Added — Unified tool context

- **`UnifiedToolContext`** (`lauren_ai._tools.UnifiedToolContext`) — runtime-checkable
  `Protocol` satisfied by both `ToolContext` and `McpToolContext`; allows tool functions to be
  registered as both native `@tool()` and `@mcp_tool()` without modification.

- **`ToolContextAdapter`** (`lauren_ai._tools.ToolContextAdapter`) — wraps an `McpToolContext`
  to present the `ToolContext`-compatible interface, including `agent_context`, `turn`,
  `session_id`, and MCP-specific pass-through methods like `report_progress` and `log`.

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
