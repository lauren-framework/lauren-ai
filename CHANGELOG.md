# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — Memory trim user-anchor guard (prevents assistant-leading histories)

- **`ShortTermMemory.messages()`** and **`trim_to_fit()`** now stop trimming
  when dropping the next oldest turn would remove the last *conversational*
  user message, leaving a history that starts with an assistant or tool-result
  message.  All providers require the first non-system message to have
  `role:"user"`.  The previous floor guard (PRD-118) only prevented an
  *empty* message list; it did not prevent an *assistant-leading* list.

  The failure scenario: user sends `"please fix the docs"`, agent makes 3
  parallel tool calls including `list_directory(recursive=True)`, the
  consolidated results exceed the 128 KB token budget.  The trim loop dropped
  the small original user intent (20 chars), leaving
  `[assistant_3_tools, user_results]` — which Anthropic rejected with
  `400: messages: at least one message is required`.

- **`_is_conversational_user(message)`** — new helper that distinguishes a
  real user turn from a `role:"user"` tool-result response.  A message whose
  content is entirely `tool_result` blocks cannot open a conversation; all
  other user messages can.  The guard fires only when no conversational user
  message would remain after the drop.

### Fixed — Streaming tool-use atomicity: closed the full cancellation window

Three bugs were fixed together (prd-streaming-tool-use-atomicity):

**Bug 1 — Cancellation window between `add_assistant` and the inner `try:`**

`_stream_loop` called `memory.add_assistant()` at line 1159, then had four
`await` calls (ModelCallComplete, on_turn_complete hook, AgentTurnComplete,
budget check) before the inner `try: ... except BaseException: ensure_valid()`
block at line 1216.  A `GeneratorExit`, `CancelledError`, or
`AgentBudgetExceededError` raised at any of those awaits left memory with an
orphaned `assistant([tool_use])` and no matching `tool_result`.  Fix: moved
`memory.add_assistant()` inside the `try:` block so `ensure_valid()` is
guaranteed to fire on any exception or cancellation after the assistant message
is committed.

**Bug 2 — `ensure_valid()` not called before each `memory.messages()`**

`ShortTermMemory.messages()` returns a healed snapshot but does not persist
the heal into `self._messages`.  The next `add_user()` call (from the
following `run_stream()` invocation) appended to the unhealed `_messages`,
potentially accumulating inconsistency across turns.  Fix: `ensure_valid()`
is now called at the top of every `_stream_loop` iteration, immediately before
`memory.messages()`, to persist heals into `_messages` before further
mutations.

**Bug 3 — `MockTransport._completion_as_stream` never emitted `tool_call_delta` chunks**

`_stream_loop` builds `accumulated_tool_calls` exclusively from
`chunk.tool_call_delta`.  The previous `_completion_as_stream` only emitted
a text chunk and a stop-reason chunk — no `tool_call_delta` — so every test
using `queue_tool_use()` + `stream=True` silently produced
`accumulated_tool_calls=[]` and never triggered tool execution.  Fix:
`_completion_as_stream` now emits one `CompletionChunk(tool_call_delta=...)`
per tool call with the full input JSON as `input_delta`.

### Fixed — Real parallel tool results now consolidated into one message

- **`ShortTermMemory.add_tool_results(results)`** — new batch method that adds
  all results for a parallel tool-call turn as **one** `role:"user"` message
  with multiple `tool_result` content blocks.  `_stream_loop` now calls this
  instead of looping over `add_tool_result`.  Without this, 3 parallel tool
  calls produced 3 separate user messages; Anthropic checked only the
  immediately-following message, saw just the first result, and returned
  `400: tool_use ids found without tool_result blocks immediately after`
  even though all three tools had completed successfully.

### Fixed — Parallel tool_use healer now produces one consolidated message

- **`_heal_dangling_tail` / `_heal_dangling_tail_unconditional`** now produce
  a **single** `role:"user"` message containing **all** missing `tool_result`
  content blocks, instead of N separate messages (one per missing ID).
  Anthropic requires every `tool_use` in an assistant turn to have its
  `tool_result` in the *immediately following* user message.  When 7 parallel
  tool calls were interrupted before any results arrived, the previous healer
  inserted 7 separate user messages.  Anthropic saw the first one contained
  only one result, declared the other 6 missing, and returned
  `400: tool_use ids found without tool_result blocks immediately after`.

### Fixed — `ShortTermMemory` trim floor guard

- **`ShortTermMemory.messages()`** no longer returns an empty list when a
  single user message exceeds the token budget.  Previously the trim loop
  (`while snapshot and total_chars > budget_chars`) would drop the only
  remaining message, resulting in an empty `messages` list that caused
  Anthropic to return `400: messages: at least one message is required`.
  This happened in practice when agenthicc prepended large `@mention` file
  contents to the user intent (e.g. a 200 KB source file).  The fix adds a
  floor guard: the loop stops before dropping the last non-system turn,
  and the oversized message is sent as-is.

- **`ShortTermMemory.trim_to_fit()`** has the same floor guard applied.

- Both methods now emit a `UserWarning` when a single message cannot be
  trimmed to fit the token budget, so callers can detect the situation and
  consider truncating large payloads upstream.

### Fixed — Anthropic `tool_use` / `tool_result` conversation healing

- **`_heal_dangling_tail` / `_heal_dangling_tail_unconditional`** now produce
  canonical `role:"user"` / `{"type": "tool_result", "tool_use_id": ...}`
  synthetic messages instead of the previous OpenAI-only `role:"tool"` /
  `tool_call_id` format.  Anthropic rejected the old format with
  `400 — tool_use ids found without tool_result blocks immediately after`,
  which caused the phase loop to retry until exhaustion and show a misleading
  "exhausted N attempts" failure.

- **`_message_to_anthropic()`** now converts any legacy `role:"tool"` message
  (OpenAI-format synthetic result already persisted in memory from an older
  session) to a valid Anthropic `role:"user"` / `tool_result` content block,
  preventing the 400 on resuming sessions that were serialised before this fix.

- **`_stream_loop()`** (`_agents/_runner.py`) now wraps tool execution and
  `memory.add_tool_result()` in a `try/except BaseException` block that calls
  `memory.ensure_valid()` before re-raising.  This closes the corruption window
  that existed between `memory.add_assistant()` (tool_use committed) and
  `memory.add_tool_result()` (tool_result committed): a `GeneratorExit` or
  `CancelledError` from the caller no longer leaves an orphaned `tool_use` in
  the conversation history.



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
