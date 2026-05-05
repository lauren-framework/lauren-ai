# Integrating `lauren-ai` with the `lauren` Framework

This guide shows how to wire `lauren-ai` agents, tools, and signals into a
production `lauren` web application.  Every example is drawn from the
SecureBank chatbot — a banking assistant with authenticated SSE streaming,
multi-agent delegation, and real-time WebSocket event push.

All patterns shown here compose cleanly with the rest of the `lauren` surface:
guards, middlewares, lifecycle hooks, and the full DI container.

---

## Overview — why integrate?

Running an `AgentRunner` inside a plain `async def` endpoint works, but you
lose everything the framework provides:

- **DI container** — singletons are created once at startup, not on every
  request.  `BankDatabase`, `AgentRunner`, and `CostTracker` share a single
  process-wide instance without any global variables.
- **Module scoping** — `AgentModule.for_root()` returns an ordinary `@module`,
  so imports, exports, and visibility rules apply exactly as for any other
  module.
- **Lifecycle hooks** — `@post_construct` / `@pre_destruct` let you warm up
  connection pools or flush state when the app shuts down.
- **Request context** — `ExecutionContext` and `Request` are injectable
  parameters; guards run before handlers, so by the time the agent starts the
  request is already authenticated.
- **Zero global state** — multiple app instances (test isolation) coexist in
  one process without interfering.

The integration is additive: you bring `@module`, `@controller`, DI, and SSE
from `lauren`; you bring `@agent()`, `@tool()`, `AgentRunner`, and `SignalBus`
from `lauren-ai`.

---

## Wiring with `LLMModule` and `AgentModule`

Both `LLMModule.for_root()` and `AgentModule.for_root()` return standard
`@module` classes that slot directly into the `imports=` list of any parent
module.

### 1. Configure the LLM provider

```python title="app/ai/ai_module.py"
import os
from lauren_ai import LLMConfig
from lauren_ai._module import LLMModule

_llm_config = LLMConfig(
    provider="openai",
    model=os.environ.get("LLM_MODEL", "poolside/laguna-xs.2:free"),
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",   # OpenRouter is OpenAI-compatible
)

LLMProvider = LLMModule.for_root(_llm_config)
```

`LLMModule.for_root()` registers `Transport`, `LLMService`, and `LLMConfig`
as singleton providers and exports all three.  Any `AgentModule` that lists
`LLMProvider` in its `imports=` will automatically see these tokens.

### 2. Register agents and tools

```python title="app/ai/ai_module.py"
from lauren_ai import InMemoryConversationStore
from lauren_ai._module import AgentModule

from app.ai.crm_agent import BankingCRMAgent
from app.ai.banking_tools import GetBalanceTool, GetTransactionHistoryTool
from app.ai.banking_delegation import DelegateToBankingTransfer
from app.ai.signals import signal_bus
from app.banking.banking_module import BankingModule

_conversation_store = InMemoryConversationStore()

_CRMAgentModule = AgentModule.for_root(
    agents=[BankingCRMAgent],
    tools=[DelegateToBankingTransfer, GetBalanceTool, GetTransactionHistoryTool],
    imports=[LLMProvider, BankingModule],
    signals=signal_bus,
    conversation_store=_conversation_store,
)
```

`AgentModule.for_root()` generates the wiring automatically:

- Each class in `agents=` is registered as a singleton injectable.
- Each class in `tools=` is auto-applied `@injectable(scope=SINGLETON)` and
  registered as a provider (constructor DI happens here — `BankDatabase` is
  injected into `GetBalanceTool.__init__` by the container).
- A unique `AgentRunnerBase` subclass is registered as the module's runner token.
  Inject it with `runner: AgentRunner` (Protocol annotation) in any provider
  belonging to the same module. When two or more AgentModules are in scope,
  use `injects=[MyRunner]` with an explicit named subclass instead (see below).
- `signals=signal_bus` wires the shared `SignalBus` so every model call emits
  `ModelCallComplete` and friends.

### 3. Compose the feature module

```python title="app/ai/ai_module.py"
from lauren import module
from lauren_ai import CostTracker, default_pricing_table
from lauren_ai._module import LLMService
from lauren import use_value

_cost_tracker = CostTracker(pricing=default_pricing_table())
_cost_tracker_provider = use_value(provide=CostTracker, value=_cost_tracker)

@module(
    imports=[LLMProvider, _CRMAgentModule, BankingModule],
    providers=[_cost_tracker_provider],
    exports=[LLMService, BankingCRMAgent, CostTracker],
    controllers=[BankingChatController],
)
class AIModule:
    """Provides banking AI services: agents, runners, and cost tracker."""
```

> **Export what controllers need.**  `BankingCRMAgent` is exported so sibling
> modules can inject it.  `CostTracker` is exported so the metrics controller can
> inject it without a circular import.  The runner is not exported directly — the
> controller is in the same `AIModule` scope and receives it via DI Protocol scan.

---

## Class-form tools with DI

Function-form tools (`@tool()` on an `async def`) are fine for stateless
operations.  When a tool needs a database client, HTTP session, or any other
service, use the class form — the DI container handles construction:

```python title="app/ai/banking_tools.py"
# NOTE: Do NOT add `from __future__ import annotations` to this file.
# The @tool() decorator uses inspect.signature() at decoration time to build
# the JSON schema, and PEP 563 lazy evaluation breaks that introspection.
from lauren_ai import ToolContext, tool
from app.banking.bank_db import BankDatabase

@tool()
class GetBalanceTool:
    """Get the current account balance and details for any SecureBank customer.

    Use this to look up any user's balance (alice, bob, or charlie).

    Args:
        user_id: The user whose balance to look up. Must be alice, bob, or charlie.
    """

    def __init__(self, db: BankDatabase) -> None:
        self._db = db

    async def run(self, user_id: str, ctx: ToolContext) -> dict:
        ...
```

Key rules:

- `@tool()` **automatically** applies `@injectable(scope=Scope.SINGLETON)` if
  not already present.  You rarely need to add it explicitly.
- The LLM schema is built from `run()`'s parameters, not `__init__`.
- `ctx: ToolContext` is injected at call time by `ToolExecutor` and is
  **never** included in the schema sent to the LLM.
- `AgentModule.for_root(tools=[...])` registers each class as a provider in
  the DI container.  Constructor dependencies (like `BankDatabase`) are
  resolved from the enclosing module graph via `imports=`.

> **Important — `from __future__ import annotations` is banned in tool files.**
> The `@tool()` decorator calls `inspect.signature()` at import time to build
> the JSON schema.  PEP 563 lazy evaluation (triggered by the future import)
> turns annotations into strings and breaks schema generation silently.  Add
> the comment shown above as a reminder.

---

## Streaming responses (SSE)

The banking controller streams agent output as Server-Sent Events using
`EventStream` from `lauren`:

```python title="app/ai/chat_banking_controller.py"
from lauren import EventStream, Json, Request, ServerSentEvent, controller, post, use_guards
from lauren.types import ExecutionContext

from app.ai.crm_agent import BankingCRMAgent
from app.banking.bank_db import BankDatabase
from app.ai.chat_schemas import ChatRequest
from app.crypto.signature_guard import SignatureGuard
from app.ws.context import current_user_id
from app.ai.banking_delegation import CRMAgentRunner


@use_guards(SignatureGuard)
@controller("/api/banking")
class BankingChatController:
    """Streams banking CRM agent responses as Server-Sent Events."""

    def __init__(
        self,
        runner: CRMAgentRunner,   # named subclass — unambiguous when two AgentModules are in scope
        db: BankDatabase,
        crm_agent: BankingCRMAgent,
    ) -> None:
        self._runner = runner
        self._db = db
        self._crm_agent = crm_agent

    @post("/chat")
    async def stream(self, body: Json[ChatRequest], exec_ctx: ExecutionContext) -> EventStream:
        ...
        async def generate():
            current_user_id.set(account.user_id)
            try:
                response = await self._runner.run(
                    self._crm_agent,
                    full_prompt,
                    conversation_id=body.conversation_id,
                    execution_context=exec_ctx,
                )
                content = response.content or ""
                chunk_size = 40
                for i in range(0, len(content), chunk_size):
                    yield ServerSentEvent(event="token", data=content[i : i + chunk_size])
                yield ServerSentEvent(event="done", data="")
            except Exception as exc:
                yield ServerSentEvent(event="error", data=str(exc))

        return EventStream(generate(), keep_alive=15.0)
```

A few design decisions worth noting:

**`AgentRunner` and agent as constructor params, not per-request.**  Both are
singletons.  The controller is a `@controller` which is itself a singleton; DI
resolves all three once at startup.

**`current_user_id.set(...)` before `runner.run()`.**  This pins the
`ContextVar` to the current async task's context.  `asyncio.gather()` (used by
`SignalBus.emit`) copies `ContextVar` state into each spawned task, so signal
handlers can call `current_user_id.get()` to route events to the right
WebSocket client.  Do NOT call `token.reset()` inside the generator — the
keep-alive mechanism in `EventStream` wraps each `__anext__()` in a new
`asyncio.Task`, so a token from an earlier task cannot be used to reset a
later one.

**`keep_alive=15.0`** tells the framework to send SSE comment lines every 15
seconds on idle connections, preventing proxy timeouts during long agent runs.

For real-time token streaming (not batch-then-chunk), replace `runner.run()`
with `runner.run_stream()` and forward each `chunk.delta` as a `token` event:

```python
async def generate():
    current_user_id.set(account.user_id)
    try:
        async for chunk in await self._runner.run_stream(
            self._crm_agent,
            full_prompt,
            conversation_id=body.conversation_id,
            execution_context=exec_ctx,
        ):
            if chunk.delta:
                yield ServerSentEvent(event="token", data=chunk.delta)
        yield ServerSentEvent(event="done", data="")
    except Exception as exc:
        yield ServerSentEvent(event="error", data=str(exc))
```

See the [streaming guide](streaming.md) for `CompletionChunk` field details.

---

## `ExecutionContext` as the security anchor

**Never trust the LLM to supply a user identity.**  The LLM is a language
model — it will do what the prompt says, including what an attacker's
prompt-injection attack says.  Identity must flow from the authenticated HTTP
layer to the tool layer via a path the LLM cannot touch.

`lauren` provides `ExecutionContext` for exactly this purpose.  Pass it from
the controller to `runner.run()` and it flows through the entire call chain:

```
SignatureGuard.can_activate(ExecutionContext)
    └─ request.state.user_id = <HMAC-verified value>
            │
AgentRunner.run(..., execution_context=exec_ctx)
    └─ AgentContext.execution_context   (same ExecutionContext object)
            │
ToolExecutor._execute_single_tool(...)
    └─ ToolContext.execution_context    (forwarded from AgentContext)
            │
tool.run(ctx: ToolContext, ...)
    └─ ctx.execution_context.request.state.get("user_id")   ← HERE
```

The `_auth_uid` helper in `banking_tools.py` captures this pattern:

```python title="app/ai/banking_tools.py"
def _auth_uid(ctx: ToolContext) -> str:
    """Extract the guard-verified user_id from the execution context."""
    exec_ctx = ctx.execution_context
    if exec_ctx is None:
        return ""
    request = getattr(exec_ctx, "request", None)
    if request is None:
        return ""
    state = getattr(request, "state", None)
    if state is None:
        return ""
    return (state.get("user_id") or "").lower()
```

`TransferFundsTool` calls `_auth_uid(ctx)` to determine the sender — the LLM
supplies only `to_user` and `amount`:

```python title="app/ai/banking_tools.py"
@tool()
class TransferFundsTool:
    """Execute a verified fund transfer between SecureBank accounts.

    The sender is always the user authenticated in the current session — the
    agent cannot override this.  Only the recipient and amount are LLM-supplied.

    Args:
        to_user: Recipient user ID (alice, bob, or charlie — cannot equal the
                 authenticated sender).
        amount: Amount in USD to transfer (must be positive, max $100,000).
        description: Optional memo or transfer description.
    """

    def __init__(self, db: BankDatabase) -> None:
        self._db = db

    async def run(
        self,
        ctx: ToolContext,
        to_user: str,
        amount: float,
        description: str = "",
    ) -> dict:
        auth_uid = _auth_uid(ctx)
        if not auth_uid:
            return {"error": "Security error: no authenticated user found in ExecutionContext."}
        # ... transfer logic using auth_uid as the sender
```

The controller enriches `request.state` with the full account details after
the guard runs:

```python title="app/ai/chat_banking_controller.py"
# exec_ctx is a lauren ExecutionContext injected into the handler.
# SignatureGuard already ran and set request.state.user_id.
request = exec_ctx.request
user_id = (request.state.get("user_id") or body.user_id).lower()

account = self._db.get_account(user_id)

# Enrich request.state with canonical account details.
request.state.user_id = account.user_id
request.state.user_name = account.name
request.state.account_id = account.account_id

response = await self._runner.run(
    self._crm_agent,
    full_prompt,
    conversation_id=body.conversation_id,
    execution_context=exec_ctx,   # <-- security anchor
)
```

> **Never read identity from tool arguments.**  If a tool receives
> `user_id: str` as an LLM-supplied argument, a prompt injection attack can
> supply an arbitrary value.  Read from `ctx.execution_context.request.state`
> exclusively.

---

## Delegation with a distinct DI token

Multi-agent delegation introduces a DI challenge: if the CRM `AgentRunner`
depends on `DelegateToBankingTransfer`, and `DelegateToBankingTransfer`
depends on `AgentRunner`, the container sees a cycle and raises
`CircularDependencyError` at startup.

The solution is a **named runner subclass** used as a distinct DI token:

```python title="app/ai/banking_delegation.py"
from lauren import injectable, Scope
from lauren_ai import AgentRunnerBase

@injectable(scope=Scope.SINGLETON)
class TransferAgentRunner(AgentRunnerBase):
    """Dedicated runner subclass for the Transfer Agent.

    Named DI token so DelegateToBankingTransfer can inject it specifically,
    avoiding ProtocolAmbiguityError in the CRM module scope that sees both runners.
    """

@injectable(scope=Scope.SINGLETON)
class CRMAgentRunner(AgentRunnerBase):
    """Dedicated runner subclass for the CRM Agent."""
```

`DelegateToBankingTransfer` injects `TransferAgentRunner`, not `AgentRunner`:

```python title="app/ai/banking_delegation.py"
@tool()
class DelegateToBankingTransfer:
    """Delegate a banking transfer task to the Transfer Agent.

    Args:
        task: Full description of what the Transfer Agent should do.
              Include amount, recipient, and any relevant context.
    """

    def __init__(
        self,
        transfer_agent: BankingTransferAgent,
        runner: TransferAgentRunner,    # distinct token — no cycle
    ) -> None:
        self._transfer_agent = transfer_agent
        self._runner = runner

    async def run(self, ctx: ToolContext, task: str) -> dict:
        # Forward execution_context so TransferFundsTool can read user_id.
        response = await self._runner.run(
            self._transfer_agent,
            task,
            execution_context=ctx.execution_context,
        )
        return {"result": response.content, "stop_reason": response.stop_reason}
```

The two modules use separate `AgentModule.for_root()` calls, each with its own
named runner token. The delegation tool lives in the **calling module's `tools=`**
(CRM module); the CRM module imports the Transfer module so `TransferAgentRunner`
is visible to DI when resolving `DelegateToBankingTransfer`:

```python title="app/ai/ai_module.py"
from app.ai.banking_delegation import (
    DelegateToBankingTransfer, TransferAgentRunner, CRMAgentRunner,
)
from app.ai.banking_tools import GetBalanceTool, GetTransactionHistoryTool, TransferFundsTool
from app.ai.crm_agent import BankingCRMAgent
from app.ai.transfer_agent import BankingTransferAgent

# Step 1: wire the Transfer Agent module under its own named runner token.
_TransferAgentModule = AgentModule.for_root(
    agents=[BankingTransferAgent],
    tools=[TransferFundsTool],
    imports=[LLMProvider, BankingModule],
    signals=signal_bus,
    conversation_store=_conversation_store,
    injects=[TransferAgentRunner],   # registers TransferAgentRunner as this module's runner
)

# Step 2: wire the CRM Agent module; import TransferAgentModule so that
# DelegateToBankingTransfer can resolve TransferAgentRunner.
# The delegation tool lives here — it belongs to the calling module.
_CRMAgentModule = AgentModule.for_root(
    agents=[BankingCRMAgent],
    tools=[DelegateToBankingTransfer, GetBalanceTool, GetTransactionHistoryTool],
    imports=[LLMProvider, _TransferAgentModule, BankingModule],
    signals=signal_bus,
    conversation_store=_conversation_store,
    injects=[CRMAgentRunner],        # registers CRMAgentRunner as this module's runner
)
```

The dependency graph the container sees:

```
CRMAgentRunner
    └─ DelegateToBankingTransfer
            └─ TransferAgentRunner   ← distinct token; resolved independently
```

No cycle.  Startup succeeds.  The `execution_context` forwarded by
`DelegateToBankingTransfer.run()` ensures `TransferFundsTool` reads the same
HMAC-verified `user_id` that the original HTTP guard pinned.

---

## `SignalBus` routing to WebSocket clients

`AgentRunner` emits signals at every stage of the agentic loop:
`ModelCallComplete`, `ToolCallStarted`, `ToolCallComplete`, `AgentRunComplete`.
Routing these to the right WebSocket client requires two things: a shared
`SignalBus` singleton and a `ContextVar` that carries the current user's
identity through `asyncio.gather`.

### Shared `SignalBus`

Define the bus in its own module to avoid circular imports:

```python title="app/ai/signals.py"
from __future__ import annotations
from lauren_ai import SignalBus

signal_bus: SignalBus = SignalBus()
```

Both `ai_module.py` and any module that listens to signals import from here.
The same instance is passed to all `AgentModule.for_root(signals=signal_bus)`
calls, so every runner — CRM and Transfer — emits onto the same bus.

### `ContextVar` for per-request routing

```python title="app/ws/context.py"
from __future__ import annotations
from contextvars import ContextVar

# Holds the authenticated user_id for the current agent run.
# asyncio.gather() copies the ContextVar context into spawned tasks, so
# this value is visible inside SignalBus handlers called via emit().
current_user_id: ContextVar[str | None] = ContextVar("ws_current_user_id", default=None)
```

The controller sets this before calling `runner.run()`:

```python title="app/ai/chat_banking_controller.py"
current_user_id.set(account.user_id)
response = await self._runner.run(
    self._crm_agent,
    full_prompt,
    conversation_id=body.conversation_id,
    execution_context=exec_ctx,
)
```

### `EventForwarder` — singleton that routes events to sockets

```python title="app/ws/event_forwarder.py"
from __future__ import annotations

import asyncio
from typing import Any

from lauren import Scope, injectable
from lauren.websockets import WebSocket
from lauren_ai import AgentRunComplete, ModelCallComplete, ToolCallComplete, ToolCallStarted

from app.ai.signals import signal_bus
from app.ws.context import current_user_id


@injectable(scope=Scope.SINGLETON)
class EventForwarder:
    """Singleton that maintains per-user WebSocket registrations and routes events."""

    def __init__(self, db: BankDatabase) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

        # Register handlers once at construction time.
        signal_bus.on(ModelCallComplete)(self._on_model_complete)
        signal_bus.on(ToolCallStarted)(self._on_tool_started)
        signal_bus.on(ToolCallComplete)(self._on_tool_complete)
        signal_bus.on(AgentRunComplete)(self._on_run_complete)

        db.add_transfer_listener(self._on_transfer)

    async def _on_model_complete(self, event: ModelCallComplete) -> None:
        user_id = current_user_id.get()   # reads ContextVar from asyncio.gather copy
        if not user_id:
            return
        usage = event.usage
        await self.send_to_user(
            user_id,
            {
                "type": "token_usage",
                "model": event.model,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "cost_usd": event.cost_usd,
                "duration_ms": round(event.duration_ms),
            },
        )
```

Because `SignalBus.emit()` uses `asyncio.gather()` internally, each handler
task inherits the `ContextVar` context from the task that called `emit()`,
which is the task that called `runner.run()`, which is the task that called
`current_user_id.set(account.user_id)`.  The user identity propagates through
the entire chain without any explicit argument passing.

### WebSocket gateway

`BankingWsGateway` is a `@ws_controller` that maps connections to user IDs via
a short-lived token:

```python title="app/ws/ws_gateway.py"
from lauren import Query
from lauren.websockets import (
    WebSocket,
    WebSocketDisconnect,
    on_connect,
    on_disconnect,
    ws_controller,
)
from app.ws.event_forwarder import EventForwarder
from app.ws.token_service import WsTokenService


@ws_controller("/ws/banking")
class BankingWsGateway:
    """WebSocket gateway: authenticates via short-lived token then forwards events."""

    def __init__(self, forwarder: EventForwarder, token_service: WsTokenService) -> None:
        self._forwarder = forwarder
        self._token_service = token_service
        self._user_id: str | None = None

    @on_connect
    async def connect(self, ws: WebSocket, token: Query[str]) -> None:
        user_id = self._token_service.verify_token(token)
        if not user_id:
            await ws.close(code=4401, reason="invalid or expired token")
            raise WebSocketDisconnect("unauthorized", close_code=4401)
        self._user_id = user_id
        await self._forwarder.register(user_id, ws)

    @on_disconnect
    async def disconnect(self, ws: WebSocket) -> None:
        if self._user_id:
            await self._forwarder.unregister(self._user_id, ws)
```

`@ws_controller` automatically makes the class `@injectable(scope=Scope.REQUEST)`,
so each connection gets its own gateway instance and `self._user_id` is
connection-private.  `EventForwarder` is a singleton injected into the
constructor — it holds the cross-connection registry.

Events pushed to clients after each agent run:

| Event type | Trigger |
|---|---|
| `token_usage` | After each LLM call (`ModelCallComplete`) |
| `tool_started` | When a tool call is dispatched (`ToolCallStarted`) |
| `tool_complete` | When a tool call finishes (`ToolCallComplete`) |
| `run_complete` | When the full agent run terminates (`AgentRunComplete`) |
| `balance_changed` | Broadcast to ALL users after any transfer (database callback) |

---

## Module structure and wiring

The final module graph for the banking chatbot:

```python title="app/app_module.py"
from lauren import module
from app.banking.banking_module import BankingModule
from app.health.health_module import HealthModule
from app.ws.ws_module import WsModule
from app.ai.ai_module import AIModule
from app.metrics.metrics_module import MetricsModule

@module(imports=[BankingModule, HealthModule, WsModule, AIModule, MetricsModule])
class AppModule:
    pass
```

```python title="app/ws/ws_module.py"
from lauren import module
from app.banking.banking_module import BankingModule
from app.crypto.crypto_module import CryptoModule
from app.ws.event_forwarder import EventForwarder
from app.ws.token_service import WsTokenService
from app.ws.ws_gateway import BankingWsGateway
from app.ws.ws_token_controller import WsTokenController

@module(
    imports=[CryptoModule, BankingModule],
    providers=[EventForwarder, WsTokenService],
    controllers=[BankingWsGateway, WsTokenController],
    exports=[EventForwarder],
)
class WsModule:
    """Real-time WebSocket module for live banking events."""
```

```python title="app/ai/ai_module.py"
@module(
    imports=[LLMProvider, _CRMAgentModule, _TransferAgentModule, BankingModule, CryptoModule],
    providers=[_cost_tracker_provider],
    exports=[LLMService, BankingCRMAgent, CostTracker],
    controllers=[BankingChatController],
)
class AIModule:
    """Provides banking AI services: agents, runners, and cost tracker."""
```

Module boundaries and visibility at a glance:

```
AppModule
├── BankingModule          — BankDatabase singleton (exported)
├── HealthModule           — /health endpoint
├── WsModule               — EventForwarder (exported), BankingWsGateway
│   ├── imports: CryptoModule, BankingModule
│   └── exports: EventForwarder
├── AIModule               — BankingChatController
│   ├── imports: LLMProvider, _CRMAgentModule, _TransferAgentModule
│   │              BankingModule, CryptoModule
│   └── exports: LLMService, BankingCRMAgent, CostTracker
│       ├── _CRMAgentModule  (injects=[CRMAgentRunner])
│       │   ├── BankingCRMAgent, GetBalanceTool, GetTransactionHistoryTool
│       │   └── DelegateToBankingTransfer → runner: TransferAgentRunner
│       └── _TransferAgentModule  (injects=[TransferAgentRunner])
│           └── BankingTransferAgent, TransferFundsTool
└── MetricsModule          — /api/metrics/* (injects CostTracker from AIModule)
```

---

## Production checklist

Before shipping an integration like this, verify:

- [ ] All `@tool()` files omit `from __future__ import annotations` — or use
  the class form exclusively (class-form tools use `typing.get_type_hints()`
  inside `AgentModule` after startup, which is safe with the future import).
- [ ] Every privileged tool reads identity from
  `ctx.execution_context.request.state`, not from LLM-supplied arguments.
- [ ] Circular DI dependencies are broken via distinct runner subclass tokens,
  not via `__post_init__` hacks or module-level globals.
- [ ] The `SignalBus` singleton is defined in its own module to avoid circular
  imports between `ai_module.py` and any listener module.
- [ ] `current_user_id.set(...)` is called inside the SSE generator, not
  outside it, so the `ContextVar` assignment is visible to
  `asyncio.gather`-spawned signal handler tasks.
- [ ] `EventForwarder` registers signal handlers in `__init__`, not in a
  `@post_construct` hook — the DI container calls `__init__` at startup, which
  is the right time to bind to the bus.
- [ ] Runner and agent singletons are constructor-injected into controllers, not
  resolved per-request.  Singletons are cheap to inject and expensive to construct.
  Use `runner: AgentRunner` (Protocol) for single-module scope; use the named
  concrete subclass (`injects=[MyRunner]`) when two or more AgentModules are
  in scope to avoid `ProtocolAmbiguityError`.

---

## Related guides

- [Agents](agents.md) — `@agent()`, lifecycle hooks, `AgentRunner.run()`
- [Tools](tools.md) — `@tool()`, `ToolContext`, schema generation rules
- [Streaming](streaming.md) — `AgentRunner.run_stream()`, SSE controller patterns
- [Multi-agent](multi-agent.md) — delegation patterns, tool-based handoff
- [Tracing](tracing.md) — `@traced()`, `Span`, `TraceStore`
- [Cost tracking](cost-tracking.md) — `CostTracker`, `ModelCallComplete`, pricing tables
