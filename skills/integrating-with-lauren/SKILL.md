---
name: integrating-with-lauren
description: Wires lauren-ai into a lauren web framework application. Use when setting up LLMModule/AgentModule, streaming agent responses as SSE, securing tools with ExecutionContext, routing SignalBus events to WebSocket clients, or breaking circular DI with a runner subclass.
---

# Integrating lauren-ai with the lauren Framework

## Module wiring

### Step 1 — `LLMModule`

```python
from lauren_ai import LLMConfig, LLMModule

LLMProvider = LLMModule.for_root(
    LLMConfig.for_anthropic(model="claude-opus-4-6")  # reads ANTHROPIC_API_KEY
    # or: LLMConfig.for_openai(model="openai/gpt-4o-mini")  # reads OPENAI_API_KEY
)
```

`LLMModule.for_root()` registers `LLMService`, `EmbedService`, `LLMConfig`, and the
transport layer as singletons. Pass it to the parent `@module`'s `imports=`.

### Step 2 — `AgentModule`

```python
from lauren_ai import AgentModule, InMemoryConversationStore, SignalBus

signal_bus = SignalBus()   # shared singleton — keep in a dedicated module

AgentProvider = AgentModule.for_root(
    agents=[BankingCRMAgent, BankingTransferAgent],
    tools=[GetBalanceTool, TransferFundsTool, DelegateToBankingTransfer],
    imports=[LLMProvider],
    signals=signal_bus,
    conversation_store=InMemoryConversationStore(),
)
```

`AgentModule.for_root()` auto-registers each agent class and tool class as a
singleton provider and generates a unique `AgentRunnerBase` subclass as the module's
runner token. Inject it with `runner: AgentRunner` (single-module scope) or with the
named concrete subclass via `runner=MyRunner` (multi-module scope).

### Step 3 — parent `@module`

```python
from lauren import module

@module(
    imports=[AgentProvider, WsModule],
    controllers=[BankingChatController, MetricsController],
    providers=[EventForwarder, TransferAgentRunner],  # wiring singletons
    exports=[AgentProvider],
)
class AIModule: ...
```

Any class in `providers=` that subscribes to `SignalBus` events or wires runner
references in its `__init__` must appear here so Lauren's lifecycle scheduler
instantiates it eagerly at startup.

---

## Streaming agent responses as SSE

```python
from lauren import controller, post, EventStream, ServerSentEvent, use_guards
from lauren.types import ExecutionContext
from lauren_ai import AgentRunner
from app.ws.context import current_user_id

@use_guards(SignatureGuard)
@controller("/api/banking")
class BankingChatController:
    def __init__(self, runner: AgentRunner, agent: BankingCRMAgent) -> None:
        self._runner = runner
        self._agent = agent

    @post("/chat")
    async def stream(self, body: Json[ChatRequest], exec_ctx: ExecutionContext) -> EventStream:
        request = exec_ctx.request
        user_id = request.state.get("user_id") or body.user_id

        async def generate():
            # Set ContextVar BEFORE runner.run() so signal handlers emitted
            # during the agent loop can route events to the right WebSocket.
            # Do NOT call ContextVar.reset() — each SSE keep-alive chunk runs
            # in its own asyncio.Task with a copied context; reset() across
            # task boundaries raises ValueError.
            current_user_id.set(user_id)
            try:
                response = await self._runner.run(
                    self._agent,
                    body.messages[-1].content,
                    conversation_id=body.conversation_id,
                    execution_context=exec_ctx,   # ← security anchor
                )
                for i in range(0, len(response.content), 40):
                    yield ServerSentEvent(event="token", data=response.content[i:i+40])
                yield ServerSentEvent(event="done", data="")
            except Exception as exc:
                yield ServerSentEvent(event="error", data=str(exc))

        return EventStream(generate(), keep_alive=15.0)
```

---

## `ExecutionContext` as the security anchor

`ExecutionContext` flows from the HTTP handler through `AgentRunner.run()` into every
`ToolContext`. Identity should always be read from `ctx.execution_context.request.state`,
never from LLM-supplied parameters.

```
SignatureGuard.can_activate()
  └─ sets request.state.user_id from HMAC-verified payload body
BankingChatController.stream()
  └─ exec_ctx received via ExecutionContext injection (no marker needed)
  └─ runner.run(..., execution_context=exec_ctx)
       └─ AgentContext.execution_context = exec_ctx
            └─ ToolContext.execution_context = exec_ctx
                 └─ TransferFundsTool._auth_uid(ctx) reads
                    ctx.execution_context.request.state.get("user_id")
```

```python
def _auth_uid(ctx: ToolContext | None) -> str:
    if ctx is None:
        return ""
    ec = ctx.execution_context
    if ec is None:
        return ""
    state = getattr(getattr(ec, "request", None), "state", None)
    if state is None:
        return ""
    return state.get("user_id") or ""
```

Key rule: **never add `from_user` or `authenticated_user` to a tool's parameter
list** — that exposes them in the LLM-visible JSON schema and creates a
prompt-injection surface.

---

## Routing `SignalBus` events to WebSocket clients

```python
# app/ai/signals.py — shared singleton to avoid circular imports
from lauren_ai import SignalBus
signal_bus: SignalBus = SignalBus()

# app/ws/context.py — ContextVar propagated by asyncio.gather
from contextvars import ContextVar
current_user_id: ContextVar[str | None] = ContextVar("ws_current_user_id", default=None)
```

```python
from lauren import Scope, injectable
from lauren_ai import ModelCallComplete, ToolCallStarted, AgentRunComplete
from app.ai.signals import signal_bus
from app.ws.context import current_user_id

@injectable(scope=Scope.SINGLETON)
class EventForwarder:
    def __init__(self, db: BankDatabase) -> None:
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()
        # Subscribe once at construction — stable for the app's lifetime
        signal_bus.on(ModelCallComplete)(self._on_model_complete)
        signal_bus.on(ToolCallStarted)(self._on_tool_started)
        signal_bus.on(AgentRunComplete)(self._on_run_complete)

    async def _on_model_complete(self, event: ModelCallComplete) -> None:
        user_id = current_user_id.get()   # asyncio.gather copies ContextVar state
        if not user_id:
            return
        await self.send_to_user(user_id, {"type": "token_usage", ...})
```

`SignalBus.emit` uses `asyncio.gather` internally, which copies the calling
task's `Context` (including all `ContextVar` values) into every spawned coroutine.
Any `ContextVar` set before `runner.run()` is therefore visible inside signal
handlers without any extra wiring.

---

## Multi-runner disambiguation — named `AgentRunnerBase` subclass per module

Every `AgentModule.for_root()` call MUST have its own dedicated runner token.
When a controller or delegation tool can see runners from two modules simultaneously,
use `runner=MyRunner` with an explicit `AgentRunnerBase` subclass:

```python
from lauren import injectable, Scope
from lauren_ai import AgentRunnerBase

@injectable(scope=Scope.SINGLETON)
class TransferAgentRunner(AgentRunnerBase):
    """Distinct DI token for the Transfer module's runner."""

@injectable(scope=Scope.SINGLETON)
class CRMAgentRunner(AgentRunnerBase):
    """Distinct DI token for the CRM module's runner."""
```

```python
# Transfer module — registers TransferAgentRunner as its runner token
TransferMod = AgentModule.for_root(
    agents=[BankingTransferAgent],
    tools=[TransferFundsTool],
    imports=[LLMProvider],
    signals=signal_bus,
    runner=TransferAgentRunner,
)

# CRM module — imports Transfer module, owns the delegation tool
CRMMod = AgentModule.for_root(
    agents=[BankingCRMAgent],
    tools=[DelegateToBankingTransfer],     # ← delegation tool lives in the calling module
    imports=[LLMProvider, TransferMod],    # ← makes TransferAgentRunner visible
    signals=signal_bus,
    runner=CRMAgentRunner,
)
```

The delegation tool uses `runner: TransferAgentRunner` (named subclass, not
`AgentRunner` Protocol) — DI resolves it unambiguously from the imported scope.

Always forward `execution_context` in the delegation tool:

```python
async def run(self, ctx: ToolContext, transfer_request: str) -> dict:
    response = await self._runner.run(
        self._agent,
        transfer_request,
        execution_context=ctx.execution_context,
    )
    return {"content": response.content}
```

---

## Reference files

| File | Contents |
|------|----------|
| [docs/guides/lauren-integration.md](../../docs/guides/lauren-integration.md) | Full guide: module wiring, SSE, security, delegation, SignalBus |
| [../securing-agents/SKILL.md](../securing-agents/SKILL.md) | `ExecutionContext` security patterns and the full trust chain |
| [../building-tools/SKILL.md](../building-tools/SKILL.md) | `@tool()`, `ToolContext`, schema generation |
| [../building-agents/SKILL.md](../building-agents/SKILL.md) | `@agent()`, lifecycle hooks, `@use_tools` |
