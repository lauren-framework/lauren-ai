# Customer Service Bot

Multi-turn bot with conversation memory, output guardrails, and agent handoff. Suitable for production.

## Agent definition

```python
from lauren_ai import agent, tool, use_tools, use_guardrails
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._guardrails import LLMOutputGuardrail   # or use a custom one

_SYSTEM = """\
You are a SecureBank customer service agent for authenticated customers.
You can help with: account balances, transaction history, and general questions.
For fund transfers, say: "I'll connect you to our Transfer Specialist."
For disputes or fraud, say: "I'll connect you to our Disputes team."
Never make up balances or transaction data — only use tool results.\
"""

_SCOPE = "Answer questions about account balances, transaction history, and general banking."
_REDIRECT = "I can only help with account questions. Would you like me to transfer you to the right team?"

@agent(
    name="Customer Service Bot",
    model=None,
    system=_SYSTEM,
    max_turns=8,
    conversation_store=InMemoryConversationStore(),
)
@use_guardrails(
    output=[ScopeGuard(allowed_scope=_SCOPE, redirect_message=_REDIRECT)],
)
@use_tools(GetBalanceTool, GetTransactionHistoryTool, HandoffTool)
class CustomerServiceAgent: ...
```

## Account tools

```python
from lauren_ai import tool, ToolContext

@tool()
async def get_balance(ctx: ToolContext) -> dict:
    """Return the authenticated user's current balance.
    No arguments needed — identity comes from execution context.
    """
    user_id = ctx.execution_context.request.state.get("user_id")
    if not user_id:
        return {"error": "Not authenticated"}
    balance = await bank_db.get_balance(user_id)
    return {"balance_usd": balance, "user_id": user_id}

@tool()
async def get_transaction_history(ctx: ToolContext, limit: int = 10) -> dict:
    """Return the last N transactions for the authenticated user.
    Args:
        limit: Maximum number of transactions to return (default 10).
    """
    user_id = ctx.execution_context.request.state.get("user_id")
    if not user_id:
        return {"error": "Not authenticated"}
    txns = await bank_db.get_transactions(user_id, limit=limit)
    return {"transactions": txns}
```

Identity always comes from `ctx.execution_context.request.state` (set by the guard), never from LLM-supplied parameters.

## Handoff tool

```python
from lauren_ai import tool, ToolContext
from app.ai.tools.active_agent_store import ActiveAgentStore

@tool()
class HandoffTool:
    """Hand the conversation off to a specialist agent.
    Args:
        to_agent: Target agent name.
        summary: Brief summary of the conversation so far.
    """
    def __init__(self, store: ActiveAgentStore) -> None:
        self._store = store

    async def run(self, ctx: ToolContext, to_agent: str, summary: str) -> dict:
        conv_id = ctx.agent_context.metadata.get("conversation_id", "")
        if conv_id:
            self._store.set(conv_id, to_agent)
            self._store.set_pending_summary(conv_id, summary)
        return {"status": "handed_off", "to_agent": to_agent}
```

## Multi-turn controller

```python
from lauren import controller, post, EventStream, ServerSentEvent, Json, use_guards
from lauren.types import ExecutionContext
from lauren_ai import AgentRunner

@use_guards(SignatureGuard)
@controller("/api/chat")
class ChatController:
    def __init__(
        self,
        runner: AgentRunner[CustomerServiceAgent],
        agent: CustomerServiceAgent,
        active_store: ActiveAgentStore,
    ) -> None:
        self._runner = runner
        self._agent = agent
        self._store = active_store

    @post("/message")
    async def message(self, body: Json[ChatRequest], exec_ctx: ExecutionContext) -> EventStream:
        user_id = exec_ctx.request.state.get("user_id", "")
        conv_id = body.conversation_id

        async def generate():
            async for chunk in await self._runner.run_stream(
                self._agent, body.message,
                conversation_id=conv_id,
                execution_context=exec_ctx,
                metadata={"conversation_id": conv_id},
            ):
                if chunk.delta:
                    yield ServerSentEvent(event="token", data=chunk.delta)
                elif chunk.guardrail_override is not None:
                    yield ServerSentEvent(event="guardrail_override", data=chunk.guardrail_override)
            yield ServerSentEvent(event="done", data="")

        return EventStream(generate(), keep_alive=15.0)
```
