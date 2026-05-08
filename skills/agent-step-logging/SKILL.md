---
name: agent-step-logging
description: Log each agentic loop step using lifecycle hooks on the @agent class. Use when you need structured per-step logs for debugging, auditing, or monitoring without modifying the runner or adding external infrastructure.
---

> Use `codemap find "AgentContext"` to check the lifecycle hook signatures.

# Agent Execution Step Logging

Define `on_start`, `on_turn_complete`, `on_tool_result`, and `on_finish`
lifecycle hooks on your `@agent()`-decorated class to log each step.

## Hook signatures

| Hook | Parameters | Called when |
|---|---|---|
| `on_start` | `(ctx: AgentContext)` | Before first LLM call |
| `on_turn_complete` | `(completion: Completion, ctx: AgentContext)` | After each LLM call |
| `on_tool_result` | `(result: ToolResult, ctx: AgentContext) -> ToolResult \| None` | After each tool execution |
| `on_finish` | `(response: AgentResponse, ctx: AgentContext)` | After the run completes |

## Pattern

```python
import logging
from lauren_ai import agent, use_tools
from lauren_ai._agents import AgentContext, AgentResponse
from lauren_ai._transport import Completion
from lauren_ai._tools import ToolResult

logger = logging.getLogger(__name__)

@agent(model="claude-sonnet-4-6", system="You are a helpful assistant.")
class LoggingAgent:
    async def on_start(self, ctx: AgentContext) -> None:
        logger.info("Agent started: run_id=%s turn=%d", ctx.agent_run_id, ctx.turn)

    async def on_turn_complete(self, completion: Completion, ctx: AgentContext) -> None:
        logger.info(
            "Turn %d complete: stop_reason=%s tokens_out=%d",
            ctx.turn,
            completion.stop_reason,
            completion.usage.output_tokens,
        )

    async def on_tool_result(self, result: ToolResult, ctx: AgentContext) -> ToolResult | None:
        logger.info(
            "Tool %s result: is_error=%s",
            result.tool_use_id,
            result.is_error,
        )
        return None  # return None to leave result unchanged

    async def on_finish(self, response: AgentResponse, ctx: AgentContext) -> None:
        logger.info(
            "Agent finished: turns=%d total_in=%d total_out=%d",
            response.turns,
            response.total_usage.input_tokens,
            response.total_usage.output_tokens,
        )
```

## Capture logs in tests

Use a list-based handler to assert on log output without configuring a real
logging destination:

```python
import logging

class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []
    def emit(self, record):
        self.records.append(self.format(record))

handler = ListHandler()
logging.getLogger("myapp.agents").addHandler(handler)
logging.getLogger("myapp.agents").setLevel(logging.INFO)
```

Or use pytest's built-in `caplog` fixture:

```python
@pytest.mark.asyncio
async def test_hooks_called(caplog):
    with caplog.at_level(logging.INFO):
        response = await runner.run(agent_instance, "Hello")
    assert any("Agent started" in r for r in caplog.messages)
    assert any("Agent finished" in r for r in caplog.messages)
```

## Notes

- Hooks are **optional** — define only the ones you need.
- `on_tool_result` returning `None` leaves the result unchanged; returning a
  new `ToolResult` replaces it (useful for sanitising sensitive outputs).
- Hooks run synchronously or asynchronously — both `def` and `async def` work.
- The runner swallows hook exceptions (logging a warning) so a broken hook
  never aborts the agent run.
