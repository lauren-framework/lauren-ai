---
name: agent-supervisor-router
description: Implements a supervisor/router pattern where a central agent classifies user intent and dispatches to specialist agents. Use when building multi-agent systems that route requests based on domain (billing, technical support, general), implementing intent classification before delegation, or orchestrating specialist agents from a coordinator.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Agent Supervisor / Router Pattern

## Overview

The supervisor pattern separates **routing logic** from **domain logic**. A
central supervisor agent classifies intent and hands off to the appropriate
specialist. Each specialist only handles its domain; the supervisor never
answers domain questions directly.

```
User Message
     │
     ▼
SupervisorAgent
  ├── classify_intent() ──► {intent: "billing", confidence: 0.9}
  └── route to ──► BillingAgent / TechSupportAgent / GeneralAgent
```

---

## Quick start

```python
from lauren_ai import agent, tool, use_tools
from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import tool

SPECIALISTS = {
    "billing": BillingAgent,
    "technical": TechSupportAgent,
    "general": GeneralAgent,
}

@tool()
async def classify_intent(message: str) -> dict:
    """Classify the intent of a user message.

    Args:
        message: The user message to classify.
    """
    if any(w in message.lower() for w in ["invoice", "payment", "charge"]):
        return {"intent": "billing", "confidence": 0.9}
    elif any(w in message.lower() for w in ["error", "bug", "crash", "slow"]):
        return {"intent": "technical", "confidence": 0.85}
    return {"intent": "general", "confidence": 0.7}


@agent(model="claude-opus-4-6", system="Route user to the correct specialist. Call classify_intent first.")
@use_tools(classify_intent)
class SupervisorAgent: ...
```

---

## Wiring with AgentModule

```python
from lauren_ai import AgentModule, LLMModule

AIModule = AgentModule.for_root(
    agents=[SupervisorAgent, BillingAgent, TechSupportAgent, GeneralAgent],
    tools=[classify_intent],
    imports=[LLMModule.for_root(cfg)],
)
```

---

## Dispatch pattern

After the supervisor calls `classify_intent`, your application logic dispatches
to the appropriate specialist runner:

```python
async def handle(message: str, runner) -> str:
    # Step 1: classify
    supervisor_resp = await runner.run(SupervisorAgent(), message)

    # Step 2: extract intent from tool call results
    intent = extract_intent_from_response(supervisor_resp)

    # Step 3: dispatch to specialist
    specialist_cls = SPECIALISTS.get(intent, GeneralAgent)
    specialist_resp = await runner.run(specialist_cls(), message)
    return specialist_resp.content
```

---

## Testing

Use `MockTransport` to queue the intent classification tool call response and
then the final answer. The runner handles the tool call → result → final turn
loop automatically.

```python
from lauren_ai._transport._mock import MockTransport
from lauren_ai._transport import Completion, TokenUsage, ToolCall

mock = MockTransport()
# Turn 1: supervisor calls classify_intent
mock.queue_tool_use("classify_intent", {"message": "I need help with my invoice"})
# Turn 2: supervisor gets tool result, produces routing decision
mock.queue_response(Completion(..., content="Routing to billing specialist"))
```

---

## Common mistakes

- **Do not** let the supervisor answer domain questions — keep it a pure router.
- **Do not** hardcode domain logic inside `classify_intent`; in production,
  replace with an LLM call or a dedicated classification model.
- Always set `confidence` thresholds — route to `general` when confidence is
  below 0.6 to avoid mis-routing.

---

## Reference

- `lauren_ai._agents`: `agent`, `use_tools`, `AgentResponse`
- `lauren_ai._tools`: `tool`, `ToolContext`
- `lauren_ai._transport._mock`: `MockTransport`
- Skills: `multi-agent-delegation`, `building-tools`
