# Lauren AI Skills Index

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> the exact file + line range and is faster than grep.

| Skill | Purpose |
|---|---|
| [building-agents](building-agents/) | `@agent`, lifecycle hooks (`on_start`, `on_finish`, …), streaming, agentic loop |
| [building-tools](building-tools/) | `@tool` function and class forms, `ToolContext` DI, HITL, caching |
| [building-teams](building-teams/) | `@team`, coordinator vs collaborate modes, `TeamRunner`, streaming events |
| [managing-memory](managing-memory/) | `InMemoryConversationStore`, `UserMemoryStore`, `@remember`, `VectorStore` |
| [adding-guardrails](adding-guardrails/) | Input/output filters, PII redaction, prompt-injection defence, `@guardrail` |
| [securing-agents](securing-agents/) | Identity trust chain, `ToolContext`, `ExecutionContext`, never trusting LLM identity |
| [testing-agents](testing-agents/) | `MockTransport`, `AgentTestClient`, multi-turn flows, memory assertions |
| [integrating-with-lauren](integrating-with-lauren/) | `LLMModule`, `AgentModule`, SSE streaming, `ExecutionContext`, `AgentRunner[X]` |
| [inspecting-streams](inspecting-streams/) | Raw token stream, debug LLM output, chunk boundaries |
| [migrating-to-lauren-ai](migrating-to-lauren-ai/) | LangChain → lauren-ai; OpenAI SDK → `LLMService` / `@tool` / `@agent` |
| [common-agent-patterns](common-agent-patterns/) | Copy-paste: research agent, customer service bot, data-analysis agent |

## Quick nav by error

| Error / symptom | Go to |
|---|---|
| Tool schema `{}` or missing params | [building-tools](building-tools/) §Schema generation |
| `ProtocolAmbiguityError` on `AgentRunner` | [integrating-with-lauren](integrating-with-lauren/) §Cross-module DI |
| Guardrail fires but SSE not replaced | [adding-guardrails](adding-guardrails/) §SSE integration |
| Memory not persisted across turns | [managing-memory](managing-memory/) §Conversation store |
| Signal handler fires N times | [integrating-with-lauren](integrating-with-lauren/) §SignalBus |
| `TypeError` passing agent class (not instance) | [building-agents](building-agents/) §Runner |

## See also

- [`AGENTS.md`](../AGENTS.md) — by-task lookup, common errors, full decorator API reference
- [`CLAUDE.md`](../CLAUDE.md) — architecture invariants, pattern selection, codemap navigation
- [`llms-full.txt`](../llms-full.txt) — complete API reference
