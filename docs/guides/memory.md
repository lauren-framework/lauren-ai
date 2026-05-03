# Memory

`lauren-ai` has three memory tiers:

| Tier | Class | Scope | Purpose |
|---|---|---|---|
| Short-term | `ShortTermMemory` | Per run | Rolling message window passed to the LLM each turn |
| Conversation | `ConversationStore` | Per session | Persists full message history across runs |
| User memory | `UserMemoryStore` / `@remember()` | Per user | Long-term facts extracted by the `@remember()` decorator |

---

## ShortTermMemory

`ShortTermMemory` is managed automatically by `AgentRunner`.  One instance is
created per `run()` call.  It trims the message window to fit within a token
budget (default: 40 000 tokens, using a 4-chars-per-token heuristic) and is
accessible in lifecycle hooks via `ctx.memory`.

```python
from lauren_ai._memory import ShortTermMemory

memory = ShortTermMemory(max_tokens=8000)
memory.add_user("Hello!")
memory.add_assistant(completion)

msgs = memory.messages()           # trimmed to token budget
print(memory.token_estimate)       # heuristic token count
print(len(memory))                 # number of messages in buffer
```

The `max_tokens` for an agent's `ShortTermMemory` is set via
`AgentConfig.memory_window_tokens`:

```python
from lauren_ai import agent

@agent(model="claude-opus-4-6", memory_window_tokens=20_000)
class MyAgent: ...
```

---

## ConversationStore

`ConversationStore` is a protocol for persisting message histories across
multiple `run()` calls within the same session.  Pass a `conversation_id` to
`AgentRunner.run()` to enable history loading:

```python
response = await runner.run(
    agent,
    "Continue our conversation.",
    conversation_id="session-42",
)
```

The built-in `InMemoryConversationStore` is wired automatically.  For
production, implement the protocol backed by Redis, a database, or another
store and pass it to `AgentModule.for_root(conversation_store=...)`.

---

## User memory — @remember()

For per-user long-term memory that persists across conversations, see the full
guide: [user-memory.md](user-memory.md).

`@remember()` attaches a `UserMemoryStore` to an agent.  After each turn, the
runner extracts facts from the conversation and stores them.  On subsequent
runs, the top-k relevant facts are injected into the system prompt automatically.

```python
from lauren_ai import agent, remember, use_tools

@agent(model="claude-opus-4-6", system="You are a personal assistant.")
@remember(store_token=UserMemoryStore, top_k=5)
@use_tools(WebSearchTool)
class PersonalAssistant: ...
```

See [user-memory.md](user-memory.md) for the full `@remember()` API,
`MemoryFact`, `UserMemoryStore` protocol, and `InMemoryUserMemoryStore`.
