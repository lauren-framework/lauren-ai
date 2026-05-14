# Memory

`lauren-ai` has four memory tiers:

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

`ConversationStore` persists full message history across multiple `run()` calls
within a session. Declare the store on the agent itself, then supply a
`conversation_id` on every `run()` call:

```python
from lauren_ai import agent, InMemoryConversationStore

store = InMemoryConversationStore()

@agent(model="claude-opus-4-6", conversation_store=store)
class MyAgent: ...
```

When `runner.run()` receives a `conversation_id`, the runner automatically:
1. **Loads** the prior history from the store and seeds `ShortTermMemory`.
2. **Saves** the updated history back to the store after the run.

```python
# Turn 1 — agent sees no prior history
resp1 = await runner.run(agent, "My name is Alice.", conversation_id="sess-1")

# Turn 2 — agent sees the Turn 1 exchange
resp2 = await runner.run(agent, "What is my name?", conversation_id="sess-1")
# resp2.content → "Your name is Alice."
```

Without `conversation_store` the `conversation_id` is accepted but ignored —
each call starts with an empty `ShortTermMemory`.  Different IDs are completely
isolated from each other.

For production, implement the `ConversationStore` protocol backed by Redis,
PostgreSQL, or any other store and attach that instance with
`@agent(conversation_store=...)`. Per-call overrides via
`runner.run(agent, ..., conversation_store=other_store)` still win when needed.

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
