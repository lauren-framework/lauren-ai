---
name: inspecting-streams
description: Inspects the raw token stream a real LLM emits for a given system + user prompt, with chunk boundaries preserved. Use when debugging dense / unformatted model output (run-on text, missing newlines, glued-on emoji), validating that a system-prompt change produces the markdown structure you expected, or comparing two prompts / models side-by-side without spinning up the agentic loop or any tools.
---

# Inspecting LLM Streams

## When to use

You added a "use blank lines between paragraphs" directive to your agent's
system prompt and want proof the model now emits `\n\n` between sentences.
You have a chat UI that renders messages as run-on text and you suspect the
model itself is the source rather than the renderer.  You want to compare
how two models (Claude 3.5 Sonnet vs. a free open-source model) handle the
same prompt structurally.

In all these cases you want the **raw token deltas** from the transport,
not an aggregated `Completion` object — chunk boundaries reveal what the
model actually emitted versus what your wire format / parser preserved.

## What it does

`inspect_stream.py` calls a transport's `complete(..., stream=True)`
directly, prints each `chunk.delta` with `repr()` (so `\n` shows
explicitly), then prints the full accumulated text and a count of
newlines.  No agent loop, no tool calls, no SSE framing — just LLM input
→ LLM output.

```
--- STREAMED CHUNK DELTAS (repr'd, \n preserved) ---
'Here'
' are'
' the'
' details'
':'
'\n\n'
'-'
' **'
'Recipient'
'**:'
' Bob'
...

--- FULL ACCUMULATED TEXT ---
Here are the details:

- **Recipient**: Bob
...

--- BACKSLASH-N COUNT: 6  --  TOTAL LEN: 91 ---
```

## Running the script

```bash
# Set transport credentials (Anthropic / OpenAI / OpenRouter all work).
export OPENROUTER_API_KEY=sk-or-v1-...
export LLM_MODEL="poolside/laguna-xs.2:free"

uv run python skills/inspecting-streams/inspect_stream.py \
    --system "You are a banking assistant.  Use blank lines between paragraphs and put each list item on its own line." \
    --prompt "Summarise the products SecureBank offers."
```

Or import the helper into a one-off REPL session:

```python
import asyncio
from skills.inspecting_streams.inspect_stream import trace
from lauren_ai import LLMConfig

cfg = LLMConfig.for_anthropic(model="claude-haiku-4-5")
asyncio.run(trace(cfg, system="You are a poet.", user_prompt="Write three lines about the sea."))
```

## Interpreting the output

- **`'\\n\\n'` between chunks** → the model is emitting paragraph breaks.
  Renderer should produce real paragraphs.
- **`' '` (single space) where you expected `'\\n'`** → the model is
  treating the boundary as a soft break.  Either tighten the system
  prompt's formatting directive or accept the result.
- **`-` followed immediately by content with no preceding `'\\n'`** →
  the model is jamming list items onto the same line as the heading.
  System-prompt fix: show a concrete bullet-list example.
- **Trailing `'  \\n'` (two spaces + newline)** → CommonMark hard line
  break.  GFM renders this as `<br>`; harmless but worth noting.

## Why bypass the agent loop

The agentic loop adds tool execution, hooks, signal emission, and
sometimes a follow-up turn.  When you only want to know "what does this
model emit for this exact prompt", those layers obscure the picture.
This skill reaches past `AgentRunner.run_stream` straight to
`Transport.complete(..., stream=True)` for the cleanest signal.

## Anti-patterns

- **Don't use this to test agent behaviour end-to-end.**  Use
  `AgentTestClient` (see `testing-agents` skill) for that — it covers
  tools, hooks, multi-turn flows, and assertions.
- **Don't add ad-hoc whitespace heuristics on the renderer side because
  this script shows dense output.**  Fix the system prompt.  Renderer
  heuristics inevitably misfire on legitimate identifiers
  (`SecureBank`, `OAuth`, `GitHub`) that look structurally identical to
  missing-space sentence boundaries.
