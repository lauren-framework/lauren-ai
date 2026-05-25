# Examples

This directory contains runnable examples that demonstrate how to compose the
public `lauren-ai` APIs into realistic workflows.

## Environment Setup

For examples that use a live model provider, copy `examples/.env.example` to
`examples/.env` and fill in the provider-specific values you need.

The current inter-agent messaging example does not require any API key because
it runs entirely in memory.

## Inter-Agent Messaging

File: `examples/inter_agent_messaging_workflow.py`

Run it from the repository root:

```bash
uv run python examples/inter_agent_messaging_workflow.py
```

What it demonstrates:

- `AgentMessageBus` as the shared runtime for peer-to-peer coordination
- `AgentMessage` as the strongly typed envelope for workflow messages
- direct request/response between a planner and a researcher
- topic publication from the researcher to the writer
- streaming draft generation from the writer back to the planner
- correlation IDs, `session_id`, and `task_id` scoping
- `SignalBus` observability hooks for sent/completed message events
- graceful shutdown with zero dead letters in the happy path

Workflow shape:

1. The planner broadcasts a kickoff notification.
2. The planner sends a `TASK_REQUEST` to the researcher.
3. The researcher publishes shared findings on the `research.findings` topic and replies with an outline.
4. The planner sends a second `TASK_REQUEST` to the writer.
5. The writer consumes the previously published findings and streams draft sections back to the planner.
6. The planner assembles the streamed sections into a final brief.

Why this example exists:

- End users can see how to wire a multi-agent workflow without needing a live LLM provider.
- Repository maintainers can use it as an executable reference for the intended messaging semantics and public API ergonomics.
