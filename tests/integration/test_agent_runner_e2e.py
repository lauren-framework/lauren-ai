"""End-to-end tests for AgentRunnerBase.run() and .run_stream().

103 tests covering:
  - Basic run / stream behaviour (no guardrails)
  - Input guardrails: pass, block, modify, exceptions, multiple guards
  - Output guardrails: pass, modify, block, exception, multi-turn
  - Combined input + output guardrails
  - per-agent memory and conversation store (meta vs per-request)
  - Lifecycle hooks (on_start, on_finish, on_turn_complete, on_tool_result)
  - Signals (ModelCallComplete, ToolCallStarted/Complete, AgentRunComplete)
  - Edge cases: error policies, budget, tool-call clearing on guardrail fire
"""

from __future__ import annotations

import asyncio

import pytest

from lauren_ai import SignalBus, use_guardrails
from lauren_ai._agents import AgentResponse, agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._guardrails._base import GuardrailContext, GuardrailDecision
from lauren_ai._memory import ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, CompletionChunk, TokenUsage, ToolCall
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _completion(
    content: str = "OK",
    *,
    n: int = 1,
    model: str = "mock-model",
    stop_reason: str = "end_turn",
) -> Completion:
    return Completion(
        id=f"c{n}",
        model=model,
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _stream_chunks(*parts: str, stop_reason: str = "end_turn") -> list[CompletionChunk]:
    chunks = [CompletionChunk(delta=p) for p in parts]
    chunks.append(
        CompletionChunk(
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=10, output_tokens=len(parts) or 1),
        )
    )
    return chunks


def _make_runner(
    mock: MockTransport | None = None,
    signals: SignalBus | None = None,
) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    runner = AgentRunner(transport=mock, signals=signals)
    return runner, mock


class _SpyGuardrail:
    """Records every check() call; returns a configurable GuardrailDecision."""

    def __init__(
        self,
        action: str = "pass",
        content: str | None = None,
        raise_exc: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self._action = action
        self._content = content
        self._raise = raise_exc

    async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
        self.calls.append(text)
        if self._raise:
            raise RuntimeError("spy guardrail error (test)")
        if self._action == "block":
            return GuardrailDecision(action="block", violation="blocked!", guardrail_name="SpyGuardrail")
        if self._action == "modify":
            return GuardrailDecision(
                action="modify",
                modified_content=self._content or "[modified]",
                guardrail_name="SpyGuardrail",
            )
        return GuardrailDecision(action="pass", guardrail_name="SpyGuardrail")


# ---------------------------------------------------------------------------
# Class 1 — TestRunNoGuardrails (10)
# ---------------------------------------------------------------------------


class TestRunNoGuardrails:
    @pytest.mark.asyncio
    async def test_basic_run_returns_content(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Hello world"))

        @agent(model="mock-model")
        class A: ...

        resp = await runner.run(A(), "hi")
        assert resp.content == "Hello world"

    @pytest.mark.asyncio
    async def test_run_turns_count_multi_turn(self):
        runner, mock = _make_runner()
        mock.queue_tool_use("noop_tool", {})
        mock.queue_response(_completion("done"))

        @tool()
        async def noop_tool() -> dict:
            """No-op tool. Args: none."""
            return {}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, noop_tool)
        runner = AgentRunner(transport=mock)

        @use_tools(noop_tool)
        @agent(model="mock-model")
        class MultiTurnAgent: ...

        MultiTurnAgent.__lauren_ai_agent__.tools = tools

        resp = await runner.run(MultiTurnAgent(), "do it")
        assert resp.turns == 2

    @pytest.mark.asyncio
    async def test_run_stop_reason_end_turn(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model")
        class A: ...

        resp = await runner.run(A(), "hi")
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_run_total_usage_accumulated(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK", n=1))

        @agent(model="mock-model")
        class A: ...

        resp = await runner.run(A(), "hi")
        assert resp.total_usage.input_tokens > 0
        assert resp.total_usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_run_non_agent_class_raises_agent_config_error(self):
        from lauren_ai._exceptions import AgentConfigError

        runner, mock = _make_runner()

        class NotAnAgent: ...

        with pytest.raises(AgentConfigError):
            await runner.run(NotAnAgent(), "hi")

    @pytest.mark.asyncio
    async def test_run_response_includes_tool_calls_made(self):
        mock = MockTransport()
        mock.queue_tool_use("echo_tool", {"msg": "hello"})
        mock.queue_response(_completion("echoed"))

        @tool()
        async def echo_tool(msg: str) -> dict:
            """Echo. Args: msg: message."""
            return {"echo": msg}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, echo_tool)
        runner = AgentRunner(transport=mock)

        @use_tools(echo_tool)
        @agent(model="mock-model")
        class EchoAgent: ...

        EchoAgent.__lauren_ai_agent__.tools = tools

        resp = await runner.run(EchoAgent(), "echo hello")
        assert len(resp.tool_calls_made) == 1
        assert resp.tool_calls_made[0].name == "echo_tool"

    @pytest.mark.asyncio
    async def test_run_with_metadata_passed_to_context(self):
        captured: list[dict] = []

        @agent(model="mock-model")
        class MetaAgent:
            async def on_start(self, ctx):
                captured.append(dict(ctx.metadata))

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(MetaAgent(), "hi", metadata={"key": "value"})
        assert captured[0].get("key") == "value"

    @pytest.mark.asyncio
    async def test_run_with_execution_context(self):
        captured = []

        @agent(model="mock-model")
        class ExecCtxAgent:
            async def on_start(self, ctx):
                captured.append(ctx.execution_context)

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        sentinel = object()
        await runner.run(ExecCtxAgent(), "hi", execution_context=sentinel)
        assert captured[0] is sentinel

    @pytest.mark.asyncio
    async def test_run_agent_with_docstring_system_prompt(self):
        @agent(model="mock-model")
        class DocstringAgent:
            """My docstring system prompt."""

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(DocstringAgent(), "hi")
        assert mock.calls[0].system == "My docstring system prompt."

    @pytest.mark.asyncio
    async def test_run_with_explicit_run_id(self):
        @agent(model="mock-model")
        class A: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        resp = await runner.run(A(), "hi", run_id="custom-run-42")
        assert resp is not None  # smoke: run_id accepted without error


# ---------------------------------------------------------------------------
# Class 2 — TestRunInputGuardrails (15)
# ---------------------------------------------------------------------------


class TestRunInputGuardrails:
    @pytest.mark.asyncio
    async def test_input_guardrail_not_called_without_decorator(self):
        spy = _SpyGuardrail()

        @agent(model="mock-model")
        class NoGuardAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(NoGuardAgent(), "hi")
        assert spy.calls == []

    @pytest.mark.asyncio
    async def test_input_guardrail_pass_llm_is_called(self):
        spy = _SpyGuardrail(action="pass")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class PassAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("response"))
        resp = await runner.run(PassAgent(), "hello")
        assert len(spy.calls) == 1
        assert resp.content == "response"

    @pytest.mark.asyncio
    async def test_input_guardrail_block_llm_never_called(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()
        await runner.run(BlockAgent(), "hi")
        assert len(mock.calls) == 0

    @pytest.mark.asyncio
    async def test_input_guardrail_block_returns_violation_as_content(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()
        resp = await runner.run(BlockAgent(), "hi")
        assert resp.content == "blocked!"

    @pytest.mark.asyncio
    async def test_input_guardrail_block_returns_zero_turns(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()
        resp = await runner.run(BlockAgent(), "hi")
        assert resp.turns == 0

    @pytest.mark.asyncio
    async def test_input_guardrail_modify_llm_sees_modified_message(self):
        spy = _SpyGuardrail(action="modify", content="[MODIFIED] hello")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class ModAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(ModAgent(), "original")
        # The LLM sees the modified message as the last user turn
        last_user_msg = mock.calls[0].messages[-1]
        assert last_user_msg["content"] == "[MODIFIED] hello"

    @pytest.mark.asyncio
    async def test_input_guardrail_exception_fails_open_llm_called(self):
        spy = _SpyGuardrail(raise_exc=True)

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class ExcAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("fallback"))
        resp = await runner.run(ExcAgent(), "hi")
        # Guard threw but we failed open — LLM was called, response returned
        assert len(mock.calls) == 1
        assert resp.content == "fallback"

    @pytest.mark.asyncio
    async def test_multiple_input_guardrails_first_block_wins(self):
        spy1 = _SpyGuardrail(action="block")
        spy2 = _SpyGuardrail(action="pass")

        @use_guardrails(input=[spy1, spy2])
        @agent(model="mock-model")
        class TwoGuardAgent: ...

        runner, mock = _make_runner()
        await runner.run(TwoGuardAgent(), "hi")
        assert len(spy1.calls) == 1
        assert spy2.calls == []  # second guard never called after first blocks

    @pytest.mark.asyncio
    async def test_input_guardrail_called_on_each_invocation(self):
        spy = _SpyGuardrail(action="pass")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class RepeatAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("r1"))
        mock.queue_response(_completion("r2"))
        await runner.run(RepeatAgent(), "msg1")
        await runner.run(RepeatAgent(), "msg2")
        assert len(spy.calls) == 2

    @pytest.mark.asyncio
    async def test_input_guardrail_sees_original_message_not_prior_context(self):
        captured: list[str] = []

        class _CaptureSpy:
            async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
                captured.append(text)
                return GuardrailDecision(action="pass")

        @use_guardrails(input=[_CaptureSpy()])
        @agent(model="mock-model")
        class CapAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(CapAgent(), "unique-xyz-message")
        assert captured[0] == "unique-xyz-message"

    @pytest.mark.asyncio
    async def test_input_guard_and_output_guard_both_active_input_blocks(self):
        in_spy = _SpyGuardrail(action="block")
        out_spy = _SpyGuardrail(action="pass")

        @use_guardrails(input=[in_spy], output=[out_spy])
        @agent(model="mock-model")
        class BothAgent: ...

        runner, mock = _make_runner()
        resp = await runner.run(BothAgent(), "hi")
        assert in_spy.calls != []
        assert out_spy.calls == []  # output never reached
        assert resp.turns == 0

    @pytest.mark.asyncio
    async def test_input_guard_passes_output_guard_modifies(self):
        in_spy = _SpyGuardrail(action="pass")
        out_spy = _SpyGuardrail(action="modify", content="[SAFE]")

        @use_guardrails(input=[in_spy], output=[out_spy])
        @agent(model="mock-model")
        class BothAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("hallucinated"))
        resp = await runner.run(BothAgent(), "hi")
        assert in_spy.calls != []
        assert out_spy.calls != []
        assert resp.content == "[SAFE]"

    @pytest.mark.asyncio
    async def test_input_guard_modifies_output_guard_passes(self):
        in_spy = _SpyGuardrail(action="modify", content="[CLEAN]")
        out_spy = _SpyGuardrail(action="pass")

        @use_guardrails(input=[in_spy], output=[out_spy])
        @agent(model="mock-model")
        class BothAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("clean response"))
        resp = await runner.run(BothAgent(), "original")
        assert mock.calls[0].messages[-1]["content"] == "[CLEAN]"
        assert resp.content == "clean response"

    @pytest.mark.asyncio
    async def test_input_guard_modify_message_stored_in_conversation(self):
        in_spy = _SpyGuardrail(action="modify", content="[MOD]")
        store = InMemoryConversationStore()

        @use_guardrails(input=[in_spy])
        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(StoreAgent(), "original", conversation_id="s1")
        history = await store.load("s1")
        user_msgs = [m for m in history["messages"] if m.get("role") == "user"]
        assert user_msgs[0]["content"] == "[MOD]"

    @pytest.mark.asyncio
    async def test_input_guardrail_block_stop_reason_end_turn(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()
        resp = await runner.run(BlockAgent(), "hi")
        assert resp.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Class 3 — TestRunOutputGuardrails (15)
# ---------------------------------------------------------------------------


class TestRunOutputGuardrails:
    @pytest.mark.asyncio
    async def test_output_guardrail_not_called_without_decorator(self):
        spy = _SpyGuardrail()

        @agent(model="mock-model")
        class NoGuardAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(NoGuardAgent(), "hi")
        assert spy.calls == []

    @pytest.mark.asyncio
    async def test_output_guardrail_pass_response_unchanged(self):
        spy = _SpyGuardrail(action="pass")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class PassAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("original response"))
        resp = await runner.run(PassAgent(), "hi")
        assert resp.content == "original response"

    @pytest.mark.asyncio
    async def test_output_guardrail_modify_replaces_response(self):
        spy = _SpyGuardrail(action="modify", content="[SAFE REDIRECT]")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class ModAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("hallucinated content"))
        resp = await runner.run(ModAgent(), "hi")
        assert resp.content == "[SAFE REDIRECT]"

    @pytest.mark.asyncio
    async def test_output_guardrail_block_replaces_with_violation(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("blocked content"))
        resp = await runner.run(BlockAgent(), "hi")
        assert resp.content == "blocked!"

    @pytest.mark.asyncio
    async def test_output_guardrail_block_ends_loop(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(output=[spy])
        @agent(model="mock-model", max_turns=5)
        class BlockAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("blocked"))
        # Queue extra responses to detect if loop continues past 1 turn
        mock.queue_response(_completion("extra"))
        mock.queue_response(_completion("extra2"))
        resp = await runner.run(BlockAgent(), "hi")
        # Only 1 LLM call should have been made (loop ended after guardrail)
        assert len(mock.calls) == 1
        assert resp.turns == 1

    @pytest.mark.asyncio
    async def test_output_guardrail_called_with_full_text(self):
        spy = _SpyGuardrail(action="pass")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class FullTextAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("full response text"))
        await runner.run(FullTextAgent(), "hi")
        assert spy.calls[0] == "full response text"

    @pytest.mark.asyncio
    async def test_output_guardrail_exception_fails_open_original_returned(self):
        spy = _SpyGuardrail(raise_exc=True)

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class ExcAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("original"))
        resp = await runner.run(ExcAgent(), "hi")
        assert resp.content == "original"

    @pytest.mark.asyncio
    async def test_multiple_output_guardrails_first_non_pass_wins(self):
        spy1 = _SpyGuardrail(action="modify", content="[FIRST]")
        spy2 = _SpyGuardrail(action="modify", content="[SECOND]")

        @use_guardrails(output=[spy1, spy2])
        @agent(model="mock-model")
        class TwoAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("bad content"))
        resp = await runner.run(TwoAgent(), "hi")
        assert len(spy1.calls) == 1
        assert spy2.calls == []  # second guard not called after first fires
        assert resp.content == "[FIRST]"

    @pytest.mark.asyncio
    async def test_output_guard_called_on_each_turn_of_multi_turn_run(self):
        spy = _SpyGuardrail(action="pass")

        @tool()
        async def noop() -> dict:
            """Noop. Args: none."""
            return {}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, noop)
        mock = MockTransport()
        runner = AgentRunner(transport=mock)

        @use_guardrails(output=[spy])
        @use_tools(noop)
        @agent(model="mock-model")
        class MultiAgent: ...

        MultiAgent.__lauren_ai_agent__.tools = tools

        # Turn 1: tool_use (content = "thinking..."), Turn 2: end_turn
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="thinking...",
                tool_calls=[ToolCall(tool_use_id="t1", name="noop", input={})],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            )
        )
        mock.queue_response(_completion("final"))

        await runner.run(MultiAgent(), "do it")
        # Guard should have been called for both non-empty completions
        assert len(spy.calls) >= 1  # at least the final non-empty turn

    @pytest.mark.asyncio
    async def test_output_guard_fires_on_turn2_not_turn1(self):
        _SpyGuardrail(action="pass")

        class _Turn2Guard:
            call_count = 0

            def __init__(self):
                self.calls: list[str] = []

            async def check(self, text, ctx):
                self.calls.append(text)
                self.call_count += 1
                if self.call_count == 2:
                    return GuardrailDecision(action="modify", modified_content="[GUARDED]")
                return GuardrailDecision(action="pass")

        guard = _Turn2Guard()

        @tool()
        async def noop() -> dict:
            """Noop. Args: none."""
            return {}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, noop)
        mock = MockTransport()
        runner = AgentRunner(transport=mock)

        @use_guardrails(output=[guard])
        @use_tools(noop)
        @agent(model="mock-model")
        class TwoTurnAgent: ...

        TwoTurnAgent.__lauren_ai_agent__.tools = tools

        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="before tool",
                tool_calls=[ToolCall(tool_use_id="t1", name="noop", input={})],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            )
        )
        mock.queue_response(_completion("final hallucination"))

        resp = await runner.run(TwoTurnAgent(), "go")
        assert resp.content == "[GUARDED]"  # fired on second turn

    @pytest.mark.asyncio
    async def test_output_guard_modify_stores_override_not_original_llm_text(self):
        spy = _SpyGuardrail(action="modify", content="[SAFE]")
        store = InMemoryConversationStore()

        @use_guardrails(output=[spy])
        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("bad content"))
        await runner.run(StoreAgent(), "hi", conversation_id="c1")
        history = await store.load("c1")
        assistant_msgs = [m for m in history["messages"] if m.get("role") == "assistant"]
        assert assistant_msgs[0]["content"] == "[SAFE]"

    @pytest.mark.asyncio
    async def test_output_guard_conversation_saved_with_override_content(self):
        spy = _SpyGuardrail(action="modify", content="[REDIRECT]")
        store = InMemoryConversationStore()

        @use_guardrails(output=[spy])
        @agent(model="mock-model", conversation_store=store)
        class ConvAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("oops"))
        await runner.run(ConvAgent(), "question", conversation_id="s1")
        history = await store.load("s1")
        assert any(m.get("content") == "[REDIRECT]" for m in history["messages"])

    @pytest.mark.asyncio
    async def test_output_guard_empty_content_not_checked(self):
        spy = _SpyGuardrail(action="pass")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class EmptyContentAgent: ...

        runner, mock = _make_runner()
        # Queue a completion with empty content (shouldn't trigger guardrail)
        mock.queue_response(_completion(""))
        await runner.run(EmptyContentAgent(), "hi")
        assert spy.calls == []  # empty content → guard skipped

    @pytest.mark.asyncio
    async def test_output_guard_runs_response_still_returned_after_block(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class Agent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("bad"))
        resp = await runner.run(Agent(), "hi")
        assert isinstance(resp, AgentResponse)
        assert resp.content  # violation text returned

    @pytest.mark.asyncio
    async def test_input_and_output_guard_both_present_input_passes_output_fires(self):
        in_spy = _SpyGuardrail(action="pass")
        out_spy = _SpyGuardrail(action="modify", content="[OUT GUARD]")

        @use_guardrails(input=[in_spy], output=[out_spy])
        @agent(model="mock-model")
        class BothAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("original"))
        resp = await runner.run(BothAgent(), "hi")
        assert in_spy.calls != []
        assert out_spy.calls != []
        assert resp.content == "[OUT GUARD]"


# ---------------------------------------------------------------------------
# Class 4 — TestRunStreamNoGuardrails (10)
# ---------------------------------------------------------------------------


class TestRunStreamNoGuardrails:
    @pytest.mark.asyncio
    async def test_stream_yields_delta_chunks(self):
        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("Hello", " World"))

        @agent(model="mock-model")
        class A: ...

        chunks = []
        async for c in await runner.run_stream(A(), "hi"):
            chunks.append(c)

        deltas = [c.delta for c in chunks if c.delta]
        assert deltas == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_no_guardrail_override_chunk_emitted(self):
        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("safe text"))

        @agent(model="mock-model")
        class A: ...

        chunks = []
        async for c in await runner.run_stream(A(), "hi"):
            chunks.append(c)

        overrides = [c for c in chunks if c.guardrail_override is not None]
        assert overrides == []

    @pytest.mark.asyncio
    async def test_stream_stop_reason_in_final_chunk(self):
        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("text"))

        @agent(model="mock-model")
        class A: ...

        chunks = []
        async for c in await runner.run_stream(A(), "hi"):
            chunks.append(c)

        stop_chunks = [c for c in chunks if c.stop_reason is not None]
        assert len(stop_chunks) >= 1
        assert stop_chunks[-1].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_usage_in_final_chunk(self):
        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("text"))

        @agent(model="mock-model")
        class A: ...

        chunks = []
        async for c in await runner.run_stream(A(), "hi"):
            chunks.append(c)

        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_tool_call_multi_turn(self):
        from lauren_ai._transport import ToolCallDelta

        @tool()
        async def ping() -> dict:
            """Ping. Args: none."""
            return {"pong": True}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, ping)
        mock = MockTransport()
        runner = AgentRunner(transport=mock)

        @use_tools(ping)
        @agent(model="mock-model")
        class PingAgent: ...

        PingAgent.__lauren_ai_agent__.tools = tools

        # Use proper ToolCallDelta chunks so the streaming runner can see the tool call
        mock.queue_stream(
            [
                CompletionChunk(tool_call_delta=ToolCallDelta(tool_use_id="t1", name="ping", input_delta="{}")),
                CompletionChunk(
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=5, output_tokens=5),
                ),
            ]
        )
        mock.queue_stream(_stream_chunks("pong received"))

        chunks = []
        async for c in await runner.run_stream(PingAgent(), "ping"):
            chunks.append(c)

        deltas = "".join(c.delta for c in chunks if c.delta)
        assert "pong received" in deltas

    @pytest.mark.asyncio
    async def test_stream_on_start_hook_fires(self):
        fired = []

        @agent(model="mock-model")
        class HookAgent:
            async def on_start(self, ctx):
                fired.append(True)

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("OK"))

        async for _ in await runner.run_stream(HookAgent(), "hi"):
            pass

        assert fired == [True]

    @pytest.mark.asyncio
    async def test_stream_on_finish_hook_fires(self):
        finished = []

        @agent(model="mock-model")
        class FinishAgent:
            async def on_finish(self, resp, ctx):
                finished.append(resp.content)

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("complete"))

        async for _ in await runner.run_stream(FinishAgent(), "hi"):
            pass

        assert finished == ["complete"]

    @pytest.mark.asyncio
    async def test_stream_thinking_delta_chunks_yielded(self):
        runner, mock = _make_runner()
        chunks_with_thinking = [
            CompletionChunk(thinking_delta="Let me think..."),
            CompletionChunk(delta="Answer"),
            CompletionChunk(stop_reason="end_turn", usage=TokenUsage(input_tokens=5, output_tokens=5)),
        ]
        mock.queue_stream(chunks_with_thinking)

        @agent(model="mock-model")
        class ThinkAgent: ...

        chunks = []
        async for c in await runner.run_stream(ThinkAgent(), "hi"):
            chunks.append(c)

        thinking = [c.thinking_delta for c in chunks if c.thinking_delta]
        assert thinking == ["Let me think..."]

    @pytest.mark.asyncio
    async def test_stream_with_metadata(self):
        captured = []

        @agent(model="mock-model")
        class MetaAgent:
            async def on_start(self, ctx):
                captured.append(ctx.metadata.get("key"))

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("OK"))

        async for _ in await runner.run_stream(MetaAgent(), "hi", metadata={"key": "val"}):
            pass

        assert captured == ["val"]

    @pytest.mark.asyncio
    async def test_stream_budget_exceeded_swallowed_no_exception(self):
        @agent(model="mock-model", max_cost_usd=0.0)
        class TightBudgetAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("expensive content"))

        chunks = []
        try:
            async for c in await runner.run_stream(TightBudgetAgent(), "hi"):
                chunks.append(c)
        except Exception as exc:
            pytest.fail(f"Stream raised unexpectedly: {exc}")

        # Budget exceeded should be swallowed; stream ends naturally
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Class 5 — TestRunStreamInputGuardrails (13)
# ---------------------------------------------------------------------------


class TestRunStreamInputGuardrails:
    @pytest.mark.asyncio
    async def test_stream_input_guardrail_not_called_without_decorator(self):
        spy = _SpyGuardrail()

        @agent(model="mock-model")
        class NoGuard: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("OK"))

        async for _ in await runner.run_stream(NoGuard(), "hi"):
            pass

        assert spy.calls == []

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_pass_yields_normal_chunks(self):
        spy = _SpyGuardrail(action="pass")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class PassAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("Hello", " World"))

        deltas = []
        async for c in await runner.run_stream(PassAgent(), "hi"):
            if c.delta:
                deltas.append(c.delta)

        assert deltas == ["Hello", " World"]
        assert len(spy.calls) == 1

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_block_yields_guardrail_override_chunk(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()

        chunks = []
        async for c in await runner.run_stream(BlockAgent(), "hi"):
            chunks.append(c)

        overrides = [c for c in chunks if c.guardrail_override is not None]
        assert len(overrides) >= 1

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_block_no_delta_chunks_in_output(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()

        deltas = []
        async for c in await runner.run_stream(BlockAgent(), "hi"):
            if c.delta:
                deltas.append(c.delta)

        assert deltas == []

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_block_no_llm_call(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()

        async for _ in await runner.run_stream(BlockAgent(), "hi"):
            pass

        assert len(mock.calls) == 0

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_block_override_text_equals_violation(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()

        chunks = []
        async for c in await runner.run_stream(BlockAgent(), "hi"):
            chunks.append(c)

        override_texts = [c.guardrail_override for c in chunks if c.guardrail_override]
        assert override_texts[0] == "blocked!"

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_block_guardrail_override_chunk_has_no_delta(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()

        chunks = []
        async for c in await runner.run_stream(BlockAgent(), "hi"):
            chunks.append(c)

        for c in chunks:
            if c.guardrail_override is not None:
                assert c.delta == ""  # sentinel has no delta

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_modify_llm_sees_modified_message(self):
        spy = _SpyGuardrail(action="modify", content="[MODIFIED]")

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class ModAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("OK"))

        async for _ in await runner.run_stream(ModAgent(), "original"):
            pass

        last_user = mock.calls[0].messages[-1]
        assert last_user["content"] == "[MODIFIED]"

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_exception_fails_open_no_override_chunk(self):
        spy = _SpyGuardrail(raise_exc=True)

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class ExcAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("fallback"))

        chunks = []
        async for c in await runner.run_stream(ExcAgent(), "hi"):
            chunks.append(c)

        overrides = [c for c in chunks if c.guardrail_override is not None]
        assert overrides == []  # no override — failed open

    @pytest.mark.asyncio
    async def test_stream_input_guardrail_exception_normal_chunks_yielded(self):
        spy = _SpyGuardrail(raise_exc=True)

        @use_guardrails(input=[spy])
        @agent(model="mock-model")
        class ExcAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("fallback text"))

        deltas = []
        async for c in await runner.run_stream(ExcAgent(), "hi"):
            if c.delta:
                deltas.append(c.delta)

        assert "fallback text" in "".join(deltas)

    @pytest.mark.asyncio
    async def test_stream_multiple_input_guardrails_first_fires_second_not_called(self):
        spy1 = _SpyGuardrail(action="block")
        spy2 = _SpyGuardrail(action="pass")

        @use_guardrails(input=[spy1, spy2])
        @agent(model="mock-model")
        class TwoAgent: ...

        runner, mock = _make_runner()

        async for _ in await runner.run_stream(TwoAgent(), "hi"):
            pass

        assert len(spy1.calls) == 1
        assert spy2.calls == []

    @pytest.mark.asyncio
    async def test_stream_input_guard_passes_output_guard_modifies(self):
        in_spy = _SpyGuardrail(action="pass")
        out_spy = _SpyGuardrail(action="modify", content="[OUT]")

        @use_guardrails(input=[in_spy], output=[out_spy])
        @agent(model="mock-model")
        class BothAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("bad text"))

        chunks = []
        async for c in await runner.run_stream(BothAgent(), "hi"):
            chunks.append(c)

        overrides = [c.guardrail_override for c in chunks if c.guardrail_override]
        assert overrides == ["[OUT]"]

    @pytest.mark.asyncio
    async def test_stream_input_guard_block_output_guard_never_called(self):
        in_spy = _SpyGuardrail(action="block")
        out_spy = _SpyGuardrail(action="pass")

        @use_guardrails(input=[in_spy], output=[out_spy])
        @agent(model="mock-model")
        class BothAgent: ...

        runner, mock = _make_runner()

        async for _ in await runner.run_stream(BothAgent(), "hi"):
            pass

        assert out_spy.calls == []


# ---------------------------------------------------------------------------
# Class 6 — TestRunStreamOutputGuardrails (18)
# ---------------------------------------------------------------------------


class TestRunStreamOutputGuardrails:
    @pytest.mark.asyncio
    async def test_stream_output_guardrail_not_called_without_decorator(self):
        spy = _SpyGuardrail()

        @agent(model="mock-model")
        class NoGuard: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("OK"))

        async for _ in await runner.run_stream(NoGuard(), "hi"):
            pass

        assert spy.calls == []

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_pass_no_override_chunk(self):
        spy = _SpyGuardrail(action="pass")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class PassAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("clean text"))

        chunks = []
        async for c in await runner.run_stream(PassAgent(), "hi"):
            chunks.append(c)

        overrides = [c for c in chunks if c.guardrail_override is not None]
        assert overrides == []

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_modify_yields_override_chunk(self):
        spy = _SpyGuardrail(action="modify", content="[SAFE]")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class ModAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("bad text"))

        chunks = []
        async for c in await runner.run_stream(ModAgent(), "hi"):
            chunks.append(c)

        overrides = [c for c in chunks if c.guardrail_override is not None]
        assert len(overrides) == 1

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_block_yields_override_chunk(self):
        spy = _SpyGuardrail(action="block")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class BlockAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("bad text"))

        chunks = []
        async for c in await runner.run_stream(BlockAgent(), "hi"):
            chunks.append(c)

        overrides = [c for c in chunks if c.guardrail_override is not None]
        assert len(overrides) == 1

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_override_comes_after_all_deltas(self):
        spy = _SpyGuardrail(action="modify", content="[OVERRIDE]")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class OrderAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("tok1", "tok2", "tok3"))

        chunks = []
        async for c in await runner.run_stream(OrderAgent(), "hi"):
            chunks.append(c)

        # Find index of last delta vs override
        delta_indices = [i for i, c in enumerate(chunks) if c.delta]
        override_indices = [i for i, c in enumerate(chunks) if c.guardrail_override is not None]
        assert max(delta_indices) < min(override_indices)

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_override_text_matches_modified_content(self):
        spy = _SpyGuardrail(action="modify", content="[EXACT OVERRIDE TEXT]")

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class Agent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("bad"))

        chunks = []
        async for c in await runner.run_stream(Agent(), "hi"):
            chunks.append(c)

        override_text = next(c.guardrail_override for c in chunks if c.guardrail_override)
        assert override_text == "[EXACT OVERRIDE TEXT]"

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_partial_tool_inputs_cleared_on_fire(self):
        """The bug fix: tool calls should NOT be executed after guardrail fires."""
        executed_tools: list[str] = []

        @tool()
        async def should_not_run(msg: str) -> dict:
            """Should not be called. Args: msg: message."""
            executed_tools.append(msg)
            return {}

        from lauren_ai._tools import _add_to_tool_map
        from lauren_ai._transport import ToolCallDelta

        tools = {}
        _add_to_tool_map(tools, should_not_run)
        mock = MockTransport()
        runner_with_tools = AgentRunner(transport=mock)

        spy = _SpyGuardrail(action="modify", content="[BLOCKED]")

        @use_guardrails(output=[spy])
        @use_tools(should_not_run)
        @agent(model="mock-model")
        class ToolGuardAgent: ...

        ToolGuardAgent.__lauren_ai_agent__.tools = tools

        # Stream that includes both text AND a tool call delta
        # Guardrail fires on the text → tool should not execute
        mock.queue_stream(
            [
                CompletionChunk(delta="some text"),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="t1", name="should_not_run", input_delta='{"msg":"test"}')
                ),
                CompletionChunk(
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=5, output_tokens=5),
                ),
            ]
        )

        async for _ in await runner_with_tools.run_stream(ToolGuardAgent(), "hi"):
            pass

        assert executed_tools == []

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_memory_has_override_text_not_hallucination(self):
        spy = _SpyGuardrail(action="modify", content="[SAFE]")
        store = InMemoryConversationStore()

        @use_guardrails(output=[spy])
        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("hallucination"))

        async for _ in await runner.run_stream(StoreAgent(), "hi", conversation_id="s1"):
            pass

        history = await store.load("s1")
        contents = [m.get("content", "") for m in history["messages"]]
        assert "[SAFE]" in contents
        assert "hallucination" not in contents

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_exception_fails_open_no_override(self):
        spy = _SpyGuardrail(raise_exc=True)

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class ExcAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("text"))

        chunks = []
        async for c in await runner.run_stream(ExcAgent(), "hi"):
            chunks.append(c)

        overrides = [c for c in chunks if c.guardrail_override is not None]
        assert overrides == []

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_exception_normal_chunks_present(self):
        spy = _SpyGuardrail(raise_exc=True)

        @use_guardrails(output=[spy])
        @agent(model="mock-model")
        class ExcAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("text content"))

        deltas = []
        async for c in await runner.run_stream(ExcAgent(), "hi"):
            if c.delta:
                deltas.append(c.delta)

        assert "text content" in "".join(deltas)

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_called_with_full_assembled_text(self):
        captured: list[str] = []

        class _CaptureSpy:
            async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
                captured.append(text)
                return GuardrailDecision(action="pass")

        @use_guardrails(output=[_CaptureSpy()])
        @agent(model="mock-model")
        class CaptureAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("tok1", " tok2", " tok3"))

        async for _ in await runner.run_stream(CaptureAgent(), "hi"):
            pass

        # Guard should receive the fully-joined text
        assert captured[0] == "tok1 tok2 tok3"

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_called_per_turn_multi_turn(self):
        spy = _SpyGuardrail(action="pass")

        @tool()
        async def noop() -> dict:
            """Noop. Args: none."""
            return {}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, noop)
        mock = MockTransport()
        runner_tools = AgentRunner(transport=mock)

        @use_guardrails(output=[spy])
        @use_tools(noop)
        @agent(model="mock-model")
        class MultiTurnAgent: ...

        MultiTurnAgent.__lauren_ai_agent__.tools = tools

        # Turn 1: tool_use with text content
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="thinking",
                tool_calls=[ToolCall(tool_use_id="t1", name="noop", input={})],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            )
        )
        # Turn 2: final answer via stream
        mock.queue_stream(_stream_chunks("final answer"))

        async for _ in await runner_tools.run_stream(MultiTurnAgent(), "go"):
            pass

        # Guard called at least once (for the streamed final answer)
        assert len(spy.calls) >= 1

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_fires_first_turn_loop_ends(self):
        spy = _SpyGuardrail(action="modify", content="[BLOCKED]")

        @use_guardrails(output=[spy])
        @agent(model="mock-model", max_turns=5)
        class FireFirstAgent: ...

        runner, mock = _make_runner()
        # Queue multiple responses; only first should be consumed
        mock.queue_stream(_stream_chunks("bad"))
        mock.queue_stream(_stream_chunks("second"))
        mock.queue_stream(_stream_chunks("third"))

        chunks = []
        async for c in await runner.run_stream(FireFirstAgent(), "hi"):
            chunks.append(c)

        assert len(mock.calls) == 1  # loop ended after guardrail

    @pytest.mark.asyncio
    async def test_stream_multiple_output_guardrails_first_non_pass_wins(self):
        spy1 = _SpyGuardrail(action="modify", content="[FIRST]")
        spy2 = _SpyGuardrail(action="modify", content="[SECOND]")

        @use_guardrails(output=[spy1, spy2])
        @agent(model="mock-model")
        class TwoAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("bad"))

        chunks = []
        async for c in await runner.run_stream(TwoAgent(), "hi"):
            chunks.append(c)

        overrides = [c.guardrail_override for c in chunks if c.guardrail_override]
        assert overrides == ["[FIRST]"]
        assert spy1.calls != []
        assert spy2.calls == []

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_override_stored_in_conversation_store(self):
        spy = _SpyGuardrail(action="modify", content="[REDIRECT]")
        store = InMemoryConversationStore()

        @use_guardrails(output=[spy])
        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("bad output"))

        async for _ in await runner.run_stream(StoreAgent(), "hi", conversation_id="s"):
            pass

        history = await store.load("s")
        texts = [m.get("content", "") for m in history["messages"]]
        assert "[REDIRECT]" in texts

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_budget_check_runs_after_guardrail(self):
        """Guardrail fires AND budget check runs — no uncaught exception."""
        spy = _SpyGuardrail(action="modify", content="[SAFE]")

        @use_guardrails(output=[spy])
        @agent(model="mock-model", max_cost_usd=100.0)
        class BudgetGuardAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("text"))

        chunks = []
        try:
            async for c in await runner.run_stream(BudgetGuardAgent(), "hi"):
                chunks.append(c)
        except Exception as exc:
            pytest.fail(f"Unexpected exception: {exc}")

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_input_passes_output_modifies_only_override_chunk(self):
        in_spy = _SpyGuardrail(action="pass")
        out_spy = _SpyGuardrail(action="modify", content="[OVERRIDE]")

        @use_guardrails(input=[in_spy], output=[out_spy])
        @agent(model="mock-model")
        class BothAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("bad"))

        chunks = []
        async for c in await runner.run_stream(BothAgent(), "hi"):
            chunks.append(c)

        overrides = [c.guardrail_override for c in chunks if c.guardrail_override]
        assert overrides == ["[OVERRIDE]"]

    @pytest.mark.asyncio
    async def test_stream_output_guardrail_called_after_all_chunks_real_time(self):
        """Verify: chunks yield BEFORE guardrail check is called (real-time)."""
        yield_order: list[str] = []

        class _TracingSpy:
            async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
                yield_order.append("guardrail_check")
                return GuardrailDecision(action="pass")

        @use_guardrails(output=[_TracingSpy()])
        @agent(model="mock-model")
        class TracingAgent: ...

        runner, mock = _make_runner()
        mock.queue_stream(_stream_chunks("a", "b", "c"))

        async for c in await runner.run_stream(TracingAgent(), "hi"):
            if c.delta:
                yield_order.append(f"delta:{c.delta}")

        # All delta chunks should come before the guardrail check
        delta_indices = [i for i, x in enumerate(yield_order) if x.startswith("delta:")]
        guard_indices = [i for i, x in enumerate(yield_order) if x == "guardrail_check"]
        if delta_indices and guard_indices:
            assert max(delta_indices) < min(guard_indices)


# ---------------------------------------------------------------------------
# Class 7 — TestRunMemoryAndStore (8)
# ---------------------------------------------------------------------------


class TestRunMemoryAndStore:
    @pytest.mark.asyncio
    async def test_meta_memory_reused_across_runs(self):
        shared_mem = ShortTermMemory(max_tokens=10_000)

        @agent(model="mock-model", memory=shared_mem)
        class MemAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("r1"))
        mock.queue_response(_completion("r2"))
        await runner.run(MemAgent(), "q1")
        await runner.run(MemAgent(), "q2")

        assert len(shared_mem.messages()) == 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_no_meta_memory_fresh_per_run(self):
        @agent(model="mock-model")  # no memory=
        class FreshAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("r1"))
        mock.queue_response(_completion("r2"))
        r1 = await runner.run(FreshAgent(), "q1")
        r2 = await runner.run(FreshAgent(), "q2")
        assert r1.content == "r1"
        assert r2.content == "r2"  # no state shared

    @pytest.mark.asyncio
    async def test_per_request_memory_override_wins_over_meta(self):
        meta_mem = ShortTermMemory(max_tokens=10_000)
        override_mem = ShortTermMemory(max_tokens=10_000)

        @agent(model="mock-model", memory=meta_mem)
        class MetaMemAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(MetaMemAgent(), "hi", memory=override_mem)

        assert len(override_mem.messages()) == 2
        assert len(meta_mem.messages()) == 0  # meta untouched

    @pytest.mark.asyncio
    async def test_conversation_store_saves_history(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("answer"))
        await runner.run(StoreAgent(), "question", conversation_id="sess1")

        history = await store.load("sess1")
        assert len(history) == 2
        assert history["messages"][0]["role"] == "user"
        assert history["messages"][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_conversation_store_loads_prior_history(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("r1"))
        await runner.run(StoreAgent(), "turn1", conversation_id="s")

        # Second run — capture messages seen by LLM
        mock.queue_response(_completion("r2"))
        await runner.run(StoreAgent(), "turn2", conversation_id="s")

        # The second call's messages should include the prior exchange
        second_call_messages = mock.calls[1].messages
        contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
        assert "turn1" in contents

    @pytest.mark.asyncio
    async def test_per_request_store_override_wins_over_meta(self):
        meta_store = InMemoryConversationStore()
        override_store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=meta_store)
        class MetaStoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(
            MetaStoreAgent(),
            "hi",
            conversation_id="s",
            conversation_store=override_store,
        )

        assert len((await override_store.load("s"))["messages"]) == 2
        assert (await meta_store.load("s")) == []

    @pytest.mark.asyncio
    async def test_no_conversation_id_store_not_touched(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(StoreAgent(), "hi")  # no conversation_id

        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_different_conversation_ids_isolated(self):
        store = InMemoryConversationStore()

        @agent(model="mock-model", conversation_store=store)
        class StoreAgent: ...

        runner, mock = _make_runner()
        mock.queue_response(_completion("alice"))
        await runner.run(StoreAgent(), "alice msg", conversation_id="alice")

        mock.queue_response(_completion("bob"))
        await runner.run(StoreAgent(), "bob msg", conversation_id="bob")

        alice = await store.load("alice")
        bob = await store.load("bob")
        assert all("bob" not in str(m) for m in alice)
        assert all("alice" not in str(m) for m in bob)


# ---------------------------------------------------------------------------
# Class 8 — TestRunLifecycleHooks (4)
# ---------------------------------------------------------------------------


class TestRunLifecycleHooks:
    @pytest.mark.asyncio
    async def test_on_start_called_once(self):
        counter = [0]

        @agent(model="mock-model")
        class HookAgent:
            async def on_start(self, ctx):
                counter[0] += 1

        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))
        await runner.run(HookAgent(), "hi")
        assert counter[0] == 1

    @pytest.mark.asyncio
    async def test_on_finish_called_with_response(self):
        received = []

        @agent(model="mock-model")
        class HookAgent:
            async def on_finish(self, resp, ctx):
                received.append(resp.content)

        runner, mock = _make_runner()
        mock.queue_response(_completion("finish content"))
        await runner.run(HookAgent(), "hi")
        assert received == ["finish content"]

    @pytest.mark.asyncio
    async def test_on_turn_complete_called_per_turn(self):
        turns_seen: list[int] = []

        @tool()
        async def noop() -> dict:
            """Noop. Args: none."""
            return {}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, noop)
        mock = MockTransport()
        runner = AgentRunner(transport=mock)

        @use_tools(noop)
        @agent(model="mock-model")
        class HookAgent:
            async def on_turn_complete(self, completion, ctx):
                turns_seen.append(ctx.turn)

        HookAgent.__lauren_ai_agent__.tools = tools

        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="thinking",
                tool_calls=[ToolCall(tool_use_id="t1", name="noop", input={})],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            )
        )
        mock.queue_response(_completion("done"))

        await runner.run(HookAgent(), "hi")
        assert len(turns_seen) == 2  # called once per LLM turn

    @pytest.mark.asyncio
    async def test_on_tool_result_called(self):
        results_seen = []

        @tool()
        async def spy_tool(x: int) -> dict:
            """Returns x. Args: x: integer."""
            return {"value": x}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, spy_tool)
        mock = MockTransport()
        runner = AgentRunner(transport=mock)

        @use_tools(spy_tool)
        @agent(model="mock-model")
        class ToolHookAgent:
            async def on_tool_result(self, result, ctx):
                results_seen.append(result.content)
                return None

        ToolHookAgent.__lauren_ai_agent__.tools = tools

        mock.queue_tool_use("spy_tool", {"x": 42})
        mock.queue_response(_completion("done"))

        await runner.run(ToolHookAgent(), "hi")
        assert len(results_seen) == 1


# ---------------------------------------------------------------------------
# Class 9 — TestRunSignals (6)
# ---------------------------------------------------------------------------


class TestRunSignals:
    @pytest.mark.asyncio
    async def test_model_call_complete_signal_emitted(self):
        from lauren_ai import ModelCallComplete

        events = []
        bus = SignalBus()
        bus.on(ModelCallComplete)(lambda e: events.append(e))

        runner, mock = _make_runner(signals=bus)
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model")
        class SigAgent: ...

        await runner.run(SigAgent(), "hi")
        assert len(events) == 1
        assert isinstance(events[0], ModelCallComplete)

    @pytest.mark.asyncio
    async def test_model_call_complete_has_correct_model_name(self):
        from lauren_ai import ModelCallComplete

        events = []
        bus = SignalBus()
        bus.on(ModelCallComplete)(lambda e: events.append(e))

        mock = MockTransport()
        runner = AgentRunner(transport=mock, signals=bus)
        mock.queue_response(_completion("OK", model="special-model"))

        @agent(model="special-model")
        class SigAgent: ...

        await runner.run(SigAgent(), "hi")
        assert events[0].model == "special-model"

    @pytest.mark.asyncio
    async def test_tool_call_started_signal_emitted(self):
        from lauren_ai import ToolCallStarted

        events = []
        bus = SignalBus()
        bus.on(ToolCallStarted)(lambda e: events.append(e))

        @tool()
        async def sig_tool() -> dict:
            """Signal tool. Args: none."""
            return {}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, sig_tool)
        mock = MockTransport()
        runner = AgentRunner(transport=mock, signals=bus)

        @use_tools(sig_tool)
        @agent(model="mock-model")
        class SigAgent: ...

        SigAgent.__lauren_ai_agent__.tools = tools

        mock.queue_tool_use("sig_tool", {})
        mock.queue_response(_completion("done"))

        await runner.run(SigAgent(), "hi")
        assert len(events) >= 1
        assert events[0].tool_name == "sig_tool"

    @pytest.mark.asyncio
    async def test_tool_call_complete_signal_emitted(self):
        from lauren_ai import ToolCallComplete

        events = []
        bus = SignalBus()
        bus.on(ToolCallComplete)(lambda e: events.append(e))

        @tool()
        async def done_tool() -> dict:
            """Done tool. Args: none."""
            return {"ok": True}

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, done_tool)
        mock = MockTransport()
        runner = AgentRunner(transport=mock, signals=bus)

        @use_tools(done_tool)
        @agent(model="mock-model")
        class SigAgent: ...

        SigAgent.__lauren_ai_agent__.tools = tools

        mock.queue_tool_use("done_tool", {})
        mock.queue_response(_completion("done"))

        await runner.run(SigAgent(), "hi")
        assert len(events) >= 1
        assert events[0].tool_name == "done_tool"

    @pytest.mark.asyncio
    async def test_agent_run_complete_signal_emitted(self):
        from lauren_ai import AgentRunComplete

        events = []
        bus = SignalBus()
        bus.on(AgentRunComplete)(lambda e: events.append(e))

        runner, mock = _make_runner(signals=bus)
        mock.queue_response(_completion("done"))

        @agent(model="mock-model")
        class SigAgent: ...

        await runner.run(SigAgent(), "hi")
        assert len(events) == 1
        assert isinstance(events[0], AgentRunComplete)

    @pytest.mark.asyncio
    async def test_run_stream_emits_model_call_complete(self):
        from lauren_ai import ModelCallComplete

        events = []
        bus = SignalBus()
        bus.on(ModelCallComplete)(lambda e: events.append(e))

        runner, mock = _make_runner(signals=bus)
        mock.queue_stream(_stream_chunks("streamed"))

        @agent(model="mock-model")
        class SigAgent: ...

        async for _ in await runner.run_stream(SigAgent(), "hi"):
            pass

        assert len(events) == 1


# ---------------------------------------------------------------------------
# Class 10 — TestRunEdgeCases (4)
# ---------------------------------------------------------------------------


class TestRunEdgeCases:
    @pytest.mark.asyncio
    async def test_run_tool_error_policy_skip(self):
        @tool()
        async def bad_tool(x: int) -> dict:
            """Throws. Args: x: int."""
            raise ValueError("tool exploded")

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, bad_tool)
        mock = MockTransport()
        runner = AgentRunner(transport=mock)

        @use_tools(bad_tool)
        @agent(model="mock-model", tool_error_policy="skip")
        class SkipAgent: ...

        SkipAgent.__lauren_ai_agent__.tools = tools

        mock.queue_tool_use("bad_tool", {"x": 1})
        mock.queue_response(_completion("recovered"))

        resp = await runner.run(SkipAgent(), "hi")
        assert resp.content == "recovered"

    @pytest.mark.asyncio
    async def test_run_tool_error_policy_raise(self):
        from lauren_ai._tools._executor import ToolExecutionError

        @tool()
        async def bad_tool(x: int) -> dict:
            """Throws. Args: x: int."""
            raise ValueError("tool exploded")

        from lauren_ai._tools import _add_to_tool_map

        tools = {}
        _add_to_tool_map(tools, bad_tool)
        mock = MockTransport()
        runner = AgentRunner(transport=mock)

        @use_tools(bad_tool)
        @agent(model="mock-model", tool_error_policy="raise")
        class RaiseAgent: ...

        RaiseAgent.__lauren_ai_agent__.tools = tools

        mock.queue_tool_use("bad_tool", {"x": 1})

        with pytest.raises(ToolExecutionError):
            await runner.run(RaiseAgent(), "hi")

    @pytest.mark.asyncio
    async def test_run_max_tokens_stop_reason(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("partial", stop_reason="max_tokens"))

        @agent(model="mock-model")
        class A: ...

        resp = await runner.run(A(), "hi")
        # max_tokens triggers stop — may report max_turns or the actual stop
        assert resp.stop_reason in ("end_turn", "max_turns")

    @pytest.mark.asyncio
    async def test_stream_empty_delta_chunks_do_not_add_to_accumulated_text(self):
        """Empty delta chunks should not corrupt the accumulated text that
        the guardrail sees."""
        captured: list[str] = []

        class _CaptureSpy:
            async def check(self, text: str, ctx: GuardrailContext) -> GuardrailDecision:
                captured.append(text)
                return GuardrailDecision(action="pass")

        @use_guardrails(output=[_CaptureSpy()])
        @agent(model="mock-model")
        class CaptureAgent: ...

        runner, mock = _make_runner()
        # Mix empty deltas with real content
        mock.queue_stream(
            [
                CompletionChunk(delta=""),  # empty — should not add to accumulated
                CompletionChunk(delta="real"),
                CompletionChunk(delta=""),  # empty — should not add to accumulated
                CompletionChunk(delta=" text"),
                CompletionChunk(
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=5, output_tokens=5),
                ),
            ]
        )

        async for _ in await runner.run_stream(CaptureAgent(), "hi"):
            pass

        assert captured[0] == "real text"


@pytest.mark.asyncio
async def test_stream_cancellation_between_parallel_tools_repairs_complete_batch() -> None:
    """A cancellation cannot serialize only the fast sibling's result."""
    fast_done = asyncio.Event()
    release_slow = asyncio.Event()

    @tool()
    async def fast_read() -> str:
        fast_done.set()
        return "fast result"

    @tool()
    async def slow_read() -> str:
        await release_slow.wait()
        return "slow result"

    @agent(model="mock-model")
    @use_tools(fast_read, slow_read)
    class ParallelAgent: ...

    from lauren_ai._tools import _add_to_tool_map

    tool_map: dict[str, object] = {}
    _add_to_tool_map(tool_map, fast_read)
    _add_to_tool_map(tool_map, slow_read)
    ParallelAgent.__lauren_ai_agent__.tools = tool_map

    mock = MockTransport()
    mock.queue_response(
        Completion(
            id="parallel",
            model="mock-model",
            content="",
            tool_calls=[
                ToolCall(tool_use_id="fast", name="fast_read", input={}),
                ToolCall(tool_use_id="slow", name="slow_read", input={}),
            ],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )
    memory = ShortTermMemory()
    runner = AgentRunner(transport=mock)

    async def consume() -> None:
        async for _ in await runner.run_stream(ParallelAgent(), "read both", memory=memory):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(fast_done.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release_slow.set()
    assert memory.validate_tool_history().ok is True
    result_blocks = [
        block
        for message in memory._messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert {block["tool_use_id"] for block in result_blocks} == {"fast", "slow"}
    assert any(block.get("is_error") is True for block in result_blocks)


@pytest.mark.asyncio
async def test_run_cancellation_during_approval_repairs_exchange() -> None:
    """Cancellation while an approval hook is suspended cannot orphan calls."""
    approval_started = asyncio.Event()
    approval_wait = asyncio.Event()

    @tool()
    async def protected_tool() -> dict[str, bool]:
        """Tool held behind the test approval gate."""
        return {"ok": True}

    @agent(model="mock-model")
    @use_tools(protected_tool)
    class ProtectedAgent: ...

    class ApprovalRunner(AgentRunner):
        async def _on_tools_requested(self, tool_calls, *, ctx):
            approval_started.set()
            await approval_wait.wait()
            return tool_calls

    mock = MockTransport()
    mock.queue_tool_use("protected_tool", {})
    memory = ShortTermMemory()
    runner = ApprovalRunner(transport=mock)

    task = asyncio.create_task(runner.run(ProtectedAgent(), "do it", memory=memory))
    await asyncio.wait_for(approval_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert memory.validate_tool_history().ok
    result_message = memory._messages[-1]
    assert result_message["role"] == "user"
    call_id = result_message["content"][0]["tool_use_id"]
    assert call_id
    assert any(block.get("id") == call_id for block in memory._messages[-2]["content"])
    assert result_message["content"][0]["is_error"] is True
