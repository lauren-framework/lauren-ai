"""Unit tests for the AgentRunner agentic loop."""

from __future__ import annotations

import pytest

from lauren_ai._agents import AgentResponse, agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


def _make_tool_map(*tool_funcs) -> dict:
    tools = {}
    for t in tool_funcs:
        m = getattr(t, TOOL_META)
        tools[m.name] = (t, m)
    return tools


def make_runner(mock: MockTransport, max_turns: int = 10) -> AgentRunner:
    """Build a runner + simple agent class for testing."""
    runner = AgentRunner(transport=mock)
    return runner


@pytest.fixture()
def mock() -> MockTransport:
    return MockTransport()


class TestAgentRunnerBasic:
    @pytest.mark.asyncio
    async def test_single_turn_response(self, mock):
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="Hello! How can I help?",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=8),
            )
        )

        @agent(model="mock-model")
        class SimpleAgent:
            pass

        runner = make_runner(mock)
        instance = SimpleAgent()
        response = await runner.run(instance, "Hi!")
        assert isinstance(response, AgentResponse)
        assert response.content == "Hello! How can I help?"
        assert response.turns == 1
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_tool_call_and_result(self, mock):
        @tool()
        async def get_time() -> str:
            """Get the current time."""
            return "12:00 PM UTC"

        tools = _make_tool_map(get_time)
        runner = AgentRunner(transport=mock)

        # Turn 1: model calls the tool
        mock.queue_tool_use("get_time", {})
        # Turn 2: model responds with final answer
        mock.queue_response(
            Completion(
                id="c2",
                model="mock",
                content="The time is 12:00 PM UTC.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=20, output_tokens=10),
            )
        )

        @agent(model="mock-model")
        @use_tools(get_time)
        class TimeAgent:
            pass

        TimeAgent.__lauren_ai_agent__.tools = tools

        instance = TimeAgent()
        response = await runner.run(instance, "What time is it?")
        assert "12:00" in response.content
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0].name == "get_time"

    @pytest.mark.asyncio
    async def test_max_turns_stop_reason(self, mock):
        # Queue completions that always call tools — loop exhausts turns
        for _ in range(5):
            mock.queue_tool_use("fake_tool", {})

        @agent(model="mock-model", max_turns=2)
        class StuckAgent:
            pass

        runner = AgentRunner(transport=mock)

        instance = StuckAgent()
        response = await runner.run(instance, "Do something")
        # The runner sets stop_reason to "max_turns" when the loop exhausts
        assert response.stop_reason == "max_turns"

    @pytest.mark.asyncio
    async def test_cumulative_token_usage(self, mock):
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="Result",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        )

        @agent(model="mock-model")
        class TokenAgent:
            pass

        runner = make_runner(mock)
        instance = TokenAgent()
        response = await runner.run(instance, "Hello")
        assert response.total_usage.input_tokens == 100
        assert response.total_usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_called(self, mock):
        mock.queue_response(
            Completion(
                id="c1",
                model="mock",
                content="Done",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            )
        )

        on_start_called = []
        on_finish_called = []

        @agent(model="mock-model")
        class HookAgent:
            async def on_start(self, ctx):
                on_start_called.append(True)

            async def on_finish(self, response, ctx):
                on_finish_called.append(True)

        runner = make_runner(mock)
        instance = HookAgent()
        await runner.run(instance, "Hi")

        assert len(on_start_called) == 1
        assert len(on_finish_called) == 1

    @pytest.mark.asyncio
    async def test_tool_error_policy_return_error(self, mock):
        @tool()
        async def failing_tool() -> str:
            """A tool that always fails."""
            raise RuntimeError("Tool exploded!")

        tools = _make_tool_map(failing_tool)
        runner = AgentRunner(transport=mock)

        mock.queue_tool_use("failing_tool", {})
        mock.queue_response(
            Completion(
                id="c2",
                model="mock",
                content="I see the tool encountered an error.",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        )

        @agent(model="mock-model")
        @use_tools(failing_tool)
        class ErrorAgent:
            pass

        ErrorAgent.__lauren_ai_agent__.tools = tools

        instance = ErrorAgent()
        # Should NOT raise — policy is "return_error" by default
        response = await runner.run(instance, "Use the tool")
        assert response.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Conversation memory persistence
# ---------------------------------------------------------------------------

from lauren_ai._memory._stores import InMemoryConversationStore  # noqa: E402


def _compl(content: str, *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def make_runner_with_store(mock: MockTransport, store: InMemoryConversationStore) -> AgentRunner:
    """Configure ``_MemAgent``'s AgentMeta with *store* and return a runner.

    Stores live on AgentMeta now (set by ``@agent(conversation_store=…)`` or
    written here for testing).  The runner consults AgentMeta on every call.
    """
    _MemAgent.__lauren_ai_agent__.conversation_store = store
    return AgentRunner(transport=mock)


@agent(model="mock-model")
class _MemAgent:
    """Minimal agent for conversation-memory tests."""


class TestAgentRunnerConversationMemory:
    """AgentRunner loads and saves conversation history via ConversationStore."""

    @pytest.mark.asyncio
    async def test_first_run_saves_history(self, mock):
        """After the first run the exchange is persisted in the store."""
        store = InMemoryConversationStore()
        runner = make_runner_with_store(mock, store)
        mock.queue_response(_compl("Hello there!", n=1))

        await runner.run(_MemAgent(), "Hi", conversation_id="s1")

        saved = await store.load("s1")
        assert len(saved) == 2
        assert saved[0] == {"role": "user", "content": "Hi"}
        assert saved[1]["role"] == "assistant"
        assert saved[1]["content"] == "Hello there!"

    @pytest.mark.asyncio
    async def test_second_run_sees_prior_messages(self, mock):
        """The transport receives the full prior exchange on the second call."""
        store = InMemoryConversationStore()
        runner = make_runner_with_store(mock, store)

        # Turn 1
        mock.queue_response(_compl("I'm fine, thanks.", n=1))
        await runner.run(_MemAgent(), "How are you?", conversation_id="s2")

        # Turn 2 — capture what the transport receives
        sent: list = []
        original = mock.complete

        async def spy(messages, **kw):
            sent.extend(messages)
            return await original(messages, **kw)

        mock.complete = spy
        mock.queue_response(_compl("You asked how I was.", n=2))
        await runner.run(_MemAgent(), "What did I ask?", conversation_id="s2")

        # Messages seen by the LLM: [prior user, prior assistant, new user]
        assert len(sent) == 3
        assert sent[0]["content"] == "How are you?"
        assert sent[1]["content"] == "I'm fine, thanks."
        assert sent[2]["content"] == "What did I ask?"

    @pytest.mark.asyncio
    async def test_no_conversation_id_does_not_write_store(self, mock):
        """Without a conversation_id the store is never touched."""
        store = InMemoryConversationStore()
        runner = make_runner_with_store(mock, store)
        mock.queue_response(_compl("OK", n=1))

        await runner.run(_MemAgent(), "Hello")  # no conversation_id

        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_no_store_runs_normally(self, mock):
        """Runner without a store behaves exactly as before — no regression."""
        # Reset class-level meta in case a sibling test set it.
        _MemAgent.__lauren_ai_agent__.conversation_store = None
        runner = make_runner(mock)  # no conversation_store
        mock.queue_response(_compl("Fine", n=1))

        resp = await runner.run(_MemAgent(), "Hey", conversation_id="irrelevant")

        assert resp.content == "Fine"

    @pytest.mark.asyncio
    async def test_different_conversation_ids_are_isolated(self, mock):
        """Separate conversation IDs never share message history."""
        store = InMemoryConversationStore()
        runner = make_runner_with_store(mock, store)
        inst = _MemAgent()

        mock.queue_response(_compl("Alice answer", n=1))
        await runner.run(inst, "Alice question", conversation_id="alice")

        mock.queue_response(_compl("Bob answer", n=2))
        await runner.run(inst, "Bob question", conversation_id="bob")

        alice = await store.load("alice")
        bob = await store.load("bob")

        assert all("Bob" not in str(m) for m in alice)
        assert all("Alice" not in str(m) for m in bob)

    @pytest.mark.asyncio
    async def test_history_accumulates_across_runs(self, mock):
        """Each run appends one user + one assistant entry to the stored history."""
        store = InMemoryConversationStore()
        runner = make_runner_with_store(mock, store)
        inst = _MemAgent()
        conv_id = "multi"

        for i in range(4):
            mock.queue_response(_compl(f"Reply {i}", n=i))
            await runner.run(inst, f"Msg {i}", conversation_id=conv_id)

        history = await store.load(conv_id)
        assert len(history) == 8  # 4 user + 4 assistant
        user_contents = [m["content"] for m in history if m["role"] == "user"]
        assert user_contents == ["Msg 0", "Msg 1", "Msg 2", "Msg 3"]


# ---------------------------------------------------------------------------
# run_stream() parity — budget enforcement, on_finish hook, history persistence
# ---------------------------------------------------------------------------

from lauren_ai._transport import CompletionChunk  # noqa: E402


def _stream_chunks(*parts: str, stop_reason: str = "end_turn") -> list[CompletionChunk]:
    chunks = [CompletionChunk(delta=p) for p in parts]
    chunks.append(
        CompletionChunk(
            delta="",
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=10, output_tokens=len(parts) or 1),
        )
    )
    return chunks


class TestRunStreamParity:
    """run_stream() honours budgets, hooks, and conversation persistence."""

    @pytest.mark.asyncio
    async def test_run_stream_budget_exceeded_swallows_and_yields(self, mock):
        """Budget breach swallows the exception and ends the stream cleanly."""
        runner = AgentRunner(transport=mock)

        @agent(model="mock-model", max_cost_usd=0.0)
        class TightBudgetAgent: ...

        mock.queue_stream(_stream_chunks("Hello"))

        # Iterating must NOT raise — the budget exception is swallowed and
        # the generator ends naturally (matches run() semantics).
        chunks = []
        async for chunk in await runner.run_stream(TightBudgetAgent(), "Hi"):
            chunks.append(chunk)

        # All chunks were yielded before the budget check fired
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_run_stream_calls_on_finish_with_response(self, mock):
        """on_finish receives an AgentResponse with the streamed content."""
        captured: list[AgentResponse] = []

        @agent(model="mock-model")
        class CaptureAgent:
            async def on_finish(self, response: AgentResponse, ctx) -> None:
                captured.append(response)

        runner = AgentRunner(transport=mock)

        mock.queue_stream(_stream_chunks("Hello", " world", "!"))

        async for _ in await runner.run_stream(CaptureAgent(), "Hi"):
            pass

        assert len(captured) == 1
        assert captured[0].content == "Hello world!"
        assert captured[0].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_run_stream_persists_conversation_history(self, mock):
        """run_stream loads prior history and saves the new turn back."""
        store = InMemoryConversationStore()
        _MemAgent.__lauren_ai_agent__.conversation_store = store
        runner = AgentRunner(transport=mock)

        mock.queue_stream(_stream_chunks("First reply"))
        async for _ in await runner.run_stream(_MemAgent(), "Hi", conversation_id="sess"):
            pass

        saved = await store.load("sess")
        assert len(saved) == 2
        assert saved[0] == {"role": "user", "content": "Hi"}
        assert saved[1]["role"] == "assistant"
        assert saved[1]["content"] == "First reply"

        # Second run should see the prior exchange
        captured_messages: list = []

        async def capture_complete(messages, **kwargs):
            captured_messages.extend(messages)
            return _stream_chunks("Second reply").__iter__()

        mock.queue_stream(_stream_chunks("Second reply"))
        async for _ in await runner.run_stream(_MemAgent(), "Follow up", conversation_id="sess"):
            pass

        history = await store.load("sess")
        assert len(history) == 4  # 2 prior + 2 new
        assert history[2] == {"role": "user", "content": "Follow up"}
