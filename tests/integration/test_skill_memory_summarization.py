"""Integration tests for context-window summarisation on overflow (Feature 16).

When AgentConfig.summarize_at is set and the memory token estimate crosses that
fraction of memory_window_tokens, the runner:
 1. Calls the LLM (optionally a cheaper summary_model) to compress older turns.
 2. Trims the internal buffer to keep_recent turns.
 3. Stores the summary text in ShortTermMemory.summary.
 4. Injects the summary into the system prompt for all subsequent turns.
 5. Saves the summary in the conversation snapshot so resumed sessions carry
    it forward.

summarize_at=None (default) → zero behaviour change (silent drop as before).
"""

from __future__ import annotations

import pytest

from lauren_ai import AgentConfig, agent
from lauren_ai._agents._runner import AgentRunnerBase, _should_summarize, _summarize_memory
from lauren_ai._memory import ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compl(content: str, *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=8),
    )


@agent(model="mock-model", system="You are a helpful assistant.")
class SimpleAgent:
    pass


def _make_runner(mock: MockTransport) -> AgentRunnerBase:
    return AgentRunnerBase(transport=mock)


# ---------------------------------------------------------------------------
# Unit-level: _should_summarize helper
# ---------------------------------------------------------------------------


class TestShouldSummarize:
    def test_returns_false_when_summarize_at_is_none(self):
        mem = ShortTermMemory(max_tokens=100)
        mem.add_user("hello " * 100)
        cfg = AgentConfig(memory_window_tokens=100, summarize_at=None)
        assert _should_summarize(mem, cfg) is False

    def test_returns_false_below_threshold(self):
        mem = ShortTermMemory(max_tokens=1000)
        mem.add_user("hi")  # very small
        cfg = AgentConfig(memory_window_tokens=1000, summarize_at=0.8)
        assert _should_summarize(mem, cfg) is False

    def test_returns_true_at_threshold(self):
        mem = ShortTermMemory(max_tokens=100)
        # Fill to exactly 80% of 100 tokens = 80 tokens = 320 chars
        mem.add_user("x" * 320)
        cfg = AgentConfig(memory_window_tokens=100, summarize_at=0.8)
        assert _should_summarize(mem, cfg) is True

    def test_returns_true_above_threshold(self):
        mem = ShortTermMemory(max_tokens=100)
        mem.add_user("x" * 500)  # >> 80% of 100 tokens
        cfg = AgentConfig(memory_window_tokens=100, summarize_at=0.8)
        assert _should_summarize(mem, cfg) is True


# ---------------------------------------------------------------------------
# Unit-level: _summarize_memory helper
# ---------------------------------------------------------------------------


class TestSummarizeMemory:
    @pytest.mark.asyncio
    async def test_compresses_old_turns_and_stores_summary(self):
        mock = MockTransport()
        mock.queue_response(_compl("Summary: Alice asked about weather."))

        mem = ShortTermMemory(max_tokens=1000)
        for i in range(8):
            mem.add_user(f"User turn {i}")
            mem.add_assistant(_compl(f"Assistant turn {i}"))

        initial_count = len(mem)
        assert initial_count == 16

        await _summarize_memory(mem, mock, model="mock-model", keep_recent=6)

        # Summary stored
        assert mem.summary is not None
        assert "Alice" in mem.summary or "Summary" in mem.summary

        # Old turns trimmed — only keep_recent=6 non-system messages remain
        assert len(mem) == 6

    @pytest.mark.asyncio
    async def test_keeps_exactly_keep_recent_messages(self):
        mock = MockTransport()
        mock.queue_response(_compl("Summary text."))

        mem = ShortTermMemory()
        for i in range(10):
            mem.add_user(f"msg {i}")
            mem.add_assistant(_compl(f"reply {i}"))

        await _summarize_memory(mem, mock, model="mock-model", keep_recent=4)

        assert len(mem) == 4
        # Most-recent messages are preserved
        assert mem._messages[-1]["content"] == "reply 9"
        assert mem._messages[-2]["content"] == "msg 9"

    @pytest.mark.asyncio
    async def test_no_op_when_too_few_messages(self):
        """If all messages fit in keep_recent, nothing is compressed."""
        mock = MockTransport()

        mem = ShortTermMemory()
        mem.add_user("only message")

        await _summarize_memory(mem, mock, model="mock-model", keep_recent=6)

        # No summary call was made and nothing changed
        assert mem.summary is None
        assert len(mem) == 1
        assert len(mock._calls) == 0

    @pytest.mark.asyncio
    async def test_transport_failure_is_swallowed(self):
        """A summarisation error must never break the agent run."""

        class FailingTransport:
            async def complete(self, *args, **kwargs):
                raise RuntimeError("network error")

        mem = ShortTermMemory()
        for i in range(10):
            mem.add_user(f"msg {i}")
            mem.add_assistant(_compl(f"reply {i}"))

        initial = len(mem)
        # Should not raise
        await _summarize_memory(mem, FailingTransport(), model="any")

        # Memory unchanged
        assert mem.summary is None
        assert len(mem) == initial


# ---------------------------------------------------------------------------
# Integration: summarise_at triggers in the agent run loop
# ---------------------------------------------------------------------------


class TestSummarizationInRunLoop:
    @pytest.mark.asyncio
    async def test_summary_is_injected_into_system_prompt(self):
        """After summarisation, the system prompt sent to transport contains the summary."""
        mock = MockTransport()
        # The first complete() call is the summarisation call
        mock.queue_response(_compl("Summary: user asked many questions."))
        # The second is the actual agent response
        mock.queue_response(_compl("I remember everything!"))

        @agent(
            model="mock-model",
            system="You are helpful.",
            memory_window_tokens=50,  # tiny window
            summarize_at=0.5,  # trigger at 50%
        )
        class TinyAgent:
            pass

        TinyAgent.__lauren_ai_agent__.tools = {}
        runner = AgentRunnerBase(transport=mock)

        # Pre-fill memory past the threshold (50% of 50 = 25 tokens = 100 chars)
        mem = ShortTermMemory(max_tokens=50)
        for _ in range(4):
            mem.add_user("x" * 30)  # ~7 tokens each
            mem.add_assistant(_compl("ok"))
        TinyAgent.__lauren_ai_agent__.model = "mock-model"

        response = await runner.run(
            TinyAgent(),
            "Final question",
            memory=mem,
        )

        assert response.content == "I remember everything!"
        # Summarisation call was made first (system="You are a conversation summariser.")
        assert len(mock._calls) >= 2
        summary_call = mock._calls[0]
        assert "summariser" in (summary_call.system or "")

    @pytest.mark.asyncio
    async def test_summary_injected_in_subsequent_turns(self):
        """Transport receives [Earlier conversation summary] in system for turns after compress."""
        mock = MockTransport()
        mock.queue_response(_compl("Summary: early turns compressed."))
        mock.queue_response(_compl("Using summary!"))

        @agent(
            model="mock-model",
            system="Agent system.",
            memory_window_tokens=50,
            summarize_at=0.5,
        )
        class CheckAgent:
            pass

        CheckAgent.__lauren_ai_agent__.tools = {}
        CheckAgent.__lauren_ai_agent__.model = "mock-model"
        runner = AgentRunnerBase(transport=mock)

        mem = ShortTermMemory(max_tokens=50)
        for _ in range(4):
            mem.add_user("word " * 20)  # forces threshold
            mem.add_assistant(_compl("ok"))

        await runner.run(CheckAgent(), "Next question", memory=mem)

        # The agent turn (second call) must have the summary in system
        agent_call = mock._calls[1]
        system_sent = agent_call.system or ""
        assert "[Earlier conversation summary]" in system_sent
        assert "Summary: early turns compressed." in system_sent

    @pytest.mark.asyncio
    async def test_summarize_at_none_disables_summarization(self):
        """Default (summarize_at=None) → silent drop, no summarisation call."""
        mock = MockTransport()
        mock.queue_response(_compl("reply"))

        @agent(model="mock-model", memory_window_tokens=20)
        class NoSummaryAgent:
            pass

        NoSummaryAgent.__lauren_ai_agent__.tools = {}
        NoSummaryAgent.__lauren_ai_agent__.model = "mock-model"
        runner = AgentRunnerBase(transport=mock)

        # Overfill memory way past the window
        mem = ShortTermMemory(max_tokens=20)
        for _ in range(10):
            mem.add_user("x" * 100)
            mem.add_assistant(_compl("ok"))

        await runner.run(NoSummaryAgent(), "hello", memory=mem)

        # Only one transport call (the actual agent turn — no summarisation call)
        assert len(mock._calls) == 1
        assert mem.summary is None

    @pytest.mark.asyncio
    async def test_cheaper_summary_model_used(self):
        """When summary_model is set, the summarisation call uses that model."""
        mock = MockTransport()
        mock.queue_response(_compl("Cheap summary."))
        mock.queue_response(_compl("Expensive reply."))

        @agent(
            model="expensive-model",
            memory_window_tokens=50,
            summarize_at=0.5,
            summary_model="cheap-haiku",
        )
        class DualModelAgent:
            pass

        DualModelAgent.__lauren_ai_agent__.tools = {}
        DualModelAgent.__lauren_ai_agent__.model = "expensive-model"
        runner = AgentRunnerBase(transport=mock)

        mem = ShortTermMemory(max_tokens=50)
        for _ in range(4):
            mem.add_user("x" * 30)
            mem.add_assistant(_compl("ok"))

        await runner.run(DualModelAgent(), "hi", memory=mem)

        # First call (summarisation) uses summary_model
        summary_call = mock._calls[0]
        assert summary_call.model == "cheap-haiku"

        # Second call (agent turn) uses the main model
        agent_call = mock._calls[1]
        assert agent_call.model == "expensive-model"


# ---------------------------------------------------------------------------
# Persistence: snapshot / restore carries summary forward
# ---------------------------------------------------------------------------


class TestSummarizationPersistence:
    def test_snapshot_includes_summary(self):
        mem = ShortTermMemory()
        mem.add_user("hello")
        mem.set_summary("Earlier: user greeted agent.")

        snap = mem.snapshot()
        assert snap["summary"] == "Earlier: user greeted agent."
        assert len(snap["messages"]) == 1

    def test_restore_from_new_format_loads_summary(self):
        snap = {
            "messages": [{"role": "user", "content": "hello"}],
            "summary": "Earlier: user greeted agent.",
        }
        mem = ShortTermMemory()
        mem.restore(snap)
        assert mem.summary == "Earlier: user greeted agent."
        assert len(mem) == 1

    def test_restore_from_legacy_list_format(self):
        """Old snapshots (plain list) restore cleanly with no summary."""
        legacy = [{"role": "user", "content": "hello"}]
        mem = ShortTermMemory()
        mem.restore(legacy)
        assert mem.summary is None
        assert len(mem) == 1

    def test_summary_preserved_across_conversation_store_roundtrip(self):
        """Summary survives save → load via InMemoryConversationStore."""
        import asyncio

        async def run():
            store = InMemoryConversationStore()
            mem = ShortTermMemory()
            mem.add_user("turn 1")
            mem.set_summary("Old turns: user said hello.")

            await store.save("conv1", mem.snapshot())

            mem2 = ShortTermMemory()
            prior = await store.load("conv1")
            mem2.restore(prior)
            return mem2

        mem_restored = asyncio.run(run())
        assert mem_restored.summary == "Old turns: user said hello."
        assert len(mem_restored) == 1

    @pytest.mark.asyncio
    async def test_resumed_session_injects_summary_into_system_prompt(self):
        """A resumed session that has a saved summary injects it immediately."""
        store = InMemoryConversationStore()

        # Save a snapshot with a summary
        snap = {
            "messages": [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
            "summary": "Saved summary: user asked about Python.",
        }
        await store.save("sess1", snap)

        mock = MockTransport()
        mock.queue_response(_compl("Based on our earlier chat..."))

        @agent(
            model="mock-model",
            memory_window_tokens=10000,  # no trigger this turn
        )
        class ResumingAgent:
            pass

        ResumingAgent.__lauren_ai_agent__.tools = {}
        ResumingAgent.__lauren_ai_agent__.model = "mock-model"
        runner = AgentRunnerBase(transport=mock)

        await runner.run(
            ResumingAgent(),
            "New question",
            conversation_id="sess1",
            conversation_store=store,
        )

        # The transport call must include the summary in the system prompt
        call = mock._calls[0]
        system = call.system or ""
        assert "[Earlier conversation summary]" in system
        assert "Saved summary: user asked about Python." in system
