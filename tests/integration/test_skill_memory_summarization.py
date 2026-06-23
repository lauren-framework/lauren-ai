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
from lauren_ai._agents._runner import (
    AgentRunnerBase,
    _maybe_compact,
    _summarize_memory,
    _summarize_text,
)
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
# Unit-level: _maybe_compact (exact-count trigger, PRD-135 A)
# ---------------------------------------------------------------------------


class TestMaybeCompact:
    @pytest.mark.asyncio
    async def test_disabled_when_summarize_at_none(self):
        mock = MockTransport()
        mem = ShortTermMemory(max_tokens=100)
        mem.add_user("x" * 100_000)
        cfg = AgentConfig(summarize_at=None)
        assert await _maybe_compact(mem, mock, model="m", system=None, tools=None, config=cfg) is False
        assert len(mock._calls) == 0

    @pytest.mark.asyncio
    async def test_no_compaction_below_trigger(self):
        # Cheap gate short-circuits — not even a count_tokens round-trip.
        mock = MockTransport()
        mem = ShortTermMemory(max_tokens=2000)  # trigger@0.5 = 1000 tok
        mem.add_user("hi")
        cfg = AgentConfig(summarize_at=0.5)
        assert await _maybe_compact(mem, mock, model="m", system=None, tools=None, config=cfg) is False
        assert len(mock._calls) == 0

    @pytest.mark.asyncio
    async def test_compacts_when_buffer_exceeds_live_window(self):
        mock = MockTransport()
        mock.queue_response(_compl("ROLLING SUMMARY of older turns"))
        mem = ShortTermMemory(max_tokens=2000)  # trigger@0.5 = 1000 tok ≈ 4000 chars
        for _ in range(8):
            mem.add_user("x" * 1000)
            mem.add_assistant(_compl("ok"))  # ~8000 chars > trigger
        cfg = AgentConfig(summarize_at=0.5, context_window=200_000)
        did = await _maybe_compact(mem, mock, model="m", system="sys", tools=None, config=cfg)
        assert did is True
        assert mem.summary == "ROLLING SUMMARY of older turns"
        assert len(mem._messages) <= 6  # trimmed to keep_recent

    @pytest.mark.asyncio
    async def test_triggers_on_full_buffer_not_trimmed_view(self):
        # messages() would trim to the (small) live window; the trigger must see
        # the FULL buffer so a conversation that has outgrown the window compacts.
        mock = MockTransport()
        mock.queue_response(_compl("summary"))
        mem = ShortTermMemory(max_tokens=1500)  # messages() would cap ~1500 tok
        for _ in range(10):
            mem.add_user("y" * 1000)
            mem.add_assistant(_compl("ok"))
        cfg = AgentConfig(summarize_at=0.5, context_window=200_000)
        assert await _maybe_compact(mem, mock, model="m", system="s", tools=None, config=cfg) is True


class TestMapReduceSummary:
    @pytest.mark.asyncio
    async def test_chunks_so_no_call_exceeds_budget(self):
        # A transcript several times the chunk budget must summarise via multiple
        # bounded calls (map-reduce), never one over-budget call.
        mock = MockTransport()
        for _ in range(50):
            mock.queue_response(_compl("partial summary"))
        text = "A" * 60_000
        out = await _summarize_text(mock, text, model="m", max_input_chars=10_000)
        assert out
        assert len(mock._calls) > 1  # mapped into chunks
        biggest = max(len(c.messages[0].content) for c in mock._calls)
        assert biggest < 11_000  # every call's input stayed near the chunk budget

    @pytest.mark.asyncio
    async def test_single_call_when_within_budget(self):
        mock = MockTransport()
        mock.queue_response(_compl("one summary"))
        out = await _summarize_text(mock, "short text", model="m", max_input_chars=100_000)
        assert out == "one summary"
        assert len(mock._calls) == 1


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


# ---------------------------------------------------------------------------
# Boundary safety: the keep_recent split must never orphan a tool_result
# ---------------------------------------------------------------------------


class TestKeepRecentBoundary:
    def _mem_with_tool_pair_at_boundary(self) -> ShortTermMemory:
        mem = ShortTermMemory(max_tokens=10_000)
        # 4 plain turns, then a tool_use → tool_result pair as the most recent 2.
        for i in range(4):
            mem.add_user(f"q{i}")
            mem.add_assistant(_compl(f"a{i}"))
        mem._messages.append(
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]}
        )
        mem._messages.append(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
        )
        return mem

    def test_trim_to_recent_does_not_orphan_tool_result(self):
        mem = self._mem_with_tool_pair_at_boundary()
        # keep_recent=1 would naively keep only the tool_result (orphan).
        mem.trim_to_recent(keep_recent=1)
        first_kept = mem._messages[0]
        # The boundary snapped back to include the tool_use (or a conversational
        # anchor) — the kept window never *opens* on a bare tool_result.
        content = first_kept.get("content")
        is_tool_result = isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
        assert not is_tool_result

    def test_messages_to_summarize_matches_trim_boundary(self):
        mem = self._mem_with_tool_pair_at_boundary()
        to_compress = mem.messages_to_summarize(keep_recent=1)
        # The tool_use must be summarised together with (or after) — never split
        # from — its tool_result.  Since the boundary snaps to keep the pair, the
        # compressed slice ends before the tool_use.
        assert all(
            not (
                isinstance(m.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in m["content"])
            )
            for m in to_compress
        ) or all(
            not (
                isinstance(m.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
            )
            for m in to_compress
        )


# ---------------------------------------------------------------------------
# run_stream surfaces a compaction status notice (PRD-135 D)
# ---------------------------------------------------------------------------


class TestRunStreamCompactionNotice:
    @pytest.mark.asyncio
    async def test_emits_system_notice_chunk_on_compaction(self):
        mock = MockTransport()
        mock.queue_response(_compl("ROLLING SUMMARY"))  # consumed by summariser
        mock.queue_response(_compl("done"))  # the streamed agent turn

        @agent(model="mock-model", system="S")
        class StreamAgent:
            pass

        StreamAgent.__lauren_ai_agent__.tools = {}
        StreamAgent.__lauren_ai_agent__.model = "mock-model"
        runner = AgentRunnerBase(transport=mock)

        mem = ShortTermMemory(max_tokens=2000)  # trigger@0.5 ≈ 1000 tok
        for _ in range(8):
            mem.add_user("x" * 1000)
            mem.add_assistant(_compl("ok"))

        notices: list[str] = []
        stream = await runner.run_stream(
            StreamAgent(),
            "final question",
            memory=mem,
            config_override=AgentConfig(summarize_at=0.5, context_window=200_000, max_turns=1),
        )
        async for chunk in stream:
            if chunk.system_notice:
                notices.append(chunk.system_notice)

        assert any("Compacting" in n for n in notices)
        assert mem.summary == "ROLLING SUMMARY"
