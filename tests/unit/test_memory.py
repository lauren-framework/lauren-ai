"""Unit tests for memory subsystem."""

from __future__ import annotations

import pytest

from lauren_ai._memory import ShortTermMemory
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._memory._vector import InMemoryVectorStore
from lauren_ai._transport import Completion, TokenUsage


class TestShortTermMemory:
    def test_empty_initially(self):
        mem = ShortTermMemory()
        assert mem.messages() == []

    def test_add_user_message(self):
        mem = ShortTermMemory()
        mem.add_user("Hello")
        msgs = mem.messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_add_assistant_completion(self):
        mem = ShortTermMemory()
        mem.add_user("Hello")
        completion = Completion(
            id="c1",
            model="mock",
            content="Hi there!",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=5, output_tokens=3),
        )
        mem.add_assistant(completion)
        msgs = mem.messages()
        assert len(msgs) == 2
        assert msgs[1]["role"] == "assistant"

    def test_token_window_trimming(self):
        # With a very small window, old messages should be dropped
        mem = ShortTermMemory(max_tokens=10)
        for i in range(20):
            mem.add_user(f"Message number {i} with some extra text here to fill tokens")
        msgs = mem.messages()
        # Should be trimmed to fit within 10 tokens (~40 chars)
        assert len(msgs) < 20

    def test_snapshot_and_restore(self):
        mem = ShortTermMemory()
        mem.add_user("First")
        snapshot = mem.snapshot()
        mem.add_user("Second")
        assert len(mem.messages()) == 2
        mem.restore(snapshot)
        assert len(mem.messages()) == 1

    # ── tool-call atomicity (regression for orphaned-tool-message bug) ────────

    def test_trimming_drops_tool_call_and_result_together_openai_format(self):
        """Trimming must never drop an assistant(tool_calls) without its tool
        results — that produces role='tool' after role='user'/'assistant',
        which OpenAI rejects with a 400.  Use a tiny budget so the first
        turn gets trimmed immediately."""
        mem = ShortTermMemory(max_tokens=1)  # force trimming

        # Turn 1: user → assistant(tool_calls) → tool result
        mem.add_user("do something")
        mem._messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tc_001", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}
                ],
                "content": None,
            }
        )
        mem._messages.append(
            {
                "role": "tool",
                "tool_call_id": "tc_001",
                "content": "wrote file",
            }
        )
        mem._messages.append({"role": "assistant", "content": "done"})

        # Turn 2: enough content to push turn 1 out of the window
        mem.add_user("x" * 500)

        msgs = mem.messages()
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]

        # No tool message should appear before an assistant-with-tool-calls
        declared_ids: set[str] = set()
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    declared_ids.add(tc.get("id", ""))
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                assert tid in declared_ids, (
                    f"Orphaned tool message: tool_call_id={tid!r} has no "
                    f"preceding assistant tool_call.  Remaining roles: {roles}"
                )

    def test_trimming_drops_tool_call_and_result_together_anthropic_format(self):
        """Same invariant for Anthropic content-block format."""
        mem = ShortTermMemory(max_tokens=1)

        mem.add_user("do something")
        mem._messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc_002", "name": "read_file", "input": {}},
                ],
            }
        )
        mem._messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tc_002", "content": "file contents"},
                ],
            }
        )
        mem._messages.append({"role": "assistant", "content": "got the file"})
        mem.add_user("x" * 500)

        msgs = mem.messages()
        declared_ids: set[str] = set()
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            declared_ids.add(b.get("id", ""))
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            tid = b.get("tool_use_id", "")
                            assert tid in declared_ids, (
                                f"Orphaned Anthropic tool_result: tool_use_id={tid!r} has no preceding tool_use block."
                            )

    def test_trim_to_fit_also_atomic(self):
        """trim_to_fit() must apply the same atomicity guarantee."""
        mem = ShortTermMemory(max_tokens=100_000)
        mem.add_user("do something")
        mem._messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tc_003", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}}
                ],
                "content": None,
            }
        )
        mem._messages.append(
            {
                "role": "tool",
                "tool_call_id": "tc_003",
                "content": "exit 0",
            }
        )
        mem._messages.append({"role": "assistant", "content": "done"})
        mem.add_user("x" * 500)

        # Trim to a tiny budget — forces the tool-call turn to be dropped
        mem.trim_to_fit(max_tokens=1)

        declared_ids: set[str] = set()
        for msg in mem._messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    declared_ids.add(tc.get("id", ""))
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                assert tid in declared_ids, f"trim_to_fit left orphaned tool message tool_call_id={tid!r}"


class TestInMemoryConversationStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        store = InMemoryConversationStore()
        messages = [{"role": "user", "content": "Hi"}]
        await store.save("conv-1", messages)
        loaded = await store.load("conv-1")
        assert len(loaded["messages"]) == 1
        assert loaded["messages"][0]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_load_missing_returns_empty(self):
        store = InMemoryConversationStore()
        loaded = await store.load("nonexistent")
        assert loaded == []

    @pytest.mark.asyncio
    async def test_save_makes_copy(self):
        store = InMemoryConversationStore()
        messages = [{"role": "user", "content": "Original"}]
        await store.save("conv-1", messages)
        messages.append({"role": "assistant", "content": "Extra"})
        loaded = await store.load("conv-1")
        assert len(loaded["messages"]) == 1  # Copy was made, mutation didn't affect store

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryConversationStore()
        await store.save("conv-1", [{"role": "user", "content": "Hi"}])
        await store.delete("conv-1")
        loaded = await store.load("conv-1")
        assert loaded == []


class TestInMemoryVectorStore:
    @pytest.mark.asyncio
    async def test_upsert_and_search(self):
        store = InMemoryVectorStore()
        await store.upsert("The weather in Paris is sunny today.")
        await store.upsert("Python programming language features.")
        results = await store.search("Paris weather", k=2)
        assert len(results) >= 1
        # The Paris weather document should score higher
        assert "Paris" in results[0].content

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        store = InMemoryVectorStore()
        results = await store.search("anything", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Hello world")
        await store.delete([doc_id])  # delete takes a list
        results = await store.search("hello", k=5)
        assert not any(r.id == doc_id for r in results)

    @pytest.mark.asyncio
    async def test_get(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Test content", metadata={"tag": "test"})
        result = await store.get(doc_id)
        assert result is not None
        assert result.content == "Test content"

    @pytest.mark.asyncio
    async def test_clear(self):
        store = InMemoryVectorStore()
        doc_id = await store.upsert("Hello")
        await store.clear()
        result = await store.get(doc_id)
        assert result is None
