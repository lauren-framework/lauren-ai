"""Tests for incremental conversation prompt caching (PRD-132 L0)."""

from __future__ import annotations

import dataclasses

import pytest

from lauren_ai._config import LLMConfig
from lauren_ai._transport import Message
from lauren_ai._transport._anthropic import AnthropicTransport, _apply_conversation_cache


class TestApplyConversationCache:
    def test_marks_last_block_of_last_message(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        ]
        _apply_conversation_cache(msgs)
        assert msgs[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # earlier message is not marked
        assert "cache_control" not in msgs[0]["content"][-1]

    def test_normalises_string_content_to_text_block(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        _apply_conversation_cache(msgs)
        assert msgs[0]["content"] == [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]

    def test_marks_tool_result_block(self) -> None:
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "X"}]},
        ]
        _apply_conversation_cache(msgs)
        assert msgs[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_messages_is_noop(self) -> None:
        msgs: list[dict] = []
        _apply_conversation_cache(msgs)  # must not raise
        assert msgs == []

    def test_only_one_breakpoint_added(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        ]
        _apply_conversation_cache(msgs)
        marked = [b for b in msgs[0]["content"] if "cache_control" in b]
        assert len(marked) == 1
        assert marked[0]["text"] == "b"  # the LAST block


class TestConfigFlag:
    def test_llmconfig_has_cache_conversation_default_false(self) -> None:
        cfg = LLMConfig.for_anthropic(model="claude-x", api_key="k")
        assert cfg.cache_conversation is False

    @pytest.mark.asyncio
    async def test_transport_applies_cache_when_enabled(self) -> None:
        """End-to-end: with cache_conversation on, the request's last message is marked."""
        cfg = dataclasses.replace(
            LLMConfig.for_anthropic(model="claude-x", api_key="k"),
            cache_conversation=True,
        )
        transport = AnthropicTransport(cfg)
        captured = _capture_into(transport)

        with pytest.raises(RuntimeError):
            await transport.complete([Message.user("hello there")], model="claude-x", max_tokens=16, stream=False)

        last = captured["messages"][-1]
        assert isinstance(last["content"], list)
        assert last["content"][-1]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_transport_no_cache_when_disabled(self) -> None:
        cfg = LLMConfig.for_anthropic(model="claude-x", api_key="k")  # cache_conversation=False
        transport = AnthropicTransport(cfg)
        captured = _capture_into(transport)

        with pytest.raises(RuntimeError):
            await transport.complete([Message.user("hi")], model="claude-x", max_tokens=16, stream=False)

        last = captured["messages"][-1]
        # string content untouched (no marker)
        assert last["content"] == "hi"


def _capture_into(transport: AnthropicTransport) -> dict:
    """Inject a fake client that captures the request kwargs then aborts."""
    captured: dict = {}

    class _FakeMessages:
        async def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            raise RuntimeError("stop-after-capture")

    class _FakeClient:
        messages = _FakeMessages()

    transport._client = _FakeClient()  # type: ignore[attr-defined]
    return captured
