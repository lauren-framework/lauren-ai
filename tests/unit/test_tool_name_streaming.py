"""Tests for streaming tool-call name handling in AgentRunner and transports.

Covers three fixes:

1. **Runner** (``_runner.py``) — entries with an empty tool name are skipped
   when building ``accumulated_tool_calls`` so ``ToolExecutor`` never sees
   ``name=''``.
2. **OpenAI transport** (``_openai.py``) — accumulated name is propagated on
   every delta once it is known; blank header deltas (no name, no args) are
   dropped; when ``finish_reason="tool_calls"`` arrives with any empty name a
   non-streaming fallback fetches the real tool names.
3. **Anthropic transport** (``_anthropic.py``) — always emits the tool name on
   ``content_block_start``; confirms the runner receives it correctly.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._tools import TOOL_META, tool
from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    TokenUsage,
    ToolCallDelta,
)
from lauren_ai._transport._mock import MockTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_completion(text: str) -> Completion:
    return Completion(
        id="c2",
        model="test",
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_agent_with_tool(t):
    """Create a minimal @agent class whose meta.tools map contains *t*."""
    tool_meta = getattr(t, TOOL_META)

    @agent(model="test")
    @use_tools(t)
    class _Agent:
        pass

    _Agent.__lauren_ai_agent__.tools = {tool_meta.name: (t, tool_meta)}
    return _Agent


# ---------------------------------------------------------------------------
# Tests: Runner skips empty-name tool calls in _stream_loop
# ---------------------------------------------------------------------------


class TestRunnerEmptyNameToolCalls:
    """AgentRunner._stream_loop must silently skip ToolCalls with name==''."""

    @pytest.mark.asyncio
    async def test_normal_stream_name_in_first_delta_executes(self):
        """Standard pattern: name on first delta, args in subsequent chunks."""
        dispatch_log: list[str] = []

        @tool()
        async def greet(name: str) -> str:
            """Greet someone."""
            dispatch_log.append(name)
            return f"Hello, {name}!"

        Agent = _make_agent_with_tool(greet)
        tool_name = getattr(greet, TOOL_META).name

        mock = MockTransport()
        mock.queue_stream(
            [
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="call_001",
                        name=tool_name,  # name present in first delta
                        input_delta="",
                    )
                ),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="call_001",
                        name=None,  # subsequent deltas have no name
                        input_delta='{"name": "Alice"}',
                    )
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        mock.queue_stream([CompletionChunk(delta="Hello, Alice!"), CompletionChunk(stop_reason="end_turn")])

        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(Agent(), "Greet Alice"):
            pass

        assert dispatch_log == ["Alice"]

    @pytest.mark.asyncio
    async def test_all_deltas_empty_name_tool_not_dispatched(self):
        """All streaming deltas have name=None → tool is never called.

        When all tool calls are skipped (empty names), the loop breaks without
        making a second LLM call — the model's streamed text is the response.
        """
        dispatch_log: list[str] = []

        @tool()
        async def do_thing(x: str) -> str:
            """Do a thing."""
            dispatch_log.append(x)
            return "done"

        Agent = _make_agent_with_tool(do_thing)
        mock = MockTransport()

        # Poolside streaming bug: model streams a text prefix, then emits a
        # tool-use stop_reason with no tool name. The runner skips the empty
        # call and the stream's already-yielded text becomes the final response.
        mock.queue_stream(
            [
                CompletionChunk(delta="I'll try to help."),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="call_empty",
                        name=None,
                        input_delta='{"x": "test"}',
                    )
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        # No second stream queued — the runner breaks after skipping the tool.

        runner = AgentRunnerBase(transport=mock)
        text_chunks: list[str] = []
        async for chunk in await runner.run_stream(Agent(), "do something"):
            if chunk.delta:
                text_chunks.append(chunk.delta)

        # Tool must not have been called
        assert dispatch_log == []
        # The text streamed before the (skipped) tool call is still yielded
        assert "I'll try to help." in "".join(text_chunks)

    @pytest.mark.asyncio
    async def test_fallback_fb_id_chunk_dispatches_tool(self):
        """A fallback ToolCallDelta with 'fb_' prefix and real name is dispatched."""
        dispatch_log: list[str] = []

        @tool()
        async def search(query: str) -> str:
            """Search the menu."""
            dispatch_log.append(query)
            return f"Results: {query}"

        Agent = _make_agent_with_tool(search)
        tool_name = getattr(search, TOOL_META).name
        tool_args = json.dumps({"query": "dumplings"})
        fb_id = f"fb_{uuid.uuid4().hex[:16]}"

        mock = MockTransport()
        mock.queue_stream(
            [
                # Original streaming delta has no name (poolside bug)
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="call_orig",
                        name=None,
                        input_delta=tool_args,
                    )
                ),
                # Transport fallback emits a fresh chunk with the real name
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id=fb_id,
                        name=tool_name,
                        input_delta=tool_args,
                    )
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        mock.queue_stream([CompletionChunk(delta="Here are the results."), CompletionChunk(stop_reason="end_turn")])

        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(Agent(), "find dumplings"):
            pass

        # Only the 'fb_' entry should have been dispatched
        assert dispatch_log == ["dumplings"]

    @pytest.mark.asyncio
    async def test_phantom_empty_entry_skipped_real_entry_runs(self):
        """Phantom (empty name + empty input) skipped; real entry with name runs."""
        dispatch_log: list[str] = []

        @tool()
        async def lookup(item: str) -> str:
            """Look up an item."""
            dispatch_log.append(item)
            return f"Found {item}"

        Agent = _make_agent_with_tool(lookup)
        tool_name = getattr(lookup, TOOL_META).name
        tool_args = json.dumps({"item": "Peking Duck"})

        mock = MockTransport()
        mock.queue_stream(
            [
                # Phantom: random ID, no name, empty args (blank header delta)
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="call_phantom",
                        name=None,
                        input_delta="",
                    )
                ),
                # Real: fallback ID, real name, real args
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="fb_real_id",
                        name=tool_name,
                        input_delta=tool_args,
                    )
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        mock.queue_stream([CompletionChunk(delta="Found Peking Duck."), CompletionChunk(stop_reason="end_turn")])

        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(Agent(), "look up Peking Duck"):
            pass

        assert dispatch_log == ["Peking Duck"]

    @pytest.mark.asyncio
    async def test_multiple_empty_name_deltas_all_skipped(self):
        """Multiple empty-name entries in one turn are ALL skipped."""
        dispatch_log: list[str] = []

        @tool()
        async def act(value: str) -> str:
            """Act on a value."""
            dispatch_log.append(value)
            return "ok"

        Agent = _make_agent_with_tool(act)
        mock = MockTransport()

        mock.queue_stream(
            [
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="call_a", name=None, input_delta='{"value": "x"}')
                ),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="call_b", name=None, input_delta='{"value": "y"}')
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        mock.queue_stream([CompletionChunk(delta="Done."), CompletionChunk(stop_reason="end_turn")])

        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(Agent(), "do both"):
            pass

        assert dispatch_log == []


# ---------------------------------------------------------------------------
# Tests: OpenAI transport accumulated-name and fallback logic
# ---------------------------------------------------------------------------


def _make_chunk_obj(d: dict) -> Any:
    """Build a single fake OpenAI ChatCompletionChunk MagicMock."""
    chunk = MagicMock()
    choice = MagicMock()
    choice.finish_reason = d.get("finish_reason")
    delta = MagicMock()
    delta.content = d.get("content")
    tcs = []
    for tc in d.get("tool_calls", []):
        tc_mock = MagicMock()
        tc_mock.index = tc.get("index", 0)
        tc_mock.id = tc.get("id", None)
        fn = MagicMock()
        fn.name = tc.get("fn_name", None)
        fn.arguments = tc.get("fn_args", "")
        tc_mock.function = fn
        tcs.append(tc_mock)
    delta.tool_calls = tcs or None
    delta.function_call = None
    choice.delta = delta
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _fake_openai_create(deltas: list[dict]):
    """Return an *async def* that, when awaited, provides an async context manager
    which yields mock chunk objects — matching the SDK's ``stream=True`` API:

        stream_cm = await client.chat.completions.create(stream=True)
        async with stream_cm as stream:
            async for chunk in stream:
                ...
    """

    @asynccontextmanager
    async def _ctx_mgr():
        async def _gen():
            for d in deltas:
                yield _make_chunk_obj(d)

        yield _gen()

    async def _create(**kwargs):
        return _ctx_mgr()

    return _create


class TestOpenAITransportStreamingFix:
    """Tests for the OpenAI streaming transport tool-name handling."""

    @pytest.mark.asyncio
    async def test_name_in_first_delta_propagated(self):
        """Standard case: name on first delta → ToolCallDelta.name is non-None."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._openai import OpenAITransport

        config, _ = LLMConfig.for_testing()
        transport = OpenAITransport(config)

        stream_data = [
            {"tool_calls": [{"index": 0, "id": "call_1", "fn_name": "search_menu_tool", "fn_args": ""}]},
            {"tool_calls": [{"index": 0, "fn_name": None, "fn_args": '{"query": "noodles"}'}]},
            {"finish_reason": "tool_calls"},
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = _fake_openai_create(stream_data)

        call_kwargs = {"model": "test", "messages": [], "stream": True}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(mock_client, call_kwargs, model="test"):
            chunks.append(chunk)

        tc_deltas = [c for c in chunks if c.tool_call_delta is not None]
        assert any(d.tool_call_delta.name == "search_menu_tool" for d in tc_deltas)
        assert all(d.tool_call_delta.tool_use_id == "call_1" for d in tc_deltas)

    @pytest.mark.asyncio
    async def test_blank_header_delta_suppressed(self):
        """A delta with no name and no args must not produce a ToolCallDelta chunk."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._openai import OpenAITransport

        config, _ = LLMConfig.for_testing()
        transport = OpenAITransport(config)

        stream_data = [
            # Blank header: no fn_name, no fn_args — must be suppressed
            {"tool_calls": [{"index": 0, "id": "call_2", "fn_name": None, "fn_args": ""}]},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "fn_name": "get_menu_item",
                        "fn_args": '{"item_id": "1"}',
                    }
                ]
            },
            {"finish_reason": "tool_calls"},
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = _fake_openai_create(stream_data)

        call_kwargs = {"model": "test", "messages": [], "stream": True}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(mock_client, call_kwargs, model="test"):
            chunks.append(chunk)

        tc_deltas = [c for c in chunks if c.tool_call_delta is not None]
        # Blank header must not appear; only the real content chunk should
        assert len(tc_deltas) == 1
        assert tc_deltas[0].tool_call_delta.name == "get_menu_item"
        assert tc_deltas[0].tool_call_delta.input_delta == '{"item_id": "1"}'

    @pytest.mark.asyncio
    async def test_accumulated_name_used_on_subsequent_deltas(self):
        """Name from first delta is accumulated; all deltas share the same tool_use_id."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._openai import OpenAITransport

        config, _ = LLMConfig.for_testing()
        transport = OpenAITransport(config)

        stream_data = [
            {"tool_calls": [{"index": 0, "id": "call_3", "fn_name": "check_dietary", "fn_args": ""}]},
            {"tool_calls": [{"index": 0, "fn_name": None, "fn_args": '{"dish_name"'}]},
            {"tool_calls": [{"index": 0, "fn_name": None, "fn_args": ': "Mapo Tofu"}'}]},
            {"finish_reason": "tool_calls"},
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = _fake_openai_create(stream_data)

        call_kwargs = {"model": "test", "messages": [], "stream": True}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(mock_client, call_kwargs, model="test"):
            chunks.append(chunk)

        tc_deltas = [c for c in chunks if c.tool_call_delta is not None]
        # First delta provides name
        assert tc_deltas[0].tool_call_delta.name == "check_dietary"
        # All deltas share the same tool_use_id
        assert all(d.tool_call_delta.tool_use_id == "call_3" for d in tc_deltas)

    @pytest.mark.asyncio
    async def test_empty_name_triggers_fallback_call(self):
        """finish_reason=tool_calls with empty name triggers a non-streaming fallback."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._openai import OpenAITransport

        config, _ = LLMConfig.for_testing()
        transport = OpenAITransport(config)

        streaming_data = [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_pool",
                        "fn_name": None,  # poolside bug: name missing
                        "fn_args": '{"query": "dim sum"}',
                    }
                ]
            },
            {"finish_reason": "tool_calls"},
        ]

        # Non-streaming fallback response with the real tool name
        fallback_tc = MagicMock()
        fallback_tc.id = "call_pool_real"
        fallback_fn = MagicMock()
        fallback_fn.name = "search_menu_tool"
        fallback_fn.arguments = '{"query": "dim sum"}'
        fallback_tc.function = fallback_fn
        fallback_msg = MagicMock()
        fallback_msg.tool_calls = [fallback_tc]
        fallback_resp = MagicMock()
        fallback_resp.choices = [MagicMock(message=fallback_msg)]

        streaming_create = _fake_openai_create(streaming_data)

        async def fallback_create(**kwargs):
            return fallback_resp

        call_count = 0

        async def _dispatch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return await streaming_create(**kwargs)
            return await fallback_create(**kwargs)

        mock_client = MagicMock()
        mock_client.chat.completions.create = _dispatch

        call_kwargs = {"model": "test", "messages": [], "stream": True}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(mock_client, call_kwargs, model="test"):
            chunks.append(chunk)

        # The fallback may replace the missing name, but it must preserve the
        # already-observed provider identity so the eventual tool_result can
        # correlate with the original streamed call.
        fb_deltas = [
            c for c in chunks if c.tool_call_delta is not None and c.tool_call_delta.tool_use_id == "call_pool"
        ]
        assert len(fb_deltas) == 1
        assert fb_deltas[0].tool_call_delta.name == "search_menu_tool"
        assert "dim sum" in fb_deltas[0].tool_call_delta.input_delta
        assert call_count == 2  # streaming + fallback

    @pytest.mark.asyncio
    async def test_no_fallback_when_name_present(self):
        """When all tool names are known after streaming, no fallback call is made."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._openai import OpenAITransport

        config, _ = LLMConfig.for_testing()
        transport = OpenAITransport(config)

        stream_data = [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_ok",
                        "fn_name": "handoff_to",
                        "fn_args": '{"to_agent": "Food Expert"}',
                    }
                ]
            },
            {"finish_reason": "tool_calls"},
        ]

        call_count = 0
        streaming_create = _fake_openai_create(stream_data)

        async def _dispatch(**kwargs):
            nonlocal call_count
            call_count += 1
            return await streaming_create(**kwargs)

        mock_client = MagicMock()
        mock_client.chat.completions.create = _dispatch

        call_kwargs = {"model": "test", "messages": [], "stream": True}
        async for _ in transport._stream(mock_client, call_kwargs, model="test"):
            pass

        # Only one call — the streaming call; no fallback
        assert call_count == 1


# ---------------------------------------------------------------------------
# Tests: Anthropic transport correctly emits tool name on content_block_start
# ---------------------------------------------------------------------------


def _fake_anthropic_events(events: list[dict]) -> Any:
    """Return a fake Anthropic streaming context manager."""

    async def _aiter():
        for ev in events:
            chunk = MagicMock()
            chunk.type = ev["type"]
            if ev["type"] == "content_block_start":
                block = MagicMock()
                block.type = ev.get("block_type", "text")
                block.id = ev.get("block_id", None)
                block.name = ev.get("block_name", None)
                chunk.content_block = block
            elif ev["type"] == "content_block_delta":
                delta = MagicMock()
                delta.type = ev.get("delta_type", "text_delta")
                delta.text = ev.get("text", "")
                delta.thinking = ev.get("thinking", "")
                delta.partial_json = ev.get("partial_json", "")
                chunk.delta = delta
            elif ev["type"] == "message_delta":
                delta = MagicMock()
                delta.stop_reason = ev.get("stop_reason", "end_turn")
                chunk.delta = delta
                chunk.usage = None
            yield chunk

    class FakeStream:
        async def __aenter__(self):
            return _aiter()

        async def __aexit__(self, *_):
            pass

    return FakeStream()


class TestAnthropicTransportToolName:
    """Anthropic transport always sends name on content_block_start; verify runner sees it."""

    @pytest.mark.asyncio
    async def test_tool_name_emitted_on_content_block_start(self):
        """content_block_start with type=tool_use emits ToolCallDelta with name set."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._anthropic import AnthropicTransport

        config, _ = LLMConfig.for_testing()
        # Override provider so we can build an AnthropicTransport
        from dataclasses import replace as dc_replace

        config = dc_replace(config, provider="anthropic")
        transport = AnthropicTransport(config)

        events = [
            {
                "type": "content_block_start",
                "block_type": "tool_use",
                "block_id": "toolu_001",
                "block_name": "search_menu_tool",
            },
            {
                "type": "content_block_delta",
                "delta_type": "input_json_delta",
                "partial_json": '{"query": "spring rolls"}',
            },
            {"type": "content_block_stop"},
            {"type": "message_delta", "stop_reason": "tool_use"},
        ]

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_fake_anthropic_events(events))

        call_kwargs = {"model": "test", "messages": []}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(mock_client, call_kwargs):
            chunks.append(chunk)

        tc_deltas = [c for c in chunks if c.tool_call_delta is not None]
        # First delta (content_block_start) must carry the tool name
        assert tc_deltas[0].tool_call_delta.name == "search_menu_tool"
        assert tc_deltas[0].tool_call_delta.tool_use_id == "toolu_001"
        # Subsequent input_json_delta has name=None
        assert tc_deltas[1].tool_call_delta.name is None
        assert "spring rolls" in tc_deltas[1].tool_call_delta.input_delta

    @pytest.mark.asyncio
    async def test_runner_receives_anthropic_tool_name_correctly(self):
        """End-to-end: AgentRunner dispatches a tool via Anthropic-style streaming."""
        dispatch_log: list[str] = []

        @tool()
        async def find_dish(query: str) -> str:
            """Find dishes."""
            dispatch_log.append(query)
            return f"Found: {query}"

        Agent = _make_agent_with_tool(find_dish)
        tool_name = getattr(find_dish, TOOL_META).name
        mock = MockTransport()

        # Anthropic transport emits name on first ToolCallDelta (content_block_start)
        # and None on subsequent input_json_delta chunks — identical to what the
        # real Anthropic transport produces.
        mock.queue_stream(
            [
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="toolu_abc",
                        name=tool_name,  # Anthropic always provides name here
                        input_delta="",
                    )
                ),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(
                        tool_use_id="toolu_abc",
                        name=None,
                        input_delta='{"query": "dim sum"}',
                    )
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        mock.queue_stream([CompletionChunk(delta="Dim sum found!"), CompletionChunk(stop_reason="end_turn")])

        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(Agent(), "find dim sum"):
            pass

        assert dispatch_log == ["dim sum"]

    @pytest.mark.asyncio
    async def test_multiple_anthropic_tool_calls_both_dispatched(self):
        """Two consecutive Anthropic tool calls (separate content blocks) both run."""
        dispatch_log: list[tuple] = []

        @tool()
        async def search(query: str) -> str:
            """Search."""
            dispatch_log.append(("search", query))
            return f"Results: {query}"

        @tool()
        async def lookup(item_id: str) -> str:
            """Lookup."""
            dispatch_log.append(("lookup", item_id))
            return f"Item: {item_id}"

        from lauren_ai._tools import TOOL_META

        @agent(model="test")
        @use_tools(search, lookup)
        class MultiToolAgent:
            pass

        search_meta = getattr(search, TOOL_META)
        lookup_meta = getattr(lookup, TOOL_META)
        MultiToolAgent.__lauren_ai_agent__.tools = {
            search_meta.name: (search, search_meta),
            lookup_meta.name: (lookup, lookup_meta),
        }

        mock = MockTransport()
        mock.queue_stream(
            [
                # First tool call
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="toolu_1", name=search_meta.name, input_delta="")
                ),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="toolu_1", name=None, input_delta='{"query": "noodles"}')
                ),
                # Second tool call
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="toolu_2", name=lookup_meta.name, input_delta="")
                ),
                CompletionChunk(
                    tool_call_delta=ToolCallDelta(tool_use_id="toolu_2", name=None, input_delta='{"item_id": "mi_42"}')
                ),
                CompletionChunk(stop_reason="tool_use"),
            ]
        )
        mock.queue_stream([CompletionChunk(delta="All done!"), CompletionChunk(stop_reason="end_turn")])

        runner = AgentRunnerBase(transport=mock)
        async for _ in await runner.run_stream(MultiToolAgent(), "find and lookup"):
            pass

        assert ("search", "noodles") in dispatch_log
        assert ("lookup", "mi_42") in dispatch_log


# ---------------------------------------------------------------------------
# Tests: Anthropic transport empty-name fallback
# ---------------------------------------------------------------------------


def _fake_anthropic_events_empty_name(events: list[dict], block_index_attr: bool = False) -> Any:
    """Like _fake_anthropic_events but supports an index attribute on events."""

    async def _aiter():
        for ev in events:
            chunk = MagicMock()
            chunk.type = ev["type"]
            chunk.index = ev.get("index", 0)
            if ev["type"] == "content_block_start":
                block = MagicMock()
                block.type = ev.get("block_type", "text")
                block.id = ev.get("block_id", None)
                block.name = ev.get("block_name", None)
                chunk.content_block = block
            elif ev["type"] == "content_block_delta":
                delta = MagicMock()
                delta.type = ev.get("delta_type", "text_delta")
                delta.text = ev.get("text", "")
                delta.thinking = ev.get("thinking", "")
                delta.partial_json = ev.get("partial_json", "")
                chunk.delta = delta
            elif ev["type"] == "message_delta":
                delta = MagicMock()
                delta.stop_reason = ev.get("stop_reason", "end_turn")
                chunk.delta = delta
                chunk.usage = None
            yield chunk

    class FakeStream:
        async def __aenter__(self):
            return _aiter()

        async def __aexit__(self, *_):
            pass

    return FakeStream()


class TestAnthropicTransportEmptyNameFallback:
    """AnthropicTransport must handle empty-name tool calls via a non-streaming fallback."""

    @pytest.mark.asyncio
    async def test_empty_name_on_content_block_start_suppresses_header(self):
        """Header chunk is not emitted when content_block_start has no name."""
        from dataclasses import replace as dc_replace

        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._anthropic import AnthropicTransport

        config, _ = LLMConfig.for_testing()
        config = dc_replace(config, provider="anthropic")
        transport = AnthropicTransport(config)

        events = [
            {
                "type": "content_block_start",
                "block_type": "tool_use",
                "block_id": "toolu_empty",
                "block_name": "",  # empty name — common in some proxy implementations
            },
            {
                "type": "content_block_delta",
                "delta_type": "input_json_delta",
                "partial_json": '{"query": "tofu"}',
            },
            {"type": "content_block_stop"},
            {"type": "message_delta", "stop_reason": "tool_use"},
        ]

        # Fallback non-streaming response
        fb_block = MagicMock()
        fb_block.type = "tool_use"
        fb_block.name = "search_menu_tool"
        fb_block.id = "toolu_fb_001"
        fb_block.input = {"query": "tofu"}
        fb_resp = MagicMock()
        fb_resp.content = [fb_block]

        async def fake_create(**kwargs):
            return fb_resp

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_fake_anthropic_events_empty_name(events))
        mock_client.messages.create = fake_create

        call_kwargs = {"model": "test", "messages": []}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(mock_client, call_kwargs):
            chunks.append(chunk)

        tc_deltas = [c for c in chunks if c.tool_call_delta is not None]
        # The blank-name header is suppressed; the fallback reuses the original
        # unambiguous provider identity.
        fb_deltas = [d for d in tc_deltas if d.tool_call_delta.tool_use_id == "toolu_empty"]
        assert len(fb_deltas) == 1
        assert fb_deltas[0].tool_call_delta.name == "search_menu_tool"

    @pytest.mark.asyncio
    async def test_empty_name_triggers_fallback_api_call(self):
        """A non-streaming call is made when content_block_start has empty name."""
        from dataclasses import replace as dc_replace

        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._anthropic import AnthropicTransport

        config, _ = LLMConfig.for_testing()
        config = dc_replace(config, provider="anthropic")
        transport = AnthropicTransport(config)

        events = [
            {
                "type": "content_block_start",
                "block_type": "tool_use",
                "block_id": "toolu_noname",
                "block_name": None,  # None is also treated as empty
            },
            {"type": "content_block_stop"},
            {"type": "message_delta", "stop_reason": "tool_use"},
        ]

        fallback_call_count = 0

        fb_block = MagicMock()
        fb_block.type = "tool_use"
        fb_block.name = "check_dietary_tool"
        fb_block.id = "toolu_real"
        fb_block.input = {"dish_name": "Mapo Tofu"}
        fb_resp = MagicMock()
        fb_resp.content = [fb_block]

        async def fake_create(**kwargs):
            nonlocal fallback_call_count
            fallback_call_count += 1
            return fb_resp

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_fake_anthropic_events_empty_name(events))
        mock_client.messages.create = fake_create

        call_kwargs = {"model": "test", "messages": []}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(mock_client, call_kwargs):
            chunks.append(chunk)

        assert fallback_call_count == 1
        fb_deltas = [
            c for c in chunks if c.tool_call_delta is not None and c.tool_call_delta.tool_use_id == "toolu_noname"
        ]
        assert len(fb_deltas) == 1
        assert fb_deltas[0].tool_call_delta.name == "check_dietary_tool"

    @pytest.mark.asyncio
    async def test_no_fallback_when_name_present(self):
        """No extra API call when all tool names arrive on content_block_start."""
        from dataclasses import replace as dc_replace

        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._anthropic import AnthropicTransport

        config, _ = LLMConfig.for_testing()
        config = dc_replace(config, provider="anthropic")
        transport = AnthropicTransport(config)

        events = [
            {
                "type": "content_block_start",
                "block_type": "tool_use",
                "block_id": "toolu_good",
                "block_name": "find_restaurant_tool",
            },
            {
                "type": "content_block_delta",
                "delta_type": "input_json_delta",
                "partial_json": '{"city": "Beijing"}',
            },
            {"type": "content_block_stop"},
            {"type": "message_delta", "stop_reason": "tool_use"},
        ]

        fallback_call_count = 0

        async def fake_create(**kwargs):
            nonlocal fallback_call_count
            fallback_call_count += 1

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_fake_anthropic_events_empty_name(events))
        mock_client.messages.create = fake_create

        call_kwargs = {"model": "test", "messages": []}
        async for _ in transport._stream(mock_client, call_kwargs):
            pass

        assert fallback_call_count == 0


# ---------------------------------------------------------------------------
# Tests: Ollama transport empty-name guard and fallback
# ---------------------------------------------------------------------------


def _make_ollama_line(d: dict) -> str:
    return json.dumps(d)


class TestOllamaTransportEmptyNameFallback:
    """OllamaTransport must skip empty-name tool calls and fall back to non-streaming."""

    @pytest.mark.asyncio
    async def test_empty_name_tool_call_skipped(self):
        """Tool calls with empty name are not emitted as ToolCallDelta chunks."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._ollama import OllamaTransport

        config, _ = LLMConfig.for_testing()
        transport = OllamaTransport(config)

        lines = [
            _make_ollama_line(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "toolu_empty",
                                "function": {
                                    "name": "",  # empty name
                                    "arguments": {"query": "noodles"},
                                },
                            }
                        ],
                    },
                    "done": False,
                }
            ),
            _make_ollama_line(
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                }
            ),
        ]

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.aiter_lines = _aiter_lines

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_stream(*_a, **_kw):
            yield mock_resp

        mock_client = MagicMock()
        mock_client.stream = _fake_stream
        transport._client = mock_client

        payload = {"model": "llama3.2", "messages": [], "stream": True}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(payload):
            chunks.append(chunk)

        tc_deltas = [c for c in chunks if c.tool_call_delta is not None]
        assert len(tc_deltas) == 0  # empty-name call must be suppressed

    @pytest.mark.asyncio
    async def test_empty_name_triggers_fallback(self):
        """Empty-name tool call with done_reason=tool triggers non-streaming fallback."""

        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._ollama import OllamaTransport

        config, _ = LLMConfig.for_testing()
        transport = OllamaTransport(config)

        lines = [
            _make_ollama_line(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "toolu_bad",
                                "function": {"name": "", "arguments": {"q": "dumplings"}},
                            }
                        ],
                    },
                    "done": False,
                }
            ),
            _make_ollama_line(
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 8,
                    "eval_count": 3,
                }
            ),
        ]

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.aiter_lines = _aiter_lines

        # Fallback non-streaming response
        fallback_body = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "toolu_real", "function": {"name": "search_menu_tool", "arguments": {"q": "dumplings"}}}
                ],
            },
            "done": True,
            "done_reason": "stop",
        }
        mock_fb_resp = MagicMock()
        mock_fb_resp.status_code = 200
        mock_fb_resp.json = MagicMock(return_value=fallback_body)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_stream_ctx(*_a, **_kw):
            yield mock_stream_resp

        fallback_call_count = 0

        async def _fake_post(path, json=None):
            nonlocal fallback_call_count
            fallback_call_count += 1
            return mock_fb_resp

        mock_client = MagicMock()
        mock_client.stream = _fake_stream_ctx
        mock_client.post = _fake_post

        payload = {"model": "llama3.2", "messages": [], "stream": True}

        # Patch _get_client to return our mock
        transport._client = mock_client

        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(payload):
            chunks.append(chunk)

        # Fallback must have been called once
        assert fallback_call_count == 1
        fb_deltas = [
            c for c in chunks if c.tool_call_delta is not None and c.tool_call_delta.tool_use_id.startswith("fb_")
        ]
        assert len(fb_deltas) == 1
        assert fb_deltas[0].tool_call_delta.name == "search_menu_tool"

    @pytest.mark.asyncio
    async def test_no_fallback_when_name_present(self):
        """No fallback call when all Ollama tool calls have names."""
        from lauren_ai._config import LLMConfig
        from lauren_ai._transport._ollama import OllamaTransport

        config, _ = LLMConfig.for_testing()
        transport = OllamaTransport(config)

        lines = [
            _make_ollama_line(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "toolu_good",
                                "function": {"name": "find_dish_tool", "arguments": {"dish": "Peking Duck"}},
                            }
                        ],
                    },
                    "done": False,
                }
            ),
            _make_ollama_line(
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                }
            ),
        ]

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.aiter_lines = _aiter_lines

        fallback_count = 0

        async def _fake_post(*_a, **_kw):
            nonlocal fallback_count
            fallback_count += 1

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_stream(*_a, **_kw):
            yield mock_resp

        mock_client = MagicMock()
        mock_client.stream = _fake_stream
        mock_client.post = _fake_post
        transport._client = mock_client

        payload = {"model": "llama3.2", "messages": [], "stream": True}
        chunks: list[CompletionChunk] = []
        async for chunk in transport._stream(payload):
            chunks.append(chunk)

        assert fallback_count == 0
        tc_deltas = [c for c in chunks if c.tool_call_delta is not None]
        assert len(tc_deltas) == 1
        assert tc_deltas[0].tool_call_delta.name == "find_dish_tool"
