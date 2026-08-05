"""Regression coverage for provider-neutral tool-call conversation integrity."""

from __future__ import annotations

import pytest

from lauren_ai._exceptions import ToolConversationIntegrityError
from lauren_ai._memory import ShortTermMemory, ToolExchange
from lauren_ai._tools import ToolResult
from lauren_ai._transport import ToolCall
from lauren_ai._transport._anthropic import _message_to_anthropic
from lauren_ai._transport._openai import _message_to_openai


def _assistant(*call_ids: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": call_id, "name": "read_file", "input": {}} for call_id in call_ids],
    }


def _results(*call_ids: str) -> dict[str, object]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": call_id, "content": "ok"} for call_id in call_ids],
    }


def test_parallel_exchange_requires_exactly_one_result_per_call() -> None:
    memory = ShortTermMemory()
    memory._messages = [{"role": "user", "content": "read both"}, _assistant("a", "b"), _results("a", "b")]

    report = memory.validate_tool_history()

    assert report.ok


def test_missing_parallel_result_is_repaired_in_one_canonical_message() -> None:
    memory = ShortTermMemory()
    memory._messages = [{"role": "user", "content": "read both"}, _assistant("a", "b"), _results("a")]

    report = memory.repair_tool_history()

    assert report.ok
    result_messages = [
        message
        for message in memory._messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_result" for block in message["content"])
    ]
    assert len(result_messages) == 1
    assert {block["tool_use_id"] for block in result_messages[0]["content"]} == {"a", "b"}


@pytest.mark.parametrize(
    ("history", "code"),
    [
        ([_assistant("", "b"), _results("b")], "empty_call_id"),
        ([_assistant("a", "a"), _results("a")], "duplicate_call_id"),
        ([_assistant("a"), _results("unknown")], "unknown_result_id"),
        ([_assistant("a"), _results("a"), _results("a")], "duplicate_result_id"),
        ([_assistant("a"), {"role": "user", "content": "intervening"}, _results("a")], "non_adjacent_results"),
    ],
)
def test_unrepairable_tool_history_is_rejected(history: list[dict[str, object]], code: str) -> None:
    memory = ShortTermMemory()
    memory._messages = [{"role": "user", "content": "task"}, *history]

    with pytest.raises(ToolConversationIntegrityError) as caught:
        memory.ensure_valid()

    assert caught.value.code == code
    assert "tool" not in str(caught.value).lower() or caught.value.expected_count >= 0


def test_commit_exchange_orders_results_and_synthesizes_filtered_call() -> None:
    memory = ShortTermMemory()
    memory.add_user("read both")
    calls = [
        ToolCall(tool_use_id="a", name="read_file", input={}),
        ToolCall(tool_use_id="b", name="read_file", input={}),
    ]
    memory.add_assistant(type("Completion", (), {"content": "", "tool_calls": calls})())
    exchange = memory.begin_tool_exchange(calls, run_id="run-1")

    committed = memory.commit_tool_exchange(
        exchange,
        [ToolResult.ok("A", tool_use_id="a")],
    )

    assert committed.state == "committed"
    assert memory.validate_tool_history().ok
    blocks = memory._messages[-1]["content"]
    assert [block["tool_use_id"] for block in blocks] == ["a", "b"]
    assert blocks[1]["is_error"] is True
    assert [outcome.status for outcome in committed.outcomes] == ["executed", "synthetic"]


def test_commit_exchange_is_idempotent_after_the_first_commit() -> None:
    memory = ShortTermMemory()
    memory.add_user("read both")
    calls = [
        ToolCall(tool_use_id="a", name="read_file", input={}),
        ToolCall(tool_use_id="b", name="read_file", input={}),
    ]
    memory.add_assistant(type("Completion", (), {"content": "", "tool_calls": calls})())
    exchange = memory.begin_tool_exchange(calls, run_id="run-1")

    committed = memory.commit_tool_exchange(exchange, [ToolResult.ok("A", tool_use_id="a")])
    repeated = memory.commit_tool_exchange(committed, [])
    aborted = memory.abort_tool_exchange(committed, repaired=True)

    assert repeated == committed
    assert aborted == committed
    assert len(memory._messages) == 3
    assert memory.validate_tool_history().ok


def test_serializer_rejects_empty_tool_result_ids_before_network_io() -> None:
    malformed = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ""}]}

    with pytest.raises(ToolConversationIntegrityError, match="without an ID"):
        _message_to_openai(malformed)
    with pytest.raises(ToolConversationIntegrityError, match="without an ID"):
        _message_to_anthropic(malformed)


def test_exchange_rejects_duplicate_call_identity() -> None:
    calls = [ToolCall(tool_use_id="same", name="one", input={}), ToolCall(tool_use_id="same", name="two", input={})]

    with pytest.raises(ValueError, match="unique"):
        ToolExchange.from_tool_calls(calls)
