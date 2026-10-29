"""Regression coverage for provider-neutral tool-call conversation integrity."""

from __future__ import annotations

from random import Random
from unittest.mock import patch

import pytest

import lauren_ai._memory as memory_module
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


def test_valid_appends_use_the_incremental_tool_history_index() -> None:
    memory = ShortTermMemory()
    memory.add_user("read both")
    calls = [
        ToolCall(tool_use_id="a", name="read_file", input={}),
        ToolCall(tool_use_id="b", name="read_file", input={}),
    ]
    memory.add_assistant(type("Completion", (), {"content": "", "tool_calls": calls})())
    with patch("lauren_ai._memory._validate_tool_history", wraps=memory_module._validate_tool_history) as full_scan:
        memory.add_tool_results(
            [
                ToolResult.ok("A", tool_use_id="a"),
                ToolResult.ok("B", tool_use_id="b"),
            ]
        )
        assert memory.validate_tool_history().ok
        memory.add_user("continue")
        assert memory.validate_tool_history().ok

    assert full_scan.call_count == 0


def test_incremental_index_matches_full_validation_for_seeded_histories() -> None:
    random = Random(138)
    for _ in range(10_000):
        memory = ShortTermMemory()
        history: list[dict[str, object]] = []
        for _ in range(random.randint(0, 8)):
            message_type = random.choice(("user", "assistant", "calls", "results"))
            if message_type == "user":
                message = {"role": "user", "content": "text"}
            elif message_type == "assistant":
                message = {"role": "assistant", "content": "done"}
            elif message_type == "calls":
                message = _assistant(*(random.choice(("a", "b", "")) for _ in range(random.randint(1, 2))))
            else:
                message = _results(*(random.choice(("a", "b", "unknown", "")) for _ in range(random.randint(1, 2))))

            history.append(message)
            memory._append_message(message)
            assert memory.validate_tool_history() == memory_module._validate_tool_history(history)


def test_open_tool_exchange_report_is_reused_until_an_append() -> None:
    memory = ShortTermMemory()
    memory.add_user("read")
    call = ToolCall(tool_use_id="a", name="read_file", input={})
    memory.add_assistant(type("Completion", (), {"content": "", "tool_calls": [call]})())
    with patch("lauren_ai._memory._validate_tool_history", wraps=memory_module._validate_tool_history) as full_scan:
        first = memory.validate_tool_history()
        second = memory.validate_tool_history()
        assert first == second
        assert first.first_issue is not None
        assert first.first_issue.code == "missing_results"
        assert full_scan.call_count == 1

        memory.add_tool_result(ToolResult.ok("A", tool_use_id="a"))
        assert memory.validate_tool_history().ok

    assert full_scan.call_count == 1


def test_replacing_history_invalidates_the_incremental_index() -> None:
    memory = ShortTermMemory()
    memory.add_user("valid")
    memory.add_assistant({"role": "assistant", "content": "done"})
    assert memory.validate_tool_history().ok

    memory._messages = [
        {"role": "user", "content": "task"},
        _assistant("a"),
        {"role": "user", "content": "intervening"},
        _results("a"),
    ]

    report = memory.validate_tool_history()
    assert not report.ok
    assert report.first_issue is not None
    assert report.first_issue.code == "non_adjacent_results"


def test_direct_append_invalidates_a_cached_tool_history_report() -> None:
    memory = ShortTermMemory()
    memory.add_user("valid")
    memory.add_assistant({"role": "assistant", "content": "done"})
    assert memory.validate_tool_history().ok
    memory._messages.append(_results("orphan"))

    with patch("lauren_ai._memory._validate_tool_history", wraps=memory_module._validate_tool_history) as full_scan:
        report = memory.validate_tool_history()

    assert not report.ok
    assert report.first_issue is not None
    assert report.first_issue.code == "orphan_result"
    assert full_scan.call_count == 1


def test_repair_and_ensure_skip_valid_history_rewrites() -> None:
    memory = ShortTermMemory()
    memory.add_user("valid")
    memory.add_assistant({"role": "assistant", "content": "done"})

    with patch("lauren_ai._memory._repair_tool_history", wraps=memory_module._repair_tool_history) as repair:
        assert memory.repair_tool_history().ok
        memory.ensure_valid()

    assert repair.call_count == 0


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
