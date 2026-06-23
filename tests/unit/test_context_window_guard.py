"""Tests for the model-aware context-window guard (PRD-133 B + D + E).

Covers:
* the model→context-window registry and ``usable_context_budget`` (B),
* the dict-safe shared token estimator every transport now uses (D),
* the runner's hard pre-send guard ``_fit_to_context`` — truncate-to-fit and
  the graceful ``AgentContextOverflowError`` for an irreducible request (D + E).
"""

from __future__ import annotations

import pytest

from lauren_ai._agents._runner import _cheap_token_estimate, _fit_to_context
from lauren_ai._config import (
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    AgentConfig,
    context_window_for,
)
from lauren_ai._exceptions import AgentContextOverflowError
from lauren_ai._transport import estimate_message_tokens
from lauren_ai._transport._mock import MockTransport

# ── helpers ──────────────────────────────────────────────────────────────────


def _tool_use_msg(tid: str, name: str, inp: dict) -> dict:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}


def _tool_result_msg(tid: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": False}],
    }


def _blocks(messages: list, btype: str) -> list:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            out.extend(b for b in content if isinstance(b, dict) and b.get("type") == btype)
    return out


# ── B: registry ──────────────────────────────────────────────────────────────


class TestContextWindowRegistry:
    def test_known_family_prefixes(self) -> None:
        assert context_window_for("claude-opus-4-8") == 1_000_000
        assert context_window_for("claude-haiku-4-5") == 200_000
        assert context_window_for("gpt-4o") == 128_000

    def test_longest_key_wins(self) -> None:
        # "gpt-4" and "gpt-4o" both match "gpt-4o"; the longer key must win.
        assert context_window_for("gpt-4o-2024") == MODEL_CONTEXT_WINDOWS["gpt-4o"]
        # bare gpt-4 resolves to the small window, not gpt-4o's.
        assert context_window_for("gpt-4-0613") == MODEL_CONTEXT_WINDOWS["gpt-4"]

    def test_unknown_model_falls_back_to_default(self) -> None:
        assert context_window_for("deepseek-v4-flash") == DEFAULT_CONTEXT_WINDOW
        assert context_window_for("some-random-model") == DEFAULT_CONTEXT_WINDOW

    def test_case_insensitive(self) -> None:
        assert context_window_for("Claude-Opus-4-8") == 1_000_000


class TestUsableContextBudget:
    def test_disabled_when_window_zero(self) -> None:
        assert AgentConfig().usable_context_budget == 0

    def test_subtracts_output_and_reserve(self) -> None:
        cfg = AgentConfig(context_window=200_000, max_tokens_per_turn=4096)
        # reserve = max(4000, 200000//25=8000) = 8000
        assert cfg.usable_context_budget == 200_000 - 4096 - 8000

    def test_reserve_scales_with_window(self) -> None:
        cfg = AgentConfig(context_window=1_000_000, max_tokens_per_turn=4096)
        # reserve = max(4000, 1_000_000//25=40_000) = 40_000
        assert cfg.usable_context_budget == 1_000_000 - 4096 - 40_000

    def test_never_negative(self) -> None:
        cfg = AgentConfig(context_window=1000, max_tokens_per_turn=4096)
        assert cfg.usable_context_budget == 0


# ── D: dict-safe estimator ───────────────────────────────────────────────────


class TestDictSafeEstimator:
    def test_dict_messages_do_not_raise(self) -> None:
        # The runner passes dict messages; the heuristic must not assume objects.
        msgs = [
            {"role": "user", "content": "hello"},
            _tool_use_msg("t1", "read_file", {"path": "x.py"}),
            _tool_result_msg("t1", "Z" * 4000),
        ]
        assert estimate_message_tokens(msgs, "system", None) > 0

    def test_counts_tool_use_input(self) -> None:
        small = estimate_message_tokens([_tool_use_msg("t", "f", {"a": 1})])
        big = estimate_message_tokens([_tool_use_msg("t", "f", {"a": "Z" * 8000})])
        assert big > small + 1000  # the input is part of the count

    def test_dict_form_tool_schemas_do_not_raise(self) -> None:
        # The runner passes tool schemas as plain JSON dicts (not ToolSchema
        # dataclasses); the estimator must read their fields without assuming
        # attribute access — regression for 'dict' object has no attribute 'name'.
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
        assert estimate_message_tokens([{"role": "user", "content": "hi"}], "sys", tools) > 0

    def test_cheap_estimate_dict_tools_do_not_raise(self) -> None:
        tools = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
        assert _cheap_token_estimate([{"role": "user", "content": "hi"}], "sys", tools) > 0

    @pytest.mark.asyncio
    async def test_mock_count_tokens_dict_safe(self) -> None:
        # MockTransport.count_tokens delegates to the same shared estimator that
        # the OpenAI/Ollama transports and the Anthropic fallback now use.
        tr = MockTransport()
        n = await tr.count_tokens([_tool_result_msg("t", "Q" * 1000)], model="gpt-4")
        assert n >= 250


class TestCheapEstimate:
    def test_includes_tool_use_input(self) -> None:
        # The cheap gate must see tool_use inputs (which _message_char_length
        # treats as zero) or it would skip the exact check on the exact request
        # that overflows.
        est = _cheap_token_estimate([_tool_use_msg("t", "f", {"blob": "Z" * 40_000})], None, None)
        assert est > 8000

    def test_conservative_vs_4char(self) -> None:
        msgs = [{"role": "user", "content": "x" * 3500}]
        # 3.5 chars/token over-counts relative to the 4-char base.
        assert _cheap_token_estimate(msgs, None, None) >= estimate_message_tokens(msgs)


# ── D + E: the pre-send guard ────────────────────────────────────────────────


class TestFitToContext:
    @pytest.mark.asyncio
    async def test_disabled_budget_is_identity(self) -> None:
        tr = MockTransport()
        msgs = [{"role": "user", "content": "x" * 100_000}]
        out = await _fit_to_context(tr, msgs, model="gpt-4", system="s", tools=None, budget=0)
        assert out is msgs

    @pytest.mark.asyncio
    async def test_within_budget_is_identity(self) -> None:
        tr = MockTransport()
        msgs = [{"role": "user", "content": "hello"}]
        out = await _fit_to_context(tr, msgs, model="gpt-4", system="s", tools=None, budget=10_000)
        assert out is msgs

    @pytest.mark.asyncio
    async def test_oversized_tool_result_truncated_to_fit(self) -> None:
        tr = MockTransport()
        budget = AgentConfig(context_window=8_192, max_tokens_per_turn=1024).usable_context_budget
        msgs = [
            {"role": "user", "content": "please summarise the file"},
            _tool_use_msg("t1", "read_file", {"path": "big.txt"}),
            _tool_result_msg("t1", "Q" * 200_000),
        ]
        out = await _fit_to_context(tr, msgs, model="gpt-4", system="sys", tools=None, budget=budget)
        # fits the budget…
        assert await tr.count_tokens(out, model="gpt-4", system="sys", tools=None) <= budget
        # …and the tool_use/tool_result pairing is preserved (never removed).
        tus = _blocks(out, "tool_use")
        trs = _blocks(out, "tool_result")
        assert len(tus) == 1 and len(trs) == 1
        assert trs[0]["tool_use_id"] == "t1"

    @pytest.mark.asyncio
    async def test_irreducible_request_raises(self) -> None:
        # A single huge tool_use INPUT can never be truncated (truncating a tool
        # call would corrupt it) → graceful overflow error, not a provider 400.
        tr = MockTransport()
        budget = AgentConfig(context_window=8_192, max_tokens_per_turn=1024).usable_context_budget
        msgs = [
            {"role": "user", "content": "go"},
            _tool_use_msg("t2", "f", {"blob": "Z" * 200_000}),
            _tool_result_msg("t2", "ok"),
        ]
        with pytest.raises(AgentContextOverflowError) as exc:
            await _fit_to_context(tr, msgs, model="gpt-4", system="s", tools=None, budget=budget)
        assert exc.value.required_tokens > exc.value.budget_tokens
        assert exc.value.model == "gpt-4"
        assert "truncation" in str(exc.value).lower()
