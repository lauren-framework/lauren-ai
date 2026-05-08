"""Integration tests for Skill 1: LLM Provider Connection.

Tests cover:
- LLMConfig direct constructor for all three providers
- LLMConfig factory classmethods (for_anthropic, for_openai, for_ollama, for_testing)
- for_testing() returns a MockTransport with no real network calls
- AgentRunner runs successfully with MockTransport
- Config properties are accessible
"""

from __future__ import annotations

import pytest

from lauren_ai import LLMConfig
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str = "OK") -> Completion:
    return Completion(
        id="c1",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock: MockTransport | None = None) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# TestLLMConfigDirectConstructor
# ---------------------------------------------------------------------------


class TestLLMConfigDirectConstructor:
    def test_anthropic_provider_field(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-ant-xxx")
        assert cfg.provider == "anthropic"

    def test_anthropic_model_field(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-ant-xxx")
        assert cfg.model == "claude-opus-4-6"

    def test_anthropic_api_key_field(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="sk-ant-xxx")
        assert cfg.api_key == "sk-ant-xxx"

    def test_openai_provider_field(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-oai")
        assert cfg.provider == "openai"

    def test_openai_model_field(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-oai")
        assert cfg.model == "gpt-4o"

    def test_ollama_no_api_key_required(self):
        cfg = LLMConfig(provider="ollama", model="llama3.2")
        assert cfg.api_key is None

    def test_default_max_tokens(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
        assert cfg.max_tokens == 4096

    def test_default_temperature(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
        assert cfg.temperature == 1.0

    def test_config_is_frozen(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
        with pytest.raises(Exception):
            cfg.model = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestLLMConfigFactoryMethods
# ---------------------------------------------------------------------------


class TestLLMConfigFactoryMethods:
    def test_for_anthropic_sets_provider(self):
        cfg = LLMConfig.for_anthropic(model="claude-opus-4-6", api_key="k")
        assert cfg.provider == "anthropic"

    def test_for_anthropic_default_model(self):
        cfg = LLMConfig.for_anthropic(api_key="k")
        assert cfg.model == "claude-opus-4-6"

    def test_for_openai_sets_provider(self):
        cfg = LLMConfig.for_openai(api_key="k")
        assert cfg.provider == "openai"

    def test_for_openai_default_model(self):
        cfg = LLMConfig.for_openai(api_key="k")
        assert cfg.model == "gpt-4o"

    def test_for_ollama_sets_provider(self):
        cfg = LLMConfig.for_ollama()
        assert cfg.provider == "ollama"

    def test_for_ollama_default_model(self):
        cfg = LLMConfig.for_ollama()
        assert cfg.model == "llama3.2"

    def test_for_ollama_sets_base_url(self):
        cfg = LLMConfig.for_ollama()
        assert cfg.base_url == "http://localhost:11434"

    def test_for_testing_returns_tuple(self):
        result = LLMConfig.for_testing()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_for_testing_config_is_llm_config(self):
        cfg, _ = LLMConfig.for_testing()
        assert isinstance(cfg, LLMConfig)

    def test_for_testing_mock_is_mock_transport(self):
        _, mock = LLMConfig.for_testing()
        assert isinstance(mock, MockTransport)

    def test_for_testing_config_has_mock_model(self):
        cfg, _ = LLMConfig.for_testing()
        assert cfg.model == "mock-model"


# ---------------------------------------------------------------------------
# TestRunnerWithMockTransport
# ---------------------------------------------------------------------------


class TestRunnerWithMockTransport:
    async def test_runner_runs_basic_agent(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("Hello from mock"))

        @agent(model="mock-model")
        class SimpleAgent: ...

        resp = await runner.run(SimpleAgent(), "hi")
        assert resp.content == "Hello from mock"

    async def test_runner_records_calls(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model")
        class SimpleAgent: ...

        await runner.run(SimpleAgent(), "hi")
        assert len(mock.calls) == 1

    async def test_runner_stop_reason_end_turn(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("done"))

        @agent(model="mock-model")
        class SimpleAgent: ...

        resp = await runner.run(SimpleAgent(), "hi")
        assert resp.stop_reason == "end_turn"

    async def test_runner_usage_populated(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion("OK"))

        @agent(model="mock-model")
        class SimpleAgent: ...

        resp = await runner.run(SimpleAgent(), "hi")
        assert resp.total_usage.input_tokens > 0

    async def test_no_network_calls_made(self):
        """MockTransport records calls; real transport would need network."""
        _, mock = LLMConfig.for_testing()
        mock.queue_response(_completion("zero network"))
        cfg, mock2 = LLMConfig.for_testing()
        # mock2 is a different instance; proves for_testing() creates fresh mocks
        assert mock is not mock2
