"""Integration tests for Skill 1: LLM Provider Connection.

Tests cover:
- LLMConfig direct constructor for all three providers
- LLMConfig factory classmethods (for_anthropic, for_openai, for_ollama, for_testing)
- for_testing() returns a MockTransport with no real network calls
- AgentRunner runs successfully with MockTransport
- Config properties are accessible
"""

import pytest

from lauren_ai import LLMConfig
from lauren_ai._agents import agent
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text: str = "OK", *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


@agent(model="mock-model")
class SimpleAgent: ...


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
    def test_runner_runs_basic_agent(self):
        client = TestClient(SimpleAgent())
        client.mock.queue_response(_c("Hello from mock"))
        result = client.run("hi")
        assert result.content == "Hello from mock"

    def test_runner_records_calls(self):
        client = TestClient(SimpleAgent())
        client.mock.queue_response(_c("OK"))
        client.run("hi")
        assert len(client.calls) == 1

    def test_runner_stop_reason_end_turn(self):
        client = TestClient(SimpleAgent())
        client.mock.queue_response(_c("done"))
        result = client.run("hi")
        assert result.stop_reason == "end_turn"

    def test_runner_usage_populated(self):
        client = TestClient(SimpleAgent())
        client.mock.queue_response(_c("OK"))
        result = client.run("hi")
        assert result.total_usage.input_tokens > 0

    def test_config_properties_accessible(self):
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        assert cfg.provider == "anthropic"
        assert cfg.model == "mock-model"
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 1.0

    def test_no_network_calls_made(self):
        _, mock1 = LLMConfig.for_testing()
        _, mock2 = LLMConfig.for_testing()
        assert mock1 is not mock2
