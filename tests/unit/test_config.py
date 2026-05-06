"""Unit tests for LLMConfig and AgentConfig."""

from __future__ import annotations

import pytest

from lauren_ai._config import AgentConfig, LLMConfig
from lauren_ai._transport._mock import MockTransport


class TestLLMConfig:
    def test_for_anthropic_defaults(self):
        cfg = LLMConfig.for_anthropic()
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-opus-4-6"

    def test_for_anthropic_custom_model(self):
        cfg = LLMConfig.for_anthropic(model="claude-haiku-4-5", api_key="sk-test")
        assert cfg.model == "claude-haiku-4-5"
        assert cfg.api_key == "sk-test"

    def test_for_openai_defaults(self):
        cfg = LLMConfig.for_openai()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"

    def test_for_ollama_defaults(self):
        cfg = LLMConfig.for_ollama()
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3.2"
        assert cfg.base_url == "http://localhost:11434"

    def test_for_testing_returns_tuple(self):
        result = LLMConfig.for_testing()
        assert isinstance(result, tuple)
        assert len(result) == 2
        cfg, mock = result
        assert isinstance(cfg, LLMConfig)
        assert isinstance(mock, MockTransport)

    def test_frozen_raises_on_mutation(self):
        cfg = LLMConfig.for_anthropic()
        with pytest.raises((AttributeError, TypeError)):
            cfg.model = "changed"  # type: ignore[misc]

    def test_temperature_default(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
        assert cfg.temperature == 1.0

    def test_max_tokens_default(self):
        cfg = LLMConfig(provider="anthropic", model="claude-opus-4-6")
        assert cfg.max_tokens == 4096


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_turns == 10
        assert cfg.temperature == 1.0
        assert cfg.tool_error_policy == "return_error"
        assert cfg.thinking is False

    def test_custom_max_turns(self):
        cfg = AgentConfig(max_turns=5)
        assert cfg.max_turns == 5

    def test_frozen_raises_on_mutation(self):
        cfg = AgentConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_turns = 99  # type: ignore[misc]

    def test_thinking_config(self):
        cfg = AgentConfig(thinking=True, thinking_budget_tokens=12000)
        assert cfg.thinking is True
        assert cfg.thinking_budget_tokens == 12000
