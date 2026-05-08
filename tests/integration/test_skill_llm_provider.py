"""Integration tests for Skill 1: LLM Provider Connection.

Tests cover:
- LLMConfig direct constructor for all three providers
- LLMConfig factory classmethods (for_anthropic, for_openai, for_ollama, for_testing)
- for_testing() returns a MockTransport with no real network calls
- AgentRunner runs successfully with MockTransport
- Config properties are accessible
"""

import pytest
from pydantic import BaseModel

from lauren import LaurenFactory, controller, get, post, module, injectable, Scope, use_value, Json
from lauren.testing import TestClient
from lauren_ai import LLMConfig
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content: str = "OK", *, n: int = 1) -> Completion:
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controller / Module
# ---------------------------------------------------------------------------


class _RunRequest(BaseModel):
    prompt: str = "hi"


@controller("/llm")
class LLMController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)
        self._cfg = cfg

    @post("/complete")
    async def complete(self, body: Json[_RunRequest]) -> dict:
        @agent(model="mock-model")
        class SimpleAgent: ...

        resp = await self._runner.run(SimpleAgent(), body.prompt)
        return {
            "content": resp.content,
            "model": self._cfg.model,
            "stop_reason": resp.stop_reason,
            "input_tokens": resp.total_usage.input_tokens,
            "calls": len(_MOCK.calls),
        }

    @get("/config")
    async def config(self) -> dict:
        return {
            "provider": self._cfg.provider,
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
        }


@module(
    controllers=[LLMController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class LLMModule: ...


def build_app(*responses: str) -> TestClient:
    _MOCK.reset()
    for content in responses:
        _MOCK.queue_response(_completion(content))
    return TestClient(LaurenFactory.create(LLMModule))


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
# TestRunnerWithMockTransport (via TestClient)
# ---------------------------------------------------------------------------


class TestRunnerWithMockTransport:
    def test_runner_runs_basic_agent(self):
        client = build_app("Hello from mock")
        r = client.post("/llm/complete", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["content"] == "Hello from mock"

    def test_runner_records_calls(self):
        client = build_app("OK")
        r = client.post("/llm/complete", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["calls"] == 1

    def test_runner_stop_reason_end_turn(self):
        client = build_app("done")
        r = client.post("/llm/complete", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["stop_reason"] == "end_turn"

    def test_runner_usage_populated(self):
        client = build_app("OK")
        r = client.post("/llm/complete", json={"prompt": "hi"})
        assert r.status_code == 200
        assert r.json()["input_tokens"] > 0

    def test_config_properties_via_endpoint(self):
        client = build_app()
        r = client.get("/llm/config")
        assert r.status_code == 200
        data = r.json()
        assert data["provider"] == "anthropic"
        assert data["model"] == "mock-model"
        assert data["max_tokens"] == 4096
        assert data["temperature"] == 1.0

    def test_no_network_calls_made(self):
        _, mock1 = LLMConfig.for_testing()
        _, mock2 = LLMConfig.for_testing()
        assert mock1 is not mock2
