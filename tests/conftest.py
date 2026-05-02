"""Shared pytest fixtures for lauren-ai tests."""
from __future__ import annotations

import pytest

from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, Message, TokenUsage
from lauren_ai._transport._mock import MockTransport


@pytest.fixture()
def mock_transport() -> MockTransport:
    """Return a fresh MockTransport for each test."""
    return MockTransport()


@pytest.fixture()
def llm_config() -> LLMConfig:
    """Return a test LLMConfig (no real API calls)."""
    config, _ = LLMConfig.for_testing()
    return config


@pytest.fixture()
def test_llm_pair() -> tuple[LLMConfig, MockTransport]:
    """Return (LLMConfig, MockTransport) test pair."""
    return LLMConfig.for_testing()


@pytest.fixture()
def simple_completion() -> Completion:
    """Return a minimal Completion for use in mock queues."""
    return Completion(
        id="test-1",
        model="mock-model",
        content="Hello from mock!",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


@pytest.fixture()
def user_message() -> Message:
    """Return a simple user Message."""
    return Message(role="user", content="Hello!")
