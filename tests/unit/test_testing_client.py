"""Unit tests for AgentTestClient."""
from __future__ import annotations

import pytest

from lauren_ai._agents import AgentResponse, agent
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import AgentTestClient


@pytest.fixture()
def simple_agent_and_client():
    """Return (agent_instance, AgentTestClient)."""
    @agent(model="mock-model")
    class EchoAgent:
        pass

    cfg, mock = LLMConfig.for_testing()
    mock.queue_response(
        Completion(
            id="c1", model="mock", content="Echo response",
            tool_calls=[], stop_reason="end_turn",
            usage=TokenUsage(input_tokens=5, output_tokens=3),
        )
    )
    instance = EchoAgent()
    client = AgentTestClient(instance, mock)
    return instance, client


class TestAgentTestClient:
    @pytest.mark.asyncio
    async def test_run_async(self, simple_agent_and_client):
        _, client = simple_agent_and_client
        response = await client.run_async("Hello!")
        assert isinstance(response, AgentResponse)
        assert response.content == "Echo response"

    def test_mock_property(self, simple_agent_and_client):
        _, client = simple_agent_and_client
        assert isinstance(client.mock, MockTransport)

    def test_reset(self, simple_agent_and_client):
        _, client = simple_agent_and_client
        client.reset()
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_calls_recorded(self, simple_agent_and_client):
        _, client = simple_agent_and_client
        # Queue a response first since we already consumed the initial one
        client.mock.queue_response(
            Completion(
                id="c2", model="mock", content="Another response",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=3, output_tokens=2),
            )
        )
        client.reset()  # Reset so calls list is fresh
        # Re-queue after reset
        client.mock.queue_response(
            Completion(
                id="c3", model="mock", content="Third response",
                tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=3, output_tokens=2),
            )
        )
        await client.run_async("ping")
        assert len(client.calls) == 1
