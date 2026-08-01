"""End-to-end compatibility workflow tests without network access."""

from __future__ import annotations

import pytest

from lauren_ai import AgentConfig, RequestOptions, agent
from lauren_ai._agents._runner import AgentRunnerBase
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


@agent(
    model="compatibility-model",
    config=AgentConfig(
        top_p=0.92,
        max_completion_tokens=256,
        request_options=RequestOptions(
            provider={"reasoning_effort": "none"},
            extra_body={"gateway_trace": True},
        ),
    ),
)
class CompatibilityAgent:
    """Agent used to verify the public provider-options workflow."""


@pytest.mark.asyncio
async def test_agent_run_propagates_provider_options_to_transport() -> None:
    """A complete agent run preserves provider-specific request settings."""
    transport = MockTransport()
    transport.queue_response(
        Completion(
            id="compatibility-completion",
            model="compatibility-model",
            content="The compatibility path is working.",
            tool_calls=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=8, output_tokens=6),
        )
    )
    runner = AgentRunnerBase(transport=transport)

    response = await runner.run(CompatibilityAgent(), "Verify the SDK compatibility path.")

    assert response.content == "The compatibility path is working."
    assert response.stop_reason == "end_turn"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call.model == "compatibility-model"
    assert call.top_p == 0.92
    assert call.max_completion_tokens == 256
    assert call.request_options is not None
    assert call.request_options.provider == {"reasoning_effort": "none"}
    assert call.request_options.extra_body == {"gateway_trace": True}
