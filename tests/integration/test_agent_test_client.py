"""Integration tests for the AgentTestClient helper.

Tests cover:
- Creating an agent and running it through AgentTestClient
- Checking response content and stop_reason
- Verifying mock.calls tracking
- Testing reset() clears call history and response queue
- Testing synchronous run() and async run_async()
- Testing with tools attached
"""

import pytest

from lauren_ai._agents import agent, use_tools
from lauren_ai._tools import tool
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport
from lauren_ai.testing import AgentTestClient

# ---------------------------------------------------------------------------
# Tool definitions (no from __future__ import annotations)
# ---------------------------------------------------------------------------


@tool()
async def double_tool(value: int) -> int:
    """Double the given integer value.

    Args:
        value: The integer to double.
    """
    return value * 2


# ---------------------------------------------------------------------------
# Agent class definitions
# ---------------------------------------------------------------------------


@agent(model="mock-model", system="You are a test assistant.")
class BasicTestAgent:
    pass


@agent(model="mock-model", system="You are a tool-using test agent.")
@use_tools(double_tool)
class ToolTestAgent:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simple_completion(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Tests: basic run
# ---------------------------------------------------------------------------


class TestAgentTestClientBasicRun:
    def test_sync_run_returns_agent_response(self):
        """sync run() returns a populated AgentResponse."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Sync response!"))

        client = AgentTestClient(BasicTestAgent(), mock)
        response = client.run("Hello sync!")

        assert response.content == "Sync response!"
        assert response.stop_reason == "end_turn"
        assert response.turns == 1

    @pytest.mark.asyncio
    async def test_async_run_returns_agent_response(self):
        """async run_async() returns a populated AgentResponse."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Async response!"))

        client = AgentTestClient(BasicTestAgent(), mock)
        response = await client.run_async("Hello async!")

        assert response.content == "Async response!"
        assert response.stop_reason == "end_turn"
        assert response.turns == 1

    @pytest.mark.asyncio
    async def test_run_async_with_metadata(self):
        """run_async() accepts optional metadata dict."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Meta response."))

        client = AgentTestClient(BasicTestAgent(), mock)
        response = await client.run_async("Hello", metadata={"user": "alice"})

        assert response.content == "Meta response."

    @pytest.mark.asyncio
    async def test_run_async_with_conversation_id(self):
        """run_async() accepts an optional conversation_id."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Conv response."))

        client = AgentTestClient(BasicTestAgent(), mock)
        response = await client.run_async("Hello", conversation_id="session-42")

        assert response.content == "Conv response."


# ---------------------------------------------------------------------------
# Tests: mock.calls tracking
# ---------------------------------------------------------------------------


class TestAgentTestClientCallTracking:
    @pytest.mark.asyncio
    async def test_calls_recorded_after_run(self):
        """mock.calls and client.calls are populated after a run."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Tracked!"))

        client = AgentTestClient(BasicTestAgent(), mock)
        await client.run_async("Track me")

        assert len(client.calls) == 1
        assert len(mock.calls) == 1

    @pytest.mark.asyncio
    async def test_calls_accumulate_across_runs(self):
        """Consecutive runs accumulate calls until reset()."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Run 1"))
        mock.queue_response(simple_completion("Run 2"))

        client = AgentTestClient(BasicTestAgent(), mock)
        await client.run_async("First message")
        await client.run_async("Second message")

        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_calls_have_correct_messages(self):
        """CompletionCall records contain the messages passed to the transport."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Got it."))

        client = AgentTestClient(BasicTestAgent(), mock)
        await client.run_async("Tell me something")

        call = client.calls[0]
        # The first message in the call should be the user message
        assert len(call.messages) >= 1
        user_msgs = [m for m in call.messages if getattr(m, "role", None) == "user" or
                     (isinstance(m, dict) and m.get("role") == "user")]
        assert len(user_msgs) >= 1

    @pytest.mark.asyncio
    async def test_two_calls_for_tool_use_run(self):
        """A tool-use run results in two transport calls tracked."""
        mock = MockTransport()
        mock.queue_tool_use("double_tool", {"value": 21})
        mock.queue_response(simple_completion("Doubled: 42.", id="c2"))

        client = AgentTestClient(ToolTestAgent(), mock)
        await client.run_async("Double 21")

        assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestAgentTestClientReset:
    @pytest.mark.asyncio
    async def test_reset_clears_call_history(self):
        """After reset(), calls is empty."""
        mock = MockTransport()
        mock.queue_response(simple_completion("Before reset."))

        client = AgentTestClient(BasicTestAgent(), mock)
        await client.run_async("Run before reset")

        assert len(client.calls) == 1

        client.reset()
        assert len(client.calls) == 0

    @pytest.mark.asyncio
    async def test_reset_clears_response_queue(self):
        """After reset(), queued responses are discarded."""
        from lauren_ai._exceptions import EmptyQueueError

        mock = MockTransport()
        mock.queue_response(simple_completion("Will be discarded."))

        client = AgentTestClient(BasicTestAgent(), mock)
        client.reset()

        # Attempting to run now should raise EmptyQueueError since queue was cleared
        with pytest.raises(EmptyQueueError):
            await client.run_async("This should fail")

    @pytest.mark.asyncio
    async def test_reset_allows_fresh_start(self):
        """After reset(), a new queue allows a fresh run."""
        mock = MockTransport()
        mock.queue_response(simple_completion("First run."))

        client = AgentTestClient(BasicTestAgent(), mock)
        await client.run_async("First")

        client.reset()
        mock.queue_response(simple_completion("Second run."))

        response = await client.run_async("Second")
        assert response.content == "Second run."
        assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# Tests: mock property
# ---------------------------------------------------------------------------


class TestAgentTestClientMockProperty:
    def test_mock_property_returns_mock_transport(self):
        """client.mock returns the MockTransport instance."""
        mock = MockTransport()
        client = AgentTestClient(BasicTestAgent(), mock)
        assert client.mock is mock

    def test_mock_property_allows_queuing(self):
        """Queuing via client.mock works the same as queuing via the original mock."""
        mock = MockTransport()
        client = AgentTestClient(BasicTestAgent(), mock)

        client.mock.queue_response(simple_completion("From mock property."))

        response = client.run("Test via mock property")
        assert response.content == "From mock property."


# ---------------------------------------------------------------------------
# Tests: tool-using agent via AgentTestClient
# ---------------------------------------------------------------------------


class TestAgentTestClientWithTools:
    @pytest.mark.asyncio
    async def test_tool_call_tracked_in_response(self):
        """Tool calls made during the run are tracked in the response."""
        mock = MockTransport()
        mock.queue_tool_use("double_tool", {"value": 5})
        mock.queue_response(simple_completion("Double 5 is 10.", id="c2"))

        client = AgentTestClient(ToolTestAgent(), mock)
        response = await client.run_async("Double 5")

        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0].name == "double_tool"
        assert response.tool_calls_made[0].input == {"value": 5}

    @pytest.mark.asyncio
    async def test_tool_result_included_in_model_context(self):
        """After a tool call, the second model invocation receives more messages."""
        mock = MockTransport()
        mock.queue_tool_use("double_tool", {"value": 7})
        mock.queue_response(simple_completion("7 doubled is 14.", id="c2"))

        client = AgentTestClient(ToolTestAgent(), mock)
        await client.run_async("Double 7")

        # Second call should have more messages (including the tool result)
        assert len(client.calls) == 2
        first_call_msgs = len(client.calls[0].messages)
        second_call_msgs = len(client.calls[1].messages)
        assert second_call_msgs > first_call_msgs

    def test_sync_run_with_tool(self):
        """Synchronous run() works correctly with a tool-using agent."""
        mock = MockTransport()
        mock.queue_tool_use("double_tool", {"value": 3})
        mock.queue_response(simple_completion("3 doubled is 6.", id="c2"))

        client = AgentTestClient(ToolTestAgent(), mock)
        response = client.run("Double 3")

        assert response.stop_reason == "end_turn"
        assert len(response.tool_calls_made) == 1

    @pytest.mark.asyncio
    async def test_run_multiple_times_with_reset_between(self):
        """Multiple isolated runs using reset() each work correctly."""
        mock = MockTransport()
        client = AgentTestClient(ToolTestAgent(), mock)

        # First isolated run
        mock.queue_tool_use("double_tool", {"value": 10})
        mock.queue_response(simple_completion("10 doubled is 20.", id="c2"))
        response1 = await client.run_async("Double 10")
        assert response1.stop_reason == "end_turn"
        assert len(client.calls) == 2

        client.reset()

        # Second isolated run
        mock.queue_response(simple_completion("Simple answer.", id="c1"))
        response2 = await client.run_async("Simple question")
        assert response2.content == "Simple answer."
        assert len(client.calls) == 1  # fresh count after reset


# ---------------------------------------------------------------------------
# Tests: _build_runner edge cases (coverage for None tool_ref and failed
# registration)
# ---------------------------------------------------------------------------


class TestAgentTestClientBuildRunnerEdgeCases:
    def test_none_tool_ref_in_meta_is_skipped(self):
        """None entries in tool_classes are silently skipped (line 176 coverage)."""
        from dataclasses import replace

        from lauren_ai._agents import AGENT_META, AgentMeta
        from lauren_ai._agents._runner import AgentRunner

        @agent(model="mock-model", system="Edge case agent.")
        class EdgeAgent:
            pass

        # Inject a None into tool_classes on a fresh copy of the meta.
        original_meta: AgentMeta = getattr(EdgeAgent, AGENT_META)
        patched_meta = replace(original_meta, tool_classes=(None,))
        setattr(EdgeAgent, AGENT_META, patched_meta)

        try:
            mock = MockTransport()
            mock.queue_response(simple_completion("Edge!"))
            client = AgentTestClient(EdgeAgent(), mock)
            assert isinstance(client._runner, AgentRunner)
        finally:
            setattr(EdgeAgent, AGENT_META, original_meta)

    def test_build_runner_tolerates_no_tool_meta(self):
        """AgentTestClient builds a runner even when agent has no tools."""
        from lauren_ai._agents._runner import AgentRunner

        @agent()
        class BrokenToolAgent:
            pass

        mock = MockTransport()
        mock.queue_response(simple_completion("OK!"))
        client = AgentTestClient(BrokenToolAgent(), mock)
        assert isinstance(client._runner, AgentRunner)
