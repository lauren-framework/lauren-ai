"""Unit tests for the exception hierarchy."""

from __future__ import annotations

from lauren_ai._exceptions import (
    AgentBudgetExceededError,
    AgentMaxTurnsError,
    AuthTransportError,
    DecoratorUsageError,
    EmptyQueueError,
    KnowledgeLoadError,
    LaurenAIError,
    ToolExecutionError,
    TransientTransportError,
    TransportError,
    WorkflowError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        exc = LaurenAIError("Something went wrong")
        assert isinstance(exc, Exception)
        assert "Something went wrong" in str(exc)

    def test_transport_error_inherits_base(self):
        exc = TransportError("Connection failed", status_code=500)
        assert isinstance(exc, LaurenAIError)
        assert exc.status_code == 500

    def test_transient_inherits_transport(self):
        exc = TransientTransportError("Rate limited", status_code=429)
        assert isinstance(exc, TransportError)
        assert isinstance(exc, LaurenAIError)

    def test_auth_error(self):
        exc = AuthTransportError("Unauthorized", status_code=401)
        assert isinstance(exc, TransportError)

    def test_tool_execution_error(self):
        exc = ToolExecutionError("Tool failed", tool_name="my_tool", tool_use_id="tc1")
        assert isinstance(exc, LaurenAIError)
        assert exc.tool_name == "my_tool"

    def test_agent_max_turns_error(self):
        exc = AgentMaxTurnsError("Too many turns", turns=10)
        assert isinstance(exc, LaurenAIError)

    def test_agent_budget_exceeded(self):
        exc = AgentBudgetExceededError("Over budget", budget_type="tokens", limit=1000.0, used=1200.0)
        assert isinstance(exc, LaurenAIError)
        assert exc.budget_type == "tokens"

    def test_decorator_usage_error(self):
        exc = DecoratorUsageError("Use @tool()", decorator_name="tool")
        assert isinstance(exc, LaurenAIError)
        assert exc.decorator_name == "tool"

    def test_empty_queue_error(self):
        exc = EmptyQueueError("Queue is empty")
        assert isinstance(exc, LaurenAIError)

    def test_knowledge_load_error(self):
        exc = KnowledgeLoadError("Cannot read file", source="/path/to/file")
        assert isinstance(exc, LaurenAIError)

    def test_workflow_error(self):
        exc = WorkflowError("Step failed", step_name="summarise")
        assert isinstance(exc, LaurenAIError)
