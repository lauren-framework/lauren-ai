"""Extended unit tests for the exception hierarchy — covers all __str__ methods."""

from __future__ import annotations

import pytest

from lauren_ai._exceptions import (
    AgentBudgetExceededError,
    AgentConfigError,
    AgentMaxTurnsError,
    AuthTransportError,
    DecoratorUsageError,
    EmptyQueueError,
    EvalError,
    KnowledgeLoadError,
    LaurenAIError,
    ToolConfigError,
    ToolConfirmationRejectedError,
    ToolExecutionError,
    ToolSchemaError,
    TransientTransportError,
    TransportError,
    WorkflowError,
)


class TestLaurenAIErrorStr:
    def test_str_without_cause(self):
        exc = LaurenAIError("base error")
        assert str(exc) == "base error"

    def test_str_with_cause(self):
        cause = ValueError("underlying")
        exc = LaurenAIError("base error", cause=cause)
        s = str(exc)
        assert "base error" in s
        assert "caused by" in s

    def test_cause_set_as_dunder(self):
        cause = ValueError("root")
        exc = LaurenAIError("msg", cause=cause)
        assert exc.__cause__ is cause

    def test_no_cause(self):
        exc = LaurenAIError("msg")
        assert exc.cause is None
        assert exc.__cause__ is None


class TestTransportErrorStr:
    def test_str_with_all_fields(self):
        exc = TransportError("fail", status_code=503, provider="anthropic", cause=RuntimeError("x"))
        s = str(exc)
        assert "fail" in s
        assert "provider='anthropic'" in s
        assert "status_code=503" in s
        assert "caused by" in s

    def test_str_minimal(self):
        exc = TransportError("fail")
        assert str(exc) == "fail"

    def test_str_provider_only(self):
        exc = TransportError("fail", provider="openai")
        s = str(exc)
        assert "provider='openai'" in s
        assert "status_code" not in s

    def test_str_status_only(self):
        exc = TransportError("fail", status_code=429)
        s = str(exc)
        assert "status_code=429" in s
        assert "provider" not in s


class TestTransientTransportErrorStr:
    def test_str_with_retry_after(self):
        exc = TransientTransportError("Rate limited", status_code=429, retry_after=5.0, provider="anthropic")
        s = str(exc)
        assert "retry_after=5.0s" in s
        assert "status_code=429" in s
        assert "provider='anthropic'" in s

    def test_str_without_retry_after(self):
        exc = TransientTransportError("Rate limited", status_code=429)
        s = str(exc)
        assert "retry_after" not in s

    def test_retry_after_attribute(self):
        exc = TransientTransportError("x", retry_after=10.5)
        assert exc.retry_after == pytest.approx(10.5)

    def test_with_cause(self):
        exc = TransientTransportError("x", cause=ValueError("y"))
        s = str(exc)
        assert "caused by" in s


class TestAuthTransportError:
    def test_inherits_transport_error(self):
        exc = AuthTransportError("Unauthorized", status_code=401, provider="anthropic")
        assert isinstance(exc, TransportError)
        assert exc.status_code == 401
        assert exc.provider == "anthropic"


class TestToolExecutionErrorStr:
    def test_str_with_cause(self):
        exc = ToolExecutionError("boom", tool_name="my_tool", tool_use_id="tc1", cause=ValueError("bad input"))
        s = str(exc)
        assert "my_tool" in s
        assert "tc1" in s
        assert "caused by" in s

    def test_str_without_cause(self):
        exc = ToolExecutionError("boom", tool_name="tool_a", tool_use_id="id1")
        s = str(exc)
        assert "tool_a" in s
        assert "id1" in s
        assert "caused by" not in s

    def test_attributes(self):
        exc = ToolExecutionError("x", tool_name="t", tool_use_id="u")
        assert exc.tool_name == "t"
        assert exc.tool_use_id == "u"


class TestToolSchemaErrorStr:
    def test_str_with_tool_and_param(self):
        exc = ToolSchemaError("bad schema", tool_name="my_tool", parameter="x")
        s = str(exc)
        assert "bad schema" in s
        assert "tool='my_tool'" in s
        assert "parameter='x'" in s

    def test_str_with_tool_only(self):
        exc = ToolSchemaError("bad schema", tool_name="my_tool")
        s = str(exc)
        assert "tool='my_tool'" in s
        assert "parameter" not in s

    def test_str_with_param_only(self):
        exc = ToolSchemaError("bad schema", parameter="q")
        s = str(exc)
        assert "parameter='q'" in s

    def test_str_no_extras(self):
        exc = ToolSchemaError("bad schema")
        assert str(exc) == "bad schema"


class TestToolConfigErrorStr:
    def test_str_with_tool_name(self):
        exc = ToolConfigError("bad config", tool_name="my_tool")
        s = str(exc)
        assert "my_tool" in s
        assert "bad config" in s

    def test_str_without_tool_name(self):
        exc = ToolConfigError("bad config")
        s = str(exc)
        assert "Tool config error:" in s
        assert "bad config" in s

    def test_attributes(self):
        exc = ToolConfigError("x", tool_name="t")
        assert exc.tool_name == "t"


class TestAgentMaxTurnsErrorStr:
    def test_str_with_agent_class(self):
        class MyAgent:
            pass

        exc = AgentMaxTurnsError("exceeded", turns=5, agent_class=MyAgent)
        s = str(exc)
        assert "MyAgent" in s
        assert "5" in s

    def test_str_without_agent_class(self):
        exc = AgentMaxTurnsError("exceeded", turns=3)
        s = str(exc)
        assert "unknown" in s
        assert "3" in s

    def test_attributes(self):
        exc = AgentMaxTurnsError("x", turns=7)
        assert exc.turns == 7
        assert exc.agent_class is None


class TestAgentBudgetExceededErrorStr:
    def test_str_with_agent_class(self):
        class BudgetAgent:
            pass

        exc = AgentBudgetExceededError(
            "over budget", budget_type="cost_usd", limit=0.5, used=0.7, agent_class=BudgetAgent
        )
        s = str(exc)
        assert "BudgetAgent" in s
        assert "cost_usd" in s
        assert "0.7" in s
        assert "0.5" in s

    def test_str_without_agent_class(self):
        exc = AgentBudgetExceededError("over budget", budget_type="tokens", limit=1000.0, used=1200.0)
        s = str(exc)
        assert "unknown" in s
        assert "tokens" in s

    def test_attributes(self):
        exc = AgentBudgetExceededError("x", budget_type="cost_usd", limit=1.0, used=2.0)
        assert exc.budget_type == "cost_usd"
        assert exc.limit == pytest.approx(1.0)
        assert exc.used == pytest.approx(2.0)


class TestAgentConfigErrorStr:
    def test_str_with_agent_class(self):
        class BadAgent:
            pass

        exc = AgentConfigError("misconfigured", agent_class=BadAgent)
        s = str(exc)
        assert "BadAgent" in s
        assert "misconfigured" in s

    def test_str_without_agent_class(self):
        exc = AgentConfigError("misconfigured")
        s = str(exc)
        assert "unknown" in s

    def test_with_cause(self):
        exc = AgentConfigError("bad", cause=ValueError("root"))
        assert exc.cause is not None


class TestDecoratorUsageErrorStr:
    def test_str_with_name(self):
        exc = DecoratorUsageError("use parens", decorator_name="tool")
        s = str(exc)
        assert "@tool" in s
        assert "use parens" in s

    def test_str_without_name(self):
        exc = DecoratorUsageError("use parens")
        s = str(exc)
        assert "Decorator misuse:" in s

    def test_attribute(self):
        exc = DecoratorUsageError("x", decorator_name="agent")
        assert exc.decorator_name == "agent"


class TestEmptyQueueErrorStr:
    def test_default_message(self):
        exc = EmptyQueueError()
        assert "empty" in str(exc).lower() or "MockTransport" in str(exc)

    def test_custom_message(self):
        exc = EmptyQueueError("custom msg")
        assert "custom msg" in str(exc)


class TestToolConfirmationRejectedErrorStr:
    def test_str_with_reason(self):
        exc = ToolConfirmationRejectedError(
            "rejected", tool_name="delete_file", tool_use_id="tc1", reason="too dangerous"
        )
        s = str(exc)
        assert "delete_file" in s
        assert "tc1" in s
        assert "too dangerous" in s

    def test_str_without_reason(self):
        exc = ToolConfirmationRejectedError("rejected", tool_name="delete_file", tool_use_id="tc1")
        s = str(exc)
        assert "delete_file" in s
        # reason should not appear
        assert "Reason" not in s

    def test_attributes(self):
        exc = ToolConfirmationRejectedError("x", tool_name="t", tool_use_id="u", reason="r")
        assert exc.tool_name == "t"
        assert exc.tool_use_id == "u"
        assert exc.reason == "r"


class TestKnowledgeLoadErrorStr:
    def test_str_with_source(self):
        exc = KnowledgeLoadError("cannot load", source="/path/to/file.pdf")
        s = str(exc)
        assert "/path/to/file.pdf" in s
        assert "cannot load" in s

    def test_str_without_source(self):
        exc = KnowledgeLoadError("cannot load")
        s = str(exc)
        assert "Knowledge load error:" in s

    def test_with_cause(self):
        exc = KnowledgeLoadError("x", cause=FileNotFoundError("missing"))
        assert exc.cause is not None
        assert exc.source is None


class TestWorkflowErrorStr:
    def test_str_with_step(self):
        exc = WorkflowError("step failed", step_name="summarise")
        s = str(exc)
        assert "summarise" in s
        assert "step failed" in s

    def test_str_without_step(self):
        exc = WorkflowError("step failed")
        s = str(exc)
        assert "Workflow error:" in s

    def test_with_cause(self):
        exc = WorkflowError("x", cause=RuntimeError("root"))
        assert exc.cause is not None
        assert exc.step_name is None


class TestEvalErrorStr:
    def test_str_with_eval_name(self):
        exc = EvalError("failed", eval_name="accuracy_eval")
        s = str(exc)
        assert "accuracy_eval" in s
        assert "failed" in s

    def test_str_without_eval_name(self):
        exc = EvalError("failed")
        s = str(exc)
        assert "Eval error:" in s

    def test_with_cause(self):
        exc = EvalError("x", cause=AssertionError("y"))
        assert exc.cause is not None

    def test_attributes(self):
        exc = EvalError("x", eval_name="my_eval")
        assert exc.eval_name == "my_eval"
