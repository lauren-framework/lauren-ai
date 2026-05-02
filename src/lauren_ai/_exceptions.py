"""Exception hierarchy for ``lauren-ai``.

All exceptions inherit from :class:`LaurenAIError`, which itself inherits
from the built-in :class:`Exception`.  The hierarchy is:

.. code-block:: text

    LaurenAIError
    ├── TransportError
    │   ├── TransientTransportError
    │   └── AuthTransportError
    ├── ToolExecutionError
    ├── ToolSchemaError
    ├── ToolConfigError
    ├── AgentMaxTurnsError
    ├── AgentBudgetExceededError
    ├── AgentConfigError
    ├── DecoratorUsageError
    ├── DelegateToAgent
    ├── EmptyQueueError
    ├── ToolConfirmationRejectedError
    ├── KnowledgeLoadError
    ├── WorkflowError
    ├── OutputParserError
    └── EvalError
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LaurenAIError",
    "TransportError",
    "TransientTransportError",
    "AuthTransportError",
    "ToolExecutionError",
    "ToolSchemaError",
    "ToolConfigError",
    "AgentMaxTurnsError",
    "AgentBudgetExceededError",
    "AgentConfigError",
    "DecoratorUsageError",
    "DelegateToAgent",
    "EmptyQueueError",
    "ToolConfirmationRejectedError",
    "KnowledgeLoadError",
    "WorkflowError",
    "OutputParserError",
    "EvalError",
    "TracingError",
]


class LaurenAIError(Exception):
    """Base class for all ``lauren-ai`` exceptions.

    :param message: Human-readable description of what went wrong.
    :type message: str
    :param cause: The underlying exception that caused this error, if any.
    :type cause: BaseException | None
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        """Initialise the error.

        :param message: Human-readable description of what went wrong.
        :type message: str
        :param cause: The underlying exception that caused this error, if any.
        :type cause: BaseException | None
        """
        super().__init__(message)
        self.message: str = message
        self.cause: BaseException | None = cause
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return a string representation of the error.

        :return: The error message, optionally followed by the cause.
        :rtype: str
        """
        if self.cause is not None:
            return f"{self.message} (caused by: {self.cause!r})"
        return self.message


# ---------------------------------------------------------------------------
# Transport layer
# ---------------------------------------------------------------------------


class TransportError(LaurenAIError):
    """Raised when an LLM provider returns or raises any error.

    :param message: Human-readable description of the transport failure.
    :type message: str
    :param status_code: HTTP status code returned by the provider, if applicable.
    :type status_code: int | None
    :param provider: Name of the provider (e.g. ``"anthropic"``).
    :type provider: str | None
    :param cause: The underlying exception from the provider SDK.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.status_code: int | None = status_code
        self.provider: str | None = provider

    def __str__(self) -> str:
        parts: list[str] = [self.message]
        if self.provider is not None:
            parts.append(f"provider={self.provider!r}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.cause is not None:
            parts.append(f"caused by: {self.cause!r}")
        return " | ".join(parts)


class TransientTransportError(TransportError):
    """Raised for retryable transport failures: rate limits (429) and server errors (5xx).

    The :class:`~lauren_ai._config.LLMConfig` ``max_retries`` setting controls
    how many times the transport layer will retry before re-raising this exception.

    :param message: Human-readable description of the transient failure.
    :type message: str
    :param status_code: HTTP status code (e.g. ``429``, ``529``, ``500``).
    :type status_code: int | None
    :param retry_after: Number of seconds to wait before retrying, if provided
        by the provider (e.g. ``Retry-After`` header).
    :type retry_after: float | None
    :param provider: Name of the provider.
    :type provider: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        provider: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            provider=provider,
            cause=cause,
        )
        self.retry_after: float | None = retry_after

    def __str__(self) -> str:
        parts: list[str] = [self.message]
        if self.provider is not None:
            parts.append(f"provider={self.provider!r}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.retry_after is not None:
            parts.append(f"retry_after={self.retry_after}s")
        if self.cause is not None:
            parts.append(f"caused by: {self.cause!r}")
        return " | ".join(parts)


class AuthTransportError(TransportError):
    """Raised for authentication / authorisation failures (HTTP 401 or 403).

    This error is **never retried** — a missing or invalid API key will not
    become valid on a subsequent attempt.

    :param message: Human-readable description of the auth failure.
    :type message: str
    :param status_code: HTTP status code (``401`` or ``403``).
    :type status_code: int | None
    :param provider: Name of the provider.
    :type provider: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------


class ToolExecutionError(LaurenAIError):
    """Raised when a tool raises an unexpected exception during execution.

    :param message: Human-readable description of the failure.
    :type message: str
    :param tool_name: The registered name of the tool that failed.
    :type tool_name: str
    :param tool_use_id: The provider-assigned identifier for this tool call.
    :type tool_use_id: str
    :param cause: The original exception raised by the tool.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        tool_use_id: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.tool_name: str = tool_name
        self.tool_use_id: str = tool_use_id

    def __str__(self) -> str:
        base = f"Tool {self.tool_name!r} (id={self.tool_use_id!r}): {self.message}"
        if self.cause is not None:
            base += f" (caused by: {self.cause!r})"
        return base


class ToolSchemaError(LaurenAIError):
    """Raised at startup when a tool's JSON schema cannot be generated.

    This typically means a parameter type annotation is not supported
    (e.g. an unannotated parameter or a type that cannot be serialised to
    JSON Schema).

    :param message: Human-readable description of the schema error.
    :type message: str
    :param tool_name: The name of the tool with the bad schema.
    :type tool_name: str | None
    :param parameter: The name of the offending parameter, if known.
    :type parameter: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        parameter: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.tool_name: str | None = tool_name
        self.parameter: str | None = parameter

    def __str__(self) -> str:
        parts: list[str] = []
        if self.tool_name is not None:
            parts.append(f"tool={self.tool_name!r}")
        if self.parameter is not None:
            parts.append(f"parameter={self.parameter!r}")
        if parts:
            return f"{self.message} ({', '.join(parts)})"
        return self.message


class ToolConfigError(LaurenAIError):
    """Raised at startup when a ``@tool()`` decorator is misconfigured.

    :param message: Human-readable description of the configuration error.
    :type message: str
    :param tool_name: The name of the offending tool, if known.
    :type tool_name: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.tool_name: str | None = tool_name

    def __str__(self) -> str:
        if self.tool_name is not None:
            return f"Tool config error for {self.tool_name!r}: {self.message}"
        return f"Tool config error: {self.message}"


# ---------------------------------------------------------------------------
# Agent layer
# ---------------------------------------------------------------------------


class AgentMaxTurnsError(LaurenAIError):
    """Raised when an agent exceeds its configured ``max_turns`` limit.

    :param message: Human-readable description of the limit exceeded.
    :type message: str
    :param turns: The number of turns that were executed before stopping.
    :type turns: int
    :param agent_class: The agent class that exceeded the limit.
    :type agent_class: type | None
    """

    def __init__(
        self,
        message: str,
        *,
        turns: int,
        agent_class: type | None = None,
    ) -> None:
        super().__init__(message)
        self.turns: int = turns
        self.agent_class: type | None = agent_class

    def __str__(self) -> str:
        agent_name = (
            self.agent_class.__name__ if self.agent_class is not None else "unknown"
        )
        return (
            f"Agent {agent_name!r} exceeded max_turns after {self.turns} turns: "
            f"{self.message}"
        )


class AgentBudgetExceededError(LaurenAIError):
    """Raised mid-run when an agent crosses its ``max_cost_usd`` or token budget.

    :param message: Human-readable description of the budget exceeded.
    :type message: str
    :param budget_type: Either ``"cost_usd"`` or ``"tokens"``.
    :type budget_type: str
    :param limit: The configured budget limit.
    :type limit: float
    :param used: The actual amount used when the budget was exceeded.
    :type used: float
    :param agent_class: The agent class that exceeded the budget.
    :type agent_class: type | None
    """

    def __init__(
        self,
        message: str,
        *,
        budget_type: str,
        limit: float,
        used: float,
        agent_class: type | None = None,
    ) -> None:
        super().__init__(message)
        self.budget_type: str = budget_type
        self.limit: float = limit
        self.used: float = used
        self.agent_class: type | None = agent_class

    def __str__(self) -> str:
        agent_name = (
            self.agent_class.__name__ if self.agent_class is not None else "unknown"
        )
        return (
            f"Agent {agent_name!r} exceeded {self.budget_type} budget "
            f"({self.used} / {self.limit}): {self.message}"
        )


class AgentConfigError(LaurenAIError):
    """Raised at startup when an ``@agent()`` decorator is misconfigured.

    :param message: Human-readable description of the configuration error.
    :type message: str
    :param agent_class: The offending agent class, if known.
    :type agent_class: type | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        agent_class: type | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.agent_class: type | None = agent_class

    def __str__(self) -> str:
        agent_name = (
            self.agent_class.__name__ if self.agent_class is not None else "unknown"
        )
        return f"Agent config error for {agent_name!r}: {self.message}"


# ---------------------------------------------------------------------------
# Decorator usage guard
# ---------------------------------------------------------------------------


class DecoratorUsageError(LaurenAIError):
    """Raised when a decorator is used incorrectly, e.g. bare ``@tool`` without parentheses.

    :param message: Human-readable description of the misuse.
    :type message: str
    :param decorator_name: The name of the decorator that was misused.
    :type decorator_name: str | None
    """

    def __init__(
        self,
        message: str,
        *,
        decorator_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.decorator_name: str | None = decorator_name

    def __str__(self) -> str:
        if self.decorator_name is not None:
            return f"Decorator @{self.decorator_name} misuse: {self.message}"
        return f"Decorator misuse: {self.message}"


# ---------------------------------------------------------------------------
# Multi-agent handoff (internal control-flow exception)
# ---------------------------------------------------------------------------


class DelegateToAgent(LaurenAIError):
    """Internal exception used to request a multi-agent handoff.

    An agent's hook method raises this to instruct the runner to transfer
    control to another agent.  This is **not** a fatal error — the runner
    catches it and performs the delegation.

    :param agent: The agent instance or class to delegate to.
    :type agent: Any
    :param message: The message to pass to the target agent.
    :type message: str
    """

    def __init__(self, agent: Any, message: str = "") -> None:
        """Initialise the delegation request.

        :param agent: The agent instance or class to delegate to.
        :type agent: Any
        :param message: The message to pass to the target agent.
        :type message: str
        """
        super().__init__(f"Delegate to {agent!r}: {message}" if message else f"Delegate to {agent!r}")
        self.agent: Any = agent
        self.message: str = message

    def __str__(self) -> str:
        agent_name = (
            self.agent.__name__
            if isinstance(self.agent, type)
            else type(self.agent).__name__
        )
        if self.message:
            return f"Handoff to {agent_name!r} with message: {self.message!r}"
        return f"Handoff to {agent_name!r}"


# ---------------------------------------------------------------------------
# MockTransport
# ---------------------------------------------------------------------------


class EmptyQueueError(LaurenAIError):
    """Raised by :class:`~lauren_ai._transport._mock.MockTransport` when the
    response queue is exhausted but another ``complete()`` call is made.

    :param message: Human-readable description.
    :type message: str
    """

    def __init__(self, message: str = "MockTransport response queue is empty") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Human-in-the-loop confirmation
# ---------------------------------------------------------------------------


class ToolConfirmationRejectedError(LaurenAIError):
    """Raised when a human-in-the-loop confirmation request is rejected.

    :param message: Human-readable description of the rejection.
    :type message: str
    :param tool_name: The name of the tool whose call was rejected.
    :type tool_name: str
    :param tool_use_id: The provider-assigned identifier for the rejected call.
    :type tool_use_id: str
    :param reason: The human-provided reason for rejecting the call.
    :type reason: str
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        tool_use_id: str,
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.tool_name: str = tool_name
        self.tool_use_id: str = tool_use_id
        self.reason: str = reason

    def __str__(self) -> str:
        base = (
            f"Tool call {self.tool_name!r} (id={self.tool_use_id!r}) rejected: "
            f"{self.message}"
        )
        if self.reason:
            base += f" Reason: {self.reason!r}"
        return base


# ---------------------------------------------------------------------------
# Knowledge / workflow / eval
# ---------------------------------------------------------------------------


class KnowledgeLoadError(LaurenAIError):
    """Raised when a knowledge base fails to load or initialise.

    :param message: Human-readable description of the failure.
    :type message: str
    :param source: The knowledge source path or identifier, if known.
    :type source: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.source: str | None = source

    def __str__(self) -> str:
        if self.source is not None:
            return f"Failed to load knowledge from {self.source!r}: {self.message}"
        return f"Knowledge load error: {self.message}"


class WorkflowError(LaurenAIError):
    """Raised when a workflow step fails.

    :param message: Human-readable description of the failure.
    :type message: str
    :param step_name: The name of the workflow step that failed, if known.
    :type step_name: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        step_name: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.step_name: str | None = step_name

    def __str__(self) -> str:
        if self.step_name is not None:
            return f"Workflow step {self.step_name!r} failed: {self.message}"
        return f"Workflow error: {self.message}"


class OutputParserError(LaurenAIError):
    """Raised when an output parser fails to parse LLM text.

    :param message: Human-readable description of the parse failure.
    :type message: str
    :param raw_output: The raw LLM output that could not be parsed.
    :type raw_output: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Initialise the error.

        :param message: Human-readable description of the parse failure.
        :type message: str
        :param raw_output: The raw LLM output that could not be parsed.
        :type raw_output: str | None
        :param cause: The underlying exception.
        :type cause: BaseException | None
        """
        super().__init__(message, cause=cause)
        self.raw_output: str | None = raw_output

    def __str__(self) -> str:
        """Return a string representation of the error.

        :return: Error message, optionally followed by a snippet of bad output.
        :rtype: str
        """
        base = self.message
        if self.raw_output is not None:
            snippet = self.raw_output[:80]
            base += f" | raw_output={snippet!r}"
        if self.cause is not None:
            base += f" | caused by: {self.cause!r}"
        return base


class EvalError(LaurenAIError):
    """Raised when an evaluation framework operation fails.

    :param message: Human-readable description of the failure.
    :type message: str
    :param eval_name: The name of the evaluation that failed, if known.
    :type eval_name: str | None
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """

    def __init__(
        self,
        message: str,
        *,
        eval_name: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.eval_name: str | None = eval_name

    def __str__(self) -> str:
        if self.eval_name is not None:
            return f"Eval {self.eval_name!r} failed: {self.message}"
        return f"Eval error: {self.message}"


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class TracingError(LaurenAIError):
    """Base class for tracing and observability errors.

    Raised when the tracing subsystem encounters an unrecoverable condition,
    such as a misconfigured exporter or a failed export operation.

    :param message: Human-readable description of the failure.
    :type message: str
    :param cause: The underlying exception.
    :type cause: BaseException | None
    """
