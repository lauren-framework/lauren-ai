"""Base types for output parsers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from lauren_ai._exceptions import LaurenAIError, OutputParserError  # noqa: F401 — re-exported

# OutputParserError is imported from _exceptions and re-exported so that
# existing code importing from _output_parsers._base keeps working.
__all__ = ["OutputParserError", "MaxRetryError", "OutputParser"]


class MaxRetryError(LaurenAIError):
    """Raised when :class:`RetryOutputParser` exhausts its retry budget.

    :param message: Human-readable description of the exhaustion.
    :type message: str
    :param attempts: Total number of parse attempts made.
    :type attempts: int
    """

    def __init__(self, message: str, attempts: int) -> None:
        """Initialise the error.

        :param message: Human-readable description.
        :type message: str
        :param attempts: Total attempts made before giving up.
        :type attempts: int
        """
        super().__init__(message)
        self.attempts: int = attempts


@runtime_checkable
class OutputParser(Protocol):
    """Protocol for output parsers — convert raw LLM text to typed values.

    Any class that exposes :meth:`parse` and the :attr:`format_instructions`
    property satisfies this protocol and can be used as the last step in a
    :class:`~lauren_ai._chains.Chain`.
    """

    def parse(self, text: str) -> Any:
        """Parse *text* and return a typed value.

        :param text: Raw LLM output text.
        :type text: str
        :return: Parsed value.
        """
        ...

    @property
    def format_instructions(self) -> str:
        """Human-readable instructions for the LLM describing the expected format.

        :return: Format instructions string.
        :rtype: str
        """
        ...
