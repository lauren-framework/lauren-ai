"""RetryOutputParser — auto-retry with LLM correction on parse failure."""

from __future__ import annotations

from typing import Any

from lauren_ai._output_parsers._base import MaxRetryError, OutputParserError
from lauren_ai._transport import Message


class RetryOutputParser:
    """Wrap any :class:`~lauren_ai._output_parsers._base.OutputParser` and
    automatically retry with error feedback when parsing fails.

    On each failure a correction turn is appended to the conversation and the
    LLM is asked to produce a valid response.  Retries continue up to
    *max_retries* additional attempts before :class:`MaxRetryError` is raised.

    Usage::

        parser = RetryOutputParser(
            parser=PydanticOutputParser(model=UserInfo),
            llm=llm_service,
            max_retries=3,
        )
        result = await parser.parse_with_retry(
            original_messages=messages,
            completion=completion,
        )

    :param parser: The underlying parser to wrap.
    :type parser: Any
    :param llm: The :class:`~lauren_ai._module.LLMService` to call for
        correction turns.
    :type llm: Any
    :param max_retries: Maximum number of additional retry attempts after the
        first failure.  Defaults to ``3``.
    :type max_retries: int
    """

    def __init__(
        self,
        parser: Any,
        llm: Any,
        max_retries: int = 3,
    ) -> None:
        """Initialise the retry wrapper.

        :param parser: Underlying output parser.
        :param llm: LLM service for correction turns.
        :param max_retries: Maximum correction attempts.
        """
        self._parser = parser
        self._llm = llm
        self._max_retries = max_retries

    def parse(self, text: str) -> Any:
        """Synchronous parse without retry — delegates to the wrapped parser.

        Use :meth:`parse_with_retry` to get the automatic retry behaviour.

        :param text: Raw LLM output text.
        :type text: str
        :return: Parsed value.
        """
        return self._parser.parse(text)

    async def invoke(self, input: Any) -> Any:
        """Invoke as a :class:`~lauren_ai._chains.Runnable` (no retry).

        Delegates synchronously to the wrapped parser.  Use
        :meth:`parse_with_retry` when you need LLM-assisted correction.

        :param input: LLM output or completion object.
        :type input: Any
        :return: Parsed value.
        """
        text = input.content if hasattr(input, "content") else str(input)
        return self.parse(text)

    @property
    def format_instructions(self) -> str:
        """Delegate to the wrapped parser's format instructions.

        :return: Format instructions string.
        :rtype: str
        """
        return self._parser.format_instructions

    async def parse_with_retry(
        self,
        original_messages: list[Message],
        completion: Any,  # Completion
    ) -> Any:
        """Parse *completion* with automatic LLM-assisted retry on failure.

        :param original_messages: The conversation messages that produced
            *completion*.
        :type original_messages: list[Message]
        :param completion: The initial :class:`~lauren_ai._transport.Completion`
            to parse.
        :return: A successfully parsed value of the wrapped parser's return type.
        :raises MaxRetryError: When all attempts are exhausted without success.
        """
        from lauren_ai._transport import Completion, TokenUsage  # noqa: PLC0415

        messages = list(original_messages)
        # Append the assistant's (potentially malformed) response to the history
        messages.append(Message(role="assistant", content=completion.content))

        last_error: OutputParserError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return self._parser.parse(completion.content)
            except OutputParserError as e:
                last_error = e
                if attempt >= self._max_retries:
                    break

                # Build a correction prompt
                correction = (
                    f"Your previous response was invalid and could not be parsed.\n"
                    f"Error: {e}\n\n"
                    f"Please provide a corrected response following this format:\n"
                    f"{self._parser.format_instructions}"
                )
                messages.append(Message(role="user", content=correction))

                # Request a correction from the LLM
                result = await self._llm.complete(messages)
                if isinstance(result, Completion):
                    completion = result
                else:
                    # Streaming response — collect all chunks
                    chunks: list[str] = []
                    async for chunk in result:
                        if chunk.delta:
                            chunks.append(chunk.delta)
                    completion = Completion(
                        id="retry",
                        model="",
                        content="".join(chunks),
                        tool_calls=[],
                        stop_reason="end_turn",
                        usage=TokenUsage(0, 0),
                    )
                messages.append(Message(role="assistant", content=completion.content))

        raise MaxRetryError(
            f"Failed to parse after {self._max_retries + 1} attempts. Last error: {last_error}",
            attempts=self._max_retries + 1,
        )

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])
