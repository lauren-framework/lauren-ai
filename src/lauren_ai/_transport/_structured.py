from __future__ import annotations

"""StructuredLLM — force schema-valid outputs via native provider mechanisms."""

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class StructuredLLM(Generic[T]):
    """Typed wrapper over LLMService that forces structured output.

    Created via ``llm.with_structured_output(MyModel)``.

    Usage::

        structured = llm.with_structured_output(SentimentResult)
        result: SentimentResult = await structured.complete([...])
    """

    def __init__(self, llm: Any, model_cls: type[T]) -> None:
        """Initialise the structured LLM wrapper.

        :param llm: The underlying :class:`~lauren_ai._module.LLMService`.
        :type llm: Any
        :param model_cls: Pydantic model class that the output must conform to.
        :type model_cls: type[T]
        """
        self._llm = llm
        self._model_cls = model_cls
        self._schema: dict[str, Any] = self._build_schema()

    def _build_schema(self) -> dict[str, Any]:
        """Extract JSON Schema from *model_cls* if it is a Pydantic model.

        Falls back to an empty schema when the class does not expose
        ``model_json_schema()``.

        :return: JSON Schema dict for the model.
        :rtype: dict[str, Any]
        """
        try:
            return self._model_cls.model_json_schema()  # type: ignore[attr-defined]
        except AttributeError:
            return {}

    async def complete(self, messages: list[Any]) -> T:
        """Complete *messages* and return a validated model instance.

        Uses tool-calling to force the model to emit JSON that matches the
        schema, then constructs and returns a ``model_cls`` instance.

        :param messages: Conversation messages.
        :type messages: list[Any]
        :return: A validated instance of *model_cls*.
        :rtype: T
        :raises OutputParserError: When the model's response cannot be parsed
            or validated against the schema.
        """
        return await self._complete_messages(messages)

    async def _complete_messages(self, messages: list[Any]) -> T:
        """Internal completion method used by :class:`~lauren_ai._chains.Chain`.

        :param messages: Conversation messages.
        :type messages: list[Any]
        :return: A validated instance of *model_cls*.
        :rtype: T
        """
        from lauren_ai._output_parsers._base import OutputParserError
        from lauren_ai._transport import Completion, ToolChoice, ToolSchema

        model_name = self._model_cls.__name__
        tool = ToolSchema(
            name="structured_output",
            description=f"Return a structured {model_name} object.",
            input_schema=self._schema,
        )

        result = await self._llm.complete(
            messages,
            tools=[tool],
            tool_choice=ToolChoice.specific("structured_output"),
        )

        if isinstance(result, Completion):
            # Extract from tool call
            if result.tool_calls:
                tc = result.tool_calls[0]
                try:
                    return self._model_cls(**tc.input)  # type: ignore[return-value]
                except Exception as e:
                    raise OutputParserError(
                        f"Structured output validation failed: {e}"
                    ) from e
            # Fallback: try to parse content as JSON
            import json
            try:
                data = json.loads(result.content)
                return self._model_cls(**data)  # type: ignore[return-value]
            except Exception as e:
                raise OutputParserError(
                    f"Could not extract structured output: {e}"
                ) from e
        else:
            # AsyncIterator — collect all chunks and parse
            chunks = []
            async for chunk in result:
                if chunk.delta:
                    chunks.append(chunk.delta)
            import json
            try:
                data = json.loads("".join(chunks))
                return self._model_cls(**data)  # type: ignore[return-value]
            except Exception as e:
                raise OutputParserError(
                    f"Structured output from stream failed: {e}"
                ) from e

    def __or__(self, other: Any) -> Any:
        """Compose this structured LLM into a :class:`~lauren_ai._chains.Chain`.

        :param other: The next step in the chain (parser, function, etc.).
        :type other: Any
        :return: A :class:`~lauren_ai._chains.Chain` containing both steps.
        :rtype: Chain
        """
        from lauren_ai._chains import Chain

        return Chain(steps=[self, other])
