"""Pydantic-backed output parser.

Do NOT add ``from __future__ import annotations`` to this file — Pydantic
needs concrete annotation types at parse time.
"""

import json
import re
from typing import Any, Generic, TypeVar

from lauren_ai._output_parsers._base import OutputParserError

T = TypeVar("T")


class PydanticOutputParser(Generic[T]):
    """Parse LLM text as a Pydantic model instance.

    Strips Markdown fenced code blocks before JSON-decoding, then validates
    the resulting dict against the provided Pydantic model class.

    Usage::

        class UserInfo(BaseModel):
            name: str
            age: int

        parser = PydanticOutputParser(model=UserInfo)
        user = parser.parse('{"name": "Alice", "age": 30}')

    :param model: The Pydantic model class to validate against.
    :type model: type[T]
    """

    def __init__(self, model: type[T]) -> None:
        """Initialise the parser with a Pydantic model class.

        :param model: The Pydantic v2 model class.
        :type model: type[T]
        """
        self._model = model

    def parse(self, text: str) -> T:
        """Parse *text* as a Pydantic model instance.

        :param text: Raw LLM output, optionally wrapped in a Markdown code
            block.
        :type text: str
        :return: A validated instance of the configured model.
        :rtype: T
        :raises OutputParserError: When *text* is not valid JSON or fails
            Pydantic validation.
        """
        # Strip markdown fenced code blocks
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            raise OutputParserError(f"Invalid JSON: {e}. Text: {text!r}") from e
        try:
            return self._model(**data)  # type: ignore[return-value]
        except Exception as e:
            raise OutputParserError(f"Pydantic validation failed: {e}") from e

    async def invoke(self, input: Any) -> T:
        """Invoke as a :class:`~lauren_ai._chains.Runnable`.

        Accepts a raw string or any object with a ``content`` attribute
        (e.g. :class:`~lauren_ai._transport.Completion`).

        :param input: LLM output or completion object.
        :type input: Any
        :return: A validated instance of the configured model.
        :rtype: T
        :raises OutputParserError: When parsing or validation fails.
        """
        text = input.content if hasattr(input, "content") else str(input)
        return self.parse(text)

    @property
    def format_instructions(self) -> str:
        """Return format instructions including the model's JSON schema.

        Falls back to a simpler message when schema generation is unavailable.

        :return: Format instructions string.
        :rtype: str
        """
        try:
            schema = self._model.model_json_schema()  # type: ignore[attr-defined]
            import json as _json

            return f"Respond with a JSON object matching this schema:\n{_json.dumps(schema, indent=2)}"
        except Exception:
            return f"Respond with a JSON object that can be parsed as {self._model.__name__}."

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])
