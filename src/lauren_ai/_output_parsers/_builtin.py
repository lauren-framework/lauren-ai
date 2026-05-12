"""Built-in output parsers for common LLM response formats."""

from __future__ import annotations

import json
import re
from typing import Any

from lauren_ai._output_parsers._base import OutputParserError


def _extract_text(value: Any) -> str:
    """Extract a plain string from *value*.

    When *value* has a ``content`` attribute (e.g. a
    :class:`~lauren_ai._transport.Completion`) that attribute is returned;
    otherwise ``str(value)`` is used.

    :param value: The value to extract text from.
    :type value: Any
    :return: Plain text string.
    :rtype: str
    """
    if hasattr(value, "content"):
        return str(value.content)
    return str(value)


class StrOutputParser:
    """Strip whitespace from the LLM response and return it as a plain string."""

    def parse(self, text: str) -> str:
        """Return *text* with leading/trailing whitespace removed.

        :param text: Raw LLM output.
        :type text: str
        :return: Stripped string.
        :rtype: str
        """
        return text.strip()

    async def invoke(self, input: Any) -> str:
        """Invoke as a :class:`~lauren_ai._chains.Runnable`.

        Accepts a raw string or any object with a ``content`` attribute
        (e.g. :class:`~lauren_ai._transport.Completion`).

        :param input: LLM output or completion object.
        :type input: Any
        :return: Parsed string.
        :rtype: str
        """
        return self.parse(_extract_text(input))

    @property
    def format_instructions(self) -> str:
        """Return format instructions for plain-text responses.

        :return: Format instructions string.
        :rtype: str
        """
        return "Respond with plain text."

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])


class JSONOutputParser:
    """Parse a JSON value from LLM text.

    Strips Markdown fenced code blocks (````json ... ````) before parsing
    so the LLM can wrap its JSON in a code block without breaking the parser.
    """

    def parse(self, text: str) -> Any:
        """Parse *text* as JSON.

        :param text: Raw LLM output, optionally wrapped in a Markdown code block.
        :type text: str
        :return: Parsed Python value (dict, list, str, int, …).
        :raises OutputParserError: When the text cannot be decoded as JSON.
        """
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            raise OutputParserError(f"Failed to parse JSON: {e}. Text: {text!r}") from e

    async def invoke(self, input: Any) -> Any:
        """Invoke as a :class:`~lauren_ai._chains.Runnable`.

        :param input: LLM output or completion object.
        :type input: Any
        :return: Parsed Python value.
        :raises OutputParserError: When input cannot be decoded as JSON.
        """
        return self.parse(_extract_text(input))

    @property
    def format_instructions(self) -> str:
        """Return format instructions for JSON responses.

        :return: Format instructions string.
        :rtype: str
        """
        return "Respond with valid JSON."

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])


class RegexParser:
    """Extract named capture groups from LLM text using a regex pattern.

    :param pattern: A regular expression string containing at least one
        named group ``(?P<name>...)``.
    :type pattern: str
    """

    def __init__(self, pattern: str) -> None:
        """Compile the regex pattern.

        :param pattern: Regular expression string.
        :type pattern: str
        """
        self.pattern = re.compile(pattern)

    def parse(self, text: str) -> dict[str, str]:
        """Search *text* for the pattern and return named groups.

        :param text: Raw LLM output.
        :type text: str
        :return: Dictionary of named capture group values.
        :rtype: dict[str, str]
        :raises OutputParserError: When the pattern does not match *text*.
        """
        m = self.pattern.search(text)
        if m is None:
            raise OutputParserError(f"Pattern {self.pattern.pattern!r} did not match: {text!r}")
        return m.groupdict()

    async def invoke(self, input: Any) -> dict[str, str]:
        """Invoke as a :class:`~lauren_ai._chains.Runnable`.

        :param input: LLM output or completion object.
        :type input: Any
        :return: Dictionary of named capture group values.
        :raises OutputParserError: When the pattern does not match.
        """
        return self.parse(_extract_text(input))

    @property
    def format_instructions(self) -> str:
        """Return format instructions showing the required pattern.

        :return: Format instructions string.
        :rtype: str
        """
        return f"Your response must match the pattern: {self.pattern.pattern}"

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])


class CommaSeparatedListParser:
    """Parse a comma-separated list from LLM text."""

    def parse(self, text: str) -> list[str]:
        """Split *text* on commas and strip whitespace from each item.

        :param text: Raw LLM output.
        :type text: str
        :return: List of stripped, non-empty items.
        :rtype: list[str]
        """
        return [item.strip() for item in text.split(",") if item.strip()]

    async def invoke(self, input: Any) -> list[str]:
        """Invoke as a :class:`~lauren_ai._chains.Runnable`.

        :param input: LLM output or completion object.
        :type input: Any
        :return: List of stripped, non-empty items.
        :rtype: list[str]
        """
        return self.parse(_extract_text(input))

    @property
    def format_instructions(self) -> str:
        """Return format instructions for comma-separated lists.

        :return: Format instructions string.
        :rtype: str
        """
        return "Respond with a comma-separated list of values."

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])


class MarkdownCodeBlockParser:
    """Extract the first fenced code block from LLM text.

    :param language: Optional language tag to match (e.g. ``"python"``).
        When provided the parser first attempts to find a block with that
        language tag before falling back to any fenced block.
    :type language: str
    """

    def __init__(self, language: str = "") -> None:
        """Initialise the parser with an optional language specifier.

        :param language: Expected fenced-block language tag.
        :type language: str
        """
        self.language = language

    def parse(self, text: str) -> str:
        """Extract the first fenced code block from *text*.

        :param text: Raw LLM output containing a fenced code block.
        :type text: str
        :return: Content of the first matching fenced code block, stripped of
            leading/trailing whitespace.
        :rtype: str
        :raises OutputParserError: When no fenced code block is found.
        """
        if self.language:
            pattern = r"```" + re.escape(self.language) + r"?\s*\n(.*?)\n```"
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1).strip()
        # Fallback: any fenced block
        m = re.search(r"```\w*\s*\n(.*?)\n```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        raise OutputParserError(f"No fenced code block found in: {text!r}")

    async def invoke(self, input: Any) -> str:
        """Invoke as a :class:`~lauren_ai._chains.Runnable`.

        :param input: LLM output or completion object.
        :type input: Any
        :return: Extracted code block content.
        :raises OutputParserError: When no fenced code block is found.
        """
        return self.parse(_extract_text(input))

    @property
    def format_instructions(self) -> str:
        """Return format instructions for fenced code block responses.

        :return: Format instructions string.
        :rtype: str
        """
        lang = self.language or "code"
        return f"Wrap your response in a ```{lang}``` fenced code block."

    def __or__(self, other: Any) -> Any:
        from lauren_ai._chains import Chain  # noqa: PLC0415

        return Chain(steps=[self, other])
