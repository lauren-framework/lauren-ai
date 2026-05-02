"""Unit tests for output parsers."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from lauren_ai._output_parsers import (
    CommaSeparatedListParser,
    JSONOutputParser,
    MarkdownCodeBlockParser,
    MaxRetryError,
    OutputParserError,
    PydanticOutputParser,
    RegexParser,
    RetryOutputParser,
    StrOutputParser,
)
from lauren_ai._transport import Completion, Message, TokenUsage
from lauren_ai._config import LLMConfig
from lauren_ai._module import LLMService


class UserInfo(BaseModel):
    name: str
    age: int


class TestStrOutputParser:
    def test_strips_whitespace(self):
        assert StrOutputParser().parse("  hello  ") == "hello"

    def test_format_instructions(self):
        assert "plain text" in StrOutputParser().format_instructions.lower()

    def test_pipe_creates_chain(self):
        from lauren_ai._chains import Chain

        chain = StrOutputParser() | "next"
        assert isinstance(chain, Chain)


class TestJSONOutputParser:
    def test_parses_valid_json(self):
        result = JSONOutputParser().parse('{"key": 42}')
        assert result == {"key": 42}

    def test_strips_markdown_fence(self):
        text = "```json\n{\"a\": 1}\n```"
        assert JSONOutputParser().parse(text) == {"a": 1}

    def test_invalid_json_raises(self):
        with pytest.raises(OutputParserError, match="JSON"):
            JSONOutputParser().parse("not json")

    def test_parses_list(self):
        assert JSONOutputParser().parse("[1, 2, 3]") == [1, 2, 3]


class TestRegexParser:
    def test_extracts_groups(self):
        parser = RegexParser(r"Name: (?P<name>\w+), Age: (?P<age>\d+)")
        result = parser.parse("Name: Alice, Age: 30")
        assert result == {"name": "Alice", "age": "30"}

    def test_no_match_raises(self):
        parser = RegexParser(r"(?P<x>\d+)")
        with pytest.raises(OutputParserError):
            parser.parse("no numbers here spelled out")


class TestCommaSeparatedListParser:
    def test_basic_parse(self):
        result = CommaSeparatedListParser().parse("apple, banana, cherry")
        assert result == ["apple", "banana", "cherry"]

    def test_strips_whitespace(self):
        assert CommaSeparatedListParser().parse("  a ,  b  , c  ") == ["a", "b", "c"]

    def test_empty_string(self):
        assert CommaSeparatedListParser().parse("") == []


class TestMarkdownCodeBlockParser:
    def test_extracts_python_block(self):
        text = "Here is code:\n```python\nprint('hi')\n```"
        assert MarkdownCodeBlockParser("python").parse(text) == "print('hi')"

    def test_extracts_without_language(self):
        text = "```\nhello world\n```"
        assert "hello world" in MarkdownCodeBlockParser().parse(text)

    def test_no_block_raises(self):
        with pytest.raises(OutputParserError):
            MarkdownCodeBlockParser("python").parse("no code block here")


class TestPydanticOutputParser:
    def test_parses_valid_json(self):
        parser = PydanticOutputParser(model=UserInfo)
        result = parser.parse('{"name": "Bob", "age": 25}')
        assert isinstance(result, UserInfo)
        assert result.name == "Bob"
        assert result.age == 25

    def test_strips_markdown_fence(self):
        parser = PydanticOutputParser(model=UserInfo)
        text = '```json\n{"name": "Eve", "age": 40}\n```'
        result = parser.parse(text)
        assert result.name == "Eve"

    def test_invalid_json_raises_output_parser_error(self):
        parser = PydanticOutputParser(model=UserInfo)
        with pytest.raises(OutputParserError):
            parser.parse("not json")

    def test_missing_field_raises_output_parser_error(self):
        parser = PydanticOutputParser(model=UserInfo)
        with pytest.raises(OutputParserError):
            parser.parse('{"name": "Alice"}')

    def test_format_instructions_contains_schema(self):
        parser = PydanticOutputParser(model=UserInfo)
        instructions = parser.format_instructions
        assert "name" in instructions or "UserInfo" in instructions


def _make_completion(content: str) -> Completion:
    return Completion(
        id="test",
        model="test",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(0, 0),
    )


class TestRetryOutputParser:
    async def test_succeeds_on_first_attempt(self):
        from lauren_ai._transport._mock import MockTransport

        transport = MockTransport()
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=transport, config=config)
        parser = RetryOutputParser(
            parser=PydanticOutputParser(model=UserInfo),
            llm=llm,
            max_retries=2,
        )
        completion = _make_completion('{"name": "Alice", "age": 30}')
        result = await parser.parse_with_retry(
            original_messages=[Message(role="user", content="q")],
            completion=completion,
        )
        assert isinstance(result, UserInfo)
        assert result.name == "Alice"

    async def test_retries_on_bad_json_then_succeeds(self):
        from lauren_ai._transport._mock import MockTransport

        transport = MockTransport()
        # Queue a valid response for the correction turn
        transport.queue_response(
            _make_completion('{"name": "Bob", "age": 20}')
        )
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=transport, config=config)
        parser = RetryOutputParser(
            parser=PydanticOutputParser(model=UserInfo),
            llm=llm,
            max_retries=2,
        )
        bad_completion = _make_completion("this is not json at all")
        result = await parser.parse_with_retry(
            original_messages=[Message(role="user", content="q")],
            completion=bad_completion,
        )
        assert isinstance(result, UserInfo)
        assert result.name == "Bob"

    async def test_raises_max_retry_after_exhaustion(self):
        from lauren_ai._transport._mock import MockTransport

        transport = MockTransport()
        # Queue bad JSON for every correction turn
        for _ in range(5):
            transport.queue_response(_make_completion("still not json"))
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=transport, config=config)
        parser = RetryOutputParser(
            parser=JSONOutputParser(),
            llm=llm,
            max_retries=2,
        )
        bad_completion = _make_completion("bad json")
        with pytest.raises(MaxRetryError) as exc_info:
            await parser.parse_with_retry(
                original_messages=[Message(role="user", content="q")],
                completion=bad_completion,
            )
        assert exc_info.value.attempts >= 2
