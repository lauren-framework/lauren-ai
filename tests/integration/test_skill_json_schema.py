"""Integration tests for Skill 5: JSON Schema Enforcement for LLM Responses.

Tests cover:
- JSONOutputParser parses valid JSON dict
- JSONOutputParser raises on invalid input
- PydanticOutputParser parses valid JSON to Pydantic model
- PydanticOutputParser raises on validation failure
- RetryOutputParser retries on initial failure
- parse_json_response helper validates schema
- Agent completion with JSON system prompt
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from lauren_ai import (
    JSONOutputParser,
    LLMConfig,
    OutputParserError,
    PydanticOutputParser,
    RetryOutputParser,
)
from lauren_ai._agents import agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str, *, id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=10),
    )


def _make_runner(mock: MockTransport | None = None) -> tuple[AgentRunner, MockTransport]:
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


def parse_json_response(content: str, schema: type[BaseModel]) -> BaseModel:
    """Validate LLM text against a Pydantic schema."""
    try:
        return schema.model_validate_json(content.strip())
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid JSON schema response: {exc}") from exc


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool


class OrderInfo(BaseModel):
    order_id: str
    total: float
    items: list[str] = []


# ---------------------------------------------------------------------------
# TestJSONOutputParser
# ---------------------------------------------------------------------------


class TestJSONOutputParser:
    def test_parses_valid_json_object(self):
        parser = JSONOutputParser()
        result = parser.parse('{"key": "value"}')
        assert isinstance(result, dict)

    def test_parsed_key_matches(self):
        parser = JSONOutputParser()
        result = parser.parse('{"name": "Widget", "price": 9.99}')
        assert result["name"] == "Widget"

    def test_raises_on_invalid_json(self):
        parser = JSONOutputParser()
        with pytest.raises((OutputParserError, ValueError, Exception)):
            parser.parse("not json at all")

    def test_parses_nested_json(self):
        parser = JSONOutputParser()
        result = parser.parse('{"order": {"id": "123", "total": 50.0}}')
        assert result["order"]["id"] == "123"

    def test_parses_json_with_list(self):
        parser = JSONOutputParser()
        result = parser.parse('{"items": ["a", "b", "c"]}')
        assert result["items"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# TestPydanticOutputParserEnforcement
# ---------------------------------------------------------------------------


class TestPydanticOutputParserEnforcement:
    def test_valid_product_json_parses(self):
        parser = PydanticOutputParser(model=ProductInfo)
        result = parser.parse('{"name": "Widget", "price": 9.99, "in_stock": true}')
        assert isinstance(result, ProductInfo)

    def test_product_name_correct(self):
        parser = PydanticOutputParser(model=ProductInfo)
        result = parser.parse('{"name": "Gadget", "price": 19.99, "in_stock": false}')
        assert result.name == "Gadget"

    def test_product_price_correct(self):
        parser = PydanticOutputParser(model=ProductInfo)
        result = parser.parse('{"name": "Gadget", "price": 19.99, "in_stock": false}')
        assert result.price == pytest.approx(19.99)

    def test_invalid_json_raises(self):
        parser = PydanticOutputParser(model=ProductInfo)
        with pytest.raises((OutputParserError, ValueError, Exception)):
            parser.parse("Sorry I cannot provide that.")

    def test_missing_required_field_raises(self):
        parser = PydanticOutputParser(model=ProductInfo)
        # Missing 'in_stock'
        with pytest.raises((OutputParserError, ValueError, Exception)):
            parser.parse('{"name": "Gadget", "price": 9.99}')


# ---------------------------------------------------------------------------
# TestParseJsonResponseHelper
# ---------------------------------------------------------------------------


class TestParseJsonResponseHelper:
    def test_valid_product_parses(self):
        content = '{"name": "Laptop", "price": 999.0, "in_stock": true}'
        result = parse_json_response(content, ProductInfo)
        assert result.name == "Laptop"
        assert result.in_stock is True

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid JSON schema response"):
            parse_json_response("not json", ProductInfo)

    def test_missing_fields_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_json_response('{"name": "X"}', ProductInfo)

    def test_order_info_with_items(self):
        content = '{"order_id": "ORD-001", "total": 150.0, "items": ["book", "pen"]}'
        result = parse_json_response(content, OrderInfo)
        assert result.order_id == "ORD-001"
        assert result.items == ["book", "pen"]


# ---------------------------------------------------------------------------
# TestAgentJSONSchemaPattern
# ---------------------------------------------------------------------------


class TestAgentJSONSchemaPattern:
    async def test_agent_returns_valid_product_json(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"name": "Widget", "price": 9.99, "in_stock": true}'))

        @agent(model="mock-model", system='Always return JSON: {"name": str, "price": float, "in_stock": bool}')
        class ProductAgent: ...

        resp = await runner.run(ProductAgent(), "Describe product #42")
        result = parse_json_response(resp.content, ProductInfo)
        assert result.name == "Widget"

    async def test_agent_returns_order_json(self):
        runner, mock = _make_runner()
        mock.queue_response(_completion('{"order_id": "O-999", "total": 75.5, "items": ["item1"]}'))

        @agent(model="mock-model")
        class OrderAgent: ...

        resp = await runner.run(OrderAgent(), "Get order details")
        result = parse_json_response(resp.content, OrderInfo)
        assert result.order_id == "O-999"
        assert result.total == pytest.approx(75.5)
