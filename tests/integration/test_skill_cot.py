"""Integration tests for the chain-of-thought-prompting skill (Skill 24).

Verifies CoT suffix addition and structured CoT response parsing.
"""
import pytest

from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------

import re

COT_SUFFIX = "\n\nThink through this step by step before answering."


def add_cot(prompt: str) -> str:
    return prompt + COT_SUFFIX


def extract_cot_answer(response: str) -> tuple[str, str]:
    """Extract <reasoning> and <answer> blocks from a structured CoT response."""
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else response.strip()
    return reasoning, answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _make_runner(mock=None):
    if mock is None:
        mock = MockTransport()
    cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
    runner = AgentRunner(transport=mock, tools={}, config=cfg)
    return runner, mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCotSuffix:
    def test_cot_suffix_added_to_prompt(self):
        prompt = "What is 2+2?"
        result = add_cot(prompt)
        assert result.startswith(prompt)
        assert "step by step" in result

    def test_cot_suffix_content(self):
        assert "Think through this step by step" in COT_SUFFIX

    def test_original_prompt_preserved(self):
        prompt = "Explain quantum entanglement."
        result = add_cot(prompt)
        assert prompt in result

    def test_suffix_appended_not_prepended(self):
        prompt = "My question"
        result = add_cot(prompt)
        assert result.index(prompt) == 0
        assert result.index(COT_SUFFIX) == len(prompt)


class TestExtractCotAnswer:
    def test_extracts_reasoning_and_answer(self):
        response = """<reasoning>
Step 1: Consider the problem.
Step 2: Apply logic.
</reasoning>
<answer>
42
</answer>"""
        reasoning, answer = extract_cot_answer(response)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert answer == "42"

    def test_answer_only_returns_full_response_when_no_tags(self):
        response = "The answer is 42."
        reasoning, answer = extract_cot_answer(response)
        assert reasoning == ""
        assert answer == "The answer is 42."

    def test_strips_whitespace_from_extracted_sections(self):
        response = "<reasoning>   some reasoning   </reasoning><answer>   result   </answer>"
        reasoning, answer = extract_cot_answer(response)
        assert reasoning == "some reasoning"
        assert answer == "result"

    def test_multiline_reasoning(self):
        response = "<reasoning>\nLine 1.\nLine 2.\nLine 3.\n</reasoning>\n<answer>Final</answer>"
        reasoning, answer = extract_cot_answer(response)
        assert "Line 1." in reasoning
        assert "Line 2." in reasoning
        assert answer == "Final"

    def test_missing_answer_tag_returns_full_response(self):
        response = "<reasoning>Thought about it.</reasoning>\nJust the answer text."
        reasoning, answer = extract_cot_answer(response)
        assert "Thought about it." in reasoning
        assert "Just the answer text." in answer

    def test_empty_response(self):
        reasoning, answer = extract_cot_answer("")
        assert reasoning == ""
        assert answer == ""


class TestCotWithRunner:
    @pytest.mark.asyncio
    async def test_agent_receives_cot_suffixed_message(self):
        mock = MockTransport()
        cot_response = "<reasoning>I think step by step.</reasoning><answer>Yes</answer>"
        mock.queue_response(_completion(cot_response))

        received_messages = []
        orig_complete = mock.complete

        async def spy(messages, **kw):
            received_messages.extend(messages)
            return await orig_complete(messages, **kw)

        mock.complete = spy

        @agent(model="mock-model", system="You are a reasoning assistant.")
        class CotAgent: ...

        runner, _ = _make_runner(mock)
        prompt = add_cot("What is the capital of France?")
        response = await runner.run(CotAgent(), prompt)

        assert "step by step" in received_messages[-1]["content"]
        reasoning, answer = extract_cot_answer(response.content)
        assert "I think step by step" in reasoning
        assert answer == "Yes"
