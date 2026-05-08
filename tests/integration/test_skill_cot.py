"""Integration tests for the chain-of-thought-prompting skill (Skill 24).

Verifies CoT suffix addition and structured CoT response parsing directly.
Agent-based CoT tested via TestClient.
"""

import re

import pytest

from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent
from lauren_ai.testing import TestClient


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------

COT_SUFFIX = "\n\nThink through this step by step before answering."


def add_cot(prompt: str) -> str:
    return prompt + COT_SUFFIX


def extract_cot_answer(response: str) -> tuple[str, str]:
    """Extract <reasoning> and <answer> blocks from a structured CoT response."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else response.strip()
    return reasoning, answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(text, *, n=1, stop="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=text,
        tool_calls=[],
        stop_reason=stop,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Tests: CoT suffix
# ---------------------------------------------------------------------------


class TestCotSuffix:
    def test_cot_suffix_added_to_prompt(self):
        result = add_cot("What is 2+2?")
        assert result.startswith("What is 2+2?")
        assert "step by step" in result

    def test_cot_suffix_content(self):
        assert "Think through this step by step" in add_cot("Hello")

    def test_original_prompt_preserved(self):
        prompt = "Explain quantum entanglement."
        assert prompt in add_cot(prompt)

    def test_suffix_appended_not_prepended(self):
        prompt = "My question"
        result = add_cot(prompt)
        assert result.index(prompt) == 0
        assert result.index(COT_SUFFIX) == len(prompt)


# ---------------------------------------------------------------------------
# Tests: CoT extraction
# ---------------------------------------------------------------------------


class TestExtractCotAnswer:
    def test_extracts_reasoning_and_answer(self):
        response = (
            "<reasoning>\nStep 1: Consider the problem.\nStep 2: Apply logic.\n</reasoning>\n"
            "<answer>\n42\n</answer>"
        )
        reasoning, answer = extract_cot_answer(response)
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert answer == "42"

    def test_answer_only_returns_full_response_when_no_tags(self):
        reasoning, answer = extract_cot_answer("The answer is 42.")
        assert reasoning == ""
        assert answer == "The answer is 42."

    def test_strips_whitespace_from_extracted_sections(self):
        reasoning, answer = extract_cot_answer(
            "<reasoning>   some reasoning   </reasoning><answer>   result   </answer>"
        )
        assert reasoning == "some reasoning"
        assert answer == "result"

    def test_multiline_reasoning(self):
        reasoning, answer = extract_cot_answer(
            "<reasoning>\nLine 1.\nLine 2.\nLine 3.\n</reasoning>\n<answer>Final</answer>"
        )
        assert "Line 1." in reasoning
        assert "Line 2." in reasoning
        assert answer == "Final"

    def test_missing_answer_tag_returns_full_response(self):
        reasoning, answer = extract_cot_answer(
            "<reasoning>Thought about it.</reasoning>\nJust the answer text."
        )
        assert "Thought about it." in reasoning
        assert "Just the answer text." in answer

    def test_empty_response(self):
        reasoning, answer = extract_cot_answer("")
        assert reasoning == ""
        assert answer == ""


# ---------------------------------------------------------------------------
# Tests: CoT via agent TestClient
# ---------------------------------------------------------------------------


class TestCotWithAgent:
    def test_agent_receives_cot_suffixed_message(self):
        @agent(model="mock-model", system="You are a reasoning assistant.")
        class CotAgent:
            pass

        cot_response = "<reasoning>I think step by step.</reasoning><answer>Yes</answer>"
        client = TestClient(CotAgent())
        client.mock.queue_response(_c(cot_response))

        prompt = "What is the capital of France?"
        cot_prompt = add_cot(prompt)
        result = client.run(cot_prompt)

        assert "step by step" in cot_prompt
        reasoning, answer = extract_cot_answer(result.content)
        assert "I think step by step" in reasoning
        assert answer == "Yes"

    @pytest.mark.asyncio
    async def test_agent_cot_async(self):
        @agent(model="mock-model", system="You are a reasoning assistant.")
        class CotAgentAsync:
            pass

        cot_response = "<reasoning>Async reasoning.</reasoning><answer>42</answer>"
        client = TestClient(CotAgentAsync())
        client.mock.queue_response(_c(cot_response))

        result = await client.run_async(add_cot("What is 6 times 7?"))
        reasoning, answer = extract_cot_answer(result.content)
        assert "Async reasoning" in reasoning
        assert answer == "42"
