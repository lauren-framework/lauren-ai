"""Integration tests for the chain-of-thought-prompting skill (Skill 24).

Verifies CoT suffix addition and structured CoT response parsing via HTTP
through a Lauren TestClient. Agent-based CoT tested via Pattern B.
"""

import re

from lauren import LaurenFactory, controller, post, module, Json, use_value
from lauren.testing import TestClient
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._transport._mock import MockTransport
from lauren_ai._config import LLMConfig
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._agents import agent
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------

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
# Request models
# ---------------------------------------------------------------------------


class AddSuffixRequest(BaseModel):
    prompt: str


class ExtractRequest(BaseModel):
    response: str


class AgentRunRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controllers / Module / build_app
# ---------------------------------------------------------------------------


@controller("/cot")
class CotController:
    @post("/add-suffix")
    async def add_suffix(self, body: Json[AddSuffixRequest]) -> dict:
        return {"prompt": add_cot(body.prompt)}

    @post("/extract")
    async def extract(self, body: Json[ExtractRequest]) -> dict:
        reasoning, answer = extract_cot_answer(body.response)
        return {"reasoning": reasoning, "answer": answer}


@controller("/cot-agent")
class CotAgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._runner = AgentRunner(transport=mock, tools={}, config=cfg)

    @post("/run")
    async def run(self, body: Json[AgentRunRequest]) -> dict:
        @agent(model="mock-model", system="You are a reasoning assistant.")
        class CotAgent: ...

        prompt = add_cot(body.prompt)
        response = await self._runner.run(CotAgent(), prompt)
        reasoning, answer = extract_cot_answer(response.content)
        return {
            "content": response.content,
            "reasoning": reasoning,
            "answer": answer,
            "cot_prompt": prompt,
        }


@module(
    controllers=[CotController, CotAgentController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class CotModule: ...


def build_app(*responses: str):
    _MOCK.reset()
    for c in responses:
        _MOCK.queue_response(_completion(c))
    return TestClient(LaurenFactory.create(CotModule))


# ---------------------------------------------------------------------------
# Tests: CoT suffix
# ---------------------------------------------------------------------------


class TestCotSuffix:
    def test_cot_suffix_added_to_prompt(self):
        client = build_app()
        resp = client.post("/cot/add-suffix", json={"prompt": "What is 2+2?"})
        assert resp.status_code == 200
        result = resp.json()["prompt"]
        assert result.startswith("What is 2+2?")
        assert "step by step" in result

    def test_cot_suffix_content(self):
        client = build_app()
        resp = client.post("/cot/add-suffix", json={"prompt": "Hello"})
        assert resp.status_code == 200
        assert "Think through this step by step" in resp.json()["prompt"]

    def test_original_prompt_preserved(self):
        client = build_app()
        prompt = "Explain quantum entanglement."
        resp = client.post("/cot/add-suffix", json={"prompt": prompt})
        assert resp.status_code == 200
        assert prompt in resp.json()["prompt"]

    def test_suffix_appended_not_prepended(self):
        client = build_app()
        prompt = "My question"
        resp = client.post("/cot/add-suffix", json={"prompt": prompt})
        assert resp.status_code == 200
        result = resp.json()["prompt"]
        assert result.index(prompt) == 0
        assert result.index(COT_SUFFIX) == len(prompt)


# ---------------------------------------------------------------------------
# Tests: CoT extraction
# ---------------------------------------------------------------------------


class TestExtractCotAnswer:
    def test_extracts_reasoning_and_answer(self):
        client = build_app()
        response = "<reasoning>\nStep 1: Consider the problem.\nStep 2: Apply logic.\n</reasoning>\n<answer>\n42\n</answer>"
        resp = client.post("/cot/extract", json={"response": response})
        assert resp.status_code == 200
        data = resp.json()
        assert "Step 1" in data["reasoning"]
        assert "Step 2" in data["reasoning"]
        assert data["answer"] == "42"

    def test_answer_only_returns_full_response_when_no_tags(self):
        client = build_app()
        resp = client.post("/cot/extract", json={"response": "The answer is 42."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reasoning"] == ""
        assert data["answer"] == "The answer is 42."

    def test_strips_whitespace_from_extracted_sections(self):
        client = build_app()
        resp = client.post("/cot/extract", json={
            "response": "<reasoning>   some reasoning   </reasoning><answer>   result   </answer>",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reasoning"] == "some reasoning"
        assert data["answer"] == "result"

    def test_multiline_reasoning(self):
        client = build_app()
        resp = client.post("/cot/extract", json={
            "response": "<reasoning>\nLine 1.\nLine 2.\nLine 3.\n</reasoning>\n<answer>Final</answer>",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Line 1." in data["reasoning"]
        assert "Line 2." in data["reasoning"]
        assert data["answer"] == "Final"

    def test_missing_answer_tag_returns_full_response(self):
        client = build_app()
        resp = client.post("/cot/extract", json={
            "response": "<reasoning>Thought about it.</reasoning>\nJust the answer text.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Thought about it." in data["reasoning"]
        assert "Just the answer text." in data["answer"]

    def test_empty_response(self):
        client = build_app()
        resp = client.post("/cot/extract", json={"response": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reasoning"] == ""
        assert data["answer"] == ""


# ---------------------------------------------------------------------------
# Tests: CoT via agent runner
# ---------------------------------------------------------------------------


class TestCotWithRunner:
    def test_agent_receives_cot_suffixed_message(self):
        cot_response = "<reasoning>I think step by step.</reasoning><answer>Yes</answer>"
        client = build_app(cot_response)
        resp = client.post("/cot-agent/run", json={"prompt": "What is the capital of France?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "step by step" in data["cot_prompt"]
        assert "I think step by step" in data["reasoning"]
        assert data["answer"] == "Yes"
