"""Integration tests for the few-shot-injection skill (Skill 23).

Verifies FewShotPromptBuilder builds system prompts and message history
correctly via HTTP through a Lauren TestClient, and that the built-in
FewShotPromptTemplate works as expected.
"""

from dataclasses import dataclass

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient
from lauren_ai import FewShotPromptTemplate, FewShotExample as BuiltinFewShotExample
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------


@dataclass
class FewShotExample:
    input: str
    output: str


class FewShotPromptBuilder:
    def __init__(self, task_description: str, examples: list[FewShotExample]):
        self._task = task_description
        self._examples = examples

    def build_system_prompt(self) -> str:
        lines = [self._task, "", "Examples:"]
        for i, ex in enumerate(self._examples, 1):
            lines.append(f"\nExample {i}:")
            lines.append(f"Input: {ex.input}")
            lines.append(f"Output: {ex.output}")
        lines.append("\nNow answer the user's input in the same format.")
        return "\n".join(lines)

    def build_messages(self) -> list[dict]:
        messages = []
        for ex in self._examples:
            messages.append({"role": "user", "content": ex.input})
            messages.append({"role": "assistant", "content": ex.output})
        return messages


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ExampleItem(BaseModel):
    input: str
    output: str


class FewShotRequest(BaseModel):
    task: str
    examples: list[ExampleItem]


class BuiltinFewShotRequest(BaseModel):
    prefix: str
    examples: list[ExampleItem]
    example_template: str
    suffix: str
    input_variables: list[str]
    render_vars: dict
    extra_examples: list[ExampleItem] = []


# ---------------------------------------------------------------------------
# Controller / Module / build_app
# ---------------------------------------------------------------------------


@controller("/few-shot")
class FewShotController:
    @post("/system")
    async def build_system(self, body: Json[FewShotRequest]) -> dict:
        examples = [FewShotExample(input=e.input, output=e.output) for e in body.examples]
        builder = FewShotPromptBuilder(task_description=body.task, examples=examples)
        return {"prompt": builder.build_system_prompt()}

    @post("/messages")
    async def build_messages(self, body: Json[FewShotRequest]) -> dict:
        examples = [FewShotExample(input=e.input, output=e.output) for e in body.examples]
        builder = FewShotPromptBuilder(task_description=body.task, examples=examples)
        return {"messages": builder.build_messages()}

    @post("/builtin-render")
    async def builtin_render(self, body: Json[BuiltinFewShotRequest]) -> dict:
        tpl = FewShotPromptTemplate(
            prefix=body.prefix,
            examples=[BuiltinFewShotExample(input=e.input, output=e.output) for e in body.examples],
            example_template=body.example_template,
            suffix=body.suffix,
            input_variables=body.input_variables,
        )
        extra = [BuiltinFewShotExample(input=e.input, output=e.output) for e in body.extra_examples]
        msg = tpl.render(extra_examples=extra if extra else None, **body.render_vars)
        return {"content": msg.content}


@module(controllers=[FewShotController])
class FewShotModule: ...


def build_app():
    return TestClient(LaurenFactory.create(FewShotModule))


_EXAMPLES = [
    {"input": "I love this product!", "output": "positive"},
    {"input": "This is terrible.", "output": "negative"},
    {"input": "Works okay.", "output": "neutral"},
]

_TASK = "Classify sentiment as positive or negative."


# ---------------------------------------------------------------------------
# Tests: FewShotPromptBuilder — system prompt
# ---------------------------------------------------------------------------


class TestFewShotPromptBuilder:
    def test_system_prompt_contains_task_description(self):
        client = build_app()
        resp = client.post("/few-shot/system", json={"task": _TASK, "examples": _EXAMPLES})
        assert resp.status_code == 200
        assert "Classify sentiment" in resp.json()["prompt"]

    def test_system_prompt_contains_all_examples(self):
        client = build_app()
        resp = client.post("/few-shot/system", json={"task": _TASK, "examples": _EXAMPLES})
        assert resp.status_code == 200
        prompt = resp.json()["prompt"]
        assert "I love this product!" in prompt
        assert "positive" in prompt
        assert "This is terrible." in prompt
        assert "negative" in prompt
        assert "Works okay." in prompt
        assert "neutral" in prompt

    def test_system_prompt_has_example_numbers(self):
        client = build_app()
        resp = client.post("/few-shot/system", json={"task": _TASK, "examples": _EXAMPLES})
        assert resp.status_code == 200
        prompt = resp.json()["prompt"]
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" in prompt

    def test_system_prompt_ends_with_instruction(self):
        client = build_app()
        resp = client.post("/few-shot/system", json={"task": _TASK, "examples": _EXAMPLES})
        assert resp.status_code == 200
        assert "Now answer the user's input in the same format." in resp.json()["prompt"]

    def test_empty_examples_produces_minimal_prompt(self):
        client = build_app()
        resp = client.post("/few-shot/system", json={"task": "Do something.", "examples": []})
        assert resp.status_code == 200
        prompt = resp.json()["prompt"]
        assert "Do something." in prompt
        assert "Examples:" in prompt

    def test_empty_examples_produces_empty_message_list(self):
        client = build_app()
        resp = client.post("/few-shot/messages", json={"task": "Do something.", "examples": []})
        assert resp.status_code == 200
        assert resp.json()["messages"] == []


# ---------------------------------------------------------------------------
# Tests: build_messages — alternating roles
# ---------------------------------------------------------------------------


class TestFewShotMessages:
    def test_build_messages_alternates_user_assistant(self):
        client = build_app()
        resp = client.post("/few-shot/messages", json={"task": _TASK, "examples": _EXAMPLES})
        assert resp.status_code == 200
        messages = resp.json()["messages"]
        assert len(messages) == 6
        for i in range(0, len(messages), 2):
            assert messages[i]["role"] == "user"
            assert messages[i + 1]["role"] == "assistant"

    def test_build_messages_contains_example_content(self):
        client = build_app()
        resp = client.post("/few-shot/messages", json={"task": _TASK, "examples": _EXAMPLES})
        assert resp.status_code == 200
        messages = resp.json()["messages"]
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
        assert "I love this product!" in user_contents
        assert "positive" in assistant_contents


# ---------------------------------------------------------------------------
# Tests: Built-in FewShotPromptTemplate
# ---------------------------------------------------------------------------


class TestBuiltinFewShotPromptTemplate:
    def test_builtin_renders_examples_in_prompt(self):
        client = build_app()
        resp = client.post("/few-shot/builtin-render", json={
            "prefix": "Classify sentiment:\n",
            "examples": [
                {"input": "Great!", "output": "positive"},
                {"input": "Awful.", "output": "negative"},
            ],
            "example_template": "{input} -> {output}",
            "suffix": "Input: {review}\nSentiment:",
            "input_variables": ["review"],
            "render_vars": {"review": "Pretty good."},
        })
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "Great!" in content
        assert "positive" in content
        assert "Awful." in content
        assert "negative" in content
        assert "Pretty good." in content

    def test_builtin_extra_examples_appended(self):
        client = build_app()
        resp = client.post("/few-shot/builtin-render", json={
            "prefix": "Examples:",
            "examples": [{"input": "A", "output": "1"}],
            "example_template": "{input}->{output}",
            "suffix": "Q: {q}",
            "input_variables": ["q"],
            "render_vars": {"q": "test"},
            "extra_examples": [{"input": "B", "output": "2"}],
        })
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "B->2" in content
        assert "A->1" in content
