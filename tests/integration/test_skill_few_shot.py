"""Integration tests for the few-shot-injection skill (Skill 23).

Verifies FewShotPromptBuilder builds system prompts and message history
correctly, and that the built-in FewShotPromptTemplate works as expected.
"""
import pytest

from lauren_ai import FewShotPromptTemplate, FewShotExample as BuiltinFewShotExample


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


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
# Tests
# ---------------------------------------------------------------------------


class TestFewShotPromptBuilder:
    def _make_builder(self) -> FewShotPromptBuilder:
        return FewShotPromptBuilder(
            task_description="Classify sentiment as positive or negative.",
            examples=[
                FewShotExample("I love this product!", "positive"),
                FewShotExample("This is terrible.", "negative"),
                FewShotExample("Works okay.", "neutral"),
            ],
        )

    def test_system_prompt_contains_task_description(self):
        builder = self._make_builder()
        prompt = builder.build_system_prompt()
        assert "Classify sentiment" in prompt

    def test_system_prompt_contains_all_examples(self):
        builder = self._make_builder()
        prompt = builder.build_system_prompt()
        assert "I love this product!" in prompt
        assert "positive" in prompt
        assert "This is terrible." in prompt
        assert "negative" in prompt
        assert "Works okay." in prompt
        assert "neutral" in prompt

    def test_system_prompt_has_example_numbers(self):
        builder = self._make_builder()
        prompt = builder.build_system_prompt()
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" in prompt

    def test_system_prompt_ends_with_instruction(self):
        builder = self._make_builder()
        prompt = builder.build_system_prompt()
        assert "Now answer the user's input in the same format." in prompt

    def test_build_messages_alternates_user_assistant(self):
        builder = self._make_builder()
        messages = builder.build_messages()
        assert len(messages) == 6  # 3 examples * 2 messages each
        for i in range(0, len(messages), 2):
            assert messages[i]["role"] == "user"
            assert messages[i + 1]["role"] == "assistant"

    def test_build_messages_contains_example_content(self):
        builder = self._make_builder()
        messages = builder.build_messages()
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
        assert "I love this product!" in user_contents
        assert "positive" in assistant_contents

    def test_empty_examples_produces_minimal_prompt(self):
        builder = FewShotPromptBuilder(task_description="Do something.", examples=[])
        prompt = builder.build_system_prompt()
        assert "Do something." in prompt
        assert "Examples:" in prompt

    def test_empty_examples_produces_empty_message_list(self):
        builder = FewShotPromptBuilder(task_description="Do something.", examples=[])
        messages = builder.build_messages()
        assert messages == []


class TestBuiltinFewShotPromptTemplate:
    def test_builtin_renders_examples_in_prompt(self):
        tpl = FewShotPromptTemplate(
            prefix="Classify sentiment:\n",
            examples=[
                BuiltinFewShotExample(input="Great!", output="positive"),
                BuiltinFewShotExample(input="Awful.", output="negative"),
            ],
            example_template="{input} -> {output}",
            suffix="Input: {review}\nSentiment:",
            input_variables=["review"],
        )
        msg = tpl.render(review="Pretty good.")
        assert "Great!" in msg.content
        assert "positive" in msg.content
        assert "Awful." in msg.content
        assert "negative" in msg.content
        assert "Pretty good." in msg.content

    def test_builtin_extra_examples_appended(self):
        tpl = FewShotPromptTemplate(
            prefix="Examples:",
            examples=[BuiltinFewShotExample(input="A", output="1")],
            example_template="{input}->{output}",
            suffix="Q: {q}",
            input_variables=["q"],
        )
        extra = [BuiltinFewShotExample(input="B", output="2")]
        msg = tpl.render(extra_examples=extra, q="test")
        assert "B->2" in msg.content
        assert "A->1" in msg.content
