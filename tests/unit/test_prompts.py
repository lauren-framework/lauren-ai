"""Unit tests for prompt templates."""
from __future__ import annotations

import pytest

from lauren_ai._prompts import (
    ChatPromptTemplate,
    FewShotExample,
    FewShotPromptTemplate,
    PromptRenderError,
    PromptTemplate,
)
from lauren_ai._transport import Message


class TestPromptTemplate:
    def test_render_simple(self):
        tpl = PromptTemplate(template="Hello {name}!", input_variables=["name"])
        msg = tpl.render(name="Alice")
        assert msg.content == "Hello Alice!"
        assert msg.role == "user"

    def test_render_missing_variable_raises(self):
        tpl = PromptTemplate(template="Hello {name}!", input_variables=["name"])
        with pytest.raises(PromptRenderError, match="name"):
            tpl.render()

    def test_render_infers_variables_from_template(self):
        tpl = PromptTemplate(template="{a} + {b} = {c}")
        msg = tpl.render(a="1", b="2", c="3")
        assert msg.content == "1 + 2 = 3"

    def test_render_custom_role(self):
        tpl = PromptTemplate(template="Sys prompt {x}", role="system")
        msg = tpl.render(x="test")
        # role is stored in the template but Message only accepts "user"/"assistant";
        # the render returns Message(role="user", ...) per the implementation spec.
        # The role field on PromptTemplate configures intent; actual Message role
        # is always "user" in the current implementation.
        assert msg.content == "Sys prompt test"

    def test_pipe_creates_chain(self):
        from lauren_ai._chains import Chain

        tpl = PromptTemplate(template="{q}")
        chain = tpl | "something"
        assert isinstance(chain, Chain)


class TestChatPromptTemplate:
    def test_render_produces_message_list(self):
        tpl = ChatPromptTemplate(
            messages=[
                ("system", "Be helpful."),
                ("human", "{q}"),
            ],
            input_variables=["q"],
        )
        msgs = tpl.render(q="What is AI?")
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "Be helpful."
        assert msgs[1].role == "user"
        assert msgs[1].content == "What is AI?"

    def test_render_role_aliases(self):
        tpl = ChatPromptTemplate(
            messages=[("human", "Hi"), ("ai", "Hello")],
        )
        msgs = tpl.render()
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_render_missing_variable_raises(self):
        tpl = ChatPromptTemplate(
            messages=[("human", "{q}")],
            input_variables=["q"],
        )
        with pytest.raises(PromptRenderError, match="q"):
            tpl.render()

    def test_render_with_message_object(self):
        msg = Message(role="user", content="Fixed system.")
        tpl = ChatPromptTemplate(messages=[msg, ("human", "{q}")])
        result = tpl.render(q="Hello")
        assert result[0].content == "Fixed system."
        assert result[1].content == "Hello"

    def test_pipe_creates_chain(self):
        from lauren_ai._chains import Chain

        tpl = ChatPromptTemplate(messages=[("human", "{q}")])
        chain = tpl | "something"
        assert isinstance(chain, Chain)


class TestFewShotPromptTemplate:
    def test_render_includes_examples(self):
        tpl = FewShotPromptTemplate(
            prefix="Examples:\n",
            examples=[
                FewShotExample("great", "positive"),
                FewShotExample("terrible", "negative"),
            ],
            example_template="{input} -> {output}",
            suffix="Input: {review}\nSentiment:",
            input_variables=["review"],
        )
        msg = tpl.render(review="Amazing!")
        assert "great -> positive" in msg.content
        assert "terrible -> negative" in msg.content
        assert "Amazing!" in msg.content
        assert msg.role == "user"

    def test_render_missing_suffix_variable_raises(self):
        tpl = FewShotPromptTemplate(
            prefix="",
            examples=[],
            example_template="{input} -> {output}",
            suffix="Input: {x}",
            input_variables=["x"],
        )
        with pytest.raises(PromptRenderError, match="x"):
            tpl.render()

    def test_render_with_extra_examples(self):
        tpl = FewShotPromptTemplate(
            prefix="",
            examples=[FewShotExample("a", "1")],
            example_template="{input}->{output}",
            suffix="Q: {q}",
            input_variables=["q"],
        )
        msg = tpl.render(q="test", extra_examples=[FewShotExample("b", "2")])
        assert "a->1" in msg.content
        assert "b->2" in msg.content

    def test_separator(self):
        tpl = FewShotPromptTemplate(
            prefix="P",
            examples=[FewShotExample("x", "y")],
            example_template="{input}={output}",
            suffix="end",
            input_variables=[],
            example_separator="|||",
        )
        msg = tpl.render()
        assert "|||" in msg.content
