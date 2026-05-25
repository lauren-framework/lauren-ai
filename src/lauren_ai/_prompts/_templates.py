"""Prompt template implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lauren_ai._exceptions import LaurenAIError
from lauren_ai._transport import Message

# Avoid circular imports; Chain imported lazily inside __or__ and invoke.


class PromptRenderError(LaurenAIError):
    """Raised when a prompt template is rendered with missing variables."""


@dataclass
class FewShotExample:
    """A single input/output example for few-shot prompting.

    :param input: The example input text.
    :type input: str
    :param output: The expected output text.
    :type output: str
    """

    input: str
    output: str


@dataclass
class PromptTemplate:
    """Single-message prompt template with {variable} interpolation.

    Variables in the template string are identified by curly-brace syntax
    ``{variable_name}``.  When ``input_variables`` is provided, only those
    names are required at render time; otherwise every ``{name}`` found in
    the template string is treated as required.

    Usage::

        tpl = PromptTemplate(
            template="Summarise in {words} words: {text}",
            input_variables=["text", "words"],
        )
        msg = tpl.render(text="Hello world", words=10)

    :param template: The template string with ``{variable}`` placeholders.
    :type template: str
    :param input_variables: Explicit list of required variable names.  When
        empty the variables are inferred from the template string.
    :type input_variables: list[str]
    :param role: The role of the produced :class:`~lauren_ai._transport.Message`.
        Defaults to ``"user"``.
    :type role: str
    """

    template: str
    input_variables: list[str] = field(default_factory=list)
    role: str = "user"

    def _extract_variables(self) -> set[str]:
        """Extract ``{var}`` names from the template string.

        :return: Set of variable names found in the template.
        :rtype: set[str]
        """
        return set(re.findall(r"\{(\w+)\}", self.template))

    def format(self, **kwargs: Any) -> str:
        """Render the template and return the resulting string.

        :param kwargs: Variable values to interpolate into the template.
        :type kwargs: Any
        :return: Rendered template string.
        :rtype: str
        :raises PromptRenderError: When required variables are missing from *kwargs*.
        """
        required = set(self.input_variables) if self.input_variables else self._extract_variables()
        missing = required - set(kwargs.keys())
        if missing:
            raise PromptRenderError(
                f"PromptTemplate missing variables: {sorted(missing)}. Provided: {sorted(kwargs.keys())}"
            )
        return self.template.format(**kwargs)

    def render(self, **kwargs: Any) -> Message:
        """Render the template and return a :class:`~lauren_ai._transport.Message`.

        :param kwargs: Variable values to interpolate into the template.
        :return: A :class:`~lauren_ai._transport.Message` with the rendered content.
        :rtype: Message
        :raises PromptRenderError: When required variables are missing from *kwargs*.
        """
        content = self.format(**kwargs)
        return Message(role="user", content=content)  # type: ignore[arg-type]

    async def invoke(self, input: Any) -> Any:
        """Invoke this template as a :class:`~lauren_ai._chains.Runnable`.

        When *input* is a dict the keys are used as template variables.
        When *input* is a string it is forwarded verbatim and the rendered
        content is also returned as a string.

        :param input: A dict of variable values or a plain string.
        :type input: Any
        :return: Rendered :class:`~lauren_ai._transport.Message` when *input*
            is a dict, otherwise the rendered string.
        :rtype: Any
        """
        if isinstance(input, dict):
            return self.render(**input)
        return self.format(**({} if input is None else {}))

    def __or__(self, other: Any) -> Chain:
        from lauren_ai._chains import Chain

        return Chain(steps=[self, other])


@dataclass
class ChatPromptTemplate:
    """Multi-turn prompt template producing a list of
    :class:`~lauren_ai._transport.Message` objects.

    Supports ``("role", "template {var}")`` tuples and bare
    :class:`~lauren_ai._transport.Message` instances in the *messages* list.
    Role aliases ``"human"`` → ``"user"`` and ``"ai"`` → ``"assistant"`` are
    resolved automatically.

    Usage::

        tpl = ChatPromptTemplate(
            messages=[
                ("system", "You speak {language}."),
                ("human", "{user_message}"),
            ],
            input_variables=["language", "user_message"],
        )
        msgs = tpl.render(language="French", user_message="Hello!")

    :param messages: Ordered list of ``(role, template)`` tuples or
        :class:`~lauren_ai._transport.Message` instances.
    :type messages: list[tuple[str, str] | Message]
    :param input_variables: Explicit list of required variable names.  When
        empty the variables are inferred by scanning all tuple templates.
    :type input_variables: list[str]
    """

    messages: list[tuple[str, str] | Message]
    input_variables: list[str] = field(default_factory=list)

    def _role_alias(self, role: str) -> str:
        """Resolve common role aliases to canonical values.

        :param role: Raw role string (possibly an alias).
        :type role: str
        :return: Canonical role string.
        :rtype: str
        """
        aliases = {"human": "user", "ai": "assistant", "system": "system"}
        return aliases.get(role, role)

    def _all_variables(self) -> set[str]:
        """Collect all ``{var}`` names from tuple message templates.

        :return: Set of variable names found across all template strings.
        :rtype: set[str]
        """
        vars_found: set[str] = set()
        for item in self.messages:
            if isinstance(item, tuple):
                vars_found |= set(re.findall(r"\{(\w+)\}", item[1]))
        return vars_found

    def format_messages(self, **kwargs: Any) -> list[Message]:
        """Render all messages and return them as a list.

        This is the canonical method name for producing
        :class:`~lauren_ai._transport.Message` objects from a
        :class:`ChatPromptTemplate`.  :meth:`render` is a backwards-compatible
        alias.

        :param kwargs: Variable values to interpolate into template strings.
        :type kwargs: Any
        :return: List of rendered :class:`~lauren_ai._transport.Message` objects.
        :rtype: list[Message]
        :raises PromptRenderError: When required variables are missing from *kwargs*.
        """
        required = set(self.input_variables) if self.input_variables else self._all_variables()
        missing = required - set(kwargs.keys())
        if missing:
            raise PromptRenderError(f"ChatPromptTemplate missing variables: {sorted(missing)}")
        result: list[Message] = []
        for item in self.messages:
            if isinstance(item, Message):
                result.append(item)
            else:
                role, template = item
                content = template.format(**kwargs)
                result.append(Message(role=self._role_alias(role), content=content))  # type: ignore[arg-type]
        return result

    def render(self, **kwargs: Any) -> list[Message]:
        """Backwards-compatible alias for :meth:`format_messages`.

        :param kwargs: Variable values to interpolate into template strings.
        :return: List of rendered :class:`~lauren_ai._transport.Message` objects.
        :rtype: list[Message]
        """
        return self.format_messages(**kwargs)

    async def invoke(self, input: Any) -> Any:
        """Invoke this template as a :class:`~lauren_ai._chains.Runnable`.

        :param input: A dict of variable values.
        :type input: Any
        :return: List of rendered :class:`~lauren_ai._transport.Message` objects.
        :rtype: list[Message]
        """
        if isinstance(input, dict):
            return self.format_messages(**input)
        return self.format_messages()

    def __or__(self, other: Any) -> Chain:
        from lauren_ai._chains import Chain

        return Chain(steps=[self, other])


@dataclass
class FewShotPromptTemplate:
    """Few-shot prompt template with injected examples.

    The rendered content is assembled as:
    ``<prefix> <example_separator> <example_1> <example_separator> ... <suffix>``

    Usage::

        tpl = FewShotPromptTemplate(
            prefix="Classify sentiment:\\n",
            examples=[FewShotExample("Great!", "positive")],
            example_template="{input} -> {output}",
            suffix="Input: {review}\\nSentiment:",
            input_variables=["review"],
        )
        msg = tpl.render(review="Terrible.")

    :param prefix: Text prepended before the examples.
    :type prefix: str
    :param examples: Static few-shot examples included in every render.
    :type examples: list[FewShotExample]
    :param example_template: Template string for each example.  Must contain
        ``{input}`` and ``{output}`` placeholders.
    :type example_template: str
    :param suffix: Template string appended after the examples.  May contain
        ``{variable}`` placeholders resolved from *kwargs*.
    :type suffix: str
    :param input_variables: Explicit list of required variable names for the
        *suffix*.  When empty they are inferred from the suffix template.
    :type input_variables: list[str]
    :param example_separator: String used to join prefix, examples, and suffix.
        Defaults to ``"\\n\\n"``.
    :type example_separator: str
    :param role: Role of the produced :class:`~lauren_ai._transport.Message`.
        Defaults to ``"user"``.
    :type role: str
    """

    prefix: str
    examples: list[FewShotExample]
    example_template: str
    suffix: str
    input_variables: list[str] = field(default_factory=list)
    example_separator: str = "\n\n"
    role: str = "user"

    def format(self, **kwargs: Any) -> str:
        """Render the few-shot prompt and return the resulting string.

        :param kwargs: Variable values for the *suffix* template.
        :type kwargs: Any
        :return: Rendered prompt string.
        :rtype: str
        :raises PromptRenderError: When required suffix variables are missing.
        """
        required = set(self.input_variables) if self.input_variables else set(re.findall(r"\{(\w+)\}", self.suffix))
        missing = required - set(kwargs.keys())
        if missing:
            raise PromptRenderError(f"FewShotPromptTemplate missing variables: {sorted(missing)}")
        all_examples = list(self.examples)
        example_strs = [self.example_template.format(input=ex.input, output=ex.output) for ex in all_examples]
        suffix_str = self.suffix.format(**kwargs)
        parts = [self.prefix] + example_strs + [suffix_str]
        return self.example_separator.join(p for p in parts if p)

    def render(
        self,
        *,
        extra_examples: list[FewShotExample] | None = None,
        **kwargs: Any,
    ) -> Message:
        """Render the few-shot prompt and return a single message.

        :param extra_examples: Additional examples appended after the static
            *examples* list.  Useful for dynamic in-context examples.
        :type extra_examples: list[FewShotExample] | None
        :param kwargs: Variable values for the *suffix* template.
        :return: A rendered :class:`~lauren_ai._transport.Message`.
        :rtype: Message
        :raises PromptRenderError: When required suffix variables are missing
            from *kwargs*.
        """
        required = set(self.input_variables) if self.input_variables else set(re.findall(r"\{(\w+)\}", self.suffix))
        missing = required - set(kwargs.keys())
        if missing:
            raise PromptRenderError(f"FewShotPromptTemplate missing variables: {sorted(missing)}")
        all_examples = list(self.examples) + (extra_examples or [])
        example_strs = [self.example_template.format(input=ex.input, output=ex.output) for ex in all_examples]
        suffix_str = self.suffix.format(**kwargs)
        parts = [self.prefix] + example_strs + [suffix_str]
        content = self.example_separator.join(p for p in parts if p)
        return Message(role="user", content=content)  # type: ignore[arg-type]

    async def invoke(self, input: Any) -> Any:
        """Invoke this template as a :class:`~lauren_ai._chains.Runnable`.

        :param input: A dict of variable values.
        :type input: Any
        :return: A rendered :class:`~lauren_ai._transport.Message`.
        :rtype: Message
        """
        if isinstance(input, dict):
            return self.render(**input)
        return self.render()

    def __or__(self, other: Any) -> Chain:
        from lauren_ai._chains import Chain

        return Chain(steps=[self, other])


# Avoid circular import — Chain is defined in _chains; referenced only from __or__
# at call time, so the forward reference via TYPE_CHECKING is not needed here.
Chain = Any  # noqa: N816 — overwritten at call sites by real import
