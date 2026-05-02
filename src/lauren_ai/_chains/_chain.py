from __future__ import annotations

"""Chain — composable pipeline for template | llm | parser sequences."""

import inspect
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Runnable(Protocol):
    """Protocol for objects that can participate in a :class:`Chain`.

    Any object that exposes ``async invoke(input) -> Any`` satisfies this
    protocol and can be composed with the ``|`` operator.

    :Example:

        class MyStep:
            async def invoke(self, input: Any) -> Any:
                return str(input).upper()
    """

    async def invoke(self, input: Any) -> Any:
        """Execute this step with *input* and return the result.

        :param input: The input value from the previous step (or the initial
            chain input for the first step).
        :type input: Any
        :return: Output to pass to the next step.
        :rtype: Any
        """
        ...


class RunnableLambda:
    """Wraps a sync or async callable as a :class:`Runnable`.

    The wrapped callable receives the chain's current value as its sole
    positional argument and may return any value.

    :param fn: A sync or async callable to wrap.
    :type fn: Callable

    :Example:

        upper = RunnableLambda(lambda x: x.upper())
        chain = template | llm | upper
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        """Initialise the lambda wrapper.

        :param fn: The callable to wrap.
        :type fn: Callable
        """
        self._fn = fn

    async def invoke(self, input: Any) -> Any:
        """Call the wrapped function with *input*.

        :param input: Value passed to the wrapped callable.
        :type input: Any
        :return: Result of the callable.
        :rtype: Any
        """
        result = self._fn(input)
        if inspect.isawaitable(result):
            return await result
        return result

    def __or__(self, other: Any) -> Chain:
        """Compose this lambda into a :class:`Chain`.

        :param other: The next step.
        :type other: Any
        :return: A new :class:`Chain`.
        :rtype: Chain
        """
        return Chain(steps=[self, other])


class Chain:
    """Composable pipeline: template | llm | parser.

    Created via the ``|`` operator on :class:`~lauren_ai._prompts.PromptTemplate`,
    :class:`~lauren_ai._prompts.ChatPromptTemplate`,
    :class:`~lauren_ai._module.LLMService`, or output-parser objects.

    Each step receives the output of the previous step as its sole argument
    (or the initial input for the first step).

    Usage::

        chain = template | llm_service | parser
        result = await chain.invoke({"query": "Hello"})

    :param steps: The pipeline steps, left to right.
    :type steps: list[Any]
    """

    def __init__(self, steps: list[Any]) -> None:
        """Initialise the chain with an ordered list of steps.

        :param steps: Ordered pipeline steps.
        :type steps: list[Any]
        """
        self.steps = steps

    def __or__(self, other: Any) -> Chain:
        """Append *other* to this chain and return a new chain.

        :param other: The next step to append.
        :return: A new :class:`Chain` with *other* appended.
        :rtype: Chain
        """
        return Chain(steps=self.steps + [other])

    def __ror__(self, other: Any) -> Chain:
        """Prepend *other* before this chain and return a new chain.

        :param other: The step to prepend.
        :return: A new :class:`Chain` with *other* prepended.
        :rtype: Chain
        """
        return Chain(steps=[other] + self.steps)

    async def invoke(self, input: Any = None, **kwargs: Any) -> Any:
        """Execute the pipeline left to right.

        Accepts either a positional *input* value (the canonical
        :class:`Runnable` interface) **or** keyword arguments (legacy
        ``**kwargs`` interface used by the first template step).

        When keyword arguments are provided and *input* is ``None`` they are
        merged into the initial value so that template steps still work with
        ``chain.invoke(query="Hello")``.

        :param input: Initial input value forwarded to the first step.
        :type input: Any
        :param kwargs: Keyword arguments merged into *input* when *input* is
            ``None`` (legacy interface for template-first chains).
        :return: The output of the final step.
        :raises ValueError: When an LLM step is required but none is found.
        """
        # Normalise: if called with only kwargs, treat them as the input dict.
        if input is None and kwargs:
            input = kwargs

        from lauren_ai._prompts._templates import (  # noqa: PLC0415
            ChatPromptTemplate,
            FewShotPromptTemplate,
            PromptTemplate,
        )
        from lauren_ai._module import LLMService  # noqa: PLC0415
        from lauren_ai._transport import Completion, Message  # noqa: PLC0415

        current: Any = None

        # Track which steps to skip (e.g. when a StructuredLLM wrapper consumed
        # both the LLM step and itself).
        skip_indices: set[int] = set()

        for i, step in enumerate(self.steps):
            if i in skip_indices:
                continue

            if i == 0:
                # --- First step: receive initial input ---
                if isinstance(step, (PromptTemplate, FewShotPromptTemplate)):
                    kw = input if isinstance(input, dict) else {}
                    current = step.render(**kw)
                elif isinstance(step, ChatPromptTemplate):
                    kw = input if isinstance(input, dict) else {}
                    current = step.format_messages(**kw)
                elif hasattr(step, "invoke"):
                    current = await step.invoke(input)
                elif callable(step):
                    result = step(input) if not isinstance(input, dict) else step(**input)
                    if inspect.isawaitable(result):
                        current = await result
                    else:
                        current = result
                else:
                    current = step
            else:
                # --- Subsequent steps: feed previous output ---
                if isinstance(step, LLMService):
                    # Normalise current to list[Message]
                    if isinstance(current, Message):
                        messages: list[Message] = [current]
                    elif isinstance(current, list):
                        messages = current
                    else:
                        messages = [Message(role="user", content=str(current))]  # type: ignore[arg-type]

                    # Check whether the immediately-next step is a StructuredLLM
                    # wrapper so we can route through it instead of plain complete().
                    next_step = self.steps[i + 1] if i + 1 < len(self.steps) else None
                    structured_llm = _try_get_structured_llm(next_step)
                    if structured_llm is not None:
                        current = await structured_llm._complete_messages(messages)
                        skip_indices.add(i + 1)
                    else:
                        result = await step.complete(messages)
                        if isinstance(result, Completion):
                            current = result
                        else:
                            # AsyncIterator — collect all chunks
                            chunks: list[str] = []
                            async for chunk in result:
                                if chunk.delta:
                                    chunks.append(chunk.delta)
                            from lauren_ai._transport import TokenUsage  # noqa: PLC0415

                            current = Completion(
                                id="chain",
                                model="",
                                content="".join(chunks),
                                tool_calls=[],
                                stop_reason="end_turn",
                                usage=TokenUsage(input_tokens=0, output_tokens=0),
                            )

                elif hasattr(step, "invoke"):
                    # Runnable protocol
                    current = await step.invoke(current)

                elif hasattr(step, "parse"):
                    # OutputParser protocol — sync parse()
                    if isinstance(current, Completion):
                        result = step.parse(current.content)
                    else:
                        result = step.parse(str(current))
                    if inspect.isawaitable(result):
                        current = await result
                    else:
                        current = result

                elif callable(step):
                    result = step(current)
                    if inspect.isawaitable(result):
                        current = await result
                    else:
                        current = result

        return current

    async def stream(self, **kwargs: Any) -> Any:
        """Return a streaming async iterator from the LLM step in this chain.

        Template steps before the LLM are rendered; parser steps after it are
        **not** applied during streaming (apply them to the aggregated result
        from :meth:`invoke` instead).

        :param kwargs: Keyword arguments forwarded to the first (template) step.
        :return: An async iterator of
            :class:`~lauren_ai._transport.CompletionChunk` objects.
        :raises ValueError: When no :class:`~lauren_ai._module.LLMService`
            step is found in the chain.
        """
        from lauren_ai._prompts._templates import (  # noqa: PLC0415
            ChatPromptTemplate,
            FewShotPromptTemplate,
            PromptTemplate,
        )
        from lauren_ai._module import LLMService  # noqa: PLC0415
        from lauren_ai._transport import Message  # noqa: PLC0415

        messages: list[Message] = []

        for step in self.steps:
            if isinstance(step, (PromptTemplate, FewShotPromptTemplate)):
                msg = step.render(**kwargs)
                messages = [msg]
            elif isinstance(step, ChatPromptTemplate):
                messages = step.format_messages(**kwargs)
            elif isinstance(step, LLMService):
                return step.complete_stream(messages)

        raise ValueError("No LLMService found in chain for streaming")


def chain(*steps: Any) -> Chain:
    """Convenience factory for :class:`Chain`.

    :param steps: Pipeline steps in left-to-right order.
    :type steps: Any
    :return: A :class:`Chain` containing *steps*.
    :rtype: Chain

    :Example:

        pipeline = chain(template, llm, StrOutputParser())
        result = await pipeline.invoke({"topic": "Python"})
    """
    return Chain(steps=list(steps))


def _try_get_structured_llm(step: Any) -> Any | None:
    """Return *step* if it is a StructuredLLM, otherwise ``None``.

    Uses a lazy import so the absence of the ``_structured`` module (which may
    not yet exist) does not break the chain machinery.

    :param step: The pipeline step to inspect.
    :return: The step if it is a StructuredLLM instance, else ``None``.
    :rtype: Any | None
    """
    try:
        from lauren_ai._transport._structured import StructuredLLM  # noqa: PLC0415

        if isinstance(step, StructuredLLM):
            return step
    except ImportError:
        pass
    return None
