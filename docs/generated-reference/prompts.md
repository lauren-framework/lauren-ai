# Prompt Templates

Reusable, composable prompt builders.

### `PromptTemplate`

Single-message prompt template with {variable} interpolation.

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

### `ChatPromptTemplate`

Multi-turn prompt template producing a list of :class:`~lauren_ai._transport.Message` objects.

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

### `FewShotPromptTemplate`

Few-shot prompt template with injected examples.

The rendered content is assembled as:
``<prefix> <example_separator> <example_1> <example_separator> ... <suffix>``

Usage::

    tpl = FewShotPromptTemplate(
        prefix="Classify sentiment:\n",
        examples=[FewShotExample("Great!", "positive")],
        example_template="{input} -> {output}",
        suffix="Input: {review}\nSentiment:",
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
    Defaults to ``"\n\n"``.
:type example_separator: str
:param role: Role of the produced :class:`~lauren_ai._transport.Message`.
    Defaults to ``"user"``.
:type role: str

### `FewShotExample`

A single input/output example for few-shot prompting.

:param input: The example input text.
:type input: str
:param output: The expected output text.
:type output: str

