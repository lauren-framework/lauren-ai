# Prompt Templates

Reusable, composable prompt builders.

### `PromptTemplate`

Single-message prompt template with {variable} interpolation.

Variables in the template string are identified by curly-brace syntax
`{variable_name}`.  When `input_variables` is provided, only those
names are required at render time; otherwise every `{name}` found in
the template string is treated as required.

Usage::

    tpl = PromptTemplate(
        template="Summarise in {words} words: {text}",
        input_variables=["text", "words"],
    )
    msg = tpl.render(text="Hello world", words=10)

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `template` | `str` | The template string with `{variable}` placeholders. |
| `input_variables` | `list[str]` | Explicit list of required variable names.  When
empty the variables are inferred from the template string. |
| `role` | `str` | The role of the produced `Message`.
Defaults to `"user"`. |

### `ChatPromptTemplate`

Multi-turn prompt template producing a list of
`Message` objects.

Supports `("role", "template {var}")` tuples and bare
`Message` instances in the *messages* list.
Role aliases `"human"` → `"user"` and `"ai"` → `"assistant"` are
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

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `messages` | `list[tuple[str, str] | Message]` | Ordered list of `(role, template)` tuples or
`Message` instances. |
| `input_variables` | `list[str]` | Explicit list of required variable names.  When
empty the variables are inferred by scanning all tuple templates. |

### `FewShotPromptTemplate`

Few-shot prompt template with injected examples.

The rendered content is assembled as:
`<prefix> <example_separator> <example_1> <example_separator> ... <suffix>`

Usage::

    tpl = FewShotPromptTemplate(
        prefix="Classify sentiment:\n",
        examples=[FewShotExample("Great!", "positive")],
        example_template="{input} -> {output}",
        suffix="Input: {review}\nSentiment:",
        input_variables=["review"],
    )
    msg = tpl.render(review="Terrible.")

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `prefix` | `str` | Text prepended before the examples. |
| `examples` | `list[FewShotExample]` | Static few-shot examples included in every render. |
| `example_template` | `str` | Template string for each example.  Must contain
`{input}` and `{output}` placeholders. |
| `suffix` | `str` | Template string appended after the examples.  May contain
`{variable}` placeholders resolved from *kwargs*. |
| `input_variables` | `list[str]` | Explicit list of required variable names for the
*suffix*.  When empty they are inferred from the suffix template. |
| `example_separator` | `str` | String used to join prefix, examples, and suffix.
Defaults to `"\n\n"`. |
| `role` | `str` | Role of the produced `Message`.
Defaults to `"user"`. |

### `FewShotExample`

A single input/output example for few-shot prompting.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `input` | `str` | The example input text. |
| `output` | `str` | The expected output text. |
