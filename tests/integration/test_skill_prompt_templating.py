"""Integration tests for the system-prompt-templating skill (Skill 22).

Verifies SystemPromptBuilder and PromptTemplate behaviour directly.
"""

from lauren_ai import PromptTemplate


# ---------------------------------------------------------------------------
# Implementation (inlined)
# ---------------------------------------------------------------------------


class SimplePromptTemplate:
    """Simple {key} substitution template for system prompts."""

    def __init__(self, template: str):
        self._template = template

    def render(self, **kwargs) -> str:
        result = self._template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class SystemPromptBuilder:
    def __init__(self):
        self._sections: list[str] = []

    def add_role(self, role: str) -> "SystemPromptBuilder":
        self._sections.append(f"You are {role}.")
        return self

    def add_context(self, context: str) -> "SystemPromptBuilder":
        self._sections.append(f"Context: {context}")
        return self

    def add_instruction(self, instruction: str) -> "SystemPromptBuilder":
        self._sections.append(instruction)
        return self

    def add_constraints(self, *constraints: str) -> "SystemPromptBuilder":
        if constraints:
            self._sections.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
        return self

    def build(self) -> str:
        return "\n\n".join(self._sections)


# ---------------------------------------------------------------------------
# Tests: SystemPromptBuilder
# ---------------------------------------------------------------------------


class TestSystemPromptBuilder:
    def test_add_role_appears_in_prompt(self):
        system = SystemPromptBuilder().add_role("a financial analyst").build()
        assert "You are a financial analyst." in system

    def test_add_context_appears_in_prompt(self):
        system = SystemPromptBuilder().add_context("User is a CEO.").build()
        assert "Context: User is a CEO." in system

    def test_add_instruction_appears_in_prompt(self):
        instruction = "Always provide quantitative estimates."
        system = SystemPromptBuilder().add_instruction(instruction).build()
        assert instruction in system

    def test_add_constraints_formats_bullet_list(self):
        system = SystemPromptBuilder().add_constraints(
            "Never give stock picks", "Always mention risks"
        ).build()
        assert "- Never give stock picks" in system
        assert "- Always mention risks" in system
        assert "Constraints:" in system

    def test_sections_separated_by_double_newline(self):
        system = SystemPromptBuilder().add_role("an assistant").add_context("Testing").build()
        assert "\n\n" in system

    def test_full_builder_chain(self):
        system = (
            SystemPromptBuilder()
            .add_role("a financial analyst specializing in risk assessment")
            .add_context("User is a portfolio manager at a hedge fund")
            .add_instruction("Always provide quantitative estimates with confidence intervals.")
            .add_constraints("Never give specific stock picks", "Always mention risks")
            .build()
        )
        assert "financial analyst" in system
        assert "portfolio manager" in system
        assert "confidence intervals" in system
        assert "Never give specific stock picks" in system
        assert "Always mention risks" in system

    def test_empty_builder_returns_empty_string(self):
        assert SystemPromptBuilder().build() == ""

    def test_add_constraints_with_no_constraints_omits_section(self):
        system = SystemPromptBuilder().add_role("an assistant").add_constraints().build()
        assert "Constraints:" not in system


# ---------------------------------------------------------------------------
# Tests: SimplePromptTemplate
# ---------------------------------------------------------------------------


class TestSimplePromptTemplate:
    def test_single_variable_substitution(self):
        tpl = SimplePromptTemplate("You are {role}.")
        assert tpl.render(role="an assistant") == "You are an assistant."

    def test_multiple_variable_substitution(self):
        tpl = SimplePromptTemplate("You are {role} working for {company}.")
        assert tpl.render(role="a lawyer", company="Acme") == "You are a lawyer working for Acme."

    def test_missing_variable_leaves_placeholder(self):
        tpl = SimplePromptTemplate("Hello {name}!")
        assert "{name}" in tpl.render()

    def test_extra_variables_are_ignored(self):
        tpl = SimplePromptTemplate("Hello {name}!")
        assert tpl.render(name="World", extra="ignored") == "Hello World!"


# ---------------------------------------------------------------------------
# Tests: Built-in PromptTemplate
# ---------------------------------------------------------------------------


class TestBuiltinPromptTemplate:
    def test_builtin_prompt_template_render_returns_message(self):
        tpl = PromptTemplate(
            template="Summarise in {words} words: {text}",
            input_variables=["words", "text"],
        )
        msg = tpl.render(words="10", text="Hello world")
        assert "10" in msg.content
        assert "Hello world" in msg.content

    def test_builtin_prompt_template_format_returns_string(self):
        tpl = PromptTemplate(template="Role: {role}")
        text = tpl.format(role="analyst")
        assert text == "Role: analyst"
