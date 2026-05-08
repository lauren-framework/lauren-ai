"""Integration tests for the system-prompt-templating skill (Skill 22).

Verifies SystemPromptBuilder and PromptTemplate behaviour via HTTP through a
Lauren TestClient.
"""

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient
from lauren_ai import PromptTemplate
from pydantic import BaseModel


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
# Request models
# ---------------------------------------------------------------------------


class TemplateRequest(BaseModel):
    template: str
    vars: dict


class BuildRequest(BaseModel):
    role: str = ""
    context: str = ""
    instruction: str = ""
    constraints: list[str] = []


class BuiltinTemplateRequest(BaseModel):
    template: str
    input_variables: list[str] = []
    vars: dict


# ---------------------------------------------------------------------------
# Controller / Module / build_app
# ---------------------------------------------------------------------------


@controller("/prompt")
class PromptController:
    @post("/template")
    async def render_template(self, body: Json[TemplateRequest]) -> dict:
        tpl = SimplePromptTemplate(body.template)
        rendered = tpl.render(**body.vars)
        return {"rendered": rendered}

    @post("/build")
    async def build_system(self, body: Json[BuildRequest]) -> dict:
        builder = SystemPromptBuilder()
        if body.role:
            builder.add_role(body.role)
        if body.context:
            builder.add_context(body.context)
        if body.instruction:
            builder.add_instruction(body.instruction)
        if body.constraints:
            builder.add_constraints(*body.constraints)
        return {"system": builder.build()}

    @post("/builtin-template")
    async def builtin_template(self, body: Json[BuiltinTemplateRequest]) -> dict:
        tpl = PromptTemplate(
            template=body.template,
            input_variables=body.input_variables or list(body.vars.keys()),
        )
        msg = tpl.render(**body.vars)
        return {"content": msg.content}

    @post("/builtin-format")
    async def builtin_format(self, body: Json[BuiltinTemplateRequest]) -> dict:
        tpl = PromptTemplate(template=body.template)
        text = tpl.format(**body.vars)
        return {"text": text}


@module(controllers=[PromptController])
class PromptModule: ...


def build_app():
    return TestClient(LaurenFactory.create(PromptModule))


# ---------------------------------------------------------------------------
# Tests: SystemPromptBuilder
# ---------------------------------------------------------------------------


class TestSystemPromptBuilder:
    def test_add_role_appears_in_prompt(self):
        client = build_app()
        resp = client.post("/prompt/build", json={"role": "a financial analyst"})
        assert resp.status_code == 200
        assert "You are a financial analyst." in resp.json()["system"]

    def test_add_context_appears_in_prompt(self):
        client = build_app()
        resp = client.post("/prompt/build", json={"context": "User is a CEO."})
        assert resp.status_code == 200
        assert "Context: User is a CEO." in resp.json()["system"]

    def test_add_instruction_appears_in_prompt(self):
        client = build_app()
        instruction = "Always provide quantitative estimates."
        resp = client.post("/prompt/build", json={"instruction": instruction})
        assert resp.status_code == 200
        assert instruction in resp.json()["system"]

    def test_add_constraints_formats_bullet_list(self):
        client = build_app()
        resp = client.post("/prompt/build", json={
            "constraints": ["Never give stock picks", "Always mention risks"],
        })
        assert resp.status_code == 200
        system = resp.json()["system"]
        assert "- Never give stock picks" in system
        assert "- Always mention risks" in system
        assert "Constraints:" in system

    def test_sections_separated_by_double_newline(self):
        client = build_app()
        resp = client.post("/prompt/build", json={"role": "an assistant", "context": "Testing"})
        assert resp.status_code == 200
        assert "\n\n" in resp.json()["system"]

    def test_full_builder_chain(self):
        client = build_app()
        resp = client.post("/prompt/build", json={
            "role": "a financial analyst specializing in risk assessment",
            "context": "User is a portfolio manager at a hedge fund",
            "instruction": "Always provide quantitative estimates with confidence intervals.",
            "constraints": ["Never give specific stock picks", "Always mention risks"],
        })
        assert resp.status_code == 200
        system = resp.json()["system"]
        assert "financial analyst" in system
        assert "portfolio manager" in system
        assert "confidence intervals" in system
        assert "Never give specific stock picks" in system
        assert "Always mention risks" in system

    def test_empty_builder_returns_empty_string(self):
        client = build_app()
        resp = client.post("/prompt/build", json={})
        assert resp.status_code == 200
        assert resp.json()["system"] == ""

    def test_add_constraints_with_no_constraints_omits_section(self):
        client = build_app()
        resp = client.post("/prompt/build", json={"role": "an assistant", "constraints": []})
        assert resp.status_code == 200
        assert "Constraints:" not in resp.json()["system"]


# ---------------------------------------------------------------------------
# Tests: SimplePromptTemplate
# ---------------------------------------------------------------------------


class TestSimplePromptTemplate:
    def test_single_variable_substitution(self):
        client = build_app()
        resp = client.post("/prompt/template", json={
            "template": "You are {role}.",
            "vars": {"role": "an assistant"},
        })
        assert resp.status_code == 200
        assert resp.json()["rendered"] == "You are an assistant."

    def test_multiple_variable_substitution(self):
        client = build_app()
        resp = client.post("/prompt/template", json={
            "template": "You are {role} working for {company}.",
            "vars": {"role": "a lawyer", "company": "Acme"},
        })
        assert resp.status_code == 200
        assert resp.json()["rendered"] == "You are a lawyer working for Acme."

    def test_missing_variable_leaves_placeholder(self):
        client = build_app()
        resp = client.post("/prompt/template", json={
            "template": "Hello {name}!",
            "vars": {},
        })
        assert resp.status_code == 200
        assert "{name}" in resp.json()["rendered"]

    def test_extra_variables_are_ignored(self):
        client = build_app()
        resp = client.post("/prompt/template", json={
            "template": "Hello {name}!",
            "vars": {"name": "World", "extra": "ignored"},
        })
        assert resp.status_code == 200
        assert resp.json()["rendered"] == "Hello World!"


# ---------------------------------------------------------------------------
# Tests: Built-in PromptTemplate
# ---------------------------------------------------------------------------


class TestBuiltinPromptTemplate:
    def test_builtin_prompt_template_render_returns_message(self):
        client = build_app()
        resp = client.post("/prompt/builtin-template", json={
            "template": "Summarise in {words} words: {text}",
            "input_variables": ["words", "text"],
            "vars": {"words": "10", "text": "Hello world"},
        })
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "10" in content
        assert "Hello world" in content

    def test_builtin_prompt_template_format_returns_string(self):
        client = build_app()
        resp = client.post("/prompt/builtin-format", json={
            "template": "Role: {role}",
            "vars": {"role": "analyst"},
        })
        assert resp.status_code == 200
        assert resp.json()["text"] == "Role: analyst"
