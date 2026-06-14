"""Extra unit tests for lauren_ai.mcp._prompt_template.

Existing coverage in test_mcp_new_features.py already covers:
- test_name_property
- test_render_single_user_message_returns_string
- test_render_multi_message_returns_list
- test_argument_names_fetches_from_list
- test_callable_interface

These tests extend into additional edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai.mcp._prompt_template import (
    McpPromptTemplate,
    McpSystemPromptBuilder,
    _convert_get_prompt_result,
    list_mcp_prompts,
)

# ---------------------------------------------------------------------------
# _convert_get_prompt_result edge cases
# ---------------------------------------------------------------------------


class TestConvertGetPromptResult:
    def test_none_returns_empty_string(self):
        result = _convert_get_prompt_result(None)
        assert result == ""

    def test_empty_messages_returns_empty_string(self):
        mock_result = MagicMock()
        mock_result.messages = []
        result = _convert_get_prompt_result(mock_result)
        assert result == ""

    def test_single_user_message_dict_content_returns_string(self):
        """Single user message whose content is a dict with 'text' key."""
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        # content is a dict (not an object with .text)
        msg.content = {"text": "dict content"}
        mock_result.messages = [msg]

        result = _convert_get_prompt_result(mock_result)
        assert result == "dict content"

    def test_single_user_message_str_content_returns_string(self):
        """Single user message whose content is already a plain string."""
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = "plain string content"
        mock_result.messages = [msg]

        result = _convert_get_prompt_result(mock_result)
        assert result == "plain string content"

    def test_single_non_user_role_falls_through_to_list(self):
        """Single message with non-user role should return a list (not a string)."""
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "system"
        msg.content = MagicMock()
        msg.content.text = "Be helpful."
        mock_result.messages = [msg]

        result = _convert_get_prompt_result(mock_result)
        # Single non-user message falls through to the multi-message list path
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful."

    def test_multi_message_dict_format(self):
        """dict-style result with 'messages' key is also handled."""
        result = _convert_get_prompt_result({"messages": []})
        assert result == ""

    def test_multi_message_list_content_with_fallback(self):
        """Content with no text attr and no text key falls back to str()."""
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "assistant"
        content = MagicMock(spec=[])  # no 'text' attribute
        msg.content = content
        mock_result.messages = [msg, msg]  # 2 messages → multi path

        # Should not raise; content falls back to str(content)
        result = _convert_get_prompt_result(mock_result)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# McpPromptTemplate.render edge cases
# ---------------------------------------------------------------------------


class TestMcpPromptTemplateExtra:
    async def test_render_empty_messages_returns_empty_string(self):
        client = AsyncMock()
        mock_result = MagicMock()
        mock_result.messages = []
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "empty_prompt")
        result = await tmpl.render()
        assert result == ""

    async def test_render_non_user_role_falls_through_to_list(self):
        """Single assistant-role message → list, not str."""
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = MagicMock()
        msg.content.text = "I am the assistant."
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "assistant_only")
        result = await tmpl.render()
        assert isinstance(result, list)
        assert result[0]["role"] == "assistant"

    async def test_invoke_with_dict_input(self):
        """invoke({}) calls render with no keyword args."""
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.text = "empty kwargs"
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "p")
        result = await tmpl.invoke({})
        assert result == "empty kwargs"
        # get_prompt should be called with no arguments (None) because dict is empty
        client.get_prompt.assert_awaited_once_with("p", None)

    async def test_invoke_with_non_dict_input(self):
        """invoke('foo') treats input as non-dict, renders with no args."""
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.text = "non-dict"
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "p")
        result = await tmpl.invoke("some string input")
        assert result == "non-dict"
        # With non-dict, kwargs is {}, so arguments=None
        client.get_prompt.assert_awaited_once_with("p", None)

    def test_pipe_operator_returns_chain(self):
        """template | other should return a Chain."""
        from lauren_ai._chains import Chain

        client = MagicMock()
        tmpl = McpPromptTemplate(client, "p")
        other = MagicMock()
        result = tmpl | other
        assert isinstance(result, Chain)

    async def test_argument_names_raises_when_not_found(self):
        """argument_names raises ValueError when prompt not in catalogue."""
        client = AsyncMock()
        schema = MagicMock()
        schema.name = "other_prompt"
        client.list_prompts.return_value = [schema]

        tmpl = McpPromptTemplate(client, "missing_prompt")
        with pytest.raises(ValueError, match="not found"):
            await tmpl.argument_names()

    async def test_argument_names_empty_arguments_field(self):
        """argument_names returns [] when schema.arguments is None/empty."""
        client = AsyncMock()
        schema = MagicMock()
        schema.name = "my_prompt"
        schema.arguments = None
        client.list_prompts.return_value = [schema]

        tmpl = McpPromptTemplate(client, "my_prompt")
        names = await tmpl.argument_names()
        assert names == []

    async def test_render_with_arguments_passed_through(self):
        """Named arguments are forwarded to get_prompt."""
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.text = "rendered with args"
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        tmpl = McpPromptTemplate(client, "templated")
        result = await tmpl.render(topic="AI", language="en")
        assert result == "rendered with args"
        client.get_prompt.assert_awaited_once_with("templated", {"topic": "AI", "language": "en"})


# ---------------------------------------------------------------------------
# McpSystemPromptBuilder
# ---------------------------------------------------------------------------


class TestMcpSystemPromptBuilder:
    async def test_builder_returns_string_from_single_user_message(self):
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.text = "You are a helpful assistant."
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        builder = McpSystemPromptBuilder(client, "system")
        result = await builder()
        assert result == "You are a helpful assistant."

    async def test_mcp_system_prompt_builder_list_result_joined_with_newlines(self):
        """When render returns a list, the builder joins with newlines."""
        client = AsyncMock()
        mock_result = MagicMock()
        msg1 = MagicMock()
        msg1.role = "system"
        msg1.content = MagicMock()
        msg1.content.text = "Line one."
        msg2 = MagicMock()
        msg2.role = "user"
        msg2.content = MagicMock()
        msg2.content.text = "Line two."
        # Two messages → list result
        mock_result.messages = [msg1, msg2]
        client.get_prompt.return_value = mock_result

        builder = McpSystemPromptBuilder(client, "multi")
        result = await builder()
        # The builder joins list items via newline
        assert "Line one." in result
        assert "Line two." in result
        assert "\n" in result

    async def test_builder_with_static_arguments(self):
        """Static keyword arguments are forwarded to render."""
        client = AsyncMock()
        mock_result = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = MagicMock()
        msg.content.text = "Parameterised system prompt."
        mock_result.messages = [msg]
        client.get_prompt.return_value = mock_result

        builder = McpSystemPromptBuilder(client, "sys_tmpl", lang="en", style="formal")
        await builder()
        client.get_prompt.assert_awaited_once_with("sys_tmpl", {"lang": "en", "style": "formal"})

    async def test_builder_list_items_with_dict_content(self):
        """Builder joins list where items have dict content (get 'content' key)."""
        client = AsyncMock()
        mock_result = MagicMock()
        # Build two messages so render returns list
        msgs = []
        for text in ["First line.", "Second line."]:
            m = MagicMock()
            m.role = "system"
            m.content = MagicMock()
            m.content.text = text
            msgs.append(m)
        mock_result.messages = msgs
        client.get_prompt.return_value = mock_result

        builder = McpSystemPromptBuilder(client, "joined")
        result = await builder()
        assert "First line." in result
        assert "Second line." in result


# ---------------------------------------------------------------------------
# list_mcp_prompts helper
# ---------------------------------------------------------------------------


class TestListMcpPrompts:
    async def test_list_mcp_prompts_returns_templates(self):
        client = AsyncMock()
        s1 = MagicMock()
        s1.name = "greeting"
        s2 = MagicMock()
        s2.name = "farewell"
        client.list_prompts.return_value = [s1, s2]

        templates = await list_mcp_prompts(client)
        assert len(templates) == 2
        names = [t.prompt_name for t in templates]
        assert "greeting" in names
        assert "farewell" in names

    async def test_list_mcp_prompts_empty(self):
        client = AsyncMock()
        client.list_prompts.return_value = []
        templates = await list_mcp_prompts(client)
        assert templates == []
