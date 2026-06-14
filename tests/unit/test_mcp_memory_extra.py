"""Extra unit tests for lauren_ai.mcp._memory.

Existing coverage in test_mcp_new_features.py already covers:
- McpConversationStore.load (not found via McpCallError)
- McpConversationStore.load (parsed JSON)
- McpConversationStore.save
- McpConversationStore.delete
- McpUserMemoryStore.save
- McpUserMemoryStore.get_all
- McpUserMemoryStore.delete

These tests extend into additional edge cases.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from lauren_ai.mcp._memory import (
    McpConversationStore,
    McpMemoryFact,
    McpUserMemoryStore,
    _extract_resource_text,
)

# ---------------------------------------------------------------------------
# _extract_resource_text helper
# ---------------------------------------------------------------------------


class TestExtractResourceText:
    def test_none_returns_empty(self):
        assert _extract_resource_text(None) == ""

    def test_empty_contents_returns_empty(self):
        raw = MagicMock()
        raw.contents = []
        assert _extract_resource_text(raw) == ""

    def test_item_with_text_attr(self):
        item = MagicMock()
        item.text = "hello"
        raw = MagicMock()
        raw.contents = [item]
        assert _extract_resource_text(raw) == "hello"

    def test_item_as_dict(self):
        raw = {"contents": [{"text": "from dict"}]}
        assert _extract_resource_text(raw) == "from dict"

    def test_item_with_empty_text_skipped(self):
        item1 = MagicMock()
        item1.text = ""  # falsy — skip
        item2 = MagicMock()
        item2.text = "second"
        raw = MagicMock()
        raw.contents = [item1, item2]
        assert _extract_resource_text(raw) == "second"


# ---------------------------------------------------------------------------
# McpConversationStore
# ---------------------------------------------------------------------------


class TestMcpConversationStoreExtra:
    async def test_load_returns_empty_on_unknown_exception(self):
        """Any exception that is NOT a known MCP error code still returns []."""
        client = AsyncMock()
        client.read_resource.side_effect = RuntimeError("something unexpected")

        store = McpConversationStore(client)
        result = await store.load("conv-unknown")
        assert result == []

    async def test_load_returns_empty_when_no_text(self):
        """Resource found but has empty text → returns []."""
        client = AsyncMock()
        mock_result = MagicMock()
        # contents list with item that has no useful text
        item = MagicMock()
        item.text = ""
        mock_result.contents = [item]
        client.read_resource.return_value = mock_result

        store = McpConversationStore(client)
        result = await store.load("conv-empty-text")
        assert result == []

    async def test_load_parses_valid_json(self):
        """Content item with text field that contains valid JSON is parsed."""
        client = AsyncMock()
        content_item = MagicMock()
        content_item.text = json.dumps({"messages": [{"role": "user", "content": "hello"}]})
        mock_result = MagicMock()
        mock_result.contents = [content_item]
        client.read_resource.return_value = mock_result

        store = McpConversationStore(client)
        result = await store.load("conv-valid")
        assert isinstance(result, dict)
        assert result["messages"][0]["content"] == "hello"

    async def test_load_returns_empty_on_invalid_json(self):
        """Invalid JSON in text field returns []."""
        client = AsyncMock()
        content_item = MagicMock()
        content_item.text = "not valid json {{{"
        mock_result = MagicMock()
        mock_result.contents = [content_item]
        client.read_resource.return_value = mock_result

        store = McpConversationStore(client)
        result = await store.load("conv-bad-json")
        assert result == []

    async def test_save_normalises_list_snapshot(self):
        """List snapshot should be wrapped in {'messages': ..., 'summary': None}."""
        client = AsyncMock()
        client.call_tool.return_value = []

        store = McpConversationStore(client)
        messages = [{"role": "user", "content": "hi"}]
        await store.save("conv-list", messages)

        client.call_tool.assert_awaited_once()
        call_args = client.call_tool.call_args[0]
        snapshot_str = call_args[1]["snapshot"]
        snapshot = json.loads(snapshot_str)
        assert "messages" in snapshot
        assert snapshot["summary"] is None
        assert snapshot["messages"] == messages

    async def test_save_skips_empty_conversation_id(self):
        """Empty conversation_id → early return, call_tool NOT called."""
        client = AsyncMock()
        store = McpConversationStore(client)
        await store.save("", {"messages": []})
        client.call_tool.assert_not_awaited()

    async def test_save_swallows_exception(self):
        """Exceptions from call_tool are caught silently."""
        client = AsyncMock()
        client.call_tool.side_effect = RuntimeError("server down")

        store = McpConversationStore(client)
        # Should not raise
        await store.save("conv-err", {"messages": []})

    async def test_save_uses_tool_prefix(self):
        """tool_prefix is prepended to save_conversation."""
        client = AsyncMock()
        client.call_tool.return_value = []

        store = McpConversationStore(client, tool_prefix="v2_")
        await store.save("conv-prefix", {"messages": []})

        call_args = client.call_tool.call_args[0]
        assert call_args[0] == "v2_save_conversation"

    async def test_delete_swallows_exception(self):
        """Exceptions from call_tool during delete are suppressed."""
        client = AsyncMock()
        client.call_tool.side_effect = RuntimeError("delete failed")

        store = McpConversationStore(client)
        await store.delete("conv-del")  # should not raise

    async def test_load_with_non_standard_error_code(self):
        """Exception with non-standard code returns [] (falls through to generic handler)."""
        client = AsyncMock()
        exc = Exception("generic error")
        exc.code = -99999  # not in (-32002, -32600, -32601)
        client.read_resource.side_effect = exc

        store = McpConversationStore(client)
        result = await store.load("conv-generic-err")
        assert result == []

    async def test_load_with_known_error_codes(self):
        """Error codes -32002, -32600, -32601 all return []."""
        for code in (-32002, -32600, -32601):
            client = AsyncMock()
            exc = Exception("mcp error")
            exc.code = code
            client.read_resource.side_effect = exc

            store = McpConversationStore(client)
            result = await store.load("c")
            assert result == [], f"Expected [] for code {code}"


# ---------------------------------------------------------------------------
# McpUserMemoryStore
# ---------------------------------------------------------------------------


class TestMcpUserMemoryStoreExtra:
    async def test_get_all_returns_empty_on_error(self):
        """call_tool raising an exception returns []."""
        client = AsyncMock()
        client.call_tool.side_effect = RuntimeError("network error")

        store = McpUserMemoryStore(client)
        result = await store.get_all("user-err")
        assert result == []

    async def test_get_all_handles_string_raw_result(self):
        """If call_tool returns a raw string, it should still be parsed."""
        client = AsyncMock()
        facts = [{"key": "color", "value": "blue"}]
        # Simulate call_tool returning a plain string (not a list of dicts)
        client.call_tool.return_value = json.dumps(facts)

        store = McpUserMemoryStore(client)
        result = await store.get_all("user-str")
        assert len(result) == 1
        assert result[0].key == "color"
        assert result[0].value == "blue"

    async def test_get_all_empty_list_result(self):
        """call_tool returns empty list → get_all returns []."""
        client = AsyncMock()
        client.call_tool.return_value = []

        store = McpUserMemoryStore(client)
        result = await store.get_all("user-empty")
        assert result == []

    async def test_get_all_invalid_json(self):
        """Invalid JSON in text field returns []."""
        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": "not-json"}]

        store = McpUserMemoryStore(client)
        result = await store.get_all("user-bad")
        assert result == []

    async def test_get_all_with_tool_prefix(self):
        """tool_prefix is applied to get_user_facts tool name."""
        client = AsyncMock()
        client.call_tool.return_value = [{"type": "text", "text": json.dumps([])}]

        store = McpUserMemoryStore(client, tool_prefix="mem_")
        await store.get_all("user-pfx")

        call_args = client.call_tool.call_args[0]
        assert call_args[0] == "mem_get_user_facts"

    async def test_save_swallows_exception(self):
        """Exceptions from save call_tool are silently suppressed."""
        client = AsyncMock()
        client.call_tool.side_effect = RuntimeError("save failed")

        store = McpUserMemoryStore(client)
        await store.save("user-1", "key", "val")  # should not raise

    async def test_delete_swallows_exception(self):
        """Exceptions from delete call_tool are silently suppressed."""
        client = AsyncMock()
        client.call_tool.side_effect = RuntimeError("delete failed")

        store = McpUserMemoryStore(client)
        await store.delete("user-1", "key")  # should not raise

    async def test_get_all_object_items_fallback(self):
        """Items that are not dicts but have .key / .value attributes are handled."""
        client = AsyncMock()
        # Simulate a list of objects (not dicts)
        fact_obj = MagicMock()
        fact_obj.key = "theme"
        fact_obj.value = "dark"
        facts_json = json.dumps([{"key": "theme", "value": "dark"}])
        client.call_tool.return_value = [{"type": "text", "text": facts_json}]

        store = McpUserMemoryStore(client)
        result = await store.get_all("user-obj")
        assert result[0].key == "theme"
        assert result[0].value == "dark"


# ---------------------------------------------------------------------------
# McpMemoryFact dataclass
# ---------------------------------------------------------------------------


class TestMcpMemoryFact:
    def test_defaults(self):
        fact = McpMemoryFact(key="lang", value="Python")
        assert fact.key == "lang"
        assert fact.value == "Python"
        assert fact.user_id == ""
        assert fact.metadata == {}

    def test_explicit_user_id(self):
        fact = McpMemoryFact(key="k", value="v", user_id="u-1", metadata={"src": "chat"})
        assert fact.user_id == "u-1"
        assert fact.metadata["src"] == "chat"
