"""Integration tests for document loaders (multiple formats) (Skill 20).

Tests:
  - StringLoader returns text and metadata with correct type
  - TextLoader reads from a temp file
  - MarkdownLoader strips headings and bold/italic/code syntax
  - HTMLLoader strips HTML tags
  - All loaders return list of dicts with 'text' and 'metadata' keys
  - Metadata contains 'source' and 'type' keys
  - StringLoader metadata type is 'text'
  - TextLoader metadata type is 'text'
  - MarkdownLoader metadata type is 'markdown'
  - HTMLLoader metadata type is 'html'
  - Loaders return non-empty text for non-empty input
  - Multiple formats produce consistent output structure

NOTE: No from __future__ import annotations.
"""

import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, module, post
from lauren.testing import TestClient


# ---------------------------------------------------------------------------
# Document loader implementations under test
# ---------------------------------------------------------------------------


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str | Path) -> list[dict]:
        """Load a document and return list of {text, metadata} dicts."""
        ...


class TextLoader(DocumentLoader):
    def load(self, source: str | Path) -> list[dict]:
        text = Path(source).read_text(encoding="utf-8")
        return [{"text": text, "metadata": {"source": str(source), "type": "text"}}]


class MarkdownLoader(DocumentLoader):
    def load(self, source: str | Path) -> list[dict]:
        text = Path(source).read_text(encoding="utf-8")
        # Remove headings
        clean = re.sub(r"#{1,6}\s+", "", text)
        # Remove bold/italic markers
        clean = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", clean)
        # Remove inline code markers
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        return [{"text": clean.strip(), "metadata": {"source": str(source), "type": "markdown"}}]


class HTMLLoader(DocumentLoader):
    def load(self, source: str | Path) -> list[dict]:
        html = Path(source).read_text(encoding="utf-8")
        # Strip tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return [{"text": text, "metadata": {"source": str(source), "type": "html"}}]


class StringLoader(DocumentLoader):
    """Loads a document from a raw string (useful for testing)."""

    def load(self, source: str | Path) -> list[dict]:
        return [{"text": str(source), "metadata": {"source": "string", "type": "text"}}]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class _ContentRequest(BaseModel):
    content: str


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


@controller("/loaders")
class LoaderController:
    @post("/string")
    async def string_loader(self, body: Json[_ContentRequest]) -> dict:
        loader = StringLoader()
        result = loader.load(body.content)
        return {
            "count": len(result),
            "text": result[0]["text"],
            "source": result[0]["metadata"]["source"],
            "type": result[0]["metadata"]["type"],
            "has_text_key": "text" in result[0],
            "has_metadata_key": "metadata" in result[0],
            "has_source_in_meta": "source" in result[0]["metadata"],
            "has_type_in_meta": "type" in result[0]["metadata"],
        }

    @post("/text-file")
    async def text_file_loader(self, body: Json[_ContentRequest]) -> dict:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write(body.content)
            tmp_path = f.name
        loader = TextLoader()
        result = loader.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        return {
            "text": result[0]["text"],
            "type": result[0]["metadata"]["type"],
            "source_has_path": tmp_path in result[0]["metadata"]["source"],
        }

    @post("/markdown")
    async def markdown_loader(self, body: Json[_ContentRequest]) -> dict:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", delete=False
        ) as f:
            f.write(body.content)
            tmp_path = f.name
        loader = MarkdownLoader()
        result = loader.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        return {
            "text": result[0]["text"],
            "type": result[0]["metadata"]["type"],
            "no_hashes": "#" not in result[0]["text"],
            "no_double_stars": "**" not in result[0]["text"],
            "no_backticks": "`" not in result[0]["text"],
        }

    @post("/html")
    async def html_loader(self, body: Json[_ContentRequest]) -> dict:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", encoding="utf-8", delete=False
        ) as f:
            f.write(body.content)
            tmp_path = f.name
        loader = HTMLLoader()
        result = loader.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        text = result[0]["text"]
        return {
            "text": text,
            "type": result[0]["metadata"]["type"],
            "no_lt": "<" not in text,
            "no_gt": ">" not in text,
            "no_double_space": "  " not in text.strip(),
        }

    @post("/structure-check")
    async def structure_check(self, body: Json[dict]) -> dict:
        loader_type = body.get("loader_type", "string")
        content = body.get("content", "test")

        if loader_type == "string":
            loader = StringLoader()
            result = loader.load(content)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f".{loader_type}",
                encoding="utf-8",
                delete=False,
            ) as f:
                f.write(content)
                tmp_path = f.name
            if loader_type == "txt":
                loader = TextLoader()
            elif loader_type == "md":
                loader = MarkdownLoader()
            else:
                loader = HTMLLoader()
            result = loader.load(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

        valid = (
            isinstance(result, list)
            and len(result) >= 1
            and "text" in result[0]
            and "metadata" in result[0]
            and "source" in result[0]["metadata"]
            and "type" in result[0]["metadata"]
        )
        return {"valid": valid}


@module(controllers=[LoaderController])
class DocumentLoadersModule: ...


def build_app() -> TestClient:
    return TestClient(LaurenFactory.create(DocumentLoadersModule))


# ---------------------------------------------------------------------------
# Tests: StringLoader
# ---------------------------------------------------------------------------


class TestStringLoader:
    def test_returns_list_with_one_dict(self):
        client = build_app()
        r = client.post("/loaders/string", json={"content": "Hello world!"})
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_result_has_text_and_metadata_keys(self):
        client = build_app()
        r = client.post("/loaders/string", json={"content": "Hello world!"})
        assert r.status_code == 200
        data = r.json()
        assert data["has_text_key"] is True
        assert data["has_metadata_key"] is True

    def test_text_matches_source(self):
        client = build_app()
        r = client.post("/loaders/string", json={"content": "My test content"})
        assert r.status_code == 200
        assert r.json()["text"] == "My test content"

    def test_metadata_source_is_string(self):
        client = build_app()
        r = client.post("/loaders/string", json={"content": "content"})
        assert r.status_code == 200
        assert r.json()["source"] == "string"

    def test_metadata_type_is_text(self):
        client = build_app()
        r = client.post("/loaders/string", json={"content": "content"})
        assert r.status_code == 200
        assert r.json()["type"] == "text"

    def test_empty_string_returns_empty_text(self):
        client = build_app()
        r = client.post("/loaders/string", json={"content": ""})
        assert r.status_code == 200
        assert r.json()["text"] == ""


# ---------------------------------------------------------------------------
# Tests: TextLoader (file-based)
# ---------------------------------------------------------------------------


class TestTextLoader:
    def test_reads_file_content(self):
        client = build_app()
        r = client.post("/loaders/text-file", json={"content": "File content here."})
        assert r.status_code == 200
        assert r.json()["text"] == "File content here."

    def test_metadata_type_is_text(self):
        client = build_app()
        r = client.post("/loaders/text-file", json={"content": "content"})
        assert r.status_code == 200
        assert r.json()["type"] == "text"

    def test_metadata_source_is_file_path(self):
        client = build_app()
        r = client.post("/loaders/text-file", json={"content": "content"})
        assert r.status_code == 200
        assert r.json()["source_has_path"] is True

    def test_multiline_content(self):
        client = build_app()
        content = "Line 1\nLine 2\nLine 3"
        r = client.post("/loaders/text-file", json={"content": content})
        assert r.status_code == 200
        assert r.json()["text"] == content


# ---------------------------------------------------------------------------
# Tests: MarkdownLoader
# ---------------------------------------------------------------------------


class TestMarkdownLoader:
    def test_strips_headings(self):
        client = build_app()
        r = client.post("/loaders/markdown", json={"content": "# Title\n## Section\nBody text"})
        assert r.status_code == 200
        data = r.json()
        assert data["no_hashes"] is True
        assert "Body text" in data["text"]

    def test_strips_bold_markers(self):
        client = build_app()
        r = client.post("/loaders/markdown", json={"content": "This is **bold** text"})
        assert r.status_code == 200
        data = r.json()
        assert data["no_double_stars"] is True
        assert "bold" in data["text"]

    def test_strips_inline_code_markers(self):
        client = build_app()
        r = client.post("/loaders/markdown", json={"content": "Use `print()` to output"})
        assert r.status_code == 200
        data = r.json()
        assert data["no_backticks"] is True
        assert "print()" in data["text"]

    def test_metadata_type_is_markdown(self):
        client = build_app()
        r = client.post("/loaders/markdown", json={"content": "# Hello"})
        assert r.status_code == 200
        assert r.json()["type"] == "markdown"

    def test_returns_non_empty_text_for_real_content(self):
        client = build_app()
        r = client.post(
            "/loaders/markdown", json={"content": "# Title\n\nSome body text here."}
        )
        assert r.status_code == 200
        assert len(r.json()["text"]) > 0


# ---------------------------------------------------------------------------
# Tests: HTMLLoader
# ---------------------------------------------------------------------------


class TestHTMLLoader:
    def test_strips_html_tags(self):
        client = build_app()
        r = client.post(
            "/loaders/html",
            json={"content": "<html><body><h1>Title</h1><p>Content</p></body></html>"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["no_lt"] is True
        assert data["no_gt"] is True

    def test_preserves_text_content(self):
        client = build_app()
        r = client.post(
            "/loaders/html",
            json={"content": "<p>Hello <strong>world</strong></p>"},
        )
        assert r.status_code == 200
        text = r.json()["text"]
        assert "Hello" in text
        assert "world" in text

    def test_metadata_type_is_html(self):
        client = build_app()
        r = client.post("/loaders/html", json={"content": "<p>content</p>"})
        assert r.status_code == 200
        assert r.json()["type"] == "html"

    def test_collapses_whitespace(self):
        client = build_app()
        r = client.post(
            "/loaders/html", json={"content": "<p>Hello   </p>   <p>World</p>"}
        )
        assert r.status_code == 200
        assert r.json()["no_double_space"] is True


# ---------------------------------------------------------------------------
# Tests: Consistent output structure across all loaders
# ---------------------------------------------------------------------------


class TestLoaderOutputStructure:
    def test_string_loader_output_structure(self):
        client = build_app()
        r = client.post("/loaders/structure-check", json={"loader_type": "string", "content": "test content"})
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_text_loader_output_structure(self):
        client = build_app()
        r = client.post("/loaders/structure-check", json={"loader_type": "txt", "content": "content"})
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_markdown_loader_output_structure(self):
        client = build_app()
        r = client.post(
            "/loaders/structure-check", json={"loader_type": "md", "content": "# Title\nBody"}
        )
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_html_loader_output_structure(self):
        client = build_app()
        r = client.post(
            "/loaders/structure-check", json={"loader_type": "html", "content": "<p>content</p>"}
        )
        assert r.status_code == 200
        assert r.json()["valid"] is True
