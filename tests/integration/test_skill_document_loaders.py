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
# Helper: write temp file and return path
# ---------------------------------------------------------------------------


def _write_temp(content: str, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, encoding="utf-8", delete=False) as f:
        f.write(content)
        return f.name


# ---------------------------------------------------------------------------
# Tests: StringLoader (direct Python)
# ---------------------------------------------------------------------------


class TestStringLoader:
    def test_returns_list_with_one_dict(self):
        loader = StringLoader()
        result = loader.load("Hello world!")
        assert len(result) == 1

    def test_result_has_text_and_metadata_keys(self):
        loader = StringLoader()
        result = loader.load("Hello world!")
        assert "text" in result[0]
        assert "metadata" in result[0]

    def test_text_matches_source(self):
        loader = StringLoader()
        result = loader.load("My test content")
        assert result[0]["text"] == "My test content"

    def test_metadata_source_is_string(self):
        loader = StringLoader()
        result = loader.load("content")
        assert result[0]["metadata"]["source"] == "string"

    def test_metadata_type_is_text(self):
        loader = StringLoader()
        result = loader.load("content")
        assert result[0]["metadata"]["type"] == "text"

    def test_empty_string_returns_empty_text(self):
        loader = StringLoader()
        result = loader.load("")
        assert result[0]["text"] == ""


# ---------------------------------------------------------------------------
# Tests: TextLoader (file-based, direct Python)
# ---------------------------------------------------------------------------


class TestTextLoader:
    def test_reads_file_content(self):
        tmp_path = _write_temp("File content here.", ".txt")
        try:
            loader = TextLoader()
            result = loader.load(tmp_path)
            assert result[0]["text"] == "File content here."
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_metadata_type_is_text(self):
        tmp_path = _write_temp("content", ".txt")
        try:
            loader = TextLoader()
            result = loader.load(tmp_path)
            assert result[0]["metadata"]["type"] == "text"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_metadata_source_is_file_path(self):
        tmp_path = _write_temp("content", ".txt")
        try:
            loader = TextLoader()
            result = loader.load(tmp_path)
            assert tmp_path in result[0]["metadata"]["source"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multiline_content(self):
        content = "Line 1\nLine 2\nLine 3"
        tmp_path = _write_temp(content, ".txt")
        try:
            loader = TextLoader()
            result = loader.load(tmp_path)
            assert result[0]["text"] == content
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: MarkdownLoader (direct Python)
# ---------------------------------------------------------------------------


class TestMarkdownLoader:
    def test_strips_headings(self):
        tmp_path = _write_temp("# Title\n## Section\nBody text", ".md")
        try:
            loader = MarkdownLoader()
            result = loader.load(tmp_path)
            assert "#" not in result[0]["text"]
            assert "Body text" in result[0]["text"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_strips_bold_markers(self):
        tmp_path = _write_temp("This is **bold** text", ".md")
        try:
            loader = MarkdownLoader()
            result = loader.load(tmp_path)
            assert "**" not in result[0]["text"]
            assert "bold" in result[0]["text"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_strips_inline_code_markers(self):
        tmp_path = _write_temp("Use `print()` to output", ".md")
        try:
            loader = MarkdownLoader()
            result = loader.load(tmp_path)
            assert "`" not in result[0]["text"]
            assert "print()" in result[0]["text"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_metadata_type_is_markdown(self):
        tmp_path = _write_temp("# Hello", ".md")
        try:
            loader = MarkdownLoader()
            result = loader.load(tmp_path)
            assert result[0]["metadata"]["type"] == "markdown"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_returns_non_empty_text_for_real_content(self):
        tmp_path = _write_temp("# Title\n\nSome body text here.", ".md")
        try:
            loader = MarkdownLoader()
            result = loader.load(tmp_path)
            assert len(result[0]["text"]) > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: HTMLLoader (direct Python)
# ---------------------------------------------------------------------------


class TestHTMLLoader:
    def test_strips_html_tags(self):
        tmp_path = _write_temp("<html><body><h1>Title</h1><p>Content</p></body></html>", ".html")
        try:
            loader = HTMLLoader()
            result = loader.load(tmp_path)
            text = result[0]["text"]
            assert "<" not in text
            assert ">" not in text
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_preserves_text_content(self):
        tmp_path = _write_temp("<p>Hello <strong>world</strong></p>", ".html")
        try:
            loader = HTMLLoader()
            result = loader.load(tmp_path)
            text = result[0]["text"]
            assert "Hello" in text
            assert "world" in text
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_metadata_type_is_html(self):
        tmp_path = _write_temp("<p>content</p>", ".html")
        try:
            loader = HTMLLoader()
            result = loader.load(tmp_path)
            assert result[0]["metadata"]["type"] == "html"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_collapses_whitespace(self):
        tmp_path = _write_temp("<p>Hello   </p>   <p>World</p>", ".html")
        try:
            loader = HTMLLoader()
            result = loader.load(tmp_path)
            assert "  " not in result[0]["text"].strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: Consistent output structure across all loaders (direct Python)
# ---------------------------------------------------------------------------


class TestLoaderOutputStructure:
    def _check_structure(self, result: list[dict]) -> bool:
        return (
            isinstance(result, list)
            and len(result) >= 1
            and "text" in result[0]
            and "metadata" in result[0]
            and "source" in result[0]["metadata"]
            and "type" in result[0]["metadata"]
        )

    def test_string_loader_output_structure(self):
        loader = StringLoader()
        result = loader.load("test content")
        assert self._check_structure(result)

    def test_text_loader_output_structure(self):
        tmp_path = _write_temp("content", ".txt")
        try:
            loader = TextLoader()
            result = loader.load(tmp_path)
            assert self._check_structure(result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_markdown_loader_output_structure(self):
        tmp_path = _write_temp("# Title\nBody", ".md")
        try:
            loader = MarkdownLoader()
            result = loader.load(tmp_path)
            assert self._check_structure(result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_html_loader_output_structure(self):
        tmp_path = _write_temp("<p>content</p>", ".html")
        try:
            loader = HTMLLoader()
            result = loader.load(tmp_path)
            assert self._check_structure(result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
