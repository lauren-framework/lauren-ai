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
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path

import pytest


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
# Tests: StringLoader
# ---------------------------------------------------------------------------


class TestStringLoader:
    def test_returns_list_with_one_dict(self):
        loader = StringLoader()
        result = loader.load("Hello world!")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_result_has_text_and_metadata_keys(self):
        loader = StringLoader()
        result = loader.load("Hello world!")[0]
        assert "text" in result
        assert "metadata" in result

    def test_text_matches_source(self):
        loader = StringLoader()
        result = loader.load("My test content")[0]
        assert result["text"] == "My test content"

    def test_metadata_source_is_string(self):
        loader = StringLoader()
        result = loader.load("content")[0]
        assert result["metadata"]["source"] == "string"

    def test_metadata_type_is_text(self):
        loader = StringLoader()
        result = loader.load("content")[0]
        assert result["metadata"]["type"] == "text"

    def test_empty_string_returns_empty_text(self):
        loader = StringLoader()
        result = loader.load("")[0]
        assert result["text"] == ""


# ---------------------------------------------------------------------------
# Tests: TextLoader (file-based)
# ---------------------------------------------------------------------------


class TestTextLoader:
    def test_reads_file_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("File content here.", encoding="utf-8")
        loader = TextLoader()
        result = loader.load(f)[0]
        assert result["text"] == "File content here."

    def test_metadata_type_is_text(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")
        loader = TextLoader()
        result = loader.load(f)[0]
        assert result["metadata"]["type"] == "text"

    def test_metadata_source_is_file_path(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")
        loader = TextLoader()
        result = loader.load(f)[0]
        assert str(f) in result["metadata"]["source"]

    def test_multiline_content(self, tmp_path):
        content = "Line 1\nLine 2\nLine 3"
        f = tmp_path / "multi.txt"
        f.write_text(content, encoding="utf-8")
        loader = TextLoader()
        result = loader.load(f)[0]
        assert result["text"] == content


# ---------------------------------------------------------------------------
# Tests: MarkdownLoader
# ---------------------------------------------------------------------------


class TestMarkdownLoader:
    def test_strips_headings(self, tmp_path):
        content = "# Title\n## Section\nBody text"
        f = tmp_path / "doc.md"
        f.write_text(content, encoding="utf-8")
        loader = MarkdownLoader()
        result = loader.load(f)[0]
        assert "#" not in result["text"]
        assert "Body text" in result["text"]

    def test_strips_bold_markers(self, tmp_path):
        content = "This is **bold** text"
        f = tmp_path / "bold.md"
        f.write_text(content, encoding="utf-8")
        loader = MarkdownLoader()
        result = loader.load(f)[0]
        assert "**" not in result["text"]
        assert "bold" in result["text"]

    def test_strips_inline_code_markers(self, tmp_path):
        content = "Use `print()` to output"
        f = tmp_path / "code.md"
        f.write_text(content, encoding="utf-8")
        loader = MarkdownLoader()
        result = loader.load(f)[0]
        assert "`" not in result["text"]
        assert "print()" in result["text"]

    def test_metadata_type_is_markdown(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello", encoding="utf-8")
        loader = MarkdownLoader()
        result = loader.load(f)[0]
        assert result["metadata"]["type"] == "markdown"

    def test_returns_non_empty_text_for_real_content(self, tmp_path):
        f = tmp_path / "content.md"
        f.write_text("# Title\n\nSome body text here.", encoding="utf-8")
        loader = MarkdownLoader()
        result = loader.load(f)[0]
        assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Tests: HTMLLoader
# ---------------------------------------------------------------------------


class TestHTMLLoader:
    def test_strips_html_tags(self, tmp_path):
        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        f = tmp_path / "doc.html"
        f.write_text(html, encoding="utf-8")
        loader = HTMLLoader()
        result = loader.load(f)[0]
        assert "<" not in result["text"]
        assert ">" not in result["text"]

    def test_preserves_text_content(self, tmp_path):
        html = "<p>Hello <strong>world</strong></p>"
        f = tmp_path / "doc.html"
        f.write_text(html, encoding="utf-8")
        loader = HTMLLoader()
        result = loader.load(f)[0]
        assert "Hello" in result["text"]
        assert "world" in result["text"]

    def test_metadata_type_is_html(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<p>content</p>", encoding="utf-8")
        loader = HTMLLoader()
        result = loader.load(f)[0]
        assert result["metadata"]["type"] == "html"

    def test_collapses_whitespace(self, tmp_path):
        html = "<p>Hello   </p>   <p>World</p>"
        f = tmp_path / "doc.html"
        f.write_text(html, encoding="utf-8")
        loader = HTMLLoader()
        result = loader.load(f)[0]
        # Multiple spaces should be collapsed
        assert "  " not in result["text"].strip()


# ---------------------------------------------------------------------------
# Tests: Consistent output structure across all loaders
# ---------------------------------------------------------------------------


class TestLoaderOutputStructure:
    def _verify_structure(self, result: list[dict]) -> None:
        assert isinstance(result, list)
        assert len(result) >= 1
        for item in result:
            assert "text" in item
            assert "metadata" in item
            assert "source" in item["metadata"]
            assert "type" in item["metadata"]

    def test_string_loader_output_structure(self):
        loader = StringLoader()
        self._verify_structure(loader.load("test content"))

    def test_text_loader_output_structure(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("content", encoding="utf-8")
        loader = TextLoader()
        self._verify_structure(loader.load(f))

    def test_markdown_loader_output_structure(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("# Title\nBody", encoding="utf-8")
        loader = MarkdownLoader()
        self._verify_structure(loader.load(f))

    def test_html_loader_output_structure(self, tmp_path):
        f = tmp_path / "t.html"
        f.write_text("<p>content</p>", encoding="utf-8")
        loader = HTMLLoader()
        self._verify_structure(loader.load(f))
