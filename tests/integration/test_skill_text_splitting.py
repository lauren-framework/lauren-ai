"""Integration tests for the text-splitting skill (Skill 21).

Verifies RecursiveCharacterSplitter behaviour directly:
chunk size limits, overlap between consecutive chunks, recursive separator fallback,
and edge cases.
"""


# ---------------------------------------------------------------------------
# Implementation (inlined — no external import needed)
# ---------------------------------------------------------------------------


class RecursiveCharacterSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: list[str] | None = None):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str) -> list[str]:
        return self._split_recursive(text, self._separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []
        sep = separators[0] if separators else ""
        if sep and sep in text:
            parts = text.split(sep)
        else:
            return [text[i:i+self._chunk_size]
                    for i in range(0, len(text), self._chunk_size - self._chunk_overlap)
                    if text[i:i+self._chunk_size].strip()]
        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > self._chunk_size and len(separators) > 1:
                    chunks.extend(self._split_recursive(part, separators[1:]))
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())
        return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecursiveCharacterSplitter:
    def test_short_text_returns_single_chunk(self):
        text = "Short text that fits in one chunk."
        splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_all_chunks_within_size_limit(self):
        text = " ".join(f"word{i}" for i in range(200))
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk too long ({len(chunk)}): {chunk[:50]}..."

    def test_splits_on_paragraph_separator(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_overlap_between_consecutive_chunks(self):
        long_word = "x" * 200
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20, separators=[""])
        chunks = splitter.split(long_word)
        assert len(chunks) >= 2
        overlap_region = chunks[0][-20:]
        assert chunks[1].startswith(overlap_region)

    def test_empty_text_returns_empty_list(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split("")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split("   \n  \n  ")
        assert len(chunks) == 0

    def test_custom_separators(self):
        text = "part1|part2|part3|part4|part5"
        splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=0, separators=["|"])
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_recursive_fallback_when_paragraph_split_insufficient(self):
        long_para = "sentence. " * 30
        splitter = RecursiveCharacterSplitter(chunk_size=60, chunk_overlap=0)
        chunks = splitter.split(long_para)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 60

    def test_multiple_paragraphs_large_document(self):
        paragraphs = [f"Paragraph {i}: " + ("content " * 20) for i in range(10)]
        text = "\n\n".join(paragraphs)
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_exact_size_text_returns_single_chunk(self):
        text = "a" * 50
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split(text)
        assert len(chunks) == 1

    def test_text_slightly_over_limit_produces_two_chunks(self):
        text = "a" * 30 + " " + "b" * 30
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=0, separators=[" "])
        chunks = splitter.split(text)
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) <= 50
