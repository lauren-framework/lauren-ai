"""Integration tests for the text-splitting skill (Skill 21).

Verifies RecursiveCharacterSplitter behaviour via HTTP through a Lauren TestClient:
chunk size limits, overlap between consecutive chunks, recursive separator fallback,
and edge cases.
"""

from lauren import LaurenFactory, controller, post, module, Json
from lauren.testing import TestClient
from pydantic import BaseModel


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
# Request model
# ---------------------------------------------------------------------------


class SplitRequest(BaseModel):
    text: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None


# ---------------------------------------------------------------------------
# Controller / Module / build_app
# ---------------------------------------------------------------------------


@controller("/split")
class SplitController:
    @post("")
    async def split(self, body: Json[SplitRequest]) -> dict:
        splitter = RecursiveCharacterSplitter(
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
            separators=body.separators,
        )
        chunks = splitter.split(body.text)
        return {"chunks": chunks, "count": len(chunks)}


@module(controllers=[SplitController])
class SplitModule: ...


def build_app():
    return TestClient(LaurenFactory.create(SplitModule))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecursiveCharacterSplitter:
    def test_short_text_returns_single_chunk(self):
        client = build_app()
        text = "Short text that fits in one chunk."
        resp = client.post("/split", json={"text": text, "chunk_size": 1000, "chunk_overlap": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["chunks"][0] == text

    def test_all_chunks_within_size_limit(self):
        client = build_app()
        text = " ".join(f"word{i}" for i in range(200))
        resp = client.post("/split", json={"text": text, "chunk_size": 100, "chunk_overlap": 20})
        assert resp.status_code == 200
        data = resp.json()
        for chunk in data["chunks"]:
            assert len(chunk) <= 100, f"Chunk too long ({len(chunk)}): {chunk[:50]}..."

    def test_splits_on_paragraph_separator(self):
        client = build_app()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        resp = client.post("/split", json={"text": text, "chunk_size": 50, "chunk_overlap": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2
        for chunk in data["chunks"]:
            assert len(chunk) <= 50

    def test_overlap_between_consecutive_chunks(self):
        client = build_app()
        long_word = "x" * 200
        resp = client.post("/split", json={
            "text": long_word,
            "chunk_size": 100,
            "chunk_overlap": 20,
            "separators": [""],
        })
        assert resp.status_code == 200
        data = resp.json()
        chunks = data["chunks"]
        assert len(chunks) >= 2
        overlap_region = chunks[0][-20:]
        assert chunks[1].startswith(overlap_region)

    def test_empty_text_returns_empty_list(self):
        client = build_app()
        resp = client.post("/split", json={"text": "", "chunk_size": 100, "chunk_overlap": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["chunks"] == []

    def test_whitespace_only_returns_empty_list(self):
        client = build_app()
        resp = client.post("/split", json={"text": "   \n  \n  ", "chunk_size": 100, "chunk_overlap": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_custom_separators(self):
        client = build_app()
        text = "part1|part2|part3|part4|part5"
        resp = client.post("/split", json={
            "text": text,
            "chunk_size": 30,
            "chunk_overlap": 0,
            "separators": ["|"],
        })
        assert resp.status_code == 200
        data = resp.json()
        for chunk in data["chunks"]:
            assert len(chunk) <= 30

    def test_recursive_fallback_when_paragraph_split_insufficient(self):
        client = build_app()
        long_para = "sentence. " * 30
        resp = client.post("/split", json={"text": long_para, "chunk_size": 60, "chunk_overlap": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2
        for chunk in data["chunks"]:
            assert len(chunk) <= 60

    def test_multiple_paragraphs_large_document(self):
        client = build_app()
        paragraphs = [f"Paragraph {i}: " + ("content " * 20) for i in range(10)]
        text = "\n\n".join(paragraphs)
        resp = client.post("/split", json={"text": text, "chunk_size": 200, "chunk_overlap": 30})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 1
        for chunk in data["chunks"]:
            assert len(chunk) <= 200

    def test_exact_size_text_returns_single_chunk(self):
        client = build_app()
        text = "a" * 50
        resp = client.post("/split", json={"text": text, "chunk_size": 50, "chunk_overlap": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    def test_text_slightly_over_limit_produces_two_chunks(self):
        client = build_app()
        text = "a" * 30 + " " + "b" * 30
        resp = client.post("/split", json={
            "text": text,
            "chunk_size": 50,
            "chunk_overlap": 0,
            "separators": [" "],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        for chunk in data["chunks"]:
            assert len(chunk) <= 50
