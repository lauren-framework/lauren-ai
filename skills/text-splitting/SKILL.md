---
name: text-splitting
description: Splits long documents into overlapping chunks for RAG pipelines and context window management. Use when a document exceeds model context limits, when building vector store indexes, or when chunking text for batch embedding. Provides RecursiveCharacterSplitter that tries paragraph, line, sentence, word, and character separators in priority order.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading.

# Recursive / Semantic Text Splitting

## Quick start

```python
# NOTE: Do NOT add `from __future__ import annotations` to tool files.

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
            # Hard split by size when no separator matches
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
```

## Usage

```python
splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split(long_document_text)
# chunks: list[str], each <= 500 chars, with ~50-char overlap between consecutive chunks
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Characters of overlap between consecutive chunks |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Tried in order; falls back to hard split on `""` |

## Separator priority

1. `"\n\n"` — paragraph boundaries (preferred; preserves semantic units)
2. `"\n"` — line breaks
3. `". "` — sentence endings
4. `" "` — word boundaries
5. `""` — hard character split (last resort)

When a part is still too large after splitting on the current separator, the
splitter recurses with the next separator in the list.

## Integration with vector stores

```python
from lauren_ai import InMemoryVectorStore

splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split(document_text)

store = InMemoryVectorStore()
for i, chunk in enumerate(chunks):
    await store.add(text=chunk, metadata={"chunk_index": i})
```

## Pitfalls

- **chunk_overlap >= chunk_size** raises no error but produces infinite loops
  for hard splits — keep `chunk_overlap < chunk_size`.
- Empty strings and whitespace-only parts are filtered out automatically.
- When `chunk_size` is very small (< 10) the hard-split branch dominates and
  semantic separators are rarely used.
