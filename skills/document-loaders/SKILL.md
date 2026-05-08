---
name: document-loaders
description: Loads documents from multiple formats (plain text, Markdown, HTML, PDF) into a unified list of {text, metadata} dicts for indexing into a KnowledgeBase. Use when ingesting diverse document types into a RAG pipeline, preprocessing documents before chunking, or building a multi-format document processing pipeline.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Document Loaders for Multiple Formats

## Overview

All loaders implement a common interface: `load(source) -> list[dict]` where
each dict has `{text: str, metadata: dict}`.  This makes them composable and
easy to swap.

---

## Loader implementations

```python
from abc import ABC, abstractmethod
from pathlib import Path
import re


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
        clean = re.sub(r"#{1,6}\s+", "", text)
        clean = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", clean)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        return [{"text": clean.strip(), "metadata": {"source": str(source), "type": "markdown"}}]


class HTMLLoader(DocumentLoader):
    def load(self, source: str | Path) -> list[dict]:
        html = Path(source).read_text(encoding="utf-8")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return [{"text": text, "metadata": {"source": str(source), "type": "html"}}]


class StringLoader(DocumentLoader):
    """Loads a document from a raw string (useful for testing)."""
    def load(self, source: str | Path) -> list[dict]:
        return [{"text": str(source), "metadata": {"source": "string", "type": "text"}}]
```

---

## PDF loader (pypdf)

```python
class PDFLoader(DocumentLoader):
    def load(self, source: str | Path) -> list[dict]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF loading: pip install pypdf")

        reader = PdfReader(str(source))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text,
                    "metadata": {"source": str(source), "type": "pdf", "page": i + 1},
                })
        return pages
```

---

## Feeding loaders into KnowledgeBase

The built-in `lauren_ai._knowledge.TextLoader` (a different class) expects
a string or file path.  To use custom loaders with `KnowledgeBase`, convert
each `{text, metadata}` dict to a `Document` and call `store.upsert()`:

```python
from lauren_ai._knowledge import Document, KnowledgeBase
from lauren_ai._memory._vector import InMemoryVectorStore

kb = KnowledgeBase(store=InMemoryVectorStore())

loader = MarkdownLoader()
for doc_dict in loader.load("README.md"):
    doc = Document(content=doc_dict["text"], metadata=doc_dict["metadata"])
    await kb._store.upsert(doc.content, id=doc.id, metadata=doc.metadata)
```

Or use `TextLoader` with `is_file=False` to pass pre-loaded text:

```python
from lauren_ai._knowledge import TextLoader as KBTextLoader

for doc_dict in MarkdownLoader().load("README.md"):
    await kb.load(KBTextLoader(doc_dict["text"], is_file=False))
```

---

## Batch loading multiple files

```python
import glob

def load_directory(directory: str, loader: DocumentLoader) -> list[dict]:
    docs = []
    for path in glob.glob(f"{directory}/**/*", recursive=True):
        try:
            docs.extend(loader.load(path))
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return docs
```

---

## Reference

- `lauren_ai._knowledge`: `KnowledgeBase`, `TextLoader` (built-in), `Document`
- `lauren_ai._memory._vector`: `InMemoryVectorStore`
- Skills: `rag-pipeline`, `embedding-model-ingestion`
