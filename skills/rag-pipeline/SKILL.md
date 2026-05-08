---
name: rag-pipeline
description: Builds a full RAG (Retrieval-Augmented Generation) pipeline using KnowledgeBase, InMemoryVectorStore, TextLoader, and FixedSizeChunker. Use when grounding agent answers in document corpora, indexing company docs/FAQs for search, or attaching a search tool to an agent via kb.as_tool() or @use_knowledge_sources.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# RAG Pipeline (Document Chunking + Embedding + Retrieval)

## Overview

```
Documents
    │
    ▼  TextLoader / MarkdownLoader
    │
    ▼  FixedSizeChunker / SentenceChunker
    │
    ▼  InMemoryVectorStore (TF-IDF by default)
    │
    ▼  KnowledgeBase.search(query)
    │
    ▼  Agent tool call → grounded answer
```

---

## Quick start

```python
from lauren_ai._knowledge import KnowledgeBase, TextLoader, FixedSizeChunker
from lauren_ai._memory._vector import InMemoryVectorStore

kb = KnowledgeBase(
    store=InMemoryVectorStore(),
    chunker=FixedSizeChunker(chunk_size=512, overlap=64),
)

# Index documents
await kb.load(TextLoader("docs/faq.txt"))
await kb.load(TextLoader("The quick brown fox jumps over the lazy dog.", is_file=False))

# Search
results = await kb.search("quick fox", top_k=3)
for r in results:
    print(f"{r.score:.3f} — {r.content}")
```

---

## Attaching KB to an agent (as a tool)

```python
from lauren_ai import agent, use_tools

@agent(model="claude-opus-4-6", system="Answer questions using the knowledge base.")
@use_tools(kb.as_tool(name="search_company_docs", top_k=5))
class RAGAgent: ...
```

The `as_tool()` method returns a `@tool()`-decorated async function whose
schema the LLM can call.  Results include `content`, `score`, and any metadata
stored at index time.

---

## Attaching KB via KnowledgeSource (module-level)

```python
from lauren_ai import agent, use_knowledge_sources, AgentModule
from lauren_ai._knowledge import KnowledgeBase, KnowledgeSource, TextLoader
from lauren_ai._memory._vector import InMemoryVectorStore

kb = KnowledgeBase(store=InMemoryVectorStore())
ks = KnowledgeSource(
    kb=kb,
    tool_name="search_docs",
    top_k=5,
    loaders=[TextLoader("docs/faq.txt")],  # loaded at app startup
)

@agent(model="claude-opus-4-6")
@use_knowledge_sources("search_docs")
class SupportAgent: ...

AIModule = AgentModule.for_root(
    agents=[SupportAgent],
    knowledge=[ks],
    imports=[LLMModule.for_root(cfg)],
)
```

---

## Chunkers

| Chunker | Description |
|---------|-------------|
| `FixedSizeChunker(chunk_size=512, overlap=64)` | Fixed character windows with optional overlap |
| `SentenceChunker(max_chunk_size=512)` | Splits at sentence boundaries |

---

## Testing a RAG pipeline

Use `InMemoryVectorStore` directly in tests — no external dependencies needed.
The TF-IDF implementation handles basic keyword matching out of the box.

```python
kb = KnowledgeBase(store=InMemoryVectorStore())
await kb.load(TextLoader("Python is a programming language.", is_file=False))
results = await kb.search("programming language")
assert len(results) > 0
assert "Python" in results[0].content
```

For testing the full agent flow, queue a search tool call in `MockTransport`:

```python
mock.queue_tool_use("search_company_docs", {"query": "refund policy"})
mock.queue_response(completion("Our refund policy allows 30-day returns."))
```

---

## Reference

- `lauren_ai._knowledge`: `KnowledgeBase`, `KnowledgeSource`, `TextLoader`, `FixedSizeChunker`, `SentenceChunker`
- `lauren_ai._memory._vector`: `InMemoryVectorStore`
- `lauren_ai._memory`: `MemoryResult`
- Skills: `managing-memory`, `vector-store-integration`, `embedding-model-ingestion`
