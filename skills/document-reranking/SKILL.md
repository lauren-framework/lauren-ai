---
name: document-reranking
description: Re-ranks retrieved documents after initial vector search using a cross-encoder or LLM judge for higher precision. Use when improving retrieval quality in RAG pipelines, implementing cross-encoder re-ranking, using Cohere Re-rank API, or building an LLM-as-judge re-ranker.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Re-ranking Retrieved Documents

## Overview

Initial vector search retrieves by similarity but may include off-topic results.
Re-ranking scores query–document pairs jointly, improving precision at the cost
of an extra model call.

```
Query + Top-K candidates
       │
       ▼
  DocumentReranker.rerank()
       │
       ▼
  Re-scored and sorted results (top-N returned)
```

---

## DocumentReranker (in-process)

A lightweight word-overlap re-ranker for development and testing:

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float


class DocumentReranker:
    def __init__(self, score_fn: Callable[[str, str], float] | None = None):
        self._score_fn = score_fn or self._simple_overlap

    def _simple_overlap(self, query: str, doc: str) -> float:
        q_words = set(query.lower().split())
        d_words = set(doc.lower().split())
        return len(q_words & d_words) / max(len(q_words), 1)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        reranked = sorted(
            results,
            key=lambda r: self._score_fn(query, r.text),
            reverse=True,
        )
        return reranked[:top_k] if top_k else reranked
```

---

## Cohere Re-rank API

For production-grade re-ranking, use the Cohere re-rank model:

```python
import cohere

co = cohere.AsyncClient()

class CohereReranker:
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        documents = [r.text for r in results]
        resp = await co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=top_k or len(documents),
        )
        reranked = []
        for item in resp.results:
            original = results[item.index]
            reranked.append(SearchResult(
                doc_id=original.doc_id,
                text=original.text,
                score=item.relevance_score,
            ))
        return reranked
```

---

## Custom cross-encoder

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_score(query: str, doc: str) -> float:
    return float(model.predict([(query, doc)])[0])

reranker = DocumentReranker(score_fn=cross_encoder_score)
```

---

## Pipeline: retrieve then re-rank

```python
kb = KnowledgeBase(store=InMemoryVectorStore())
await kb.load(TextLoader("docs/faq.txt"))

reranker = DocumentReranker()

async def retrieve_and_rerank(query: str, top_k: int = 3) -> list[SearchResult]:
    # Step 1: broad retrieval
    memory_results = await kb.search(query, top_k=20)

    # Step 2: convert to SearchResult
    candidates = [
        SearchResult(doc_id=r.id, text=r.content, score=r.score)
        for r in memory_results
    ]

    # Step 3: re-rank to top_k
    return reranker.rerank(query, candidates, top_k=top_k)
```

---

## Reference

- `lauren_ai._knowledge`: `KnowledgeBase`, `TextLoader`
- `lauren_ai._memory._vector`: `InMemoryVectorStore`
- Skills: `rag-pipeline`, `hybrid-search`, `vector-store-integration`
