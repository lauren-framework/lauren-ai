---
name: hybrid-search
description: Combines dense vector similarity (cosine) with sparse keyword (BM25) search for improved retrieval accuracy. Use when TF-IDF alone misses exact keyword matches, building production search that balances semantic and lexical relevance, or tuning alpha to weight semantic vs keyword scores.
---

> Use `codemap find "SymbolName"` to locate any symbol before reading — it gives
> exact file + line range and is faster than grep across the whole repo.

# Hybrid Search (Dense + Sparse / BM25)

## Overview

Pure semantic search misses exact keyword matches.  Pure keyword search misses
synonyms and paraphrases.  Hybrid search combines both:

```
hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
```

- `alpha = 1.0` → pure semantic (dense only)
- `alpha = 0.0` → pure keyword (BM25 only)
- `alpha = 0.5` → balanced (recommended starting point)

---

## HybridSearch implementation

```python
import math
from collections import Counter


class HybridSearch:
    def __init__(self):
        self._docs: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}

    def index(self, doc_id: str, text: str, embedding: list[float]) -> None:
        self._docs[doc_id] = text
        self._embeddings[doc_id] = embedding

    def _bm25_score(self, query: str, doc_text: str) -> float:
        query_terms = query.lower().split()
        doc_terms = doc_text.lower().split()
        doc_len = len(doc_terms)
        term_freq = Counter(doc_terms)
        avg_len = sum(len(d.split()) for d in self._docs.values()) / max(len(self._docs), 1)
        k1, b = 1.5, 0.75
        score = 0.0
        for term in query_terms:
            tf = term_freq.get(term, 0)
            idf = math.log(
                (len(self._docs) + 1)
                / (sum(1 for d in self._docs.values() if term in d.lower()) + 1)
                + 1
            )
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        return score

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 3,
        alpha: float = 0.5,
    ) -> list[dict]:
        results = []
        for doc_id, text in self._docs.items():
            dense_score = self._cosine_similarity(query_embedding, self._embeddings[doc_id])
            sparse_score = self._bm25_score(query, text)
            # Normalise sparse to [0, 1] roughly
            hybrid_score = alpha * dense_score + (1 - alpha) * min(sparse_score / 10.0, 1.0)
            results.append({"id": doc_id, "text": text, "score": hybrid_score})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
```

---

## Tuning alpha

```python
# Semantic-heavy — good for paraphrase matching
results = hs.search(query, query_emb, alpha=0.8)

# Keyword-heavy — good for exact product names, codes, IDs
results = hs.search(query, query_emb, alpha=0.2)
```

---

## Integration with KnowledgeBase

```python
from lauren_ai._knowledge import KnowledgeBase
from lauren_ai._memory._vector import InMemoryVectorStore

kb = KnowledgeBase(store=InMemoryVectorStore())
hybrid = HybridSearch()

await kb.load(TextLoader("docs/manual.txt"))

# Also index in the hybrid searcher (with your dense embeddings)
for doc_id, text in corpus:
    hybrid.index(doc_id, text, embed(text))

# At query time
semantic_results = await kb.search(query, top_k=10)
hybrid_results = hybrid.search(query, embed(query), top_k=5, alpha=0.6)
```

---

## Reference

- `lauren_ai._memory._vector`: `InMemoryVectorStore` (built-in TF-IDF)
- Skills: `rag-pipeline`, `vector-store-integration`, `document-reranking`
