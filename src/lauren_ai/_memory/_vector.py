"""In-memory vector store using TF-IDF cosine similarity.

``InMemoryVectorStore`` implements the ``MemoryStore`` protocol without any
external dependencies (no embedding API calls, no database).  It is suitable
for development, testing, and small-scale deployments.

All embeddings are computed from text using a simple bag-of-words TF-IDF
approach:

1. Tokenise by splitting on whitespace and lowercasing.
2. Compute term frequency for the document.
3. Build an IDF weight from the full corpus (``log(N / df)``).
4. Multiply TF × IDF for each token; store as a sparse-like dict.
5. Normalise the result vector to unit length.
6. Cosine similarity = dot product of two unit vectors.

``numpy`` is used for performance when available; the implementation falls back
to pure Python otherwise.
"""

from __future__ import annotations

__all__ = [
    "InMemoryVectorStore",
]

import logging
import math
import re
import uuid
from typing import Any

from . import MemoryResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# numpy opt-in
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore[import]
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split *text* into word tokens.

    Strips punctuation; only alphanumeric sequences are kept.

    :param text: Input text.
    :type text: str
    :return: List of lowercase token strings.
    :rtype: list[str]
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    """Compute normalised term frequency.

    :param tokens: Token list produced by ``_tokenize``.
    :type tokens: list[str]
    :return: Mapping of token → TF value (count / total_tokens).
    :rtype: dict[str, float]
    """
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    n = len(tokens)
    return {tok: cnt / n for tok, cnt in counts.items()}


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """In-memory vector store using TF-IDF cosine similarity.

    Implements the ``MemoryStore`` protocol.  Suitable for development and
    testing; no external dependencies required.

    :Example:

    .. code-block:: python

        store = InMemoryVectorStore()
        doc_id = await store.upsert("The quick brown fox", metadata={"tag": "test"})
        results = await store.search("quick fox", k=3)
    """

    def __init__(self) -> None:
        # id → (content, metadata, embedding_vector)
        # embedding_vector is stored as a dict[str, float] (sparse) internally
        self._documents: dict[str, tuple[str, dict[str, Any], dict[str, float]]] = {}
        # document-frequency table: token → number of docs containing it
        self._df: dict[str, int] = {}

    # ------------------------------------------------------------------
    # MemoryStore protocol implementation
    # ------------------------------------------------------------------

    async def upsert(
        self,
        content: str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Insert or update a document.

        :param content: The text content to store.
        :type content: str
        :param id: Optional stable identifier.  A UUID4 is generated when
            ``None``.
        :type id: str | None
        :param metadata: Optional key/value metadata dict.
        :type metadata: dict[str, Any] | None
        :param embedding: Pre-computed embedding vector as a list of floats.
            When provided it is used directly (after L2-normalisation); the
            TF-IDF computation is skipped.  Must be dense and compatible with
            the cosine-similarity computation.
        :type embedding: list[float] | None
        :return: The document's identifier.
        :rtype: str
        """
        doc_id = id if id is not None else str(uuid.uuid4())
        safe_meta = metadata or {}

        if embedding is not None:
            # Use provided embedding (normalise to unit length)
            vec = self._normalise_dense(embedding)
            # Convert to sparse dict for uniform storage
            sparse_vec = {str(i): v for i, v in enumerate(vec) if v != 0.0}
        else:
            tokens = _tokenize(content)
            tf = _term_frequency(tokens)
            sparse_vec = tf  # raw TF; IDF applied at search time

        # Update document-frequency table
        if doc_id in self._documents:
            # Remove old DF contributions
            old_content, _, old_vec = self._documents[doc_id]
            old_tokens = set(_tokenize(old_content))
            for tok in old_tokens:
                if tok in self._df:
                    self._df[tok] -= 1
                    if self._df[tok] <= 0:
                        del self._df[tok]

        # Add new DF contributions
        new_tokens_set = set(_tokenize(content))
        for tok in new_tokens_set:
            self._df[tok] = self._df.get(tok, 0) + 1

        self._documents[doc_id] = (content, safe_meta, sparse_vec)
        return doc_id

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        """Search for documents semantically similar to *query*.

        :param query: Natural-language query string.
        :type query: str
        :param k: Maximum number of results to return.
        :type k: int
        :param filter: Optional metadata filter.  Only documents whose
            metadata contains **all** specified key/value pairs are returned.
        :type filter: dict[str, Any] | None
        :return: Up to *k* results ordered by descending cosine similarity.
        :rtype: list[MemoryResult]
        """
        if not self._documents:
            return []
        if k <= 0:
            return []
        if not query.strip():
            return []

        query_vec = self._embed_text(query)

        results: list[tuple[float, str]] = []
        for doc_id, (content, meta, doc_vec) in self._documents.items():
            # Apply metadata filter
            if filter is not None:
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue

            # Compute TF-IDF weighted doc vector
            tfidf_vec = self._apply_idf(doc_vec)
            score = self._cosine_similarity_sparse(query_vec, tfidf_vec)
            results.append((score, doc_id))

        # Sort by descending score
        results.sort(key=lambda x: x[0], reverse=True)

        output: list[MemoryResult] = []
        for score, doc_id in results[:k]:
            content, meta, _ = self._documents[doc_id]
            output.append(MemoryResult(
                id=doc_id,
                content=content,
                score=float(score),
                metadata=dict(meta),
            ))
        return output

    async def get(self, id: str) -> MemoryResult | None:
        """Retrieve a document by its identifier.

        :param id: Document identifier.
        :type id: str
        :return: The ``MemoryResult``, or ``None`` when not found.
        :rtype: MemoryResult | None
        """
        entry = self._documents.get(id)
        if entry is None:
            return None
        content, meta, _ = entry
        return MemoryResult(id=id, content=content, score=1.0, metadata=dict(meta))

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by their identifiers.

        :param ids: List of document identifiers to remove.
        :type ids: list[str]
        """
        for doc_id in ids:
            entry = self._documents.pop(doc_id, None)
            if entry is not None:
                content, _, _ = entry
                old_tokens_set = set(_tokenize(content))
                for tok in old_tokens_set:
                    if tok in self._df:
                        self._df[tok] -= 1
                        if self._df[tok] <= 0:
                            del self._df[tok]

    async def clear(self) -> None:
        """Remove all documents from the store.

        :return: None
        :rtype: None
        """
        self._documents.clear()
        self._df.clear()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> dict[str, float]:
        """Compute a TF-IDF-weighted, normalised sparse embedding for *text*.

        :param text: Input text.
        :type text: str
        :return: Sparse dict mapping token → weighted float.
        :rtype: dict[str, float]
        """
        tokens = _tokenize(text)
        tf = _term_frequency(tokens)
        tfidf = self._apply_idf(tf)
        return tfidf

    def _apply_idf(self, tf: dict[str, float]) -> dict[str, float]:
        """Apply IDF weights to a TF dict and L2-normalise.

        IDF formula: ``log(1 + N) - log(1 + df(t))``  (smoothed).

        :param tf: Term-frequency dict from ``_term_frequency``.
        :type tf: dict[str, float]
        :return: Normalised TF-IDF dict.
        :rtype: dict[str, float]
        """
        n_docs = len(self._documents)
        if n_docs == 0:
            # No documents in corpus yet; use raw TF normalised
            return self._normalise_sparse(tf)

        tfidf: dict[str, float] = {}
        log_n1 = math.log(1 + n_docs)
        for tok, tf_val in tf.items():
            df = self._df.get(tok, 0)
            idf = log_n1 - math.log(1 + df)
            tfidf[tok] = tf_val * idf

        return self._normalise_sparse(tfidf)

    @staticmethod
    def _normalise_sparse(vec: dict[str, float]) -> dict[str, float]:
        """L2-normalise a sparse float vector.

        :param vec: Input sparse vector.
        :type vec: dict[str, float]
        :return: Unit-length sparse vector.
        :rtype: dict[str, float]
        """
        if not vec:
            return {}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0.0:
            return dict(vec)
        return {k: v / norm for k, v in vec.items()}

    @staticmethod
    def _normalise_dense(vec: list[float]) -> list[float]:
        """L2-normalise a dense float vector.

        :param vec: Input dense vector.
        :type vec: list[float]
        :return: Unit-length dense vector.
        :rtype: list[float]
        """
        if not vec:
            return vec
        if _HAS_NUMPY:
            arr = np.array(vec, dtype=float)
            norm = float(np.linalg.norm(arr))
            if norm == 0.0:
                return list(arr)
            return list(arr / norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return list(vec)
        return [v / norm for v in vec]

    @staticmethod
    def _cosine_similarity_sparse(
        a: dict[str, float],
        b: dict[str, float],
    ) -> float:
        """Compute the cosine similarity between two **unit** sparse vectors.

        Because both vectors are already L2-normalised, the cosine similarity
        equals their dot product.

        :param a: First unit-length sparse vector.
        :type a: dict[str, float]
        :param b: Second unit-length sparse vector.
        :type b: dict[str, float]
        :return: Cosine similarity in ``[-1.0, 1.0]`` (typically ``[0.0, 1.0]``
            for TF-IDF vectors which are always non-negative).
        :rtype: float
        """
        if not a or not b:
            return 0.0
        # Dot product over the smaller set for efficiency
        if len(a) > len(b):
            a, b = b, a
        return sum(a_val * b.get(tok, 0.0) for tok, a_val in a.items())

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two dense float vectors.

        Vectors are assumed to be L2-normalised (unit length); the similarity
        is therefore their dot product.

        :param a: First dense vector.
        :type a: list[float]
        :param b: Second dense vector.
        :type b: list[float]
        :return: Cosine similarity.
        :rtype: float
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        if _HAS_NUMPY:
            return float(np.dot(np.array(a), np.array(b)))
        return sum(x * y for x, y in zip(a, b))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._documents)

    def __repr__(self) -> str:  # pragma: no cover
        return f"InMemoryVectorStore(documents={len(self._documents)})"
