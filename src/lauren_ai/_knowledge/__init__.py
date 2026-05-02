"""Knowledge base and agentic RAG for ``lauren-ai``.

Provides document loading, chunking, embedding-based retrieval, and hybrid
search (keyword + semantic) for building knowledge-augmented agents.

Typical usage::

    from lauren_ai.knowledge import KnowledgeBase, TextLoader, FixedSizeChunker

    kb = KnowledgeBase(
        store=InMemoryVectorStore(),
        llm_service=llm,
        chunker=FixedSizeChunker(chunk_size=512),
    )
    await kb.load(TextLoader("docs/faq.txt"))

    # Attach as a tool to an agent
    @use_tools(kb.as_tool())
    @agent(model="claude-opus-4-6")
    class SupportAgent: ...
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from lauren_ai._memory import MemoryResult, MemoryStore

__all__ = [
    "KnowledgeBase",
    "Document",
    "TextLoader",
    "FixedSizeChunker",
    "SentenceChunker",
    "KnowledgeModule",
]


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A single document chunk stored in the knowledge base.

    :param id: Unique document identifier.
    :type id: str
    :param content: Text content of the chunk.
    :type content: str
    :param metadata: Arbitrary metadata (source, page number, etc.).
    :type metadata: dict[str, Any]
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class TextLoader:
    """Load plain text from a file path or string.

    :param source: File path or raw text content.
    :type source: str
    :param is_file: When ``True``, *source* is interpreted as a file path.
    :type is_file: bool
    """

    def __init__(self, source: str, *, is_file: bool = True) -> None:
        self._source = source
        self._is_file = is_file

    async def load(self) -> list[Document]:
        """Load and return a list of :class:`Document` objects.

        :return: Documents loaded from the source.
        :rtype: list[Document]
        :raises KnowledgeLoadError: When the file cannot be read.
        """
        from lauren_ai._exceptions import KnowledgeLoadError  # noqa: PLC0415

        try:
            if self._is_file:
                with open(self._source, encoding="utf-8") as f:
                    text = f.read()
            else:
                text = self._source
        except OSError as exc:
            raise KnowledgeLoadError(
                f"Cannot read {self._source!r}: {exc}", cause=exc
            ) from exc

        return [Document(content=text, metadata={"source": self._source})]


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


class FixedSizeChunker:
    """Split documents into fixed-size chunks with optional overlap.

    :param chunk_size: Maximum characters per chunk.
    :type chunk_size: int
    :param overlap: Overlap in characters between consecutive chunks.
    :type overlap: int
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, doc: Document) -> list[Document]:
        """Split *doc* into fixed-size chunks.

        :param doc: The document to split.
        :type doc: Document
        :return: List of chunk documents.
        :rtype: list[Document]
        """
        text = doc.content
        chunks: list[Document] = []
        step = max(1, self._chunk_size - self._overlap)
        idx = 0
        while idx < len(text):
            chunk_text = text[idx : idx + self._chunk_size]
            if chunk_text.strip():
                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={**doc.metadata, "chunk_index": len(chunks)},
                    )
                )
            idx += step
        return chunks or [doc]


class SentenceChunker:
    """Split documents into chunks at sentence boundaries.

    :param max_chunk_size: Maximum characters per chunk.
    :type max_chunk_size: int
    """

    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chunk_size: int = 512) -> None:
        self._max = max_chunk_size

    def chunk(self, doc: Document) -> list[Document]:
        """Split *doc* at sentence boundaries.

        :param doc: The document to split.
        :type doc: Document
        :return: List of chunk documents.
        :rtype: list[Document]
        """
        sentences = self._SENTENCE_BOUNDARY.split(doc.content)
        chunks: list[Document] = []
        buf: list[str] = []
        buf_len = 0

        for sent in sentences:
            if buf_len + len(sent) > self._max and buf:
                chunks.append(
                    Document(
                        content=" ".join(buf),
                        metadata={**doc.metadata, "chunk_index": len(chunks)},
                    )
                )
                buf = []
                buf_len = 0
            buf.append(sent)
            buf_len += len(sent) + 1

        if buf:
            chunks.append(
                Document(
                    content=" ".join(buf),
                    metadata={**doc.metadata, "chunk_index": len(chunks)},
                )
            )
        return chunks or [doc]


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """A document knowledge base backed by a vector store.

    Supports loading documents from various loaders, chunking them, generating
    embeddings, and performing semantic search.

    :param store: The vector store backend.
    :type store: MemoryStore
    :param llm_service: Optional LLM service for embedding generation.
    :type llm_service: Any | None
    :param chunker: Document chunker.  Defaults to :class:`FixedSizeChunker`.
    :type chunker: Any | None
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        llm_service: Any = None,
        chunker: Any = None,
    ) -> None:
        self._store = store
        self._llm = llm_service
        self._chunker = chunker or FixedSizeChunker()

    async def load(self, loader: Any) -> int:
        """Load documents from *loader* into the knowledge base.

        :param loader: A loader object with an async ``load()`` method.
        :type loader: Any
        :return: Number of chunks indexed.
        :rtype: int
        """
        docs = await loader.load()
        total = 0
        for doc in docs:
            chunks = self._chunker.chunk(doc)
            for chunk in chunks:
                await self._store.upsert(
                    chunk.content,
                    id=chunk.id,
                    metadata=chunk.metadata,
                )
                total += 1
        return total

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemoryResult]:
        """Search the knowledge base for documents relevant to *query*.

        :param query: Search query string.
        :type query: str
        :param top_k: Maximum number of results to return.
        :type top_k: int
        :param filter_metadata: Optional metadata filter dict.
        :type filter_metadata: dict[str, Any] | None
        :return: Ranked list of :class:`~lauren_ai._memory.MemoryResult`.
        :rtype: list[MemoryResult]
        """
        kwargs: dict[str, Any] = {}
        if hasattr(self._store, "search"):
            # Support both (k=, filter=) and (top_k=, filter_metadata=) signatures
            import inspect  # noqa: PLC0415
            sig = inspect.signature(self._store.search)
            if "k" in sig.parameters:
                kwargs["k"] = top_k
            else:
                kwargs["top_k"] = top_k
            if "filter" in sig.parameters:
                kwargs["filter"] = filter_metadata
            else:
                kwargs["filter_metadata"] = filter_metadata
        return await self._store.search(query, **kwargs)

    def as_tool(self, *, name: str = "search_knowledge_base", top_k: int = 5) -> Any:
        """Return a ``@tool()``-decorated function backed by this knowledge base.

        Attach to an agent via ``@use_tools(kb.as_tool())``::

            @use_tools(kb.as_tool(name="search_docs"))
            @agent(model="claude-opus-4-6")
            class SupportAgent: ...

        :param name: Tool name.
        :type name: str
        :param top_k: Maximum results to return per search.
        :type top_k: int
        :return: A ``@tool()``-decorated async function.
        :rtype: Any
        """
        from lauren_ai._tools import tool  # noqa: PLC0415

        kb_ref = self
        _top_k = top_k

        @tool(name=name)
        async def _kb_search(query: str) -> list[dict[str, Any]]:
            """Search the knowledge base for relevant information.

            Args:
                query: The search query.

            Returns a list of relevant text snippets.
            """
            results = await kb_ref.search(query, top_k=_top_k)
            return [
                {"content": r.content, "score": r.score, **r.metadata}
                for r in results
            ]

        return _kb_search


# ---------------------------------------------------------------------------
# KnowledgeModule
# ---------------------------------------------------------------------------


class KnowledgeModule:
    """Module that registers a :class:`KnowledgeBase` in the DI container.

    Usage::

        KnowledgeProviderModule = KnowledgeModule.for_root(
            store=InMemoryVectorStore(),
            loaders=[TextLoader("docs/faq.txt")],
        )
    """

    def __init__(
        self,
        store: Any,
        *,
        loaders: list[Any] | None = None,
        chunker: Any = None,
    ) -> None:
        self._store = store
        self._loaders = loaders or []
        self._chunker = chunker

    @classmethod
    def for_root(
        cls,
        store: Any,
        *,
        loaders: list[Any] | None = None,
        chunker: Any = None,
    ) -> KnowledgeModule:
        """Create a :class:`KnowledgeModule`.

        :param store: The vector store backend.
        :type store: MemoryStore
        :param loaders: Optional list of loaders to eagerly index at startup.
        :type loaders: list[Any] | None
        :param chunker: Optional chunker override.
        :type chunker: Any | None
        :return: Configured :class:`KnowledgeModule`.
        :rtype: KnowledgeModule
        """
        return cls(store, loaders=loaders, chunker=chunker)

    def register(self, container: Any) -> None:
        """Register the :class:`KnowledgeBase` in the DI container.

        :param container: The DI container.
        :type container: Any
        """
        kb = KnowledgeBase(
            self._store,
            chunker=self._chunker,
        )

        try:
            from lauren import Scope  # type: ignore[import]  # noqa: PLC0415

            container.provide(KnowledgeBase, lambda: kb, scope=Scope.SINGLETON)
        except ImportError:
            pass
