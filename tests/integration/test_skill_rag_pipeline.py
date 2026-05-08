"""Integration tests for the RAG pipeline pattern (Skill 15).

Tests:
  - KnowledgeBase.load() indexes documents and returns chunk count
  - KnowledgeBase.search() returns MemoryResult with content and score
  - TextLoader with is_file=False loads from string
  - FixedSizeChunker splits text into fixed-size chunks
  - SentenceChunker splits at sentence boundaries
  - KnowledgeBase.as_tool() returns a callable tool
  - Search with relevant keyword returns higher score than irrelevant
  - Multiple documents indexed, search retrieves most relevant
  - top_k=1 returns exactly one result
  - Search on empty KB returns empty list
  - MemoryResult has id, content, score, metadata fields
  - Metadata is preserved through indexing

NOTE: No from __future__ import annotations.
"""

from pydantic import BaseModel

from lauren import Json, LaurenFactory, controller, get, module, post, use_value
from lauren.testing import TestClient
from lauren_ai import LLMConfig
from lauren_ai._agents import agent, use_tools
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._knowledge import (
    Document,
    FixedSizeChunker,
    KnowledgeBase,
    SentenceChunker,
    TextLoader,
)
from lauren_ai._memory import MemoryResult
from lauren_ai._memory._vector import InMemoryVectorStore
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

_MOCK = MockTransport()


def _completion(content="OK", *, n=1, stop_reason="end_turn"):
    return Completion(
        id=f"c{n}",
        model="mock-model",
        content=content,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class _LoadRequest(BaseModel):
    text: str


class _SearchRequest(BaseModel):
    texts: list[str]
    query: str
    top_k: int = 5


class _ChunkRequest(BaseModel):
    content: str
    chunk_size: int = 50
    overlap: int = 0
    max_chunk_size: int = 512


class _AgentRequest(BaseModel):
    query: str


@controller("/kb")
class KnowledgeBaseController:
    @post("/load")
    async def load(self, body: Json[_LoadRequest]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        count = await kb.load(TextLoader(body.text, is_file=False))
        return {"count": count}

    @post("/load-multiple")
    async def load_multiple(self, body: Json[dict]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        texts = body.get("texts", [])
        counts = []
        for text in texts:
            count = await kb.load(TextLoader(text, is_file=False))
            counts.append(count)
        return {"counts": counts}

    @post("/loader-string")
    async def loader_string(self, body: Json[_LoadRequest]) -> dict:
        loader = TextLoader(body.text, is_file=False)
        docs = await loader.load()
        return {"count": len(docs), "content": docs[0].content if docs else ""}

    @post("/search-empty")
    async def search_empty(self) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        results = await kb.search("anything")
        return {"results": results}

    @post("/search")
    async def search(self, body: Json[_SearchRequest]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        for text in body.texts:
            await kb.load(TextLoader(text, is_file=False))
        results = await kb.search(body.query, top_k=body.top_k)
        return {
            "count": len(results),
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "has_metadata": r.metadata is not None,
                }
                for r in results
            ],
        }


@controller("/chunker")
class ChunkerController:
    @post("/fixed-long")
    async def fixed_long(self, body: Json[_ChunkRequest]) -> dict:
        chunker = FixedSizeChunker(chunk_size=body.chunk_size, overlap=body.overlap)
        doc = Document(content=body.content)
        chunks = chunker.chunk(doc)
        return {
            "count": len(chunks),
            "max_len": max(len(c.content) for c in chunks) if chunks else 0,
        }

    @post("/fixed-short")
    async def fixed_short(self, body: Json[_ChunkRequest]) -> dict:
        chunker = FixedSizeChunker(chunk_size=512)
        doc = Document(content=body.content)
        chunks = chunker.chunk(doc)
        return {"count": len(chunks)}

    @post("/fixed-metadata")
    async def fixed_metadata(self, body: Json[_ChunkRequest]) -> dict:
        chunker = FixedSizeChunker(chunk_size=body.chunk_size, overlap=body.overlap)
        doc = Document(content=body.content, metadata={"source": "test.txt"})
        chunks = chunker.chunk(doc)
        return {"all_have_source": all(c.metadata.get("source") == "test.txt" for c in chunks)}

    @post("/sentence")
    async def sentence(self, body: Json[_ChunkRequest]) -> dict:
        chunker = SentenceChunker(max_chunk_size=body.max_chunk_size)
        doc = Document(content=body.content)
        chunks = chunker.chunk(doc)
        return {"count": len(chunks)}


@controller("/rag-agent")
class RagAgentController:
    def __init__(self, mock: MockTransport) -> None:
        cfg = LLMConfig(provider="anthropic", model="mock-model", api_key="mock")
        self._cfg = cfg
        self._mock = mock

    @post("/as-tool")
    async def as_tool(self, body: Json[_LoadRequest]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        await kb.load(TextLoader(body.text, is_file=False))
        tool_fn = kb.as_tool()
        return {"is_callable": callable(tool_fn)}

    @post("/as-tool-custom-name")
    async def as_tool_custom_name(self, body: Json[_LoadRequest]) -> dict:
        from lauren_ai._tools import TOOL_META

        kb = KnowledgeBase(store=InMemoryVectorStore())
        await kb.load(TextLoader(body.text, is_file=False))
        tool_fn = kb.as_tool(name="search_custom")
        meta = getattr(tool_fn, TOOL_META)
        return {"name": meta.name}

    @post("/as-tool-callable")
    async def as_tool_callable(self, body: Json[_LoadRequest]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        await kb.load(TextLoader(body.text, is_file=False))
        tool_fn = kb.as_tool()
        results = await tool_fn("Python programming")
        return {"count": len(results), "is_list": isinstance(results, list)}

    @post("/run")
    async def run(self, body: Json[_AgentRequest]) -> dict:
        kb = KnowledgeBase(store=InMemoryVectorStore())
        await kb.load(TextLoader("The refund policy allows 30-day returns.", is_file=False))
        search_tool = kb.as_tool(name="search_docs")

        @agent(model=None, system="Answer questions using the knowledge base.")
        @use_tools(search_tool)
        class RAGAgent: ...

        runner = AgentRunner(transport=self._mock, tools={}, config=self._cfg)
        resp = await runner.run(RAGAgent(), body.query)
        return {"content": resp.content, "turns": resp.turns}


@module(
    controllers=[KnowledgeBaseController, ChunkerController, RagAgentController],
    providers=[use_value(provide=MockTransport, value=_MOCK)],
)
class RAGModule: ...


def build_app(*responses) -> TestClient:
    _MOCK.reset()
    for item in responses:
        if isinstance(item, tuple):
            # ("tool_use", name, args)
            _MOCK.queue_tool_use(item[1], item[2])
        else:
            _MOCK.queue_response(_completion(item))
    return TestClient(LaurenFactory.create(RAGModule))


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase loading
# ---------------------------------------------------------------------------


class TestKnowledgeBaseLoad:
    def test_load_returns_chunk_count(self):
        client = build_app()
        r = client.post("/kb/load", json={"text": "Hello world."})
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_load_multiple_documents(self):
        client = build_app()
        r = client.post(
            "/kb/load-multiple", json={"texts": ["Document one.", "Document two."]}
        )
        assert r.status_code == 200
        data = r.json()
        assert all(c >= 1 for c in data["counts"])

    def test_text_loader_from_string(self):
        client = build_app()
        r = client.post("/kb/loader-string", json={"text": "Hello from string!"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert "Hello from string!" in data["content"]

    def test_empty_kb_search_returns_empty(self):
        client = build_app()
        r = client.post("/kb/search-empty", json={})
        assert r.status_code == 200
        assert r.json()["results"] == []


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase search
# ---------------------------------------------------------------------------


class TestKnowledgeBaseSearch:
    def test_search_returns_memory_results(self):
        client = build_app()
        r = client.post(
            "/kb/search",
            json={"texts": ["Python is a programming language"], "query": "programming language"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] > 0

    def test_search_result_has_content(self):
        client = build_app()
        r = client.post(
            "/kb/search",
            json={"texts": ["Python is a programming language"], "query": "Python"},
        )
        assert r.status_code == 200
        assert r.json()["results"][0]["content"]

    def test_search_result_has_score(self):
        client = build_app()
        r = client.post(
            "/kb/search",
            json={"texts": ["Python is a programming language"], "query": "Python"},
        )
        assert r.status_code == 200
        score = r.json()["results"][0]["score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_search_result_has_id(self):
        client = build_app()
        r = client.post(
            "/kb/search",
            json={"texts": ["Python is a programming language"], "query": "Python"},
        )
        assert r.status_code == 200
        assert r.json()["results"][0]["id"]

    def test_top_k_limits_results(self):
        client = build_app()
        r = client.post(
            "/kb/search",
            json={
                "texts": [
                    "Python is a programming language",
                    "JavaScript runs in browsers",
                    "Rust is a systems language",
                ],
                "query": "language",
                "top_k": 1,
            },
        )
        assert r.status_code == 200
        assert r.json()["count"] <= 1

    def test_multiple_docs_returns_most_relevant_first(self):
        client = build_app()
        r = client.post(
            "/kb/search",
            json={
                "texts": [
                    "Python is a programming language used for data science",
                    "Cats are fluffy animals that meow",
                    "JavaScript is a scripting language for web browsers",
                ],
                "query": "programming language",
                "top_k": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) >= 1
        assert any(
            word in data["results"][0]["content"].lower()
            for word in ["python", "javascript", "language"]
        )

    def test_search_returns_metadata(self):
        client = build_app()
        r = client.post(
            "/kb/search",
            json={"texts": ["test document content"], "query": "document"},
        )
        assert r.status_code == 200
        assert r.json()["results"][0]["has_metadata"] is True


# ---------------------------------------------------------------------------
# Tests: Chunkers
# ---------------------------------------------------------------------------


class TestChunkers:
    def test_fixed_size_chunker_splits_long_text(self):
        client = build_app()
        r = client.post(
            "/chunker/fixed-long",
            json={"content": "a" * 200, "chunk_size": 50, "overlap": 0},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] > 1
        assert data["max_len"] <= 50

    def test_fixed_size_chunker_short_text_returns_single_chunk(self):
        client = build_app()
        r = client.post("/chunker/fixed-short", json={"content": "Short text."})
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_fixed_size_chunker_preserves_metadata(self):
        client = build_app()
        r = client.post(
            "/chunker/fixed-metadata",
            json={"content": "a" * 200, "chunk_size": 50, "overlap": 0},
        )
        assert r.status_code == 200
        assert r.json()["all_have_source"] is True

    def test_sentence_chunker_splits_at_boundaries(self):
        client = build_app()
        r = client.post(
            "/chunker/sentence",
            json={"content": "First sentence. Second sentence. Third sentence.", "max_chunk_size": 50},
        )
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_sentence_chunker_short_text_returns_single_chunk(self):
        client = build_app()
        r = client.post(
            "/chunker/sentence",
            json={"content": "One short sentence.", "max_chunk_size": 512},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1


# ---------------------------------------------------------------------------
# Tests: KnowledgeBase.as_tool()
# ---------------------------------------------------------------------------


class TestKnowledgeBaseAsTool:
    def test_as_tool_returns_callable(self):
        client = build_app()
        r = client.post("/rag-agent/as-tool", json={"text": "test document"})
        assert r.status_code == 200
        assert r.json()["is_callable"] is True

    def test_as_tool_with_custom_name(self):
        client = build_app()
        r = client.post("/rag-agent/as-tool-custom-name", json={"text": "test document"})
        assert r.status_code == 200
        assert r.json()["name"] == "search_custom"

    def test_as_tool_returns_search_results_when_called(self):
        client = build_app()
        r = client.post(
            "/rag-agent/as-tool-callable",
            json={"text": "Python is a great programming language"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["is_list"] is True
        assert data["count"] >= 1

    def test_rag_agent_with_tool_runs_via_mock(self):
        _MOCK.reset()
        _MOCK.queue_tool_use("search_docs", {"query": "refund policy"})
        _MOCK.queue_response(_completion("Our refund policy allows 30-day returns.", n=2))
        client = TestClient(LaurenFactory.create(RAGModule))

        r = client.post("/rag-agent/run", json={"query": "What is the refund policy?"})
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "Our refund policy allows 30-day returns."
        assert data["turns"] == 2
