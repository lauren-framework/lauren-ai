# Research Agent

An agent that searches the web, synthesises results, and cites sources. Uses the built-in `WebSearchTool` and streams tokens to the caller.

```python
from lauren_ai import agent, use_tools, use_knowledge_sources
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._skills import WebSearchTool

_SYSTEM = """\
You are a research assistant. For every factual question:
1. Call the web_search tool with a focused query.
2. Read the results carefully.
3. Write a concise, well-structured answer with citations.

Always cite your sources with the URL. If you cannot find reliable information, say so.\
"""

@agent(
    model=None,                             # inherits from LLMModule.for_root()
    system=_SYSTEM,
    max_turns=6,
    conversation_store=InMemoryConversationStore(),
)
@use_tools(WebSearchTool)
class ResearchAgent: ...
```

**Streaming in a Lauren controller:**

```python
from lauren import controller, get, EventStream, ServerSentEvent
from lauren_ai import AgentRunner

@controller("/api/research")
class ResearchController:
    def __init__(self, runner: AgentRunner[ResearchAgent], agent: ResearchAgent) -> None:
        self._runner = runner
        self._agent = agent

    @get("/search")
    async def search(self, q: str) -> EventStream:
        async def generate():
            async for chunk in await self._runner.run_stream(
                self._agent, q,
                conversation_id=f"research-{q[:20]}",
            ):
                if chunk.delta:
                    yield ServerSentEvent(event="token", data=chunk.delta)
            yield ServerSentEvent(event="done", data="")
        return EventStream(generate(), keep_alive=15.0)
```

**With a knowledge base (RAG):**

```python
from lauren_ai import KnowledgeBase, use_knowledge_sources
from lauren_ai._memory._vector import InMemoryVectorStore

PUBLIC_KB = KnowledgeBase(
    name="public_info",
    description="Company products, pricing, and FAQ",
    store=InMemoryVectorStore(),
)

# Load documents at startup
await PUBLIC_KB.add_texts(["Product A costs $10/month...", "FAQ: ..."])

@agent(model=None, system=_SYSTEM, max_turns=6)
@use_knowledge_sources(PUBLIC_KB)   # adds search_public_info tool automatically
@use_tools(WebSearchTool)
class HybridResearchAgent: ...
```

The `@use_knowledge_sources` decorator adds a `search_<kb_name>` tool that the agent can call to query the vector store.
