# Knowledge Base

`lauren_ai.knowledge` provides document loading, chunking, and semantic
retrieval for building knowledge-augmented agents.

```python
from lauren_ai.knowledge import KnowledgeBase, TextLoader, FixedSizeChunker, SentenceChunker
```

---

## Document

All knowledge base content is stored as `Document` objects:

```python
from lauren_ai.knowledge import Document

doc = Document(
    content="The Eiffel Tower was completed in 1889.",
    metadata={"source": "history.txt", "page": 1},
)
```

`id` is auto-generated as a UUID hex string when omitted.

---

## Loading documents — TextLoader

`TextLoader` reads a plain text file from disk:

```python
from lauren_ai.knowledge import TextLoader

# From a file path
loader = TextLoader("docs/faq.txt")

# From a raw string (is_file=False)
loader = TextLoader("The capital of France is Paris.", is_file=False)

docs = await loader.load()   # returns list[Document]
```

Pass `is_file=False` to load directly from a string instead of reading a file.

---

## Chunking

Chunkers split long documents into smaller pieces before indexing.  Pass a
chunker to `KnowledgeBase` or call `.chunk()` directly.

### FixedSizeChunker

Splits at fixed character boundaries with optional overlap:

```python
from lauren_ai.knowledge import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,   # max characters per chunk
    overlap=64,       # characters of overlap between consecutive chunks
)

chunks = chunker.chunk(doc)   # returns list[Document]
```

### SentenceChunker

Splits at sentence boundaries (`.`, `!`, `?`) to preserve semantic units:

```python
from lauren_ai.knowledge import SentenceChunker

chunker = SentenceChunker(
    max_chunk_size=512,   # max characters per chunk
)

chunks = chunker.chunk(doc)
```

Use `SentenceChunker` when the document is prose and you want coherent
retrievable units.  Use `FixedSizeChunker` for code, tables, or when you need
predictable chunk sizes.

Each chunk inherits the parent document's `metadata` plus a `chunk_index` key.

---

## KnowledgeBase — indexing and retrieval

`KnowledgeBase` wraps a vector store, optionally an LLM service for embedding
generation, and a chunker.

```python
from lauren_ai.knowledge import KnowledgeBase, TextLoader, SentenceChunker
from lauren_ai._memory._vector import InMemoryVectorStore

store = InMemoryVectorStore()
kb = KnowledgeBase(
    store=store,
    llm_service=llm_service,     # optional; used for embedding generation
    chunker=SentenceChunker(max_chunk_size=512),
)
```

### Loading documents

```python
n = await kb.load(TextLoader("docs/faq.txt"))
print(f"Indexed {n} chunks")

# Load multiple sources
for path in ["docs/faq.txt", "docs/guide.txt"]:
    await kb.load(TextLoader(path))
```

`load()` passes each document through the chunker and upserts all chunks into
the vector store.  Returns the total number of chunks indexed.

### Searching

```python
results = await kb.search(
    "How do I reset my password?",
    top_k=5,
    filter_metadata={"source": "docs/faq.txt"},   # optional metadata filter
)

for result in results:
    print(f"[score={result.score:.3f}] {result.content[:120]}")
    print(f"  source: {result.metadata.get('source')}")
```

Each `MemoryResult` has `.id`, `.content`, `.score`, and `.metadata`.

### Using a knowledge base as an agent tool

`kb.as_tool()` returns a `@tool()`-decorated function backed by this knowledge
base.  Attach it to an agent via `@use_tools()`:

```python
from lauren_ai import agent, use_tools

search_docs = kb.as_tool(name="search_knowledge_base", top_k=5)

@agent(model="claude-opus-4-6", system="You are a helpful support agent.")
@use_tools(search_docs)
class SupportAgent: ...
```

The tool's JSON schema exposes a single `query: str` parameter.  Results are
returned as a list of dicts with `content`, `score`, and any document metadata
keys.

### Attaching via `AgentModule.for_root(knowledge=...)`

For module-level wiring, pass `KnowledgeSource` instances (or bare
`KnowledgeBase`s) to `AgentModule.for_root` via the `knowledge=`
parameter.  The framework calls `kb.as_tool()` internally and
auto-attaches the resulting tool to **every** agent in the module —
no `@use_tools()` declaration needed.

When a `KnowledgeSource` ships `loaders=[…]`, the framework also
generates a `Scope.SINGLETON` `@post_construct` hook per source.  At
app startup (`LifecycleScheduler.run_post_construct`), the framework
iterates the loaders and populates each KB via `await kb.load(loader)`
— **the user never calls `await` themselves**.

```python
from lauren_ai import AgentModule, LLMModule
from lauren_ai._knowledge import (
    KnowledgeBase, KnowledgeSource, SentenceChunker, TextLoader,
)
from lauren_ai._memory._vector import InMemoryVectorStore

@agent(model="claude-opus-4-6", system="Answer using the product manual.")
class ManualAgent: ...                                # NO @use_tools needed

LLMProvider = LLMModule.for_root(LLMConfig.for_anthropic())

AIModule = AgentModule.for_root(
    agents=[ManualAgent],
    imports=LLMProvider,
    knowledge=[
        KnowledgeSource(
            # KB is empty here; the framework populates it at startup
            kb=KnowledgeBase(
                store=InMemoryVectorStore(),
                chunker=SentenceChunker(),
            ),
            tool_name="search_manual",
            top_k=3,
            loaders=[TextLoader("docs/product_manual.txt")],
        ),
    ],
)
# Loading happens at app startup via a generated @post_construct hook —
# no asyncio.run at module-import time, safe inside any async context
# (uvicorn, pytest-asyncio mode=auto, Modal).
```

For multiple KBs in the same module, give each a distinct `tool_name`:

```python
AIModule = AgentModule.for_root(
    agents=[ManualAgent],
    imports=LLMProvider,
    knowledge=[
        KnowledgeSource(
            kb=KnowledgeBase(store=InMemoryVectorStore()),
            tool_name="search_products",
            top_k=3,
            loaders=[TextLoader("docs/products.md")],
        ),
        KnowledgeSource(
            kb=KnowledgeBase(store=InMemoryVectorStore()),
            tool_name="search_policies",
            top_k=5,
            loaders=[TextLoader("docs/policies.md")],
        ),
    ],
)
```

Two KBs with the same `tool_name` raise `DecoratorUsageError` at
module-build time so the collision is caught before the first request.

#### Pre-populated KB (no loaders)

If you've already loaded the KB elsewhere (e.g. fetching from a
remote source asynchronously before app construction), pass the bare
populated KB and omit `loaders=`:

```python
# Caller has already done: await kb.load(TextLoader(...))
AIModule = AgentModule.for_root(
    agents=[ManualAgent],
    imports=LLMProvider,
    knowledge=[kb],          # bare KnowledgeBase; default tool name
)
```

---

## Full example

```python
import os
from lauren_ai import LLMConfig, agent, use_tools
from lauren_ai._module import LLMModule, LLMService
from lauren_ai.knowledge import KnowledgeBase, TextLoader, SentenceChunker
from lauren_ai._memory._vector import InMemoryVectorStore
from lauren import module, LaurenFactory

# Build the knowledge base at startup
store = InMemoryVectorStore()
kb = KnowledgeBase(store=store, chunker=SentenceChunker())
await kb.load(TextLoader("docs/product_manual.txt"))

search_docs = kb.as_tool(name="search_manual", top_k=3)

@agent(model="claude-opus-4-6", system="Answer questions using the product manual.")
@use_tools(search_docs)
class ManualAgent: ...

LLMProvider = LLMModule.for_root(
    LLMConfig.for_anthropic(model="claude-opus-4-6")
)

from lauren_ai._module import AgentModule

AIModule = AgentModule.for_root(
    agents=[ManualAgent],
    tools=[search_docs],
    imports=LLMProvider,
)

@module(imports=[LLMProvider, AIModule])
class AppModule: ...

app = LaurenFactory.create(AppModule)
```

---

## KnowledgeModule — DI registration

Use `KnowledgeModule.for_root()` to register a `KnowledgeBase` as a singleton
in the DI container:

```python
from lauren_ai.knowledge import KnowledgeModule, TextLoader
from lauren_ai._memory._vector import InMemoryVectorStore

KnowledgeProvider = KnowledgeModule.for_root(
    store=InMemoryVectorStore(),
    loaders=[TextLoader("docs/faq.txt")],
)
```

Then import it alongside `LLMProvider` and `AgentModule` in your `AppModule`.
The `KnowledgeBase` singleton is resolved from the container and can be
injected into controllers or other injectables.
