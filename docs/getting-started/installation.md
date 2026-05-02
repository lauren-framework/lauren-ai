# Installation

## Requirements

- Python 3.11+
- Lauren framework ≥ 1.0

## Install with pip / uv

```bash
# Core (no LLM provider included)
pip install lauren-ai

# With Anthropic SDK
pip install "lauren-ai[anthropic]"

# With OpenAI SDK
pip install "lauren-ai[openai]"

# With Ollama (no extra dep — uses httpx which is bundled)
pip install lauren-ai

# With LiteLLM (catch-all for 100+ models)
pip install "lauren-ai[litellm]"

# With everything
pip install "lauren-ai[all]"
```

## Optional backends

| Extra | Provides |
|---|---|
| `anthropic` | `AnthropicTransport` (Claude models) |
| `openai` | `OpenAITransport` (GPT-4o, o1, o3) |
| `litellm` | `LiteLLMTransport` (Gemini, Cohere, Bedrock, Azure, …) |
| `pgvector` | `PgVectorStore` (PostgreSQL + pgvector) |
| `chroma` | `ChromaStore` |
| `qdrant` | `QdrantStore` |
| `redis` | `RedisCacheBackend`, `RedisConversationStore` |
| `pypdf` | `PDFLoader` for knowledge bases |

## Local development

```bash
git clone https://github.com/lauren-framework/lauren-ai
cd lauren-ai
uv sync --extra dev --extra anthropic
uv run pytest tests/unit
```
