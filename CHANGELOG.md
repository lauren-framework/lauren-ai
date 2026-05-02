# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `lauren-ai`
- Transport layer: `Transport` protocol, `AnthropicTransport`, `OpenAITransport`, `OllamaTransport`, `LiteLLMTransport`, `MockTransport`
- Tool system: `@tool()` decorator, `ToolContext`, `ToolResult`, `ToolRegistry`, `ToolSchema` auto-generation
- Memory system: `ShortTermMemory`, `MemoryStore` protocol, `InMemoryVectorStore`, `ConversationStore`, `InMemoryConversationStore`
- Agent system: `@agent()` decorator, `use_tools()`, `AgentRunner`, `AgentContext`, `AgentResponse`
- Extractors: `Agent[T]`, `Completion[T]`, `Embed[T]`, `StreamCompletion[T]`
- Module factories: `LLMModule.for_root()`, `AgentModule.for_root()`
- Signals: `ModelCallStarted`, `ModelCallComplete`, `ToolCallStarted`, `ToolCallComplete`, `AgentTurnComplete`, `AgentRunComplete`, `EmbeddingGenerated`
- Middleware: `conversation_middleware()`, `ai_rate_limit()`
- Guards: `token_budget_guard()`, `safety_guard()`, `requires_capability()`
- Interceptors: `AIMetricsInterceptor`, `TokenUsageResponseInterceptor`
- Pre-built skills: `WebSearchTool`, `CodeExecutionTool`, `HttpFetchTool`
- Knowledge base: `KnowledgeBase`, `TextLoader`, `PDFLoader`, `URLLoader`, hybrid retrieval
- Structured workflows: `Workflow`, `Step`, `Parallel`, `Condition`, `Loop`
- Tool enhancements: HITL confirmation, pre/post hooks, result caching
- Extended thinking support for Anthropic models
- Evaluation framework: `AccuracyEval`, `AgentJudge`, `TrajectoryEval`, `PerformanceEval`
- Full `llms.txt` and `llms-full.txt` AI reference files
- Comprehensive test suite (unit + integration)
- MkDocs documentation site
