"""lauren-ai — First-party AI/LLM companion for the Lauren web framework.

``lauren-ai`` integrates large-language-model completions, agentic loops,
tool execution, memory management, and structured workflows into Lauren
applications using the same decorator-first, DI-driven, module-scoped
paradigm as the framework itself.

Quick start::

    from lauren_ai import LLMModule, LLMConfig, LLMService, Message

    # Single completion
    class AIController:
        def __init__(self, llm: LLMService) -> None:
            self._llm = llm

        @get("/hello")
        async def hello(self) -> dict:
            completion = await self._llm.complete(
                [Message(role="user", content="Say hello!")]
            )
            return {"message": completion.content}

    # Agent with tools — @agent() outermost, @use_tools() below it
    @tool()
    async def get_weather(city: str) -> dict:
        \"\"\"Get weather for a city. Args: city: City name.\"\"\"
        return {"city": city, "temperature": 22}

    @agent(model="claude-opus-4-6", system="You are a helpful assistant.")
    @use_tools(get_weather)
    class WeatherAgent: ...

See the documentation at https://lauren-ai.readthedocs.io for full details.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__: str = version("lauren-ai")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # Config
    "LLMConfig",
    "AgentConfig",
    # Transport types
    "Message",
    "Completion",
    "CompletionChunk",
    "TokenUsage",
    "ToolCall",
    "ToolSchema",
    "Embedding",
    # Multimodal content types (Section 38)
    "ImageContent",
    "AudioContent",
    "DocumentContent",
    "ContentPart",
    "UnsupportedContentError",
    # Structured output (Section 37)
    "StructuredLLM",
    # Exceptions
    "LaurenAIError",
    "TransportError",
    "AgentMaxTurnsError",
    "AgentBudgetExceededError",
    "AgentConfigError",
    "DecoratorUsageError",
    "ToolExecutionError",
    # Decorators
    "agent",
    "use_tools",
    "use_knowledge_sources",
    "tool",
    # Agent types
    "AgentMeta",
    "AgentContext",
    "AgentResponse",
    # Tool types
    "ToolContext",
    "ToolResult",
    "get_tool_context_from_func_args",
    "set_metadata",
    # Memory
    "ShortTermMemory",
    "MemoryStore",
    "ConversationStore",
    "InMemoryVectorStore",
    "InMemoryConversationStore",
    # User memory (Section 36)
    "MemoryFact",
    "UserMemoryStore",
    "InMemoryUserMemoryStore",
    "remember",
    "RememberMeta",
    "MemoryConfigError",
    "REMEMBER_META",
    # Signals
    "SignalBus",
    "ModelCallStarted",
    "ModelCallComplete",
    "ToolCallStarted",
    "ToolCallComplete",
    "AgentRunComplete",
    # Modules / services
    "LLMModule",
    "AgentModule",
    "LLMService",
    # Extractors
    "Agent",
    "Embed",
    "StreamCompletion",
    # Guards
    "token_budget_guard",
    "requires_capability",
    "safety_guard",
    "SafetyPolicy",
    # Middleware factories
    "conversation_middleware",
    "ai_rate_limit",
    # Interceptors
    "ai_metrics_interceptor",
    "token_usage_response_interceptor",
    # Runner
    "AgentRunner",
    "AgentRunnerBase",
    # Prompt templates (Section 32)
    "PromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotExample",
    "PromptRenderError",
    # Chains (Section 32)
    "Chain",
    "Runnable",
    "RunnableLambda",
    "chain",
    # Output parsers (Section 33)
    "OutputParserError",
    "MaxRetryError",
    "StrOutputParser",
    "JSONOutputParser",
    "RegexParser",
    "CommaSeparatedListParser",
    "MarkdownCodeBlockParser",
    "PydanticOutputParser",
    "RetryOutputParser",
    # Semantic router (Section 39)
    "SemanticRouter",
    "Route",
    "RouteMatch",
    "RouterConfigError",
    # Cost & rate tracking (Section 40)
    "ModelPricing",
    "CostEstimate",
    "PricingTable",
    "default_pricing_table",
    "CostTracker",
    "CostSession",
    "CostReport",
    "TokenBudget",
    "BudgetExceededError",
    "RateLimiter",
    "RateLimitExhaustedError",
    # Guardrails (Section 41)
    # @guardrail() — class decorator, makes a guardrail DI-injectable
    "guardrail",
    "GuardrailClassMeta",
    "GUARDRAIL_CLASS_META",
    # @use_guardrails() — agent decorator, attaches guardrail instances to @agent()
    "use_guardrails",
    "UseGuardrailsMeta",
    "USE_GUARDRAILS_META",
    "GuardrailDecision",
    "GuardrailContext",
    "GuardrailViolated",
    "InputGuardrail",
    "OutputGuardrail",
    "TopicFilter",
    "PIIRedactor",
    "LengthFilter",
    "PromptInjectionFilter",
    "LLMGuardrail",
    # Agent teams (Section 34)
    "team",
    "TeamMeta",
    "TEAM_META",
    "TeamConfigError",
    "TeamMemory",
    "TeamRunner",
    "TeamResult",
    "TeamEvent",
    "TeamWorkerStarted",
    "TeamWorkerFinished",
    "TeamCoordinatorDecision",
    "TeamFinalAnswer",
    # Tracing (Section 35)
    "SpanKind",
    "Span",
    "Trace",
    "TraceStore",
    "TracingConfig",
    "TraceExporter",
    "InMemoryTraceExporter",
    "ConsoleTraceExporter",
    "FileTraceExporter",
    "traced",
    "set_trace_store",
    "get_trace_store",
    "TracingError",
    # Routing extras
    "RouterNotCompiledError",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
from lauren_ai._agents import (
    AgentContext,
    AgentMeta,
    AgentResponse,
    agent,
    use_knowledge_sources,
    use_tools,
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
from lauren_ai._agents._runner import AgentRunner, AgentRunnerBase

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------
from lauren_ai._chains import Chain, Runnable, RunnableLambda, chain
from lauren_ai._config import AgentConfig, LLMConfig

# ---------------------------------------------------------------------------
# Cost & rate tracking (Section 40)
# ---------------------------------------------------------------------------
from lauren_ai._cost import (
    BudgetExceededError,
    CostEstimate,
    CostReport,
    CostSession,
    CostTracker,
    ModelPricing,
    PricingTable,
    RateLimiter,
    RateLimitExhaustedError,
    TokenBudget,
    default_pricing_table,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
from lauren_ai._exceptions import (
    AgentBudgetExceededError,
    AgentConfigError,
    AgentMaxTurnsError,
    DecoratorUsageError,
    LaurenAIError,
    OutputParserError,
    ToolExecutionError,
    TracingError,
    TransportError,
)

# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
from lauren_ai._extractors import Agent, Embed, StreamCompletion

# ---------------------------------------------------------------------------
# Guardrails (Section 41)
# ---------------------------------------------------------------------------
from lauren_ai._guardrails import (
    GUARDRAIL_CLASS_META,
    USE_GUARDRAILS_META,
    GuardrailClassMeta,
    GuardrailContext,
    GuardrailDecision,
    GuardrailViolated,
    InputGuardrail,
    LengthFilter,
    LLMGuardrail,
    OutputGuardrail,
    PIIRedactor,
    PromptInjectionFilter,
    TopicFilter,
    UseGuardrailsMeta,
    guardrail,
    use_guardrails,
)

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
from lauren_ai._guards import (
    SafetyPolicy,
    requires_capability,
    safety_guard,
    token_budget_guard,
)

# ---------------------------------------------------------------------------
# Interceptors
# ---------------------------------------------------------------------------
from lauren_ai._interceptors import (
    ai_metrics_interceptor,
    token_usage_response_interceptor,
)

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# User memory (Section 36)
# ---------------------------------------------------------------------------
from lauren_ai._memory import (
    REMEMBER_META,
    ConversationStore,
    InMemoryUserMemoryStore,
    MemoryConfigError,
    MemoryFact,
    MemoryStore,
    RememberMeta,
    ShortTermMemory,
    UserMemoryStore,
    remember,
)
from lauren_ai._memory._stores import InMemoryConversationStore
from lauren_ai._memory._vector import InMemoryVectorStore

# ---------------------------------------------------------------------------
# Middleware factories
# ---------------------------------------------------------------------------
from lauren_ai._middleware import ai_rate_limit, conversation_middleware

# ---------------------------------------------------------------------------
# Modules and services
# ---------------------------------------------------------------------------
from lauren_ai._module import AgentModule, LLMModule, LLMService

# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------
from lauren_ai._output_parsers import (
    CommaSeparatedListParser,
    JSONOutputParser,
    MarkdownCodeBlockParser,
    MaxRetryError,
    OutputParserError,
    PydanticOutputParser,
    RegexParser,
    RetryOutputParser,
    StrOutputParser,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
from lauren_ai._prompts import (
    ChatPromptTemplate,
    FewShotExample,
    FewShotPromptTemplate,
    PromptRenderError,
    PromptTemplate,
)

# ---------------------------------------------------------------------------
# Semantic router (Section 39)
# ---------------------------------------------------------------------------
from lauren_ai._routing import (
    Route,
    RouteMatch,
    RouterConfigError,
    RouterNotCompiledError,
    SemanticRouter,
)

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
from lauren_ai._signals import (
    AgentRunComplete,
    ModelCallComplete,
    ModelCallStarted,
    SignalBus,
    ToolCallComplete,
    ToolCallStarted,
)

# ---------------------------------------------------------------------------
# Agent teams (Section 34)
# ---------------------------------------------------------------------------
from lauren_ai._teams import (
    TEAM_META,
    TeamConfigError,
    TeamCoordinatorDecision,
    TeamEvent,
    TeamFinalAnswer,
    TeamMemory,
    TeamMeta,
    TeamResult,
    TeamRunner,
    TeamWorkerFinished,
    TeamWorkerStarted,
    team,
)
from lauren_ai._tools import (
    ToolContext,
    ToolResult,
    get_tool_context_from_func_args,
    set_metadata,
    tool,
)

# ---------------------------------------------------------------------------
# Tracing (Section 35)
# ---------------------------------------------------------------------------
from lauren_ai._tracing import (
    ConsoleTraceExporter,
    FileTraceExporter,
    InMemoryTraceExporter,
    Span,
    SpanKind,
    Trace,
    TraceExporter,
    TraceStore,
    TracingConfig,
    get_trace_store,
    set_trace_store,
    traced,
)

# ---------------------------------------------------------------------------
# Transport types
# ---------------------------------------------------------------------------
from lauren_ai._transport import (
    Completion,
    CompletionChunk,
    Embedding,
    Message,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Multimodal content types (Section 38)
# ---------------------------------------------------------------------------
from lauren_ai._transport._multimodal import (
    AudioContent,
    ContentPart,
    DocumentContent,
    ImageContent,
    UnsupportedContentError,
)

# ---------------------------------------------------------------------------
# Structured output (Section 37)
# ---------------------------------------------------------------------------
from lauren_ai._transport._structured import StructuredLLM
