"""Unit tests for the isolated subagent runtime."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from lauren_ai._agents import AgentContext, agent
from lauren_ai._agents._runner import AgentRunnerBase as AgentRunner
from lauren_ai._config import AgentConfig, LLMConfig
from lauren_ai._memory import ShortTermMemory
from lauren_ai._module import LLMService
from lauren_ai._signals import SignalBus, SubagentCompleted, SubagentStarted
from lauren_ai._subagent import (
    LlmCompiler,
    PassThroughCompiler,
    ReturnMode,
    SubagentConfig,
    SubagentPool,
    SubagentTool,
    TemplateCompiler,
)
from lauren_ai._tools import ToolContext
from lauren_ai._transport import Completion, TokenUsage
from lauren_ai._transport._mock import MockTransport


class ResearchBrief(BaseModel):
    summary: str
    confidence: float


def _completion(content: str, *, model: str = "mock-model") -> Completion:
    return Completion(
        id="c1",
        model=model,
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _parent_agent_context(
    *,
    metadata: dict[str, object] | None = None,
    signals: SignalBus | None = None,
    runner: AgentRunner | None = None,
) -> AgentContext:
    memory = ShortTermMemory()
    memory.add_user("Parent request")
    memory.add_assistant("Parent analysis")
    return AgentContext(
        agent_id="parent-agent",
        agent_run_id="parent-run",
        agent_class=type("ParentAgentClass", (), {}),
        config=AgentConfig(),
        memory=memory,
        turn=0,
        metadata=dict(metadata or {}),
        conversation_id="conversation-1",
        signals=signals,
        runner=runner,
    )


def _tool_context(
    *,
    metadata: dict[str, object] | None = None,
    signals: SignalBus | None = None,
    runner: AgentRunner | None = None,
) -> ToolContext:
    return ToolContext(
        agent_context=_parent_agent_context(metadata=metadata, signals=signals, runner=runner),
        tool_use_id="tool-1",
        turn=0,
    )


class TestBriefCompilers:
    @pytest.mark.asyncio
    async def test_pass_through_compiler_uses_task_field(self) -> None:
        compiler = PassThroughCompiler()
        brief = await compiler.compile({"task": "Summarize runner budget handling"}, _tool_context())
        assert brief == "Summarize runner budget handling"

    @pytest.mark.asyncio
    async def test_template_compiler_uses_tool_input_and_metadata(self) -> None:
        compiler = TemplateCompiler(
            template="Task: {{ task }} / Repo: {{ repo_name }} / Missing: {{ unknown }}",
            metadata_keys=("repo_name",),
        )
        brief = await compiler.compile(
            {"task": "Review _runner.py"},
            _tool_context(metadata={"repo_name": "lauren-ai"}),
        )
        assert brief == "Task: Review _runner.py / Repo: lauren-ai / Missing: "

    @pytest.mark.asyncio
    async def test_llm_compiler_summarizes_recent_history(self) -> None:
        transport = MockTransport()
        transport.queue_response(_completion("Focused subagent brief"))
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=transport, config=config)

        compiler = LlmCompiler(llm=llm, window=1)
        brief = await compiler.compile({"task": "Investigate tracing"}, _tool_context(metadata={"repo": "lauren-ai"}))

        assert brief == "Focused subagent brief"
        assert "Investigate tracing" in str(transport.calls[0].messages[0].content)


class TestSubagentTool:
    @pytest.mark.asyncio
    async def test_subagent_tool_uses_empty_memory_and_applies_overrides(self) -> None:
        subagent_transport = MockTransport()
        subagent_transport.queue_response(_completion('{"summary":"done","confidence":0.9}', model="override-model"))
        runner = AgentRunner(transport=subagent_transport)

        extraction_transport = MockTransport()
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=extraction_transport, config=config)

        seen_message_counts: list[int] = []
        seen_max_turns: list[int] = []

        @agent(model="research-model", system="Research precisely.")
        class ResearchAgent:
            async def on_start(self, ctx: AgentContext) -> None:
                seen_message_counts.append(len(ctx.memory.messages()))
                seen_max_turns.append(ctx.config.max_turns)

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
            config=SubagentConfig(
                max_turns=2,
                max_tokens_per_turn=33,
                model="override-model",
                return_mode=ReturnMode.DIRECT_JSON,
            ),
        )

        tool_instance = ResearchTool(agent=ResearchAgent(), llm=llm)
        result = await tool_instance.run(_tool_context(runner=runner), "Inspect the retry path")

        assert result == {"summary": "done", "confidence": 0.9}
        assert seen_message_counts == [1]
        assert seen_max_turns == [2]
        assert subagent_transport.calls[0].model == "override-model"
        assert subagent_transport.calls[0].max_tokens == 33
        assert "Inspect the retry path" in str(subagent_transport.calls[0].messages)

    @pytest.mark.asyncio
    async def test_subagent_tool_uses_structured_llm_extraction(self) -> None:
        subagent_transport = MockTransport()
        subagent_transport.queue_response(_completion("Summary: done. Confidence: 0.8"))
        runner = AgentRunner(transport=subagent_transport)

        extraction_transport = MockTransport()
        extraction_transport.queue_structured(ResearchBrief(summary="done", confidence=0.8))
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=extraction_transport, config=config)

        @agent(model="research-model", system="Research precisely.")
        class ResearchAgent:
            pass

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
        )

        tool_instance = ResearchTool(agent=ResearchAgent(), llm=llm)
        result = await tool_instance.run(_tool_context(runner=runner), "Inspect the retry path")

        assert result == {"summary": "done", "confidence": 0.8}
        assert extraction_transport.calls[0].tool_choice is not None

    @pytest.mark.asyncio
    async def test_subagent_tool_failure_returns_error_dict(self) -> None:
        subagent_transport = MockTransport()
        subagent_transport.queue_response(_completion("not valid json"))
        runner = AgentRunner(transport=subagent_transport)

        extraction_transport = MockTransport()
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=extraction_transport, config=config)

        @agent(model="research-model", system="Research precisely.")
        class ResearchAgent:
            pass

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
            config=SubagentConfig(return_mode=ReturnMode.DIRECT_JSON),
        )

        tool_instance = ResearchTool(agent=ResearchAgent(), llm=llm)
        result = await tool_instance.run(_tool_context(runner=runner), "Inspect the retry path")

        assert result["subagent"] == "ResearchAgent"
        assert result["task"] == "Inspect the retry path"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_subagent_signals_emitted_for_successful_run(self) -> None:
        subagent_transport = MockTransport()
        subagent_transport.queue_response(_completion('{"summary":"done","confidence":0.9}'))
        runner = AgentRunner(transport=subagent_transport)

        extraction_transport = MockTransport()
        config, _ = LLMConfig.for_testing()
        llm = LLMService(transport=extraction_transport, config=config)

        @agent(model="research-model", system="Research precisely.")
        class ResearchAgent:
            pass

        bus = SignalBus()
        started: list[SubagentStarted] = []
        completed: list[SubagentCompleted] = []

        @bus.on(SubagentStarted)
        async def _on_started(event: SubagentStarted) -> None:
            started.append(event)

        @bus.on(SubagentCompleted)
        async def _on_completed(event: SubagentCompleted) -> None:
            completed.append(event)

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
            config=SubagentConfig(return_mode=ReturnMode.DIRECT_JSON),
        )

        tool_instance = ResearchTool(agent=ResearchAgent(), llm=llm)
        await tool_instance.run(_tool_context(signals=bus, runner=runner), "Inspect the retry path")

        assert len(started) == 1
        assert started[0].parent_agent_name == "ParentAgentClass"
        assert started[0].subagent_name == "ResearchAgent"
        assert len(completed) == 1
        assert completed[0].success is True
        assert completed[0].error is None


class TestSubagentPool:
    @pytest.mark.asyncio
    async def test_pool_returns_results_in_input_order(self) -> None:
        class FakeTool:
            async def run(self, ctx: ToolContext, task: str) -> dict[str, object]:
                await asyncio.sleep(0.01 if task == "b" else 0.0)
                return {"task": task}

        pool = SubagentPool(FakeTool())
        results = await pool.run_all(["a", "b", "c"], _tool_context())
        assert results == [{"task": "a"}, {"task": "b"}, {"task": "c"}]

    @pytest.mark.asyncio
    async def test_pool_partial_failure_returns_error_dict(self) -> None:
        class FakeTool:
            async def run(self, ctx: ToolContext, task: str) -> dict[str, object]:
                if task == "bad":
                    raise RuntimeError("boom")
                return {"task": task}

        pool = SubagentPool(FakeTool())
        results = await pool.run_all(["good", "bad"], _tool_context())
        assert results[0] == {"task": "good"}
        assert results[1] == {"error": "boom", "task": "bad"}

    @pytest.mark.asyncio
    async def test_pool_respects_max_concurrent(self) -> None:
        active = 0
        max_active = 0

        class FakeTool:
            async def run(self, ctx: ToolContext, task: str) -> dict[str, object]:
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.02)
                active -= 1
                return {"task": task}

        pool = SubagentPool(FakeTool(), max_concurrent=1)
        results = await pool.run_all(["a", "b", "c"], _tool_context())

        assert [item["task"] for item in results] == ["a", "b", "c"]
        assert max_active == 1
