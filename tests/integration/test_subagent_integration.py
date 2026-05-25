"""Integration tests for isolated subagent execution through AgentModule."""

from __future__ import annotations

from lauren import LaurenFactory
from pydantic import BaseModel

from lauren_ai import (
    AgentMessageBus,
    AgentModule,
    InMemoryAgentMessageTransport,
    LLMConfig,
    LLMModule,
    ReturnMode,
    SignalBus,
    SubagentCompleted,
    SubagentConfig,
    SubagentStarted,
    SubagentTool,
    agent,
    tool,
    use_tools,
)
from lauren_ai._agents import AgentContext
from lauren_ai._transport import Completion, TokenUsage


class ResearchBrief(BaseModel):
    summary: str
    confidence: float


def _completion(content: str, *, model: str = "mock-model", id: str = "c1") -> Completion:
    return Completion(
        id=id,
        model=model,
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


class TestSubagentModuleIntegration:
    async def test_subagent_tool_registered_in_module(self) -> None:
        @agent(model="research-model", system="Research precisely.")
        class ResearchAgent:
            pass

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
        )

        @agent(model="planner-model", system="Plan carefully.")
        @use_tools(ResearchTool)
        class PlannerAgent:
            pass

        cfg, mock = LLMConfig.for_testing()
        llm_module = LLMModule.for_root(cfg, transport_override=mock)
        agent_module = AgentModule.for_root(
            agents=[PlannerAgent, ResearchAgent],
            tools=[ResearchTool],
            imports=[llm_module],
        )

        app = LaurenFactory.create(agent_module)
        tool_instance = await app.container.resolve(ResearchTool)
        assert tool_instance is not None

    async def test_parent_can_call_subagent_with_isolated_memory_and_shared_bus(self) -> None:
        seen_subagent_message_counts: list[int] = []
        seen_message_bus: list[bool] = []
        started_events: list[SubagentStarted] = []
        completed_events: list[SubagentCompleted] = []

        @agent(model="research-model", system="Research precisely.")
        class ResearchAgent:
            async def on_start(self, ctx: AgentContext) -> None:
                seen_subagent_message_counts.append(len(ctx.memory.messages()))
                seen_message_bus.append(ctx.message_bus is not None)

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
            config=SubagentConfig(return_mode=ReturnMode.DIRECT_JSON),
        )

        @agent(model="planner-model", system="Plan carefully.")
        @use_tools(ResearchTool)
        class PlannerAgent:
            pass

        signals = SignalBus()

        @signals.on(SubagentStarted)
        async def _on_started(event: SubagentStarted) -> None:
            started_events.append(event)

        @signals.on(SubagentCompleted)
        async def _on_completed(event: SubagentCompleted) -> None:
            completed_events.append(event)

        message_bus = AgentMessageBus(transport=InMemoryAgentMessageTransport())

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("research_code", {"task": "Summarize the retry path"})
        mock.queue_response(_completion('{"summary":"done","confidence":0.9}', model="research-model", id="c2"))
        mock.queue_response(_completion("Planner done.", model="planner-model", id="c3"))

        llm_module = LLMModule.for_root(cfg, transport_override=mock)
        agent_module = AgentModule.for_root(
            agents=[PlannerAgent, ResearchAgent],
            tools=[ResearchTool],
            imports=[llm_module],
            signals=signals,
            message_bus=message_bus,
        )

        app = LaurenFactory.create(agent_module)
        planner = await app.container.resolve(PlannerAgent)
        runner = await app.container.resolve(agent_module.runner_class)
        result = await runner.run(planner, "Do the research")

        assert result.content == "Planner done."
        assert [tool_call.name for tool_call in result.tool_calls_made] == ["research_code"]
        assert seen_subagent_message_counts == [1]
        assert seen_message_bus == [True]
        assert len(started_events) == 1
        assert started_events[0].parent_agent_name == "PlannerAgent"
        assert len(completed_events) == 1
        assert completed_events[0].success is True

    async def test_subagent_uses_only_its_own_tools(self) -> None:
        @tool()
        async def parent_only_tool(subject: str) -> dict[str, str]:
            """Run only in the parent agent.

            Args:
                subject: The subject.
            """

            return {"subject": subject}

        @tool()
        async def subagent_only_tool(subject: str) -> dict[str, str]:
            """Run only in the subagent.

            Args:
                subject: The subject.
            """

            return {"subject": subject}

        @agent(model="research-model", system="Research precisely.")
        @use_tools(subagent_only_tool)
        class ResearchAgent:
            pass

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
        )

        @agent(model="planner-model", system="Plan carefully.")
        @use_tools(ResearchTool, parent_only_tool)
        class PlannerAgent:
            pass

        cfg, mock = LLMConfig.for_testing()
        llm_module = LLMModule.for_root(cfg, transport_override=mock)
        agent_module = AgentModule.for_root(
            agents=[PlannerAgent, ResearchAgent],
            tools=[ResearchTool, parent_only_tool, subagent_only_tool],
            imports=[llm_module],
        )

        app = LaurenFactory.create(agent_module)
        await app.container.resolve(agent_module.runner_class)

        planner_tools = PlannerAgent.__lauren_ai_agent__.tools
        research_tools = ResearchAgent.__lauren_ai_agent__.tools

        assert "research_code" in planner_tools
        assert "parent_only_tool" in planner_tools
        assert "subagent_only_tool" not in planner_tools
        assert "subagent_only_tool" in research_tools
        assert "parent_only_tool" not in research_tools

    async def test_subagent_model_override_and_extraction_model_are_used(self) -> None:
        @agent(model="research-model", system="Research precisely.")
        class ResearchAgent:
            pass

        ResearchTool = SubagentTool(
            subagent_cls=ResearchAgent,
            return_type=ResearchBrief,
            name="research_code",
            description="Research code and return a brief",
            config=SubagentConfig(
                model="sub-model",
                extraction_model="extract-model",
            ),
        )

        @agent(model="planner-model", system="Plan carefully.")
        @use_tools(ResearchTool)
        class PlannerAgent:
            pass

        cfg, mock = LLMConfig.for_testing()
        mock.queue_tool_use("research_code", {"task": "Summarize the retry path"})
        mock.queue_response(_completion("Summary: done. Confidence: 0.8", model="sub-model", id="c2"))
        mock.queue_structured(ResearchBrief(summary="done", confidence=0.8))
        mock.queue_response(_completion("Planner done.", model="planner-model", id="c4"))

        llm_module = LLMModule.for_root(cfg, transport_override=mock)
        agent_module = AgentModule.for_root(
            agents=[PlannerAgent, ResearchAgent],
            tools=[ResearchTool],
            imports=[llm_module],
        )

        app = LaurenFactory.create(agent_module)
        planner = await app.container.resolve(PlannerAgent)
        runner = await app.container.resolve(agent_module.runner_class)
        result = await runner.run(planner, "Do the research")

        assert result.content == "Planner done."
        assert mock.calls[1].model == "sub-model"
        assert mock.calls[2].model == "extract-model"
