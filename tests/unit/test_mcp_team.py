"""Unit tests for lauren_ai.mcp._team (McpAgentTeam, McpTeamResult, TeamWorkerResult)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lauren_ai import agent
from lauren_ai.mcp._team import McpAgentTeam, McpTeamResult, TeamWorkerResult

# ---------------------------------------------------------------------------
# TeamWorkerResult / McpTeamResult dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_team_worker_result_fields(self):
        result = TeamWorkerResult(worker_name="researcher", content="done")
        assert result.worker_name == "researcher"
        assert result.content == "done"
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_team_worker_result_explicit_tokens(self):
        result = TeamWorkerResult(
            worker_name="w",
            content="hi",
            input_tokens=10,
            output_tokens=20,
        )
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    def test_mcp_team_result_fields(self):
        result = McpTeamResult(final_answer="all done")
        assert result.final_answer == "all done"
        assert result.worker_results == {}
        assert result.rounds == 0
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0

    def test_mcp_team_result_explicit_fields(self):
        wr = TeamWorkerResult(worker_name="w", content="c")
        result = McpTeamResult(
            final_answer="ok",
            worker_results={"w": wr},
            rounds=3,
            total_input_tokens=100,
            total_output_tokens=200,
        )
        assert result.rounds == 3
        assert result.total_input_tokens == 100
        assert "w" in result.worker_results


# ---------------------------------------------------------------------------
# McpAgentTeam construction validation
# ---------------------------------------------------------------------------


class TestMcpAgentTeamValidation:
    def test_requires_at_least_one_worker(self):
        @agent(model="claude-sonnet-4-6")
        class Coord:
            pass

        with pytest.raises(ValueError, match="at least one worker"):
            McpAgentTeam(
                coordinator=Coord,
                workers={},
                llm_module=MagicMock(),
            )

    def test_requires_agent_class(self):
        class NotAnAgent:
            pass

        mock_client = AsyncMock()
        with pytest.raises(ValueError, match="not decorated with @agent"):
            McpAgentTeam(
                coordinator=NotAnAgent,
                workers={"w": mock_client},
                llm_module=MagicMock(),
            )

    def test_accepts_valid_agent_and_workers(self):
        @agent(model="claude-sonnet-4-6")
        class MyCoord:
            pass

        mock_client = AsyncMock()
        team = McpAgentTeam(
            coordinator=MyCoord,
            workers={"worker1": mock_client},
            llm_module=MagicMock(),
            max_rounds=3,
            health_check_timeout=2.0,
        )
        assert team._max_rounds == 3
        assert team._health_check_timeout == 2.0
        assert "worker1" in team._workers


# ---------------------------------------------------------------------------
# McpAgentTeam._build_app
# ---------------------------------------------------------------------------


class TestMcpAgentTeamBuildApp:
    def test_build_app_constructs_module(self):
        """_build_app should call LaurenFactory.create and return the app."""
        import sys

        @agent(model="claude-sonnet-4-6")
        class BuildCoord:
            pass

        mock_client = AsyncMock()
        mock_llm_module = MagicMock()
        team = McpAgentTeam(
            coordinator=BuildCoord,
            workers={"w": mock_client},
            llm_module=mock_llm_module,
        )

        mock_app = MagicMock()
        mock_lauren_factory = MagicMock()
        mock_lauren_factory.create.return_value = mock_app

        # _build_app uses lazy imports inside the method body; patch via sys.modules
        fake_lauren = MagicMock()
        fake_lauren.LaurenFactory = mock_lauren_factory
        fake_lauren.module = MagicMock(return_value=lambda cls: cls)

        fake_agent_module_inst = MagicMock()
        fake_lauren_ai_module = MagicMock()
        fake_lauren_ai_module.AgentModule.for_root.return_value = fake_agent_module_inst

        fake_mcp_bridge = MagicMock()
        fake_mcp_bridge.McpServerConfig = MagicMock(return_value=MagicMock())

        orig_lauren = sys.modules.get("lauren")
        orig_ai_module = sys.modules.get("lauren_ai._module")
        orig_mcp_bridge = sys.modules.get("lauren_mcp._bridge")
        try:
            sys.modules["lauren"] = fake_lauren
            sys.modules["lauren_ai._module"] = fake_lauren_ai_module
            sys.modules["lauren_mcp._bridge"] = fake_mcp_bridge

            result = team._build_app()
            assert result is mock_app
            mock_lauren_factory.create.assert_called_once()
        finally:
            if orig_lauren is None:
                sys.modules.pop("lauren", None)
            else:
                sys.modules["lauren"] = orig_lauren
            if orig_ai_module is None:
                sys.modules.pop("lauren_ai._module", None)
            else:
                sys.modules["lauren_ai._module"] = orig_ai_module
            if orig_mcp_bridge is None:
                sys.modules.pop("lauren_mcp._bridge", None)
            else:
                sys.modules["lauren_mcp._bridge"] = orig_mcp_bridge


# ---------------------------------------------------------------------------
# McpAgentTeam.close — swallows exceptions from client.close()
# ---------------------------------------------------------------------------


class TestMcpAgentTeamClose:
    async def test_close_suppresses_errors(self):
        @agent(model="claude-sonnet-4-6")
        class CloseCoord:
            pass

        bad_client = AsyncMock()
        bad_client.close.side_effect = RuntimeError("network error")
        good_client = AsyncMock()

        team = McpAgentTeam(
            coordinator=CloseCoord,
            workers={"bad": bad_client, "good": good_client},
            llm_module=MagicMock(),
        )

        # Should not raise even if one client.close() fails
        await team.close()

        bad_client.close.assert_awaited_once()
        good_client.close.assert_awaited_once()

    async def test_close_calls_all_clients(self):
        @agent(model="claude-sonnet-4-6")
        class CloseCoord2:
            pass

        clients = {f"w{i}": AsyncMock() for i in range(3)}
        team = McpAgentTeam(
            coordinator=CloseCoord2,
            workers=clients,
            llm_module=MagicMock(),
        )
        await team.close()
        for c in clients.values():
            c.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# McpAgentTeam.run — unit test with mocked _ensure_ready / app.get
# ---------------------------------------------------------------------------


class TestMcpAgentTeamRun:
    async def test_run_returns_mcp_team_result(self):
        @agent(model="claude-sonnet-4-6")
        class RunCoord:
            pass

        mock_client = AsyncMock()
        mock_llm = MagicMock()

        team = McpAgentTeam(
            coordinator=RunCoord,
            workers={"w": mock_client},
            llm_module=mock_llm,
        )

        # Build fake response
        mock_usage = MagicMock()
        mock_usage.input_tokens = 50
        mock_usage.output_tokens = 25
        mock_response = MagicMock()
        mock_response.content = "Final answer text"
        mock_response.turns = 2
        mock_response.total_usage = mock_usage

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_response)

        mock_app = MagicMock()
        mock_app.get.return_value = MagicMock()  # coordinator_instance

        # Patch _ensure_ready to return our mock runner and pre-set _app
        async def fake_ensure_ready() -> MagicMock:
            team._app = mock_app
            return mock_runner

        team._ensure_ready = fake_ensure_ready  # type: ignore[method-assign]

        result = await team.run("Do something", conversation_id="conv-42")

        assert isinstance(result, McpTeamResult)
        assert result.final_answer == "Final answer text"
        assert result.rounds == 2
        assert result.total_input_tokens == 50
        assert result.total_output_tokens == 25

    async def test_run_handles_missing_usage(self):
        @agent(model="claude-sonnet-4-6")
        class RunCoord2:
            pass

        mock_client = AsyncMock()
        team = McpAgentTeam(
            coordinator=RunCoord2,
            workers={"w": mock_client},
            llm_module=MagicMock(),
        )

        mock_response = MagicMock()
        mock_response.content = "answer"
        mock_response.turns = 1
        mock_response.total_usage = None

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_response)

        mock_app = MagicMock()
        mock_app.get.return_value = MagicMock()

        async def fake_ensure_ready() -> MagicMock:
            team._app = mock_app
            return mock_runner

        team._ensure_ready = fake_ensure_ready  # type: ignore[method-assign]

        result = await team.run("task")
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
