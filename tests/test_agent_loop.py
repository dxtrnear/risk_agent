"""Agent loop control flow, exercised with a scripted fake client -- no LLM calls."""

from dataclasses import dataclass, field

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

import config
from agents.loop import ToolCallRequest, ToolTurn, run_agent


class Result(BaseModel):
    answer: str


@dataclass
class FakeClient:
    tool_turns: list[ToolTurn]
    final_output: dict
    calls: list = field(default_factory=list)

    async def call_with_tools(self, agent_name, prompt_version, messages, tools):
        self.calls.append(("tools", list(messages)))
        return self.tool_turns.pop(0)

    async def call_for_schema(self, agent_name, prompt_version, messages, schema):
        self.calls.append(("schema", list(messages)))
        return self.final_output


async def _echo_tool(**kwargs):
    return "ok"


async def test_tool_call_then_final_answer():
    turns = [
        ToolTurn([ToolCallRequest(name="get_contract", arguments={})]),
        ToolTurn([]),
    ]
    client = FakeClient(tool_turns=turns, final_output={"answer": "done"})
    run = await run_agent(client, "coverage", "v1", "claim text", [], {"get_contract": _echo_tool}, Result)

    assert run.output == Result(answer="done")
    assert [c.name for c in run.tool_calls] == ["get_contract"]
    assert run.steps == 2


async def test_unknown_tool_is_refused_not_executed():
    calls = []

    async def tracked(**kwargs):
        calls.append(kwargs)
        return "should not run"

    turns = [ToolTurn([ToolCallRequest(name="not_a_real_tool", arguments={})]), ToolTurn([])]
    client = FakeClient(tool_turns=turns, final_output={"answer": "done"})
    run = await run_agent(client, "coverage", "v1", "claim text", [], {"get_contract": tracked}, Result)

    assert calls == []
    assert "unknown tool" in run.tool_calls[0].result


async def test_tool_error_message_passed_back_loop_continues():
    async def failing_tool(**kwargs):
        raise ToolError("bad arguments")

    turns = [ToolTurn([ToolCallRequest(name="risk_score", arguments={})]), ToolTurn([])]
    client = FakeClient(tool_turns=turns, final_output={"answer": "done"})
    run = await run_agent(client, "risk", "v1", "claim text", [], {"risk_score": failing_tool}, Result)

    assert "bad arguments" in run.tool_calls[0].result
    assert run.output == Result(answer="done")


async def test_step_limit_exceeded_raises():
    never_ending_turn = ToolTurn([ToolCallRequest(name="get_contract", arguments={})])
    turns = [never_ending_turn] * config.AGENT_STEP_LIMIT
    client = FakeClient(tool_turns=turns, final_output={"answer": "done"})

    with pytest.raises(RuntimeError, match="step limit"):
        await run_agent(client, "coverage", "v1", "claim text", [], {"get_contract": _echo_tool}, Result)


async def test_timeout_exceeded_raises(monkeypatch):
    import agents.loop as loop_module

    times = iter([0, 1000])
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: next(times, 1000))
    client = FakeClient(tool_turns=[], final_output={"answer": "done"})

    with pytest.raises(TimeoutError):
        await run_agent(client, "coverage", "v1", "claim text", [], {}, Result)


async def test_final_output_failing_schema_raises():
    turns = [ToolTurn([])]
    client = FakeClient(tool_turns=turns, final_output={"wrong_field": "nope"})

    with pytest.raises(Exception):
        await run_agent(client, "coverage", "v1", "claim text", [], {}, Result)
