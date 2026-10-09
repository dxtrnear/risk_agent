"""Hand-written agent loop: alternates tool-calling turns with the model, then
makes one separate final call with a response schema and no tools (DECISIONS #45
-- combining active function-calling with response_schema is an unreliable Gemini
pattern). Deliberately client-agnostic: step 2 wires a real Gemini-backed
AgentClient; tests here use a scripted fake one.

Async throughout: tool_fns call out to a real MCP ClientSession (async-only, no
sync wrapper in the installed mcp version), so the loop that awaits them must be
async too. The Gemini client's own calls are sync-blocking under the hood but its
AgentClient methods are still declared async for one uniform calling convention.
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

import config


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any]
    # Opaque Gemini-specific field (an opaque signature over the model's own
    # reasoning for this call) that must be replayed verbatim on the next turn or
    # the API rejects the request. The loop never reads this, only stores and
    # forwards it; only agents/gemini_client.py gives it meaning.
    thought_signature: bytes | None = None


@dataclass
class ToolTurn:
    tool_calls: list[ToolCallRequest]  # empty -> model is done calling tools


class AgentClient(Protocol):
    async def call_with_tools(
        self, agent_name: str, prompt_version: str, messages: list[dict], tools: list
    ) -> ToolTurn: ...

    async def call_for_schema(
        self, agent_name: str, prompt_version: str, messages: list[dict], schema: type[BaseModel]
    ) -> dict: ...


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class AgentRunResult:
    output: BaseModel
    tool_calls: list[ToolCallRecord]
    steps: int


async def run_agent(
    client: AgentClient,
    agent_name: str,
    prompt_version: str,
    user_content: str,
    tools: list,
    tool_fns: dict[str, Any],
    response_schema: type[BaseModel],
) -> AgentRunResult:
    """Run one agent to completion. `user_content` carries the claim; the system
    prompt lives with the client/model config, never formatted in here."""
    messages = [{"role": "user", "content": user_content}]
    tool_call_records: list[ToolCallRecord] = []
    deadline = time.monotonic() + config.AGENT_TIMEOUT_SECONDS

    for step in range(1, config.AGENT_STEP_LIMIT + 1):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{agent_name}: exceeded {config.AGENT_TIMEOUT_SECONDS}s")

        turn = await client.call_with_tools(agent_name, prompt_version, messages, tools)
        if not turn.tool_calls:
            break

        # Record the model's own function-call turn before the results, so a real
        # client can faithfully replay it (Gemini requires the function_call parts
        # it emitted back in history, paired with their function_response parts).
        messages.append(
            {"role": "model", "tool_calls": [c.model_dump() for c in turn.tool_calls]}
        )
        for call in turn.tool_calls:
            if call.name not in tool_fns:
                result = f"error: unknown tool {call.name!r}, call refused"
            else:
                try:
                    result = await tool_fns[call.name](**call.arguments)
                except ToolError as exc:
                    result = f"error: {exc}"
            tool_call_records.append(ToolCallRecord(call.name, call.arguments, result))
            messages.append({"role": "tool", "name": call.name, "content": result})
    else:
        raise RuntimeError(f"{agent_name}: exceeded step limit ({config.AGENT_STEP_LIMIT})")

    if time.monotonic() > deadline:
        raise TimeoutError(f"{agent_name}: exceeded {config.AGENT_TIMEOUT_SECONDS}s")

    raw = await client.call_for_schema(agent_name, prompt_version, messages, response_schema)
    output = response_schema.model_validate(raw)
    return AgentRunResult(output=output, tool_calls=tool_call_records, steps=step)
