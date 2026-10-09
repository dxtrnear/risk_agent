"""Builds the `tool_fns` dict agents/loop.py's run_agent expects, backed by one
shared MCP ClientSession (one subprocess per claim, reused across both agents).
Each call is also a Langfuse "tool" span, explicitly parented via `trace_context`
(see agents/gemini_client.py's docstring for why ambient nesting isn't used).
"""

import functools
from typing import Any, Callable

from dotenv import load_dotenv
from langfuse import get_client
from mcp import ClientSession
from mcp.server.mcpserver.exceptions import ToolError

load_dotenv()


async def _call_tool(
    session: ClientSession, name: str, parent_trace_id: str, parent_span_id: str, **arguments: Any
) -> str:
    langfuse_client = get_client()
    with langfuse_client.start_as_current_observation(
        name=name,
        as_type="tool",
        input=arguments,
        trace_context={"trace_id": parent_trace_id, "parent_span_id": parent_span_id},
    ) as span:
        result = await session.call_tool(name, arguments)
        if result.is_error:
            span.update(output={"error": result.content[0].text})
            raise ToolError(result.content[0].text)
        span.update(output=result.content[0].text)
        return result.content[0].text


def make_tool_fns(
    session: ClientSession, tool_names: list[str], parent_trace_id: str, parent_span_id: str
) -> dict[str, Callable]:
    """One async wrapper per tool name, each a thin partial over `_call_tool`."""
    return {
        name: functools.partial(_call_tool, session, name, parent_trace_id, parent_span_id)
        for name in tool_names
    }
