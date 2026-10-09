"""Builds the `tool_fns` dict agents/loop.py's run_agent expects, backed by one
shared MCP ClientSession (one subprocess per claim, reused across both agents).
"""

import functools
from typing import Any, Callable

from mcp import ClientSession
from mcp.server.mcpserver.exceptions import ToolError


async def _call_tool(session: ClientSession, name: str, **arguments: Any) -> str:
    result = await session.call_tool(name, arguments)
    if result.is_error:
        raise ToolError(result.content[0].text)
    return result.content[0].text


def make_tool_fns(session: ClientSession, tool_names: list[str]) -> dict[str, Callable]:
    """One async wrapper per tool name, each a thin partial over `_call_tool`."""
    return {name: functools.partial(_call_tool, session, name) for name in tool_names}
