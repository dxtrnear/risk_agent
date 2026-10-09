"""make_tool_fns / _call_tool against a fake ClientSession -- no real subprocess."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from agents.mcp_tools import make_tool_fns


@dataclass
class FakeSession:
    responses: dict  # name -> (is_error, text)
    calls: list = field(default_factory=list)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        is_error, text = self.responses[name]
        return SimpleNamespace(is_error=is_error, content=[SimpleNamespace(text=text)])


# Must be valid hex: Langfuse's trace_context parses these with int(x, 16)
# (128-bit trace_id, 64-bit span_id, matching OTel ID widths).
PARENT = ("a" * 32, "b" * 16)


async def test_successful_call_returns_text():
    session = FakeSession(responses={"get_contract": (False, '{"formule": "tous_risques"}')})
    tool_fns = make_tool_fns(session, ["get_contract"], *PARENT)

    result = await tool_fns["get_contract"]()

    assert result == '{"formule": "tous_risques"}'
    assert session.calls == [("get_contract", {})]


async def test_arguments_forwarded():
    session = FakeSession(responses={"search_clauses": (False, "{}")})
    tool_fns = make_tool_fns(session, ["search_clauses"], *PARENT)

    await tool_fns["search_clauses"](query="vol", k=3)

    assert session.calls == [("search_clauses", {"query": "vol", "k": 3})]


async def test_error_result_raises_tool_error():
    session = FakeSession(responses={"search_clauses": (True, "query must not be empty")})
    tool_fns = make_tool_fns(session, ["search_clauses"], *PARENT)

    with pytest.raises(ToolError, match="query must not be empty"):
        await tool_fns["search_clauses"](query="")


async def test_only_named_tools_are_exposed():
    session = FakeSession(responses={})
    tool_fns = make_tool_fns(session, ["get_contract", "search_clauses"], *PARENT)

    assert set(tool_fns) == {"get_contract", "search_clauses"}
