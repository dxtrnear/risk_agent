"""MCP server tests: no LLM calls, covers the Phase 3 security checkpoint."""

import json
import sqlite3

import joblib
import pandas as pd
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import config
from mcp_server.server import create_server

EXISTING_CONTRACT_ID = 1
OTHER_CONTRACT_ID = 3
UNKNOWN_CONTRACT_ID = 999_999_999


def _result_json(call_tool_result) -> dict:
    return json.loads(call_tool_result.content[0].text)


def _stub_search_must_not_be_called(query: str, k: int = 5) -> list:
    raise AssertionError("search_fn must not be called when validation fails")


# --- identity / session binding -------------------------------------------------


async def test_get_contract_returns_bound_contract():
    server = create_server(EXISTING_CONTRACT_ID)
    result = await server.call_tool("get_contract", {})
    assert _result_json(result)["contract_id"] == EXISTING_CONTRACT_ID


async def test_cross_customer_access_impossible():
    """Injecting a different contract_id as a tool argument has zero effect.

    get_contract/risk_score take no parameters, so there is no argument path for a
    model to override the bound session. This also pins the mcp 2.3.0 SDK's own
    behaviour of silently dropping unrecognized arguments, which this design relies
    on — if a future mcp upgrade starts rejecting unknown arguments instead, this
    test still passes; if it ever started *honoring* them, this test would catch it.
    """
    server = create_server(EXISTING_CONTRACT_ID)
    result = await server.call_tool("get_contract", {"contract_id": OTHER_CONTRACT_ID})
    assert _result_json(result)["contract_id"] == EXISTING_CONTRACT_ID


def test_insert_fails_on_readonly_connection():
    conn = sqlite3.connect(f"file:{config.CONTRACTS_DB}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO contracts (contract_id) VALUES (999999998)")
    conn.close()


def test_non_integer_contract_id_rejected_at_startup():
    with pytest.raises(TypeError):
        create_server("1 OR 1=1")


def test_unknown_contract_id_fails_loudly_at_startup():
    with pytest.raises(ValueError):
        create_server(UNKNOWN_CONTRACT_ID)


# --- risk_score parity ------------------------------------------------------------


async def test_risk_score_matches_training_style_prediction():
    """Catches serving-time feature construction silently diverging from training."""
    conn = sqlite3.connect(config.CONTRACTS_DB)
    df = pd.read_sql("SELECT * FROM contracts", conn)
    conn.close()
    for col in config.CATEGORICAL:
        df[col] = df[col].astype("category")
    row = df[df["contract_id"] == EXISTING_CONTRACT_ID]

    model = joblib.load(config.RISK_MODEL_PATH)
    expected_rate = float(model.predict(row[config.FEATURES])[0])

    server = create_server(EXISTING_CONTRACT_ID)
    result = await server.call_tool("risk_score", {})
    actual_rate = _result_json(result)["expected_claims_per_year"]

    assert actual_rate == pytest.approx(expected_rate)


# --- search_clauses: validation must precede any search_fn call -----------------


async def test_search_clauses_rejects_empty_query_before_calling_search_fn():
    server = create_server(
        EXISTING_CONTRACT_ID, search_fn=_stub_search_must_not_be_called
    )
    with pytest.raises(ToolError, match="(?i)empty"):
        await server.call_tool("search_clauses", {"query": "   ", "k": 3})


async def test_search_clauses_rejects_k_out_of_range_before_calling_search_fn():
    server = create_server(
        EXISTING_CONTRACT_ID, search_fn=_stub_search_must_not_be_called
    )
    with pytest.raises(ToolError, match="k must be between 1 and 10"):
        await server.call_tool("search_clauses", {"query": "pare-brise", "k": 99})


async def test_search_clauses_rejects_bad_type_for_k():
    server = create_server(
        EXISTING_CONTRACT_ID, search_fn=_stub_search_must_not_be_called
    )
    with pytest.raises(ToolError):
        await server.call_tool("search_clauses", {"query": "pare-brise", "k": "five"})


async def test_search_clauses_rejects_missing_required_query():
    server = create_server(
        EXISTING_CONTRACT_ID, search_fn=_stub_search_must_not_be_called
    )
    with pytest.raises(ToolError):
        await server.call_tool("search_clauses", {"k": 3})


async def test_search_clauses_happy_path_uses_injected_stub():
    canned = [
        {"clause_id": "6.1", "title": "Bris de glace", "text": "...", "score": 0.9}
    ]

    def stub_search(query: str, k: int = 5) -> list:
        return canned

    server = create_server(EXISTING_CONTRACT_ID, search_fn=stub_search)
    result = await server.call_tool(
        "search_clauses", {"query": "pare-brise cassé", "k": 3}
    )
    assert _result_json(result)["results"] == canned
