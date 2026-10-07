"""MCP server exposing get_contract, risk_score, search_clauses for one contract."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import config
from policy.index_policy import search as default_search

CONTRACT_COLUMNS = [
    "contract_id",
    "area",
    "veh_power",
    "veh_age",
    "driv_age",
    "bonus_malus",
    "veh_brand",
    "veh_gas",
    "density",
    "region",
    "exposure",
    "claim_nb",
    "formule",
    "franchise",
]


def _fetch_contract(contract_id: int) -> dict:
    """Read one contract row by its primary key. Raises if it doesn't exist."""
    columns_sql = ", ".join(CONTRACT_COLUMNS)
    conn = sqlite3.connect(f"file:{config.CONTRACTS_DB}?mode=ro", uri=True)
    try:
        row = conn.execute(
            f"SELECT {columns_sql} FROM contracts WHERE contract_id = ?", (contract_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"Unknown contract_id: {contract_id}")
    return dict(zip(CONTRACT_COLUMNS, row))


def create_server(contract_id: int, search_fn=default_search) -> MCPServer:
    """Build an MCP server bound to one contract for the lifetime of the process.

    `contract_id` is set by code at startup (e.g. from a CLI argument) — it is the
    session identity for the claim being processed, never a tool argument the model
    can set. Fails loudly at construction if `contract_id` is not an int or does not
    exist, rather than letting a bad session start and returning empty results later.
    """
    if not isinstance(contract_id, int):
        raise TypeError(f"contract_id must be an int, got {type(contract_id).__name__}")

    contract = _fetch_contract(contract_id)
    model = joblib.load(config.RISK_MODEL_PATH)

    server = MCPServer("claims-triage")

    @server.tool()
    def get_contract() -> dict:
        """Return the current customer's contract: coverage tier (formule), deductible
        (franchise), vehicle and driver profile, past claim count, and
        subscribed_covers (the list of clause_ids actually included in this formule
        -- use it to tell what this contract covers without guessing from the
        formule name alone). Takes no arguments — always returns the contract bound
        to this session."""
        result = dict(contract)
        result["subscribed_covers"] = config.FORMULE_COVERAGES[contract["formule"]]
        return result

    @server.tool()
    def risk_score() -> dict:
        """Return the risk model's expected claim frequency for this session's
        contract, as claims per year. This is a rate, not scaled by the contract's
        exposure — scale it yourself before comparing to an actual claim count.
        Takes no arguments — always scores the contract bound to this session."""
        row = pd.DataFrame([contract])
        for col in config.CATEGORICAL:
            row[col] = row[col].astype("category")
        rate = float(model.predict(row[config.FEATURES])[0])
        return {"expected_claims_per_year": rate}

    @server.tool()
    def search_clauses(query: str, k: int = 3) -> dict:
        """Search the policy's conditions générales for clauses relevant to a
        French-language description of a claim or coverage question. Returns up to
        k clauses ranked by relevance, each with its clause_id, title and text. Use
        the clause_id to cite the policy basis for a coverage decision."""
        # Raise ToolError, not a plain exception: the SDK hides a plain exception's
        # message from the model (treats it as a crash) but preserves ToolError's.
        if not query.strip():
            raise ToolError("query must not be empty")
        if not (1 <= k <= 10):
            raise ToolError("k must be between 1 and 10")
        return {"results": search_fn(query, k=k)}

    return server


if __name__ == "__main__":
    import asyncio

    bound_contract_id = int(sys.argv[1])
    srv = create_server(bound_contract_id)
    asyncio.run(srv.run_stdio_async())
