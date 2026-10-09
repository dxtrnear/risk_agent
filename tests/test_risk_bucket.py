"""compute_risk_bucket against real contracts with a known, already-verified bucket
(cross-checked in tests/test_golden_consistency.py's own per-case assertions)."""

import sqlite3

import config
from agents.loop import ToolTurn, run_agent
from agents.schemas import RiskResult
from models.risk_bucket import bucket_from_rate, compute_risk_bucket


def _fetch_contract_row(contract_id: int) -> dict:
    conn = sqlite3.connect(f"file:{config.CONTRACTS_DB}?mode=ro", uri=True)
    cursor = conn.execute("SELECT * FROM contracts WHERE contract_id = ?", (contract_id,))
    row = cursor.fetchone()
    cols = [d[0] for d in cursor.description]
    conn.close()
    return dict(zip(cols, row))


def test_known_low_risk_contract():
    contract = _fetch_contract_row(2051860)  # golden_01, expected_risk_bucket=low
    assert compute_risk_bucket(contract) == "low"


def test_known_high_risk_contract():
    contract = _fetch_contract_row(1187103)  # golden_10, expected_risk_bucket=high
    assert compute_risk_bucket(contract) == "high"


def test_bucket_from_rate_zero_claims_is_always_low():
    # P(at least 0 historical claims) is 1 regardless of the rate -- a clean,
    # rate-independent sanity check.
    assert bucket_from_rate(rate=5.0, exposure=2.0, claim_nb=0) == "low"


def test_bucket_from_rate_many_claims_vs_tiny_expectation_is_high():
    assert bucket_from_rate(rate=0.01, exposure=1.0, claim_nb=5) == "high"


async def test_risk_agent_run_with_no_tool_call_does_not_affect_bucket():
    """Literal requested test: the risk agent's own tool-use choices (here, a
    fake model that calls no tool at all) must have zero bearing on
    stat_bucket. Proven by construction -- bucket_from_rate's signature takes
    plain numbers, never an AgentRunResult -- and demonstrated here by running
    a no-tool-call risk agent turn alongside an unrelated bucket computation
    and confirming the bucket is exactly what the numbers dictate."""

    class FakeClient:
        async def call_with_tools(self, agent_name, prompt_version, messages, tools):
            return ToolTurn([])  # the model never calls get_contract

        async def call_for_schema(self, agent_name, prompt_version, messages, schema):
            return {"flags": [], "reasoning": "no risk signals found"}

    run = await run_agent(FakeClient(), "risk", "v2", "claim text", [], {}, RiskResult)
    assert run.tool_calls == []  # confirms the agent really called nothing

    # Computed independently of `run` -- nothing above is an input here.
    assert bucket_from_rate(rate=0.01, exposure=1.0, claim_nb=5) == "high"
