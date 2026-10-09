"""The pipeline: deterministic code that calls the coverage and risk agents, then
applies the decision table in CLAUDE.md in code. Never call this "the orchestrator
agent" -- it has no judgment of its own, only the agents' validated outputs and a
fixed table (DECISIONS #47).
"""

import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

from langfuse import get_client, propagate_attributes
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
from agents.coverage import PROMPT_VERSION as COVERAGE_PROMPT_VERSION
from agents.coverage import run_coverage_agent
from agents.loop import AgentRunResult
from agents.mcp_tools import make_tool_fns
from agents.risk import PROMPT_VERSION as RISK_PROMPT_VERSION
from agents.risk import run_risk_agent
from agents.schemas import CoverageResult, RedFlag
from models.risk_bucket import bucket_from_rate

SERVER_SCRIPT = config.ROOT / "mcp_server" / "server.py"

Decision = Literal["approve", "reject", "ask_info", "escalate"]


@dataclass
class TriageResult:
    decision: Decision
    decision_clause_id: str | None
    coverage: CoverageResult
    stat_bucket: Literal["low", "medium", "high"]
    effective_amount_eur: float | None
    validated_flags: list[RedFlag]
    dropped_flags: list[RedFlag]  # proposed by the risk agent, evidence check failed
    coverage_run: AgentRunResult
    risk_run: AgentRunResult
    trace_id: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _validate_flags(flags: list[RedFlag], claim_text: str) -> tuple[list[RedFlag], list[RedFlag]]:
    """A flag's evidence must be an exact, whitespace-normalized substring of the
    claim text -- the LLM proposing a flag is not sufficient grounds to act on it."""
    normalized_claim = _normalize(claim_text)
    validated, dropped = [], []
    for flag in flags:
        if _normalize(flag.evidence) in normalized_claim:
            validated.append(flag)
        else:
            dropped.append(flag)
    return validated, dropped


def _valid_clause_ids() -> set[str]:
    clauses = json.loads(config.CLAUSES_JSON.read_text())
    return {c["clause_id"] for c in clauses}


def _check_clause_id(clause_id: str | None) -> None:
    if clause_id is not None and clause_id not in _valid_clause_ids():
        raise ValueError(f"hallucinated clause_id from the coverage agent: {clause_id!r}")


def _effective_amount(claim_amount_eur: float | None, limitation: str | None) -> float | None:
    if claim_amount_eur is None:
        return None
    fraction = config.PARTIAL_COVERAGE_FRACTIONS.get(limitation, 1.0)
    return claim_amount_eur * fraction


def _decide(
    coverage: CoverageResult,
    claim_amount_eur: float | None,
    effective_amount_eur: float | None,
    franchise: float,
    stat_bucket: str,
    validated_flags: list[RedFlag],
) -> tuple[Decision, str | None]:
    """The decision table from CLAUDE.md, precedence in order (first match wins)."""
    if coverage.coverage == "missing_info":
        return "ask_info", None
    if coverage.coverage == "not_covered":
        return "reject", coverage.clause_id
    if claim_amount_eur is None:
        return "ask_info", None

    is_rc = (coverage.clause_id or "").startswith("4.")
    if not is_rc and effective_amount_eur is not None and effective_amount_eur <= franchise:
        return "reject", config.FRANCHISE_CLAUSE_ID

    amount_over = (
        effective_amount_eur is not None and effective_amount_eur > config.CLAIM_AMOUNT_THRESHOLD_EUR
    )
    if stat_bucket == "high" or amount_over or validated_flags:
        return "escalate", coverage.clause_id
    return "approve", coverage.clause_id


def git_commit_hash() -> str:
    """Public: also used by evals/run_eval.py to stamp every record."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=config.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class AgentFailure(Exception):
    """Raised when either agent errors out; the caller should treat this claim as
    needing human escalation rather than silently failing the whole pipeline."""


async def triage(
    contract_id: int,
    claim_text: str,
    claim_amount_eur: float | None,
    run_id: str = "single",
    eval_set: str = "golden",
) -> TriageResult:
    """`run_id` groups every claim from one evaluation run under one Langfuse
    session (e.g. a smoke-set or baseline run); `eval_set` tags which case
    population it's from (smoke/golden/production)."""
    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_SCRIPT), str(contract_id)]
    )
    # Catch the agent failure *inside* the nested context managers and only raise
    # it after both have exited cleanly. stdio_client/ClientSession are each
    # backed by an anyio task group, which wraps an exception propagating through
    # their __aexit__ in a BaseExceptionGroup (doubly, for the two nesting levels)
    # -- raising AgentFailure directly from inside here would reach the caller as
    # an ExceptionGroup, not a plain AgentFailure, and a simple `except
    # AgentFailure` would never match it.
    agent_error: Exception | None = None
    coverage_run = risk_run = contract = None
    langfuse_client = get_client()

    with langfuse_client.start_as_current_observation(
        name="process_claim",
        as_type="span",
        input={"claim_text": claim_text, "claim_amount_eur": claim_amount_eur},
        metadata={
            "contract_id": contract_id,
            "coverage_prompt_version": COVERAGE_PROMPT_VERSION,
            "risk_prompt_version": RISK_PROMPT_VERSION,
            "model": config.GEMINI_MODEL_LIGHT,
            "git_commit": git_commit_hash(),
        },
    ) as root:
        # Every span created below is explicitly parented via trace_context
        # (passed down through run_coverage_agent/run_risk_agent into
        # agents/gemini_client.py and agents/mcp_tools.py) rather than relying on
        # ambient OTel context -- ambient nesting does not reliably survive the
        # stdio_client/ClientSession anyio task-group boundaries a few frames
        # down (confirmed broken in practice: every call was its own trace).
        # propagate_attributes still uses ambient (baggage) propagation for
        # session_id/tags, which is a separate, more robust mechanism in OTel.
        with propagate_attributes(
            session_id=run_id, tags=[eval_set, "stage_a"], as_baggage=True
        ):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    try:
                        coverage_run = await run_coverage_agent(
                            session, claim_text, root.trace_id, root.id
                        )
                        risk_run = await run_risk_agent(
                            session, claim_text, root.trace_id, root.id
                        )
                    except Exception as exc:
                        agent_error = exc
                    else:
                        # The pipeline calls get_contract and risk_score itself,
                        # unconditionally, as plain code -- stat_bucket must
                        # never depend on whether either agent chose to call a
                        # tool (DECISIONS #111). Traced as pipeline-level tool
                        # spans, parented to root directly (not to either agent).
                        pipeline_tools = make_tool_fns(
                            session, ["get_contract", "risk_score"], root.trace_id, root.id
                        )
                        contract = json.loads(await pipeline_tools["get_contract"]())
                        rate = json.loads(await pipeline_tools["risk_score"]())[
                            "expected_claims_per_year"
                        ]

        trace_id = root.trace_id

        if agent_error is not None:
            root.update(output={"error": str(agent_error)})
            langfuse_client.flush()
            # Agents disagreeing or erroring is a separate, runtime-only
            # escalation path, not part of the main decision table.
            raise AgentFailure(
                f"agent run failed, this claim must escalate: {agent_error}"
            ) from agent_error

        coverage = coverage_run.output
        _check_clause_id(coverage.clause_id)

        stat_bucket = bucket_from_rate(rate, contract["exposure"], contract["claim_nb"])
        validated_flags, dropped_flags = _validate_flags(risk_run.output.flags, claim_text)
        effective_amount_eur = _effective_amount(claim_amount_eur, coverage.limitation)

        decision, decision_clause_id = _decide(
            coverage,
            claim_amount_eur,
            effective_amount_eur,
            contract["franchise"],
            stat_bucket,
            validated_flags,
        )
        _check_clause_id(decision_clause_id)
        root.update(output={"decision": decision, "decision_clause_id": decision_clause_id})

    langfuse_client.flush()

    return TriageResult(
        decision=decision,
        decision_clause_id=decision_clause_id,
        coverage=coverage,
        stat_bucket=stat_bucket,
        effective_amount_eur=effective_amount_eur,
        validated_flags=validated_flags,
        dropped_flags=dropped_flags,
        coverage_run=coverage_run,
        risk_run=risk_run,
        trace_id=trace_id,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one claim through the pipeline.")
    parser.add_argument("contract_id", type=int)
    parser.add_argument("claim_text")
    parser.add_argument("--amount", type=float, default=None, dest="claim_amount_eur")
    args = parser.parse_args()

    result = asyncio.run(triage(args.contract_id, args.claim_text, args.claim_amount_eur))

    print(f"trace_id: {result.trace_id}")
    print(f"decision: {result.decision}")
    print(f"decision_clause_id: {result.decision_clause_id}")
    print(f"coverage: {result.coverage}")
    print(f"stat_bucket: {result.stat_bucket}")
    print(f"effective_amount_eur: {result.effective_amount_eur}")
    print(f"validated_flags: {result.validated_flags}")
    print(f"dropped_flags: {result.dropped_flags}")
