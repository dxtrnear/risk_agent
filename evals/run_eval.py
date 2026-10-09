"""Phase 6 core: runs a named case set N times, one Langfuse session per run,
writes one JSONL record per (case, run) to evals/results/, then prints the
score table. No judge model here -- see judge_calibration.py for that, later.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.coverage import PROMPT_VERSION as COVERAGE_PROMPT_VERSION
from agents.orchestrator import AgentFailure, git_commit_hash, triage
from agents.risk import PROMPT_VERSION as RISK_PROMPT_VERSION
from evals.metrics import print_score_table
from evals.mlflow_log import log_run

GOLDEN_PATH = config.ROOT / "evals" / "golden" / "golden.json"
SMOKE_SET_PATH = config.ROOT / "evals" / "smoke_set.json"
RESULTS_DIR = config.ROOT / "evals" / "results"


def _load_cases(eval_set: str) -> list[dict]:
    all_cases = {c["case_id"]: c for c in json.loads(GOLDEN_PATH.read_text())}
    if eval_set == "golden":
        return list(all_cases.values())
    if eval_set == "smoke":
        case_ids = json.loads(SMOKE_SET_PATH.read_text())
        return [all_cases[cid] for cid in case_ids]
    raise ValueError(f"unknown eval_set: {eval_set!r}")


def _retrieved_clause_ids(tool_calls) -> list[str]:
    ids = []
    for call in tool_calls:
        if call.name == "search_clauses" and not call.result.startswith("error:"):
            data = json.loads(call.result)
            ids.extend(r["clause_id"] for r in data["results"])
    return ids


async def _run_one(case: dict, run_id: str, run_index: int, eval_set: str, git_commit: str) -> dict:
    base = {
        "run_id": run_id,
        "run_index": run_index,
        "case_id": case["case_id"],
        "eval_set": eval_set,
        "scenario": case["scenario"],
        "coverage_prompt_version": COVERAGE_PROMPT_VERSION,
        "risk_prompt_version": RISK_PROMPT_VERSION,
        "model": config.GEMINI_MODEL_LIGHT,
        "git_commit": git_commit,
        "expected_decision": case["expected_decision"],
        "expected_coverage": case["expected_coverage"],
        "agent_clause_ids_expected": case["agent_clause_ids"],
        "expected_limitation": case["expected_limitation"],
        "expected_risk_bucket": case["expected_risk_bucket"],
        "expected_flags": case["expected_flags"],
    }
    start = time.monotonic()
    try:
        result = await triage(
            case["contract_id"], case["claim_text"], case["claim_amount_eur"],
            run_id=run_id, eval_set=eval_set,
        )
        latency = time.monotonic() - start
        return {
            **base,
            "error": None,
            "latency_seconds": latency,
            "decision": result.decision,
            "coverage": result.coverage.coverage,
            "coverage_clause_id": result.coverage.clause_id,
            "decision_clause_id": result.decision_clause_id,
            "retrieved_clause_ids": _retrieved_clause_ids(result.coverage_run.tool_calls),
            "limitation": result.coverage.limitation,
            "stat_bucket": result.stat_bucket,
            "validated_flags": [f.model_dump() for f in result.validated_flags],
            "dropped_flags": [f.model_dump() for f in result.dropped_flags],
            "coverage_tool_calls": [c.name for c in result.coverage_run.tool_calls],
            "risk_tool_calls": [c.name for c in result.risk_run.tool_calls],
            "trace_id": result.trace_id,
            "input_tokens": result.coverage_run.input_tokens + result.risk_run.input_tokens,
            "output_tokens": result.coverage_run.output_tokens + result.risk_run.output_tokens,
        }
    except AgentFailure as exc:
        latency = time.monotonic() - start
        return {
            **base,
            "error": str(exc),
            "latency_seconds": latency,
            "decision": None,
            "coverage": None,
            "coverage_clause_id": None,
            "decision_clause_id": None,
            "retrieved_clause_ids": None,
            "limitation": None,
            "stat_bucket": None,
            "validated_flags": None,
            "dropped_flags": None,
            "coverage_tool_calls": None,
            "risk_tool_calls": None,
            "trace_id": None,
            "input_tokens": None,
            "output_tokens": None,
        }


async def main(eval_set: str, num_runs: int) -> None:
    cases = _load_cases(eval_set)
    git_commit = git_commit_hash()
    timestamp = int(time.time())
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{eval_set}-{timestamp}.jsonl"

    records = []
    with out_path.open("w") as f:
        for run_index in range(1, num_runs + 1):
            run_id = f"{eval_set}-{timestamp}-run{run_index}"
            print(f"\n=== run {run_index}/{num_runs} (session: {run_id}) ===")
            for case in cases:
                print(f"  {case['case_id']}...", end=" ", flush=True)
                record = await _run_one(case, run_id, run_index, eval_set, git_commit)
                print(record["decision"] or f"ERROR: {record['error']}")
                f.write(json.dumps(record) + "\n")
                records.append(record)

    print(f"\nwrote {len(records)} records to {out_path}")
    print()
    print_score_table(records)

    log_run(
        eval_set=eval_set,
        num_runs=num_runs,
        coverage_prompt_version=COVERAGE_PROMPT_VERSION,
        risk_prompt_version=RISK_PROMPT_VERSION,
        model=config.GEMINI_MODEL_LIGHT,
        git_commit=git_commit,
        records=records,
        jsonl_path=out_path,
    )
    print("\nlogged to MLflow experiment 'claims-triage-eval' -- run `mlflow ui` to view it")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a named eval set N times and score it.")
    parser.add_argument("--set", dest="eval_set", choices=["smoke", "golden"], required=True)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main(args.eval_set, args.runs))
