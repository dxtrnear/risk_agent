"""Phase 5 Stage A smoke test, 10 cases: same pipeline as run_one_case.py, looped.
Baseline only -- no tuning. One case failing (either agent erroring) doesn't stop
the rest; it's recorded as a mismatch so the summary table stays complete."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.orchestrator import AgentFailure, triage

GOLDEN_PATH = config.ROOT / "evals" / "golden" / "golden.json"
SMOKE_CASE_IDS = [f"golden_{i:02d}" for i in range(1, 11)]


def _load_cases() -> list[dict]:
    all_cases = {c["case_id"]: c for c in json.loads(GOLDEN_PATH.read_text())}
    return [all_cases[case_id] for case_id in SMOKE_CASE_IDS]


async def main() -> None:
    rows = []
    for case in _load_cases():
        print(f"running {case['case_id']}...", end=" ", flush=True)
        try:
            result = await triage(case["contract_id"], case["claim_text"], case["claim_amount_eur"])
            decision = result.decision
            bucket = result.stat_bucket
            flags = len(result.validated_flags)
        except AgentFailure as exc:
            decision = f"ERROR: {exc}"
            bucket = "n/a"
            flags = "n/a"
        print(decision)
        rows.append((case["case_id"], decision, case["expected_decision"], bucket, case["expected_risk_bucket"], flags))

    print(f"{'case':<12}{'decision':<12}{'expected':<12}{'match':<7}{'bucket':<10}{'exp_bucket':<12}{'flags'}")
    correct = 0
    for case_id, decision, expected, bucket, expected_bucket, flags in rows:
        match = decision == expected
        correct += match
        print(f"{case_id:<12}{decision!s:<12}{expected:<12}{str(match):<7}{bucket!s:<10}{expected_bucket:<12}{flags}")
    print(f"\n{correct}/{len(rows)} decisions matched expected (baseline, no tuning yet)")


if __name__ == "__main__":
    asyncio.run(main())
