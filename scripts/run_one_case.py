"""Phase 5 Stage A smoke test: run golden_01 through the real pipeline end to end
(pipeline -> coverage agent -> risk agent -> MCP tools), one real Gemini call per
agent turn, traced in Langfuse. No tuning -- this just records a baseline."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.orchestrator import triage

GOLDEN_PATH = config.ROOT / "evals" / "golden" / "golden.json"


def _load_case(case_id: str) -> dict:
    cases = json.loads(GOLDEN_PATH.read_text())
    return next(c for c in cases if c["case_id"] == case_id)


async def main() -> None:
    case = _load_case("golden_01")
    result = await triage(case["contract_id"], case["claim_text"], case["claim_amount_eur"])

    print(f"case: {case['case_id']}")
    print(f"decision: {result.decision} (expected: {case['expected_decision']})")
    print(f"decision_clause_id: {result.decision_clause_id}")
    print(f"coverage: {result.coverage}")
    print(f"stat_bucket: {result.stat_bucket} (expected: {case['expected_risk_bucket']})")
    print(f"effective_amount_eur: {result.effective_amount_eur}")
    print(f"validated_flags: {result.validated_flags}")
    print(f"dropped_flags: {result.dropped_flags}")
    print(f"coverage agent steps: {result.coverage_run.steps}, tool calls: {len(result.coverage_run.tool_calls)}")
    print(f"risk agent steps: {result.risk_run.steps}, tool calls: {len(result.risk_run.tool_calls)}")


if __name__ == "__main__":
    asyncio.run(main())
