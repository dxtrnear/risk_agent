"""Run one named case N times (same case, repeated under one shared Langfuse
session) to measure its own stability, independent of the rest of the eval
set. Reuses run_eval.py's record-building and scoring -- same JSONL shape,
same score table. No MLflow logging here; this is ad hoc stability
measurement, not a tracked baseline run.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.orchestrator import git_commit_hash
from evals.metrics import print_score_table
from evals.run_eval import GOLDEN_PATH, RESULTS_DIR, _run_one


def _load_case(case_id: str) -> dict:
    cases = {c["case_id"]: c for c in json.loads(GOLDEN_PATH.read_text())}
    return cases[case_id]


async def main(case_id: str, n: int) -> None:
    case = _load_case(case_id)
    git_commit = git_commit_hash()
    timestamp = int(time.time())
    run_id = f"repeat-{case_id}-{timestamp}"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"repeat-{case_id}-{timestamp}.jsonl"
    print(f"session: {run_id}")

    records = []
    with out_path.open("w") as f:
        for run_index in range(1, n + 1):
            print(f"  run {run_index}/{n}...", end=" ", flush=True)
            record = await _run_one(case, run_id, run_index, "golden", git_commit)
            print(record["decision"] or f"ERROR: {record['error']}")
            f.write(json.dumps(record) + "\n")
            records.append(record)

    print(f"\nwrote {len(records)} records to {out_path}")
    print()
    print_score_table(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one case N times to measure its stability.")
    parser.add_argument("case_id")
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    asyncio.run(main(args.case_id, args.n))
