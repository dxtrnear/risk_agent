"""Phase 2 checkpoint: expected clause in top-5 for at least 13 of 15 test queries."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from policy.index_policy import search

PASS_THRESHOLD = 13


def evaluate() -> None:
    queries = json.loads((config.ROOT / "policy" / "test_queries.json").read_text())

    hits = 0
    for case in queries:
        results = search(case["query"], k=5)
        found_ids = [r["clause_id"] for r in results]
        hit = case["expected_clause_id"] in found_ids
        hits += hit
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] {case['query'][:60]:60} expected={case['expected_clause_id']:10} got={found_ids}")

    print(f"\n{hits}/{len(queries)} passed (need >= {PASS_THRESHOLD})")
    if hits < PASS_THRESHOLD:
        raise RuntimeError(f"Only {hits}/{len(queries)} queries hit top-5; checkpoint not met.")


if __name__ == "__main__":
    evaluate()
