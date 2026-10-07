"""Diagnostic (not an assertion): for each golden case, check whether its
agent_clause_ids are consistent with the contract's formule under config.FORMULE_COVERAGES.
Run before hardening this into a test assertion, to see what it would break.
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

KNOWN_COVER_CLAUSES = {cid for covers in config.FORMULE_COVERAGES.values() for cid in covers}


def _formule(contract_id: int) -> str:
    conn = sqlite3.connect(f"file:{config.CONTRACTS_DB}?mode=ro", uri=True)
    row = conn.execute("SELECT formule FROM contracts WHERE contract_id = ?", (contract_id,)).fetchone()
    conn.close()
    return row[0]


if __name__ == "__main__":
    cases = json.loads((config.ROOT / "evals" / "golden" / "golden.json").read_text())
    problems = []
    for case in cases:
        formule = _formule(case["contract_id"])
        subscribed = config.FORMULE_COVERAGES[formule]
        cover_clauses = [c for c in case["agent_clause_ids"] if c in KNOWN_COVER_CLAUSES]

        if case["expected_coverage"] == "covered":
            if cover_clauses and not any(c in subscribed for c in cover_clauses):
                problems.append(
                    f"{case['case_id']}: covered, formule={formule}, "
                    f"agent_clause_ids={case['agent_clause_ids']} -- NONE in subscribed covers"
                )
        elif case["expected_coverage"] == "not_covered":
            bad = [c for c in cover_clauses if c in subscribed]
            if bad:
                problems.append(
                    f"{case['case_id']}: not_covered, formule={formule}, "
                    f"but {bad} IS in subscribed covers (not a real formule-gap)"
                )

    print(f"{len(cases)} cases checked, {len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
