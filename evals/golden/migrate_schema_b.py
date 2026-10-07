"""One-time migration: acceptable_clause_ids -> agent_clause_ids + decision_clause_id,
add expected_limitation, drop expected_tools.

Franchise-reject cases need their cover clause (currently absent) split out from the
franchise clause -- known precisely for golden.json (hardcoded below, since the
clause depends on which peril each case is about, not something inferable from the
case's own fields). For production.json, only the safe generic transform runs; cases
matching the same structural pattern (reject + covered, i.e. likely franchise) are
printed by case_id only -- never their content -- for manual follow-up.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

GOLDEN_PATH = config.ROOT / "evals" / "golden" / "golden.json"
PRODUCTION_PATH = config.ROOT / "evals" / "golden" / "production.json"

# case_id -> cover clause (the garantie clause, as opposed to the franchise clause
# already in acceptable_clause_ids). Known precisely only for golden.json.
GOLDEN_FRANCHISE_COVER_CLAUSE = {
    "golden_05": "6.5",
    "golden_16": "6.1",
    "golden_26": "6.5",
    "golden_46": "6.1",
}
GOLDEN_KEYS_IN_VEHICLE = {"golden_10", "golden_50"}


def _migrate_case(case: dict) -> dict:
    agent_clause_ids = list(case["acceptable_clause_ids"])
    decision_clause_id = agent_clause_ids[0] if agent_clause_ids else None

    cover_clause = GOLDEN_FRANCHISE_COVER_CLAUSE.get(case["case_id"])
    if cover_clause is not None:
        decision_clause_id = agent_clause_ids[0]  # the franchise clause, e.g. "8.1"
        agent_clause_ids = [cover_clause]

    case["agent_clause_ids"] = agent_clause_ids
    case["decision_clause_id"] = decision_clause_id
    case["expected_limitation"] = "keys_in_vehicle" if case["case_id"] in GOLDEN_KEYS_IN_VEHICLE else None
    del case["acceptable_clause_ids"]
    case.pop("expected_tools", None)
    return case


def migrate(path: Path) -> list[str]:
    """Migrate every case in `path`. Returns case_ids matching the reject+covered
    (likely franchise) pattern, for manual review when the file is production.json."""
    cases = json.loads(path.read_text())
    flagged = [
        c["case_id"]
        for c in cases
        if c["expected_decision"] == "reject" and c["expected_coverage"] == "covered"
    ]
    cases = [_migrate_case(c) for c in cases]
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    return flagged


if __name__ == "__main__":
    golden_flagged = migrate(GOLDEN_PATH)
    print(f"golden.json: {len(golden_flagged)} reject+covered cases (all handled above)")

    prod_flagged = migrate(PRODUCTION_PATH)
    print(f"production.json: migrated. {len(prod_flagged)} reject+covered case(s) need your manual")
    print("agent_clause_ids/decision_clause_id split (cover clause currently missing,")
    print("franchise clause is in decision_clause_id by default -- fix if that's wrong):")
    for case_id in prod_flagged:
        print(f"  - {case_id}")
