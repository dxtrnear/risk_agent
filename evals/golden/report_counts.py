"""Report final per-decision and per-reject-reason counts for golden.json."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

if __name__ == "__main__":
    cases = json.loads((config.ROOT / "evals" / "golden" / "golden.json").read_text())
    print(f"total: {len(cases)}")
    print("by decision:", dict(Counter(c["expected_decision"] for c in cases)))

    rejects = [c for c in cases if c["expected_decision"] == "reject"]
    print("reject by reason:", dict(Counter(c["reject_reason"] for c in rejects)))

    escalates = [c for c in cases if c["expected_decision"] == "escalate"]

    def esc_type(c):
        high = c["expected_risk_bucket"] == "high"
        amt = c["effective_amount_eur"] and c["effective_amount_eur"] > config.CLAIM_AMOUNT_THRESHOLD_EUR
        flag = len(c["expected_flags"]) > 0
        triggers = [t for t, v in [("risk", high), ("amount", amt), ("flag", flag)] if v]
        return "+".join(triggers)

    print("escalate by trigger:", dict(Counter(esc_type(c) for c in escalates)))
    print("flagged cases:", sum(1 for c in cases if c["expected_flags"]))
