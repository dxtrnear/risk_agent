"""Code-level eval metrics (no judge model) computed from evals/run_eval.py's
JSONL records. Every rate is returned with its raw counts (k of n), never a bare
float -- limitation and flag positives are rare enough that a percentage alone
would hide whether k is 1 or 100.

A record with error != null is a failure for every applicable metric and is
never dropped from a denominator (it still counts as "wrong"), except
evidence_exists_rate: that's a ratio over flags *proposed*, and an error record
proposes none, so it contributes to neither side -- there's nothing to be wrong
about there, unlike every other metric where the case itself still has an
expected answer the error failed to produce.
"""

from collections import Counter

import config

Rate = dict  # {"rate": float, "k": int, "n": int}


def _rate(k: int, n: int) -> Rate:
    return {"rate": (k / n) if n else 0.0, "k": k, "n": n}


def decision_match(records: list[dict]) -> Rate:
    k = sum(1 for r in records if r["error"] is None and r["decision"] == r["expected_decision"])
    return _rate(k, len(records))


def coverage_match(records: list[dict]) -> Rate:
    k = sum(1 for r in records if r["error"] is None and r["coverage"] == r["expected_coverage"])
    return _rate(k, len(records))


def cited_clause_rate(records: list[dict]) -> Rate:
    """Denominator: cases whose golden `agent_clause_ids_expected` is non-empty
    (i.e. not a missing_info case) -- see missing_info_clause_null_rate for those."""
    eligible = [r for r in records if r["agent_clause_ids_expected"]]
    k = sum(
        1 for r in eligible
        if r["error"] is None and r["coverage_clause_id"] in r["agent_clause_ids_expected"]
    )
    return _rate(k, len(eligible))


def missing_info_clause_null_rate(records: list[dict]) -> Rate:
    """For missing_info cases specifically: a non-null citation there is itself
    a hallucination symptom, checked separately from the covered/not_covered case."""
    eligible = [r for r in records if r["expected_coverage"] == "missing_info"]
    k = sum(1 for r in eligible if r["error"] is None and r["coverage_clause_id"] is None)
    return _rate(k, len(eligible))


def limitation_match(records: list[dict]) -> dict:
    """TP/FP/FN over the rare "a limitation applies" positive, not a bare rate."""
    tp = fp = fn = 0
    for r in records:
        expected = r["expected_limitation"]
        actual = None if r["error"] is not None else r["limitation"]
        if expected is not None and actual == expected:
            tp += 1
        elif expected is not None and actual != expected:
            fn += 1
        elif expected is None and actual is not None:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def _flag_types_in(flags: list[dict] | None) -> set[str]:
    return {f["flag"] for f in flags} if flags else set()


def flag_metrics(records: list[dict]) -> dict:
    """Per flag type and overall: TP/FP/FN/precision/recall over validated flags
    vs. the golden case's expected flags (type-level, per case). Plus a separate
    false-positive check restricted to hard-negative cases (scenario starts with
    "hard negative:"), where ANY validated flag -- of any type -- is a failure."""
    per_type: dict[str, dict] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in config.RED_FLAG_TYPES}
    overall = {"tp": 0, "fp": 0, "fn": 0}

    for r in records:
        validated = set() if r["error"] is not None else _flag_types_in(r["validated_flags"])
        expected = _flag_types_in(r["expected_flags"])
        for t in validated | expected:
            bucket = per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
            if t in validated and t in expected:
                bucket["tp"] += 1
                overall["tp"] += 1
            elif t in validated:
                bucket["fp"] += 1
                overall["fp"] += 1
            else:
                bucket["fn"] += 1
                overall["fn"] += 1

    def _with_rates(counts: dict) -> dict:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return {
            **counts,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
        }

    hard_negatives = [r for r in records if (r.get("scenario") or "").startswith("hard negative")]
    hn_failures = sum(
        1 for r in hard_negatives
        if r["error"] is not None or (r["validated_flags"] or [])
    )

    return {
        "overall": _with_rates(overall),
        "per_type": {t: _with_rates(c) for t, c in per_type.items()},
        "hard_negative_false_positive": _rate(hn_failures, len(hard_negatives)),
    }


def evidence_exists_rate(records: list[dict]) -> Rate:
    """Validated / (validated + dropped) flags, pooled across all records.
    Error records propose zero flags and are excluded -- see module docstring."""
    validated = sum(len(r["validated_flags"] or []) for r in records if r["error"] is None)
    dropped = sum(len(r["dropped_flags"] or []) for r in records if r["error"] is None)
    return _rate(validated, validated + dropped)


def contract_read_rate_coverage(records: list[dict]) -> Rate:
    k = sum(1 for r in records if r["error"] is None and "get_contract" in (r["coverage_tool_calls"] or []))
    return _rate(k, len(records))


def contract_read_rate_risk(records: list[dict]) -> Rate:
    """Informational only -- most cases have no reason for the risk agent to
    read the contract at all. See contract_read_when_needed for the metric
    that actually means something."""
    k = sum(1 for r in records if r["error"] is None and "get_contract" in (r["risk_tool_calls"] or []))
    return _rate(k, len(records))


# Mirrors tests/test_golden_consistency.py::CONTRACT_FLAG_CONTRADICTIONS -- the
# one place that already distinguishes "this flag genuinely contradicts a
# contract field" (vehicle_mismatch always is; this driver_mismatch sub-type,
# a claimed age/record, is) from the other flag cases (e.g. golden_21's
# "someone else drives" driver_mismatch, which isn't contract-based).
CONTRACT_BASED_FLAG_CASE_IDS = {"golden_13", "golden_33", "golden_54"}


def contract_read_when_needed(records: list[dict]) -> Rate:
    """Over cases whose expected flag is actually contract-based: did the risk
    agent call get_contract at all. The real version of contract_read_rate_risk."""
    eligible = [r for r in records if r["case_id"] in CONTRACT_BASED_FLAG_CASE_IDS]
    k = sum(
        1 for r in eligible
        if r["error"] is None and "get_contract" in (r["risk_tool_calls"] or [])
    )
    return _rate(k, len(eligible))


def bucket_match_rate(records: list[dict]) -> Rate:
    """stat_bucket (code) vs expected_risk_bucket (golden) -- a regression guard
    on the risk model/Poisson bucketing, independent of any agent."""
    k = sum(1 for r in records if r["error"] is None and r["stat_bucket"] == r["expected_risk_bucket"])
    return _rate(k, len(records))


def retrieval_hit_rate(records: list[dict]) -> Rate:
    """Denominator: cases with a non-empty expected clause set. k: at least one
    expected clause was among search_clauses' results."""
    eligible = [r for r in records if r["agent_clause_ids_expected"]]
    k = sum(
        1 for r in eligible
        if r["error"] is None
        and set(r["agent_clause_ids_expected"]) & set(r["retrieved_clause_ids"] or [])
    )
    return _rate(k, len(eligible))


def cited_clause_retrieved_rate(records: list[dict]) -> Rate:
    """Denominator: cases where the agent actually cited a clause. k: that
    clause was among the ones search_clauses returned (not invented from the
    system prompt or elsewhere)."""
    eligible = [r for r in records if r["coverage_clause_id"] is not None]
    k = sum(
        1 for r in eligible
        if r["error"] is None and r["coverage_clause_id"] in (r["retrieved_clause_ids"] or [])
    )
    return _rate(k, len(eligible))


def error_rate(records: list[dict]) -> Rate:
    return _rate(sum(1 for r in records if r["error"] is not None), len(records))


def _outcome(r: dict, field: str) -> str:
    return "ERROR" if r["error"] is not None else (r[field] or "null")


def decision_confusion_matrix(records: list[dict]) -> dict[tuple[str, str], int]:
    """Keys are (expected, actual) -- "ERROR" is a real column, not hidden."""
    return dict(Counter((r["expected_decision"], _outcome(r, "decision")) for r in records))


def coverage_confusion_matrix(records: list[dict]) -> dict[tuple[str, str], int]:
    return dict(Counter((r["expected_coverage"], _outcome(r, "coverage")) for r in records))


def wrongful_reject_rate(records: list[dict]) -> Rate:
    """Actual decision is reject while expected is anything else at all. Errors
    stay in the denominator (a different failure happened) but don't count as
    this specific one."""
    eligible = [r for r in records if r["expected_decision"] != "reject"]
    k = sum(1 for r in eligible if r["error"] is None and r["decision"] == "reject")
    return _rate(k, len(eligible))


def wrongful_approve_rate(records: list[dict]) -> Rate:
    """Actual decision is approve while expected is anything else at all."""
    eligible = [r for r in records if r["expected_decision"] != "approve"]
    k = sum(1 for r in eligible if r["error"] is None and r["decision"] == "approve")
    return _rate(k, len(eligible))


def _agreement(records: list[dict], key) -> Rate:
    by_case: dict[str, list] = {}
    for r in records:
        by_case.setdefault(r["case_id"], []).append(key(r))
    k = sum(1 for values in by_case.values() if len(set(values)) == 1)
    return _rate(k, len(by_case))


def per_case_agreement_decision(records: list[dict]) -> Rate:
    return _agreement(records, lambda r: _outcome(r, "decision"))


def per_case_agreement_full(records: list[dict]) -> Rate:
    def key(r: dict):
        if r["error"] is not None:
            return ("ERROR",)
        return (
            r["decision"],
            r["coverage"],
            r["coverage_clause_id"],
            frozenset(_flag_types_in(r["validated_flags"])),
        )

    return _agreement(records, key)


def compute_all(records: list[dict]) -> dict:
    return {
        "decision_match": decision_match(records),
        "coverage_match": coverage_match(records),
        "cited_clause_rate": cited_clause_rate(records),
        "missing_info_clause_null_rate": missing_info_clause_null_rate(records),
        "limitation_match": limitation_match(records),
        "flags": flag_metrics(records),
        "evidence_exists_rate": evidence_exists_rate(records),
        "contract_read_rate_coverage": contract_read_rate_coverage(records),
        "contract_read_when_needed": contract_read_when_needed(records),
        "contract_read_rate_risk": contract_read_rate_risk(records),
        "bucket_match_rate": bucket_match_rate(records),
        "retrieval_hit_rate": retrieval_hit_rate(records),
        "cited_clause_retrieved_rate": cited_clause_retrieved_rate(records),
        "error_rate": error_rate(records),
        "wrongful_reject_rate": wrongful_reject_rate(records),
        "wrongful_approve_rate": wrongful_approve_rate(records),
        "per_case_agreement_decision": per_case_agreement_decision(records),
        "per_case_agreement_full": per_case_agreement_full(records),
        "decision_confusion_matrix": decision_confusion_matrix(records),
        "coverage_confusion_matrix": coverage_confusion_matrix(records),
    }


def _fmt_rate(name: str, r: Rate) -> str:
    pct = f"{r['rate']:.0%}"
    return f"{name:<32}{pct:>6}  ({r['k']}/{r['n']})"


def print_score_table(records: list[dict]) -> None:
    results = compute_all(records)
    print(f"n = {len(records)} records")
    print()
    for name in [
        "decision_match", "coverage_match", "cited_clause_rate",
        "missing_info_clause_null_rate", "evidence_exists_rate",
        "contract_read_rate_coverage", "contract_read_when_needed", "contract_read_rate_risk",
        "bucket_match_rate", "retrieval_hit_rate", "cited_clause_retrieved_rate",
        "error_rate", "wrongful_reject_rate", "wrongful_approve_rate",
    ]:
        print(_fmt_rate(name, results[name]))

    num_runs = len({r["run_index"] for r in records})
    for name in ["per_case_agreement_decision", "per_case_agreement_full"]:
        if num_runs < 2:
            print(f"{name:<32}n/a (1 run)")
        else:
            print(_fmt_rate(name, results[name]))

    lim = results["limitation_match"]
    print(f"{'limitation_match':<32}TP={lim['tp']} FP={lim['fp']} FN={lim['fn']}  "
          f"precision={lim['precision']} recall={lim['recall']}")

    print("\nflags, overall:")
    o = results["flags"]["overall"]
    print(f"  TP={o['tp']} FP={o['fp']} FN={o['fn']}  precision={o['precision']} recall={o['recall']}")
    print("flags, per type:")
    for flag_type, c in results["flags"]["per_type"].items():
        print(f"  {flag_type:<22}TP={c['tp']} FP={c['fp']} FN={c['fn']}  "
              f"precision={c['precision']} recall={c['recall']}")
    print(_fmt_rate("hard_negative_false_positive", results["flags"]["hard_negative_false_positive"]))

    print("\ndecision confusion matrix (expected, actual) -> count:")
    for (expected, actual), count in sorted(results["decision_confusion_matrix"].items()):
        print(f"  {expected:<10} -> {actual:<10}: {count}")
    print("coverage confusion matrix (expected, actual) -> count:")
    for (expected, actual), count in sorted(results["coverage_confusion_matrix"].items()):
        print(f"  {expected:<14} -> {actual:<14}: {count}")
