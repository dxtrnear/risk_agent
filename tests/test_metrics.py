"""evals/metrics.py against synthetic records -- no LLM, no real eval run."""

import evals.metrics as metrics


def _record(**overrides) -> dict:
    base = {
        "run_id": "r1",
        "run_index": 1,
        "case_id": "golden_01",
        "scenario": "",
        "error": None,
        "decision": "approve",
        "expected_decision": "approve",
        "coverage": "covered",
        "expected_coverage": "covered",
        "coverage_clause_id": "6.1",
        "decision_clause_id": "6.1",
        "agent_clause_ids_expected": ["6.1"],
        "retrieved_clause_ids": ["6.1"],
        "limitation": None,
        "expected_limitation": None,
        "stat_bucket": "low",
        "expected_risk_bucket": "low",
        "validated_flags": [],
        "dropped_flags": [],
        "expected_flags": [],
        "coverage_tool_calls": ["get_contract", "search_clauses"],
        "risk_tool_calls": ["get_contract"],
    }
    base.update(overrides)
    return base


def test_decision_match_counts_correctly():
    records = [_record(), _record(decision="reject")]
    result = metrics.decision_match(records)
    assert result == {"rate": 0.5, "k": 1, "n": 2}


def test_error_record_is_always_a_decision_failure_but_stays_in_denominator():
    records = [_record(), _record(error="boom", decision=None)]
    result = metrics.decision_match(records)
    assert result == {"rate": 0.5, "k": 1, "n": 2}


def test_cited_clause_rate_excludes_missing_info_cases_from_denominator():
    records = [
        _record(agent_clause_ids_expected=["6.1"], coverage_clause_id="6.1"),
        _record(agent_clause_ids_expected=[], coverage_clause_id=None),  # missing_info case
    ]
    result = metrics.cited_clause_rate(records)
    assert result["n"] == 1
    assert result["k"] == 1


def test_missing_info_clause_null_rate():
    records = [
        _record(expected_coverage="missing_info", coverage_clause_id=None),
        _record(expected_coverage="missing_info", coverage_clause_id="6.1"),  # hallucinated
    ]
    result = metrics.missing_info_clause_null_rate(records)
    assert result == {"rate": 0.5, "k": 1, "n": 2}


def test_limitation_match_tp_fp_fn():
    records = [
        _record(expected_limitation="keys_in_vehicle", limitation="keys_in_vehicle"),  # TP
        _record(expected_limitation="keys_in_vehicle", limitation=None),  # FN
        _record(expected_limitation=None, limitation="keys_in_vehicle"),  # FP
        _record(expected_limitation=None, limitation=None),  # true negative, ignored
    ]
    result = metrics.limitation_match(records)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1


def test_flag_metrics_per_type_and_overall():
    records = [
        _record(
            validated_flags=[{"flag": "payment_pressure", "evidence": "e"}],
            expected_flags=[{"flag": "payment_pressure", "evidence": "e"}],
        ),  # TP
        _record(
            validated_flags=[{"flag": "vehicle_mismatch", "evidence": "e"}],
            expected_flags=[],
        ),  # FP
        _record(
            validated_flags=[],
            expected_flags=[{"flag": "driver_mismatch", "evidence": "e"}],
        ),  # FN
    ]
    result = metrics.flag_metrics(records)
    assert result["overall"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5}
    assert result["per_type"]["payment_pressure"]["tp"] == 1
    assert result["per_type"]["vehicle_mismatch"]["fp"] == 1
    assert result["per_type"]["driver_mismatch"]["fn"] == 1


def test_hard_negative_false_positive_detected():
    records = [
        _record(scenario="hard negative: payment_pressure", validated_flags=[]),  # correct
        _record(
            scenario="hard negative: vehicle_mismatch",
            validated_flags=[{"flag": "vehicle_mismatch", "evidence": "e"}],
        ),  # false positive
        _record(scenario="flag-only escalate: driver_mismatch", validated_flags=[]),  # not a hard negative
    ]
    result = metrics.flag_metrics(records)
    assert result["hard_negative_false_positive"] == {"rate": 0.5, "k": 1, "n": 2}


def test_evidence_exists_rate_excludes_errors():
    records = [
        _record(validated_flags=[{"flag": "payment_pressure", "evidence": "e"}], dropped_flags=[]),
        _record(error="boom", validated_flags=None, dropped_flags=None),
    ]
    result = metrics.evidence_exists_rate(records)
    assert result == {"rate": 1.0, "k": 1, "n": 1}  # error record contributes to neither side


def test_contract_read_when_needed_only_considers_contract_based_cases():
    records = [
        _record(case_id="golden_13", risk_tool_calls=["get_contract"]),  # contract-based, read
        _record(case_id="golden_33", risk_tool_calls=[]),  # contract-based, NOT read
        _record(case_id="golden_21", risk_tool_calls=[]),  # not contract-based, excluded
    ]
    result = metrics.contract_read_when_needed(records)
    assert result == {"rate": 0.5, "k": 1, "n": 2}


def test_contract_read_rate_coverage():
    records = [
        _record(coverage_tool_calls=["get_contract", "search_clauses"]),
        _record(coverage_tool_calls=["search_clauses"]),
    ]
    result = metrics.contract_read_rate_coverage(records)
    assert result == {"rate": 0.5, "k": 1, "n": 2}


def test_retrieval_hit_rate_and_cited_clause_retrieved():
    records = [
        _record(agent_clause_ids_expected=["6.1"], retrieved_clause_ids=["6.1", "6.2"], coverage_clause_id="6.1"),
        _record(agent_clause_ids_expected=["6.5"], retrieved_clause_ids=["6.1"], coverage_clause_id="6.1"),
    ]
    hit = metrics.retrieval_hit_rate(records)
    assert hit == {"rate": 0.5, "k": 1, "n": 2}
    cited = metrics.cited_clause_retrieved_rate(records)
    # both records cite 6.1, and 6.1 is retrieved in both
    assert cited == {"rate": 1.0, "k": 2, "n": 2}


def test_wrongful_reject_and_approve_rates():
    records = [
        _record(expected_decision="approve", decision="reject"),  # wrongful reject
        _record(expected_decision="escalate", decision="reject"),  # wrongful reject
        _record(expected_decision="reject", decision="approve"),  # wrongful approve
        _record(expected_decision="ask_info", decision="ask_info"),  # neither
    ]
    # wrongful_reject denominator: expected != reject -> records 1,2,4 (3); k=2 (1,2)
    assert metrics.wrongful_reject_rate(records) == {"rate": 2 / 3, "k": 2, "n": 3}
    # wrongful_approve denominator: expected != approve -> records 2,3,4 (3); k=1 (3)
    assert metrics.wrongful_approve_rate(records) == {"rate": 1 / 3, "k": 1, "n": 3}


def test_per_case_agreement_decision():
    records = [
        _record(case_id="golden_01", run_index=1, decision="approve"),
        _record(case_id="golden_01", run_index=2, decision="approve"),
        _record(case_id="golden_02", run_index=1, decision="reject"),
        _record(case_id="golden_02", run_index=2, decision="escalate"),
    ]
    result = metrics.per_case_agreement_decision(records)
    assert result == {"rate": 0.5, "k": 1, "n": 2}


def test_per_case_agreement_full_is_stricter_than_decision_agreement():
    records = [
        _record(case_id="golden_01", run_index=1, decision="approve", coverage_clause_id="6.1"),
        _record(case_id="golden_01", run_index=2, decision="approve", coverage_clause_id="6.5"),
    ]
    assert metrics.per_case_agreement_decision(records) == {"rate": 1.0, "k": 1, "n": 1}
    assert metrics.per_case_agreement_full(records) == {"rate": 0.0, "k": 0, "n": 1}


def test_error_rate():
    records = [_record(), _record(error="boom")]
    assert metrics.error_rate(records) == {"rate": 0.5, "k": 1, "n": 2}


def test_decision_confusion_matrix_includes_error_as_outcome():
    records = [
        _record(expected_decision="approve", decision="approve"),
        _record(expected_decision="approve", error="boom", decision=None),
    ]
    matrix = metrics.decision_confusion_matrix(records)
    assert matrix[("approve", "approve")] == 1
    assert matrix[("approve", "ERROR")] == 1
