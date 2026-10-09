"""log_run against a temporary MLflow tracking store -- never touches the
project's real ./mlruns/, and makes no LLM or network call."""

import mlflow

from evals.mlflow_log import log_run

RECORDS = [
    {
        "case_id": "golden_01", "run_index": 1, "error": None,
        "decision": "approve", "expected_decision": "approve",
        "coverage": "covered", "expected_coverage": "covered",
        "coverage_clause_id": "6.1", "decision_clause_id": "6.1",
        "agent_clause_ids_expected": ["6.1"], "retrieved_clause_ids": ["6.1"],
        "limitation": None, "expected_limitation": None,
        "stat_bucket": "low", "expected_risk_bucket": "low",
        "validated_flags": [], "dropped_flags": [], "expected_flags": [],
        "coverage_tool_calls": ["get_contract"], "risk_tool_calls": ["get_contract"],
        "scenario": "",
    },
]


def test_log_run_writes_params_metrics_and_artifact(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    jsonl_path = tmp_path / "smoke-123.jsonl"
    jsonl_path.write_text("{}\n")

    log_run(
        eval_set="smoke", num_runs=1,
        coverage_prompt_version="v1", risk_prompt_version="v2",
        model="gemini-3.5-flash-lite", git_commit="abc123",
        records=RECORDS, jsonl_path=jsonl_path,
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("claims-triage-eval")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert run.data.params["git_commit"] == "abc123"
    assert run.data.params["coverage_prompt_version"] == "v1"
    assert run.data.metrics["decision_match_rate"] == 1.0
    assert run.data.metrics["decision_match_k"] == 1.0
    assert run.data.metrics["decision_match_n"] == 1.0

    artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
    assert "smoke-123.jsonl" in artifacts
