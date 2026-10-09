"""log_run against a temporary MLflow tracking store -- never touches the
project's real ./mlruns/ or mlflow.db, and makes no LLM or network call."""

import mlflow

from evals.mlflow_log import log_run

RECORDS = [
    {
        "run_id": "smoke-123-run1", "case_id": "golden_01", "run_index": 1, "error": None,
        "decision": "approve", "expected_decision": "approve",
        "coverage": "covered", "expected_coverage": "covered",
        "coverage_clause_id": "6.1", "decision_clause_id": "6.1",
        "agent_clause_ids_expected": ["6.1"], "retrieved_clause_ids": ["6.1"],
        "limitation": None, "expected_limitation": None,
        "stat_bucket": "low", "expected_risk_bucket": "low",
        "validated_flags": [], "dropped_flags": [], "expected_flags": [],
        "coverage_tool_calls": ["get_contract"], "risk_tool_calls": ["get_contract"],
        "scenario": "", "latency_seconds": 12.5,
        "input_tokens": 5000, "output_tokens": 300,
    },
]


def test_log_run_writes_params_metrics_and_artifact(tmp_path):
    # Overrides evals.mlflow_log's module-level sqlite URI (the real project's
    # mlflow.db) with an isolated temp one -- this test must never touch the
    # real tracking store.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    jsonl_path = tmp_path / "smoke-123.jsonl"
    jsonl_path.write_text("{}\n")

    log_run(
        eval_set="smoke", timestamp=123, num_runs=1,
        coverage_prompt_version="v1", risk_prompt_version="v2",
        model="gemini-3.5-flash-lite", git_commit="abc123",
        dataset_sha256="d" * 64, num_cases=1,
        langfuse_session_ids=["smoke-123-run1"],
        records=RECORDS, jsonl_path=jsonl_path,
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("claims-triage-eval")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert run.info.run_name == "smoke-123-abc123"
    assert run.data.tags["eval_set"] == "smoke"
    assert run.data.tags["stage"] == "stage_a"

    assert run.data.params["git_commit"] == "abc123"
    assert run.data.params["coverage_prompt_version"] == "v1"
    assert run.data.params["dataset_sha256"] == "d" * 12
    assert run.data.params["num_cases"] == "1"
    assert run.data.params["langfuse_session_ids"] == "smoke-123-run1"

    assert run.data.metrics["decision_match_rate"] == 1.0
    assert run.data.metrics["decision_match_k"] == 1.0
    assert run.data.metrics["decision_match_n"] == 1.0
    # The double-"_rate" bug: cited_clause_rate must log as cited_clause_rate/_k/_n,
    # never cited_clause_rate_rate.
    assert "cited_clause_rate" in run.data.metrics
    assert "cited_clause_rate_rate" not in run.data.metrics

    assert run.data.metrics["latency_mean_seconds"] == 12.5
    assert run.data.metrics["mean_input_tokens"] == 5000

    artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
    assert "smoke-123.jsonl" in artifacts
