"""MLflow experiment tracking for eval runs. One MLflow run per run_eval.py
invocation. Params are the run's configuration (prompt versions, model, git
commit, eval set, repetition count, dataset hash/size, Langfuse session ids);
metrics are every score's rate/k/n (and TP/FP/FN for limitation and flags,
never precision/recall directly -- those are None for a flag type with zero
occurrences, and MLflow metrics must be floats); the JSONL file itself is
logged as an artifact, so any run's exact raw records are recoverable later,
not just its summary numbers.
"""

import mlflow

import config
from evals.metrics import compute_all

EXPERIMENT_NAME = "claims-triage-eval"

# The installed MLflow version put the plain filesystem backend ("./mlruns")
# into maintenance mode and refuses to use it at all, raising instead ("Please
# migrate to a database backend"). A local sqlite file is the recommended
# replacement and still requires no server -- matches the already-gitignored
# mlflow.db this project anticipated from Phase 0 but never actually wired up.
mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")


def _log_rate(name: str, rate: dict) -> None:
    # Several metric names (e.g. "cited_clause_rate") already end in "_rate";
    # appending "_rate" again without stripping first would log
    # "cited_clause_rate_rate". Strip once, then always append uniformly.
    base = name[:-len("_rate")] if name.endswith("_rate") else name
    mlflow.log_metric(f"{base}_rate", rate["rate"])
    mlflow.log_metric(f"{base}_k", rate["k"])
    mlflow.log_metric(f"{base}_n", rate["n"])


def log_run(
    *,
    eval_set: str,
    timestamp: int,
    num_runs: int,
    coverage_prompt_version: str,
    risk_prompt_version: str,
    model: str,
    git_commit: str,
    dataset_sha256: str,
    num_cases: int,
    langfuse_session_ids: list[str],
    records: list[dict],
    jsonl_path,
) -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"{eval_set}-{timestamp}-{git_commit}"
    with mlflow.start_run(run_name=run_name, tags={"eval_set": eval_set, "stage": config.PROJECT_STAGE}):
        mlflow.log_param("eval_set", eval_set)
        mlflow.log_param("num_runs", num_runs)
        mlflow.log_param("coverage_prompt_version", coverage_prompt_version)
        mlflow.log_param("risk_prompt_version", risk_prompt_version)
        mlflow.log_param("model", model)
        mlflow.log_param("git_commit", git_commit)
        # Two runs against a different golden.json must be visibly, immediately
        # not comparable -- the hash changes the moment the case file does.
        mlflow.log_param("dataset_sha256", dataset_sha256[:12])
        mlflow.log_param("num_cases", num_cases)
        # Lets you go from an MLflow run straight to its Langfuse traces.
        mlflow.log_param("langfuse_session_ids", ",".join(langfuse_session_ids))

        results = compute_all(records)
        for name, value in results.items():
            if name in ("decision_confusion_matrix", "coverage_confusion_matrix", "latency", "tokens_cost"):
                continue  # handled separately below; not Rate-shaped
            if name == "limitation_match":
                mlflow.log_metric("limitation_tp", value["tp"])
                mlflow.log_metric("limitation_fp", value["fp"])
                mlflow.log_metric("limitation_fn", value["fn"])
            elif name == "flags":
                overall = value["overall"]
                mlflow.log_metric("flags_overall_tp", overall["tp"])
                mlflow.log_metric("flags_overall_fp", overall["fp"])
                mlflow.log_metric("flags_overall_fn", overall["fn"])
                for flag_type, counts in value["per_type"].items():
                    mlflow.log_metric(f"flags_{flag_type}_tp", counts["tp"])
                    mlflow.log_metric(f"flags_{flag_type}_fp", counts["fp"])
                    mlflow.log_metric(f"flags_{flag_type}_fn", counts["fn"])
                _log_rate("hard_negative_false_positive", value["hard_negative_false_positive"])
            else:
                _log_rate(name, value)

        latency = results["latency"]
        if latency["mean"] is not None:
            mlflow.log_metric("latency_mean_seconds", latency["mean"])
            mlflow.log_metric("latency_p95_seconds", latency["p95"])

        tokens_cost = results["tokens_cost"]
        if tokens_cost["mean_input_tokens"] is not None:
            mlflow.log_metric("mean_input_tokens", tokens_cost["mean_input_tokens"])
            mlflow.log_metric("mean_output_tokens", tokens_cost["mean_output_tokens"])
            mlflow.log_metric("mean_cost_usd", tokens_cost["mean_cost_usd"])

        mlflow.log_artifact(str(jsonl_path))
