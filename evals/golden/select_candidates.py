"""Authoring tool: list holdout contracts with their risk bucket, for picking
real contract_ids when writing golden/production cases. Not used at runtime.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
from scipy.stats import poisson
from sklearn.model_selection import train_test_split

import config
from data.load_contracts import load_contracts


def load_holdout_with_risk():
    """Holdout contracts with expected_claims, p (Poisson tail prob) and bucket."""
    df = load_contracts()
    target = df["claim_nb"] / df["exposure"]
    _, test_df, _, _ = train_test_split(
        df, target, test_size=config.HOLDOUT_FRACTION, random_state=config.RANDOM_SEED
    )
    test_df = test_df.copy()

    model = joblib.load(config.RISK_MODEL_PATH)
    pred_rate = model.predict(test_df[config.FEATURES])
    test_df["expected_claims"] = pred_rate * test_df["exposure"]
    test_df["p"] = poisson.sf(test_df["claim_nb"] - 1, test_df["expected_claims"])

    def bucket(p: float) -> str:
        if p >= config.RISK_P_LOW_MIN:
            return "low"
        if p >= config.RISK_P_MEDIUM_MIN:
            return "medium"
        return "high"

    test_df["risk_bucket"] = test_df["p"].apply(bucket)
    return test_df


if __name__ == "__main__":
    df = load_holdout_with_risk()
    print(df["risk_bucket"].value_counts())
