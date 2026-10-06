"""Train the risk model: expected claims/year via a Poisson HGB, vs a predict-the-mean baseline."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split

import config

CATEGORICAL = ["area", "veh_brand", "veh_gas", "region"]
FEATURES = CATEGORICAL + [
    "veh_power",
    "veh_age",
    "driv_age",
    "bonus_malus",
    "density",
]


def _load_data() -> pd.DataFrame:
    conn = sqlite3.connect(config.CONTRACTS_DB)
    df = pd.read_sql("SELECT * FROM contracts", conn)
    conn.close()
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def train() -> None:
    """Fit the model, compare it to a predict-the-mean baseline, save it only if it wins."""
    df = _load_data()
    target = df["claim_nb"] / df["exposure"]

    train_df, test_df, train_y, test_y = train_test_split(
        df, target, test_size=config.HOLDOUT_FRACTION, random_state=config.RANDOM_SEED
    )

    model = HistGradientBoostingRegressor(loss="poisson", random_state=config.RANDOM_SEED)
    model.fit(train_df[FEATURES], train_y, sample_weight=train_df["exposure"])
    pred = model.predict(test_df[FEATURES])

    baseline_rate = train_df["claim_nb"].sum() / train_df["exposure"].sum()
    baseline_pred = np.full(len(test_df), baseline_rate)

    weight = test_df["exposure"].to_numpy()
    model_dev = mean_poisson_deviance(test_y, pred, sample_weight=weight)
    baseline_dev = mean_poisson_deviance(test_y, baseline_pred, sample_weight=weight)

    print(f"Model Poisson deviance:    {model_dev:.4f}")
    print(f"Baseline Poisson deviance: {baseline_dev:.4f}")

    if model_dev >= baseline_dev:
        raise RuntimeError("Model did not beat the predict-the-mean baseline; not saving.")

    config.RISK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.RISK_MODEL_PATH)
    print(f"Saved {config.RISK_MODEL_PATH}")


if __name__ == "__main__":
    train()
