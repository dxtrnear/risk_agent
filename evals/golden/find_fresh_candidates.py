"""Authoring aid: list holdout contracts NOT already used in golden.json or
production.json, by formule/risk profile, for picking contract_ids for new cases.
Reads production.json's contract_id/case_id fields only (not claim content).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

import config
from evals.golden.select_candidates import load_holdout_with_risk

GOLDEN_PATH = config.ROOT / "evals" / "golden" / "golden.json"
PRODUCTION_PATH = config.ROOT / "evals" / "golden" / "production.json"


def used_contract_ids() -> set[int]:
    golden = json.loads(GOLDEN_PATH.read_text())
    production = json.loads(PRODUCTION_PATH.read_text())
    return {c["contract_id"] for c in golden} | {c["contract_id"] for c in production}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    used = used_contract_ids()
    df = load_holdout_with_risk()
    df = df[~df.contract_id.isin(used)]
    cols = ["contract_id", "formule", "franchise", "veh_age", "driv_age", "risk_bucket"]

    print("--- low/medium risk, tous_risques, fresh ---")
    print(df[(df.risk_bucket.isin(["low", "medium"])) & (df.formule == "tous_risques")][cols].head(10))

    print()
    print("--- low/medium risk, tiers_etendu, fresh ---")
    print(df[(df.risk_bucket.isin(["low", "medium"])) & (df.formule == "tiers_etendu")][cols].head(6))

    print()
    print("--- HIGH risk, any formule, fresh ---")
    print(df[df.risk_bucket == "high"][cols].head(10))

    print()
    print("--- HIGH risk, formule=tiers specifically, fresh ---")
    print(df[(df.risk_bucket == "high") & (df.formule == "tiers")][cols].head(6))
