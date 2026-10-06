"""Add synthetic `formule` and `franchise` to the raw contracts, seeded for reproducibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config

FORMULES = ["tiers", "tiers_etendu", "tous_risques"]

RENAME = {
    "IDpol": "contract_id",
    "ClaimNb": "claim_nb",
    "Exposure": "exposure",
    "Area": "area",
    "VehPower": "veh_power",
    "VehAge": "veh_age",
    "DrivAge": "driv_age",
    "BonusMalus": "bonus_malus",
    "VehBrand": "veh_brand",
    "VehGas": "veh_gas",
    "Density": "density",
    "Region": "region",
}


def _assign_formule(veh_age: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Newer cars skew toward richer coverage; older cars skew toward liability-only."""
    probs = np.empty((len(veh_age), 3))
    probs[veh_age <= 3] = [0.15, 0.35, 0.50]
    probs[(veh_age > 3) & (veh_age < 10)] = [0.35, 0.40, 0.25]
    probs[veh_age >= 10] = [0.60, 0.30, 0.10]

    cum_probs = probs.cumsum(axis=1)
    draw = rng.random((len(veh_age), 1))
    idx = (draw > cum_probs).sum(axis=1)
    return np.array(FORMULES)[idx]


def _assign_franchise(formule: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """No own-damage cover under `tiers` -> no franchise; richer cover -> higher franchise."""
    franchise = np.zeros(len(formule))
    etendu = formule == "tiers_etendu"
    tous = formule == "tous_risques"
    franchise[etendu] = rng.integers(150, 300, etendu.sum())
    franchise[tous] = rng.integers(300, 800, tous.sum())
    return franchise


def enrich(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Clean raw OpenML columns and add synthetic `formule`/`franchise`."""
    df = df.rename(columns=RENAME).copy()
    df["veh_gas"] = df["veh_gas"].str.strip("'")
    df["contract_id"] = df["contract_id"].astype(int)

    rng = np.random.default_rng(seed)
    df["formule"] = _assign_formule(df["veh_age"], rng)
    df["franchise"] = _assign_franchise(df["formule"].to_numpy(), rng)
    return df


if __name__ == "__main__":
    raw_path = config.RAW_DATA_DIR / "contracts_raw.csv"
    raw = pd.read_csv(raw_path)
    enriched = enrich(raw, config.RANDOM_SEED)
    enriched.to_csv(config.CONTRACTS_CSV, index=False)
    print(f"Saved {config.CONTRACTS_CSV} ({len(enriched)} rows)")
