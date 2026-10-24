"""Data access: UCI download, local CSV, or an offline synthetic stand-in."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BINARY_COLS, NOMINAL_COLS, PASS_MARK, TARGET,
                     UCI_DATASET_ID)


def load_uci(dataset_id: int = UCI_DATASET_ID) -> pd.DataFrame:
    """Fetch the student-performance table from the UCI repository."""
    from ucimlrepo import fetch_ucirepo  # imported lazily: optional at test time

    ds = fetch_ucirepo(id=dataset_id)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def make_synthetic(n: int = 649, seed: int = 7) -> pd.DataFrame:
    """Generate a table with the same schema as the real dataset.

    Intended only for smoke tests, demos and offline development; the
    relationships are invented and carry no scientific meaning.
    """
    rng = np.random.default_rng(seed)

    def pick(options, p=None):
        return rng.choice(options, n, p=p)

    df = pd.DataFrame({
        "school": pick(["GP", "MS"], [0.65, 0.35]),
        "sex": pick(["F", "M"]),
        "age": rng.integers(15, 20, n),
        "address": pick(["U", "R"], [0.7, 0.3]),
        "famsize": pick(["GT3", "LE3"], [0.7, 0.3]),
        "Pstatus": pick(["T", "A"], [0.88, 0.12]),
        "Medu": rng.integers(0, 5, n),
        "Fedu": rng.integers(0, 5, n),
        "Mjob": pick(["teacher", "health", "services", "at_home", "other"]),
        "Fjob": pick(["teacher", "health", "services", "at_home", "other"]),
        "reason": pick(["home", "reputation", "course", "other"]),
        "guardian": pick(["mother", "father", "other"], [0.7, 0.22, 0.08]),
        "traveltime": rng.integers(1, 5, n),
        "studytime": rng.integers(1, 5, n),
        "failures": rng.choice([0, 1, 2, 3], n, p=[0.84, 0.1, 0.04, 0.02]),
        "schoolsup": pick(["yes", "no"], [0.12, 0.88]),
        "famsup": pick(["yes", "no"], [0.6, 0.4]),
        "paid": pick(["yes", "no"], [0.05, 0.95]),
        "activities": pick(["yes", "no"]),
        "nursery": pick(["yes", "no"], [0.8, 0.2]),
        "higher": pick(["yes", "no"], [0.9, 0.1]),
        "internet": pick(["yes", "no"], [0.77, 0.23]),
        "romantic": pick(["yes", "no"], [0.35, 0.65]),
        "famrel": rng.integers(1, 6, n),
        "freetime": rng.integers(1, 6, n),
        "goout": rng.integers(1, 6, n),
        "Dalc": rng.integers(1, 6, n),
        "Walc": rng.integers(1, 6, n),
        "health": rng.integers(1, 6, n),
        "absences": np.clip(rng.poisson(3.5, n), 0, 32),
    })

    ability = rng.normal(0, 1, n)
    base = (11.5 + 2.2 * ability + 0.7 * (df.studytime - 2.5)
            - 1.4 * df.failures + 1.2 * (df.higher == "yes")
            - 0.05 * df.absences)
    df["G1"] = np.clip(np.rint(base + rng.normal(0, 1.2, n)), 0, 20).astype(int)
    df["G2"] = np.clip(np.rint(base + rng.normal(0, 1.0, n)), 0, 20).astype(int)
    df[TARGET] = np.clip(
        np.rint(0.55 * df.G2 + 0.35 * df.G1 + 0.1 * base + rng.normal(0, 0.9, n)),
        0, 20).astype(int)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    required = set(BINARY_COLS) | set(NOMINAL_COLS) | {TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")


def to_features_and_label(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw table into feature matrix and binary pass/fail label."""
    validate_schema(df)
    y = (df[TARGET] >= PASS_MARK).astype(int).rename("passed")
    return df.drop(columns=[TARGET]), y
