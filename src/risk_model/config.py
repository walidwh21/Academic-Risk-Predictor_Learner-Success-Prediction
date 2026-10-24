"""Central configuration: column groups, split ratios and search space."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "docs" / "images"

UCI_DATASET_ID = 320
TARGET = "G3"
PASS_MARK = 10  # final grade >= PASS_MARK  ->  label 1 (pass)

# Column groups -------------------------------------------------------------
BINARY_COLS = [
    "sex", "address", "Pstatus", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic",
]
NOMINAL_COLS = ["school", "famsize", "Mjob", "Fjob", "reason", "guardian"]
NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2",
]
# Prior-period grades: strongest signal, but unavailable in an "early" setting.
PRIOR_GRADE_COLS = ["G1", "G2"]


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.60
    valid: float = 0.20
    test: float = 0.20
    seed: int = 42


@dataclass(frozen=True)
class SearchConfig:
    n_iter: int = 25
    cv_folds: int = 5
    scoring: str = "f1"
    seed: int = 42
    space: dict = field(default_factory=lambda: {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 6, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    })
