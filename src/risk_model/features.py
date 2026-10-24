"""Leak-free preprocessing built on sklearn's ColumnTransformer."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import BINARY_COLS, NOMINAL_COLS, NUMERIC_COLS, PRIOR_GRADE_COLS


def build_preprocessor(use_prior_grades: bool = True) -> ColumnTransformer:
    """Return an unfitted preprocessor; it only learns from data passed to fit().

    Parameters
    ----------
    use_prior_grades:
        When False the G1/G2 columns are dropped, which gives an "early
        warning" model that does not rely on mid-term results.
    """
    numeric = [c for c in NUMERIC_COLS
               if use_prior_grades or c not in PRIOR_GRADE_COLS]

    return ColumnTransformer(
        transformers=[
            ("binary", OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1), BINARY_COLS),
            ("nominal", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), NOMINAL_COLS),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
