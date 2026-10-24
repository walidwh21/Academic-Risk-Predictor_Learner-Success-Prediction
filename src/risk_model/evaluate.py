"""Metric computation and result tables."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def score(pipeline, X, y) -> dict:
    """Metrics; ROC AUC uses predicted probabilities rather than hard labels."""
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def leaderboard(results: dict) -> pd.DataFrame:
    table = pd.DataFrame(results).T.round(3)
    return table.sort_values("f1", ascending=False)


def top_features(pipeline, n: int = 10) -> pd.Series:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipeline.named_steps["model"].feature_importances_,
                    index=names)
    return imp.sort_values(ascending=False).head(n)
