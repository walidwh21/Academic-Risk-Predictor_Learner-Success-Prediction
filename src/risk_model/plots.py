"""Figure generation. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402

PALETTE = {"fail": "#d1495b", "pass": "#2e86ab"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def class_balance(y, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    counts = y.map({0: "fail", 1: "pass"}).value_counts()
    ax.bar(counts.index, counts.values,
           color=[PALETTE[k] for k in counts.index])
    ax.set_title("Class balance")
    ax.set_ylabel("Students")
    return _save(fig, path)


def numeric_overview(X, y, cols, path: Path) -> Path:
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3.4))
    label = y.map({0: "fail", 1: "pass"})
    for ax, col in zip(axes, cols):
        sns.boxplot(x=label, y=X[col], hue=label, palette=PALETTE,
                    legend=False, ax=ax, order=["fail", "pass"])
        ax.set_title(col)
        ax.set_xlabel("")
    return _save(fig, path)


def correlation(X, cols, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(X[cols].corr(), annot=True, fmt=".2f", cmap="vlag",
                center=0, ax=ax)
    ax.set_title("Correlation of numeric features")
    return _save(fig, path)


def leaderboard_chart(table, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    table[["accuracy", "f1", "roc_auc"]].plot.barh(ax=ax, width=0.75)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title("Validation metrics by model")
    ax.legend(loc="lower right")
    return _save(fig, path)


def importance_chart(imp, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(x=imp.values, y=imp.index, color="#2e86ab", ax=ax)
    ax.set_title("Top feature importances")
    ax.set_xlabel("Importance")
    return _save(fig, path)


def confusion_and_roc(pipeline, X, y, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X, y, display_labels=["fail", "pass"], cmap="Blues",
        colorbar=False, ax=axes[0])
    axes[0].set_title("Confusion matrix (test)")
    RocCurveDisplay.from_estimator(pipeline, X, y, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set_title("ROC curve (test)")
    return _save(fig, path)
