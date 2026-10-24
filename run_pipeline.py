#!/usr/bin/env python3
"""End-to-end training pipeline.

Examples
--------
    python run_pipeline.py                       # real data from UCI
    python run_pipeline.py --source synthetic    # offline demo data
    python run_pipeline.py --no-prior-grades     # early-warning variant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_model import data, evaluate, models, plots  # noqa: E402
from risk_model.config import (IMAGES_DIR, NUMERIC_COLS, REPORTS_DIR,  # noqa: E402
                               SearchConfig)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", choices=["uci", "csv", "synthetic"], default="uci")
    p.add_argument("--csv", type=Path, help="path used when --source csv")
    p.add_argument("--no-prior-grades", action="store_true",
                   help="drop G1/G2 to simulate an early-warning setting")
    p.add_argument("--n-iter", type=int, default=SearchConfig().n_iter,
                   help="randomised-search iterations")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return p.parse_args(argv)


def load_table(args):
    if args.source == "synthetic":
        return data.make_synthetic()
    if args.source == "csv":
        if not args.csv:
            sys.exit("--csv is required when --source csv")
        return data.load_csv(args.csv)
    return data.load_uci()


def main(argv=None) -> dict:
    args = parse_args(argv)
    use_prior = not args.no_prior_grades

    table = load_table(args)
    X, y = data.to_features_and_label(table)
    print(f"Loaded {len(X)} rows, {X.shape[1]} features, "
          f"pass rate {y.mean():.1%}")

    X_tr, X_va, X_te, y_tr, y_va, y_te = models.three_way_split(X, y)
    print(f"Split sizes  train={len(X_tr)}  valid={len(X_va)}  test={len(X_te)}")

    # 1. Baseline comparison on the validation set -------------------------
    results = {}
    for name, est in models.candidate_estimators().items():
        pipe = models.make_pipeline(est, use_prior).fit(X_tr, y_tr)
        results[name] = evaluate.score(pipe, X_va, y_va)
    board = evaluate.leaderboard(results)
    print("\nValidation leaderboard\n", board.to_string(), sep="")

    # 2. Tune the forest, then touch the test set exactly once --------------
    cfg = SearchConfig(n_iter=args.n_iter)
    search = models.tune_random_forest(X_tr, y_tr, use_prior, cfg)
    best = search.best_estimator_
    print(f"\nBest CV {cfg.scoring}: {search.best_score_:.3f}")
    print("Best params:", search.best_params_)

    valid_scores = evaluate.score(best, X_va, y_va)
    test_scores = evaluate.score(best, X_te, y_te)
    print("Tuned forest  valid:", {k: round(v, 3) for k, v in valid_scores.items()})
    print("Tuned forest  test :", {k: round(v, 3) for k, v in test_scores.items()})

    importance = evaluate.top_features(best)

    # 3. Persist artefacts ---------------------------------------------------
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": args.source,
        "uses_prior_grades": use_prior,
        "rows": int(len(X)),
        "leaderboard_validation": board.to_dict(orient="index"),
        "best_params": {k: (v if v is None else
                            v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "tuned_validation": valid_scores,
        "tuned_test": test_scores,
        "top_features": importance.round(4).to_dict(),
    }
    (args.reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(best, args.reports_dir / "model.joblib")

    if not args.no_plots:
        img = args.images_dir
        plots.class_balance(y, img / "class_balance.png")
        plots.numeric_overview(
            X, y, [c for c in ["G1", "G2", "studytime", "absences"] if c in NUMERIC_COLS],
            img / "numeric_overview.png")
        plots.correlation(X, ["age", "studytime", "failures", "absences", "G1", "G2"],
                          img / "correlation.png")
        plots.leaderboard_chart(board, img / "leaderboard.png")
        plots.importance_chart(importance, img / "feature_importance.png")
        plots.confusion_and_roc(best, X_te, y_te, img / "test_diagnostics.png")
        print(f"\nFigures written to {img}")

    return summary


if __name__ == "__main__":
    main()
