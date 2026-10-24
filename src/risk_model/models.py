"""Candidate estimators, data splitting and hyper-parameter search."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import SearchConfig, SplitConfig
from .features import build_preprocessor


def three_way_split(X, y, cfg: SplitConfig = SplitConfig()):
    """Stratified train / validation / test split."""
    holdout = cfg.valid + cfg.test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=holdout, stratify=y, random_state=cfg.seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=cfg.test / holdout, stratify=y_tmp,
        random_state=cfg.seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def candidate_estimators(seed: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }


def make_pipeline(estimator, use_prior_grades: bool = True) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(use_prior_grades)),
        ("model", estimator),
    ])


def tune_random_forest(X_train, y_train, use_prior_grades: bool = True,
                       cfg: SearchConfig = SearchConfig()) -> RandomizedSearchCV:
    """Randomised search over the forest, with preprocessing inside each fold."""
    pipe = make_pipeline(RandomForestClassifier(random_state=cfg.seed),
                         use_prior_grades)
    search = RandomizedSearchCV(
        pipe, cfg.space, n_iter=cfg.n_iter, cv=cfg.cv_folds,
        scoring=cfg.scoring, n_jobs=-1, random_state=cfg.seed, refit=True)
    return search.fit(X_train, y_train)
