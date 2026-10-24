import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_model import data, evaluate, models  # noqa: E402


def _xy(n=300, seed=1):
    return data.to_features_and_label(data.make_synthetic(n, seed=seed))


class PipelineTests(unittest.TestCase):
    def test_label_matches_threshold(self):
        df = data.make_synthetic(200, seed=2)
        _, y = data.to_features_and_label(df)
        self.assertLessEqual(set(y.unique()), {0, 1})
        self.assertTrue(((df["G3"] >= 10).astype(int).values == y.values).all())

    def test_split_is_stratified_and_disjoint(self):
        X, y = _xy()
        X_tr, X_va, X_te, y_tr, y_va, y_te = models.three_way_split(X, y)
        self.assertEqual(len(X_tr) + len(X_va) + len(X_te), len(X))
        self.assertFalse(set(X_tr.index) & set(X_te.index))
        self.assertLess(abs(y_tr.mean() - y_te.mean()), 0.05)

    def test_unseen_category_does_not_crash(self):
        X, y = _xy()
        X_tr, X_va, _, y_tr, _, _ = models.three_way_split(X, y)
        pipe = models.make_pipeline(
            models.candidate_estimators()["Logistic Regression"]).fit(X_tr, y_tr)
        odd = X_va.copy()
        odd["Mjob"] = "astronaut"
        self.assertEqual(len(pipe.predict(odd)), len(odd))

    def test_early_warning_drops_prior_grades(self):
        X, y = _xy()
        pipe = models.make_pipeline(
            models.candidate_estimators()["Decision Tree"],
            use_prior_grades=False).fit(X, y)
        names = list(pipe.named_steps["prep"].get_feature_names_out())
        self.assertNotIn("G1", names)
        self.assertNotIn("G2", names)

    def test_scores_in_unit_interval(self):
        X, y = _xy()
        X_tr, X_va, _, y_tr, y_va, _ = models.three_way_split(X, y)
        pipe = models.make_pipeline(
            models.candidate_estimators()["Random Forest"]).fit(X_tr, y_tr)
        s = evaluate.score(pipe, X_va, y_va)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in s.values()))


if __name__ == "__main__":
    unittest.main()
