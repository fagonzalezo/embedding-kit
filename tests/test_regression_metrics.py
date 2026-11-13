"""Phase 6: regression metrics with per-point R^2 terms."""

import numpy as np
import pytest


class TestRegressionMetrics:
    """Phase 6: per-point R^2 terms for paired comparisons of regression transforms."""

    @staticmethod
    def _data(seed=0):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((300, 5))
        y = Z[:, 0] + 0.1 * rng.standard_normal(300)
        return Z[:200], y[:200], Z[200:], y[200:]

    def test_knn_r2_matches_sklearn(self):
        from sklearn.metrics import r2_score
        from sklearn.neighbors import KNeighborsRegressor

        from embedkit.evaluation.metrics import knn_regression_r2

        Ztr, ytr, Zte, yte = self._data()
        value, terms = knn_regression_r2(Ztr, ytr, Zte, yte, k=7, metric="euclidean",
                                         return_per_point=True)
        pred = KNeighborsRegressor(7).fit(Ztr, ytr).predict(Zte)
        assert value == pytest.approx(r2_score(yte, pred))
        assert terms.mean() == pytest.approx(value) and terms.shape == (100,)

    def test_ridge_probe_terms_average_to_r2(self):
        from embedkit.evaluation.metrics import linear_probe

        Ztr, ytr, Zte, yte = self._data(1)
        value, terms = linear_probe(Ztr, ytr, Zte, yte, task="regression", return_per_point=True)
        assert terms.mean() == pytest.approx(value) and value > 0.9

    def test_protocol_picks_regression_metrics(self):
        from embedkit.baselines import Identity
        from embedkit.evaluation import evaluate_transform

        Ztr, ytr, Zte, yte = self._data(2)
        df = evaluate_transform(lambda s: Identity(), Ztr, Zte, ytr, yte, seeds=[0])
        assert {"knn_r2", "linear_probe"} <= set(df.metric)
        assert "knn_accuracy" not in set(df.metric)
        assert df.set_index("metric").loc["knn_r2", "per_point"] is not None
        with pytest.raises(ValueError, match="continuous labels"):
            evaluate_transform(lambda s: Identity(), Ztr, Zte, (ytr > 0).astype(int),
                               (yte > 0).astype(int), seeds=[0], metrics=["knn_r2"])

    def test_constant_targets_raise(self):
        from embedkit.evaluation.metrics import knn_regression_r2

        Ztr, ytr, Zte, _ = self._data()
        with pytest.raises(ValueError, match="constant"):
            knn_regression_r2(Ztr, ytr, Zte, np.ones(len(Zte)))
