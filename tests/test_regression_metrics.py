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


class TestMultivariateTargets:
    """Targets of shape (n, m): R^2 averaged over outputs, with per-point terms."""

    @staticmethod
    def _data(seed=0, m=3):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((300, 6))
        Y = Z[:, :m] * np.arange(1, m + 1) + 0.1 * rng.standard_normal((300, m))
        return Z[:200], Y[:200], Z[200:], Y[200:]

    def test_knn_r2_matches_sklearn_uniform_average(self):
        from sklearn.metrics import r2_score
        from sklearn.neighbors import KNeighborsRegressor

        from embedkit.evaluation.metrics import knn_regression_r2

        Ztr, Ytr, Zte, Yte = self._data()
        value, terms = knn_regression_r2(Ztr, Ytr, Zte, Yte, k=7, metric="euclidean",
                                         return_per_point=True)
        pred = KNeighborsRegressor(7).fit(Ztr, Ytr).predict(Zte)
        assert value == pytest.approx(r2_score(Yte, pred, multioutput="uniform_average"))
        assert terms.shape == (100,) and terms.mean() == pytest.approx(value)

    def test_one_column_equals_1d(self):
        from embedkit.evaluation.metrics import knn_regression_r2

        Ztr, Ytr, Zte, Yte = self._data(m=1)
        a = knn_regression_r2(Ztr, Ytr, Zte, Yte, return_per_point=True)
        b = knn_regression_r2(Ztr, Ytr[:, 0], Zte, Yte[:, 0], return_per_point=True)
        assert a[0] == b[0]
        np.testing.assert_array_equal(a[1], b[1])

    def test_constant_output_column_raises(self):
        from embedkit.evaluation.metrics import knn_regression_r2

        Ztr, Ytr, Zte, Yte = self._data()
        Yte = Yte.copy()
        Yte[:, 1] = 2.0
        with pytest.raises(ValueError, match=r"constant.*\[1\]"):
            knn_regression_r2(Ztr, Ytr, Zte, Yte)

    def test_mismatched_output_count_raises(self):
        from embedkit.evaluation.metrics import knn_regression_r2

        Ztr, Ytr, Zte, Yte = self._data()
        with pytest.raises(ValueError, match="same m"):
            knn_regression_r2(Ztr, Ytr, Zte, Yte[:, :2])

    def test_ridge_probe_returns_terms(self):
        from embedkit.evaluation.metrics import linear_probe

        Ztr, Ytr, Zte, Yte = self._data(1)
        value, terms = linear_probe(Ztr, Ytr, Zte, Yte, task="regression",
                                    return_per_point=True)
        assert terms.shape == (100,) and terms.mean() == pytest.approx(value) and value > 0.9

    def test_compare_gives_an_interval(self):
        from embedkit.baselines import Identity, PCAReduce
        from embedkit.evaluation import compare, evaluate_transform

        Ztr, Ytr, Zte, Yte = self._data(2)
        kw = dict(seeds=[0], metrics=["knn_r2", "linear_probe"])
        res = [evaluate_transform(lambda s: Identity(), Ztr, Zte, Ytr, Yte, method="raw", **kw),
               evaluate_transform(lambda s: PCAReduce(2), Ztr, Zte, Ytr, Yte, method="pca", **kw)]
        table = compare(res, reference="raw")
        row = table[(table.method == "pca") & (table.metric == "knn_r2")].iloc[0]
        assert np.isfinite(row["diff_lo"]) and np.isfinite(row["diff_hi"])
