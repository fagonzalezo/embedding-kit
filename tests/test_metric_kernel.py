"""WP 4.4: one metric in every analyzer (D28) and the centred kernel spectrum (D29)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from embedkit.analysis.kernel import KernelDiagnostics, _center
from embedkit.analysis.report import (
    CALIBRATED_STATISTICS,
    EmbedKitAnalyzer,
    calibration_statistics,
    l2_normalize,
)


def _gauss(n, d, alpha=0.0, seed=0, mean=0.0):
    rng = np.random.default_rng(seed)
    lam = np.arange(1, d + 1, dtype=float) ** -alpha
    return (rng.standard_normal((n, d)) * np.sqrt(lam) + mean).astype(np.float32)


def _numbers(report) -> pd.Series:
    df = report.to_dataframe()
    df = df[df["value"].map(lambda v: isinstance(v, (int, float, np.floating, np.integer))
                            and not isinstance(v, bool))]
    return pd.Series(df["value"].astype(float).values, index=df["category"] + "/" + df["metric"])


# --------------------------------------------------------------------------------------
# D28: the metric reaches every analyzer
# --------------------------------------------------------------------------------------
class TestMetric:
    @pytest.mark.parametrize("metric", ["manhattan", "minkowski", "Cosine"])
    def test_other_metrics_raise(self, small_X, metric):
        with pytest.raises(ValueError, match="not supported"):
            EmbedKitAnalyzer(metric=metric, calibrate=False).fit(small_X)
        with pytest.raises(ValueError, match="not supported"):
            calibration_statistics(small_X, 5, metric=metric)

    def test_cosine_ignores_row_norms(self):
        # Every number in a cosine report depends on the directions only: rescaling each row by
        # a positive factor changes nothing. The exception is the null of the calibration, which
        # is fitted to the rows as given (the observed calibrated values do not change).
        X = _gauss(300, 24, alpha=1.0, seed=1, mean=0.5)
        scale = np.random.default_rng(2).uniform(0.2, 5.0, size=(300, 1)).astype(np.float32)
        kw = dict(metric="cosine", id_methods=["TwoNN", "MLE"], n_null_draws=4,
                  n_calibration=200, id_bootstrap=3)
        a = _numbers(EmbedKitAnalyzer(**kw).fit(X))
        b = _numbers(EmbedKitAnalyzer(**kw).fit(X * scale))
        keep = ~a.index.str.contains("_null_|_z$|_z_lo$|_z_hi$|n_outside_null", regex=True)
        assert a.index.str.endswith("_observed").sum() == len(CALIBRATED_STATISTICS)
        pd.testing.assert_series_equal(a[keep], b[keep], rtol=1e-3, atol=1e-5)

    def test_euclidean_report_depends_on_row_norms(self):
        X = _gauss(300, 24, alpha=1.0, seed=1, mean=0.5)
        scale = np.random.default_rng(2).uniform(0.2, 5.0, size=(300, 1)).astype(np.float32)
        kw = dict(id_methods=["TwoNN"], calibrate=False, id_bootstrap=0, id_scale_curve=False)
        a = EmbedKitAnalyzer(**kw).fit(X).geometry.distance_concentration.distance_cv
        b = EmbedKitAnalyzer(**kw).fit(X * scale).geometry.distance_concentration.distance_cv
        assert abs(a - b) > 0.05

    def test_cosine_equals_euclidean_on_normalised_rows(self):
        # Except the two places where cosine does more than normalise the input: the null draws
        # are normalised too, and neighbour consistency searches its perturbed (no longer unit)
        # rows by cosine.
        X = _gauss(250, 16, alpha=0.5, seed=3, mean=0.3)
        kw = dict(id_methods=["TwoNN"], calibrate=False, id_bootstrap=2)
        a = _numbers(EmbedKitAnalyzer(metric="cosine", **kw).fit(X))
        b = _numbers(EmbedKitAnalyzer(metric="euclidean", **kw).fit(l2_normalize(X)))
        keep = a.index != "Geometry/neighbor_consistency"
        pd.testing.assert_series_equal(a[keep], b[keep], rtol=1e-6, atol=1e-9)

    def test_euclidean_outputs_match_the_analyzers_on_raw_rows(self, small_X):
        from embedkit.analysis.geometry import DistanceConcentration, IsotropyAnalyzer
        from embedkit.analysis.kernel import KernelDiagnostics

        r = EmbedKitAnalyzer(id_methods=["TwoNN"], calibrate=False).fit(small_X)
        assert r.metric == "euclidean"
        assert r.geometry.distance_concentration.distance_cv == DistanceConcentration().fit(
            small_X).distance_cv
        assert r.geometry.isotropy.mean_vector_ratio == IsotropyAnalyzer().fit(
            small_X).mean_vector_ratio
        assert r.kernel.effective_rank == KernelDiagnostics().fit(small_X).effective_rank

    def test_zero_rows_raise_under_cosine(self, small_X):
        X = small_X.copy()
        X[3] = 0
        with pytest.raises(ValueError, match="zero norm"):
            EmbedKitAnalyzer(metric="cosine", calibrate=False).fit(X)
        EmbedKitAnalyzer(id_methods=["TwoNN"], calibrate=False, id_bootstrap=0).fit(X)

    def test_null_data_is_inside_the_cosine_null(self):
        # Gaussian data (the null family) stays within the null under cosine: the null is
        # fitted to the rows as given and its draws are normalised like the data. As in the
        # WP 4.1 acceptance test, a rate over repetitions (here 20, so at least 18 inside).
        from embedkit.analysis.null import calibrate
        from embedkit.analysis.report import CALIBRATED_STATISTICS

        inside = np.zeros(len(CALIBRATED_STATISTICS))
        for seed in range(20):
            rng = np.random.default_rng(seed)
            Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
            X = (_gauss(300, 32, alpha=1.0, seed=seed, mean=0.0) @ Q.T + 0.3).astype(np.float32)
            c = calibrate(lambda Z: calibration_statistics(Z, 10, "cosine", seed), X,
                          model="gaussian", n_draws=20, random_state=seed)
            inside += [abs(c[s].z_score) < 3 for s in CALIBRATED_STATISTICS]
        assert np.all(inside >= 18), dict(zip(CALIBRATED_STATISTICS, inside))


# --------------------------------------------------------------------------------------
# D29: the centred kernel spectrum
# --------------------------------------------------------------------------------------
class TestKernelSpectrum:
    def test_effective_rank_is_the_trace_ratio_of_the_centred_kernel(self):
        X = _gauss(300, 16, alpha=1.0)
        r = KernelDiagnostics().fit(X)
        K = KernelDiagnostics._rbf_kernel(X, r.sigma)
        Kc = _center(K)
        assert r.effective_rank == pytest.approx(np.trace(Kc) ** 2 / np.sum(Kc * Kc), rel=1e-9)

    def test_effective_rank_is_not_capped_by_n_components(self):
        r = KernelDiagnostics(n_components=10).fit(_gauss(1000, 256))
        assert len(r.eigenvalues) == 10
        assert r.effective_rank > 50

    def test_centring_removes_the_constant_direction(self):
        # One isotropic Gaussian has no dominant direction once the kernel is centred; the
        # uncentred kernel's top eigenvalue (the constant direction) is many times the second.
        X = _gauss(800, 32)
        r = KernelDiagnostics().fit(X)
        lam = np.linalg.eigvalsh(KernelDiagnostics._rbf_kernel(X, r.sigma))[::-1]
        assert lam[0] / lam[1] > 20
        assert r.spectral_gap < 2

    def test_two_clusters_give_a_gap(self):
        X = _gauss(600, 16)
        X[:300] += 6.0
        assert KernelDiagnostics().fit(X).spectral_gap > 5

    def test_eigenvalues_are_sorted_and_non_negative(self):
        for n in (60, 1500):  # dense and sparse eigensolver paths
            ev = KernelDiagnostics().fit(_gauss(n, 8, alpha=1.0)).eigenvalues
            assert np.all(ev >= 0) and np.all(np.diff(ev) <= 1e-9)

    def test_condition_number_is_a_deprecated_alias(self, small_X):
        r = KernelDiagnostics().fit(small_X)
        assert r.top_k_eigen_ratio >= r.spectral_gap >= 1.0
        with pytest.warns(DeprecationWarning, match="top_k_eigen_ratio"):
            assert r.condition_number == r.top_k_eigen_ratio

    def test_row_sum_skewness_uses_the_uncentred_kernel(self):
        X = _gauss(500, 8)
        X[:5] *= 0.01  # a tight core: large row sums for a few points
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert KernelDiagnostics().fit(X).row_sum_skewness != 0.0


def test_cosine_knn_matches_scikit_learn_cosine():
    # The scikit-learn backend searches normalised rows with its Euclidean reduction.
    from sklearn.neighbors import NearestNeighbors

    from embedkit.utils.neighbors import knn

    X = _gauss(400, 24, alpha=1.0, seed=5, mean=0.2)
    dist, idx = knn(X, 7, metric="cosine", backend="sklearn")
    ref_d, ref_i = NearestNeighbors(n_neighbors=8, metric="cosine", algorithm="brute").fit(
        X).kneighbors(X)
    np.testing.assert_array_equal(idx, ref_i[:, 1:])
    np.testing.assert_allclose(dist, ref_d[:, 1:], atol=1e-5)


def test_neighbor_consistency_is_translation_invariant():
    # WP 4.5: the perturbation scale ignores the coordinate means.
    from embedkit.analysis.geometry import NeighborConsistency

    X = _gauss(400, 16, alpha=1.0, seed=7)
    shift = 50.0 * np.random.default_rng(8).standard_normal(16).astype(np.float32)
    a = NeighborConsistency(k=10).fit(X).mean_consistency
    b = NeighborConsistency(k=10).fit(X + shift).mean_consistency
    assert a == pytest.approx(b, abs=0.01)
