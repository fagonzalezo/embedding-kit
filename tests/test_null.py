"""WP 4.1: null models and calibrated statistics."""

import os

import numpy as np
import pytest

from embedkit.analysis.null import (
    calibrate,
    calibrated_statistic,
    column_permuted,
    mean_shrinkage,
    null_distribution,
    shrinkage_weight,
    spectrum_matched_gaussian,
)
from embedkit.analysis.report import CALIBRATED_STATISTICS, calibration_statistics

SLOW = os.environ.get("EMBEDKIT_SLOW_TESTS") == "1"


def _gaussian(n, d, spectrum, seed):
    """Rows from a Gaussian with the given covariance spectrum, a random rotation and a mean."""
    rng = np.random.default_rng(seed)
    lam = np.ones(d) if spectrum == "flat" else np.arange(1, d + 1) ** -1.0
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    mean = 0.5 * rng.standard_normal(d)
    return ((rng.standard_normal((n, d)) * np.sqrt(lam)) @ Q.T + mean).astype(np.float32)


# --------------------------------------------------------------------------------------
# The null models
# --------------------------------------------------------------------------------------
class TestSpectrumMatchedGaussian:
    def test_shape_dtype_and_mean(self):
        X = _gaussian(500, 16, "power", seed=0)
        Y = spectrum_matched_gaussian(X, n=4000, rng=1)
        assert Y.shape == (4000, 16) and Y.dtype == np.float32
        np.testing.assert_allclose(Y.mean(0), X.mean(0), atol=0.05)

    def test_keeps_total_variance(self):
        X = _gaussian(500, 64, "power", seed=0)
        Y = spectrum_matched_gaussian(X, n=20000, rng=1)
        assert np.trace(np.cov(Y.T)) == pytest.approx(np.trace(np.cov(X.T)), rel=0.03)

    def test_iid_data_gives_an_isotropic_null(self):
        # The sample spectrum of iid data is spread (d / n = 1.7); the null undoes that.
        X = _gaussian(300, 512, "flat", seed=0)
        Y = spectrum_matched_gaussian(X, n=20000, rng=1).astype(np.float64)
        lam = np.linalg.eigvalsh(np.cov(Y.T))
        pr = lam.sum() ** 2 / (lam**2).sum()
        assert pr > 0.85 * 512  # the sample covariance itself has PR of about 0.37 d

    def test_anisotropic_data_keeps_its_spectrum(self):
        X = _gaussian(5000, 32, "power", seed=0)
        Xc = X - X.mean(0)
        top = np.linalg.svd(Xc, full_matrices=False)[2][0]
        Y = spectrum_matched_gaussian(X, n=20000, rng=1)
        top_null = np.linalg.svd(Y - Y.mean(0), full_matrices=False)[2][0]
        assert abs(top @ top_null) > 0.99

    def test_shrinkage_weight_matches_the_unbiased_second_moment(self):
        # Averaged over samples, tr(Sigma_hat^2) estimates tr(Sigma^2) without bias.
        rng = np.random.default_rng(0)
        d, n = 64, 100
        lam = np.arange(1, d + 1) ** -0.5
        ests = []
        for _ in range(200):
            X = rng.standard_normal((n, d)) * np.sqrt(lam)
            C = np.cov(X.T)
            t1, t2 = np.trace(C), np.sum(C * C)
            a = shrinkage_weight(t1, t2, n - 1, d)
            b = (1 - a) * t1 / d
            ests.append(a**2 * t2 + 2 * a * b * t1 + b**2 * d)
        assert np.mean(ests) == pytest.approx(np.sum(lam**2), rel=0.03)

    def test_constant_data(self):
        X = np.ones((50, 4), dtype=np.float32)
        Y = spectrum_matched_gaussian(X, rng=0)
        np.testing.assert_allclose(Y, X)


class TestColumnPermuted:
    def test_keeps_each_column_and_removes_dependence(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal(2000)
        X = np.stack([z, z + 0.1 * rng.standard_normal(2000)], axis=1).astype(np.float32)
        Y = column_permuted(X, rng=1)
        for j in range(2):
            np.testing.assert_array_equal(np.sort(Y[:, j]), np.sort(X[:, j]))
        assert abs(np.corrcoef(Y.T)[0, 1]) < 0.1 < np.corrcoef(X.T)[0, 1]


class TestNullDistribution:
    def test_scalar_and_dict_statistics(self):
        X = _gaussian(200, 8, "flat", seed=0)
        v = null_distribution(lambda Z: float(Z.std()), X, n_draws=5, random_state=0)
        assert v.shape == (5,)
        d = null_distribution(lambda Z: {"a": 1.0, "b": float(Z.mean())}, X, n_draws=5,
                              random_state=0)
        assert set(d) == {"a", "b"} and d["a"].shape == (5,)

    @pytest.mark.parametrize("model", ["gaussian", "permuted"])
    def test_seeded(self, model):
        X = _gaussian(200, 8, "flat", seed=0)
        a = null_distribution(lambda Z: float(Z[0, 0]), X, model, n_draws=3, random_state=7)
        b = null_distribution(lambda Z: float(Z[0, 0]), X, model, n_draws=3, random_state=7)
        np.testing.assert_array_equal(a, b)

    def test_invalid_arguments(self):
        X = _gaussian(50, 4, "flat", seed=0)
        with pytest.raises(ValueError, match="model"):
            null_distribution(lambda Z: 0.0, X, model="uniform")
        with pytest.raises(ValueError, match="n_draws"):
            null_distribution(lambda Z: 0.0, X, n_draws=1)


class TestCalibratedStatistic:
    def test_z_score(self):
        c = calibrated_statistic(3.0, np.array([0.0, 1.0, 2.0]), "gaussian", 10)
        assert c.null_mean == 1.0 and c.null_std == 1.0 and c.z_score == 2.0
        assert c.n_draws == 3 and c.n_samples == 10

    def test_null_without_spread(self):
        same = calibrated_statistic(1.0, np.ones(5), "gaussian", 10)
        below = calibrated_statistic(0.9, np.ones(5), "gaussian", 10)
        assert same.z_score == 0.0
        assert below.z_score == -np.inf


# --------------------------------------------------------------------------------------
# Acceptance: data drawn from the null model give z-scores within +-3
# --------------------------------------------------------------------------------------
def _z_scores(X, model, n_draws, seed):
    n = X.shape[0]
    k = max(5, round(np.sqrt(n) / 2))  # the report's default k
    cal = calibrate(lambda Z: calibration_statistics(Z, k), X, model=model, n_draws=n_draws,
                    random_state=seed)
    return {s: cal[s].z_score for s in CALIBRATED_STATISTICS}


def _share_within(cells, reps, model="gaussian", n_draws=20):
    out = {}
    for n, d, spectrum in cells:
        zs = [_z_scores(_gaussian(n, d, spectrum, seed=1000 + r) if model == "gaussian"
                        else _independent_columns(n, d, seed=1000 + r), model, n_draws, r)
              for r in range(reps)]
        for s in CALIBRATED_STATISTICS:
            out[(n, d, spectrum, s)] = np.mean([abs(z[s]) <= 3 for z in zs])
    return out


def _independent_columns(n, d, seed):
    """Independent, non-Gaussian columns with different scales: the permutation null."""
    rng = np.random.default_rng(seed)
    return (rng.exponential(size=(n, d)) * np.linspace(0.5, 2.0, d)).astype(np.float32)


def test_gaussian_null_is_calibrated_small():
    """Fast version of the WP 4.1 acceptance test (n = 300, 10 repetitions)."""
    cells = [(300, 32, "flat"), (300, 512, "flat"), (300, 32, "power"), (300, 512, "power")]
    share = _share_within(cells, reps=10)
    bad = {key: v for key, v in share.items() if v < 0.8}
    assert not bad, bad


def test_gaussian_null_is_calibrated_without_a_mean():
    """Zero-mean data (E3, WP 4.5): the null's mean must not add the sample mean's noise.

    Centred on the raw sample mean, the null gave uniformity z of -8 to -15 here.
    """
    zs = []
    for r in range(10):
        X = np.random.default_rng(2000 + r).standard_normal((300, 512)).astype(np.float32)
        zs.append(_z_scores(X, "gaussian", 20, r))
    for s in CALIBRATED_STATISTICS:
        assert np.mean([abs(z[s]) <= 3 for z in zs]) >= 0.8, (s, [z[s] for z in zs])


def test_mean_shrinkage():
    assert mean_shrinkage(0.0, 1.0, 10) == 0.0
    assert mean_shrinkage(1.0, 20.0, 10) == 0.0  # ||m||^2 below tr(C) / n: clipped
    assert mean_shrinkage(1.0, 5.0, 20) == pytest.approx(np.sqrt(0.75))


def test_permutation_null_is_calibrated_small():
    share = _share_within([(300, 32, "independent")], reps=10, model="permuted")
    bad = {key: v for key, v in share.items() if v < 0.8}
    assert not bad, bad


@pytest.mark.skipif(not SLOW, reason="slow (over 45 min on a laptop); set EMBEDKIT_SLOW_TESTS=1")
def test_gaussian_null_is_calibrated_full():
    """WP 4.1 acceptance: within +-3 in at least 95% of 40 repetitions, per statistic."""
    cells = [(n, d, s) for s in ("flat", "power") for n in (300, 3000) for d in (32, 512)]
    share = _share_within(cells, reps=40)
    for key, v in sorted(share.items()):
        print(key, v)
    bad = {key: v for key, v in share.items() if v < 0.95}
    assert not bad, bad


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------
class TestReportCalibration:
    def test_heavy_tails_are_flagged_as_hubness(self):
        # A multivariate t has the covariance shape of a Gaussian but many points near the
        # centre, which become hubs.
        from embedkit.analysis.report import EmbedKitAnalyzer

        rng = np.random.default_rng(0)
        Z = rng.standard_normal((1000, 128))
        X = (Z / np.sqrt(rng.chisquare(3, size=(1000, 1)) / 3)).astype(np.float32)
        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X)
        assert report.calibration["k_skewness"].z_score > 3
        assert any("More hubness than the null" in r and "MutualProximity" in r
                   for r in report.recommendations)
        # E4 (WP 7.1): under cosine the hubness rule did not predict the gain; it raises
        # nothing there, and the z-score stays in the calibration.
        cos = EmbedKitAnalyzer(id_methods=["TwoNN"], metric="cosine", id_bootstrap=0,
                               id_scale_curve=False).fit(X)
        assert not any("than the null" in r for r in cos.recommendations)

    def test_gaussian_data_raise_no_calibrated_recommendation(self):
        # Checks the report's wiring on one draw; calibration itself is tested by the
        # acceptance tests above. Seed 3 has max |z| = 0.37 (seeds 0 and 1 give a k-skewness z
        # of about 2.5, too close to 3 for a single-draw test across BLAS builds).
        from embedkit.analysis.report import EmbedKitAnalyzer

        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(_gaussian(1000, 64, "power", 3))
        assert report.n_outside_null == 0, report.calibration
        assert not any("than the null" in r for r in report.recommendations)

    def test_calibration_subsample_and_off_switch(self):
        from embedkit.analysis.report import EmbedKitAnalyzer

        X = _gaussian(600, 16, "flat", 0)
        report = EmbedKitAnalyzer(id_methods=["TwoNN"], n_calibration=300,
                                  n_null_draws=5).fit(X)
        c = report.calibration["k_skewness"]
        assert c.n_samples == 300 and c.n_draws == 5
        off = EmbedKitAnalyzer(id_methods=["TwoNN"], calibrate=False).fit(X)
        assert off.calibration is None and off.n_outside_null is None

    def test_permuted_null_and_cosine_metric(self):
        from embedkit.analysis.report import EmbedKitAnalyzer

        report = EmbedKitAnalyzer(id_methods=["TwoNN"], null_model="permuted",
                                  metric="cosine", n_null_draws=5).fit(_gaussian(300, 16,
                                                                                 "flat", 0))
        assert report.calibration["uniformity"].null_model == "permuted"

    def test_calibration_k_depends_on_the_calibration_rows_only(self):
        # E3 (WP 4.5): reusing the report's k (sqrt(n) / 2 on all rows) made the z-scores of
        # k-dependent statistics grow with n although the calibration rows were fixed.
        from embedkit.analysis.report import EmbedKitAnalyzer

        kw = dict(id_methods=["TwoNN"], id_bootstrap=0, id_scale_curve=False,
                  n_calibration=300, n_null_draws=3)
        small = EmbedKitAnalyzer(**kw).fit(_gaussian(600, 16, "flat", 0))
        large = EmbedKitAnalyzer(**kw).fit(_gaussian(2400, 16, "flat", 0))
        assert small.suggested_k == 12 and large.suggested_k == 24
        assert small.calibration_k == large.calibration_k == 9  # round(sqrt(300) / 2)
        fixed = EmbedKitAnalyzer(k=7, **kw).fit(_gaussian(600, 16, "flat", 0))
        assert fixed.calibration_k == 7

    def test_rows_normalised_upstream_are_inside_the_euclidean_null(self):
        # E3 (WP 4.5): unit rows against a Gaussian null with varying norms gave distance_cv
        # z down to -35 and a concentration recommendation in every cell.
        from embedkit.analysis.report import EmbedKitAnalyzer, l2_normalize

        kw = dict(id_methods=["TwoNN"], id_bootstrap=0, id_scale_curve=False)
        for seed in range(3):
            X = 2.0 * l2_normalize(_gaussian(800, 128, "flat", seed))
            report = EmbedKitAnalyzer(**kw).fit(X)
            c = report.calibration["distance_cv"]
            assert abs(c.z_score) < 3, report.calibration
            assert not any("concentrated" in r for r in report.recommendations)

    def test_common_norm(self):
        from embedkit.analysis.report import common_norm

        X = np.random.default_rng(0).standard_normal((50, 8))
        assert common_norm(X) is None
        U = 3.0 * X / np.linalg.norm(X, axis=1, keepdims=True)
        assert common_norm(U) == pytest.approx(3.0)
        assert common_norm(np.zeros((5, 3))) is None

    def test_antihub_null_spread_is_floored(self):
        # At low dimension antihubs are rare, and often absent from every null draw; the floor
        # keeps one or two observed antihubs from giving an infinite z.
        from embedkit.analysis.report import EmbedKitAnalyzer

        X = np.random.default_rng(0).standard_normal((1000, 3)).astype(np.float32)
        report = EmbedKitAnalyzer(id_methods=["TwoNN"], id_bootstrap=0,
                                  id_scale_curve=False).fit(X)
        c = report.calibration["antihub_ratio"]
        assert c.null_std >= 1 / 1000 and np.isfinite(c.z_score)

    def test_antihub_recommendation_comes_from_the_z_score(self):
        from embedkit.analysis.report import _calibrated_recommendations

        def cal(observed):
            null = np.array([0.10, 0.11, 0.12])
            return {"antihub_ratio": calibrated_statistic(observed, null, "gaussian", 100)}

        many = _calibrated_recommendations(cal(0.4), 3.0)
        assert any("More antihubs than the null" in r for r in many)
        assert not _calibrated_recommendations(cal(0.0), 3.0)  # fewer antihubs: no action
        # Under cosine the antihub ratio is not calibrated (E3): no recommendation from it.
        assert not _calibrated_recommendations(cal(0.4), 3.0, metric="cosine")
