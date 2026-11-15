"""WP 4.2: per-estimator intrinsic dimension, bootstrap intervals, scale curve, reference."""

import numpy as np
import pytest
import skdim

from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

DEFAULT = ["TwoNN", "MLE", "lPCA"]


def _manifold(kind, m, n, seed, ambient=256):
    """A sphere S^m (in R^{m+1}) or a cube [0, 1]^m, embedded linearly in ``ambient`` dims."""
    if kind == "sphere":
        Z = skdim.datasets.hyperSphere(n, m, random_state=seed)  # n x (m + 1)
    else:
        Z = np.random.default_rng(seed).random((n, m))
    Q, _ = np.linalg.qr(np.random.default_rng(99).standard_normal((ambient, Z.shape[1])))
    return (Z @ Q.T).astype(np.float32)


# --------------------------------------------------------------------------------------
# Acceptance (as revised with the user, 2026-11-14)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["sphere", "cube"])
def test_an_interval_contains_the_true_dimension_at_id5(kind):
    X = _manifold(kind, 5, 5000, seed=0)
    r = IntrinsicDimensionEstimator(methods=DEFAULT).fit(X)
    covered = [m for m, (lo, hi) in r.intervals.items() if lo <= 5 <= hi]
    assert covered, (r.estimates, r.intervals)


def test_bootstrap_se_matches_the_sampling_error():
    # The spread of each estimate over independent draws is the true sampling error; the
    # bootstrap standard error from one draw should match it (within a factor of 2 here; the
    # prototype at n = 5,000 found within about 30%).
    methods = ["TwoNN", "MLE", "MOM"]
    est = [IntrinsicDimensionEstimator(methods=methods, n_bootstrap=0, scale_curve=False)
           .fit(_manifold("sphere", 10, 2000, seed=s)).estimates for s in range(20)]
    se = [IntrinsicDimensionEstimator(methods=methods, scale_curve=False, random_state=s)
          .fit(_manifold("sphere", 10, 2000, seed=100 + s)).standard_errors for s in range(3)]
    for m in methods:
        true_sd = np.std([e[m] for e in est], ddof=1)
        boot_se = np.mean([s[m] for s in se])
        assert 0.5 < boot_se / true_sd < 2.0, (m, boot_se, true_sd)


# --------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------
class TestIntervals:
    def test_interval_brackets_the_estimate(self, small_X):
        r = IntrinsicDimensionEstimator(methods=DEFAULT).fit(small_X)
        for m in DEFAULT:
            lo, hi = r.intervals[m]
            assert lo <= r.estimates[m] <= hi
            assert r.standard_errors[m] >= 0

    def test_no_bootstrap(self, small_X):
        r = IntrinsicDimensionEstimator(methods=DEFAULT, n_bootstrap=0).fit(small_X)
        assert all(np.isnan(s) for s in r.standard_errors.values())
        assert all(np.isnan(lo) and np.isnan(hi) for lo, hi in r.intervals.values())

    def test_slow_methods_are_not_resampled(self, monkeypatch):
        from embedkit.analysis import intrinsic_dim as mod

        calls = {"ESS": 0, "TwoNN": 0}

        class Fake:
            def __init__(self, method):
                self.method = method
                self.dimension_ = 3.0

            def fit(self, X):
                calls[self.method] += 1
                return self

        monkeypatch.setattr(mod, "_build_estimator", lambda m, rs: Fake(m))
        r = mod.IntrinsicDimensionEstimator(methods=["ESS", "TwoNN"], n_bootstrap=5,
                                            scale_curve=False).fit(np.random.default_rng(0)
                                                                   .random((50, 4)))
        assert calls == {"ESS": 1, "TwoNN": 6}
        assert np.isnan(r.standard_errors["ESS"]) and r.standard_errors["TwoNN"] == 0.0


def test_ess_is_supported():
    # New in WP 4.2: ESS is accepted and built (it needs an entry in the estimator map too).
    X = _manifold("cube", 3, 300, seed=0, ambient=10)
    r = IntrinsicDimensionEstimator(methods=["ESS"], scale_curve=False).fit(X)
    assert 2 < r.estimates["ESS"] < 4


# --------------------------------------------------------------------------------------
# Scale curve
# --------------------------------------------------------------------------------------
class TestScaleCurve:
    def test_levels_and_plateau_on_a_clean_manifold(self):
        r = IntrinsicDimensionEstimator(methods=["TwoNN"], n_bootstrap=0).fit(
            _manifold("sphere", 5, 4000, seed=0))
        assert list(r.scale_curve) == [4000, 2000, 1000, 500]
        assert r.scale_curve[4000] == r.estimates["TwoNN"]
        assert r.scale_plateau is True

    def test_noise_breaks_the_plateau(self):
        # Small isotropic noise in all 256 dimensions inflates TwoNN at short scales (many
        # rows) far more than at long scales (Denti et al. 2022).
        X = _manifold("sphere", 3, 4000, seed=0)
        X = X + 0.01 * np.random.default_rng(1).standard_normal(X.shape).astype(np.float32)
        r = IntrinsicDimensionEstimator(methods=["TwoNN"], n_bootstrap=0).fit(X)
        assert r.scale_plateau is False, r.scale_curve

    def test_small_samples_skip_tiny_levels(self, small_X):
        r = IntrinsicDimensionEstimator(methods=["TwoNN"], n_bootstrap=0).fit(small_X)
        assert list(r.scale_curve) == [100, 50, 25]

    def test_disabled(self, small_X):
        r = IntrinsicDimensionEstimator(methods=["MLE"], scale_curve=False).fit(small_X)
        assert r.scale_curve == {} and r.scale_plateau is None

    def test_computed_without_twonn_in_methods(self, small_X):
        r = IntrinsicDimensionEstimator(methods=["MLE"], n_bootstrap=0).fit(small_X)
        assert len(r.scale_curve) == 3


# --------------------------------------------------------------------------------------
# Reference estimate and the report
# --------------------------------------------------------------------------------------
class TestReference:
    def test_twonn_is_preferred_whatever_the_order(self, small_X):
        r = IntrinsicDimensionEstimator(methods=["MLE", "TwoNN"], n_bootstrap=0).fit(small_X)
        assert r.reference_method == "TwoNN"

    def test_first_method_without_twonn(self, small_X):
        r = IntrinsicDimensionEstimator(methods=["lPCA", "MLE"], n_bootstrap=0).fit(small_X)
        assert r.reference_method == "lPCA" and r.reference_estimate == r.estimates["lPCA"]

    def test_report_target_dim_from_the_reference(self, small_X):
        from embedkit.analysis.report import EmbedKitAnalyzer

        report = EmbedKitAnalyzer(id_methods=["MLE", "TwoNN"], calibrate=False).fit(small_X)
        ref = report.intrinsic_dim.estimates["TwoNN"]
        expected = int(np.clip(round(1.5 * ref), max(4, int(ref)), small_X.shape[1]))
        assert report.suggested_target_dim == expected

    def test_embedkit_skips_bootstrap_and_curve(self, small_X):
        from embedkit import EmbedKit

        ek = EmbedKit(method="pca").fit(small_X)
        idr = ek.analysis_report.intrinsic_dim
        assert all(np.isnan(s) for s in idr.standard_errors.values())
        assert idr.scale_curve == {}
