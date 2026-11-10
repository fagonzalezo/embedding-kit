"""Regression tests for WP 1.5 (review defects D6, D7, D8, D10, D14).

Each statistic is checked on data with a known answer.
"""

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_blobs


# --------------------------------------------------------------------------------------
# D6: isotropy
# --------------------------------------------------------------------------------------
def _iid(n, d, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


class TestIsotropyD6:
    @pytest.fixture(scope="class")
    def results(self):
        from embedkit.analysis.geometry import IsotropyAnalyzer

        return {d: IsotropyAnalyzer().fit(_iid(3000, d)) for d in (1000, 2500)}

    def test_normalized_pr_same_across_old_dmax_threshold(self, results):
        a = results[1000].participation_ratio_normalized
        b = results[2500].participation_ratio_normalized
        assert abs(a - b) < 0.05
        # isotropic data: close to 1 at both
        assert a > 0.9 and b > 0.9

    def test_scores_do_not_contradict_across_threshold(self, results):
        # Isotropic data: the effective-rank based score must be high at both dimensions.
        assert results[1000].isotropy_score > 0.5 and results[2500].isotropy_score > 0.5
        # effective rank is a real count of dimensions, not capped at 512
        assert results[2500].effective_rank > 1000

    def test_no_discontinuity_at_old_dmax(self):
        # Just below and just above the old d_max = 2000 (where the method used to change).
        from embedkit.analysis.geometry import IsotropyAnalyzer

        lo = IsotropyAnalyzer().fit(_iid(3000, 1950, seed=1))
        hi = IsotropyAnalyzer().fit(_iid(3000, 2050, seed=1))
        assert abs(lo.participation_ratio_normalized - hi.participation_ratio_normalized) < 0.02
        assert abs(lo.isotropy_score - hi.isotropy_score) < 0.03
        assert abs(lo.effective_rank / 1950 - hi.effective_rank / 2050) < 0.03

    def test_plug_in_pr_is_exact_trace_formula(self):
        from embedkit.analysis.geometry import IsotropyAnalyzer

        rng = np.random.default_rng(1)
        X = (rng.standard_normal((400, 30)) * np.linspace(0.2, 3.0, 30)).astype(np.float32)
        ev = np.linalg.eigvalsh(np.cov(X.astype(np.float64).T))
        expected = ev.sum() ** 2 / (ev ** 2).sum()
        r = IsotropyAnalyzer(random_state=0).fit(X)
        assert r.participation_ratio == pytest.approx(expected, rel=1e-4)

    def test_gram_branch_matches_covariance_branch(self):
        # n < d uses the Gram matrix; the trace identities must give the same PR as the
        # covariance computed directly.
        from embedkit.analysis.geometry import IsotropyAnalyzer

        rng = np.random.default_rng(2)
        X = (rng.standard_normal((60, 200)) * np.linspace(0.2, 3.0, 200)).astype(np.float32)
        ev = np.linalg.eigvalsh(np.cov(X.astype(np.float64).T))
        expected = ev.sum() ** 2 / (ev ** 2).sum()
        r = IsotropyAnalyzer(random_state=0).fit(X)
        assert r.participation_ratio == pytest.approx(expected, rel=1e-4)

    def test_anisotropic_data_is_flagged_by_normalized_pr(self):
        from embedkit.analysis.geometry import IsotropyAnalyzer

        rng = np.random.default_rng(3)
        scale = np.ones(200)
        scale[:2] = 30.0
        X = (rng.standard_normal((2000, 200)) * scale).astype(np.float32)
        r = IsotropyAnalyzer().fit(X)
        assert r.participation_ratio_normalized < 0.1

    def test_subsample_is_seeded(self):
        from embedkit.analysis.geometry import IsotropyAnalyzer

        X = _iid(6000, 20)
        a = IsotropyAnalyzer(random_state=1).fit(X).participation_ratio
        b = IsotropyAnalyzer(random_state=1).fit(X).participation_ratio
        assert a == b

    @pytest.mark.parametrize("d", [1000, 2500])
    def test_report_has_no_anisotropy_recommendation(self, d):
        from embedkit.analysis.report import EmbedKitAnalyzer

        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(_iid(3000, d))
        assert not any("anisotrop" in r.lower() or "participation" in r.lower()
                       for r in report.recommendations), report.recommendations

    def test_spectrum_top_and_deprecated_alias(self):
        from embedkit.analysis.geometry import IsotropyAnalyzer

        r = IsotropyAnalyzer(n_components=16).fit(_iid(300, 40))
        assert len(r.eigenvalue_spectrum_top) == 16
        with pytest.warns(DeprecationWarning):
            old = r.eigenvalue_spectrum
        np.testing.assert_array_equal(old, r.eigenvalue_spectrum_top)


# --------------------------------------------------------------------------------------
# D7 / D8: kernel alignment
# --------------------------------------------------------------------------------------
class TestKernelAlignmentD7:
    @pytest.fixture(scope="class")
    def X(self):
        return _iid(1000, 32, seed=4)

    @pytest.mark.parametrize("n_classes", [2, 10])
    def test_independent_labels_give_zero(self, X, n_classes):
        from embedkit.analysis.kernel import KernelDiagnostics

        y = np.random.default_rng(5).integers(0, n_classes, len(X))
        r = KernelDiagnostics().fit(X, y)
        assert abs(r.centered_kernel_alignment) < 0.05

    def test_cluster_labels_give_high_alignment(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        X, y = make_blobs(n_samples=600, n_features=32, centers=5, random_state=0)
        r = KernelDiagnostics().fit(X.astype(np.float32), y)
        assert r.centered_kernel_alignment > 0.5

    def test_unbalanced_independent_labels_give_zero(self, X):
        # The old, uncentred statistic was ~ class balance (0.7 for two balanced classes).
        from embedkit.analysis.kernel import KernelDiagnostics

        y = (np.random.default_rng(6).random(len(X)) < 0.9).astype(int)
        r = KernelDiagnostics().fit(X, y)
        assert abs(r.centered_kernel_alignment) < 0.05

    def test_continuous_1d_smooth_target_gives_high_alignment(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        X = _iid(800, 2, seed=7)
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]  # float64, continuous
        r = KernelDiagnostics().fit(X, y)
        assert r.centered_kernel_alignment > 0.5

    def test_continuous_1d_independent_target_gives_zero(self, X):
        from embedkit.analysis.kernel import KernelDiagnostics

        y = np.random.default_rng(8).standard_normal(len(X))
        r = KernelDiagnostics().fit(X, y)
        assert abs(r.centered_kernel_alignment) < 0.05

    def test_continuous_2d_target(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        X = _iid(800, 2, seed=9)
        y_dep = np.stack([X[:, 0], X[:, 1] ** 2], axis=1)
        y_ind = np.random.default_rng(10).standard_normal((800, 2))
        assert KernelDiagnostics().fit(X, y_dep).centered_kernel_alignment > 0.5
        assert abs(KernelDiagnostics().fit(X, y_ind).centered_kernel_alignment) < 0.05

    def test_deprecated_alias(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        X, y = make_blobs(n_samples=200, n_features=8, centers=3, random_state=0)
        r = KernelDiagnostics().fit(X, y)
        with pytest.warns(DeprecationWarning):
            old = r.kernel_alignment
        assert old == r.centered_kernel_alignment

    def test_no_labels_gives_none(self, X):
        from embedkit.analysis.kernel import KernelDiagnostics

        r = KernelDiagnostics().fit(X)
        assert r.centered_kernel_alignment is None
        with pytest.warns(DeprecationWarning):
            assert r.kernel_alignment is None


class TestKernelLabelTypesD8:
    def test_string_labels(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        X, y = make_blobs(n_samples=300, n_features=8, centers=3, random_state=0)
        names = np.array(["cat", "dog", "bird"])[y]
        r_str = KernelDiagnostics().fit(X, names)
        r_int = KernelDiagnostics().fit(X, y)
        assert r_str.centered_kernel_alignment == pytest.approx(
            r_int.centered_kernel_alignment, abs=1e-5
        )

    def test_python_list_longer_than_subsample(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        X, y = make_blobs(n_samples=2500, n_features=8, centers=3, random_state=0)
        r = KernelDiagnostics(subsample=500).fit(X.astype(np.float32), list(y))
        assert r.centered_kernel_alignment > 0.5

    def test_python_list_of_strings_longer_than_subsample(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        X, y = make_blobs(n_samples=2500, n_features=8, centers=3, random_state=0)
        labels = [f"c{i}" for i in y]
        r = KernelDiagnostics(subsample=500).fit(X.astype(np.float32), labels)
        assert r.centered_kernel_alignment > 0.5

    def test_wrong_label_length_raises(self):
        from embedkit.analysis.kernel import KernelDiagnostics

        with pytest.raises(ValueError):
            KernelDiagnostics().fit(_iid(100, 4), np.zeros(99, dtype=int))


# --------------------------------------------------------------------------------------
# D10: distance concentration
# --------------------------------------------------------------------------------------
class TestDistanceConcentrationD10:
    def test_distance_cv_matches_definition(self):
        from scipy.spatial.distance import pdist

        from embedkit.analysis.geometry import DistanceConcentration

        X = _iid(300, 20, seed=11)
        u = pdist(X.astype(np.float64))
        r = DistanceConcentration().fit(X)
        assert r.distance_cv == pytest.approx(u.std() / u.mean(), rel=1e-3)

    def test_distance_cv_known_answer_for_iid_gaussians(self):
        # ||x - y|| ~ sqrt(2d) with standard deviation ~ 1, so CV ~ 1 / sqrt(2d).
        from embedkit.analysis.geometry import DistanceConcentration

        d = 500
        r = DistanceConcentration().fit(_iid(1000, d, seed=12))
        assert r.distance_cv == pytest.approx(1 / np.sqrt(2 * d), rel=0.1)

    def test_distance_cv_robust_to_one_duplicate(self):
        from embedkit.analysis.geometry import DistanceConcentration

        X = _iid(1000, 64, seed=13)
        before = DistanceConcentration().fit(X)
        X2 = X.copy()
        X2[1] = X2[0]
        after = DistanceConcentration().fit(X2)
        assert abs(after.distance_cv - before.distance_cv) / before.distance_cv < 0.01

    def test_relative_contrast_per_query_known_answer(self):
        # Points on a line at 0, 1, 3: queries see (max-min)/min of
        # {1,3} -> 2, {1,2} -> 1, {3,2} -> 0.5 ; median = 1.
        from embedkit.analysis.geometry import DistanceConcentration

        X = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
        r = DistanceConcentration().fit(X)
        assert r.relative_contrast_per_query == pytest.approx(1.0, rel=1e-5)

    def test_relative_contrast_per_query_excludes_zero_distances(self):
        from embedkit.analysis.geometry import DistanceConcentration

        X = np.array([[0.0], [0.0], [1.0], [3.0]], dtype=np.float32)
        r = DistanceConcentration().fit(X)
        # Queries: 0 -> {1,3}: 2 ; 0 -> {1,3}: 2 ; 1 -> {1,1,2}: 1 ; 3 -> {3,3,2}: 0.5
        assert r.relative_contrast_per_query == pytest.approx(1.5, rel=1e-5)

    def test_old_fields_kept(self):
        from embedkit.analysis.geometry import DistanceConcentration

        r = DistanceConcentration().fit(_iid(100, 8))
        assert r.relative_contrast > 0 and 0 <= r.concentration_ratio <= 1

    def test_report_flags_concentration_by_cv(self):
        from types import SimpleNamespace as NS

        from embedkit.analysis.report import _assess_severity_and_recommendations

        def bundle(cv):
            return NS(
                isotropy=NS(participation_ratio_normalized=1.0, participation_ratio=1.0),
                distance_concentration=NS(distance_cv=cv, concentration_ratio=0.0),
                neighbor_consistency=NS(mean_consistency=1.0),
                uniformity=NS(uniformity=-3.0),
            )

        idr = NS(consensus=float("nan"))
        hub = NS(k_skewness=0.0, antihub_ratio=0.0)
        _, low_cv = _assess_severity_and_recommendations(idr, hub, bundle(0.005), None, 100)
        _, ok_cv = _assess_severity_and_recommendations(idr, hub, bundle(0.2), None, 100)
        assert any("distance concentration" in r for r in low_cv)
        assert not any("distance concentration" in r for r in ok_cv)


# --------------------------------------------------------------------------------------
# D14: intrinsic dimension
# --------------------------------------------------------------------------------------
METHODS = ["TwoNN", "MLE", "lPCA"]


class TestIntrinsicDimensionD14:
    @pytest.fixture(scope="class")
    def digits(self):
        from sklearn.datasets import load_digits

        return load_digits().data.astype(np.float32)

    @pytest.fixture(scope="class")
    def digits_dup(self, digits):
        # digits has no exact duplicates; copy 30 rows to create them.
        return np.vstack([digits, digits[:30]])

    def test_same_random_state_gives_identical_finite_estimates(self, digits):
        # Regression: the estimators used to run in a ThreadPoolExecutor, and TwoNN returned
        # 8.89, 8.87, 8.76, ... (and sometimes NaN) for the same input and seed.
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

        runs = [
            IntrinsicDimensionEstimator(methods=METHODS, random_state=0).fit(digits)
            for _ in range(12)
        ]
        for r in runs:
            assert all(np.isfinite(v) and v > 0 for v in r.estimates.values()), r.estimates
            assert r.estimates == runs[0].estimates
            assert r.consensus == runs[0].consensus

    def test_digits_with_duplicates_gives_identical_finite_estimates(self, digits_dup):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

        runs = [
            IntrinsicDimensionEstimator(methods=METHODS, random_state=0).fit(digits_dup)
            for _ in range(4)
        ]
        for r in runs:
            assert all(np.isfinite(v) and v > 0 for v in r.estimates.values()), r.estimates
            assert r.estimates == runs[0].estimates
            assert r.n_duplicates_removed == 30

    def test_duplicates_removed_and_reported(self, digits_dup):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

        r = IntrinsicDimensionEstimator(methods=["TwoNN"]).fit(digits_dup)
        assert r.n_duplicates_removed == 30

    def test_no_duplicates_reports_zero(self):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

        r = IntrinsicDimensionEstimator(methods=["TwoNN"]).fit(_iid(200, 8))
        assert r.n_duplicates_removed == 0

    def test_duplicates_do_not_change_the_estimate(self):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

        X = _iid(500, 6, seed=14)
        Xd = np.vstack([X, X[:50], X[:50]])
        a = IntrinsicDimensionEstimator(methods=["TwoNN", "MLE"]).fit(X)
        b = IntrinsicDimensionEstimator(methods=["TwoNN", "MLE"]).fit(Xd)
        assert b.n_duplicates_removed == 100
        assert b.estimates == a.estimates

    def test_analyzer_reports_are_reproducible_on_digits(self, digits):
        from embedkit.analysis.report import EmbedKitAnalyzer

        reps = [EmbedKitAnalyzer(id_methods=["TwoNN", "MLE", "lPCA"], random_state=0).fit(digits)
                for _ in range(3)]
        assert all(r.intrinsic_dim.estimates == reps[0].intrinsic_dim.estimates for r in reps)
        assert reps[0].intrinsic_dim.estimates == reps[1].intrinsic_dim.estimates
        assert not any(np.isnan(v) for v in reps[0].intrinsic_dim.estimates.values())
        assert reps[0].suggested_target_dim == reps[1].suggested_target_dim

    def test_non_finite_estimate_is_a_failure(self, monkeypatch):
        from embedkit.analysis import intrinsic_dim as mod

        class Bad:
            def __init__(self, value):
                self.dimension_ = value

            def fit(self, X):
                return self

        values = {"TwoNN": float("inf"), "MLE": 0.0, "lPCA": 4.0}
        monkeypatch.setattr(mod, "_build_estimator", lambda m, rs: Bad(values[m]))
        with pytest.warns(UserWarning):
            r = mod.IntrinsicDimensionEstimator(methods=["TwoNN", "MLE", "lPCA"]).fit(_iid(50, 8))
        assert np.isnan(r.estimates["TwoNN"]) and np.isnan(r.estimates["MLE"])
        assert r.estimates["lPCA"] == 4.0
        assert r.consensus == 4.0  # failures excluded from the consensus


def test_old_field_aliases_do_not_warn_on_new_names():
    from embedkit.analysis.geometry import IsotropyAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = IsotropyAnalyzer().fit(_iid(100, 10))
        r.eigenvalue_spectrum_top
        r.participation_ratio_normalized
