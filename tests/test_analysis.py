"""Tests for embedkit.analysis.*"""

import numpy as np
import pytest

from embedkit.utils.neighbors import clear_cache


@pytest.fixture(autouse=True)
def reset_knn_cache():
    clear_cache()
    yield
    clear_cache()


class TestIntrinsicDim:
    def test_result_fields(self, small_X):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator
        result = IntrinsicDimensionEstimator(methods=["TwoNN"]).fit(small_X)
        assert "TwoNN" in result.estimates
        assert isinstance(result.consensus, float)
        assert result.uncertainty >= 0

    def test_consensus_positive(self, small_X):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator
        result = IntrinsicDimensionEstimator(methods=["TwoNN", "MLE"]).fit(small_X)
        assert result.consensus > 0

    def test_multiple_methods_dict(self, small_X):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator
        result = IntrinsicDimensionEstimator(methods=["TwoNN", "MLE", "lPCA"]).fit(small_X)
        assert len(result.estimates) == 3

    def test_to_dict(self, small_X):
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator
        result = IntrinsicDimensionEstimator(methods=["TwoNN"]).fit(small_X)
        d = result.to_dict()
        assert "consensus" in d


class TestHubness:
    def test_result_shape(self, small_X):
        from embedkit.analysis.hubness import HubnessAnalyzer
        result = HubnessAnalyzer(k=5).fit(small_X)
        assert result.k_occurrence.shape == (small_X.shape[0],)

    def test_k_occurrence_sum(self, small_X):
        from embedkit.analysis.hubness import HubnessAnalyzer
        k = 5
        result = HubnessAnalyzer(k=k).fit(small_X)
        # sum of k-occurrence == n * k (each point has exactly k neighbors)
        assert result.k_occurrence.sum() == small_X.shape[0] * k

    def test_ratios_in_range(self, small_X):
        from embedkit.analysis.hubness import HubnessAnalyzer
        result = HubnessAnalyzer(k=5).fit(small_X)
        assert 0.0 <= result.antihub_ratio <= 1.0
        assert 0.0 <= result.hub_ratio <= 1.0
        assert 0.0 <= result.hub_contamination <= 1.0

    def test_hub_contamination(self, small_X):
        from embedkit.analysis.hubness import HubnessAnalyzer
        result = HubnessAnalyzer(k=5).fit(small_X)
        assert 0.0 <= result.hub_contamination <= 1.0


class TestGeometry:
    def test_distance_concentration(self, small_X):
        from embedkit.analysis.geometry import DistanceConcentration
        r = DistanceConcentration().fit(small_X)
        assert r.relative_contrast > 0
        assert 0.0 <= r.concentration_ratio <= 1.0
        assert r.distance_histogram.sum() > 0

    def test_isotropy(self, small_X):
        from embedkit.analysis.geometry import IsotropyAnalyzer
        r = IsotropyAnalyzer().fit(small_X)
        n, d = small_X.shape
        assert 1.0 <= r.participation_ratio <= d + 1e-3
        assert 0.0 <= r.isotropy_score <= 1.0
        assert r.effective_rank >= 1.0
        assert len(r.eigenvalue_spectrum) == d

    def test_neighbor_consistency(self, small_X):
        from embedkit.analysis.geometry import NeighborConsistency
        r = NeighborConsistency(k=5, n_perturbations=3).fit(small_X)
        assert 0.0 <= r.mean_consistency <= 1.0
        assert r.std_consistency >= 0.0
        assert len(r.per_perturbation) == 3

    def test_uniformity_negative(self, small_X):
        from embedkit.analysis.geometry import UniformityScore
        r = UniformityScore().fit(small_X)
        assert r.uniformity < 0.0

    def test_uniformity_more_negative_for_spread(self):
        from embedkit.analysis.geometry import UniformityScore
        rng = np.random.default_rng(0)
        X_concentrated = rng.standard_normal((200, 8)).astype(np.float32) * 0.01
        X_spread = rng.standard_normal((200, 8)).astype(np.float32) * 10.0
        u_conc = UniformityScore().fit(X_concentrated).uniformity
        u_spread = UniformityScore().fit(X_spread).uniformity
        assert u_spread < u_conc


class TestKernelDiagnostics:
    def test_fields(self, small_X):
        from embedkit.analysis.kernel import KernelDiagnostics
        r = KernelDiagnostics().fit(small_X)
        assert r.effective_rank >= 1.0
        assert r.sigma > 0.0
        assert r.condition_number >= 1.0

    def test_kernel_alignment_with_labels(self, small_Xy):
        from embedkit.analysis.kernel import KernelDiagnostics
        X, y = small_Xy
        r = KernelDiagnostics().fit(X, y)
        assert r.kernel_alignment is not None
        assert 0.0 <= r.kernel_alignment <= 1.0

    def test_kernel_alignment_none_without_labels(self, small_X):
        from embedkit.analysis.kernel import KernelDiagnostics
        r = KernelDiagnostics().fit(small_X)
        assert r.kernel_alignment is None


class TestEmbedKitAnalyzer:
    def test_report_fields(self, small_X):
        from embedkit.analysis.report import EmbedKitAnalyzer
        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(small_X)
        assert report.suggested_k > 0
        assert report.suggested_target_dim > 0
        assert report.severity in ("low", "medium", "high")
        assert len(report.recommendations) > 0

    def test_to_dict(self, small_X):
        from embedkit.analysis.report import EmbedKitAnalyzer
        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(small_X)
        d = report.to_dict()
        assert "severity" in d
        assert "recommendations" in d

    def test_to_dataframe(self, small_X):
        from embedkit.analysis.report import EmbedKitAnalyzer
        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(small_X)
        df = report.to_dataframe()
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_print_summary(self, small_X, capsys):
        from embedkit.analysis.report import EmbedKitAnalyzer
        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(small_X)
        report.print_summary()
        captured = capsys.readouterr()
        assert "severity" in captured.out.lower()

    def test_input_shape(self, small_X):
        from embedkit.analysis.report import EmbedKitAnalyzer
        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(small_X)
        assert report.input_shape == small_X.shape

    def test_suggested_target_dim_bounds(self, small_X):
        from embedkit.analysis.report import EmbedKitAnalyzer
        report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(small_X)
        _, d = small_X.shape
        assert 4 <= report.suggested_target_dim <= d
