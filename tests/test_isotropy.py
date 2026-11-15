"""WP 4.3: isotropy and spectrum statistics, reported as properties."""

import numpy as np
import pytest

from embedkit.analysis.geometry import IsotropyAnalyzer


def _power_law(n, d, alpha, seed=0, mean=0.0):
    rng = np.random.default_rng(seed)
    lam = np.arange(1, d + 1, dtype=float) ** -alpha
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return ((rng.standard_normal((n, d)) * np.sqrt(lam)) @ Q.T + mean).astype(np.float32)


def _isoscore_published(X):
    """Rudman et al. (2022), Algorithm 1, step by step."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = np.var(Xc @ Vt.T, axis=0, ddof=1)
    d = X.shape[1]
    var = np.concatenate([var, np.zeros(d - len(var))])
    vhat = np.sqrt(d) * var / np.linalg.norm(var)
    delta = np.linalg.norm(vhat - 1) / np.sqrt(2 * (d - np.sqrt(d)))
    phi = (d - delta**2 * (d - np.sqrt(d))) ** 2 / d**2
    return (d * phi - 1) / (d - 1)


class TestIsoScore:
    @pytest.mark.parametrize("n,d,alpha", [(2000, 64, 0.0), (2000, 64, 1.0), (300, 512, 0.5)])
    def test_matches_the_published_algorithm(self, n, d, alpha):
        X = _power_law(n, d, alpha)
        assert IsotropyAnalyzer().fit(X).isoscore == pytest.approx(_isoscore_published(X),
                                                                   abs=1e-6)

    def test_isotropic_data_with_many_rows_score_near_one(self):
        assert IsotropyAnalyzer().fit(_power_law(5000, 32, 0.0)).isoscore > 0.95


class TestSpectrumAlpha:
    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    @pytest.mark.parametrize("n,d", [(2000, 128), (300, 512)])
    def test_recovers_the_exponent(self, alpha, n, d):
        r = IsotropyAnalyzer().fit(_power_law(n, d, alpha))
        assert r.spectrum_alpha == pytest.approx(alpha, abs=0.1)

    def test_flat_spectrum_has_a_small_exponent(self):
        assert IsotropyAnalyzer().fit(_power_law(2000, 128, 0.0)).spectrum_alpha < 0.4

    def test_too_few_eigenvalues(self):
        assert np.isnan(IsotropyAnalyzer().fit(_power_law(50, 3, 1.0)).spectrum_alpha)


class TestShares:
    def test_principal_component_shares(self):
        r = IsotropyAnalyzer().fit(_power_law(5000, 64, 2.0))
        lam = np.arange(1, 65, dtype=float) ** -2.0
        assert r.top_k_variance_share[1] == pytest.approx(lam[0] / lam.sum(), abs=0.02)
        assert r.top_k_variance_share[1] < r.top_k_variance_share[3] < r.top_k_variance_share[10]

    def test_a_rogue_coordinate(self):
        # One coordinate with 30x the standard deviation of the others: the "rogue dimension"
        # holds most of the variance by coordinate (and here also by principal component).
        X = np.random.default_rng(0).standard_normal((2000, 64)).astype(np.float32)
        X[:, 7] *= 30
        r = IsotropyAnalyzer().fit(X)
        assert r.top_k_coordinate_share[1] > 0.9
        rotated = IsotropyAnalyzer().fit(_power_law(2000, 64, 0.0))
        assert rotated.top_k_coordinate_share[1] < 0.05


class TestMeanDirection:
    def test_centred_data(self):
        r = IsotropyAnalyzer().fit(_power_law(2000, 64, 0.0))
        assert r.mean_vector_ratio < 0.05 and abs(r.mean_cosine) < 0.01

    def test_shared_offset(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal(64)
        u *= 10 / np.linalg.norm(u)  # a mean 10 times the per-row noise norm scale (~8)
        X = _power_law(2000, 64, 0.0, mean=u * 3)
        r = IsotropyAnalyzer().fit(X)
        assert r.mean_vector_ratio > 0.9 and r.mean_cosine > 0.8

    def test_mean_cosine_matches_brute_force(self):
        X = _power_law(60, 8, 0.5, mean=0.3)
        Z = X / np.linalg.norm(X, axis=1, keepdims=True)
        C = Z @ Z.T
        brute = (C.sum() - np.trace(C)) / (60 * 59)
        assert IsotropyAnalyzer().fit(X).mean_cosine == pytest.approx(brute, abs=1e-6)


class TestReportProperties:
    def test_anisotropy_is_a_property_not_a_recommendation(self):
        from embedkit.analysis.report import EmbedKitAnalyzer

        X = _power_law(1000, 64, 2.0)  # strongly anisotropic: corrected PR well under 0.1
        report = EmbedKitAnalyzer(id_methods=["TwoNN"], calibrate=False).fit(X)
        assert report.geometry.isotropy.participation_ratio_normalized < 0.1
        assert not any(w in r.lower() for r in report.recommendations
                       for w in ("anisotrop", "whiten", "participation", "alignment loss"))
        assert len(report.properties) == 3
        assert report.properties[0].startswith("Spectrum:")

    def test_uniformity_is_a_property_when_calibrated(self):
        from embedkit.analysis.report import EmbedKitAnalyzer

        report = EmbedKitAnalyzer(id_methods=["TwoNN"], n_null_draws=5, n_calibration=300).fit(
            _power_law(300, 16, 1.0))
        assert len(report.properties) == 4 and "uniformity" in report.properties[3]
        assert not any("uniformity" in r for r in report.recommendations)
