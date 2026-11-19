"""WP 8.4: statistics the null does not calibrate are flagged; z intervals; DANCo's NaN."""

import numpy as np
import pytest

from embedkit.analysis.null import calibrated_statistic
from embedkit.analysis.report import (
    EmbedKitAnalyzer,
    _calibrated_recommendations,
    uncalibrated_statistics,
)

FAST = dict(id_methods=["TwoNN"], id_bootstrap=0, id_scale_curve=False, random_state=0)


def _power_law(n, d, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.arange(1, d + 1) ** (-alpha / 2)
    return (rng.standard_normal((n, d)) * scale).astype(np.float32)


def test_flag_rule():
    assert uncalibrated_statistics("euclidean", 256, False) == {}
    assert set(uncalibrated_statistics("euclidean", 1024, False)) == {"neighbor_consistency"}
    assert set(uncalibrated_statistics("cosine", 32, False)) == {"k_skewness", "antihub_ratio"}
    assert set(uncalibrated_statistics("euclidean", 64, True)) == {
        "antihub_ratio", "distance_cv", "uniformity"}
    with pytest.raises(ValueError):
        uncalibrated_statistics("manhattan", 64, False)


def test_normalised_rows_are_flagged_and_raise_nothing():
    X = _power_law(600, 64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = EmbedKitAnalyzer(n_null_draws=5, **FAST).fit(X)
    flagged = {k for k, c in r.calibration.items() if not c.calibrated}
    assert flagged == {"antihub_ratio", "distance_cv", "uniformity"}
    assert "normalised upstream" in r.calibration["distance_cv"].uncalibrated
    assert not any("antihub" in rec for rec in r.recommendations)
    assert r.n_outside_null == sum(abs(c.z_score) > r.z_threshold
                                   for c in r.calibration.values() if c.calibrated)
    df = r.to_dataframe().set_index("metric")["value"]
    assert not df["distance_cv_calibrated"] and df["k_skewness_calibrated"]
    assert any("Uniformity is not compared" in p for p in r.properties)


def test_unnormalised_rows_at_low_d_are_all_calibrated():
    r = EmbedKitAnalyzer(n_null_draws=5, **FAST).fit(_power_law(600, 64))
    assert all(c.calibrated for c in r.calibration.values())


def test_cosine_flags_hubness_statistics():
    r = EmbedKitAnalyzer(metric="cosine", n_null_draws=5, **FAST).fit(_power_law(600, 64))
    flagged = {k for k, c in r.calibration.items() if not c.calibrated}
    assert flagged == {"k_skewness", "antihub_ratio"}


def test_z_interval_contains_z_and_honours_the_floor():
    null = np.array([0.10, 0.12, 0.11, 0.13, 0.09, 0.12, 0.10, 0.11])
    c = calibrated_statistic(0.2, null, "gaussian", 100)
    lo, hi = c.z_interval
    assert lo <= c.z_score <= hi and lo < hi
    zero = calibrated_statistic(0.02, np.zeros(10), "gaussian", 100, std_floor=0.01)
    assert zero.null_std == 0.01 and zero.z_score == pytest.approx(2.0)
    assert zero.z_interval == pytest.approx((2.0, 2.0))
    assert np.isnan(calibrated_statistic(1.0, np.array([0.0, 1.0]), "gaussian", 10)
                    .z_interval[0])


def test_near_threshold_recommendation_says_so():
    rng = np.random.default_rng(0)
    null = rng.normal(0.10, 0.01, 20)

    def z_of(obs):
        return calibrated_statistic(obs, null, "gaussian", 100)

    near = z_of(null.mean() + 3.3 * null.std(ddof=1))
    far = z_of(null.mean() + 20 * null.std(ddof=1))
    assert near.z_interval[0] <= 3.0 < near.z_score
    rec_near = _calibrated_recommendations({"k_skewness": near}, 3.0)
    rec_far = _calibrated_recommendations({"k_skewness": far}, 3.0)
    assert len(rec_near) == 1 and "near the threshold" in rec_near[0]
    assert len(rec_far) == 1 and "near the threshold" not in rec_far[0]


def _cube2_in_256d(n, seed):
    """E3's ``cube2`` cell: a 2-d uniform cube, standardised, mapped linearly into D = 256."""
    Z = np.random.default_rng(seed).random((n, 2))
    Z = (Z - Z.mean(0)) / Z.std(0)
    Q, _ = np.linalg.qr(np.random.default_rng(seed + 10_000).standard_normal((256, 2)))
    return Z @ Q.T


def test_danco_is_finite_on_a_linearly_embedded_cube():
    from embedkit.analysis.intrinsic_dim import _build_estimator

    with np.errstate(all="ignore"):
        dim = _build_estimator("DANCo", 0).fit(_cube2_in_256d(1000, 0)).dimension_
    assert np.isfinite(dim) and 1.5 < dim < 3.0


def test_danco_matches_skdim_where_skdim_is_finite():
    import skdim

    from embedkit.analysis.intrinsic_dim import _build_estimator

    X = np.random.default_rng(0).standard_normal((400, 5)) @ \
        np.random.default_rng(1).standard_normal((5, 40))
    with np.errstate(all="ignore"):
        ref = skdim.id.DANCo(random_state=0).fit(X).dimension_
        ours = _build_estimator("DANCo", 0).fit(X).dimension_
    assert np.isfinite(ref) and ours == ref
