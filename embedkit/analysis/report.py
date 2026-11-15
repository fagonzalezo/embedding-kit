"""EmbedKitAnalyzer: unified analysis report with recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from embedkit.analysis._base import BaseAnalyzer, BaseResult
from embedkit.analysis.geometry import (
    DistanceConcentration,
    DistanceConcentrationResult,
    IsotropyAnalyzer,
    IsotropyResult,
    NeighborConsistency,
    NeighborConsistencyResult,
    UniformityResult,
    UniformityScore,
)
from embedkit.analysis.hubness import HubnessAnalyzer, HubnessResult
from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator, IntrinsicDimensionResult
from embedkit.analysis.kernel import KernelDiagnostics, KernelDiagnosticsResult
from embedkit.analysis.null import CalibratedStatistic, NullModel, calibrate
from embedkit.utils.neighbors import knn

# The statistics compared with a null model when EmbedKitAnalyzer(calibrate=True).
CALIBRATED_STATISTICS = ("k_skewness", "distance_cv", "neighbor_consistency", "uniformity")


@dataclass(frozen=True)
class GeometryBundle(BaseResult):
    distance_concentration: DistanceConcentrationResult
    isotropy: IsotropyResult
    neighbor_consistency: NeighborConsistencyResult
    uniformity: UniformityResult


@dataclass(frozen=True)
class EmbedKitReport(BaseResult):
    intrinsic_dim: IntrinsicDimensionResult
    hubness: HubnessResult
    geometry: GeometryBundle
    kernel: KernelDiagnosticsResult
    suggested_k: int
    suggested_sigma: float
    suggested_target_dim: int
    recommendations: list[str]
    input_shape: tuple[int, int]
    calibration: dict[str, CalibratedStatistic] | None = None
    n_outside_null: int | None = None
    z_threshold: float = 3.0

    def to_dataframe(self) -> pd.DataFrame:
        dc = self.geometry.distance_concentration
        nc = self.geometry.neighbor_consistency
        rows = [
            *[(("ID", m, v)) for m, v in self.intrinsic_dim.estimates.items()],
            *[("ID", f"{m}_se", s) for m, s in self.intrinsic_dim.standard_errors.items()],
            ("ID", "between_method_spread", self.intrinsic_dim.between_method_spread),
            ("ID", "reference_method", self.intrinsic_dim.reference_method),
            ("ID", "scale_plateau", self.intrinsic_dim.scale_plateau),
            ("Hubness", "k_skewness", self.hubness.k_skewness),
            ("Hubness", "robinhood_index", self.hubness.robinhood_index),
            ("Hubness", "antihub_ratio", self.hubness.antihub_ratio),
            ("Hubness", "hub_ratio", self.hubness.hub_ratio),
            ("Hubness", "hub_contamination", self.hubness.hub_contamination),
            ("Geometry", "relative_contrast", dc.relative_contrast),
            ("Geometry", "concentration_ratio", dc.concentration_ratio),
            ("Geometry", "distance_cv", self.geometry.distance_concentration.distance_cv),
            (
                "Geometry",
                "relative_contrast_per_query",
                self.geometry.distance_concentration.relative_contrast_per_query,
            ),
            ("Geometry", "participation_ratio", self.geometry.isotropy.participation_ratio),
            (
                "Geometry",
                "participation_ratio_normalized",
                self.geometry.isotropy.participation_ratio_normalized,
            ),
            ("Geometry", "effective_rank", self.geometry.isotropy.effective_rank),
            ("Geometry", "isotropy_score", self.geometry.isotropy.isotropy_score),
            ("Geometry", "neighbor_consistency", nc.mean_consistency),
            ("Geometry", "uniformity", self.geometry.uniformity.uniformity),
            ("Kernel", "effective_rank", self.kernel.effective_rank),
            ("Kernel", "spectral_gap", self.kernel.spectral_gap),
            ("Kernel", "condition_number", self.kernel.condition_number),
            ("Kernel", "sigma", self.kernel.sigma),
            ("Kernel", "centered_kernel_alignment", self.kernel.centered_kernel_alignment),
            ("Summary", "suggested_k", self.suggested_k),
            ("Summary", "suggested_sigma", self.suggested_sigma),
            ("Summary", "suggested_target_dim", self.suggested_target_dim),
            ("Summary", "n_outside_null", self.n_outside_null),
        ]
        for name, c in (self.calibration or {}).items():
            rows += [
                ("Calibration", f"{name}_observed", c.observed),
                ("Calibration", f"{name}_null_mean", c.null_mean),
                ("Calibration", f"{name}_null_std", c.null_std),
                ("Calibration", f"{name}_z", c.z_score),
            ]
        return pd.DataFrame(rows, columns=["category", "metric", "value"])

    def print_summary(self) -> None:
        n, d = self.input_shape
        dc = self.geometry.distance_concentration
        print(f"\n{'='*60}")
        print(f"  EmbedKit Analysis Report  ({n} samples × {d} dims)")
        print(f"{'='*60}")
        print("\n[Intrinsic Dimension]")
        id_res = self.intrinsic_dim
        for m, v in id_res.estimates.items():
            lo, hi = id_res.intervals.get(m, (float("nan"), float("nan")))
            ci = f"  [{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else ""
            print(f"  {m:<14}: {v:.2f}{ci}")
        if id_res.scale_curve:
            curve = ", ".join(f"{v:.1f} (n={k})" for k, v in id_res.scale_curve.items())
            print(f"  TwoNN by scale: {curve}; plateau: {id_res.scale_plateau}")
        print(f"  (intervals: sampling error only; k-NN estimators read low at high ID; "
              f"target dim from {id_res.reference_method})")
        print("\n[Hubness]")
        print(f"  k-skewness       : {self.hubness.k_skewness:.3f}")
        print(f"  Robin Hood index : {self.hubness.robinhood_index:.3f}")
        print(f"  hub ratio        : {self.hubness.hub_ratio:.3f}")
        print(f"  antihub ratio    : {self.hubness.antihub_ratio:.3f}")
        print(f"  hub contamination: {self.hubness.hub_contamination:.3f}")
        print("\n[Geometry]")
        print(f"  participation_ratio : {self.geometry.isotropy.participation_ratio:.2f}")
        pr_norm = self.geometry.isotropy.participation_ratio_normalized
        print(f"  PR (normalized)     : {pr_norm:.3f}")
        print(f"  isotropy_score      : {self.geometry.isotropy.isotropy_score:.3f}")
        print(f"  relative_contrast   : {dc.relative_contrast:.3f}")
        print(f"  distance_cv         : {self.geometry.distance_concentration.distance_cv:.4f}")
        print(f"  neighbor_consistency: {self.geometry.neighbor_consistency.mean_consistency:.3f}")
        print(f"  uniformity          : {self.geometry.uniformity.uniformity:.4f}")
        print("\n[Kernel]")
        print(f"  sigma            : {self.kernel.sigma:.4f}")
        print(f"  effective_rank   : {self.kernel.effective_rank:.2f}")
        print(f"  spectral_gap     : {self.kernel.spectral_gap:.3f}")
        print(f"  condition_number : {self.kernel.condition_number:.2e}")
        if self.kernel.centered_kernel_alignment is not None:
            print(f"  centered kernel alignment: {self.kernel.centered_kernel_alignment:.4f}")
        if self.calibration:
            c0 = next(iter(self.calibration.values()))
            print(f"\n[Calibration]  {c0.null_model} null, {c0.n_draws} draws of "
                  f"{c0.n_samples} rows")
            for name, c in self.calibration.items():
                print(f"  {name:<20}: {c.observed:.4f}  null {c.null_mean:.4f} "
                      f"± {c.null_std:.4f}  z = {c.z_score:+.1f}")
            print(f"  outside ±{self.z_threshold:g}: {self.n_outside_null} of "
                  f"{len(self.calibration)}")
        print("\n[Recommendations]")
        for rec in self.recommendations:
            print(f"  • {rec}")
        print("\n[Auto-config]")
        print(f"  suggested_k          : {self.suggested_k}")
        print(f"  suggested_sigma      : {self.suggested_sigma:.4f}")
        print(f"  suggested_target_dim : {self.suggested_target_dim}")
        print(f"{'='*60}\n")


class EmbedKitAnalyzer(BaseAnalyzer):
    """Run every analyzer and summarise the results in an :class:`EmbedKitReport`.

    calibrate:
        Compare k-skewness, ``distance_cv``, neighbour consistency and uniformity with their
        distribution under ``null_model`` (see :mod:`embedkit.analysis.null`), in
        ``report.calibration``. The comparison runs on a subsample of ``n_calibration`` rows
        with the report's ``k`` and metric, against ``n_null_draws`` null data sets of the same
        size, so it costs ``n_null_draws + 1`` evaluations of the four statistics on that
        subsample. The recommendations for these four statistics come from the z-scores
        (``|z| > z_threshold`` in the direction each recommendation names), and
        ``report.n_outside_null`` counts the statistics with ``|z| > z_threshold``. With
        ``calibrate=False`` the report makes no recommendation about them.
    id_bootstrap, id_scale_curve:
        Passed to :class:`IntrinsicDimensionEstimator` as ``n_bootstrap`` and ``scale_curve``.
        ``suggested_target_dim`` is ``1.5 x`` the reference estimate (TwoNN, or the first method
        of ``id_methods`` that succeeded), clipped to ``[max(4, ID), d]``; ``max(2, d // 4)``
        stands in for the ID when every method failed.
    """

    def __init__(
        self,
        k: int | None = None,
        id_methods: list[str] | None = None,
        metric: str = "euclidean",
        n_max: int = 20_000,
        d_max: int = 2_000,
        random_state: int | None = 42,
        calibrate: bool = True,
        null_model: NullModel = "gaussian",
        n_null_draws: int = 20,
        n_calibration: int = 2_000,
        z_threshold: float = 3.0,
        id_bootstrap: int = 20,
        id_scale_curve: bool = True,
    ):
        self.k = k
        self.id_methods = id_methods or ["TwoNN", "MLE", "lPCA", "MOM"]
        self.metric = metric
        self.n_max = n_max
        self.d_max = d_max
        self.random_state = random_state
        self.calibrate = calibrate
        self.null_model = null_model
        self.n_null_draws = n_null_draws
        self.n_calibration = n_calibration
        self.z_threshold = z_threshold
        self.id_bootstrap = id_bootstrap
        self.id_scale_curve = id_scale_curve

    def fit(self, X, y=None) -> EmbedKitReport:
        X = self._prepare(X, min_n=5)
        n, d = X.shape
        k = self.k or max(5, round(np.sqrt(min(n, self.n_max)) / 2))

        id_est = IntrinsicDimensionEstimator(
            methods=self.id_methods, n_max=min(5_000, self.n_max),
            n_bootstrap=self.id_bootstrap, scale_curve=self.id_scale_curve,
            random_state=self.random_state,
        )
        id_result = id_est.fit(X)

        # The unperturbed kNN graph is computed once, on the same subsample, and shared by the
        # analyzers that need it.
        if n > self.n_max:
            rng = np.random.default_rng(self.random_state)
            X_graph = X[rng.choice(n, self.n_max, replace=False)]
        else:
            X_graph = X
        _, graph = knn(X_graph, k, metric=self.metric)

        hub = HubnessAnalyzer(
            k=k, metric=self.metric, subsample=self.n_max, random_state=self.random_state
        )
        hub_result = hub.fit(X_graph, neighbor_index=graph)

        dc = DistanceConcentration(random_state=self.random_state)
        iso = IsotropyAnalyzer(d_max=self.d_max, random_state=self.random_state)
        nc = NeighborConsistency(
            k=k, metric=self.metric, subsample=self.n_max, random_state=self.random_state
        )
        uni = UniformityScore(random_state=self.random_state)
        geo_bundle = GeometryBundle(
            distance_concentration=dc.fit(X),
            isotropy=iso.fit(X),
            neighbor_consistency=nc.fit(X_graph, neighbor_index=graph),
            uniformity=uni.fit(X),
        )

        kernel = KernelDiagnostics(random_state=self.random_state)
        kernel_result = kernel.fit(X, y)

        suggested_k = k
        suggested_sigma = kernel_result.sigma
        ref_id = id_result.reference_estimate
        if np.isnan(ref_id):
            ref_id = max(2, d // 4)
        suggested_target_dim = int(np.clip(round(1.5 * ref_id), max(4, int(ref_id)), d))

        calibration = self._calibrate(X, k) if self.calibrate else None
        n_outside = (
            None if calibration is None
            else int(sum(abs(c.z_score) > self.z_threshold for c in calibration.values()))
        )
        recommendations = _recommendations(
            id_result, hub_result, geo_bundle, calibration, d, self.z_threshold
        )

        return EmbedKitReport(
            intrinsic_dim=id_result,
            hubness=hub_result,
            geometry=geo_bundle,
            kernel=kernel_result,
            suggested_k=suggested_k,
            suggested_sigma=suggested_sigma,
            suggested_target_dim=suggested_target_dim,
            recommendations=recommendations,
            input_shape=(n, d),
            calibration=calibration,
            n_outside_null=n_outside,
            z_threshold=self.z_threshold,
        )

    def _calibrate(self, X: np.ndarray, k: int) -> dict[str, CalibratedStatistic]:
        """The calibrated statistics on a subsample of ``X``, against the null model."""
        n = X.shape[0]
        if n > self.n_calibration:
            rng = np.random.default_rng(self.random_state)
            X = X[np.sort(rng.choice(n, self.n_calibration, replace=False))]
        k = min(k, X.shape[0] - 1)
        return calibrate(
            lambda Z: calibration_statistics(Z, k, self.metric, self.random_state),
            X,
            model=self.null_model,
            n_draws=self.n_null_draws,
            random_state=self.random_state,
        )


def calibration_statistics(
    X: np.ndarray, k: int, metric: str = "euclidean", random_state: int | None = 42
) -> dict[str, float]:
    """The statistics in ``CALIBRATED_STATISTICS``, computed as the report computes them.

    One kNN graph is shared by k-skewness and neighbour consistency; ``X`` is used whole for
    both, and ``DistanceConcentration`` and ``UniformityScore`` subsample it as usual.
    """
    _, graph = knn(X, k, metric=metric)
    return {
        "k_skewness": HubnessAnalyzer(k=k, metric=metric).fit(
            X, neighbor_index=graph
        ).k_skewness,
        "distance_cv": DistanceConcentration(random_state=random_state).fit(X).distance_cv,
        "neighbor_consistency": NeighborConsistency(
            k=k, metric=metric, random_state=random_state
        ).fit(X, neighbor_index=graph).mean_consistency,
        "uniformity": UniformityScore(random_state=random_state).fit(X).uniformity,
    }


def _recommendations(
    id_result, hub_result, geo_bundle, calibration, d, z_threshold: float = 3.0
) -> list[str]:
    """Recommendations from the report.

    Hubness, distance concentration, neighbour consistency and uniformity are judged only
    through ``calibration`` (z-scores against the null model); without it nothing is said about
    them. The antihub, ID / D and participation-ratio rules are fixed thresholds that WP 4.2
    and 4.3 revisit.
    """
    recs = []

    if calibration is not None:
        recs.extend(_calibrated_recommendations(calibration, z_threshold))

    if hub_result.antihub_ratio > 0.3:
        recs.append(
            "Many antihubs (antihub_ratio > 0.3): a significant fraction of points "
            "are never retrieved as neighbors."
        )

    # ID / D ratio (curse of dimensionality)
    ref_id = id_result.reference_estimate
    if not np.isnan(ref_id) and d > 0:
        id_d_ratio = ref_id / d
        if id_d_ratio < 0.1:
            recs.append(
                f"Low intrinsic-to-ambient ratio ({id_result.reference_method} ID/D ≈ "
                f"{id_d_ratio:.2f}): "
                "the embedding is very sparse — strong dimensionality reduction is advised."
            )
        elif id_d_ratio < 0.3:
            recs.append(
                f"Moderate intrinsic-to-ambient ratio ({id_result.reference_method} ID/D ≈ "
                f"{id_d_ratio:.2f}): "
                "dimensionality reduction may improve downstream tasks."
            )

    # Isotropy / participation ratio (normalised by d, so it does not depend on the dimension)
    pr = geo_bundle.isotropy.participation_ratio_normalized
    if pr < 0.1:
        recs.append(
            f"Low participation ratio ({pr:.2f} of {d} dimensions): embeddings are highly "
            "anisotropic. Consider whitening or applying an alignment loss."
        )
    elif pr < 0.3:
        recs.append(
            f"Moderate anisotropy (normalized participation ratio={pr:.2f}): "
            "alignment losses or PCA whitening may help."
        )

    if not recs:
        recs.append("No major issues detected. Embedding geometry looks healthy.")
    return recs


def _calibrated_recommendations(calibration, z_threshold: float) -> list[str]:
    """One recommendation per calibrated statistic that is beyond the null in its direction."""

    def describe(c) -> str:
        return (f"{c.observed:.3g} against {c.null_mean:.3g} ± {c.null_std:.2g} under the "
                f"{c.null_model} null, z = {c.z_score:+.1f}")

    recs = []
    c = calibration.get("k_skewness")
    if c is not None and c.z_score > z_threshold:
        recs.append(
            f"More hubness than the null (k-skewness {describe(c)}): a few points dominate "
            "the neighbour lists. Try a hubness-reducing neighbour search such as "
            "embedkit.hubness_reduction.MutualProximity."
        )
    c = calibration.get("distance_cv")
    if c is not None and c.z_score < -z_threshold:
        recs.append(
            f"Distances are more concentrated than the null (distance_cv {describe(c)}): "
            "nearest and farthest neighbours are harder to tell apart than in a Gaussian with "
            "the same covariance spectrum."
        )
    c = calibration.get("neighbor_consistency")
    if c is not None and c.z_score < -z_threshold:
        recs.append(
            f"The kNN graph is less stable under small perturbations than the null "
            f"(neighbor_consistency {describe(c)})."
        )
    c = calibration.get("uniformity")
    if c is not None and c.z_score > z_threshold:
        recs.append(
            f"Directions are more clustered on the sphere than the null "
            f"(uniformity {describe(c)}). This describes the geometry; it is not a defect."
        )
    return recs
