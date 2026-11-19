"""EmbedKitAnalyzer: unified analysis report with recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
from embedkit.utils.validation import trusted_input

# The statistics compared with a null model when EmbedKitAnalyzer(calibrate=True).
CALIBRATED_STATISTICS = (
    "k_skewness", "antihub_ratio", "distance_cv", "neighbor_consistency", "uniformity"
)

# Rows whose norms differ by less than this (relative to the median norm) count as already
# normalised: the calibration null then normalises its draws to that norm too.
_UNIT_NORM_RTOL = 1e-3

# The metrics every analyzer of the report can honour (D28): Euclidean, and cosine through
# L2-normalised rows.
METRICS = ("euclidean", "cosine")

# WP 8.4: settings in which a calibrated statistic failed the E3 bar (z of data drawn from the
# null family inside +-3 in at least 95% of cells, WP 4.1 and 4.5). Above this dimension
# neighbour consistency under Euclidean distance is flagged (E3 tested d = 32, 256 and 1,024).
_UNCALIBRATED_ABOVE_D = 256
_NORMALISED_ROWS = (
    "rows normalised upstream (a common row norm): with an anisotropic spectrum the null, "
    "fitted to rows already on the sphere, is more isotropic than the data (E3 i^-1 spectrum: "
    "inside +-3 in {share} of null cells)")


def uncalibrated_statistics(metric: str, d: int, normalised: bool) -> dict[str, str]:
    """The calibrated statistics whose null is not calibrated in this setting, with the reason.

    From E3 (``benchmarks/results/e3/summary_gauss.csv``): data drawn from the null family,
    whose z-scores should fall inside +-3; the bar is 95% of cells (WP 8.4).

    - Rows with a common norm (``normalised``): the antihub ratio (51% of cells inside),
      ``distance_cv`` (0%) and uniformity (0%) under either metric, for rows normalised with
      an ``i^-1`` spectrum. With a flat spectrum such rows were calibrated, but the report
      cannot tell the two apart reliably, so it flags all normalised rows (the user's
      decision, 2026-11-18: flag only, no new null).
    - ``metric="cosine"``: k-skewness (76% inside with an ``i^-0.5`` spectrum, 40% at
      d = 1,024) and the antihub ratio (89% over all cells, 0% for spiked spectra at
      d = 1,024).
    - ``metric="euclidean"`` and ``d > 256``: neighbour consistency (89% at d = 1,024, 40% for
      spiked spectra; 99% at d up to 256).

    None of these feeds a recommendation except the antihub ratio under Euclidean distance on
    normalised rows: that recommendation is not made.
    """
    _check_metric(metric)
    out = {}
    if metric == "cosine":
        out["k_skewness"] = ("not calibrated under cosine: E3 null data were inside +-3 in 76% "
                             "of cells with an i^-0.5 spectrum (40% at d = 1,024)")
        out["antihub_ratio"] = ("not calibrated under cosine: E3 null data were inside +-3 in "
                                "89% of cells (0% for spiked spectra at d = 1,024)")
    elif d > _UNCALIBRATED_ABOVE_D:
        out["neighbor_consistency"] = (
            f"not calibrated under Euclidean distance above d = {_UNCALIBRATED_ABOVE_D}: E3 null "
            "data were inside +-3 in 89% of cells at d = 1,024 (40% for spiked spectra)")
    if normalised:
        for name, share in (("antihub_ratio", "51%"), ("distance_cv", "0%"),
                            ("uniformity", "0%")):
            out[name] = _NORMALISED_ROWS.format(share=share)
    return out


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(
            f"metric={metric!r} is not supported: the report honours only {METRICS} in every "
            "analyzer (intrinsic dimension, distance concentration and the kernel have no other "
            "version). HubnessAnalyzer and NeighborConsistency accept other scikit-learn "
            "metrics when used on their own."
        )


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Rows divided by their Euclidean norm. Raises ``ValueError`` on a zero row, which has no
    cosine geometry."""
    X = np.asarray(X)
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    zero = norms == 0
    if zero.any():
        raise ValueError(
            f"metric='cosine': {int(zero.sum())} row(s) have zero norm and no direction; "
            "remove them or use metric='euclidean'."
        )
    return (X / norms[:, None]).astype(X.dtype, copy=False)


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
    k: int
    suggested_target_dim: int
    recommendations: list[str]
    input_shape: tuple[int, int]
    calibration: dict[str, CalibratedStatistic] | None = None
    n_outside_null: int | None = None
    z_threshold: float = 3.0
    properties: list[str] = field(default_factory=list)
    metric: str = "euclidean"
    calibration_k: int | None = None

    def to_dataframe(self) -> pd.DataFrame:
        dc = self.geometry.distance_concentration
        nc = self.geometry.neighbor_consistency
        rows = [
            *[(("ID", m, v)) for m, v in self.intrinsic_dim.estimates.items()],
            *[("ID", f"{m}_se", s) for m, s in self.intrinsic_dim.standard_errors.items()],
            ("ID", "between_method_spread", self.intrinsic_dim.between_method_spread),
            ("ID", "reference_method", self.intrinsic_dim.reference_method),
            ("ID", "scale_max_deviation", self.intrinsic_dim.scale_max_deviation),
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
            ("Geometry", "isoscore", self.geometry.isotropy.isoscore),
            *[("Geometry", f"top_{k}_variance_share", v)
              for k, v in self.geometry.isotropy.top_k_variance_share.items()],
            *[("Geometry", f"top_{k}_coordinate_share", v)
              for k, v in self.geometry.isotropy.top_k_coordinate_share.items()],
            ("Geometry", "spectrum_alpha", self.geometry.isotropy.spectrum_alpha),
            ("Geometry", "mean_vector_ratio", self.geometry.isotropy.mean_vector_ratio),
            ("Geometry", "mean_cosine", self.geometry.isotropy.mean_cosine),
            ("Geometry", "neighbor_consistency", nc.mean_consistency),
            ("Geometry", "uniformity", self.geometry.uniformity.uniformity),
            ("Kernel", "effective_rank", self.kernel.effective_rank),
            ("Kernel", "spectral_gap", self.kernel.spectral_gap),
            ("Kernel", "top_k_eigen_ratio", self.kernel.top_k_eigen_ratio),
            ("Kernel", "sigma", self.kernel.sigma),
            ("Kernel", "centered_kernel_alignment", self.kernel.centered_kernel_alignment),
            ("Summary", "k", self.k),
            ("Summary", "suggested_target_dim", self.suggested_target_dim),
            ("Summary", "n_outside_null", self.n_outside_null),
            ("Summary", "metric", self.metric),
            ("Summary", "calibration_k", self.calibration_k),
        ]
        for name, c in (self.calibration or {}).items():
            rows += [
                ("Calibration", f"{name}_observed", c.observed),
                ("Calibration", f"{name}_null_mean", c.null_mean),
                ("Calibration", f"{name}_null_std", c.null_std),
                ("Calibration", f"{name}_z", c.z_score),
                ("Calibration", f"{name}_z_lo", c.z_interval[0]),
                ("Calibration", f"{name}_z_hi", c.z_interval[1]),
                ("Calibration", f"{name}_calibrated", c.calibrated),
            ]
        return pd.DataFrame(rows, columns=["category", "metric", "value"])

    def print_summary(self) -> None:
        n, d = self.input_shape
        dc = self.geometry.distance_concentration
        print(f"\n{'='*60}")
        print(f"  EmbedKit Analysis Report  ({n} samples × {d} dims, {self.metric})")
        print(f"{'='*60}")
        if self.metric == "cosine":
            print("  Every statistic below is computed on the L2-normalised rows.")
        print("\n[Intrinsic Dimension]")
        id_res = self.intrinsic_dim
        for m, v in id_res.estimates.items():
            lo, hi = id_res.intervals.get(m, (float("nan"), float("nan")))
            ci = f"  [{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else ""
            print(f"  {m:<14}: {v:.2f}{ci}")
        if id_res.scale_curve:
            curve = ", ".join(f"{v:.1f} (n={k})" for k, v in id_res.scale_curve.items())
            print(f"  TwoNN by scale: {curve}; max deviation: "
                  f"{id_res.scale_max_deviation:.0%}")
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
        iso = self.geometry.isotropy
        print(f"  isotropy_score      : {iso.isotropy_score:.3f}")
        print(f"  IsoScore            : {iso.isoscore:.3f}")
        print(f"  spectrum alpha      : {iso.spectrum_alpha:.2f}")
        shares = " / ".join(f"{v:.2f}" for v in iso.top_k_variance_share.values())
        print(f"  top 1/3/10 PC share : {shares}")
        shares = " / ".join(f"{v:.2f}" for v in iso.top_k_coordinate_share.values())
        print(f"  top 1/3/10 coords   : {shares}")
        print(f"  mean vector ratio   : {iso.mean_vector_ratio:.3f}")
        print(f"  mean cosine         : {iso.mean_cosine:.3f}")
        print(f"  relative_contrast   : {dc.relative_contrast:.3f}")
        print(f"  distance_cv         : {self.geometry.distance_concentration.distance_cv:.4f}")
        print(f"  neighbor_consistency: {self.geometry.neighbor_consistency.mean_consistency:.3f}")
        print(f"  uniformity          : {self.geometry.uniformity.uniformity:.4f}")
        print("\n[Kernel]")
        print(f"  sigma            : {self.kernel.sigma:.4f}")
        print(f"  effective_rank   : {self.kernel.effective_rank:.2f}  (centred kernel)")
        print(f"  spectral_gap     : {self.kernel.spectral_gap:.3f}")
        print(f"  top_k_eigen_ratio: {self.kernel.top_k_eigen_ratio:.3g}")
        if self.kernel.centered_kernel_alignment is not None:
            print(f"  centered kernel alignment: {self.kernel.centered_kernel_alignment:.4f}")
        if self.calibration:
            c0 = next(iter(self.calibration.values()))
            print(f"\n[Calibration]  {c0.null_model} null, {c0.n_draws} draws of "
                  f"{c0.n_samples} rows, k = {self.calibration_k}")
            for name, c in self.calibration.items():
                lo, hi = c.z_interval
                flag = "" if c.calibrated else "  (not calibrated)"
                print(f"  {name:<20}: {c.observed:.4f}  null {c.null_mean:.4f} "
                      f"± {c.null_std:.4f}  z = {c.z_score:+.1f} [{lo:+.1f}, {hi:+.1f}]{flag}")
            n_cal = sum(c.calibrated for c in self.calibration.values())
            print(f"  outside ±{self.z_threshold:g}: {self.n_outside_null} of {n_cal} "
                  "calibrated")
            for name, c in self.calibration.items():
                if not c.calibrated:
                    print(f"  {name} not calibrated: {c.uncalibrated}")
        print("\n[Properties]")
        for prop in self.properties:
            print(f"  • {prop}")
        print("\n[Recommendations]")
        for rec in self.recommendations:
            print(f"  • {rec}")
        print("\n[Settings]")
        print(f"  k                    : {self.k}")
        print(f"  suggested_target_dim : {self.suggested_target_dim}")
        print(f"{'='*60}\n")


class EmbedKitAnalyzer(BaseAnalyzer):
    """Run every analyzer and summarise the results in an :class:`EmbedKitReport`.

    calibrate:
        Compare k-skewness, the antihub ratio, ``distance_cv``, neighbour consistency and
        uniformity with their distribution under ``null_model`` (see
        :mod:`embedkit.analysis.null`), in ``report.calibration``. The comparison runs on a
        subsample of ``n_calibration`` rows with the report's metric, against ``n_null_draws``
        null data sets of the same size, so it costs ``n_null_draws + 1`` evaluations of the
        five statistics on that subsample. Its ``k`` (``report.calibration_k``) is ``k`` when
        given, and otherwise the default rule applied to the calibration rows,
        ``max(5, round(sqrt(rows) / 2))``, so that the z-scores depend on the calibration
        sample only and not on the size of ``X``: reusing the report's ``k`` (71 at 20,000
        rows) made the neighbour-consistency z on the same 2,000 rows grow from 9 to 23 (E3,
        WP 4.5). The antihub ratio is a proportion of a count, often zero in every null draw
        at low dimension, so its null standard deviation is floored at one count,
        ``1 / rows``. (A Poisson floor, ``sqrt(null count) / rows``, bound in about 60% of
        ordinary Gaussian cells and shrank their z-scores.) The recommendations come from the
        z-scores of the two rules experiment E4 kept (more hubness and more antihubs than the
        null, under ``metric="euclidean"`` only; WP 7.1), and ``report.n_outside_null``
        counts the calibrated statistics with ``|z| > z_threshold``. Each statistic carries a
        95% interval for its z over resamples of the null draws (``z_interval``), and a
        recommendation whose interval reaches below the threshold says so. A statistic whose
        null failed the E3 bar in the setting at hand (normalised rows; under cosine,
        k-skewness and the antihub ratio; under Euclidean distance above d = 256, neighbour
        consistency) is marked ``uncalibrated`` with the reason, raises nothing and is not
        counted (WP 8.4, :func:`uncalibrated_statistics`). With ``calibrate=False`` the report
        makes no recommendation. Rows that arrive with a common norm (relative spread under
        1e-3, for instance embeddings normalised upstream) are compared with null draws
        normalised to that norm, under either metric; a Gaussian null with varying norms made
        such rows look concentrated (``distance_cv`` z down to -35 in E3). This calibrates
        normalised rows with a flat spectrum only: with a power-law spectrum the null, fitted
        to rows already on the sphere, is more isotropic than the data (``distance_cv`` and
        uniformity z of +8 to +11, antihub z up to 4.5, at n = 2,000, d = 256 and 1,024), so
        on normalised rows those three statistics are marked uncalibrated.
    metric:
        ``"euclidean"`` or ``"cosine"`` (anything else raises ``ValueError``), honoured by every
        analyzer. With ``"cosine"`` the rows are L2-normalised once and every statistic, the
        isotropy ones included, describes the normalised rows: the geometry a cosine search
        sees. Euclidean distances between unit vectors are monotone in cosine distance, so the
        kNN graph is the cosine one; distance concentration, the kernel and intrinsic dimension
        use chord distances on the sphere. The one exception is the null model of
        ``calibrate``: it is fitted to the rows as given and its draws are normalised, so the
        null mean, spread and z-scores depend on the row norms through the null's covariance.
        A null fitted to the normalised rows was biased: on Gaussian data its draws, normalised
        a second time, came out more isotropic than the data (z up to 9 for ``distance_cv``
        and uniformity with an ``i^-1`` spectrum). Rows with zero norm raise ``ValueError``.
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
        _check_metric(self.metric)
        X = self._prepare(X, min_n=5)  # the one validation of the call: warnings are issued here
        with trusted_input():  # the analyzers below get the validated array, and stay silent
            return self._fit_validated(X, y)

    def _fit_validated(self, X: np.ndarray, y) -> EmbedKitReport:
        X_given = X  # the null model of the calibration is fitted to the rows as given
        if self.metric == "cosine":
            X = l2_normalize(X)
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

        ref_id = id_result.reference_estimate
        if np.isnan(ref_id):
            ref_id = max(2, d // 4)
        suggested_target_dim = int(np.clip(round(1.5 * ref_id), max(4, int(ref_id)), d))

        calibration, calibration_k = (
            self._calibrate(X_given) if self.calibrate else (None, None)
        )
        n_outside = (
            None if calibration is None
            else int(sum(abs(c.z_score) > self.z_threshold
                         for c in calibration.values() if c.calibrated))
        )
        properties = _properties(geo_bundle, calibration)
        recommendations = _recommendations(
            id_result, hub_result, geo_bundle, calibration, d, self.z_threshold, self.metric
        )

        return EmbedKitReport(
            intrinsic_dim=id_result,
            hubness=hub_result,
            geometry=geo_bundle,
            kernel=kernel_result,
            k=k,
            suggested_target_dim=suggested_target_dim,
            recommendations=recommendations,
            properties=properties,
            input_shape=(n, d),
            calibration=calibration,
            n_outside_null=n_outside,
            z_threshold=self.z_threshold,
            metric=self.metric,
            calibration_k=calibration_k,
        )

    def _calibrate(self, X: np.ndarray) -> tuple[dict[str, CalibratedStatistic], int]:
        """The calibrated statistics on a subsample of ``X``, against the null model, and the
        ``k`` they used."""
        n = X.shape[0]
        if n > self.n_calibration:
            rng = np.random.default_rng(self.random_state)
            X = X[np.sort(rng.choice(n, self.n_calibration, replace=False))]
        rows = X.shape[0]
        k = min(self.k or max(5, round(np.sqrt(rows) / 2)), rows - 1)
        norm = common_norm(X)
        cal = calibrate(
            lambda Z: calibration_statistics(Z, k, self.metric, self.random_state, norm=norm),
            X,
            model=self.null_model,
            n_draws=self.n_null_draws,
            random_state=self.random_state,
            std_floor={"antihub_ratio": 1.0 / rows},
        )
        flags = uncalibrated_statistics(self.metric, X.shape[1], norm is not None)
        cal = {name: replace(c, uncalibrated=flags.get(name)) for name, c in cal.items()}
        return cal, k


def common_norm(X: np.ndarray, rtol: float = _UNIT_NORM_RTOL) -> float | None:
    """The median row norm of ``X`` when every row norm is within ``rtol`` of it, else None."""
    norms = np.linalg.norm(np.asarray(X, dtype=np.float64), axis=1)
    med = float(np.median(norms))
    if med > 0 and float(np.max(np.abs(norms - med))) <= rtol * med:
        return med
    return None


def calibration_statistics(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    random_state: int | None = 42,
    norm: float | None = None,
) -> dict[str, float]:
    """The statistics in ``CALIBRATED_STATISTICS``, computed as the report computes them.

    One kNN graph is shared by k-skewness, the antihub ratio and neighbour consistency; ``X`` is
    used whole for them, and ``DistanceConcentration`` and ``UniformityScore`` subsample it as
    usual. With ``metric="cosine"`` the rows are L2-normalised first, so that the observed rows
    and a null draw (fitted to the rows as given) are both measured on the sphere. With
    ``norm``, every row is first rescaled to that norm (the report passes the common norm of
    rows normalised upstream, so that null draws are compared on the same sphere).
    """
    _check_metric(metric)
    if norm is not None:
        X = (norm * l2_normalize(X)).astype(X.dtype, copy=False)
    if metric == "cosine":
        X = l2_normalize(X)
    _, graph = knn(X, k, metric=metric)
    hub = HubnessAnalyzer(k=k, metric=metric).fit(X, neighbor_index=graph)
    return {
        "k_skewness": hub.k_skewness,
        "antihub_ratio": hub.antihub_ratio,
        "distance_cv": DistanceConcentration(random_state=random_state).fit(X).distance_cv,
        "neighbor_consistency": NeighborConsistency(
            k=k, metric=metric, random_state=random_state
        ).fit(X, neighbor_index=graph).mean_consistency,
        "uniformity": UniformityScore(random_state=random_state).fit(X).uniformity,
    }


def _recommendations(
    id_result, hub_result, geo_bundle, calibration, d, z_threshold: float = 3.0,
    metric: str = "euclidean",
) -> list[str]:
    """Recommendations from the report: only the rules that experiment E4 kept (WP 7.1).

    E4 tested each rule on 35 natural classification sets: a rule stays when the gain of its
    intervention rises with the signal (Spearman interval over source data sets excluding 0)
    and the intervention significantly beats the raw embedding on at least half of the sets
    where the rule fires (``benchmarks/results/e4/REPORT.md``). Two rules passed, both on
    Euclidean kNN: hubness (k-skewness) and antihubs, each pointing to a hubness-reducing
    neighbour search. Removed: hubness under cosine, distance concentration, neighbour
    consistency, and the fixed ID / D rule (0.1 and 0.3), which fired on all 35 sets while
    PCA to ``suggested_target_dim`` lost accuracy on 32 of them. Their statistics stay in the
    report (``calibration``, ``intrinsic_dim``); they just raise no action. ``id_result``,
    ``hub_result``, ``geo_bundle`` and ``d`` are kept for the call signature.
    """
    recs = []
    if calibration is not None:
        recs.extend(_calibrated_recommendations(calibration, z_threshold, metric))
    if not recs:
        recs.append(
            "No recommendation: no statistic with a validated rule is beyond its null range."
        )
    return recs


def _properties(geo_bundle, calibration) -> list[str]:
    """Descriptive statements about the geometry (WP 4.3), with no thresholds.

    Anisotropy, a shared mean direction and clustered directions are properties, not defects:
    strict isotropy is at odds with cluster structure, and whitening can hurt classification
    (review section 3). Which of them predict that an intervention helps is E4's question.
    """
    iso = geo_bundle.isotropy
    pcs = " / ".join(f"{v:.0%}" for v in iso.top_k_variance_share.values())
    coords = " / ".join(f"{v:.0%}" for v in iso.top_k_coordinate_share.values())
    props = [
        f"Spectrum: the top 1 / 3 / 10 principal components hold {pcs} of the variance; "
        f"the eigenvalues decay as i^-{iso.spectrum_alpha:.2f}; corrected participation ratio "
        f"{iso.participation_ratio_normalized:.3f} of the dimensions (IsoScore "
        f"{iso.isoscore:.3f}, which is biased low when n is not much larger than d).",
        f"Coordinates: the top 1 / 3 / 10 coordinates hold {coords} of the variance.",
        f"Mean direction: the mean vector is {iso.mean_vector_ratio:.2f} of the average norm, "
        f"and random pairs have mean cosine {iso.mean_cosine:.2f}.",
    ]
    c = (calibration or {}).get("uniformity")
    if c is not None and c.calibrated:
        side = "more clustered than" if c.z_score > 0 else "more spread than"
        props.append(
            f"Directions on the sphere are {side} the {c.null_model} null (uniformity "
            f"{c.observed:.3g} against {c.null_mean:.3g} ± {c.null_std:.2g}, "
            f"z = {c.z_score:+.1f})."
        )
    elif c is not None:
        props.append(f"Uniformity is not compared with the null: {c.uncalibrated}.")
    return props


def _calibrated_recommendations(
    calibration, z_threshold: float, metric: str = "euclidean"
) -> list[str]:
    """The two rules E4 kept (WP 7.1), both under ``metric="euclidean"`` only.

    - k-skewness above the null: on Euclidean kNN the gain of the best hubness reduction rose
      with the z-score (Spearman 0.36, 95% interval 0.13 to 0.65 over source data sets) and was
      significant on all 3 natural sets where it fired. Under cosine it did not predict the
      gain (0.05, -0.24 to 0.39; no significant gain on the 2 firing sets).
    - antihub ratio above the null: Spearman 0.32 (0.05 to 0.58), significant gain on 5 of the
      8 firing sets. Under cosine the antihub ratio of Gaussian data is not calibrated (inside
      +-3 in 89% of E3 null cells), so it was never a cosine rule (the user's decision,
      2026-11-16).

    Under cosine the report therefore makes no calibrated recommendation; every z-score stays
    in ``report.calibration``.
    """

    def describe(c) -> str:
        return (f"{c.observed:.3g} against {c.null_mean:.3g} ± {c.null_std:.2g} under the "
                f"{c.null_model} null, z = {c.z_score:+.1f}")

    def near(c) -> str:
        lo, hi = getattr(c, "z_interval", (float("nan"),) * 2)
        if not (lo <= z_threshold):  # also False for NaN
            return ""
        return (f" The reading is near the threshold: over resamples of the {c.n_draws} null "
                f"draws z lies between {lo:+.1f} and {hi:+.1f}; more draws (n_null_draws) "
                "would settle it.")

    if metric != "euclidean":
        return []
    search = ("A hubness-reducing neighbour search (embedkit.hubness_reduction, for instance "
              "MutualProximity or CSLS) is likely to improve Euclidean kNN.")
    recs = []
    c = calibration.get("k_skewness")
    if c is not None and c.calibrated and c.z_score > z_threshold:
        recs.append(
            f"More hubness than the null (k-skewness {describe(c)}): a few points dominate "
            f"the neighbour lists. {search}{near(c)}"
        )
    c = calibration.get("antihub_ratio")
    if c is not None and c.calibrated and c.z_score > z_threshold:
        recs.append(
            f"More antihubs than the null (antihub_ratio {describe(c)}): more points are "
            "never retrieved as a neighbour than in a Gaussian with the same covariance "
            f"spectrum. {search}{near(c)}"
        )
    return recs
