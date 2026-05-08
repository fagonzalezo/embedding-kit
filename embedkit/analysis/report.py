"""EmbedKitAnalyzer: unified analysis report with recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
    UniformityScore,
    UniformityResult,
)
from embedkit.analysis.hubness import HubnessAnalyzer, HubnessResult
from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator, IntrinsicDimensionResult
from embedkit.analysis.kernel import KernelDiagnostics, KernelDiagnosticsResult


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
    severity: Literal["low", "medium", "high"]
    recommendations: list[str]
    input_shape: tuple[int, int]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            ("ID", "consensus", self.intrinsic_dim.consensus),
            ("ID", "uncertainty", self.intrinsic_dim.uncertainty),
            ("Hubness", "k_skewness", self.hubness.k_skewness),
            ("Hubness", "robinhood_index", self.hubness.robinhood_index),
            ("Hubness", "antihub_ratio", self.hubness.antihub_ratio),
            ("Hubness", "hub_ratio", self.hubness.hub_ratio),
            ("Hubness", "hub_contamination", self.hubness.hub_contamination),
            ("Geometry", "relative_contrast", self.geometry.distance_concentration.relative_contrast),
            ("Geometry", "concentration_ratio", self.geometry.distance_concentration.concentration_ratio),
            ("Geometry", "participation_ratio", self.geometry.isotropy.participation_ratio),
            ("Geometry", "effective_rank", self.geometry.isotropy.effective_rank),
            ("Geometry", "isotropy_score", self.geometry.isotropy.isotropy_score),
            ("Geometry", "neighbor_consistency", self.geometry.neighbor_consistency.mean_consistency),
            ("Geometry", "uniformity", self.geometry.uniformity.uniformity),
            ("Kernel", "effective_rank", self.kernel.effective_rank),
            ("Kernel", "spectral_gap", self.kernel.spectral_gap),
            ("Kernel", "condition_number", self.kernel.condition_number),
            ("Kernel", "sigma", self.kernel.sigma),
            ("Summary", "suggested_k", self.suggested_k),
            ("Summary", "suggested_sigma", self.suggested_sigma),
            ("Summary", "suggested_target_dim", self.suggested_target_dim),
            ("Summary", "severity", self.severity),
        ]
        return pd.DataFrame(rows, columns=["category", "metric", "value"])

    def print_summary(self) -> None:
        n, d = self.input_shape
        print(f"\n{'='*60}")
        print(f"  EmbedKit Analysis Report  ({n} samples × {d} dims)")
        print(f"{'='*60}")
        print(f"\n[Intrinsic Dimension]")
        print(f"  consensus ID     : {self.intrinsic_dim.consensus:.2f}  (±{self.intrinsic_dim.uncertainty:.2f})")
        for m, v in self.intrinsic_dim.estimates.items():
            print(f"  {m:<14}: {v:.2f}")
        print(f"\n[Hubness]")
        print(f"  k-skewness       : {self.hubness.k_skewness:.3f}")
        print(f"  Robin Hood index : {self.hubness.robinhood_index:.3f}")
        print(f"  hub ratio        : {self.hubness.hub_ratio:.3f}")
        print(f"  antihub ratio    : {self.hubness.antihub_ratio:.3f}")
        print(f"  hub contamination: {self.hubness.hub_contamination:.3f}")
        print(f"\n[Geometry]")
        print(f"  participation_ratio : {self.geometry.isotropy.participation_ratio:.2f}")
        print(f"  isotropy_score      : {self.geometry.isotropy.isotropy_score:.3f}")
        print(f"  relative_contrast   : {self.geometry.distance_concentration.relative_contrast:.3f}")
        print(f"  neighbor_consistency: {self.geometry.neighbor_consistency.mean_consistency:.3f}")
        print(f"  uniformity          : {self.geometry.uniformity.uniformity:.4f}")
        print(f"\n[Kernel]")
        print(f"  sigma            : {self.kernel.sigma:.4f}")
        print(f"  effective_rank   : {self.kernel.effective_rank:.2f}")
        print(f"  spectral_gap     : {self.kernel.spectral_gap:.3f}")
        print(f"  condition_number : {self.kernel.condition_number:.2e}")
        if self.kernel.kernel_alignment is not None:
            print(f"  kernel_alignment : {self.kernel.kernel_alignment:.4f}")
        print(f"\n[Recommendations]  severity={self.severity.upper()}")
        for rec in self.recommendations:
            print(f"  • {rec}")
        print(f"\n[Auto-config]")
        print(f"  suggested_k          : {self.suggested_k}")
        print(f"  suggested_sigma      : {self.suggested_sigma:.4f}")
        print(f"  suggested_target_dim : {self.suggested_target_dim}")
        print(f"{'='*60}\n")


class EmbedKitAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        k: int | None = None,
        id_methods: list[str] | None = None,
        metric: str = "euclidean",
        n_max: int = 20_000,
        d_max: int = 2_000,
        random_state: int | None = 42,
    ):
        self.k = k
        self.id_methods = id_methods or ["TwoNN", "MLE", "lPCA", "MOM"]
        self.metric = metric
        self.n_max = n_max
        self.d_max = d_max
        self.random_state = random_state

    def fit(self, X, y=None) -> EmbedKitReport:
        X = self._prepare(X, min_n=5)
        n, d = X.shape
        k = self.k or max(5, round(np.sqrt(min(n, self.n_max)) / 2))

        id_est = IntrinsicDimensionEstimator(
            methods=self.id_methods, n_max=min(5_000, self.n_max), random_state=self.random_state
        )
        id_result = id_est.fit(X)

        hub = HubnessAnalyzer(k=k, metric=self.metric, subsample=self.n_max, random_state=self.random_state)
        hub_result = hub.fit(X)

        dc = DistanceConcentration(random_state=self.random_state)
        iso = IsotropyAnalyzer(d_max=self.d_max)
        nc = NeighborConsistency(k=k, metric=self.metric, subsample=self.n_max, random_state=self.random_state)
        uni = UniformityScore(random_state=self.random_state)
        geo_bundle = GeometryBundle(
            distance_concentration=dc.fit(X),
            isotropy=iso.fit(X),
            neighbor_consistency=nc.fit(X),
            uniformity=uni.fit(X),
        )

        kernel = KernelDiagnostics(random_state=self.random_state)
        kernel_result = kernel.fit(X, y)

        suggested_k = k
        suggested_sigma = kernel_result.sigma
        consensus_id = id_result.consensus if not np.isnan(id_result.consensus) else max(2, d // 4)
        suggested_target_dim = int(np.clip(round(1.5 * consensus_id), max(4, int(consensus_id)), d))

        severity, recommendations = _assess_severity_and_recommendations(
            id_result, hub_result, geo_bundle, kernel_result, d
        )

        return EmbedKitReport(
            intrinsic_dim=id_result,
            hubness=hub_result,
            geometry=geo_bundle,
            kernel=kernel_result,
            suggested_k=suggested_k,
            suggested_sigma=suggested_sigma,
            suggested_target_dim=suggested_target_dim,
            severity=severity,
            recommendations=recommendations,
            input_shape=(n, d),
        )


def _assess_severity_and_recommendations(
    id_result, hub_result, geo_bundle, kernel_result, d
) -> tuple[str, list[str]]:
    recs = []
    score = 0

    # Hubness checks
    if hub_result.k_skewness > 5:
        recs.append(
            "High hubness (k_skewness > 5): consider dimensionality reduction or "
            "hubness-aware distance weighting."
        )
        score += 2
    elif hub_result.k_skewness > 2:
        recs.append(
            "Moderate hubness (k_skewness > 2): kNN-based tasks may be unreliable."
        )
        score += 1

    if hub_result.antihub_ratio > 0.3:
        recs.append(
            "Many antihubs (antihub_ratio > 0.3): a significant fraction of points "
            "are never retrieved as neighbors."
        )
        score += 1

    # ID / D ratio (curse of dimensionality)
    consensus_id = id_result.consensus
    if not np.isnan(consensus_id) and d > 0:
        id_d_ratio = consensus_id / d
        if id_d_ratio < 0.1:
            recs.append(
                f"Low intrinsic-to-ambient ratio (ID/D ≈ {id_d_ratio:.2f}): "
                "the embedding is very sparse — strong dimensionality reduction is advised."
            )
            score += 2
        elif id_d_ratio < 0.3:
            recs.append(
                f"Moderate intrinsic-to-ambient ratio (ID/D ≈ {id_d_ratio:.2f}): "
                "dimensionality reduction may improve downstream tasks."
            )
            score += 1

    # Isotropy / participation ratio
    pr = geo_bundle.isotropy.participation_ratio
    if pr < 0.1 * d:
        recs.append(
            f"Low participation ratio ({pr:.1f} << {d}): embeddings are highly anisotropic. "
            "Consider whitening or applying an alignment loss."
        )
        score += 2
    elif pr < 0.3 * d:
        recs.append(
            f"Moderate anisotropy (participation_ratio={pr:.1f}): "
            "alignment losses or PCA whitening may help."
        )
        score += 1

    # Distance concentration
    cr = geo_bundle.distance_concentration.concentration_ratio
    if cr > 0.8:
        recs.append(
            "High distance concentration (concentration_ratio > 0.8): distances are almost "
            "indistinguishable — standard kNN will degrade."
        )
        score += 2

    # Neighbor consistency
    nc = geo_bundle.neighbor_consistency.mean_consistency
    if nc < 0.5:
        recs.append(
            "Low neighbor consistency (<0.5): the kNN graph is unstable under small perturbations. "
            "The embedding lacks local robustness."
        )
        score += 1

    # Uniformity
    u = geo_bundle.uniformity.uniformity
    if u > -1.0:
        recs.append(
            "Poor uniformity (score > -1.0): embeddings are clustered rather than spread. "
            "An alignment + uniformity loss is recommended."
        )
        score += 1

    if not recs:
        recs.append("No major issues detected. Embedding geometry looks healthy.")

    if score >= 4:
        severity: str = "high"
    elif score >= 2:
        severity = "medium"
    else:
        severity = "low"

    return severity, recs  # type: ignore[return-value]
