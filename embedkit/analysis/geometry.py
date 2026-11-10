"""Geometric diagnostics: distance concentration, isotropy, neighbor consistency, uniformity."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from embedkit.analysis._base import BaseAnalyzer, BaseResult
from embedkit.utils.neighbors import knn


@dataclass(frozen=True)
class DistanceConcentrationResult(BaseResult):
    """Distance concentration statistics on a subsample.

    ``distance_cv`` (std / mean of pairwise distances) is the statistic the report uses: it
    does not depend on a single pair of points. ``concentration_ratio`` (global ``d_min /
    d_max``) is kept for compatibility but collapses to 0 when one pair is duplicated.
    ``relative_contrast_per_query`` is the median over query points of
    ``(d_max - d_min) / d_min``, with zero distances excluded.
    """

    relative_contrast: float
    concentration_ratio: float
    distance_histogram: np.ndarray
    bin_edges: np.ndarray
    distance_cv: float
    relative_contrast_per_query: float


@dataclass(frozen=True)
class IsotropyResult(BaseResult):
    """Isotropy statistics of the (sub)sampled covariance spectrum.

    participation_ratio:
        Plug-in participation ratio ``tr(C)^2 / tr(C^2)`` of the sample covariance ``C``.
        Exact at any dimension (computed from traces). Bounded above by ``min(n - 1, d)``.
    participation_ratio_corrected:
        The same ratio with ``tr(C^2)`` replaced by its unbiased estimate of ``tr(Sigma^2)``
        (Gaussian sampling model), so that iid data gives about ``d`` whatever ``n / d`` is.
    participation_ratio_normalized:
        ``participation_ratio_corrected / d``, in ``[0, 1]``. This is the value the report
        thresholds.
    effective_rank:
        ``exp`` of the entropy of the normalised eigenvalues, over the full non-zero spectrum.
    isotropy_score:
        ``effective_rank / min(n - 1, d)``, in ``[0, 1]``.
    eigenvalue_spectrum_top, explained_variance_ratio:
        The largest ``n_components`` eigenvalues, for plotting only. The ratio is relative to
        the total variance.
    """

    participation_ratio: float
    effective_rank: float
    eigenvalue_spectrum_top: np.ndarray
    isotropy_score: float
    explained_variance_ratio: np.ndarray
    participation_ratio_normalized: float
    participation_ratio_corrected: float

    @property
    def eigenvalue_spectrum(self) -> np.ndarray:
        """Deprecated alias of ``eigenvalue_spectrum_top``."""
        warnings.warn(
            "IsotropyResult.eigenvalue_spectrum is deprecated; use eigenvalue_spectrum_top "
            "(the truncated spectrum, for plotting only).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.eigenvalue_spectrum_top


@dataclass(frozen=True)
class NeighborConsistencyResult(BaseResult):
    mean_consistency: float
    std_consistency: float
    per_perturbation: np.ndarray


@dataclass(frozen=True)
class UniformityResult(BaseResult):
    uniformity: float


class DistanceConcentration(BaseAnalyzer):
    def __init__(self, subsample: int = 2000, n_bins: int = 50, random_state: int | None = 42):
        self.subsample = subsample
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, X, y=None) -> DistanceConcentrationResult:
        X = self._prepare(X)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if n > self.subsample:
            idx = rng.choice(n, self.subsample, replace=False)
            Xs = X[idx]
        else:
            Xs = X

        # float64 cdist gives exact zeros for duplicated rows; sklearn's dot-product expansion
        # does not.
        from scipy.spatial.distance import cdist

        dists = cdist(Xs.astype(np.float64), Xs.astype(np.float64))
        upper = dists[np.triu_indices_from(dists, k=1)]
        d_min, d_max, d_mean = upper.min(), upper.max(), upper.mean()
        relative_contrast = float((d_max - d_min) / (d_mean + 1e-10))
        concentration_ratio = float(d_min / (d_max + 1e-10))
        distance_cv = float(upper.std() / (d_mean + 1e-10))

        # Per query: (max - min) / min over the other points, zero distances excluded.
        positive = dists > 0
        q_min = np.where(positive, dists, np.inf).min(axis=1)
        q_max = np.where(positive, dists, -np.inf).max(axis=1)
        valid = positive.any(axis=1)
        if valid.any():
            rcq = float(np.median((q_max[valid] - q_min[valid]) / q_min[valid]))
        else:
            rcq = float("nan")

        counts, bin_edges = np.histogram(upper, bins=self.n_bins)
        return DistanceConcentrationResult(
            relative_contrast=relative_contrast,
            concentration_ratio=concentration_ratio,
            distance_histogram=counts.astype(np.float32),
            bin_edges=bin_edges.astype(np.float32),
            distance_cv=distance_cv,
            relative_contrast_per_query=rcq,
        )


class IsotropyAnalyzer(BaseAnalyzer):
    """Participation ratio and effective rank of the covariance spectrum.

    All statistics come from the covariance (``n >= d``) or the Gram matrix (``n < d``) of a
    subsample of at most ``n_max`` rows, so they are exact at every dimension and do not
    change definition with ``d``. ``d_max`` is kept for compatibility and ignored.
    """

    def __init__(
        self,
        d_max: int = 2000,
        n_components: int = 512,
        n_max: int = 5000,
        random_state: int | None = 42,
    ):
        self.d_max = d_max
        self.n_components = n_components
        self.n_max = n_max
        self.random_state = random_state

    def fit(self, X, y=None) -> IsotropyResult:
        X = self._prepare(X)
        if X.shape[0] > self.n_max:
            rng = np.random.default_rng(self.random_state)
            X = X[rng.choice(X.shape[0], self.n_max, replace=False)]
        n, d = X.shape
        m = max(n - 1, 1)
        Xc = X.astype(np.float64)
        Xc -= Xc.mean(axis=0)

        # tr(C) = ||Xc||_F^2 / (n-1);
        # tr(C^2) = ||Xc^T Xc||_F^2 / (n-1)^2 = ||Xc Xc^T||_F^2 / (n-1)^2.
        # The smaller of the two matrices is formed; its eigenvalues are the non-zero
        # eigenvalues of the covariance (up to the factor 1 / (n-1)).
        M = Xc @ Xc.T if n < d else Xc.T @ Xc
        eigenvalues = np.linalg.eigvalsh(M)[::-1] / m
        eigenvalues = np.maximum(eigenvalues, 0.0)[: min(n - 1, d)]
        tr_c = float(np.trace(M)) / m
        tr_c2 = float(np.sum(M * M)) / m**2
        del M, Xc

        if tr_c2 > 0:
            pr = tr_c**2 / tr_c2
        else:
            pr = 0.0  # constant data: no variance in any direction

        # Unbiased estimate of tr(Sigma^2) for Gaussian data (Wishart moments with m = n-1
        # degrees of freedom): E[tr(C^2) - tr(C)^2 / m] = tr(Sigma^2) (m - 1)(m + 2) / m^2.
        pr_corr = pr
        if m > 1 and tr_c2 > 0:
            tr_sigma2 = (m**2 / ((m - 1) * (m + 2))) * (tr_c2 - tr_c**2 / m)
            pr_corr = tr_c**2 / tr_sigma2 if tr_sigma2 > 0 else float(d)
            pr_corr = float(np.clip(pr_corr, 1.0, d))
        pr_norm = float(np.clip(pr_corr / d, 0.0, 1.0)) if tr_c2 > 0 else 0.0

        total = eigenvalues.sum()
        if total > 0:
            p = eigenvalues[eigenvalues > 0] / total
            eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        else:
            eff_rank = 1.0
        isotropy_score = eff_rank / max(min(n - 1, d), 1)

        top = eigenvalues[: self.n_components]
        return IsotropyResult(
            participation_ratio=float(pr),
            effective_rank=eff_rank,
            eigenvalue_spectrum_top=top.astype(np.float32),
            isotropy_score=max(0.0, min(1.0, float(isotropy_score))),
            explained_variance_ratio=(top / (total + 1e-30)).astype(np.float32),
            participation_ratio_normalized=pr_norm,
            participation_ratio_corrected=float(pr_corr),
        )


class NeighborConsistency(BaseAnalyzer):
    def __init__(
        self,
        k: int = 10,
        n_perturbations: int = 5,
        noise_std: float = 0.01,
        metric: str = "euclidean",
        subsample: int = 10_000,
        random_state: int | None = 42,
    ):
        self.k = k
        self.n_perturbations = n_perturbations
        self.noise_std = noise_std
        self.metric = metric
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None, neighbor_index: np.ndarray | None = None) -> NeighborConsistencyResult:
        """Measure how much of the kNN graph survives small Gaussian perturbations.

        neighbor_index:
            Optional precomputed kNN graph of the unperturbed ``X`` (shape ``(n, >= k)``).
            When given, ``X`` is not subsampled and the base search is skipped.
        """
        X = self._prepare(X)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if neighbor_index is None and n > self.subsample:
            idx = rng.choice(n, self.subsample, replace=False)
            X = X[idx]
            n = self.subsample

        # Scale perturbation count down for large subsamples to keep wall time bounded.
        n_pert = self.n_perturbations if n <= 2000 else max(2, self.n_perturbations - (n // 2000 - 1))

        if neighbor_index is not None:
            from embedkit.analysis.hubness import _check_neighbor_index
            base_indices = _check_neighbor_index(neighbor_index, n, self.k)
        else:
            _, base_indices = knn(X, self.k, metric=self.metric)

        std = self.noise_std * float(np.std(X))
        noise_buf = np.empty(X.shape, dtype=np.float64)
        consistencies = []
        for _ in range(n_pert):
            rng.standard_normal(out=noise_buf)
            noise_buf *= std
            Xp = (X + noise_buf.astype(X.dtype))
            _, pert_indices = knn(Xp, self.k, metric=self.metric)
            # Vectorized intersection: (n, k, 1) == (n, 1, k) → (n, k, k), sum over both dims.
            # O(n·k²) NumPy ops instead of a Python loop with set intersections.
            matches = (
                (base_indices[:, :, None] == pert_indices[:, None, :])
                .any(axis=2)
                .sum(axis=1)
                .astype(np.float32)
            ) / self.k
            consistencies.append(float(matches.mean()))

        arr = np.array(consistencies, dtype=np.float32)
        return NeighborConsistencyResult(
            mean_consistency=float(arr.mean()),
            std_consistency=float(arr.std()),
            per_perturbation=arr,
        )


class UniformityScore(BaseAnalyzer):
    def __init__(self, t: float = 2.0, subsample: int = 2000, random_state: int | None = 42):
        self.t = t
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None) -> UniformityResult:
        X = self._prepare(X)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if n > self.subsample:
            idx = rng.choice(n, self.subsample, replace=False)
            Xs = X[idx]
        else:
            Xs = X

        norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-10
        Zs = (Xs / norms).astype(np.float32)

        # Wang & Isola uniformity: log E[exp(-t ||z_i - z_j||^2)]
        # Use pdist to avoid the N×N×D broadcast tensor.
        from scipy.spatial.distance import pdist
        sq_dists = pdist(Zs, metric="sqeuclidean")
        uniformity = float(np.log(np.mean(np.exp(-self.t * sq_dists)) + 1e-10))

        return UniformityResult(uniformity=uniformity)
