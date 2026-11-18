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
        ``participation_ratio_corrected / d``, in ``[0, 1]``: the n-robust isotropy summary.
    effective_rank:
        ``exp`` of the entropy of the normalised eigenvalues, over the full non-zero spectrum.
        A plug-in value: for iid data it is below ``min(n - 1, d)`` by a factor that depends on
        ``d / n`` (0.66 at n = 3,000, d = 2,500).
    isotropy_score:
        ``effective_rank / min(n - 1, d)``, in ``[0, 1]``; same finite-sample bias.
    isoscore:
        IsoScore (Rudman et al. 2022), computed as published, in ``[0, 1]``. Its six steps reduce
        to ``(participation_ratio - 1) / (d - 1)``, so it carries the plug-in bias: iid data at
        n = 300, d = 512 score 0.37, not 1. ``participation_ratio_normalized`` corrects that.
    top_k_variance_share:
        ``{k: share of the total variance in the top k principal components}`` for k in 1, 3, 10.
    top_k_coordinate_share:
        The same for the k coordinates with the largest variance: a few dominant coordinates are
        the "rogue dimensions" of Timkey and van Schijndel (2021).
    spectrum_alpha:
        Power-law decay exponent of the eigenvalues, ``lambda_i ~ i^-alpha`` (Agrawal et al.
        2022), from a least-squares line on the log-log spectrum over the top half of the
        non-zero eigenvalues: the finite-sample tail bends down and inflated the full-range fit
        (by up to 0.35 for alpha = 1 at n = 300, d = 512). Recovers alpha = 1 and 2 within 0.06
        from n = 300 to 5,000; a flat spectrum gives 0.1-0.35. NaN with fewer than 4 non-zero
        eigenvalues.
    mean_vector_ratio:
        ``||mean(x)|| / mean(||x||)`` on the uncentred rows, in ``[0, 1]``: how much of each
        vector is a shared offset.
    mean_cosine:
        Average cosine similarity over all pairs of distinct rows of the subsample, uncentred.
    eigenvalue_spectrum_top, explained_variance_ratio:
        The largest ``n_components`` eigenvalues, for plotting only. The ratio is relative to
        the total variance.

    These are properties of the geometry, not defects: strict isotropy is at odds with cluster
    structure, and whitening can hurt classification (review section 3).
    """

    participation_ratio: float
    effective_rank: float
    eigenvalue_spectrum_top: np.ndarray
    isotropy_score: float
    explained_variance_ratio: np.ndarray
    participation_ratio_normalized: float
    participation_ratio_corrected: float
    isoscore: float
    top_k_variance_share: dict[int, float]
    top_k_coordinate_share: dict[int, float]
    spectrum_alpha: float
    mean_vector_ratio: float
    mean_cosine: float

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

        dists = _pairwise_distances_exact_zeros(Xs)
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


def _pairwise_distances_exact_zeros(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix in float64, with exact zeros between identical rows.

    The Gram expansion ``|x|^2 + |y|^2 - 2 x.y`` uses BLAS (``scipy`` ``cdist`` does not, and
    was 15 times slower at 2,000 x 2,048), but leaves rounding error where the distance is 0.
    Identical rows are found exactly and set to 0, as ``cdist`` would give.
    """
    Xd = np.asarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", Xd, Xd)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Xd @ Xd.T)
    np.maximum(D2, 0.0, out=D2)
    rows = np.ascontiguousarray(Xd).view(np.dtype((np.void, Xd.itemsize * Xd.shape[1]))).ravel()
    _, inverse = np.unique(rows, return_inverse=True)
    inverse = inverse.ravel()
    if len(np.unique(inverse)) < len(inverse):
        D2[inverse[:, None] == inverse[None, :]] = 0.0
    else:
        np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2, out=D2)


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
        mean_vector_ratio, mean_cosine = _mean_direction(Xc)
        coord_var = Xc.var(axis=0, ddof=1) if n > 1 else np.zeros(d)
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
        isoscore = float(np.clip((pr - 1.0) / (d - 1), 0.0, 1.0)) if d > 1 and pr > 0 else 0.0

        top = eigenvalues[: self.n_components]
        return IsotropyResult(
            participation_ratio=float(pr),
            effective_rank=eff_rank,
            eigenvalue_spectrum_top=top.astype(np.float32),
            isotropy_score=max(0.0, min(1.0, float(isotropy_score))),
            explained_variance_ratio=(top / (total + 1e-30)).astype(np.float32),
            participation_ratio_normalized=pr_norm,
            participation_ratio_corrected=float(pr_corr),
            isoscore=isoscore,
            top_k_variance_share=_top_k_share(eigenvalues),
            top_k_coordinate_share=_top_k_share(coord_var),
            spectrum_alpha=_spectrum_alpha(eigenvalues),
            mean_vector_ratio=mean_vector_ratio,
            mean_cosine=mean_cosine,
        )


_TOP_K = (1, 3, 10)


def _top_k_share(variances: np.ndarray) -> dict[int, float]:
    """``{k: share of the total in the k largest values}`` for k in 1, 3, 10."""
    v = np.sort(np.asarray(variances, dtype=np.float64))[::-1]
    total = v.sum()
    if total <= 0:
        return {k: float("nan") for k in _TOP_K}
    return {k: float(v[:k].sum() / total) for k in _TOP_K}


def _spectrum_alpha(eigenvalues: np.ndarray) -> float:
    """Power-law exponent of the top half of the non-zero spectrum (see ``IsotropyResult``)."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    lam = lam[lam > 1e-12 * lam[0]] if len(lam) and lam[0] > 0 else lam[:0]
    r = len(lam) // 2
    if r < 2 or len(lam) < 4:
        return float("nan")
    i = np.arange(1, r + 1)
    slope = np.polyfit(np.log(i), np.log(lam[:r]), 1)[0]
    return float(-slope)


def _mean_direction(X: np.ndarray) -> tuple[float, float]:
    """``||mean(x)|| / mean(||x||)`` and the mean cosine over all pairs of distinct rows."""
    norms = np.linalg.norm(X, axis=1)
    if norms.mean() <= 0:
        return 0.0, 0.0
    ratio = float(np.linalg.norm(X.mean(axis=0)) / norms.mean())
    n = X.shape[0]
    if n < 2:
        return ratio, float("nan")
    Z = X[norms > 0] / norms[norms > 0, None]
    s = Z.sum(axis=0)
    # sum over i != j of z_i . z_j = ||sum z||^2 - sum ||z_i||^2; zero rows count as cosine 0.
    cos = (float(s @ s) - float(np.sum(Z * Z))) / (n * (n - 1))
    return ratio, cos


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
        n_pert = (
            self.n_perturbations
            if n <= 2000
            else max(2, self.n_perturbations - (n // 2000 - 1))
        )

        if neighbor_index is not None:
            from embedkit.analysis.hubness import _check_neighbor_index
            base_indices = _check_neighbor_index(neighbor_index, n, self.k)
        else:
            _, base_indices = knn(X, self.k, metric=self.metric)

        # RMS per-coordinate sd of the centred rows: translation-invariant. np.std(X) over all
        # entries (before 0.5) also counted the spread of the coordinate means, so adding a
        # common vector changed the perturbation and the consistency.
        std = self.noise_std * float(np.sqrt(np.mean(np.var(X, axis=0, dtype=np.float64))))
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

        Xs = Xs.astype(np.float64)
        norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-10
        Zs = Xs / norms

        # Wang & Isola uniformity: log E[exp(-t ||z_i - z_j||^2)] over pairs i < j, with
        # ||z_i - z_j||^2 = |z_i|^2 + |z_j|^2 - 2 z_i.z_j from one BLAS product.
        sq = np.einsum("ij,ij->i", Zs, Zs)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (Zs @ Zs.T)
        sq_dists = np.maximum(D2[np.triu_indices_from(D2, k=1)], 0.0)
        uniformity = float(np.log(np.mean(np.exp(-self.t * sq_dists)) + 1e-10))

        return UniformityResult(uniformity=uniformity)
