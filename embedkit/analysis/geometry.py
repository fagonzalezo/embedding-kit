"""Geometric diagnostics: distance concentration, isotropy, neighbor consistency, uniformity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedkit.analysis._base import BaseAnalyzer, BaseResult
from embedkit.utils.neighbors import knn


@dataclass(frozen=True)
class DistanceConcentrationResult(BaseResult):
    relative_contrast: float
    concentration_ratio: float
    distance_histogram: np.ndarray
    bin_edges: np.ndarray


@dataclass(frozen=True)
class IsotropyResult(BaseResult):
    participation_ratio: float
    effective_rank: float
    eigenvalue_spectrum: np.ndarray
    isotropy_score: float
    explained_variance_ratio: np.ndarray


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

        # pairwise distances (upper triangle)
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(Xs, metric="euclidean")
        upper = dists[np.triu_indices_from(dists, k=1)]
        d_min, d_max, d_mean = upper.min(), upper.max(), upper.mean()
        relative_contrast = float((d_max - d_min) / (d_mean + 1e-10))
        concentration_ratio = float(d_min / (d_max + 1e-10))

        counts, bin_edges = np.histogram(upper, bins=self.n_bins)
        return DistanceConcentrationResult(
            relative_contrast=relative_contrast,
            concentration_ratio=concentration_ratio,
            distance_histogram=counts.astype(np.float32),
            bin_edges=bin_edges.astype(np.float32),
        )


class IsotropyAnalyzer(BaseAnalyzer):
    def fit(self, X, y=None) -> IsotropyResult:
        X = self._prepare(X)
        Xc = X - X.mean(axis=0)
        cov = np.cov(Xc.T)
        if cov.ndim == 0:
            eigenvalues = np.array([float(cov)])
        else:
            eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1].astype(np.float32)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total = eigenvalues.sum()
        evr = eigenvalues / (total + 1e-10)

        pr = float((eigenvalues.sum() ** 2) / (np.sum(eigenvalues ** 2) + 1e-10))
        # effective rank via entropy
        p = evr + 1e-10
        p /= p.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        isotropy_score = float(eff_rank / len(eigenvalues))

        return IsotropyResult(
            participation_ratio=pr,
            effective_rank=eff_rank,
            eigenvalue_spectrum=eigenvalues,
            isotropy_score=max(0.0, min(1.0, isotropy_score)),
            explained_variance_ratio=evr,
        )


class NeighborConsistency(BaseAnalyzer):
    def __init__(
        self,
        k: int = 10,
        n_perturbations: int = 5,
        noise_std: float = 0.01,
        metric: str = "euclidean",
        random_state: int | None = 42,
    ):
        self.k = k
        self.n_perturbations = n_perturbations
        self.noise_std = noise_std
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None) -> NeighborConsistencyResult:
        X = self._prepare(X)
        rng = np.random.default_rng(self.random_state)
        _, base_indices = knn(X, self.k, metric=self.metric)
        base_sets = [set(row) for row in base_indices]

        consistencies = []
        for _ in range(self.n_perturbations):
            noise = rng.normal(0, self.noise_std * np.std(X), size=X.shape).astype(np.float32)
            Xp = X + noise
            _, pert_indices = knn(Xp, self.k, metric=self.metric)
            fracs = [
                len(base_sets[i] & set(pert_indices[i])) / self.k
                for i in range(X.shape[0])
            ]
            consistencies.append(np.mean(fracs))

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

        # L2 normalize
        norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-10
        Zs = Xs / norms

        # Wang & Isola uniformity: log E[exp(-t ||z_i - z_j||^2)]
        sq_diffs = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=-1)
        upper = sq_diffs[np.triu_indices_from(sq_diffs, k=1)]
        uniformity = float(np.log(np.mean(np.exp(-self.t * upper)) + 1e-10))

        return UniformityResult(uniformity=uniformity)
