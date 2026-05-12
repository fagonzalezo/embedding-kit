"""RBF kernel diagnostics: effective rank, spectral gap, alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import linalg as scipy_linalg

from embedkit.analysis._base import BaseAnalyzer, BaseResult


@dataclass(frozen=True)
class KernelDiagnosticsResult(BaseResult):
    effective_rank: float
    spectral_gap: float
    condition_number: float
    row_sum_skewness: float
    eigenvalues: np.ndarray
    kernel_alignment: float | None
    sigma: float


class KernelDiagnostics(BaseAnalyzer):
    _SUBSAMPLE_MAX = 3000

    def __init__(
        self,
        sigma: float | Literal["median"] | None = "median",
        n_components: int = 50,
        subsample: int = 2000,
        random_state: int | None = 42,
    ):
        self.sigma = sigma
        self.n_components = n_components
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None) -> KernelDiagnosticsResult:
        import warnings
        X = self._prepare(X)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        effective_sub = min(self.subsample, self._SUBSAMPLE_MAX)
        if self.subsample > self._SUBSAMPLE_MAX:
            warnings.warn(
                f"KernelDiagnostics.subsample capped at {self._SUBSAMPLE_MAX} "
                f"(requested {self.subsample}) to prevent O(N³) eigendecomposition.",
                stacklevel=2,
            )
        if n > effective_sub:
            idx = rng.choice(n, effective_sub, replace=False)
            Xs = X[idx]
            ys = y[idx] if y is not None else None
        else:
            Xs = X
            ys = y

        sigma = self._resolve_sigma(Xs)
        K = self._rbf_kernel(Xs, sigma)

        ns = Xs.shape[0]
        n_comp = min(self.n_components, ns - 1)
        # Always use eigsh (top-k only) — never the full dense spectrum.
        from scipy.sparse.linalg import eigsh
        eigenvalues = eigsh(K, k=n_comp, which="LM", return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)[::-1]

        eigenvalues = np.maximum(eigenvalues, 0.0).astype(np.float32)
        total = eigenvalues.sum() + 1e-10
        p = eigenvalues / total
        p = p + 1e-10
        p /= p.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        spectral_gap = float(eigenvalues[0] / (eigenvalues[1] + 1e-10)) if len(eigenvalues) > 1 else float("inf")
        condition_number = float(eigenvalues[0] / (eigenvalues[-1] + 1e-10))

        row_sums = K.sum(axis=1)
        rs_mean = row_sums.mean()
        rs_std = row_sums.std() + 1e-10
        row_sum_skewness = float(np.mean(((row_sums - rs_mean) / rs_std) ** 3))

        ka = None
        if ys is not None:
            ys_arr = np.asarray(ys, dtype=np.float32)
            if ys_arr.ndim == 1:
                # Class labels: indicator kernel (same-class = 1)
                label_K = (ys_arr[:, None] == ys_arr[None, :]).astype(np.float32)
            else:
                # Vector labels: linear (dot-product) kernel between label vectors
                norms = np.linalg.norm(ys_arr, axis=1, keepdims=True) + 1e-10
                ys_norm = ys_arr / norms
                label_K = ys_norm @ ys_norm.T
            ka = float(_kernel_alignment(K, label_K))

        return KernelDiagnosticsResult(
            effective_rank=eff_rank,
            spectral_gap=spectral_gap,
            condition_number=condition_number,
            row_sum_skewness=row_sum_skewness,
            eigenvalues=eigenvalues,
            kernel_alignment=ka,
            sigma=float(sigma),
        )

    def _resolve_sigma(self, X: np.ndarray) -> float:
        if isinstance(self.sigma, (int, float)):
            return float(self.sigma)
        # median heuristic
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if n > 1000:
            idx = rng.choice(n, 1000, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(Xs, metric="euclidean")
        upper = dists[np.triu_indices_from(dists, k=1)]
        return float(np.median(upper))

    @staticmethod
    def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(X, metric="euclidean")
        return np.exp(-(dists ** 2) / (2 * sigma ** 2 + 1e-10)).astype(np.float32)


def _kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    def frobenius(A, B):
        return float(np.sum(A * B))
    num = frobenius(K1, K2)
    denom = np.sqrt(frobenius(K1, K1) * frobenius(K2, K2) + 1e-10)
    return num / denom
