"""RBF kernel diagnostics: effective rank, spectral gap, centred kernel alignment."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from embedkit.analysis._base import BaseAnalyzer, BaseResult
from embedkit.utils.validation import encode_labels


@dataclass(frozen=True)
class KernelDiagnosticsResult(BaseResult):
    effective_rank: float
    spectral_gap: float
    condition_number: float
    row_sum_skewness: float
    eigenvalues: np.ndarray
    centered_kernel_alignment: float | None
    sigma: float

    @property
    def kernel_alignment(self) -> float | None:
        """Deprecated alias of ``centered_kernel_alignment``.

        The old statistic was not centred; the value returned here is the centred one.
        """
        warnings.warn(
            "KernelDiagnosticsResult.kernel_alignment is deprecated; use "
            "centered_kernel_alignment (the old, uncentred statistic mostly measured class "
            "balance).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.centered_kernel_alignment


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
        """Kernel spectrum diagnostics and, when ``y`` is given, centred kernel alignment.

        ``y`` may be a list, array or tensor of class labels (strings, booleans or integers),
        of continuous values ``(n,)``, or of vector values ``(n, m)``; see ``encode_labels``.
        """
        X = self._prepare(X)
        labels, label_kind = None, None
        if y is not None:
            labels, label_kind = encode_labels(y, n_samples=X.shape[0])
            labels = labels.numpy()
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
            ys = labels[idx] if labels is not None else None
        else:
            Xs = X
            ys = labels

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

        cka = None
        if ys is not None:
            cka = _centered_kernel_alignment(K, _label_kernel(ys, label_kind))

        return KernelDiagnosticsResult(
            effective_rank=eff_rank,
            spectral_gap=spectral_gap,
            condition_number=condition_number,
            row_sum_skewness=row_sum_skewness,
            eigenvalues=eigenvalues,
            centered_kernel_alignment=cka,
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


def _label_kernel(labels: np.ndarray, kind: str) -> np.ndarray:
    """Indicator kernel for classes; RBF with median-heuristic bandwidth for continuous labels."""
    if kind == "class":
        return (labels[:, None] == labels[None, :]).astype(np.float64)
    from scipy.spatial.distance import pdist, squareform

    Y = labels.astype(np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    pair = pdist(Y)
    positive = pair[pair > 0]
    if positive.size == 0:
        return np.ones((len(Y), len(Y)))  # constant labels
    bandwidth = float(np.median(pair)) or float(positive.mean())
    return np.exp(-(squareform(pair) ** 2) / (2 * bandwidth**2))


def _center(K: np.ndarray) -> np.ndarray:
    """``H K H`` with ``H = I - 11^T / n``."""
    K = np.asarray(K, dtype=np.float64)
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def _centered_kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """``<HK1H, HK2H>_F / (||HK1H||_F ||HK2H||_F)``; NaN when either kernel is constant."""
    Kc, Lc = _center(K1), _center(K2)
    denom = float(np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc)))
    if denom <= 1e-12:
        warnings.warn(
            "Centred kernel alignment is undefined: one of the kernels is constant "
            "(for example, all labels are equal).",
            stacklevel=3,
        )
        return float("nan")
    return float(np.sum(Kc * Lc) / denom)
