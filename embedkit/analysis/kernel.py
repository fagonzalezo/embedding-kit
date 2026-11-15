"""RBF kernel diagnostics: effective rank, spectral gap, centred kernel alignment.

The spectrum statistics come from the centred kernel ``HKH`` (``H = I - 11^T / n``): the
uncentred RBF kernel has a large eigenvalue along the constant direction that says only that the
kernel is positive, and dominated the old spectral gap (defect 29).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from embedkit.analysis._base import BaseAnalyzer, BaseResult
from embedkit.utils.validation import encode_labels


@dataclass(frozen=True)
class KernelDiagnosticsResult(BaseResult):
    """Spectrum of the centred RBF kernel on a subsample, and centred kernel alignment.

    effective_rank:
        ``tr(Kc)^2 / tr(Kc^2)`` of the centred kernel ``Kc`` (the participation ratio of its
        eigenvalues), from traces, so it uses the whole spectrum and is not capped by
        ``n_components``. Until 0.4 this was the entropy effective rank of the top 50
        eigenvalues of the uncentred kernel, capped at 50.
    spectral_gap:
        ``lambda_1 / lambda_2`` of the centred kernel, in ``[1, inf)``.
    top_k_eigen_ratio:
        ``lambda_1 / lambda_k`` of the centred kernel, with ``k`` the number of eigenvalues in
        ``eigenvalues`` (``n_components``, or fewer on small samples). Formerly
        ``condition_number``: it is not a condition number of the kernel.
    row_sum_skewness:
        Skewness of the row sums of the uncentred kernel (the row sums of the centred kernel
        are all zero): a continuous analogue of hubness.
    eigenvalues:
        The top ``n_components`` eigenvalues of the centred kernel, for plotting.
    """

    effective_rank: float
    spectral_gap: float
    top_k_eigen_ratio: float
    row_sum_skewness: float
    eigenvalues: np.ndarray
    centered_kernel_alignment: float | None
    sigma: float

    @property
    def condition_number(self) -> float:
        """Deprecated alias of ``top_k_eigen_ratio``."""
        warnings.warn(
            "KernelDiagnosticsResult.condition_number is deprecated; use top_k_eigen_ratio "
            "(lambda_1 / lambda_k of the centred kernel, not a condition number).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.top_k_eigen_ratio

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
        Kc = _center(K)

        # Effective rank from traces: the whole spectrum, no truncation.
        sq = float(np.sum(Kc * Kc))
        eff_rank = float(np.trace(Kc) ** 2 / sq) if sq > 1e-24 else 0.0

        ns = Xs.shape[0]
        # The centred kernel has rank at most ns - 1 (the constant vector is in its null space).
        n_comp = max(1, min(self.n_components, ns - 2))
        if ns <= 3 * n_comp or ns <= 200:
            eigenvalues = np.linalg.eigvalsh(Kc)[::-1][:n_comp]
        else:
            from scipy.sparse.linalg import eigsh
            eigenvalues = np.sort(eigsh(Kc, k=n_comp, which="LA", return_eigenvectors=False))[::-1]
        eigenvalues = np.maximum(eigenvalues, 0.0)
        spectral_gap = _ratio(eigenvalues[0], eigenvalues[1]) if len(eigenvalues) > 1 else float(
            "inf")
        top_k_eigen_ratio = _ratio(eigenvalues[0], eigenvalues[-1])

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
            top_k_eigen_ratio=top_k_eigen_ratio,
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
        return np.exp(-(dists.astype(np.float64) ** 2) / (2 * sigma ** 2 + 1e-10))


def _ratio(a: float, b: float) -> float:
    """``a / b``, with ``inf`` for ``b == 0 < a`` and 1 for ``a == b == 0``."""
    if b > 0:
        return float(a / b)
    return float("inf") if a > 0 else 1.0


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
