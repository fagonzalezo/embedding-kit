"""Hubness analysis using skhubness and direct N_k computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedkit.analysis._base import BaseAnalyzer, BaseResult
from embedkit.utils.neighbors import knn


@dataclass(frozen=True)
class HubnessResult(BaseResult):
    k_skewness: float
    robinhood_index: float
    antihub_ratio: float
    hub_ratio: float
    k_occurrence: np.ndarray
    hubs: np.ndarray
    antihubs: np.ndarray
    hub_contamination: float


class HubnessAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        k: int = 10,
        hub_threshold: float = 2.0,
        metric: str = "euclidean",
        subsample: int = 20_000,
        random_state: int | None = 42,
    ):
        self.k = k
        self.hub_threshold = hub_threshold
        self.metric = metric
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None, neighbor_index: np.ndarray | None = None) -> HubnessResult:
        """Compute hubness statistics.

        neighbor_index:
            Optional precomputed kNN graph of ``X`` (shape ``(n, >= k)``, row ``i`` holding
            the neighbours of row ``i`` in ascending distance, without ``i`` itself), for
            example from :func:`embedkit.utils.neighbors.knn`. When given, it is used as is:
            ``X`` is not subsampled and no search is run. Its first ``k`` columns are used.
        """
        X = self._prepare(X)
        n = X.shape[0]
        if neighbor_index is not None:
            indices = _check_neighbor_index(neighbor_index, n, self.k)
        else:
            rng = np.random.default_rng(self.random_state)
            if n > self.subsample:
                idx = rng.choice(n, self.subsample, replace=False)
                X = X[idx]
                n = self.subsample
            _, indices = knn(X, self.k, metric=self.metric)

        N_k = k_occurrence(indices, n)
        mean_N = N_k.mean()
        k_skewness = k_occurrence_skewness(N_k)

        sorted_N = np.sort(N_k)
        n_arr = np.arange(1, n + 1)
        robinhood = float(
            1.0 - 2.0 * np.sum(sorted_N * n_arr) / (n * np.sum(N_k) + 1e-10)
            + 1.0 / n
        )

        threshold = self.hub_threshold * mean_N
        hubs = np.where(N_k > threshold)[0].astype(np.int64)
        antihubs = np.where(N_k == 0)[0].astype(np.int64)
        antihub_ratio = float(len(antihubs) / n)
        hub_ratio = float(len(hubs) / n)

        if len(hubs) == 0:
            hub_contamination = 0.0
        else:
            hub_mask = N_k > threshold  # boolean (n,)
            hub_contamination = float(hub_mask[indices[hubs]].mean())

        return HubnessResult(
            k_skewness=k_skewness,
            robinhood_index=robinhood,
            antihub_ratio=antihub_ratio,
            hub_ratio=hub_ratio,
            k_occurrence=N_k,
            hubs=hubs,
            antihubs=antihubs,
            hub_contamination=hub_contamination,
        )


def k_occurrence(indices: np.ndarray, n: int) -> np.ndarray:
    """k-occurrence ``N_k``: how often each of ``n`` points appears in the neighbour lists.

    ``indices`` is a kNN graph of shape ``(n, k)``, for example from
    :func:`embedkit.utils.neighbors.knn`. Returns an ``int64`` array of length ``n``.
    """
    return np.bincount(np.asarray(indices).ravel(), minlength=n).astype(np.int64)


def k_occurrence_skewness(N_k: np.ndarray) -> float:
    """Skewness of the k-occurrence distribution (the "hubness" of Radovanovic et al. 2010).

    Population (biased) skewness ``E[(N - mean)^3] / std^3``; ``1e-10`` is added to the cubed
    standard deviation, so a constant ``N_k`` gives 0.
    """
    N_k = np.asarray(N_k)
    mean_N = N_k.mean()
    std_N = N_k.std()
    return float(np.mean((N_k - mean_N) ** 3) / (std_N ** 3 + 1e-10))


def _check_neighbor_index(neighbor_index, n: int, k: int) -> np.ndarray:
    """Validate a precomputed kNN graph for ``n`` points and return its first ``k`` columns."""
    indices = np.asarray(neighbor_index)
    if indices.ndim != 2 or indices.shape[0] != n or indices.shape[1] < k:
        raise ValueError(
            f"neighbor_index must have shape ({n}, >= {k}), got {indices.shape}"
        )
    return indices[:, :k].astype(np.int64, copy=False)
