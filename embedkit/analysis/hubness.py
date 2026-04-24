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
    def __init__(self, k: int = 10, hub_threshold: float = 2.0, metric: str = "euclidean"):
        self.k = k
        self.hub_threshold = hub_threshold
        self.metric = metric

    def fit(self, X, y=None) -> HubnessResult:
        X = self._prepare(X)
        n = X.shape[0]
        _, indices = knn(X, self.k, metric=self.metric)

        # k-occurrence: how many times each point appears as a kNN
        N_k = np.zeros(n, dtype=np.int64)
        for row in indices:
            for idx in row:
                N_k[idx] += 1

        mean_N = N_k.mean()
        std_N = N_k.std()
        k_skewness = float(
            np.mean((N_k - mean_N) ** 3) / (std_N ** 3 + 1e-10)
        )

        # Robin Hood index (Gini-like inequality measure)
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

        # Hub contamination: fraction of kNN of hubs that are other hubs
        if len(hubs) == 0:
            hub_contamination = 0.0
        else:
            hub_set = set(hubs.tolist())
            contamination_counts = []
            for h in hubs:
                neighbors = indices[h]
                contamination_counts.append(sum(1 for nb in neighbors if nb in hub_set))
            hub_contamination = float(np.mean(contamination_counts) / self.k)

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
