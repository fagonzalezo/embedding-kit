"""KNNPairs: emit kNN as positives with optional precomputed neighbor index."""

from __future__ import annotations

import numpy as np
import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class KNNPairs(BaseAugmentation):
    def __init__(
        self,
        k: int = 10,
        hard_negatives: bool = False,
        neighbor_index: np.ndarray | None = None,
    ):
        self.k = k
        self.hard_negatives = hard_negatives
        self.neighbor_index = neighbor_index
        self._X_ref: torch.Tensor | None = None  # full training set, set by precompute

    def precompute(self, X: np.ndarray) -> None:
        """Build and cache the neighbor index and reference embeddings for X."""
        from embedkit.utils.neighbors import knn
        _, indices = knn(X, self.k)
        self.neighbor_index = indices  # (N, k) int64
        self._X_ref = torch.from_numpy(X)  # kept on CPU; moved to device on demand

    def __call__(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.neighbor_index is not None and indices is not None and self._X_ref is not None:
            return self._from_precomputed(x, indices)
        return self._from_batch(x)

    def _from_precomputed(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_np = indices.cpu().numpy()
        # Randomly pick one of the k neighbors for each anchor.
        rand_col = np.random.randint(0, self.k, size=len(idx_np))
        neighbor_global = self.neighbor_index[idx_np, rand_col]  # (B,)
        x_j = self._X_ref[neighbor_global].to(x.device, dtype=x.dtype)
        return x, x_j

    def _from_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from embedkit.utils.neighbors import knn
        X_np = x.detach().cpu().numpy()
        _, indices = knn(X_np, self.k)
        idx_tensor = torch.tensor(indices[:, 0], device=x.device)
        return x, x[idx_tensor]
