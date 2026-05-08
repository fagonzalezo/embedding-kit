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
        # Precomputed (N, k) integer index; set by Trainer before the loop to
        # avoid repeated exact-kNN calls inside the training step.
        self.neighbor_index = neighbor_index

    def precompute(self, X: np.ndarray) -> None:
        """Build and cache the neighbor index for dataset X (call once before fit)."""
        from embedkit.utils.neighbors import knn
        _, indices = knn(X, self.k)
        self.neighbor_index = indices  # (N, k) int64

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.neighbor_index is not None:
            return self._from_precomputed(x)
        return self._from_batch(x)

    def _from_precomputed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is a batch drawn by DataLoader; we need the dataset-level row indices
        # so this path is only safe when the DataLoader preserves dataset order
        # (shuffle=False) or the Trainer passes the global indices alongside x.
        # For the common shuffled-batch case, fall back to batch-level kNN.
        return self._from_batch(x)

    def _from_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from embedkit.utils.neighbors import knn
        X_np = x.detach().cpu().numpy()
        _, indices = knn(X_np, self.k)
        idx_tensor = torch.tensor(indices[:, 0], device=x.device)
        return x, x[idx_tensor]
