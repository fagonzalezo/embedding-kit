"""EmbeddingMixup: interpolate embeddings with random kNN."""

from __future__ import annotations

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class EmbeddingMixup(BaseAugmentation):
    def __init__(self, k: int = 10, alpha: float = 0.4):
        self.k = k
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        from embedkit.utils.neighbors import knn
        import numpy as np

        X_np = x.detach().cpu().numpy()
        _, indices = knn(X_np, self.k)
        # pick a random neighbor for each point
        idx_arr = torch.tensor(indices, device=x.device)
        rand_col = torch.randint(0, self.k, (x.shape[0],), device=x.device)
        neighbor_idx = idx_arr[torch.arange(x.shape[0]), rand_col]
        x_neighbor = x[neighbor_idx]

        lam_i = torch.distributions.Beta(self.alpha, self.alpha).sample((x.shape[0], 1)).to(x.device)
        lam_j = torch.distributions.Beta(self.alpha, self.alpha).sample((x.shape[0], 1)).to(x.device)
        xi = lam_i * x + (1 - lam_i) * x_neighbor
        xj = lam_j * x + (1 - lam_j) * x_neighbor
        return xi, xj
