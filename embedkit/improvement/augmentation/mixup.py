"""EmbeddingMixup: interpolate embeddings with random kNN."""

from __future__ import annotations

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class EmbeddingMixup(BaseAugmentation):
    kind = "pair"

    def __init__(self, k: int = 10, alpha: float = 0.4):
        self.k = k
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:

        from embedkit.utils.neighbors import knn

        X_np = x.detach().cpu().numpy()
        k = min(self.k, x.shape[0] - 1)  # neighbours are searched within the batch
        _, indices = knn(X_np, k)
        # pick a random neighbor for each point
        idx_arr = torch.tensor(indices, device=x.device)
        rand_col = torch.randint(0, k, (x.shape[0],), device=x.device)
        neighbor_idx = idx_arr[torch.arange(x.shape[0]), rand_col]
        x_neighbor = x[neighbor_idx]

        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam_i = beta.sample((x.shape[0], 1)).to(x.device)
        lam_j = beta.sample((x.shape[0], 1)).to(x.device)
        xi = lam_i * x + (1 - lam_i) * x_neighbor
        xj = lam_j * x + (1 - lam_j) * x_neighbor
        return xi, xj
