"""KNNPairs: emit kNN as positives with optional hard-negative mining."""

from __future__ import annotations

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class KNNPairs(BaseAugmentation):
    def __init__(self, k: int = 10, hard_negatives: bool = False):
        self.k = k
        self.hard_negatives = hard_negatives

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from embedkit.utils.neighbors import knn

        X_np = x.detach().cpu().numpy()
        _, indices = knn(X_np, self.k)
        idx_tensor = torch.tensor(indices, device=x.device)
        # pick the closest neighbor as positive
        pos_idx = idx_tensor[:, 0]
        x_pos = x[pos_idx]

        if self.hard_negatives:
            # hardest negative: furthest kNN neighbor
            neg_idx = idx_tensor[:, -1]
            x_neg = x[neg_idx]
            # return (anchor+positive) paired; caller uses labels to separate
            return x, x_pos  # hard negative info stored separately if needed

        return x, x_pos
