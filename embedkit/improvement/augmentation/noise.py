"""Gaussian noise and feature dropout augmentations."""

from __future__ import annotations

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class GaussianNoise(BaseAugmentation):
    def __init__(
        self,
        std: float = 0.05,
        adaptive: bool = False,
        k: int = 10,
    ):
        self.std = std
        self.adaptive = adaptive
        self.k = k
        self._radii: torch.Tensor | None = None

    def _compute_radii(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-point neighborhood radius for adaptive noise."""
        from embedkit.utils.neighbors import knn
        import numpy as np
        X_np = x.detach().cpu().numpy()
        dists, _ = knn(X_np, self.k)
        radii = torch.tensor(dists.mean(axis=1), device=x.device, dtype=x.dtype)
        return radii.unsqueeze(1)

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.adaptive:
            radii = self._compute_radii(x)
            noise_i = torch.randn_like(x) * radii * self.std
            noise_j = torch.randn_like(x) * radii * self.std
        else:
            noise_i = torch.randn_like(x) * self.std
            noise_j = torch.randn_like(x) * self.std
        return x + noise_i, x + noise_j


class FeatureDropout(BaseAugmentation):
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        mask_i = (torch.rand_like(x) > self.p).float()
        mask_j = (torch.rand_like(x) > self.p).float()
        return x * mask_i, x * mask_j
