"""Gaussian noise and feature dropout augmentations."""

from __future__ import annotations

import numpy as np
import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class GaussianNoise(BaseAugmentation):
    """Add independent Gaussian noise to each view.

    Parameters
    ----------
    std:
        Noise scale. With ``relative=True`` (default) it is a multiple of the per-dimension
        standard deviation of the training data: dimension ``j`` receives noise with standard
        deviation ``std * sigma_j``. The expected noise norm is then about ``std`` times the
        root-mean-square norm of the centred data, whatever the scale or dimensionality of the
        embeddings. With ``relative=False`` it is an absolute standard deviation in every
        dimension.
    relative:
        See ``std``. The per-dimension standard deviations are computed by ``precompute(X)``
        (the trainer calls it with the training set). If ``precompute`` was never called, the
        standard deviations of the current batch are used instead (noisy for small batches).
    adaptive:
        Scale the noise by each point's neighbourhood radius (mean distance to its ``k`` nearest
        neighbours in the batch) times ``std``. Adaptive noise ignores ``relative``.
    k:
        Neighbours for the adaptive radius.
    """

    kind = "view"

    def __init__(
        self,
        std: float = 0.05,
        adaptive: bool = False,
        k: int = 10,
        relative: bool = True,
    ):
        self.std = std
        self.adaptive = adaptive
        self.k = k
        self.relative = relative
        self._radii: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None

    def precompute(self, X: np.ndarray) -> None:
        """Record the per-dimension standard deviation of the training data."""
        X = np.asarray(X, dtype=np.float64)
        self._feature_std = torch.from_numpy(X.std(axis=0).astype(np.float32))

    def _compute_radii(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-point neighborhood radius for adaptive noise."""
        from embedkit.utils.neighbors import knn

        X_np = x.detach().cpu().numpy()
        dists, _ = knn(X_np, min(self.k, X_np.shape[0] - 1))
        radii = torch.tensor(dists.mean(axis=1), device=x.device, dtype=x.dtype)
        return radii.unsqueeze(1)

    def _scale(self, x: torch.Tensor) -> torch.Tensor | float:
        if self.adaptive:
            return self._compute_radii(x) * self.std
        if not self.relative:
            return self.std
        if self._feature_std is not None and self._feature_std.shape[0] == x.shape[-1]:
            sigma = self._feature_std.to(x.device, dtype=x.dtype)
        else:
            sigma = x.std(dim=0, correction=0)
        return sigma * self.std

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(x)
        return x + torch.randn_like(x) * scale, x + torch.randn_like(x) * scale


class FeatureDropout(BaseAugmentation):
    kind = "view"

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        mask_i = (torch.rand_like(x) > self.p).float()
        mask_j = (torch.rand_like(x) > self.p).float()
        return x * mask_i, x * mask_j
