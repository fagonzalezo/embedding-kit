"""Feature masking (cutout-style) augmentation."""

from __future__ import annotations

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class FeatureMasking(BaseAugmentation):
    def __init__(self, mask_ratio: float = 0.15, block_size: int = 1):
        self.mask_ratio = mask_ratio
        self.block_size = block_size

    def _make_mask(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        mask = torch.ones(n, d, device=x.device, dtype=x.dtype)
        n_blocks = max(1, int(d * self.mask_ratio / self.block_size))
        for _ in range(n_blocks):
            start = torch.randint(0, max(1, d - self.block_size + 1), (1,)).item()
            mask[:, start : start + self.block_size] = 0.0
        return mask

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        return x * self._make_mask(x), x * self._make_mask(x)
