"""Feature masking (cutout-style) augmentation."""

from __future__ import annotations

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class FeatureMasking(BaseAugmentation):
    """Zero out a fraction of the features of every row.

    A separate mask is drawn for every row and for each of the two views.

    Parameters
    ----------
    mask_ratio:
        Fraction of the ``d`` features masked per row. ``max(1, round(d * mask_ratio))``
        features are masked (blocks, when ``block_size > 1``).
    block_size:
        With ``block_size=1`` (default) the masked features are distinct, uniformly chosen, so
        exactly ``max(1, round(d * mask_ratio))`` are masked per row. With ``block_size > 1``,
        ``max(1, round(d * mask_ratio / block_size))`` contiguous blocks of that many features
        start at independent random positions (blocks may overlap, so slightly fewer features can
        end up masked). Blocks are only meaningful when neighbouring features are related, for
        example ordered or spatial features; for generic embedding coordinates keep 1.
    """

    kind = "view"

    def __init__(self, mask_ratio: float = 0.15, block_size: int = 1):
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.mask_ratio = mask_ratio
        self.block_size = block_size

    def _make_mask(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        bs = min(self.block_size, d)
        n_blocks = max(1, round(d * self.mask_ratio / bs))
        if bs == 1:
            n_blocks = min(n_blocks, d)
            # the n_blocks smallest of d uniform draws: a uniform subset, independent per row
            u = self._rand(torch.empty(n, d, device=x.device))
            cols = u.topk(n_blocks, dim=1, largest=False).indices
            masked = torch.zeros(n, d, device=x.device, dtype=torch.bool)
            masked.scatter_(1, cols, True)
        else:
            starts = self._randint(d - bs + 1, (n, n_blocks), device=x.device)
            # difference array: +1 where a block starts, -1 just after it ends
            delta = torch.zeros(n, d + 1, device=x.device, dtype=torch.int32)
            delta.scatter_add_(1, starts, torch.ones_like(starts, dtype=torch.int32))
            delta.scatter_add_(1, starts + bs, -torch.ones_like(starts, dtype=torch.int32))
            masked = delta.cumsum(dim=1)[:, :d] > 0
        return (~masked).to(x.dtype)

    def __call__(self, x: torch.Tensor, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        return x * self._make_mask(x), x * self._make_mask(x)
