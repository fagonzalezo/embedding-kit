"""CombinedLoss: weighted sum of multiple losses."""

from __future__ import annotations

import torch

from embedkit.improvement.losses.base import BaseLoss


class CombinedLoss(BaseLoss):
    """Weighted sum of losses.

    ``labels`` are forwarded only to members whose ``uses_labels`` attribute is
    ``True``; the others are called as ``loss(z_i, z_j)``. Exceptions raised by a
    member (for example a label shape error) propagate unchanged. The combined loss
    has ``uses_labels = True`` if any member does.
    """

    def __init__(self, losses: list[tuple[BaseLoss, float]]):
        super().__init__()
        self.loss_fns = torch.nn.ModuleList([fn for fn, _ in losses])
        self.weights = [w for _, w in losses]
        self.uses_labels = any(getattr(fn, "uses_labels", False) for fn, _ in losses)

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        terms = []
        for loss_fn, w in zip(self.loss_fns, self.weights):
            if getattr(loss_fn, "uses_labels", False):
                terms.append(w * loss_fn(z_i, z_j, labels))
            else:
                terms.append(w * loss_fn(z_i, z_j))
        return sum(terms, torch.zeros((), device=z_i.device, dtype=z_i.dtype))
