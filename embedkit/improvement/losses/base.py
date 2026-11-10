"""Abstract base loss."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """Base class for contrastive losses.

    Attributes
    ----------
    uses_labels:
        ``True`` if the loss consumes the ``labels`` argument (supervised losses).
        :class:`~embedkit.improvement.losses.CombinedLoss` forwards labels only to
        members that declare it. Subclasses that need labels must set it to ``True``.
    """

    uses_labels: bool = False

    @abstractmethod
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
