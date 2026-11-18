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
    uses_indices:
        ``True`` if :meth:`forward` accepts an ``indices`` keyword: the global training-row
        indices of the batch (for example to look up input-space neighbours). The trainer and
        :class:`~embedkit.improvement.losses.CombinedLoss` pass it only to such losses.
    """

    uses_labels: bool = False
    uses_indices: bool = False

    def precompute(self, X) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Optional hook, called once by the trainer with the training rows ``(N, D)``."""

    @abstractmethod
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
