"""Abstract base for augmentations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAugmentation(ABC):
    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views of x.

        indices: optional global row indices (shape B,) used by KNNPairs for
        dataset-level neighbor lookup. Other augmentations ignore it.
        """
        ...
