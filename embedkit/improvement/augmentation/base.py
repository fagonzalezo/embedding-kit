"""Abstract base for augmentations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAugmentation(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views of x."""
        ...
