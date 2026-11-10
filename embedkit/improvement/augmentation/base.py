"""Abstract base for augmentations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch


class BaseAugmentation(ABC):
    """Produces two views ``(x_i, x_j)`` of a batch.

    Every augmentation has a ``kind`` that tells :class:`CompositeAugmentation` how to chain it:

    ``"pair"``
        Builds the second view from *another sample* (a neighbour): ``KNNPairs``,
        ``EmbeddingMixup``. A pair augmentation defines which two samples the views come from, so
        it must run first.
    ``"view"``
        Perturbs each view independently (noise, dropout, masking). When used on its own it
        returns two independent perturbations of ``x``; inside a composite it is applied once to
        each existing view, each with its own random draw.

    Subclasses that do not set ``kind`` are treated as ``"view"``.
    """

    kind: ClassVar[str] = "view"

    def precompute(self, X: np.ndarray) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Optional hook, called once by the trainer with the full training matrix ``(N, D)``."""

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
