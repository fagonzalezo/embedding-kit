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

    # -- random number generation ------------------------------------------------------------
    # The trainer calls set_rng() with generators derived from its random_state, so that
    # augmentation draws are reproducible and never touch the global NumPy or torch state.
    # Without set_rng() the helpers below fall back to the global torch / NumPy RNG.

    def set_rng(
        self,
        np_rng: np.random.Generator | None = None,
        torch_gen: torch.Generator | None = None,
    ) -> None:
        """Use these generators for every random draw. ``torch_gen`` must be a CPU generator."""
        self._np_rng = np_rng
        self._torch_gen = torch_gen

    @property
    def np_rng(self) -> np.random.Generator:
        rng = getattr(self, "_np_rng", None)
        if rng is None:
            return np.random.default_rng(int(torch.randint(2**31 - 1, (1,))))
        return rng

    def _randn(self, like: torch.Tensor) -> torch.Tensor:
        """Standard normal tensor with the shape, dtype and device of ``like``."""
        gen = getattr(self, "_torch_gen", None)
        if gen is None:
            return torch.randn_like(like)
        return torch.randn(like.shape, generator=gen, dtype=like.dtype).to(like.device)

    def _rand(self, like: torch.Tensor) -> torch.Tensor:
        """Uniform [0, 1) tensor with the shape, dtype and device of ``like``."""
        gen = getattr(self, "_torch_gen", None)
        if gen is None:
            return torch.rand_like(like)
        return torch.rand(like.shape, generator=gen, dtype=like.dtype).to(like.device)

    def _randint(self, high: int, size: tuple[int, ...], device=None) -> torch.Tensor:
        """Integers in ``[0, high)``; on the CPU unless ``device`` is given."""
        gen = getattr(self, "_torch_gen", None)
        out = torch.randint(0, high, size, generator=gen)
        return out if device is None else out.to(device)

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
