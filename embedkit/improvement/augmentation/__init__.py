"""Augmentations package."""

from __future__ import annotations

from typing import Literal

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation
from embedkit.improvement.augmentation.cutout import FeatureMasking
from embedkit.improvement.augmentation.knn import KNNPairs
from embedkit.improvement.augmentation.mixup import EmbeddingMixup
from embedkit.improvement.augmentation.noise import FeatureDropout, GaussianNoise


class CompositeAugmentation(BaseAugmentation):
    """Combine augmentations.

    ``mode="sequential"``: at most one ``kind == "pair"`` augmentation, which must come first; it
    decides the two samples the views come from. Every ``"view"`` augmentation is then applied
    once to each view, view ``i`` and view ``j`` each with its own independent random draw. A
    violation raises ``ValueError`` at construction time. For example
    ``[KNNPairs(), GaussianNoise()]`` gives a noised anchor and a noised neighbour, while
    ``[GaussianNoise(), KNNPairs()]`` is rejected (the pair step would discard the noise).

    ``mode="random_choice"``: each call applies one member chosen uniformly at random (torch's
    RNG); no ordering rule applies.

    ``kind`` of a composite is ``"pair"`` if any member is a pair augmentation (including nested
    composites) and ``"view"`` otherwise, so nesting is validated by the same rule.
    """

    def __init__(
        self,
        augs: list[BaseAugmentation],
        mode: Literal["sequential", "random_choice"] = "sequential",
    ):
        if mode not in ("sequential", "random_choice"):
            raise ValueError(f"mode must be 'sequential' or 'random_choice', got {mode!r}")
        augs = list(augs)
        if not augs:
            raise ValueError("CompositeAugmentation needs at least one augmentation")
        for aug in augs:
            if aug.kind not in ("pair", "view"):
                raise ValueError(
                    f"{type(aug).__name__}.kind must be 'pair' or 'view', got {aug.kind!r}"
                )
        pair_positions = [i for i, a in enumerate(augs) if a.kind == "pair"]
        if mode == "sequential":
            if len(pair_positions) > 1:
                raise ValueError(
                    "A sequential CompositeAugmentation allows at most one 'pair' augmentation "
                    f"(KNNPairs, EmbeddingMixup), got {len(pair_positions)}"
                )
            if pair_positions and pair_positions[0] != 0:
                raise ValueError(
                    "The 'pair' augmentation must come first in a sequential "
                    "CompositeAugmentation: it builds the second view from another sample, so "
                    "any view augmentation placed before it would be discarded"
                )
        self.augs = augs
        self.mode = mode
        self.kind = "pair" if pair_positions else "view"

    def precompute(self, X) -> None:
        for aug in self.augs:
            aug.precompute(X)

    def __call__(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "random_choice":
            aug = self.augs[int(torch.randint(len(self.augs), (1,)))]
            return aug(x, indices)
        # sequential: the pair augmentation (if any, always first) yields the two base views;
        # each view augmentation then perturbs each view with its own draw.
        xi, xj = x, x
        for aug in self.augs:
            if aug.kind == "pair":
                xi, xj = aug(x, indices)
            else:
                xi, _ = aug(xi, indices)
                _, xj = aug(xj, indices)
        return xi, xj


__all__ = [
    "BaseAugmentation",
    "GaussianNoise",
    "FeatureDropout",
    "FeatureMasking",
    "EmbeddingMixup",
    "KNNPairs",
    "CompositeAugmentation",
]
