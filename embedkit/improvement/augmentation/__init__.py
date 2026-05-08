"""Augmentations package."""

from __future__ import annotations

import random
from typing import Literal

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation
from embedkit.improvement.augmentation.noise import GaussianNoise, FeatureDropout
from embedkit.improvement.augmentation.cutout import FeatureMasking
from embedkit.improvement.augmentation.mixup import EmbeddingMixup
from embedkit.improvement.augmentation.knn import KNNPairs


class CompositeAugmentation(BaseAugmentation):
    def __init__(
        self,
        augs: list[BaseAugmentation],
        mode: Literal["sequential", "random_choice"] = "sequential",
    ):
        self.augs = augs
        self.mode = mode

    def precompute(self, X) -> None:
        for aug in self.augs:
            if hasattr(aug, "precompute"):
                aug.precompute(X)

    def __call__(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "random_choice":
            aug = random.choice(self.augs)
            return aug(x, indices)
        # sequential: chain views through all augmentations
        xi, xj = x, x
        for aug in self.augs:
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
