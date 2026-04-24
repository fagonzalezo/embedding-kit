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

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "random_choice":
            aug = random.choice(self.augs)
            return aug(x)
        # sequential: chain views through all augmentations
        xi, xj = x, x
        for aug in self.augs:
            xi, _ = aug(xi)
            _, xj = aug(xj)
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
