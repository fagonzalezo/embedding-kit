"""IdentityPair: two identical views, so the positives differ only by the model's dropout."""

from __future__ import annotations

import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class IdentityPair(BaseAugmentation):
    """Returns ``(x, x)``.

    The two views pass through the refiner separately in training mode, so they differ only by
    its dropout masks: the SimCSE positives of Gao et al. (2021) (review 3.4, WP 5.2). With the
    residual architecture the dropout acts inside the MLP branch only, whose output starts at
    zero. Inside a :class:`CompositeAugmentation` it is a pair augmentation (it fixes the sample
    both views come from), and view augmentations may follow it.
    """

    kind = "pair"

    def __call__(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x
