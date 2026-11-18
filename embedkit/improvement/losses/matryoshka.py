"""MatryoshkaLoss: the base loss on nested prefixes of the output (Kusupati et al. 2022)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from embedkit.improvement.losses.base import BaseLoss


class MatryoshkaLoss(BaseLoss):
    """Sum of ``base_loss`` over the prefixes ``z[:, :m]``, ``m`` in ``dims`` (review 3.3, WP 5.3).

    Training this way makes every prefix a usable embedding on its own, so one model serves
    several output dimensions: truncate to ``m`` and renormalise (``EmbedKit.transform(X,
    dim=m)``). Prefixes longer than the output are skipped, and the full output is always
    included, so ``dims`` need not contain it. The base loss must normalise its inputs (the
    contrastive losses in this package do).

    ``weights`` default to 1 for every prefix. ``labels``, ``indices`` and :meth:`precompute`
    are forwarded to the base loss as it declares.
    """

    def __init__(
        self,
        base_loss: BaseLoss,
        dims: Sequence[int] = (16, 32, 64, 128),
        weights: Sequence[float] | None = None,
    ):
        super().__init__()
        dims = [int(m) for m in dims]
        if not dims or min(dims) < 1 or dims != sorted(set(dims)):
            raise ValueError(f"dims must be distinct positive integers in increasing order: {dims}")
        if weights is not None and len(weights) != len(dims):
            raise ValueError("weights must have one entry per element of dims")
        self.base_loss = base_loss
        self.dims = tuple(dims)
        self.weights = tuple(weights) if weights is not None else (1.0,) * len(dims)
        self.uses_labels = getattr(base_loss, "uses_labels", False)
        self.uses_indices = getattr(base_loss, "uses_indices", False)

    def precompute(self, X) -> None:
        if hasattr(self.base_loss, "precompute"):
            self.base_loss.precompute(X)

    def prefixes(self, d: int) -> list[tuple[int, float]]:
        """``(m, weight)`` pairs used for an output of dimension ``d``."""
        out = [(m, w) for m, w in zip(self.dims, self.weights) if m < d]
        full_w = dict(zip(self.dims, self.weights)).get(d, 1.0)
        return out + [(d, full_w)]

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {"indices": indices} if self.uses_indices else {}
        args = (labels,) if self.uses_labels else ()
        total = torch.zeros((), device=z_i.device, dtype=z_i.dtype)
        for m, w in self.prefixes(z_i.shape[1]):
            total = total + w * self.base_loss(z_i[:, :m], z_j[:, :m], *args, **kwargs)
        return total
