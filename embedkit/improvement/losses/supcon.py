"""Supervised Contrastive Loss (Khosla et al., 2020)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class SupConLoss(BaseLoss):
    """Supervised contrastive loss (Khosla et al., 2020, Eq. 2), optionally with spread.

    Positives are the samples with an equal label (plus the other view of the anchor).
    Labels must be class identifiers of shape ``(N,)`` (or ``(N, 1)``); vector-valued labels
    are rejected with a ``ValueError``. With ``labels=None`` the only positive of each row
    is its other view (SimCLR-style).

    ``spread_weight`` (``w`` in [0, 1]) mixes in the class-conditional InfoNCE term of Chen et al.
    (2022, "Perfectly balanced"): ``(1 - w) * SupCon + w * spread``. In the spread term the only
    positive of a row is its other view, and the candidates are restricted to the samples of the
    same class, so it pushes apart the members of a class while SupCon pulls them together; this
    keeps within-class structure that pure SupCon collapses (review 3.4, WP 6.2). A row whose class
    has no other member in the batch contributes 0 to it. Without labels the spread term is 0.
    """

    uses_labels = True

    def __init__(
        self,
        temperature: float = 0.07,
        chunk_size: int | None = None,
        spread_weight: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= spread_weight <= 1.0:
            raise ValueError(f"spread_weight must be in [0, 1], got {spread_weight}")
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.spread_weight = float(spread_weight)

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)  # (2N, d)

        same = None
        if labels is not None:
            labels = _check_labels(labels, n)
            lbl = torch.cat([labels, labels], dim=0)
            same = lbl.unsqueeze(0) == lbl.unsqueeze(1)
            pos_mask = same.float()
            pos_mask.fill_diagonal_(0.0)
        else:
            pos_mask = torch.zeros(2 * n, 2 * n, device=z.device)
            idx = torch.arange(n, device=z.device)
            pos_mask[idx, idx + n] = 1.0
            pos_mask[idx + n, idx] = 1.0

        w = self.spread_weight if same is not None else 0.0
        cs = self.chunk_size if self.chunk_size is not None else 2 * n
        sup, spread = [], []
        for start in range(0, 2 * n, cs):
            end = min(start + cs, 2 * n)
            rows = torch.arange(start, end, device=z.device)
            logits = torch.mm(z[start:end], z.T) / self.temperature  # (m, 2N)
            diag = torch.zeros_like(logits, dtype=torch.bool)
            diag[torch.arange(end - start, device=z.device), rows] = True
            if w < 1.0:
                masked = logits.masked_fill(diag, float("-inf"))
                # Zero the (-inf) diagonal so 0 * (-inf) = NaN never appears below.
                log_prob = F.log_softmax(masked, dim=1).masked_fill(diag, 0.0)
                pm = pos_mask[start:end]
                sup.append(-(pm * log_prob).sum(dim=1) / pm.sum(dim=1).clamp(min=1))
            if w > 0.0:
                # Candidates: same class, not the row itself; the positive is the other view.
                allowed = same[start:end] & ~diag
                masked = logits.masked_fill(~allowed, float("-inf"))
                partner = (rows + n) % (2 * n)
                pos_logit = logits[torch.arange(end - start, device=z.device), partner]
                spread.append(torch.logsumexp(masked, dim=1) - pos_logit)
        # Means over all rows (not means of chunk means, which overweight a short last chunk).
        loss = (1.0 - w) * torch.cat(sup).mean() if w < 1.0 else 0.0
        if w > 0.0:
            loss = loss + w * torch.cat(spread).mean()
        return loss


def _check_labels(labels: torch.Tensor, n: int) -> torch.Tensor:
    if labels.dim() == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 1:
        raise ValueError(
            "SupConLoss expects 1-D class labels of shape (N,) (or (N, 1)); "
            f"got labels of shape {tuple(labels.shape)}. Encode vector-valued labels as class "
            "ids, or use RankNContrastLoss for continuous or vector targets."
        )
    if labels.shape[0] != n:
        raise ValueError(f"SupConLoss: got {labels.shape[0]} labels for {n} samples")
    return labels
