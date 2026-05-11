"""Rank-N-Contrast loss for supervised refinement with continuous labels.

Zha et al., "Rank-N-Contrast: Learning Continuous Representations for
Regression", NeurIPS 2023.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class RankNContrastLoss(BaseLoss):
    """Contrastive loss for regression targets (scalar or vector labels).

    For each anchor, samples closer in label space must be more similar in
    embedding space than those farther away.  No kernel bandwidth to tune —
    the ranking is distribution-free over the label values.

    Parameters
    ----------
    temperature:
        Cosine-similarity scale factor (lower = sharper distribution).
    chunk_size:
        If set, compute the (2N, 2N) similarity matrix in row-blocks of this
        size to reduce peak memory.  ``None`` materialises the full matrix.
    """

    def __init__(self, temperature: float = 0.07, chunk_size: int | None = None):
        super().__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError("RankNContrastLoss requires labels")

        n = z_i.shape[0]
        two_n = 2 * n
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)  # (2N, d)

        # Expand scalar labels to (2N, 1); vector labels stay (2N, k).
        lbl = torch.cat([labels, labels], dim=0).float()
        if lbl.dim() == 1:
            lbl = lbl.unsqueeze(-1)  # (2N, 1)

        # Pairwise label distances and the sort order that ranks neighbours
        # from closest to farthest in label space, per anchor.
        D = torch.cdist(lbl, lbl)  # (2N, 2N)
        eye = torch.eye(two_n, dtype=torch.bool, device=z.device)
        D_off = D[~eye].view(two_n, two_n - 1)  # (2N, 2N-1) — diagonal removed
        idx_D = D_off.argsort(dim=1)             # (2N, 2N-1)

        if self.chunk_size is None or self.chunk_size >= two_n:
            return self._loss_full(z, eye, idx_D)
        return self._loss_chunked(z, eye, idx_D)

    def _loss_full(self, z, eye, idx_D):
        two_n = z.shape[0]
        S = torch.mm(z, z.T) / self.temperature          # (2N, 2N)
        S_off = S[~eye].view(two_n, two_n - 1)           # (2N, 2N-1)
        S_sorted = S_off.gather(1, idx_D)                 # (2N, 2N-1) sorted by label dist
        # lse_tail[i, j] = logsumexp(S_sorted[i, j:]) — right-tail in one pass.
        lse_tail = S_sorted.flip(-1).logcumsumexp(-1).flip(-1)
        return (lse_tail - S_sorted).mean()

    def _loss_chunked(self, z, eye, idx_D):
        """Row-block RNC to avoid materialising full (2N)×(2N) sim matrix."""
        two_n = z.shape[0]
        cs = self.chunk_size
        losses = []
        for start in range(0, two_n, cs):
            end = min(start + cs, two_n)
            rows = z[start:end]
            logits = torch.mm(rows, z.T) / self.temperature  # (cs, 2N)
            # Remove the self-similarity column from each row in the chunk.
            eye_chunk = eye[start:end]                        # (cs, 2N)
            S_off_chunk = logits[~eye_chunk].view(end - start, two_n - 1)
            S_sorted = S_off_chunk.gather(1, idx_D[start:end])
            lse_tail = S_sorted.flip(-1).logcumsumexp(-1).flip(-1)
            losses.append((lse_tail - S_sorted).mean())
        return torch.stack(losses).mean()
