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

    For an anchor ``i`` and a candidate ``j != i`` the loss is
    ``-log(exp(s_ij / tau) / sum_{k in S_ij} exp(s_ik / tau))`` with
    ``S_ij = {k != i : d(y_i, y_k) >= d(y_i, y_j)}`` (Zha et al., Eq. 2), averaged over all
    ``(i, j)``. **Tie semantics:** candidates at exactly the same label distance from the
    anchor share one denominator, which contains the whole tie group and every farther
    candidate; the loss therefore does not depend on the order of the rows. Label distances
    are Euclidean and computed exactly, so identical labels tie. To treat nearby continuous
    labels as tied, round them before calling the loss. Each sample appears in two views with
    the same label, so ties always occur.

    Parameters
    ----------
    temperature:
        Cosine-similarity scale factor (lower = sharper distribution).
    chunk_size:
        If set, compute the (2N, 2N) similarity matrix in row-blocks of this
        size to reduce peak memory.  ``None`` materialises the full matrix.
    """

    uses_labels = True

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

        # Expand scalar labels to (2N, 1); vector labels stay (2N, k). Float64 labels keep their
        # precision; everything else is evaluated in at least float32.
        lbl = torch.cat([labels, labels], dim=0)
        lbl = lbl.to(torch.promote_types(lbl.dtype, torch.float32))
        if lbl.dim() == 1:
            lbl = lbl.unsqueeze(-1)  # (2N, 1)

        # Pairwise label distances and the sort order that ranks neighbours from closest to
        # farthest in label space, per anchor.
        D = _label_distances(lbl)  # (2N, 2N)
        eye = torch.eye(two_n, dtype=torch.bool, device=z.device)
        D_off = D[~eye].view(two_n, two_n - 1)  # (2N, 2N-1) — diagonal removed
        idx_D = D_off.argsort(dim=1)             # (2N, 2N-1)
        D_sorted = D_off.gather(1, idx_D).contiguous()
        # First sorted position of each candidate's tie group: the denominator of every
        # member of a tie group starts there (candidates at equal distance share it).
        tie_start = torch.searchsorted(D_sorted, D_sorted, right=False)  # (2N, 2N-1)

        if self.chunk_size is None or self.chunk_size >= two_n:
            return self._loss_full(z, eye, idx_D, tie_start)
        return self._loss_chunked(z, eye, idx_D, tie_start)

    def _loss_full(self, z, eye, idx_D, tie_start):
        two_n = z.shape[0]
        S = torch.mm(z, z.T) / self.temperature          # (2N, 2N)
        S_off = S[~eye].view(two_n, two_n - 1)           # (2N, 2N-1)
        S_sorted = S_off.gather(1, idx_D)                 # (2N, 2N-1) sorted by label dist
        # lse_tail[i, r] = logsumexp(S_sorted[i, r:]) — right-tail in one pass.
        lse_tail = S_sorted.flip(-1).logcumsumexp(-1).flip(-1)
        return (lse_tail.gather(1, tie_start) - S_sorted).mean()

    def _loss_chunked(self, z, eye, idx_D, tie_start):
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
            losses.append(lse_tail.gather(1, tie_start[start:end]) - S_sorted)  # (cs, 2N-1)
        # Mean over all entries, matching _loss_full even when chunk_size does not divide 2N.
        return torch.cat(losses, dim=0).mean()


def _label_distances(lbl: torch.Tensor) -> torch.Tensor:
    """Exact pairwise Euclidean label distances, so that identical labels tie exactly.

    ``torch.cdist`` switches to a matrix-product formula for more than 25 rows, which is
    inexact (identical rows can come out at ~1e-3 instead of 0).
    """
    return torch.cdist(lbl, lbl, compute_mode="donot_use_mm_for_euclid_dist")
