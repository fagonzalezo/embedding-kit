"""NT-Xent (symmetric InfoNCE) contrastive loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from embedkit.improvement.losses.base import BaseLoss


class NTXentLoss(BaseLoss):
    """NT-Xent over the ``2N`` rows of two views; row ``a`` of one view is the positive of row
    ``a`` of the other, every other row is a negative.

    Parameters
    ----------
    temperature:
        Softmax temperature.
    chunk_size:
        Compute the ``(2N, 2N)`` similarities in row blocks of this size to bound memory.
    neighbor_mask:
        Remove likely false negatives (review 3.4, WP 5.2): an integer ``k``, or a precomputed
        ``(n_train, k)`` neighbour-index array. Rows ``a`` and ``b`` of a batch do not act as
        negatives for each other (in either view) when one sample is among the ``k`` nearest
        input-space neighbours of the other. With an integer, the neighbours are computed by
        :meth:`precompute`, which the trainer calls with the training rows. Needs the global row
        ``indices`` of the batch in :meth:`forward`; without them no mask is applied.
    neighbor_metric:
        Metric for that neighbour search.
    decoupled:
        Decoupled contrastive loss (Yeh et al. 2022): the positive is left out of the
        denominator, ``loss = -s_pos + logsumexp(s_neg)``.
    """

    uses_labels = False

    def __init__(
        self,
        temperature: float = 0.07,
        chunk_size: int | None = None,
        neighbor_mask: int | np.ndarray | None = None,
        neighbor_metric: str = "euclidean",
        decoupled: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.neighbor_metric = neighbor_metric
        self.decoupled = decoupled
        self.neighbor_k: int | None = None
        self.neighbor_index: np.ndarray | None = None
        if isinstance(neighbor_mask, (int, np.integer)) and not isinstance(neighbor_mask, bool):
            if neighbor_mask < 1:
                raise ValueError(f"neighbor_mask must be >= 1, got {neighbor_mask}")
            self.neighbor_k = int(neighbor_mask)
        elif neighbor_mask is not None:
            self.neighbor_index = np.asarray(neighbor_mask, dtype=np.int64)
            if self.neighbor_index.ndim != 2:
                raise ValueError("neighbor_mask array must have shape (n_train, k)")
        self.uses_indices = neighbor_mask is not None

    def precompute(self, X: np.ndarray) -> None:
        """Input-space neighbours of the training rows (called by the trainer)."""
        if self.neighbor_k is None:
            return
        from embedkit.utils.neighbors import knn

        k = min(self.neighbor_k, len(X) - 1)
        _, self.neighbor_index = knn(np.ascontiguousarray(X), k, metric=self.neighbor_metric)

    def _batch_neighbor_mask(self, indices: torch.Tensor, device) -> torch.Tensor | None:
        """``(2N, 2N)`` bool: True where the pair is masked as a negative."""
        if self.neighbor_index is None or indices is None:
            return None
        idx = indices.detach().cpu().numpy()
        nbr = self.neighbor_index[idx]  # (N, k) neighbours of each anchor
        M = (nbr[:, :, None] == idx[None, None, :]).any(axis=1)  # M[a, b]: b in N(a)
        M = M | M.T
        np.fill_diagonal(M, False)  # the positive pairs (a, a + N) stay
        M = torch.from_numpy(M).to(device)
        return torch.cat([torch.cat([M, M], 1), torch.cat([M, M], 1)], 0)

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)  # (2N, d)
        extra = self._batch_neighbor_mask(indices, z.device)
        cs = self.chunk_size if self.chunk_size is not None else 2 * n
        cols = torch.arange(2 * n, device=z.device)
        losses = []
        # Row blocks; one block when chunk_size is None. Averaged over all 2N rows, so a short
        # last chunk is not over-weighted.
        for start in range(0, 2 * n, cs):
            end = min(start + cs, 2 * n)
            rows = torch.arange(start, end, device=z.device)
            logits = torch.mm(z[start:end], z.T) / self.temperature  # (m, 2N)
            tgt = torch.where(rows < n, rows + n, rows - n)  # the other view of each row
            mask = cols[None, :] == rows[:, None]
            if extra is not None:
                mask = mask | extra[start:end]
            if not self.decoupled:
                losses.append(F.cross_entropy(logits.masked_fill(mask, float("-inf")), tgt,
                                              reduction="none"))
                continue
            pos = logits.gather(1, tgt[:, None]).squeeze(1)
            den_mask = mask | (cols[None, :] == tgt[:, None])
            # A row whose negatives are all masked keeps them (else logsumexp is -inf).
            empty = den_mask.all(dim=1)
            if empty.any():
                base = (cols[None, :] == rows[:, None]) | (cols[None, :] == tgt[:, None])
                den_mask = torch.where(empty[:, None], base, den_mask)
            losses.append(torch.logsumexp(logits.masked_fill(den_mask, float("-inf")), 1) - pos)
        return torch.cat(losses).mean()
