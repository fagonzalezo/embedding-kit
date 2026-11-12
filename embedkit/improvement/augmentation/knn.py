"""KNNPairs: emit kNN as positives with optional precomputed neighbor index."""

from __future__ import annotations

import numpy as np
import torch

from embedkit.improvement.augmentation.base import BaseAugmentation


class KNNPairs(BaseAugmentation):
    """The anchor and one of its k nearest neighbours form the two views.

    With ``precompute`` (the trainer calls it) neighbours come from the whole training set and
    are looked up by the global row ``indices`` of the batch. Without it, neighbours are searched
    among the rows of the batch itself (``k`` is capped at ``batch_size - 1``). In both paths one
    of the k neighbours is drawn uniformly at random for every anchor, using torch's RNG.

    Parameters
    ----------
    k:
        Number of neighbours to sample from.
    neighbor_index:
        Optional precomputed ``(N, k)`` neighbour matrix; ``precompute`` overwrites it.
    metric:
        ``"euclidean"`` or ``"cosine"``, used for the neighbour search in both paths.
    mode:
        ``"all"`` (default): any of the k neighbours. ``"mutual"``: only mutual neighbours (j is
        among the k neighbours of i and i among those of j), which removes most cross-cluster
        pairs at hubs (review 3.4, WP 5.2). Points with no mutual neighbour fall back to their
        nearest neighbour.
    """

    kind = "pair"

    def __init__(
        self,
        k: int = 10,
        neighbor_index: np.ndarray | None = None,
        metric: str = "euclidean",
        mode: str = "all",
    ):
        if mode not in ("all", "mutual"):
            raise ValueError(f"mode must be 'all' or 'mutual', got {mode!r}")
        self.k = k
        self.neighbor_index = neighbor_index
        self.metric = metric
        self.mode = mode
        self._allowed: np.ndarray | None = None  # (N, k) mask of usable neighbours, mutual mode
        self._X_ref: torch.Tensor | None = None  # full training set, set by precompute

    def precompute(self, X: np.ndarray) -> None:
        """Build and cache the neighbor index and reference embeddings for X."""
        from embedkit.utils.neighbors import knn

        X = np.ascontiguousarray(X)
        _, indices = knn(X, min(self.k, len(X) - 1), metric=self.metric)
        self.neighbor_index = indices  # (N, k) int64
        self._allowed = self._allowed_mask(indices) if self.mode == "mutual" else None
        self._X_ref = torch.from_numpy(X)  # kept on CPU; moved to device on demand

    @staticmethod
    def _allowed_mask(nbr: np.ndarray) -> np.ndarray:
        """``(N, k)`` bool: neighbour ``nbr[i, c]`` is mutual; rows with none keep column 0."""
        rows = np.arange(len(nbr))[:, None, None]
        allowed = (nbr[nbr] == rows).any(axis=2)  # i in the neighbour list of nbr[i, c]
        allowed[~allowed.any(axis=1), 0] = True
        return allowed

    def _pick_columns(self, allowed: np.ndarray | None, b: int, k: int) -> np.ndarray:
        """One column per row, uniform over the allowed neighbours (all if ``allowed`` is None)."""
        if allowed is None:
            return self._randint(k, (b,)).numpy()
        keys = self._rand(torch.empty(b, k)).numpy()
        return np.where(allowed, keys, -1.0).argmax(axis=1)

    def __call__(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.neighbor_index is not None and indices is not None and self._X_ref is not None:
            return self._from_precomputed(x, indices)
        return self._from_batch(x)

    def _from_precomputed(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_np = indices.cpu().numpy()
        # Randomly pick one of the k neighbors for each anchor.
        allowed = self._allowed[idx_np] if self._allowed is not None else None
        rand_col = self._pick_columns(allowed, len(idx_np), self.neighbor_index.shape[1])
        neighbor_global = self.neighbor_index[idx_np, rand_col]  # (B,)
        x_j = self._X_ref[torch.from_numpy(neighbor_global)].to(x.device, dtype=x.dtype)
        return x, x_j

    def _from_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from embedkit.utils.neighbors import knn

        b = x.shape[0]
        if b < 2:
            raise ValueError("KNNPairs needs at least 2 rows in a batch to find a neighbour")
        k = min(self.k, b - 1)
        _, indices = knn(x.detach().cpu().numpy(), k, metric=self.metric)
        allowed = self._allowed_mask(indices) if self.mode == "mutual" else None
        rand_col = self._pick_columns(allowed, b, k)
        chosen = torch.from_numpy(indices[np.arange(b), rand_col]).to(x.device)
        return x, x[chosen]
