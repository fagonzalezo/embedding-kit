"""Hubness reduction: secondary distances for k-nearest-neighbour search (review 4.4, WP 5.5).

These methods change the distances between points, not their coordinates, so they are not
transforms. Each one is fitted on a reference set (the gallery, or the training split of a kNN
classifier) and then answers neighbour queries against it::

    mp = MutualProximity(metric="cosine").fit(Z_train)
    dist, idx = mp.kneighbors(Z_test, n_neighbors=10)   # neighbours among the Z_train rows
    dist, idx = mp.kneighbors(n_neighbors=10)           # reference rows among each other

All four start from a base distance ``d`` (``metric="euclidean"`` or ``"cosine"``, i.e.
``1 - cos``) and rescale it with statistics of each point's neighbourhood:

- :class:`LocalScaling` (Schnitzer et al. 2012): ``1 - exp(-d^2 / (s_x s_y))``, where ``s_x`` is
  the distance from ``x`` to its ``k``-th nearest reference point.
- :class:`MutualProximity` (Schnitzer et al. 2012, Gaussian variant):
  ``1 - P(D_x > d) P(D_y > d)``, where ``D_x`` is a normal fitted to the distances from ``x``
  to (a sample of) the reference set.
- :class:`CSLS` (Conneau et al. 2018): ``2 d - r_x - r_y``, where ``r_x`` is the mean distance
  from ``x`` to its ``k`` nearest reference points. With the cosine metric the ranking is that
  of the original ``2 cos - r_x - r_y`` (on similarities).
- :class:`DisSimLocal` (Hara et al. 2016): ``|x - y|^2 - |x - c_x|^2 - |y - c_y|^2``, where
  ``c_x`` is the centroid of the ``k`` nearest reference points of ``x``. With the cosine
  metric it runs on L2-normalised vectors.

The search is exact: every query is compared with every reference row, in chunks that bound
memory. A reference row is never its own neighbour when the reference set is queried
(``X=None``); the same holds for the neighbourhood statistics of the reference rows. A query
is a new point, so an exact duplicate of a reference row is returned as a neighbour.

Comparison of these methods: Feldbauer and Flexer (2019).
"""

from __future__ import annotations

import numpy as np

_METRICS = ("euclidean", "cosine")
_CHUNK_ENTRIES = 4_000_000  # base-distance entries computed at once (32 MB in float64)


def _as_float64(X) -> np.ndarray:
    try:
        import torch

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().double().numpy()
    except ImportError:  # pragma: no cover
        pass
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {X.shape}")
    return X


def _topk(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` smallest entries of each row of ``D``, sorted: ``(values, columns)``."""
    if k < D.shape[1]:
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
    else:
        part = np.broadcast_to(np.arange(D.shape[1]), D.shape).copy()
    vals = np.take_along_axis(D, part, axis=1)
    order = np.argsort(vals, axis=1, kind="stable")
    return np.take_along_axis(vals, order, axis=1), np.take_along_axis(part, order, axis=1)


class _HubnessReduction:
    """Shared fitting, chunked search and parameter handling.

    Subclasses implement ``_fit_stats(R)``, the reference statistics, and
    ``_secondary(D, Q, rows, self_query)``, the ranking key for a chunk of queries ``Q`` with
    base distances ``D`` (shape ``(m, n_reference)``). ``rows`` are the reference indices of the
    queries when ``self_query`` is true. ``_key_to_distance`` turns ranking keys into the
    reported distances (identity by default).
    """

    _param_names: tuple[str, ...] = ("metric",)

    def __init__(self, metric: str = "euclidean"):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
        self.metric = metric

    # ------------------------------------------------------------------------------ params
    def get_params(self) -> dict:
        return {name: getattr(self, name) for name in self._param_names}

    def clone(self, **overrides) -> _HubnessReduction:
        """A new, unfitted instance with the same parameters, except ``overrides``."""
        return type(self)(**{**self.get_params(), **overrides})

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    # ----------------------------------------------------------------------------- helpers
    def _prepare(self, X) -> np.ndarray:
        X = _as_float64(X)
        if self.metric == "cosine":
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X

    def _base(self, Q: np.ndarray, columns: np.ndarray | None = None) -> np.ndarray:
        """Base distances from the (prepared) rows ``Q`` to the reference rows ``columns``
        (all of them by default)."""
        R = self.reference_ if columns is None else self.reference_[columns]
        G = Q @ R.T
        if self.metric == "cosine":
            return np.maximum(1.0 - G, 0.0)
        r_sq = self._ref_sq if columns is None else self._ref_sq[columns]
        sq = np.einsum("ij,ij->i", Q, Q)[:, None] - 2.0 * G + r_sq[None, :]
        return np.sqrt(np.maximum(sq, 0.0))

    def _chunks(self, n: int):
        step = max(1, _CHUNK_ENTRIES // max(len(self.reference_), 1))
        for start in range(0, n, step):
            yield start, min(n, start + step)

    def _reference_knn(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Base-distance ``k`` nearest neighbours of each reference row among the others."""
        R = self.reference_
        dist, idx = np.empty((len(R), k)), np.empty((len(R), k), dtype=np.int64)
        for start, stop in self._chunks(len(R)):
            D = self._base(R[start:stop])
            rows = np.arange(stop - start)
            D[rows, start + rows] = np.inf
            dist[start:stop], idx[start:stop] = _topk(D, k)
        return dist, idx

    def _check_k(self, k: int, n: int) -> int:
        k = int(k)
        if not 1 <= k <= n - 1:
            raise ValueError(f"k must be between 1 and n_reference - 1 = {n - 1}, got {k}")
        return k

    # ------------------------------------------------------------------------------ public
    def fit(self, X, y=None):
        """Store the reference set and compute its neighbourhood statistics."""
        R = self._prepare(X)
        if len(R) < 2:
            raise ValueError(f"Need at least 2 reference rows, got {len(R)}")
        self.reference_ = R
        self._ref_sq = np.einsum("ij,ij->i", R, R)
        self.n_features_in_ = R.shape[1]
        self._fit_stats(R)
        return self

    def kneighbors(self, X=None, n_neighbors: int = 5, return_distance: bool = True):
        """The ``n_neighbors`` nearest reference rows of each query under the secondary distance.

        ``X=None`` queries the reference rows themselves, each excluding itself.

        Returns ``(distances, indices)``, both of shape ``(n_queries, n_neighbors)`` and sorted
        by increasing distance, or only ``indices`` with ``return_distance=False``.
        """
        if not hasattr(self, "reference_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        self_query = X is None
        Q = self.reference_ if self_query else self._prepare(X)
        if Q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {Q.shape[1]} features, the reference set has {self.n_features_in_}"
            )
        n_ref = len(self.reference_)
        max_k = n_ref - 1 if self_query else n_ref
        n_neighbors = int(n_neighbors)
        if not 1 <= n_neighbors <= max_k:
            raise ValueError(f"n_neighbors must be between 1 and {max_k}, got {n_neighbors}")
        keys = np.empty((len(Q), n_neighbors))
        idx = np.empty((len(Q), n_neighbors), dtype=np.int64)
        for start, stop in self._chunks(len(Q)):
            D = self._base(Q[start:stop])
            rows = np.arange(start, stop)
            S = self._secondary(D, Q[start:stop], rows, self_query)
            if self_query:
                S[rows - start, rows] = np.inf
            keys[start:stop], idx[start:stop] = _topk(S, n_neighbors)
        if not return_distance:
            return idx
        return self._key_to_distance(keys), idx

    def _key_to_distance(self, keys: np.ndarray) -> np.ndarray:
        return keys

    def _fit_stats(self, R: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _secondary(self, D, Q, rows, self_query):  # pragma: no cover - abstract
        raise NotImplementedError


class LocalScaling(_HubnessReduction):
    """Local scaling (Schnitzer et al. 2012): ``1 - exp(-d^2 / (s_x s_y))``.

    ``s_x`` is the base distance from ``x`` to its ``k``-th nearest reference point.
    """

    _param_names = ("k", "metric")

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        super().__init__(metric)
        self.k = k

    def _fit_stats(self, R):
        k = self._check_k(self.k, len(R))
        self.scale_ = self._reference_knn(k)[0][:, -1]

    def _secondary(self, D, Q, rows, self_query):
        s_q = self.scale_[rows] if self_query else _topk(D, self.k)[0][:, -1]
        denom = np.maximum(s_q[:, None] * self.scale_[None, :], 1e-300)
        return -np.expm1(-(D**2) / denom)


class MutualProximity(_HubnessReduction):
    """Mutual proximity, Gaussian variant (Schnitzer et al. 2012): ``1 - P(D_x > d) P(D_y > d)``.

    ``D_x`` is a normal distribution with the mean and standard deviation of the base distances
    from ``x`` to ``sample_size`` reference rows drawn once with ``random_state`` (all rows when
    ``None`` or when the reference set is smaller). Neighbours are ranked on
    ``-log P(D_x > d) - log P(D_y > d)``, which orders the same way and does not saturate at 1
    for distant pairs; the reported distances are the mutual-proximity values.
    """

    _param_names = ("metric", "sample_size", "random_state")

    def __init__(self, metric: str = "euclidean", sample_size: int | None = 2000,
                 random_state: int | None = 0):
        super().__init__(metric)
        if sample_size is not None and int(sample_size) < 2:
            raise ValueError(f"sample_size must be at least 2, got {sample_size}")
        self.sample_size = sample_size
        self.random_state = random_state

    def _fit_stats(self, R):
        n = len(R)
        if self.sample_size is None or n <= self.sample_size:
            self.sample_ = np.arange(n)
        else:
            rng = np.random.default_rng(self.random_state)
            self.sample_ = np.sort(rng.choice(n, size=int(self.sample_size), replace=False))
        pos = np.full(n, -1)
        pos[self.sample_] = np.arange(len(self.sample_))
        mu, sd = np.empty(n), np.empty(n)
        for start, stop in self._chunks(n):
            D = self._base(R[start:stop], self.sample_)
            rows = np.arange(start, stop)
            in_sample = pos[rows] >= 0
            # A reference row's distance to itself is not part of its distribution.
            D[np.flatnonzero(in_sample), pos[rows[in_sample]]] = np.nan
            mu[start:stop] = np.nanmean(D, axis=1)
            sd[start:stop] = np.nanstd(D, axis=1)
        self.mean_, self.std_ = mu, np.maximum(sd, 1e-12)

    def _secondary(self, D, Q, rows, self_query):
        from scipy.special import log_ndtr

        if self_query:
            mu_q, sd_q = self.mean_[rows], self.std_[rows]
        else:
            Ds = D[:, self.sample_]
            mu_q, sd_q = Ds.mean(axis=1), np.maximum(Ds.std(axis=1), 1e-12)
        log_sf_q = log_ndtr(-(D - mu_q[:, None]) / sd_q[:, None])
        log_sf_r = log_ndtr(-(D - self.mean_[None, :]) / self.std_[None, :])
        return -(log_sf_q + log_sf_r)

    def _key_to_distance(self, keys):
        return -np.expm1(-keys)


class CSLS(_HubnessReduction):
    """Cross-domain similarity local scaling (Conneau et al. 2018), as a distance:
    ``2 d - r_x - r_y``.

    ``r_x`` is the mean base distance from ``x`` to its ``k`` nearest reference points. With
    ``metric="cosine"`` this is ``-(2 cos - r^cos_x - r^cos_y)``, the original score negated.
    The values can be negative.
    """

    _param_names = ("k", "metric")

    def __init__(self, k: int = 10, metric: str = "cosine"):
        super().__init__(metric)
        self.k = k

    def _fit_stats(self, R):
        k = self._check_k(self.k, len(R))
        self.radius_ = self._reference_knn(k)[0].mean(axis=1)

    def _secondary(self, D, Q, rows, self_query):
        r_q = self.radius_[rows] if self_query else _topk(D, self.k)[0].mean(axis=1)
        return 2.0 * D - r_q[:, None] - self.radius_[None, :]


class DisSimLocal(_HubnessReduction):
    """DisSimLocal (Hara et al. 2016): ``|x - y|^2 - |x - c_x|^2 - |y - c_y|^2``.

    ``c_x`` is the centroid of the ``k`` nearest reference points of ``x``. The computation is
    Euclidean; ``metric="cosine"`` applies it to L2-normalised vectors. The values can be
    negative.
    """

    _param_names = ("k", "metric")

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        super().__init__(metric)
        self.k = k

    def _sq(self, D):
        # Squared Euclidean distance; on unit vectors |x - y|^2 = 2 (1 - cos).
        return D**2 if self.metric == "euclidean" else 2.0 * D

    def _fit_stats(self, R):
        k = self._check_k(self.k, len(R))
        _, idx = self._reference_knn(k)
        centroids = R[idx].mean(axis=1)
        self.offset_ = np.sum((R - centroids) ** 2, axis=1)

    def _secondary(self, D, Q, rows, self_query):
        if self_query:
            off_q = self.offset_[rows]
        else:
            _, idx = _topk(D, self.k)
            off_q = np.sum((Q - self.reference_[idx].mean(axis=1)) ** 2, axis=1)
        return self._sq(D) - off_q[:, None] - self.offset_[None, :]


HUBNESS_REDUCTIONS: dict[str, type[_HubnessReduction]] = {
    "local_scaling": LocalScaling,
    "mutual_proximity": MutualProximity,
    "csls": CSLS,
    "dissimlocal": DisSimLocal,
}

__all__ = ["CSLS", "DisSimLocal", "HUBNESS_REDUCTIONS", "LocalScaling", "MutualProximity"]
