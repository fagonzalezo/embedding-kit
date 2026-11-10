"""Shared exact kNN computation.

Contract of :func:`knn`:

- returns ``(distances, indices)``, both of shape ``(n, k)``;
- distances are true distances for every backend: Euclidean distance (not squared) for
  ``metric="euclidean"``, cosine distance ``1 - cos`` for ``metric="cosine"``; each row is
  sorted in ascending order;
- the query point itself is excluded *by index*, so a point is never its own neighbour, even
  when the data contain exact duplicate rows;
- the input array is never modified;
- nothing is cached. Callers that need the same graph twice compute it once and pass it on.

Backends: scikit-learn is the default. FAISS (``pip install embedkit[faiss]``) is opt-in with
``backend="faiss"`` or the environment variable ``EMBEDKIT_KNN_BACKEND=faiss``. Both are exact
searches; FAISS is only faster.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

_BACKENDS = ("sklearn", "faiss")
_ENV_VAR = "EMBEDKIT_KNN_BACKEND"


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        backend = os.environ.get(_ENV_VAR, "sklearn").strip().lower() or "sklearn"
        if backend == "auto":
            backend = "sklearn"
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown kNN backend {backend!r}. Choose from {('auto',) + _BACKENDS}.")
    return backend


def knn(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours of every row of ``X`` among the other rows.

    Parameters
    ----------
    X:
        Array of shape ``(n, d)``. It is not modified.
    k:
        Number of neighbours, ``1 <= k <= n - 1``.
    metric:
        ``"euclidean"`` or ``"cosine"`` for every backend. The scikit-learn backend also
        accepts any metric that ``sklearn.neighbors.NearestNeighbors`` accepts.
    backend:
        ``"auto"`` (scikit-learn, unless ``EMBEDKIT_KNN_BACKEND`` says otherwise),
        ``"sklearn"`` or ``"faiss"``.

    Returns
    -------
    distances : float32 array of shape ``(n, k)``, ascending along each row.
    indices : int64 array of shape ``(n, k)``. Row ``i`` never contains ``i``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {X.shape}")
    n = X.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be between 1 and n - 1 = {n - 1}, got {k}")

    resolved = _resolve_backend(backend)
    if resolved == "faiss":
        distances, indices = _search_faiss(X, k + 1, metric)
    else:
        distances, indices = _search_sklearn(X, k + 1, metric)
    return _drop_self(distances, indices, k)


def _drop_self(distances: np.ndarray, indices: np.ndarray, k: int):
    """Remove the query from each row of a (k + 1)-neighbour search.

    The entry equal to the row index is dropped when present; otherwise (the query was pushed
    out by exact duplicates) the last entry is dropped.
    """
    n = indices.shape[0]
    is_self = indices == np.arange(n)[:, None]
    has_self = is_self.any(axis=1)
    # For rows without the query, mark the last column for removal.
    is_self[~has_self, -1] = True
    # Keep only the first marked entry per row (a row cannot contain its own index twice,
    # but be defensive).
    first = is_self.argmax(axis=1)
    keep = np.ones_like(is_self)
    keep[np.arange(n), first] = False
    distances = distances[keep].reshape(n, k)
    indices = indices[keep].reshape(n, k)
    return distances.astype(np.float32, copy=False), indices.astype(np.int64, copy=False)


def _search_sklearn(X: np.ndarray, n_neighbors: int, metric: str):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    if metric == "cosine":
        # Rounding can give tiny negative values for identical directions.
        np.maximum(distances, 0.0, out=distances)
    return distances, indices


def _search_faiss(X: np.ndarray, n_neighbors: int, metric: str):
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='faiss' needs the faiss package. Install it with "
            "`pip install embedkit[faiss]` (or faiss-cpu), or use backend='sklearn'."
        ) from exc

    if metric not in ("euclidean", "cosine"):
        warnings.warn(
            f"The FAISS backend supports 'euclidean' and 'cosine'; using scikit-learn for "
            f"metric={metric!r}.",
            stacklevel=3,
        )
        return _search_sklearn(X, n_neighbors, metric)

    # Always copy: faiss.normalize_L2 works in place and must not touch the caller's array.
    X32 = np.array(X, dtype=np.float32, order="C", copy=True)
    d = X32.shape[1]
    if metric == "euclidean":
        index = faiss.IndexFlatL2(d)
        index.add(X32)
        sq, indices = index.search(X32, n_neighbors)
        distances = np.sqrt(np.maximum(sq, 0.0))
    else:
        faiss.normalize_L2(X32)
        index = faiss.IndexFlatIP(d)
        index.add(X32)
        sims, indices = index.search(X32, n_neighbors)
        distances = np.maximum(1.0 - sims, 0.0)  # inner products are sorted descending
    return distances, indices


def clear_cache() -> None:
    """Deprecated no-op. :func:`knn` no longer caches results."""
    warnings.warn(
        "embedkit.utils.neighbors.clear_cache() is deprecated and does nothing: knn() no "
        "longer caches results. It will be removed in the next release.",
        DeprecationWarning,
        stacklevel=2,
    )
