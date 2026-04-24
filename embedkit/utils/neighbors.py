"""Shared kNN computation with optional FAISS backend and simple result caching."""

from __future__ import annotations

import numpy as np

_cache: dict = {}


def knn(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) arrays of shape (n_samples, k).

    Uses FAISS when installed and backend='auto', otherwise sklearn.
    Results are cached by (id(X), k, metric) so repeated calls on the same
    array within a session are free.
    """
    cache_key = (id(X), k, metric)
    if cache_key in _cache:
        return _cache[cache_key]

    use_faiss = False
    if backend in ("auto", "faiss"):
        try:
            import faiss  # noqa: F401
            use_faiss = True
        except ImportError:
            if backend == "faiss":
                raise
            use_faiss = False

    if use_faiss:
        result = _knn_faiss(X, k, metric)
    else:
        result = _knn_sklearn(X, k, metric)

    _cache[cache_key] = result
    return result


def _knn_sklearn(X: np.ndarray, k: int, metric: str):
    from sklearn.neighbors import NearestNeighbors
    # Request k+1 to exclude self (index 0 is always the point itself when querying training data)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.int64)


def _knn_faiss(X: np.ndarray, k: int, metric: str):
    import faiss
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    d = X32.shape[1]
    if metric == "euclidean":
        index = faiss.IndexFlatL2(d)
    elif metric in ("cosine", "ip"):
        faiss.normalize_L2(X32)
        index = faiss.IndexFlatIP(d)
    else:
        # fallback for unsupported metrics
        return _knn_sklearn(X, k, metric)
    index.add(X32)
    distances, indices = index.search(X32, k + 1)
    # remove self
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.int64)


def clear_cache() -> None:
    _cache.clear()
