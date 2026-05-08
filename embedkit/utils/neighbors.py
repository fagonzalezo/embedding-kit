"""Shared kNN computation with optional FAISS backend and bounded result cache."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

_MAX_CACHE = 8
_cache: OrderedDict = OrderedDict()


def knn(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) arrays of shape (n_samples, k).

    Uses FAISS when installed and backend='auto', otherwise sklearn.
    Results are cached by array address+shape+dtype so repeated calls on the
    same array within a session are free. Cache is bounded to _MAX_CACHE entries
    (LRU eviction) to prevent unbounded memory growth.
    """
    cache_key = (X.ctypes.data, X.shape, X.dtype, k, metric)
    if cache_key in _cache:
        _cache.move_to_end(cache_key)
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
    if len(_cache) > _MAX_CACHE:
        _cache.popitem(last=False)
    return result


def _knn_sklearn(X: np.ndarray, k: int, metric: str):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto", n_jobs=-1)
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
        return _knn_sklearn(X, k, metric)
    index.add(X32)
    distances, indices = index.search(X32, k + 1)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.int64)


def clear_cache() -> None:
    _cache.clear()
