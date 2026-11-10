"""Regression tests for embedkit.utils.neighbors (WP 0.1 and WP 1.1: D1, D2, D12, D15, D16, D17)."""

from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import pytest

from embedkit.utils.neighbors import clear_cache, knn

# FAISS is optional and crashes in some environments (D12), so it is only imported in the
# FAISS CI job, which sets EMBEDKIT_KNN_BACKEND=faiss.
HAS_FAISS = os.environ.get("EMBEDKIT_KNN_BACKEND", "") == "faiss"


def brute_force(X: np.ndarray, k: int, metric: str):
    """Naive O(n^2) reference: full distance matrix, self excluded by index."""
    X = X.astype(np.float64)
    if metric == "euclidean":
        D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))
    else:
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        D = 1.0 - Xn @ Xn.T
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(D, order, axis=1), order, D


def _data(with_duplicates: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((120, 9)).astype(np.float32)
    if with_duplicates:
        X[1] = X[0]
        X[2] = X[0]
        X[50] = X[49]
    return X


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("with_duplicates", [False, True])
def test_matches_brute_force(metric, with_duplicates):
    X = _data(with_duplicates)
    k = 7
    d_ref, _, D_full = brute_force(X, k, metric)
    dist, idx = knn(X, k, metric=metric)
    assert dist.shape == idx.shape == (X.shape[0], k)
    # Distances equal the reference (true distances, not squared, not similarities).
    np.testing.assert_allclose(dist, d_ref, atol=1e-4)
    # Every returned index is at the distance reported for it.
    np.testing.assert_allclose(np.take_along_axis(D_full, idx, axis=1), dist, atol=1e-4)
    # Sorted ascending.
    assert (np.diff(dist, axis=1) >= -1e-6).all()


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_no_row_contains_itself_with_duplicates(metric):
    X = _data(with_duplicates=True)
    # Many exact copies of one row: the query may be pushed out of the k+1 search.
    X[3:15] = X[0]
    _, idx = knn(X, 5, metric=metric)
    assert not (idx == np.arange(X.shape[0])[:, None]).any()


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_input_not_modified(metric):
    X = _data(with_duplicates=False) * 3.0
    X = np.ascontiguousarray(X, dtype=np.float32)
    before = X.tobytes()
    knn(X, 5, metric=metric)
    assert X.tobytes() == before


def test_fresh_arrays_of_same_shape_get_their_own_neighbours():
    """D1: the old cache keyed on memory address returned another array's neighbours."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.standard_normal((300, 16)).astype(np.float32)
        knn(A, 5)
        del A
        gc.collect()
        B = rng.standard_normal((300, 16)).astype(np.float32)
        _, idx = knn(B, 5)
        _, ref, _ = brute_force(B, 5, "euclidean")
        np.testing.assert_array_equal(idx, ref)


def test_in_place_edit_is_seen():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 8)).astype(np.float32)
    _, i1 = knn(A, 5)
    A[:] = rng.standard_normal((200, 8))
    _, i2 = knn(A, 5)
    _, ref, _ = brute_force(A, 5, "euclidean")
    np.testing.assert_array_equal(i2, ref)
    assert not np.array_equal(i1, i2)


def test_neighbor_consistency_rounds_are_distinct():
    """D2: with the cache, later perturbation rounds reused earlier rounds' neighbours."""
    from embedkit.analysis.geometry import NeighborConsistency

    X = np.random.default_rng(0).standard_normal((1500, 64)).astype(np.float32)
    r = NeighborConsistency(k=10, noise_std=0.3, n_perturbations=5).fit(X)
    assert len(r.per_perturbation) == 5
    assert len(np.unique(r.per_perturbation)) == 5


def test_invalid_k_raises():
    X = _data(False)
    with pytest.raises(ValueError):
        knn(X, X.shape[0])
    with pytest.raises(ValueError):
        knn(X, 0)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        knn(_data(False), 3, backend="annoy")


def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("EMBEDKIT_KNN_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="backend"):
        knn(_data(False), 3)
    monkeypatch.setenv("EMBEDKIT_KNN_BACKEND", "sklearn")
    knn(_data(False), 3)


def test_clear_cache_is_deprecated_noop():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clear_cache()
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_hubness_accepts_precomputed_graph():
    from embedkit.analysis.hubness import HubnessAnalyzer

    X = np.random.default_rng(0).standard_normal((300, 16)).astype(np.float32)
    _, graph = knn(X, 10)
    a = HubnessAnalyzer(k=10).fit(X)
    b = HubnessAnalyzer(k=10).fit(X, neighbor_index=graph)
    assert a.k_skewness == pytest.approx(b.k_skewness)
    with pytest.raises(ValueError, match="neighbor_index"):
        HubnessAnalyzer(k=10).fit(X, neighbor_index=graph[:, :5])


@pytest.mark.skipif(not HAS_FAISS, reason="FAISS job only (EMBEDKIT_KNN_BACKEND=faiss)")
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_faiss_and_sklearn_agree(metric):
    X = _data(with_duplicates=False)
    d1, i1 = knn(X, 7, metric=metric, backend="sklearn")
    d2, i2 = knn(X, 7, metric=metric, backend="faiss")
    np.testing.assert_allclose(d1, d2, atol=1e-4)
    np.testing.assert_array_equal(i1, i2)
