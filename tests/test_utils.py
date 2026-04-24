"""Tests for embedkit.utils."""

import numpy as np
import pytest
import torch

from embedkit.utils.validation import _to_numpy, _to_tensor, check_n_samples
from embedkit.utils.neighbors import knn, clear_cache


class TestValidation:
    def test_to_numpy_from_ndarray(self):
        X = np.ones((10, 5), dtype=np.float64)
        out = _to_numpy(X)
        assert out.dtype == np.float32
        assert out.shape == (10, 5)

    def test_to_numpy_from_tensor(self):
        t = torch.randn(8, 4)
        out = _to_numpy(t)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8, 4)

    def test_to_numpy_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            _to_numpy(np.ones(10))

    def test_to_tensor_from_ndarray(self):
        X = np.ones((5, 3), dtype=np.float32)
        t = _to_tensor(X)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (5, 3)

    def test_check_n_samples_ok(self):
        X = np.ones((10, 3))
        check_n_samples(X, min_n=5)  # no error

    def test_check_n_samples_fails(self):
        X = np.ones((3, 3))
        with pytest.raises(ValueError):
            check_n_samples(X, min_n=5)


class TestNeighbors:
    def setup_method(self):
        clear_cache()

    def test_knn_shape(self):
        X = np.random.randn(50, 8).astype(np.float32)
        dists, idx = knn(X, k=5)
        assert dists.shape == (50, 5)
        assert idx.shape == (50, 5)

    def test_knn_no_self(self):
        X = np.random.randn(50, 8).astype(np.float32)
        _, idx = knn(X, k=5)
        for i, row in enumerate(idx):
            assert i not in row

    def test_knn_cache(self):
        X = np.random.randn(30, 4).astype(np.float32)
        r1 = knn(X, k=3)
        r2 = knn(X, k=3)
        assert r1 is r2  # same object from cache

    def test_knn_distances_nonneg(self):
        X = np.random.randn(40, 6).astype(np.float32)
        dists, _ = knn(X, k=4)
        assert (dists >= 0).all()
