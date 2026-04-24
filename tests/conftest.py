"""Shared fixtures for EmbedKit tests."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def gaussian_blobs():
    """500 samples, 32-D, 5 isotropic Gaussian blobs."""
    X, y = make_blobs(n_samples=500, n_features=32, centers=5, random_state=42)
    return X.astype(np.float32), y


@pytest.fixture(scope="session")
def small_X():
    """100 samples in 16 dimensions — fast fixture for smoke tests."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((100, 16)).astype(np.float32)


@pytest.fixture(scope="session")
def small_Xy(small_X):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=small_X.shape[0])
    return small_X, y


@pytest.fixture(scope="session")
def hubbed_X():
    """Anisotropic high-dim embedding known to produce hubness.

    300 samples in 128-D — most variance in the first 5 dims.
    """
    rng = np.random.default_rng(1)
    scale = np.ones(128)
    scale[:5] = 10.0
    X = (rng.standard_normal((300, 128)) * scale).astype(np.float32)
    return X


@pytest.fixture(scope="session")
def swiss_roll_embedded():
    """10-D Swiss roll embedded into 50-D (known intrinsic dim ≈ 2)."""
    from sklearn.datasets import make_swiss_roll
    X_3d, _ = make_swiss_roll(n_samples=300, noise=0.1, random_state=42)
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((3, 50)).astype(np.float32)
    return (X_3d.astype(np.float32) @ proj)
