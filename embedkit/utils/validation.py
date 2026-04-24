from __future__ import annotations

import numpy as np


def _to_numpy(X) -> np.ndarray:
    """Coerce X (ndarray or Tensor) to float32 ndarray, enforce 2-D shape."""
    try:
        import torch
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
    except ImportError:
        pass
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {X.shape}")
    return X


def _to_tensor(X, device=None):
    """Coerce X (ndarray or Tensor) to float32 Tensor on device."""
    import torch
    if isinstance(X, torch.Tensor):
        return X.float().to(device)
    return torch.tensor(np.asarray(X, dtype=np.float32), device=device)


def check_n_samples(X, min_n: int = 2) -> None:
    if X.shape[0] < min_n:
        raise ValueError(f"Need at least {min_n} samples, got {X.shape[0]}")
