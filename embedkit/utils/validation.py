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


def encode_labels(y, n_samples: int | None = None):
    """Convert user labels to a tensor and say whether they are classes or continuous.

    Accepted inputs: Python lists, NumPy arrays and tensors of strings, booleans, integers or
    floats, of shape ``(n,)``; and 2-D float arrays of shape ``(n, m)`` (vector labels).

    Rules:

    - strings, objects and booleans are classes; they are encoded as ``int64`` codes
      ``0 .. C-1`` in sorted order of the distinct values;
    - integers are classes and keep their values (``int64``), so ordered integer levels still
      carry their spacing for losses that use label distances;
    - 1-D floats are continuous (``float32``), even when every value is a whole number;
    - 2-D inputs are continuous vector labels (``float32``), shape ``(n, m)``;
    - a 2-D input with one column is treated as the 1-D array of that column.

    Returns
    -------
    labels : torch.Tensor
        ``int64`` of shape ``(n,)`` for classes, ``float32`` of shape ``(n,)`` or ``(n, m)``
        for continuous labels. On the CPU.
    kind : {"class", "continuous"}
    """
    import torch

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim not in (1, 2):
        raise ValueError(f"Labels must be 1-D or 2-D, got shape {arr.shape}")
    if n_samples is not None and arr.shape[0] != n_samples:
        raise ValueError(f"Got {arr.shape[0]} labels for {n_samples} samples")

    kind_char = arr.dtype.kind
    if arr.ndim == 2:
        if kind_char not in "biuf":
            raise ValueError(f"2-D labels must be numeric, got dtype {arr.dtype}")
        out = arr.astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Labels contain NaN or infinite values")
        return torch.from_numpy(np.ascontiguousarray(out)), "continuous"

    if kind_char in "iu":
        return torch.from_numpy(arr.astype(np.int64)), "class"
    if kind_char == "f":
        out = arr.astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Labels contain NaN or infinite values")
        return torch.from_numpy(np.ascontiguousarray(out)), "continuous"
    if kind_char in "bUSO":
        if kind_char == "O" and any(v is None for v in arr):
            raise ValueError("Labels contain None")
        _, codes = np.unique(arr.astype(str) if kind_char == "O" else arr, return_inverse=True)
        return torch.from_numpy(codes.astype(np.int64).reshape(-1)), "class"
    raise ValueError(f"Unsupported label dtype {arr.dtype}")
