from __future__ import annotations

import contextlib
import contextvars
import warnings

import numpy as np

# Set while an entry point that has already validated its input calls other entry points
# (``EmbedKit.fit`` -> ``EmbedKitAnalyzer.fit`` -> the analyzers, ``Trainer.fit``): the inner
# calls then only convert and check the shape, so every warning is issued once per user call.
_TRUSTED = contextvars.ContextVar("embedkit_trusted_input", default=False)

_HASH_BLOCK_ENTRIES = 1 << 21  # entries hashed at once (16 MB as uint64)
_HASH_SEED = 20261117


@contextlib.contextmanager
def trusted_input():
    """Inside this block :func:`validate_embeddings` skips its value scans (NaN or infinite
    values, constant columns, duplicate rows). Wrap the internal calls of an entry point that
    has validated its input."""
    token = _TRUSTED.set(True)
    try:
        yield
    finally:
        _TRUSTED.reset(token)


def _as_array(X, dtype) -> np.ndarray:
    try:
        import torch

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu()
            if X.dtype == torch.bfloat16:  # NumPy has no bfloat16
                X = X.float()
            X = X.numpy()
    except ImportError:  # pragma: no cover
        pass
    with np.errstate(over="ignore"):  # float64 beyond the float32 range: reported as infinite
        return np.asarray(X, dtype=dtype)


def _row_blocks(n: int, d: int):
    step = max(1, _HASH_BLOCK_ENTRIES // max(d, 1))
    for start in range(0, n, step):
        yield start, min(n, start + step)


def _nonfinite_counts(X: np.ndarray) -> tuple[int, int, int]:
    """``(entries, rows, NaN entries)`` that are not finite; scans in blocks, no ``(n, d)`` temp."""
    entries = rows = nans = 0
    for a, b in _row_blocks(*X.shape):
        bad = ~np.isfinite(X[a:b])
        if bad.any():
            entries += int(bad.sum())
            rows += int(bad.any(axis=1).sum())
            nans += int(np.isnan(X[a:b]).sum())
    return entries, rows, nans


def count_constant_columns(X: np.ndarray) -> int:
    """Number of columns with zero variance (all rows equal), for a finite ``(n, d)`` array."""
    return int((X.min(axis=0) == X.max(axis=0)).sum())


def count_duplicate_rows(X: np.ndarray) -> int:
    """Number of rows that exactly repeat an earlier row (``-0.0`` equals ``0.0``).

    Each row is hashed to two 64-bit integers (wrap-around dot product of its bit pattern with
    random odd coefficients, in blocks), the rows are ordered by the hashes, and neighbours
    with equal hashes are compared entry by entry. That is a sort of ``n`` pairs of integers
    instead of ``n`` rows of ``d`` entries, and the integer arithmetic is deterministic, so
    equal rows always get equal hashes. Rows that are not equal are miscounted only if both
    hashes collide and the rows interleave, which does not happen in practice. Requires finite
    values.
    """
    n, d = X.shape
    if n < 2:
        return 0
    rng = np.random.default_rng(_HASH_SEED)
    coef = rng.integers(1, 2**63, size=(d, 2), dtype=np.uint64) | np.uint64(1)
    bits = np.dtype(f"u{X.dtype.itemsize}")
    hashes = np.empty((n, 2), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for a, b in _row_blocks(n, d):
            # + 0 turns -0.0 into 0.0, so that rows equal as numbers get equal bit patterns
            U = (X[a:b] + X.dtype.type(0)).view(bits).astype(np.uint64)
            # fold the high word into the low one: a float64 such as 2.0 has 50 zero low bits,
            # and the product with a 64-bit coefficient would lose its high bits
            U ^= U >> np.uint64(32)
            hashes[a:b] = U @ coef
    order = np.lexsort((hashes[:, 1], hashes[:, 0]))
    h = hashes[order]
    same = np.flatnonzero((h[1:] == h[:-1]).all(axis=1))
    count = 0
    for a, b in _row_blocks(len(same), d):
        i = same[a:b]
        count += int((X[order[i]] == X[order[i + 1]]).all(axis=1).sum())
    return count


def validate_embeddings(
    X,
    *,
    name: str = "X",
    min_samples: int = 2,
    n_features: int | None = None,
    warn: bool = True,
    dtype=np.float32,
    stacklevel: int = 3,
) -> np.ndarray:
    """The shared check of an embedding matrix at the public entry points.

    ``X`` is an ``ndarray``, a ``torch.Tensor`` (any device and float dtype), a list or anything
    ``np.asarray`` reads. It is returned as a 2-D ``ndarray`` of ``dtype`` (a copy only when the
    dtype changes).

    Raises ``ValueError`` when

    - ``X`` is not 2-D, has no columns, or has fewer than ``min_samples`` rows;
    - ``n_features`` is given and ``X`` has a different number of columns (``transform`` checks
      against the width seen in ``fit``);
    - ``X`` has NaN or infinite values; the message gives the number of entries and rows.

    With ``warn=True`` it issues one ``UserWarning`` for constant (zero-variance) columns and one
    for exact duplicate rows, each with its counts; neither stops the computation. Pass
    ``warn=False`` for ``transform``-like calls: they check shape and values but do not scan
    for those two conditions. Inside :func:`trusted_input` only the shape is checked.

    **float64.** The default ``dtype=np.float32`` is what the torch refiner and the analysis
    code compute in (the library has always converted at its entry points, silently): float64
    input is accepted and rounded to float32 (relative error about 6e-8; a value beyond the
    float32 range becomes infinite and raises). The baselines in :mod:`embedkit.baselines` and
    the hubness reductions compute in float64 and pass ``dtype=np.float64``, so they keep the
    precision of float64 input. Integer input is converted the same way.
    """
    X = _as_array(X, dtype)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {X.shape}")
    if X.shape[0] < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {X.shape[0]}")
    if X.shape[1] < 1:
        raise ValueError(f"{name} has no columns")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"{name} has {X.shape[1]} features, but {n_features} were expected "
                         f"(the number seen in fit)")
    if _TRUSTED.get():
        return X
    entries, rows, nans = _nonfinite_counts(X)
    if entries:
        raise ValueError(
            f"{name} contains NaN or infinite values: {entries} non-finite entries "
            f"({nans} NaN, {entries - nans} infinite) in {rows} of {X.shape[0]} rows; "
            f"all entries must be finite")
    if warn and X.shape[0] >= 2:
        n, d = X.shape
        constant = count_constant_columns(X)
        if constant:
            warnings.warn(
                f"{name} has {constant} constant column(s) of {d} (zero variance); they carry no "
                f"information and distort variance-based statistics", UserWarning,
                stacklevel=stacklevel)
        duplicates = count_duplicate_rows(X)
        if duplicates:
            warnings.warn(
                f"{name} has {duplicates} duplicate row(s) of {n} (exact copies of an earlier "
                f"row); nearest-neighbour statistics and the intrinsic dimension are biased by "
                f"them", UserWarning, stacklevel=stacklevel)
    return X


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
