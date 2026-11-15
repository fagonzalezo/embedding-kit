"""Null models for calibrating embedding statistics (WP 4.1).

A statistic such as k-skewness or the coefficient of variation of distances depends on n, d, k
and the covariance spectrum, so a fixed threshold means something different on every dataset.
This module draws data sets from a null model fitted to ``X`` and reports where the observed
value falls in the null distribution, as a z-score.

Two null models:

``"gaussian"`` (:func:`spectrum_matched_gaussian`)
    A Gaussian with the mean of ``X`` and covariance ``alpha C + beta I``, where ``C`` is the
    sample covariance. ``beta`` keeps the trace, and ``alpha`` makes ``tr(Sigma^2)`` equal its
    unbiased (Wishart) estimate. A Gaussian with ``C`` itself is the wrong null: the sample
    spectrum is more spread than the population one by about ``1 + d / n``, so data drawn from
    a Gaussian look *more* isotropic than their own null (distance CV z-scores of -8 at
    n = 3,000, d = 512, and -19 at n = 300, d = 512, for iid data). With the correction, data
    drawn from a Gaussian with a flat or power-law spectrum give z-scores within +-3.
``"permuted"`` (:func:`column_permuted`)
    Each column permuted independently: the marginals are kept and every dependence between
    columns is removed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from embedkit.analysis._base import BaseResult
from embedkit.utils.validation import _to_numpy

NullModel = Literal["gaussian", "permuted"]
_MODELS = ("gaussian", "permuted")


@dataclass(frozen=True)
class CalibratedStatistic(BaseResult):
    """A statistic compared with its distribution under a null model.

    observed:
        The statistic on the calibration sample (``n_samples`` rows of the input).
    null_mean, null_std:
        Mean and standard deviation (``ddof=1``) over ``n_draws`` null data sets of the same
        shape, computed the same way.
    z_score:
        ``(observed - null_mean) / null_std``; 0 when the null has no spread and the observed
        value equals its mean, and +-inf when it has no spread and the value differs.
    """

    observed: float
    null_mean: float
    null_std: float
    z_score: float
    null_model: str
    n_draws: int
    n_samples: int


class _GaussianNull:
    """Sampler for :func:`spectrum_matched_gaussian`; factorises ``X`` once."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if n < 2:
            raise ValueError("spectrum_matched_gaussian needs at least 2 rows")
        self.d = d
        self.dtype = X.dtype
        self.mean = X.mean(axis=0)
        m = n - 1
        _, S, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        lam = S**2 / m  # non-zero eigenvalues of the sample covariance C
        t1, t2 = float(lam.sum()), float((lam**2).sum())
        self.alpha = shrinkage_weight(t1, t2, m, d)
        self.beta = (1.0 - self.alpha) * t1 / d
        self.scale = np.sqrt(self.alpha) * S / np.sqrt(m)
        self.Vt = Vt

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        Y = (rng.standard_normal((n, len(self.scale))) * self.scale) @ self.Vt
        if self.beta > 0:
            Y += np.sqrt(self.beta) * rng.standard_normal((n, self.d))
        return Y + self.mean


def shrinkage_weight(t1: float, t2: float, m: int, d: int) -> float:
    """Weight ``alpha`` of ``Sigma_hat = alpha C + (1 - alpha) tr(C) / d I`` matching two moments.

    ``t1 = tr(C)`` and ``t2 = tr(C^2)`` for a sample covariance ``C`` with ``m`` degrees of
    freedom. ``Sigma_hat`` has trace ``t1`` and ``tr(Sigma_hat^2) = f + alpha^2 (t2 - f)`` with
    ``f = t1^2 / d``. The returned ``alpha`` sets this equal to the unbiased Gaussian estimate
    ``T2 = m^2 / ((m - 1)(m + 2)) (t2 - t1^2 / m)`` of ``tr(Sigma^2)``:
    ``alpha = sqrt((T2 - f) / (t2 - f))``, with the ratio clipped to ``[0, 1]``.
    """
    if t2 <= 0:
        return 1.0  # constant data
    flat = t1**2 / d  # tr(.^2) of the isotropic matrix with the same trace
    excess = t2 - flat
    if excess <= 1e-12 * t2 or m < 2:
        return 1.0  # C is already isotropic, or there is nothing to correct with
    t2_unbiased = (m**2 / ((m - 1) * (m + 2))) * (t2 - t1**2 / m)
    return float(np.sqrt(np.clip((t2_unbiased - flat) / excess, 0.0, 1.0)))


def spectrum_matched_gaussian(
    X, n: int | None = None, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Draw ``n`` rows (default: as many as ``X``) from the Gaussian null of ``X``.

    The null has the mean of ``X`` and the covariance ``alpha C + beta I`` described in the
    module docstring. Returns an array of shape ``(n, d)`` with the dtype of ``X`` when it is
    floating point.
    """
    X = _to_numpy(X)
    rng = np.random.default_rng(rng)
    sampler = _GaussianNull(X)
    out = sampler.draw(X.shape[0] if n is None else n, rng)
    return out.astype(X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64)


def column_permuted(X, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Permute each column of ``X`` independently (keeps the marginals, removes dependence)."""
    X = _to_numpy(X)
    rng = np.random.default_rng(rng)
    return rng.permuted(X, axis=0)


def null_distribution(
    stat_fn: Callable[[np.ndarray], float | dict[str, float]],
    X,
    model: NullModel = "gaussian",
    n_draws: int = 20,
    random_state: int | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """Values of ``stat_fn`` on ``n_draws`` data sets drawn from the null model of ``X``.

    Each null data set has the shape and dtype of ``X``. ``stat_fn`` returns a float or a dict
    of floats; the result is an array of length ``n_draws``, or a dict of such arrays.
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")
    X = _to_numpy(X)
    dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
    rng = np.random.default_rng(random_state)
    sampler = _GaussianNull(X) if model == "gaussian" else None

    values = []
    for _ in range(n_draws):
        if sampler is not None:
            Y = sampler.draw(X.shape[0], rng).astype(dtype)
        else:
            Y = column_permuted(X, rng)
        values.append(stat_fn(Y))

    if isinstance(values[0], dict):
        return {key: np.array([v[key] for v in values], dtype=np.float64) for key in values[0]}
    return np.array(values, dtype=np.float64)


def calibrate(
    stat_fn: Callable[[np.ndarray], dict[str, float]],
    X,
    model: NullModel = "gaussian",
    n_draws: int = 20,
    random_state: int | None = None,
) -> dict[str, CalibratedStatistic]:
    """Compute ``stat_fn(X)`` and compare each of its values with the null distribution."""
    X = _to_numpy(X)
    observed = stat_fn(X)
    null = null_distribution(stat_fn, X, model=model, n_draws=n_draws, random_state=random_state)
    return {
        key: calibrated_statistic(observed[key], null[key], model, X.shape[0])
        for key in observed
    }


def calibrated_statistic(
    observed: float, null_values: np.ndarray, model: str, n_samples: int
) -> CalibratedStatistic:
    null_values = np.asarray(null_values, dtype=np.float64)
    mean = float(null_values.mean())
    std = float(null_values.std(ddof=1))
    diff = float(observed) - mean
    if std > 0:
        z = diff / std
    elif diff == 0:
        z = 0.0
    else:
        z = float(np.sign(diff) * np.inf)
    return CalibratedStatistic(
        observed=float(observed),
        null_mean=mean,
        null_std=std,
        z_score=float(z),
        null_model=model,
        n_draws=len(null_values),
        n_samples=int(n_samples),
    )
