"""Intrinsic dimension estimation via skdim (WP 4.2: per-estimator intervals and scale curve)."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass

import numpy as np

from embedkit.analysis._base import BaseAnalyzer, BaseResult

_SUPPORTED = {"TwoNN", "MLE", "lPCA", "DANCo", "CorrInt", "MOM", "FisherS", "ESS"}
_LOCAL_METHODS = {"TwoNN", "MLE", "lPCA"}
# About 50-110 s per fit at n = 5,000 (measured 2026-11-14), so they get no bootstrap interval.
_SLOW_METHODS = {"ESS", "DANCo"}
_SCALE_FACTORS = (1, 2, 4, 8)
_MIN_SCALE_ROWS = 20


@dataclass(frozen=True)
class IntrinsicDimensionResult(BaseResult):
    """Per-estimator intrinsic dimension, with sampling intervals and a scale curve.

    estimates:
        Point estimate of each method; NaN when it failed. The estimators measure different
        things and are not combined.
    standard_errors, intervals:
        Bootstrap standard error of each estimate (``n_bootstrap`` resamples of the rows) and the
        normal interval ``estimate +- 1.96 se``. They describe sampling variability only: the
        k-NN estimators (TwoNN, MLE, MOM) read low at high intrinsic dimension (on a 20-sphere
        with n = 5,000 they give 15-17), and no interval covers that bias. NaN for methods in
        ``_SLOW_METHODS``, for failed methods, and when ``n_bootstrap = 0``.
    between_method_spread:
        Standard deviation of the valid estimates across methods (formerly ``uncertainty``).
        It is not a sampling uncertainty.
    scale_curve:
        TwoNN on nested subsamples of ``n``, ``n/2``, ``n/4`` and ``n/8`` rows (levels with fewer
        than 20 rows are left out), as ``{n_rows: estimate}``. Fewer rows probe larger scales.
        Empty when disabled.
    scale_max_deviation:
        Largest relative deviation of a level from the full-sample value,
        ``max |curve[m] / curve[n] - 1|``; NaN without a curve. A flat curve (a plateau) gives a
        small value. On clean data the curve still drifts down at high ID (a 20-sphere reads
        17.1 at n = 5,000 and 15.8 at n = 625, a deviation of 0.08). It replaces the boolean
        ``scale_plateau`` (within 10%), which flipped between random subsets of one dataset in
        E3 (WP 4.5) and separated noisy from noise-free manifolds only weakly (plateau in 44%
        and 74% of cells); no threshold is applied to it.
    reference_method, reference_estimate:
        The single estimate the report uses for ``suggested_target_dim`` and the ID / D
        recommendation: TwoNN when it was requested and succeeded, otherwise the first method of
        ``methods`` that succeeded; None and NaN when every method failed.
    """

    estimates: dict[str, float]
    standard_errors: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    local_estimates: dict[str, np.ndarray]
    between_method_spread: float
    scale_curve: dict[int, float]
    scale_max_deviation: float
    reference_method: str | None
    reference_estimate: float
    n_samples: int = 0
    n_duplicates_removed: int = 0


class IntrinsicDimensionEstimator(BaseAnalyzer):
    """Run several skdim estimators on (a subsample of) ``X`` and report each separately.

    n_bootstrap:
        Bootstrap resamples per method for the standard errors (0 to skip). A resample draws
        ``n`` rows with replacement and drops the repeated rows, as ``fit`` does with exact
        duplicates. Measured against the spread over 40 independent draws of spheres and cubes
        (n = 5,000, ID 10 and 20), these standard errors were within about 30% of the true
        sampling error for TwoNN, MLE and MOM.
    scale_curve:
        Compute the TwoNN scale curve (see :class:`IntrinsicDimensionResult`).
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        n_max: int = 5_000,
        n_bootstrap: int = 20,
        scale_curve: bool = True,
        random_state: int | None = 42,
    ):
        self.methods = methods or ["TwoNN", "MLE", "lPCA"]
        self.n_max = n_max
        self.n_bootstrap = n_bootstrap
        self.scale_curve = scale_curve
        self.random_state = random_state

    def fit(self, X, y=None) -> IntrinsicDimensionResult:
        """Estimate the intrinsic dimension with each method.

        Exact duplicate rows are removed first (``n_duplicates_removed`` reports how many):
        TwoNN divides by the nearest-neighbour distance, which is zero for a duplicated point.
        The methods run one after the other. Running them in a thread pool made the TwoNN
        estimate vary from run to run for identical input and seed. An estimate that is not
        finite and positive counts as a failure: it is recorded as NaN with a warning.
        """
        for method in self.methods:
            if method not in _SUPPORTED:
                raise ValueError(f"Unknown ID method: {method}. Choose from {_SUPPORTED}")

        X = self._prepare(X, min_n=5)
        X, n_duplicates = _drop_exact_duplicates(X)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        if n > self.n_max:
            X = X[np.sort(rng.choice(n, self.n_max, replace=False))]
            n = self.n_max

        estimates: dict[str, float] = {}
        local_estimates: dict[str, np.ndarray] = {}
        for method in self.methods:
            dim, est = _fit_one(method, X, self.random_state, warn=True)
            estimates[method] = dim
            if est is not None and method in _LOCAL_METHODS and hasattr(est, "dimension_pw_"):
                local_estimates[method] = np.asarray(est.dimension_pw_, dtype=np.float32)

        standard_errors, intervals = self._bootstrap(X, estimates, rng)
        valid = [v for v in estimates.values() if not np.isnan(v)]
        spread = float(np.std(valid)) if len(valid) > 1 else (0.0 if valid else float("nan"))

        curve, deviation = {}, float("nan")
        if self.scale_curve:
            curve, deviation = self._scale_curve(X, estimates.get("TwoNN"), rng)

        reference = next(
            (m for m in (["TwoNN"] if "TwoNN" in estimates else []) + list(self.methods)
             if not np.isnan(estimates[m])),
            None,
        )
        return IntrinsicDimensionResult(
            estimates=estimates,
            standard_errors=standard_errors,
            intervals=intervals,
            local_estimates=local_estimates,
            between_method_spread=spread,
            scale_curve=curve,
            scale_max_deviation=deviation,
            reference_method=reference,
            reference_estimate=estimates[reference] if reference else float("nan"),
            n_samples=n,
            n_duplicates_removed=n_duplicates,
        )

    def _bootstrap(self, X, estimates, rng):
        nan = float("nan")
        se = {m: nan for m in estimates}
        resampled = [
            m for m, v in estimates.items()
            if self.n_bootstrap >= 2 and m not in _SLOW_METHODS and not np.isnan(v)
        ]
        values: dict[str, list[float]] = {m: [] for m in resampled}
        n = X.shape[0]
        for _ in range(self.n_bootstrap if resampled else 0):
            Xb, _ = _drop_exact_duplicates(X[rng.integers(0, n, n)])
            for m in resampled:
                dim, _ = _fit_one(m, Xb, self.random_state, warn=False)
                if not np.isnan(dim):
                    values[m].append(dim)
        for m in resampled:
            if len(values[m]) >= 2:
                se[m] = float(np.std(values[m], ddof=1))
        intervals = {
            m: (estimates[m] - 1.96 * s, estimates[m] + 1.96 * s) if not np.isnan(s) else (nan, nan)
            for m, s in se.items()
        }
        return se, intervals

    def _scale_curve(self, X, full_twonn, rng):
        n = X.shape[0]
        perm = rng.permutation(n)
        curve: dict[int, float] = {}
        for f in _SCALE_FACTORS:
            rows = n // f
            if rows < _MIN_SCALE_ROWS:
                break
            if f == 1 and full_twonn is not None:
                curve[rows] = full_twonn  # same rows; TwoNN does not depend on their order
            else:
                curve[rows], _ = _fit_one("TwoNN", X[np.sort(perm[:rows])], self.random_state,
                                          warn=False)
        finite = {k: v for k, v in curve.items() if not np.isnan(v)}
        if n not in finite or len(finite) < 2:
            return curve, float("nan")
        ref = finite[n]
        return curve, float(max(abs(v / ref - 1.0) for v in finite.values()))


def _fit_one(method: str, X: np.ndarray, random_state, warn: bool):
    """Fit one estimator; return ``(dimension, estimator)`` or ``(nan, None)`` on failure."""
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            if not warn:
                warnings.simplefilter("ignore")
            est = _build_estimator(method, random_state)
            est.fit(X)
        dim = float(est.dimension_)
        if not np.isfinite(dim) or dim <= 0:
            raise ValueError(f"non-finite or non-positive estimate ({dim})")
    except Exception as e:  # noqa: BLE001 - skdim raises many kinds of errors
        if warn:
            warnings.warn(f"ID method {method} failed: {e}", stacklevel=3)
        return float("nan"), None
    return dim, est


def _drop_exact_duplicates(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove exact duplicate rows, keeping the first occurrence and the original order."""
    n = X.shape[0]
    rows = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, first = np.unique(rows.ravel(), return_index=True)
    if len(first) == n:
        return X, 0
    return X[np.sort(first)], n - len(first)


_DANCO_CACHE: dict = {}


def _danco_with_clipped_angles(base):
    """skdim's DANCo with the cosine of each neighbour angle clipped to [-1, 1].

    skdim 0.3.4 clips it at 1 only. When two neighbour vectors point in exactly opposite
    directions, as on a low-dimensional cube mapped linearly into many dimensions, rounding
    gives a cosine just below -1, ``arccos`` returns NaN, and so does the estimate: DANCo was NaN
    on 15 of 20 E3 ``cube2`` cells (D = 256) and on one Swiss-roll cell. With both sides
    clipped it reads about 2.1 to 2.2 there (true ID 2), and it is unchanged wherever skdim's
    estimate was finite (the pairs are taken in the order of skdim's ``indnComb``).
    """
    if not hasattr(base, "_loc_angles"):  # another skdim version: leave it alone
        return base
    if base not in _DANCO_CACHE:

        class DANCo(base):
            def _loc_angles(self, pt, nbs):
                vec = nbs - pt
                length = np.sqrt(np.sum(vec**2, axis=1))
                i, j = np.triu_indices(len(nbs), 1)
                cos = np.sum(vec[i] * vec[j], axis=1) / (length[i] * length[j])
                return np.arccos(np.clip(cos, -1.0, 1.0))

        _DANCO_CACHE[base] = DANCo
    return _DANCO_CACHE[base]


def _build_estimator(method: str, random_state):
    # skdim.id._FisherS calls warnings.filterwarnings("ignore") at import time, which
    # silences every warning in the user's process. catch_warnings() restores the
    # global filter list when the block exits.
    with warnings.catch_warnings():
        import skdim.id as skid

    cls_map = {
        "TwoNN": skid.TwoNN,
        "MLE": skid.MLE,
        "lPCA": skid.lPCA,
        "DANCo": _danco_with_clipped_angles(skid.DANCo),
        "CorrInt": skid.CorrInt,
        "MOM": skid.MOM,
        "FisherS": skid.FisherS,
        "ESS": skid.ESS,
    }
    cls = cls_map[method]
    # DANCo and ESS accept random_state; the other estimators are deterministic.
    if "random_state" in inspect.signature(cls.__init__).parameters:
        return cls(random_state=random_state)
    return cls()
