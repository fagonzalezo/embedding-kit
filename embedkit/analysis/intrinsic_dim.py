"""Intrinsic dimension estimation via skdim."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass

import numpy as np

from embedkit.analysis._base import BaseAnalyzer, BaseResult

_SUPPORTED = {"TwoNN", "MLE", "lPCA", "DANCo", "CorrInt", "MOM", "FisherS"}
_LOCAL_METHODS = {"TwoNN", "MLE", "lPCA"}


@dataclass(frozen=True)
class IntrinsicDimensionResult(BaseResult):
    estimates: dict[str, float]
    consensus: float
    local_estimates: dict[str, np.ndarray]
    uncertainty: float
    n_duplicates_removed: int = 0


class IntrinsicDimensionEstimator(BaseAnalyzer):
    def __init__(
        self,
        methods: list[str] | None = None,
        aggregate: str = "median",
        n_max: int = 5_000,
        random_state: int | None = 42,
    ):
        self.methods = methods or ["TwoNN", "MLE", "lPCA"]
        self.aggregate = aggregate
        self.n_max = n_max
        self.random_state = random_state

    def fit(self, X, y=None) -> IntrinsicDimensionResult:
        """Estimate the intrinsic dimension with each method and aggregate the valid estimates.

        Exact duplicate rows are removed first (``n_duplicates_removed`` reports how many):
        TwoNN divides by the nearest-neighbour distance, which is zero for a duplicated point.
        The methods run one after the other. Running them in a thread pool made the TwoNN
        estimate vary from run to run for identical input and seed. An estimate that is not
        finite and positive counts as a failure: it is recorded as NaN and excluded from the
        consensus.
        """
        X = self._prepare(X, min_n=5)
        X, n_duplicates = _drop_exact_duplicates(X)
        n = X.shape[0]
        if n > self.n_max:
            rng = np.random.default_rng(self.random_state)
            X = X[np.sort(rng.choice(n, self.n_max, replace=False))]
        estimates: dict[str, float] = {}
        local_estimates: dict[str, np.ndarray] = {}

        for method in self.methods:
            if method not in _SUPPORTED:
                raise ValueError(f"Unknown ID method: {method}. Choose from {_SUPPORTED}")

        for method in self.methods:
            try:
                with np.errstate(all="ignore"):
                    est = _build_estimator(method, self.random_state)
                    est.fit(X)
                dim = float(est.dimension_)
                if not np.isfinite(dim) or dim <= 0:
                    raise ValueError(f"non-finite or non-positive estimate ({dim})")
            except Exception as e:
                warnings.warn(f"ID method {method} failed: {e}", stacklevel=2)
                estimates[method] = float("nan")
                continue
            estimates[method] = dim
            if method in _LOCAL_METHODS and hasattr(est, "dimension_pw_"):
                local_estimates[method] = np.asarray(est.dimension_pw_, dtype=np.float32)

        valid = [v for v in estimates.values() if not np.isnan(v)]
        if not valid:
            consensus = float("nan")
            uncertainty = float("nan")
        else:
            agg_fn = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max}[
                self.aggregate
            ]
            consensus = float(agg_fn(valid))
            uncertainty = float(np.std(valid)) if len(valid) > 1 else 0.0

        return IntrinsicDimensionResult(
            estimates=estimates,
            consensus=consensus,
            local_estimates=local_estimates,
            uncertainty=uncertainty,
            n_duplicates_removed=n_duplicates,
        )


def _drop_exact_duplicates(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove exact duplicate rows, keeping the first occurrence and the original order."""
    n = X.shape[0]
    rows = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, first = np.unique(rows.ravel(), return_index=True)
    if len(first) == n:
        return X, 0
    return X[np.sort(first)], n - len(first)


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
        "DANCo": skid.DANCo,
        "CorrInt": skid.CorrInt,
        "MOM": skid.MOM,
        "FisherS": skid.FisherS,
    }
    cls = cls_map[method]
    # Only DANCo accepts random_state; the other estimators are deterministic.
    if "random_state" in inspect.signature(cls.__init__).parameters:
        return cls(random_state=random_state)
    return cls()
