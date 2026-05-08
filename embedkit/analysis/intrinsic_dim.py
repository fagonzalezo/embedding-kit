"""Intrinsic dimension estimation via skdim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        X = self._prepare(X, min_n=5)
        n = X.shape[0]
        if n > self.n_max:
            rng = np.random.default_rng(self.random_state)
            X = X[rng.choice(n, self.n_max, replace=False)]
        estimates: dict[str, float] = {}
        local_estimates: dict[str, np.ndarray] = {}

        for method in self.methods:
            if method not in _SUPPORTED:
                raise ValueError(f"Unknown ID method: {method}. Choose from {_SUPPORTED}")

        def _fit_one(method: str):
            try:
                est = _build_estimator(method, self.random_state)
                est.fit(X)
                dim = float(est.dimension_)
                local = (
                    np.asarray(est.dimension_pw_, dtype=np.float32)
                    if method in _LOCAL_METHODS and hasattr(est, "dimension_pw_")
                    else None
                )
                return method, dim, local
            except Exception as e:
                import warnings
                warnings.warn(f"ID method {method} failed: {e}")
                return method, float("nan"), None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(self.methods), 4)) as pool:
            for method, dim, local in pool.map(_fit_one, self.methods):
                estimates[method] = dim
                if local is not None:
                    local_estimates[method] = local

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
        )


def _build_estimator(method: str, random_state):
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
    try:
        return cls(random_state=random_state)
    except TypeError:
        return cls()
