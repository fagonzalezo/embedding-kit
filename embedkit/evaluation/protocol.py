"""Evaluation protocol: fit a transform on the training split, score it on the test split.

:func:`evaluate_transform` turns any object with ``fit`` and ``transform`` (an
:class:`~embedkit.EmbedKit`, a class from :mod:`embedkit.baselines`, a scikit-learn transformer)
into a tidy table of held-out metrics, one row per seed and metric. :func:`compare` summarises
several such tables against a reference method, with paired bootstrap intervals over test points.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from embedkit.utils.validation import _to_numpy, encode_labels

# ---------------------------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------------------------
# Each metric receives a context with the raw and transformed splits and returns either a float
# or (float, per_point). per_point is an array over test points whose mean is the metric; it is
# what compare() bootstraps. Metrics that need labels are skipped when no labels are given.


@dataclass
class _Context:
    X_train: np.ndarray
    X_test: np.ndarray
    Z_train: np.ndarray
    Z_test: np.ndarray
    y_train: np.ndarray | None
    y_test: np.ndarray | None
    label_kind: str | None  # "class", "continuous" or None
    test_pairs: tuple | None = None  # (idx_a, idx_b, gold scores) into the test rows
    hubness_reduction: Any = None  # neighbour search for the kNN-based metrics


@dataclass(frozen=True)
class MetricSpec:
    fn: Callable[[_Context], Any]
    needs_labels: bool = False
    needs_class_labels: bool = False
    needs_continuous_labels: bool = False


def _m():
    from embedkit.evaluation import metrics

    return metrics


def _knn(metric: str, k: int = 5):
    def fn(c: _Context):
        return _m().knn_accuracy(
            c.Z_train, c.y_train, c.Z_test, c.y_test, k=k, metric=metric, return_per_point=True,
            hubness_reduction=c.hubness_reduction,
        )

    return MetricSpec(fn, needs_labels=True, needs_class_labels=True)


def _probe(c: _Context):
    return _m().linear_probe(
        c.Z_train, c.y_train, c.Z_test, c.y_test, task="auto", return_per_point=True
    )


def _probe_c1(c: _Context):
    # The review pilot's probe: logistic regression with C = 1, no tuning.
    return _m().linear_probe(
        c.Z_train, c.y_train, c.Z_test, c.y_test, task="classification", C=1.0,
        max_iter=3000, return_per_point=True,
    )


def _recall(k: int):
    def fn(c: _Context):
        return _m().recall_at_k(
            c.Z_test, c.Z_train, (c.y_test, c.y_train), k=k, metric="cosine",
            return_per_point=True, hubness_reduction=c.hubness_reduction,
        )

    return MetricSpec(fn, needs_labels=True, needs_class_labels=True)


def _sts_spearman(c: _Context):
    """Spearman correlation between cosine similarity of test pairs and their gold scores."""
    from scipy.stats import spearmanr

    a, b, gold = c.test_pairs
    Z = c.Z_test.astype(np.float64)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    cos = np.sum(Z[np.asarray(a)] * Z[np.asarray(b)], axis=1)
    return float(spearmanr(cos, np.asarray(gold)).statistic)


METRICS: dict[str, MetricSpec] = {
    # Sentence similarity, from test pairs (test_pairs=...).
    "sts_spearman": MetricSpec(_sts_spearman),
    # Downstream, with labels. Neighbours always come from the training split.
    "knn_accuracy": _knn("cosine"),
    "knn_accuracy_euclidean": _knn("euclidean"),
    "linear_probe": MetricSpec(_probe, needs_labels=True),
    "linear_probe_c1": MetricSpec(_probe_c1, needs_labels=True, needs_class_labels=True),
    "recall_at_10": _recall(10),
    # Regression (continuous targets): kNN regression, k = 10, cosine.
    "knn_r2": MetricSpec(lambda c: _m().knn_regression_r2(
        c.Z_train, c.y_train, c.Z_test, c.y_test, k=10, return_per_point=True),
        needs_labels=True, needs_continuous_labels=True),
    "nc1": MetricSpec(lambda c: _m().nc1(c.Z_test, c.y_test), True, True),
    "silhouette": MetricSpec(lambda c: _m().silhouette(c.Z_test, c.y_test), True, True),
    # Label-free, on the test split: structure preservation relative to the input ...
    "trustworthiness": MetricSpec(lambda c: _m().trustworthiness(c.X_test, c.Z_test, k=10)),
    "continuity": MetricSpec(lambda c: _m().continuity(c.X_test, c.Z_test, k=10)),
    "knn_overlap": MetricSpec(lambda c: _m().knn_overlap(c.X_test, c.Z_test, k=10)),
    "linear_cka": MetricSpec(lambda c: _m().linear_cka(c.X_test, c.Z_test)),
    # ... and geometry of the output.
    "effective_rank": MetricSpec(lambda c: _m().effective_rank(c.Z_test)),
    "k_skewness": MetricSpec(lambda c: _m().k_skewness(
        c.Z_test, k=10, metric="cosine", hubness_reduction=c.hubness_reduction)),
}

DEFAULT_LABELLED = (
    "knn_accuracy", "knn_accuracy_euclidean", "linear_probe", "linear_probe_c1", "recall_at_10",
    "knn_r2", "trustworthiness", "continuity", "k_skewness", "effective_rank",
)
DEFAULT_UNLABELLED = (
    "trustworthiness", "continuity", "knn_overlap", "linear_cka", "k_skewness", "effective_rank",
)


def _resolve_metrics(
    metrics: Sequence[str] | None, has_labels: bool, kind: str | None, has_pairs: bool = False
):
    names = list(metrics) if metrics is not None else list(
        DEFAULT_LABELLED if has_labels else DEFAULT_UNLABELLED
    )
    if metrics is None and has_pairs:
        names.insert(0, "sts_spearman")
    if "sts_spearman" in names and not has_pairs:
        raise ValueError("Metric 'sts_spearman' needs test_pairs")
    unknown = [m for m in names if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}. Available: {sorted(METRICS)}")
    out = []
    for name in names:
        spec = METRICS[name]
        if spec.needs_labels and not has_labels:
            if metrics is not None:
                raise ValueError(f"Metric {name!r} needs y_train and y_test")
            continue
        if spec.needs_class_labels and kind != "class":
            if metrics is not None:
                raise ValueError(f"Metric {name!r} needs class labels, got {kind} labels")
            continue
        if spec.needs_continuous_labels and kind != "continuous":
            if metrics is not None:
                raise ValueError(f"Metric {name!r} needs continuous labels, got {kind} labels")
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------------------------------------
# evaluate_transform
# ---------------------------------------------------------------------------------------------


def evaluate_transform(
    make_transform: Callable[[int], Any],
    X_train,
    X_test,
    y_train=None,
    y_test=None,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    metrics: Sequence[str] | None = None,
    transductive: bool = False,
    fit_with_labels: bool = False,
    method: str | None = None,
    keep_per_point: bool = True,
    test_pairs: tuple | None = None,
    hubness_reduction=None,
) -> pd.DataFrame:
    """Fit a transform on the training split and score it on the test split, once per seed.

    Parameters
    ----------
    make_transform:
        ``make_transform(seed)`` returns a new, unfitted object with ``fit(X[, y])`` and
        ``transform(X)``. ``lambda s: EmbedKit(random_state=s)`` and
        ``lambda s: PCAReduce(64)`` both qualify. A deterministic transform only needs one seed.
    X_train, X_test, y_train, y_test:
        The splits. Labels are optional; label-based metrics are skipped without them.
        Labels are used only for scoring, unless ``fit_with_labels=True``.
    seeds:
        Seeds passed to ``make_transform``.
    metrics:
        Names from :data:`METRICS`. ``None`` picks :data:`DEFAULT_LABELLED` or
        :data:`DEFAULT_UNLABELLED`, dropping metrics the labels do not support. An explicitly
        requested metric that cannot be computed raises.
    transductive:
        Fit on the concatenation of both splits, without labels. Recorded in the output.
    fit_with_labels:
        Pass ``y_train`` to ``fit`` (supervised methods). Not allowed with ``transductive``.
    method:
        Name recorded in the ``method`` column (default: the class name of the transform).
    test_pairs:
        ``(idx_a, idx_b, scores)``: pairs of test rows with gold similarity scores, for
        ``sts_spearman`` (added to the default metrics when given).
    hubness_reduction:
        An object from :mod:`embedkit.hubness_reduction` (e.g. ``CSLS()``). The kNN-based
        metrics (``knn_accuracy``, ``knn_accuracy_euclidean``, ``recall_at_10``,
        ``k_skewness``) then search neighbours under its secondary distance, built on each
        metric's own base distance; the other metrics do not use neighbour search and are
        unchanged. Recorded in ``params`` as ``hubness_reduction``.
    keep_per_point:
        Keep the per-test-point outcomes (``per_point`` column) that :func:`compare` uses for
        its paired bootstrap.

    Returns
    -------
    A DataFrame with one row per (seed, metric) and the columns ``method``, ``seed``,
    ``metric``, ``value``, ``fit_time``, ``transform_time``, ``dim``, ``transductive``,
    ``params`` and ``per_point`` (an array or ``None``).
    """
    if transductive and fit_with_labels:
        raise ValueError("transductive=True fits without labels; fit_with_labels must be False")
    X_tr = _to_numpy(X_train)
    X_te = _to_numpy(X_test)
    if X_tr.shape[1] != X_te.shape[1]:
        raise ValueError("X_train and X_test have different numbers of columns")
    has_labels = y_train is not None and y_test is not None
    if (y_train is None) != (y_test is None):
        raise ValueError("Give both y_train and y_test, or neither")
    y_tr = y_te = None
    kind = None
    if has_labels:
        y_all, kind = encode_labels(
            np.concatenate([np.asarray(y_train), np.asarray(y_test)]),
            n_samples=len(X_tr) + len(X_te),
        )
        y_all = y_all.numpy()
        y_tr, y_te = y_all[: len(X_tr)], y_all[len(X_tr):]
    if fit_with_labels and not has_labels:
        raise ValueError("fit_with_labels=True needs labels")
    names = _resolve_metrics(metrics, has_labels, kind, has_pairs=test_pairs is not None)

    rows = []
    for seed in seeds:
        transform = make_transform(seed)
        X_fit = np.concatenate([X_tr, X_te]) if transductive else X_tr
        t0 = time.perf_counter()
        if fit_with_labels:
            transform.fit(X_fit, y_train)
        else:
            transform.fit(X_fit)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        Z_tr = _to_numpy(transform.transform(X_tr))
        Z_te = _to_numpy(transform.transform(X_te))
        transform_time = time.perf_counter() - t0
        if not (np.isfinite(Z_tr).all() and np.isfinite(Z_te).all()):
            raise FloatingPointError(
                f"{method or type(transform).__name__} (seed {seed}) produced NaN or inf"
            )
        ctx = _Context(X_tr, X_te, Z_tr, Z_te, y_tr, y_te, kind, test_pairs, hubness_reduction)
        params = _params_of(transform)
        if hubness_reduction is not None:
            params = {**params, "hubness_reduction": repr(hubness_reduction)}
        for name in names:
            out = METRICS[name].fn(ctx)
            value, per_point = out if isinstance(out, tuple) else (out, None)
            rows.append(
                dict(
                    method=method or type(transform).__name__,
                    seed=int(seed),
                    metric=name,
                    value=float(value),
                    fit_time=fit_time,
                    transform_time=transform_time,
                    dim=int(Z_te.shape[1]),
                    transductive=bool(transductive),
                    params=params,
                    per_point=(
                        np.asarray(per_point, dtype=np.float32)
                        if keep_per_point and per_point is not None
                        else None
                    ),
                )
            )
    return pd.DataFrame(rows)


def _params_of(obj) -> dict:
    """Constructor parameters of a transform, when it exposes them, for the record."""
    get = getattr(obj, "get_params", None)
    if callable(get):
        try:
            return {k: v for k, v in get().items() if _jsonable(v)}
        except TypeError:
            pass
    return {k: v for k, v in vars(obj).items() if not k.startswith("_") and _jsonable(v)}


def _jsonable(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str)) or (
        isinstance(v, (list, tuple)) and all(_jsonable(x) for x in v)
    )


# ---------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------


def compare(
    results: pd.DataFrame | Sequence[pd.DataFrame],
    reference: str = "raw",
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Summarise results per method and metric, and compare each method with a reference.

    Returns one row per (method, metric) with ``mean``, ``sd`` and ``n_seeds`` over seeds,
    ``dim``, and, for methods other than the reference, ``diff`` (mean minus the reference mean)
    with a ``diff_lo`` / ``diff_hi`` interval.

    The interval is a paired bootstrap over test points when both methods have per-point
    outcomes for that metric (kNN accuracy, classification probe, recall@k): each method's
    outcomes are averaged over its seeds, the per-point differences are resampled ``n_boot``
    times, and the interval is the percentile interval of their mean. Otherwise the interval is
    NaN. ``ci_method`` says which.
    """
    df = pd.concat(results, ignore_index=True) if not isinstance(results, pd.DataFrame) else results
    if reference not in set(df["method"]):
        raise ValueError(f"Reference method {reference!r} not in results")
    rng = np.random.default_rng(random_state)
    alpha = (1.0 - ci) / 2.0

    def mean_per_point(sub: pd.DataFrame):
        pps = [p for p in sub["per_point"] if p is not None]
        if len(pps) != len(sub) or not pps:
            return None
        lengths = {len(p) for p in pps}
        if len(lengths) != 1:
            return None
        return np.mean(np.stack(pps), axis=0)

    ref = {m: g for m, g in df[df["method"] == reference].groupby("metric")}
    rows = []
    for (method, metric), g in df.groupby(["method", "metric"], sort=False):
        row = dict(
            method=method,
            metric=metric,
            mean=float(g["value"].mean()),
            sd=float(g["value"].std(ddof=1)) if len(g) > 1 else 0.0,
            n_seeds=int(g["seed"].nunique()),
            dim=int(g["dim"].iloc[0]),
            diff=np.nan,
            diff_lo=np.nan,
            diff_hi=np.nan,
            ci_method="none",
        )
        if method != reference and metric in ref:
            r = ref[metric]
            row["diff"] = row["mean"] - float(r["value"].mean())
            pp_m, pp_r = mean_per_point(g), mean_per_point(r)
            if pp_m is not None and pp_r is not None and len(pp_m) == len(pp_r):
                d = pp_m - pp_r
                idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                boot = d[idx].mean(axis=1)
                row["diff_lo"], row["diff_hi"] = (
                    float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
                )
                row["ci_method"] = "paired_bootstrap_points"
        rows.append(row)
    out = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return out.sort_values(["metric", "method"], kind="stable").reset_index(drop=True)
