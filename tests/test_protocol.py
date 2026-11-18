"""Tests for embedkit.evaluation.protocol (WP 2.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embedkit.baselines import Identity, PCAReduce, RandomProjection
from embedkit.evaluation import METRICS, compare, evaluate_transform, knn_accuracy


@pytest.fixture(scope="module")
def blobs():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=400, n_features=20, centers=4, cluster_std=4.0, random_state=0)
    X = X.astype(np.float32)
    return X[:250], y[:250], X[250:], y[250:]


class Recorder:
    """A transform that records what it was fitted on."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y=None):
        self.n_fit_, self.y_ = len(X), y
        return self

    def transform(self, X):
        return np.asarray(X) + 0.0


def test_one_row_per_seed_and_metric(blobs):
    Xtr, ytr, Xte, yte = blobs
    df = evaluate_transform(lambda s: Identity(), Xtr, Xte, ytr, yte, seeds=(0, 1),
                            metrics=["knn_accuracy", "trustworthiness"])
    assert len(df) == 4
    assert set(df["metric"]) == {"knn_accuracy", "trustworthiness"}
    assert set(df["seed"]) == {0, 1}
    for col in ("method", "value", "fit_time", "transform_time", "dim", "transductive",
                "params", "per_point"):
        assert col in df.columns
    assert (df["method"] == "Identity").all()


def test_values_match_direct_metric_call(blobs):
    Xtr, ytr, Xte, yte = blobs
    df = evaluate_transform(lambda s: PCAReduce(5), Xtr, Xte, ytr, yte, seeds=(0,),
                            metrics=["knn_accuracy"])
    p = PCAReduce(5).fit(Xtr)
    expected = knn_accuracy(p.transform(Xtr), ytr, p.transform(Xte), yte)
    assert df["value"].iloc[0] == pytest.approx(expected)
    pp = df["per_point"].iloc[0]
    assert pp.shape == (len(Xte),) and pp.mean() == pytest.approx(expected)


def test_fits_on_training_split_only_without_labels(blobs):
    Xtr, ytr, Xte, yte = blobs
    made = []

    def make(seed):
        made.append(Recorder(seed))
        return made[-1]

    evaluate_transform(make, Xtr, Xte, ytr, yte, seeds=(3,), metrics=["knn_accuracy"])
    assert made[0].n_fit_ == len(Xtr) and made[0].y_ is None and made[0].seed == 3


def test_fit_with_labels_passes_y(blobs):
    Xtr, ytr, Xte, yte = blobs
    made = []
    evaluate_transform(lambda s: made.append(Recorder(s)) or made[-1], Xtr, Xte, ytr, yte,
                       seeds=(0,), metrics=["knn_accuracy"], fit_with_labels=True)
    np.testing.assert_array_equal(made[0].y_, ytr)


def test_transductive_fits_on_both_splits(blobs):
    Xtr, ytr, Xte, yte = blobs
    made = []
    df = evaluate_transform(lambda s: made.append(Recorder(s)) or made[-1], Xtr, Xte, ytr, yte,
                            seeds=(0,), metrics=["knn_accuracy"], transductive=True)
    assert made[0].n_fit_ == len(Xtr) + len(Xte) and made[0].y_ is None
    assert df["transductive"].all()
    with pytest.raises(ValueError):
        evaluate_transform(Recorder, Xtr, Xte, ytr, yte, transductive=True, fit_with_labels=True)


def test_default_metrics_depend_on_labels(blobs):
    Xtr, ytr, Xte, yte = blobs
    labelled = evaluate_transform(lambda s: Identity(), Xtr, Xte, ytr, yte, seeds=(0,))
    unlabelled = evaluate_transform(lambda s: Identity(), Xtr, Xte, seeds=(0,))
    assert "knn_accuracy" in set(labelled["metric"])
    assert "knn_accuracy" not in set(unlabelled["metric"])
    assert {"trustworthiness", "knn_overlap"} <= set(unlabelled["metric"])
    # Identity preserves the input structure exactly.
    t = unlabelled.set_index("metric")["value"]
    assert t["trustworthiness"] == pytest.approx(1.0)
    assert t["linear_cka"] == pytest.approx(1.0)


def test_explicit_label_metric_without_labels_raises(blobs):
    Xtr, _, Xte, _ = blobs
    with pytest.raises(ValueError, match="needs"):
        evaluate_transform(lambda s: Identity(), Xtr, Xte, metrics=["knn_accuracy"])
    with pytest.raises(ValueError, match="Unknown"):
        evaluate_transform(lambda s: Identity(), Xtr, Xte, metrics=["nope"])


def test_continuous_labels_skip_class_metrics(blobs):
    Xtr, _, Xte, _ = blobs
    ytr, yte = Xtr[:, 0] * 2.0, Xte[:, 0] * 2.0
    df = evaluate_transform(lambda s: Identity(), Xtr, Xte, ytr, yte, seeds=(0,))
    assert "linear_probe" in set(df["metric"]) and "knn_accuracy" not in set(df["metric"])
    assert df.set_index("metric")["value"]["linear_probe"] > 0.99  # R^2 of a linear target


def test_nonfinite_output_raises(blobs):
    Xtr, ytr, Xte, yte = blobs

    class Bad(Recorder):
        def transform(self, X):
            Z = np.asarray(X, dtype=np.float32).copy()
            Z[0, 0] = np.nan
            return Z

    with pytest.raises(FloatingPointError):
        evaluate_transform(Bad, Xtr, Xte, ytr, yte, seeds=(0,), metrics=["knn_accuracy"])


def test_string_labels(blobs):
    Xtr, ytr, Xte, yte = blobs
    names = np.array(["a", "b", "c", "d"])
    df = evaluate_transform(lambda s: Identity(), Xtr, Xte, names[ytr], names[yte], seeds=(0,),
                            metrics=["knn_accuracy"])
    ref = evaluate_transform(lambda s: Identity(), Xtr, Xte, ytr, yte, seeds=(0,),
                             metrics=["knn_accuracy"])
    assert df["value"].iloc[0] == ref["value"].iloc[0]


def test_every_registered_metric_runs(blobs):
    Xtr, ytr, Xte, yte = blobs
    pairs = (np.arange(0, 50), np.arange(50, 100), np.linspace(0, 1, 50))
    class_metrics = [m for m in METRICS if not METRICS[m].needs_continuous_labels]
    df = evaluate_transform(lambda s: PCAReduce(8), Xtr, Xte, ytr, yte, seeds=(0,),
                            metrics=class_metrics, test_pairs=pairs)
    assert set(df["metric"]) == set(class_metrics)
    assert np.isfinite(df["value"]).all()
    # The continuous-label metrics, on a continuous target.
    cont = [m for m in METRICS if METRICS[m].needs_continuous_labels]
    df = evaluate_transform(lambda s: PCAReduce(8), Xtr, Xte, Xtr[:, 0].astype(float),
                            Xte[:, 0].astype(float), seeds=(0,), metrics=cont)
    assert set(df["metric"]) == set(cont) and np.isfinite(df["value"]).all()


# --- compare ------------------------------------------------------------------------------------

def _two_methods(blobs):
    Xtr, ytr, Xte, yte = blobs
    raw = evaluate_transform(lambda s: Identity(), Xtr, Xte, ytr, yte, seeds=(0,),
                             metrics=["knn_accuracy", "trustworthiness"], method="raw")
    rp = evaluate_transform(lambda s: RandomProjection(2, random_state=s), Xtr, Xte, ytr, yte,
                            seeds=(0, 1, 2), metrics=["knn_accuracy", "trustworthiness"],
                            method="rp2")
    return raw, rp


def test_compare_summary_and_paired_bootstrap(blobs):
    raw, rp = _two_methods(blobs)
    out = compare([raw, rp], reference="raw", n_boot=500)
    row = out[(out.method == "rp2") & (out.metric == "knn_accuracy")].iloc[0]
    vals = rp[rp.metric == "knn_accuracy"]["value"]
    assert row["mean"] == pytest.approx(vals.mean())
    assert row["sd"] == pytest.approx(vals.std(ddof=1))
    assert row["n_seeds"] == 3
    ref_mean = raw[raw.metric == "knn_accuracy"]["value"].mean()
    assert row["diff"] == pytest.approx(row["mean"] - ref_mean)
    assert row["ci_method"] == "paired_bootstrap_points"
    assert row["diff_lo"] <= row["diff"] <= row["diff_hi"]
    # A 2-d random projection of 20-d blobs loses accuracy: the interval excludes 0.
    assert row["diff_hi"] < 0
    # Trustworthiness has no per-point outcomes: no interval.
    t = out[(out.method == "rp2") & (out.metric == "trustworthiness")].iloc[0]
    assert t["ci_method"] == "none" and np.isnan(t["diff_lo"])
    # The reference has no difference to itself.
    assert np.isnan(out[(out.method == "raw")]["diff"]).all()


def test_compare_identical_methods_interval_contains_zero(blobs):
    raw, _ = _two_methods(blobs)
    twin = raw.copy()
    twin["method"] = "twin"
    out = compare(pd.concat([raw, twin]), reference="raw", n_boot=200)
    row = out[(out.method == "twin") & (out.metric == "knn_accuracy")].iloc[0]
    assert row["diff"] == 0 and row["diff_lo"] == 0 and row["diff_hi"] == 0


def test_compare_unknown_reference_raises(blobs):
    raw, rp = _two_methods(blobs)
    with pytest.raises(ValueError, match="Reference"):
        compare([raw, rp], reference="missing")


def test_sts_spearman_from_pairs():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    a, b = np.arange(0, 30), np.arange(30, 60)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    gold = np.sum(Xn[a] * Xn[b], axis=1) + 0.01 * rng.standard_normal(30)
    df = evaluate_transform(lambda s: Identity(), X, X, seeds=(0,), test_pairs=(a, b, gold))
    assert df["metric"].iloc[0] == "sts_spearman"
    expected = spearmanr(np.sum(Xn[a] * Xn[b], axis=1), gold).statistic
    assert df.set_index("metric")["value"]["sts_spearman"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="test_pairs"):
        evaluate_transform(lambda s: Identity(), X, X, metrics=["sts_spearman"])
