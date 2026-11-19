"""EmbedKit(strategy="select"), the default since WP 7.2."""

import numpy as np
import pandas as pd
import pytest
import torch

from embedkit import EmbedKit
from embedkit.baselines import CenterNormalize

KW = dict(device="cpu", random_state=0)


def _noisy_classes(n=1500, d=1000, n_classes=5, seed=0):
    """Class signal in 8 high-variance directions, plus many low-variance noise columns whose
    total variance swamps it: PCA keeps the signal and drops most of the noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    means = rng.standard_normal((n_classes, 8)) * 0.6
    signal = means[y] + rng.standard_normal((n, 8))
    noise = rng.standard_normal((n, d - 8)) * 1.5
    return np.hstack([signal * 3.0, noise]).astype(np.float32), y


def test_without_labels_the_default_is_center_l2():
    X = np.random.default_rng(0).standard_normal((200, 16)).astype(np.float32) + 3.0
    ek = EmbedKit(**KW).fit(X)
    assert ek.selected_ == "center_l2" and ek.strategy_ == "select"
    assert ek.selection_report_ is None and ek.analysis_report is None
    np.testing.assert_allclose(ek.transform(X), CenterNormalize().fit(X).transform(X),
                               atol=1e-6)
    Zt = ek.transform(torch.from_numpy(X))
    assert isinstance(Zt, torch.Tensor) and Zt.shape == (200, 16)


def test_labels_select_a_transform_that_beats_identity():
    X, y = _noisy_classes()
    ek = EmbedKit(mode="self_supervised", **KW).fit(X, y)
    t = ek.selection_report_
    assert list(t["candidate"]) == ["identity", "center_l2", "pca-64", "pca-128", "pca-256"]
    assert t["selected"].sum() == 1 and ek.selected_ == t.loc[t["selected"], "candidate"].item()
    assert ek.selected_.startswith("pca-"), t
    row = t[t["selected"]].iloc[0]
    assert row["beats_identity"] and row["diff_lo"] > 0
    assert t.attrs["criterion"] == "knn_accuracy" and t.attrs["n_val"] == 300
    k = int(ek.selected_.split("-")[1])
    assert ek.transform(X).shape == (len(X), k) and ek.method_ == "pca"


def test_identity_is_kept_without_a_significant_gain():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((600, 32)).astype(np.float32)
    y = rng.integers(0, 3, 600)  # labels unrelated to X
    ek = EmbedKit(mode="self_supervised", **KW).fit(X, y)
    assert ek.selected_ == "identity", ek.selection_report_
    np.testing.assert_array_equal(ek.transform(X), X)


def test_supervised_selection_includes_the_refiner():
    X, y = _noisy_classes(n=400, d=40)
    ek = EmbedKit(epochs=2, eval_every=1, **KW).fit(X, y)  # mode="auto" with labels
    assert ek.mode_ == "supervised"
    names = list(ek.selection_report_["candidate"])
    assert names[-1] == "refiner-20" and "pca-64" not in names  # 64 is not below d = 40
    if ek.selected_ == "refiner-20":
        assert ek.method_ == "refiner" and ek.transform(X).shape == (400, 20)


def test_regression_uses_knn_r2():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 12)).astype(np.float32)
    y = X[:, 0] * 2 + 0.1 * rng.standard_normal(300)
    ek = EmbedKit(mode="regression", epochs=1, eval_every=100, **KW).fit(X, y)
    assert ek.selection_report_.attrs["criterion"] == "knn_regression_r2"
    assert "refiner-6" in set(ek.selection_report_["candidate"])


def test_too_few_rows_return_identity_with_a_warning():
    X, y = _noisy_classes(n=60, d=20)
    with pytest.warns(UserWarning, match="Too few rows"):
        ek = EmbedKit(mode="self_supervised", **KW).fit(X, y)
    assert ek.selected_ == "identity"


def test_continuous_labels_need_regression_mode():
    X = np.random.default_rng(0).standard_normal((100, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="mode='regression'"):
        EmbedKit(**KW).fit(X, np.linspace(0, 1, 100))


def test_training_configuration_without_a_refiner_raises():
    X = np.random.default_rng(0).standard_normal((100, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="nested_dims configure training"):
        EmbedKit(nested_dims="auto", **KW).fit(X)


def test_explicit_method_means_fixed():
    X = np.random.default_rng(0).standard_normal((100, 16)).astype(np.float32)
    ek = EmbedKit(method="pca", target_dim=4, id_methods=["TwoNN"], **KW).fit(X)
    assert ek.strategy_ == "fixed" and ek.selected_ is None
    assert ek.transform(X).shape == (100, 4)


@pytest.mark.parametrize("labels", [False, True])
def test_save_load_roundtrip(tmp_path, labels):
    X, y = _noisy_classes(n=600, d=100)
    ek = EmbedKit(mode="self_supervised", **KW).fit(X, y if labels else None)
    ek.save(tmp_path / "b")
    back = EmbedKit.load(tmp_path / "b", device="cpu")
    assert back.selected_ == ek.selected_ and back.strategy_ == "select"
    assert back.mode_ == ek.mode_
    if labels:
        pd.testing.assert_frame_equal(back.selection_report_, ek.selection_report_,
                                      check_dtype=False)
        assert back.selection_report_.attrs == ek.selection_report_.attrs
    else:
        assert back.selection_report_ is None
    np.testing.assert_allclose(back.transform(X), ek.transform(X), atol=1e-5)


def test_report_k_and_removed_aliases():
    from embedkit import EmbedKitAnalyzer

    X = np.random.default_rng(0).standard_normal((300, 8)).astype(np.float32)
    r = EmbedKitAnalyzer(id_methods=["TwoNN"], calibrate=False, id_bootstrap=0,
                         id_scale_curve=False).fit(X)
    assert r.k == 9 and r.kernel.sigma > 0
    # Deprecated in 0.6.0, removed in 0.7.0 (WP 8.5).
    assert not hasattr(r, "suggested_k") and not hasattr(r, "suggested_sigma")


# ---------------------------------------------------------------------------------------------
# WP 8.2: selection on a subsample of rows
# ---------------------------------------------------------------------------------------------


def test_selection_on_a_subsample_refits_the_winner_on_all_rows():
    from embedkit.api import _l2, _pca

    X, y = _noisy_classes(n=2500)
    ek = EmbedKit(mode="self_supervised", selection_max_rows=1500, **KW).fit(X, y)
    t = ek.selection_report_
    assert t.attrs["n_rows"] == 2500 and t.attrs["n_selection_rows"] == 1500
    assert t.attrs["n_val"] == 300 and t.attrs["n_fit"] == 1200
    assert ek.selected_.startswith("pca-"), t
    # the winner is fitted on all 2500 rows, not on the subsample
    k = int(ek.selected_.split("-")[1])
    mean, comps, _ = _pca(X)
    np.testing.assert_allclose(ek.transform(X), _l2((X - mean) @ comps[:k].T), atol=1e-4)


def test_the_subsample_is_stratified_by_class():
    from embedkit.api import _selection_subsample

    y = np.repeat([0, 1, 2], [9000, 900, 100])
    sub = _selection_subsample(y, 1000, True, 0)
    assert len(sub) == 1000 and np.all(np.diff(sub) > 0)
    np.testing.assert_array_equal(np.bincount(y[sub]), [900, 90, 10])


@pytest.mark.parametrize("cap", [None, 600, 10_000])
def test_a_cap_that_does_not_bind_changes_nothing(cap):
    X, y = _noisy_classes(n=600, d=100)
    ref = EmbedKit(mode="self_supervised", selection_max_rows=None, **KW).fit(X, y)
    ek = EmbedKit(mode="self_supervised", selection_max_rows=cap, **KW).fit(X, y)
    pd.testing.assert_frame_equal(ek.selection_report_, ref.selection_report_)
    assert ek.selection_report_.attrs["n_selection_rows"] == 600


def test_regression_selection_on_a_subsample():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 12)).astype(np.float32)
    y = X[:, 0] * 2 + 0.1 * rng.standard_normal(400)
    ek = EmbedKit(mode="regression", epochs=1, eval_every=100, selection_max_rows=200,
                  **KW).fit(X, y)
    assert ek.selection_report_.attrs["n_selection_rows"] == 200
    assert ek.transform(X).shape[0] == 400


@pytest.mark.parametrize("cap", [0, 1, 2.5, True, "50000"])
def test_invalid_selection_max_rows_raises(cap):
    X, y = _noisy_classes(n=200, d=20)
    with pytest.raises(ValueError, match="selection_max_rows"):
        EmbedKit(mode="self_supervised", selection_max_rows=cap, **KW).fit(X, y)


def test_baseline_refit_matches_the_baseline_fit():
    from embedkit.api import _fitted_baseline

    X = (np.random.default_rng(0).standard_normal((500, 64)) * 3 + 7).astype(np.float32)
    fast, ref = _fitted_baseline("center_l2", X), CenterNormalize().fit(X)
    assert fast.mean_.dtype == np.float64
    np.testing.assert_allclose(fast.mean_, ref.mean_, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fast.transform(X), ref.transform(X))
