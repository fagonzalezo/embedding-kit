"""Phase 6: supervised and regression defaults (WP 6.1) and NC1 monitors (WP 6.2)."""

import numpy as np
import pytest

from embedkit import EmbedKit
from embedkit.evaluation.metrics import nc1
from embedkit.improvement.augmentation import GaussianNoise
from embedkit.improvement.losses import RankNContrastLoss, SupConLoss
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer

KW = dict(id_methods=["TwoNN"], device="cpu", random_state=0)


def _classes(n=120, d=16, c=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % c
    X = rng.standard_normal((n, d)).astype(np.float32) + 3.0 * np.eye(c, d)[y].astype(np.float32)
    return X, y


def _regression(n=120, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X, (X[:, 0] + 0.5 * X[:, 1]).astype(np.float64)


def test_supervised_defaults():
    X, y = _classes()
    ek = EmbedKit(mode="supervised", target_dim=4, epochs=4, eval_every=2, **KW).fit(X, y)
    t = ek._trainer
    assert type(t.loss) is SupConLoss and t.loss.spread_weight == 0.0  # no AlignUniform
    assert ek.monitor_ == "val_knn_accuracy" and t.monitor_mode == "max"
    assert ek.early_stopping_patience_ == 10 and t.val_split == 0.1 and t.restore_best
    for name in ("val_knn_accuracy", "train_nc1", "val_nc1"):
        assert len(t.history[name]) == 2 and np.isfinite(t.history[name]).all()


def test_regression_mode():
    X, y = _regression()
    ek = EmbedKit(mode="regression", target_dim=4, epochs=4, eval_every=2, **KW).fit(X, y)
    t = ek._trainer
    assert ek.method_ == "refiner" and isinstance(t.loss, RankNContrastLoss)
    assert ek.monitor_ == "val_ridge_r2" and t.monitor_mode == "max"
    assert len(t.history["val_ridge_r2"]) == 2
    # Integer targets are cast to float, so they are treated as continuous, not as classes.
    EmbedKit(mode="regression", target_dim=4, epochs=2, eval_every=2, **KW).fit(
        X, np.round(y).astype(int))
    assert ek.transform(X).shape == (len(X), 4)


def test_modes_need_labels_and_valid_name():
    X, _ = _classes()
    for mode in ("supervised", "regression"):
        with pytest.raises(ValueError, match="needs labels"):
            EmbedKit(mode=mode, **KW).fit(X)
    with pytest.raises(ValueError, match="mode must be"):
        EmbedKit(mode="ranking", **KW).fit(X)


def test_explicit_settings_override_auto():
    X, y = _classes()
    ek = EmbedKit(mode="supervised", target_dim=4, epochs=2, eval_every=1, monitor="uniformity",
                  early_stopping_patience=None, **KW).fit(X, y)
    assert ek.monitor_ == "uniformity" and ek._trainer.val_split == 0.0
    assert "train_nc1" in ek._trainer.history and "val_nc1" not in ek._trainer.history
    # Self-supervised refiner keeps no early stopping and k_skewness.
    ek = EmbedKit(method="refiner", target_dim=4, epochs=2, eval_every=1, **KW).fit(X)
    assert ek.monitor_ == "k_skewness" and ek.early_stopping_patience_ is None


def _trainer(loss, metrics, val_split=0.2, monitor=None):
    return Trainer(EmbeddingRefiner(16, 4, architecture="linear"), GaussianNoise(0.1), loss,
                   epochs=2, eval_every=1, eval_metrics=metrics, monitor=monitor or metrics[0],
                   val_split=val_split, device="cpu", random_state=0)


def test_nc1_monitors_match_metric():
    X, y = _classes()
    t = _trainer(SupConLoss(), ["train_nc1", "val_nc1"]).fit(X, y)
    ctx = t._eval_ctx
    Z_ref = t.transform(ctx["X_ref"])
    assert t.history["train_nc1"][-1] == pytest.approx(nc1(Z_ref, ctx["y_ref"]))
    assert len(t.history["val_nc1"]) == 2
    # train_nc1 works without a validation split; val_nc1 does not.
    _trainer(SupConLoss(), ["train_nc1"], val_split=0.0).fit(X, y)
    with pytest.raises(ValueError, match="val_split"):
        _trainer(SupConLoss(), ["val_nc1"], val_split=0.0).fit(X, y)


def test_label_kind_checks():
    Xc, yc = _classes()
    Xr, yr = _regression()
    with pytest.raises(ValueError, match="continuous labels"):
        _trainer(SupConLoss(), ["val_ridge_r2"]).fit(Xc, yc)
    with pytest.raises(ValueError, match="class labels"):
        _trainer(RankNContrastLoss(), ["val_knn_accuracy"]).fit(Xr, yr)
    with pytest.raises(ValueError, match="need labels"):
        _trainer(SupConLoss(), ["train_nc1"], val_split=0.0).fit(Xc)


def test_spread_keeps_more_within_class_variability():
    X, y = _classes(n=240, d=16, c=4, seed=1)
    out = {}
    for w in (0.0, 0.5):
        t = Trainer(EmbeddingRefiner(16, 8, architecture="linear"), GaussianNoise(0.05),
                    SupConLoss(0.1, spread_weight=w), epochs=30, batch_size=60, eval_every=30,
                    eval_metrics=["train_nc1"], monitor="train_nc1", device="cpu",
                    random_state=0, lr=1e-2).fit(X, y)
        out[w] = t.history["train_nc1"][-1]
    assert out[0.5] > out[0.0]
