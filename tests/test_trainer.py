"""Regression tests for embedkit.improvement.trainer (WP 1.3: D5, D8, D11, D18, D19, D22, D23)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import embedkit.improvement.trainer as trainer_mod
from embedkit.improvement import EmbeddingRefiner, GaussianNoise, NTXentLoss, SupConLoss, Trainer


def _data(n=128, d=16, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _trainer(**kw):
    torch.manual_seed(0)
    defaults = dict(
        model=EmbeddingRefiner(16, 4, hidden_dim=32, n_layers=1, architecture="mlp"),
        augmentation=GaussianNoise(0.1, relative=False),
        loss=NTXentLoss(0.5),
        epochs=5,
        batch_size=32,
        device="cpu",
        random_state=0,
    )
    defaults.update(kw)
    return Trainer(**defaults)


@pytest.fixture
def register_monitor(monkeypatch):
    """Register a temporary monitor in the registry."""

    def _register(name, fn, direction):
        monitors = dict(trainer_mod.MONITORS)
        monitors[name] = (fn, direction)
        monkeypatch.setattr(trainer_mod, "MONITORS", monitors)

    return _register


# --- D5: StepLR is stepped per epoch ------------------------------------------------------------

def test_step_scheduler_is_stepped_per_epoch():
    # 30 epochs, step_size = 30 // 3 = 10, 4 batches per epoch.
    t = _trainer(epochs=30, scheduler="step", lr=1e-3, eval_every=1000)
    t.fit(_data(n=128))
    assert t.history["lr"][-1] == pytest.approx(1e-3 * 1e-3, rel=1e-6)
    assert t.history["lr"][9] == pytest.approx(1e-4, rel=1e-6)


def test_cosine_scheduler_reaches_zero():
    t = _trainer(epochs=6, scheduler="cosine", lr=1e-3, eval_every=1000)
    t.fit(_data())
    assert t.history["lr"][-1] == pytest.approx(0.0, abs=1e-9)
    assert t.history["lr"][0] < 1e-3


# --- D22: warm-up and LARS ----------------------------------------------------------------------

def test_warmup_increases_lr_then_hands_over():
    t = _trainer(epochs=8, scheduler="step", warmup_epochs=3, lr=1e-2, eval_every=1000)
    t.fit(_data())
    lrs = t.history["lr"]
    assert lrs[0] < lrs[1] < lrs[2]
    assert lrs[2] == pytest.approx(1e-2)


def test_warmup_default_is_zero():
    assert _trainer().warmup_epochs == 0


def test_lars_removed():
    with pytest.raises(ValueError, match="lars"):
        _trainer(optimizer="lars")


# --- D18: monitor direction ---------------------------------------------------------------------

def test_unknown_monitor_raises_at_construction():
    with pytest.raises(ValueError, match="monitor"):
        _trainer(monitor="not_a_monitor")
    with pytest.raises(ValueError, match="monitor"):
        _trainer(eval_metrics=["not_a_monitor"])


def test_registry_directions():
    assert trainer_mod.MONITORS["isotropy"][1] == "max"
    assert trainer_mod.MONITORS["uniformity"][1] == "min"
    assert trainer_mod.MONITORS["k_skewness"][1] == "min"


def test_max_monitor_does_not_stop_while_rising(register_monitor):
    calls = []

    def rising(Z, **kw):
        calls.append(1)
        return float(len(calls))

    register_monitor("rising", rising, "max")
    t = _trainer(epochs=10, eval_every=1, early_stopping_patience=1, monitor="rising")
    t.fit(_data())
    assert t.stopped_epoch_ == 10
    assert t.best_epoch_ == 10


def test_min_monitor_stops_when_rising(register_monitor):
    calls = []

    def rising(Z, **kw):
        calls.append(1)
        return float(len(calls))

    register_monitor("rising_min", rising, "min")
    t = _trainer(epochs=10, eval_every=1, early_stopping_patience=2, monitor="rising_min")
    t.fit(_data())
    assert t.best_epoch_ == 1
    assert t.stopped_epoch_ == 3


def test_plateau_uses_monitor_direction(register_monitor):
    calls = []

    def rising(Z, **kw):
        calls.append(1)
        return float(len(calls))

    register_monitor("rising2", rising, "max")
    t = _trainer(epochs=12, eval_every=1, scheduler="plateau", monitor="rising2", lr=1e-3)
    t.fit(_data())
    # A steadily improving (rising) max-monitor must never trigger an lr reduction.
    assert t.history["lr"][-1] == pytest.approx(1e-3)


# --- D19: best weights --------------------------------------------------------------------------

def test_restore_best_returns_best_model(register_monitor):
    def first_coord_mean(Z, **kw):
        return float(Z[:, 0].mean())

    register_monitor("z0", first_coord_mean, "min")
    X = _data(n=96)
    t = _trainer(
        epochs=60, eval_every=1, early_stopping_patience=3, monitor="z0", lr=3e-2,
        scheduler=None, restore_best=True,
    )
    t.fit(X)
    assert t.best_epoch_ < t.stopped_epoch_, "test needs early stopping to trigger"
    assert t.evaluate(X)["z0"] == pytest.approx(t.best_score_, abs=1e-6)
    assert t.history["z0"][t.history["eval_epoch"].index(t.best_epoch_)] == t.best_score_


def test_restore_best_false_keeps_last_model(register_monitor):
    def first_coord_mean(Z, **kw):
        return float(Z[:, 0].mean())

    register_monitor("z0b", first_coord_mean, "min")
    X = _data(n=96)
    t = _trainer(
        epochs=60, eval_every=1, early_stopping_patience=3, monitor="z0b", lr=3e-2,
        scheduler=None, restore_best=False,
    )
    t.fit(X)
    assert t.evaluate(X)["z0b"] == pytest.approx(t.history["z0b"][-1], abs=1e-6)


# --- D23: evaluation errors and settings --------------------------------------------------------

def test_evaluation_errors_propagate(register_monitor):
    def broken(Z, **kw):
        raise RuntimeError("boom")

    register_monitor("broken", broken, "min")
    t = _trainer(epochs=2, eval_every=1, monitor="broken", early_stopping_patience=2)
    with pytest.raises(RuntimeError, match="boom"):
        t.fit(_data())


def test_monitor_receives_trainer_metric_and_k(register_monitor):
    seen = {}

    def spy(Z, **kw):
        seen.update(kw)
        return 0.0

    register_monitor("spy", spy, "min")
    t = _trainer(epochs=1, eval_every=1, monitor="spy", eval_k=7, eval_metric="cosine")
    t.fit(_data())
    assert seen["k"] == 7 and seen["metric"] == "cosine"


def test_default_eval_metric_is_cosine():
    assert _trainer().eval_metric == "cosine"


def test_k_skewness_monitor_runs_with_cosine():
    t = _trainer(epochs=2, eval_every=1, eval_metrics=["k_skewness", "isotropy"])
    t.fit(_data())
    assert np.isfinite(t.history["k_skewness"]).all()
    assert np.isfinite(t.history["isotropy"]).all()


# --- D8: labels ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "y",
    [
        np.array(["cat", "dog", "bird", "dog"] * 32),
        ["a", "b"] * 64,
        list(np.random.default_rng(0).integers(0, 4, 128)),
    ],
    ids=["string_array", "string_list", "int_list"],
)
def test_fit_accepts_string_and_list_labels(y):
    t = _trainer(loss=SupConLoss(0.5), epochs=2)
    t.fit(_data(n=128), y)
    assert np.isfinite(t.history["loss"]).all()


def test_label_length_mismatch_raises():
    t = _trainer(loss=SupConLoss(0.5), epochs=1)
    with pytest.raises(ValueError, match="labels"):
        t.fit(_data(n=128), np.zeros(10, dtype=int))


# --- D11: device --------------------------------------------------------------------------------

def test_default_device_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert trainer_mod._default_device().type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert trainer_mod._default_device().type == "cpu"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
@pytest.mark.parametrize("loss_name", ["NTXentLoss", "SupConLoss", "AlignUniformLoss",
                                       "TripletLoss", "RankNContrastLoss"])
def test_every_loss_trains_on_mps(loss_name):
    """Every built-in loss runs natively on MPS (no CPU fallback)."""
    import embedkit.improvement.losses as L

    loss = getattr(L, loss_name)()
    y = np.random.default_rng(0).integers(0, 4, 128)
    t = _trainer(loss=loss, epochs=1, device="mps")
    t.fit(_data(n=128), y)
    assert t.device.type == "mps"
    assert np.isfinite(t.history["loss"]).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_fallback_to_cpu_when_loss_unsupported():
    """The probe falls back to CPU with a warning when the loss raises on MPS."""

    class MpsHostile(NTXentLoss):
        def forward(self, z_i, z_j, labels=None):
            if z_i.device.type == "mps":
                raise NotImplementedError("op not implemented for MPS")
            return super().forward(z_i, z_j, labels)

    t = _trainer(loss=MpsHostile(0.5), epochs=1)
    t.device = torch.device("mps")
    with pytest.warns(UserWarning, match="does not run on MPS"):
        t._select_device(_data(), None)
    assert t.device.type == "cpu"


# --- WP 2.4: label-free monitors and val_knn_accuracy -------------------------------------------

def test_new_monitors_registered_with_directions():
    M = trainer_mod.MONITORS
    assert M["knn_overlap"][1] == "max"
    assert M["effective_rank"][1] == "max"
    assert M["val_loss"][1] == "min"
    assert M["val_knn_accuracy"][1] == "max"


def test_label_free_monitors_are_logged():
    t = _trainer(epochs=2, eval_every=1, val_split=0.25,
                 eval_metrics=["knn_overlap", "effective_rank", "val_loss"])
    t.fit(_data(n=128))
    for name in ("knn_overlap", "effective_rank", "val_loss"):
        assert len(t.history[name]) == 2 and np.isfinite(t.history[name]).all()
    assert 0.0 <= min(t.history["knn_overlap"]) <= max(t.history["knn_overlap"]) <= 1.0
    assert 1.0 <= min(t.history["effective_rank"]) <= 4.0 + 1e-6  # output dim is 4


def test_val_loss_is_deterministic_for_a_fixed_model():
    t = _trainer(epochs=1, eval_every=1, val_split=0.25, eval_metrics=["val_loss"])
    X = _data(n=128)
    t.fit(X)
    a, b = t._held_out_loss(X[:40]), t._held_out_loss(X[:40])
    assert a == b


def test_val_loss_matches_training_loss_definition():
    """With no augmentation randomness and no dropout, val_loss is the loss on the rows."""
    from embedkit.improvement.augmentation import FeatureDropout

    torch.manual_seed(0)
    model = EmbeddingRefiner(16, 4, hidden_dim=32, n_layers=1, dropout=0.0, architecture="mlp")
    t = Trainer(model, FeatureDropout(p=0.0), NTXentLoss(0.5), epochs=1, batch_size=64,
                eval_every=1, val_split=0.25, eval_metrics=["val_loss"], device="cpu",
                random_state=0)
    X = _data(n=128)
    t.fit(X)
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(X[:64]))
        expected = float(NTXentLoss(0.5)(z, z))
    assert t._held_out_loss(X[:64]) == pytest.approx(expected, rel=1e-5)


def test_val_monitors_need_val_split_and_labels():
    with pytest.raises(ValueError, match="val_split"):
        _trainer(epochs=1, eval_metrics=["val_loss"]).fit(_data())
    with pytest.raises(ValueError, match="labels"):
        _trainer(epochs=1, val_split=0.2, eval_metrics=["val_knn_accuracy"]).fit(_data())


def test_val_knn_accuracy_uses_validation_labels(register_monitor):
    seen = {}

    def spy(Z, **kw):
        seen.update(kw)
        return 0.0

    register_monitor("spy_ctx", spy, "min")
    X = _data(n=200)
    y = (X[:, 0] > 0).astype(int)
    t = _trainer(loss=SupConLoss(0.5), epochs=3, eval_every=1, val_split=0.25,
                 eval_metrics=["val_knn_accuracy", "spy_ctx"], monitor="val_knn_accuracy")
    t.fit(X, y)
    assert len(seen["y"]) == 50 and seen["Z_ref"] is not None and len(seen["y_ref"]) == 150
    # The validation labels are the labels of the validation rows.
    val_rows = {tuple(r) for r in np.round(seen["X"], 5)}
    lookup = {tuple(r): lab for r, lab in zip(np.round(X, 5), y)}
    assert all(lookup[tuple(r)] == lab for r, lab in zip(np.round(seen["X"], 5), seen["y"]))
    assert len(val_rows) == 50
    # The logged value is 5-NN accuracy of the validation rows against the training pool.
    from embedkit.evaluation import knn_accuracy

    pool = np.array([r for r in X if tuple(np.round(r, 5)) not in val_rows])
    pool_y = np.array([lookup[tuple(r)] for r in np.round(pool, 5)])
    expected = knn_accuracy(t.transform(pool), pool_y, t.transform(seen["X"]), seen["y"], k=5)
    assert t.history["val_knn_accuracy"][-1] == pytest.approx(expected)
    assert t.monitor_mode == "max"


def test_public_evaluate_with_reference_pool():
    X = _data(n=120)
    y = (X[:, 0] > 0).astype(int)
    t = _trainer(epochs=1, eval_every=1000, eval_metrics=["val_knn_accuracy"], val_split=0.2)
    t.fit(X, y)
    out = t.evaluate(X[:30], y[:30], X_ref=X[30:], y_ref=y[30:])
    assert 0.0 <= out["val_knn_accuracy"] <= 1.0
