"""WP 5.1: residual adapter and input standardisation."""

import numpy as np
import pytest
import torch

from embedkit.baselines import PCAReduce
from embedkit.improvement.augmentation import GaussianNoise
from embedkit.improvement.losses import NTXentLoss
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer


def _anisotropic(n=400, d=48, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.geomspace(20.0, 0.05, d)  # very different feature scales
    X = rng.standard_normal((n, d)) * scale + rng.normal(0, 30, d)  # large shared mean
    return X.astype(np.float32)


def _pca_l2(X_fit, X, k):
    Z = PCAReduce(k).fit(X_fit).transform(X).astype(np.float64)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


@pytest.mark.parametrize("architecture", ["residual", "linear"])
@pytest.mark.parametrize("k", [4, 16, 47])
def test_starts_at_pca(architecture, k):
    """Acceptance (WP 5.1): before any training step, transform = PCA(k) + L2, to 1e-4."""
    X = _anisotropic()
    X_new = _anisotropic(n=100, seed=1)
    torch.manual_seed(0)
    model = EmbeddingRefiner(48, k, architecture=architecture)
    model.initialize_from_data(X)
    model.eval()
    with torch.no_grad():
        Z = model(torch.from_numpy(X_new)).numpy()
    np.testing.assert_allclose(Z, _pca_l2(X, X_new, k), atol=1e-4)


def test_trainer_initialises_on_training_rows_only():
    """Trainer.fit calls initialize_from_data on the rows left after the validation split."""
    X = _anisotropic()
    model = EmbeddingRefiner(48, 8, architecture="residual")
    t = Trainer(model, GaussianNoise(), NTXentLoss(), epochs=1, lr=0.0, batch_size=64,
                val_split=0.25, eval_every=100, device="cpu", random_state=0)
    t.fit(X)
    mean = model.input_mean.numpy()
    assert not np.allclose(mean, X.mean(axis=0), atol=1e-3)  # not the full data
    n_val = int(len(X) * 0.25)
    val_idx = np.random.default_rng(0).choice(len(X), n_val, replace=False)
    train = np.delete(X, val_idx, axis=0)
    np.testing.assert_allclose(mean, train.mean(axis=0), rtol=1e-5, atol=1e-3)
    # lr = 0: the trained model is still PCA of the training rows
    np.testing.assert_allclose(t.transform(X), _pca_l2(train, X, 8), atol=1e-4)


def test_residual_branch_trains():
    X = _anisotropic()
    model = EmbeddingRefiner(48, 8, architecture="residual")
    t = Trainer(model, GaussianNoise(), NTXentLoss(), epochs=3, lr=1e-3, batch_size=64,
                eval_every=100, device="cpu", random_state=0)
    t.fit(X)
    assert model.net[-1].weight.abs().sum() > 0
    assert np.isfinite(t.history["loss"]).all()


def test_target_dim_above_rank_pads_with_zero_rows():
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((50, 3)) @ rng.standard_normal((3, 20))).astype(np.float32)
    model = EmbeddingRefiner(20, 6, architecture="linear")
    model.initialize_from_data(X)
    assert torch.all(model.linear.weight[3:] == 0)
    assert torch.all(model.linear.weight[:3].abs().sum(dim=1) > 0)


def test_constant_feature_is_not_divided_by_zero():
    X = _anisotropic()
    X[:, 5] = 7.0
    model = EmbeddingRefiner(48, 8, architecture="residual")
    model.initialize_from_data(X)
    assert model.input_scale[5] == 1.0
    with torch.no_grad():
        assert torch.isfinite(model(torch.from_numpy(X))).all()


def test_mlp_is_unchanged_by_initialize():
    torch.manual_seed(0)
    model = EmbeddingRefiner(48, 8, architecture="mlp")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.initialize_from_data(_anisotropic())
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    assert "input_mean" not in before


def test_unknown_architecture():
    with pytest.raises(ValueError, match="architecture"):
        EmbeddingRefiner(8, 4, architecture="transformer")


def test_save_load_round_trip(tmp_path):
    from embedkit import EmbedKit

    X = _anisotropic()
    ek = EmbedKit(method="refiner", target_dim=8, architecture="residual", epochs=2, eval_every=100,
                  device="cpu", random_state=0, id_methods=["TwoNN"]).fit(X)
    ek.save(tmp_path / "bundle")
    ek2 = EmbedKit.load(tmp_path / "bundle", device="cpu")
    assert ek2._trainer.model.architecture == "residual"
    np.testing.assert_allclose(ek2.transform(X), ek.transform(X), atol=1e-6)


def test_default_architecture_is_linear():
    model = EmbeddingRefiner(16, 4)
    assert model.architecture == "linear" and model.net is None


def test_warns_when_trained_uninitialised():
    import warnings

    X = torch.randn(32, 16)
    model = EmbeddingRefiner(16, 4)
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model(X)  # inference on an uninitialised model: no warning
    model.train()
    with pytest.warns(UserWarning, match="initialize_from_data"):
        model(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model(X)  # once per model

        init = EmbeddingRefiner(16, 4, architecture="residual")
        init.initialize_from_data(X.numpy())
        init(X)
        loaded = EmbeddingRefiner(16, 4)
        loaded.load_state_dict(EmbeddingRefiner(16, 4).state_dict())
        loaded(X)
        EmbeddingRefiner(16, 4, architecture="mlp")(X)


def test_trainer_initialises_without_warning():
    import warnings

    from embedkit.improvement.augmentation import GaussianNoise
    from embedkit.improvement.losses import NTXentLoss
    from embedkit.improvement.trainer import Trainer

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Trainer(EmbeddingRefiner(16, 4), GaussianNoise(0.1, relative=False), NTXentLoss(0.5),
                epochs=1, batch_size=32, eval_every=100, device="cpu",
                random_state=0).fit(np.random.default_rng(0).normal(size=(64, 16)))
