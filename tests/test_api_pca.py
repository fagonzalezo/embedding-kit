"""EmbedKit(method="pca"): the self-supervised default after E6/E7 (gate outcome C)."""

import numpy as np
import pytest
import torch

from embedkit import EmbedKit
from embedkit.baselines import PCAReduce

KW = dict(id_methods=["TwoNN"], device="cpu")


def _unit(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def test_self_supervised_default_is_pca(small_X):
    ek = EmbedKit(target_dim=8, **KW).fit(small_X)
    assert ek.method_ == "pca" and ek._config["method"] == "pca"
    assert ek._trainer.epochs == 0
    Z = ek.transform(small_X)
    ref = _unit(PCAReduce(8).fit_transform(small_X))
    # Same subspace and sign convention: identical up to float32 rounding.
    np.testing.assert_allclose(Z, ref, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)


def test_pca_prefixes_are_pca_at_lower_dim(small_X):
    ek = EmbedKit(target_dim=8, **KW).fit(small_X)
    np.testing.assert_allclose(ek.transform(small_X, dim=3),
                               _unit(PCAReduce(3).fit_transform(small_X)), atol=1e-4)
    Zt = ek.transform(torch.from_numpy(small_X), dim=3)
    assert isinstance(Zt, torch.Tensor)


def test_supervised_default_trains(small_X):
    y = np.arange(len(small_X)) % 3
    ek = EmbedKit(mode="supervised", target_dim=4, epochs=1, eval_every=100, **KW).fit(small_X, y)
    assert ek.method_ == "refiner" and ek._trainer.epochs == 1


def test_pca_rejects_training_configuration(small_X):
    for kw in (dict(loss=object()), dict(augmentation=object()), dict(nested_dims="auto"),
               dict(architecture="mlp")):
        with pytest.raises(ValueError, match="method='refiner'"):
            EmbedKit(target_dim=4, **kw, **KW).fit(small_X)
    with pytest.raises(ValueError, match="method must be"):
        EmbedKit(method="svd", **KW).fit(small_X)


def test_pca_save_load_roundtrip(small_X, tmp_path):
    ek = EmbedKit(target_dim=6, **KW).fit(small_X)
    ek.save(tmp_path / "b")
    ek2 = EmbedKit.load(tmp_path / "b")
    assert ek2.method_ == "pca" and ek2.method == "pca"
    np.testing.assert_allclose(ek2.transform(small_X), ek.transform(small_X), atol=1e-6)


def test_explicit_refiner_in_self_supervised_mode(small_X):
    ek = EmbedKit(method="refiner", target_dim=4, epochs=1, eval_every=100, **KW).fit(small_X)
    assert ek.method_ == "refiner" and ek._trainer.epochs == 1
