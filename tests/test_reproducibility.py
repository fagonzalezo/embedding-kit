"""Reproducibility (WP 0.2, D13, D24): same random_state gives the same result on CPU, and
training leaves the caller's global random state alone."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embedkit import EmbedKit
from embedkit.improvement import (
    CompositeAugmentation,
    EmbeddingMixup,
    EmbeddingRefiner,
    FeatureDropout,
    FeatureMasking,
    GaussianNoise,
    KNNPairs,
    NTXentLoss,
    Trainer,
)


def _data(n=160, d=16, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _ek():
    return EmbedKit(
        target_dim=4, hidden_dim=32, epochs=6, batch_size=32, eval_every=2,
        early_stopping_patience=2, id_methods=["TwoNN"], device="cpu", random_state=0,
    )


def test_embedkit_fit_transform_is_reproducible():
    X = _data()
    a = _ek().fit_transform(X)
    b = _ek().fit_transform(X)
    np.testing.assert_array_equal(a, b)


def test_embedkit_different_seeds_differ():
    X = _data()
    a = _ek().fit_transform(X)
    ek = _ek()
    ek.random_state = 1
    b = ek.fit_transform(X)
    assert not np.array_equal(a, b)


AUGMENTATIONS = {
    "knn_pairs": lambda: KNNPairs(k=5),
    "noise": lambda: GaussianNoise(0.1),
    "dropout": lambda: FeatureDropout(0.2),
    "masking": lambda: FeatureMasking(0.25),
    "mixup": lambda: EmbeddingMixup(k=5),
    "composite": lambda: CompositeAugmentation([KNNPairs(k=5), GaussianNoise(0.1)]),
    "random_choice": lambda: CompositeAugmentation(
        [GaussianNoise(0.1), FeatureDropout(0.2)], mode="random_choice"
    ),
}


def _trainer_run(make_aug):
    torch.manual_seed(123)  # the model is built under the same seed in both runs
    model = EmbeddingRefiner(16, 4, hidden_dim=32, n_layers=1, architecture="mlp")
    t = Trainer(
        model, make_aug(), NTXentLoss(0.5), epochs=3, batch_size=32, eval_every=1,
        device="cpu", random_state=7,
    )
    X = _data()
    t.fit(X)
    return t.transform(X), t.history["loss"]


@pytest.mark.parametrize("name", list(AUGMENTATIONS))
def test_trainer_is_reproducible(name):
    z1, l1 = _trainer_run(AUGMENTATIONS[name])
    z2, l2 = _trainer_run(AUGMENTATIONS[name])
    np.testing.assert_array_equal(z1, z2)
    assert l1 == l2


@pytest.mark.parametrize("name", list(AUGMENTATIONS))
def test_trainer_leaves_global_state_alone(name):
    torch.manual_seed(0)
    model = EmbeddingRefiner(16, 4, hidden_dim=32, n_layers=1, architecture="mlp")
    t = Trainer(
        model, AUGMENTATIONS[name](), NTXentLoss(0.5), epochs=2, batch_size=32,
        eval_every=1, device="cpu", random_state=7,
    )
    np.random.seed(99)
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    t.fit(_data())
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    np.testing.assert_array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


def test_embedkit_leaves_global_state_alone():
    torch.manual_seed(5)
    torch_before = torch.get_rng_state().clone()
    np.random.seed(5)
    np_before = np.random.get_state()[1].copy()
    _ek().fit(_data())
    assert torch.equal(torch_before, torch.get_rng_state())
    np.testing.assert_array_equal(np_before, np.random.get_state()[1])


def test_knn_pairs_does_not_use_numpy_global_rng(monkeypatch):
    """D24: KNNPairs used np.random.randint."""

    def forbidden(*a, **k):
        raise AssertionError("np.random global state used")

    monkeypatch.setattr(np.random, "randint", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    aug = KNNPairs(k=5)
    X = _data()
    aug.precompute(X)
    aug.set_rng(np.random.default_rng(0), torch.Generator().manual_seed(0))
    aug(torch.from_numpy(X[:32]), torch.arange(32))
    aug2 = KNNPairs(k=5)
    aug2.set_rng(np.random.default_rng(0), torch.Generator().manual_seed(0))
    aug2(torch.from_numpy(X[:32]))


@pytest.mark.parametrize("name", list(AUGMENTATIONS))
def test_set_rng_makes_augmentation_deterministic(name):
    X = _data()
    outs = []
    for _ in range(2):
        aug = AUGMENTATIONS[name]()
        aug.precompute(X)
        aug.set_rng(np.random.default_rng(3), torch.Generator().manual_seed(3))
        torch.manual_seed(np.random.randint(1_000))  # global state differs between rounds
        outs.append(aug(torch.from_numpy(X[:32]), torch.arange(32)))
    for a, b in zip(outs[0], outs[1]):
        assert torch.equal(a, b)
