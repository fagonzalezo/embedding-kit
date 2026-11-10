"""Tests for the augmentation contracts (WP 1.4: D9, D24, D25, D26)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embedkit.improvement.augmentation import (
    BaseAugmentation,
    CompositeAugmentation,
    EmbeddingMixup,
    FeatureDropout,
    FeatureMasking,
    GaussianNoise,
    KNNPairs,
)
from embedkit.utils.neighbors import knn


def _unit_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _match_rows(x_j: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Index of the row of ``ref`` that each row of ``x_j`` equals exactly (-1 if none)."""
    eq = (x_j[:, None, :] == ref[None, :, :]).all(dim=2)
    found = eq.any(dim=1)
    idx = eq.float().argmax(dim=1)
    return torch.where(found, idx, torch.full_like(idx, -1))


# ----------------------------------------------------------------------------- D9 composition


class TestCompositionContract:
    def test_kinds(self):
        assert KNNPairs.kind == "pair"
        assert EmbeddingMixup.kind == "pair"
        assert GaussianNoise.kind == "view"
        assert FeatureDropout.kind == "view"
        assert FeatureMasking.kind == "view"
        assert BaseAugmentation.kind in ("pair", "view")

    def test_view_before_pair_raises(self):
        with pytest.raises(ValueError):
            CompositeAugmentation([GaussianNoise(0.1), KNNPairs(k=3)])

    def test_two_pair_augmentations_raise(self):
        with pytest.raises(ValueError):
            CompositeAugmentation([KNNPairs(k=3), EmbeddingMixup(k=3)])
        with pytest.raises(ValueError):
            CompositeAugmentation([KNNPairs(k=3), KNNPairs(k=3)])

    def test_nested_pair_composite_is_checked(self):
        inner = CompositeAugmentation([KNNPairs(k=3), GaussianNoise(0.1)])
        assert inner.kind == "pair"
        with pytest.raises(ValueError):
            CompositeAugmentation([inner, KNNPairs(k=3)])
        with pytest.raises(ValueError):
            CompositeAugmentation([GaussianNoise(0.1), inner])
        outer = CompositeAugmentation([inner, FeatureDropout(0.1)])
        assert outer.kind == "pair"

    def test_view_only_composite_is_a_view(self):
        assert CompositeAugmentation([GaussianNoise(0.1), FeatureDropout(0.1)]).kind == "view"

    def test_pair_then_noise_gives_two_noisy_views(self):
        X = _unit_gaussian(60, 32)
        aug = CompositeAugmentation([KNNPairs(k=1), GaussianNoise(std=0.5)])
        aug.precompute(X)
        x = torch.from_numpy(X)
        idx = torch.arange(len(X))
        x_i, x_j = aug(x, idx)

        # view i is a noised anchor, view j a noised neighbour: neither is a dataset row
        assert not (x_i == x).all(dim=1).any()
        assert _match_rows(x_j, x).max().item() == -1
        assert _match_rows(x_i, x).max().item() == -1

        # view i is centred on the anchor, view j on its (deterministic, k=1) neighbour
        nbr = torch.from_numpy(knn(X, 1)[1][:, 0])
        noise_i = x_i - x
        noise_j = x_j - x[nbr]
        assert noise_i.abs().mean() > 0 and noise_j.abs().mean() > 0
        assert not torch.allclose(noise_i, noise_j)
        # both draws have the configured scale (std 0.5 times per-dimension data std)
        expected = 0.5 * float(X.std(axis=0).mean())
        assert noise_i.std().item() == pytest.approx(expected, rel=0.15)
        assert noise_j.std().item() == pytest.approx(expected, rel=0.15)

    def test_views_get_independent_draws_from_each_view_augmentation(self):
        x = torch.ones(64, 64)
        aug = CompositeAugmentation([FeatureMasking(0.25), FeatureMasking(0.25)])
        x_i, x_j = aug(x)
        assert not torch.equal(x_i, x_j)

    def test_random_choice_does_not_use_python_random(self):
        import random

        x = torch.randn(16, 8)
        aug = CompositeAugmentation(
            [GaussianNoise(0.1, relative=False), FeatureDropout(0.5)], mode="random_choice"
        )
        random.seed(0)
        torch.manual_seed(123)
        a = aug(x)
        random.seed(1)
        torch.manual_seed(123)
        b = aug(x)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        # both members get picked over many calls
        picks = set()
        for _ in range(50):
            xi, _ = aug(torch.ones(4, 8))
            picks.add("dropout" if (xi == 0).any() else "noise")
        assert picks == {"dropout", "noise"}

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            CompositeAugmentation([GaussianNoise(0.1)], mode="parallel")


# ----------------------------------------------------------------------------- D24 KNNPairs


class TestKNNPairs:
    def test_hard_negatives_argument_removed(self):
        with pytest.raises(TypeError):
            KNNPairs(k=3, hard_negatives=True)

    def test_in_batch_samples_among_k_nearest(self):
        torch.manual_seed(0)
        x = torch.randn(40, 8)
        k = 5
        _, nbrs = knn(x.numpy(), k)
        aug = KNNPairs(k=k)
        seen = [set() for _ in range(len(x))]
        for _ in range(60):
            x_i, x_j = aug(x)
            assert torch.equal(x_i, x)
            chosen = _match_rows(x_j, x)
            assert (chosen >= 0).all()
            for r, c in enumerate(chosen.tolist()):
                assert c != r
                assert c in set(nbrs[r].tolist())
                seen[r].add(c)
        # the neighbour varies (the old code always returned the nearest one)
        assert np.mean([len(s) for s in seen]) > 3

    def test_in_batch_k_is_capped_at_batch_size_minus_one(self):
        x = torch.randn(4, 6)
        _, x_j = KNNPairs(k=10)(x)
        chosen = _match_rows(x_j, x)
        assert (chosen >= 0).all()
        assert (chosen != torch.arange(4)).all()

    def test_cosine_metric_in_batch(self):
        torch.manual_seed(1)
        x = torch.randn(30, 6) * torch.rand(30, 1).mul(5).add(0.1)  # varied norms
        cos_nn = knn(x.numpy(), 1, metric="cosine")[1][:, 0]
        euc_nn = knn(x.numpy(), 1, metric="euclidean")[1][:, 0]
        assert (cos_nn != euc_nn).any()
        _, x_j = KNNPairs(k=1, metric="cosine")(x)
        assert _match_rows(x_j, x).tolist() == cos_nn.tolist()
        _, x_j = KNNPairs(k=1)(x)  # default is euclidean
        assert _match_rows(x_j, x).tolist() == euc_nn.tolist()

    def test_cosine_metric_precomputed(self):
        torch.manual_seed(2)
        x = torch.randn(30, 6) * torch.rand(30, 1).mul(5).add(0.1)
        cos_nn = knn(x.numpy(), 1, metric="cosine")[1][:, 0]
        aug = KNNPairs(k=1, metric="cosine")
        aug.precompute(x.numpy())
        _, x_j = aug(x, torch.arange(30))
        assert _match_rows(x_j, x).tolist() == cos_nn.tolist()

    @pytest.mark.parametrize("precomputed", [False, True])
    def test_uses_torch_rng(self, precomputed):
        x = torch.randn(50, 6)
        aug = KNNPairs(k=8)
        idx = None
        if precomputed:
            aug.precompute(x.numpy())
            idx = torch.arange(50)
        torch.manual_seed(7)
        _, a = aug(x, idx)
        torch.manual_seed(7)
        _, b = aug(x, idx)
        assert torch.equal(a, b)
        _, c = aug(x, idx)
        assert not torch.equal(a, c)


# ----------------------------------------------------------------------------- D25 FeatureMasking


class TestFeatureMasking:
    def test_masks_differ_across_rows(self):
        x = torch.ones(64, 40)
        x_i, x_j = FeatureMasking(mask_ratio=0.25)(x)
        for v in (x_i, x_j):
            assert len({tuple(r.tolist()) for r in v}) > 32
        assert not torch.equal(x_i, x_j)

    def test_per_row_masked_count_block_size_one(self):
        x = torch.ones(200, 40)
        for ratio, expected in [(0.25, 10), (0.5, 20), (0.01, 1)]:
            x_i, x_j = FeatureMasking(mask_ratio=ratio, block_size=1)(x)
            for v in (x_i, x_j):
                assert ((v == 0).sum(dim=1) == expected).all()
                assert set(v.unique().tolist()) <= {0.0, 1.0}

    def test_masks_are_not_shared_across_views(self):
        x = torch.ones(200, 40)
        x_i, x_j = FeatureMasking(mask_ratio=0.25)(x)
        assert ((x_i == 0) != (x_j == 0)).any(dim=1).float().mean() > 0.9

    def test_blocks_are_contiguous_and_per_row(self):
        x = torch.ones(100, 60)
        x_i, _ = FeatureMasking(mask_ratio=0.1, block_size=6)(x)  # one block of 6
        masked = x_i == 0
        assert (masked.sum(dim=1) == 6).all()
        for row in masked:
            pos = row.nonzero().flatten()
            assert (pos[1:] - pos[:-1] == 1).all()
        starts = {int(r.nonzero()[0]) for r in masked}
        assert len(starts) > 10

    def test_preserves_values_and_dtype(self):
        x = torch.randn(10, 20, dtype=torch.float64)
        x_i, _ = FeatureMasking(0.3)(x)
        assert x_i.dtype == torch.float64
        kept = x_i != 0
        assert torch.equal(x_i[kept], x[kept])


# ----------------------------------------------------------------------------- D26 GaussianNoise


class TestGaussianNoise:
    def test_relative_noise_scales_with_centred_signal(self):
        X = _unit_gaussian(2000, 768)
        aug = GaussianNoise(std=0.1)
        assert aug.relative is True
        aug.precompute(X)
        x = torch.from_numpy(X[:500])
        x_i, x_j = aug(x)
        signal = (x - torch.from_numpy(X.mean(axis=0))).norm(dim=1).mean().item()
        for v in (x_i, x_j):
            noise = (v - x).norm(dim=1).mean().item()
            assert noise / signal == pytest.approx(0.1, rel=0.1)

    def test_relative_false_keeps_absolute_std(self):
        x = torch.zeros(500, 100)
        x_i, _ = GaussianNoise(std=0.3, relative=False)(x)
        assert x_i.std().item() == pytest.approx(0.3, rel=0.05)

    def test_relative_per_dimension(self):
        rng = np.random.default_rng(0)
        scales = np.array([0.01, 1.0, 100.0], dtype=np.float32)
        X = rng.standard_normal((4000, 3)).astype(np.float32) * scales
        aug = GaussianNoise(std=0.2)
        aug.precompute(X)
        x = torch.zeros(4000, 3)
        noise = aug(x)[0]
        assert noise.std(dim=0).numpy() == pytest.approx(0.2 * scales, rel=0.1)

    def test_fallback_to_batch_std_without_precompute(self):
        rng = np.random.default_rng(1)
        scales = np.array([0.1, 10.0], dtype=np.float32)
        x = torch.from_numpy(rng.standard_normal((4000, 2)).astype(np.float32) * scales)
        noise = GaussianNoise(std=0.5)(x)[0] - x
        assert noise.std(dim=0).numpy() == pytest.approx(0.5 * scales, rel=0.1)

    def test_adaptive_ignores_relative(self):
        torch.manual_seed(0)
        x = torch.randn(50, 6)
        outs = []
        for rel in (True, False):
            aug = GaussianNoise(std=0.2, adaptive=True, k=5, relative=rel)
            aug.precompute(x.numpy())
            torch.manual_seed(5)
            outs.append(aug(x))
        assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_in_batch_k_is_capped_for_mixup_and_adaptive_noise():
    """k larger than the batch used to make the in-batch kNN search fail."""
    from embedkit.improvement.augmentation import EmbeddingMixup, GaussianNoise

    x = torch.randn(6, 4)
    xi, xj = EmbeddingMixup(k=10)(x)
    assert xi.shape == xj.shape == x.shape
    xi, xj = GaussianNoise(0.1, adaptive=True, k=10)(x)
    assert torch.isfinite(xi).all() and torch.isfinite(xj).all()
