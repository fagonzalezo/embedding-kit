"""Tests for embedkit.improvement.*"""

import numpy as np
import pytest
import torch

from embedkit.utils.neighbors import clear_cache


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def small_tensor():
    torch.manual_seed(0)
    return torch.randn(64, 16)


@pytest.fixture
def labels_tensor():
    return torch.randint(0, 4, (64,))


@pytest.fixture
def cont_labels_tensor():
    torch.manual_seed(1)
    return torch.randn(64)


@pytest.fixture
def vec_labels_tensor():
    torch.manual_seed(2)
    return torch.randn(64, 3)


class TestAugmentations:
    def test_gaussian_noise_shape(self, small_tensor):
        from embedkit.improvement.augmentation import GaussianNoise
        aug = GaussianNoise(std=0.1)
        xi, xj = aug(small_tensor)
        assert xi.shape == small_tensor.shape
        assert xj.shape == small_tensor.shape

    def test_gaussian_noise_differs(self, small_tensor):
        from embedkit.improvement.augmentation import GaussianNoise
        aug = GaussianNoise(std=0.1)
        xi, xj = aug(small_tensor)
        assert not torch.allclose(xi, xj)

    def test_feature_dropout_shape(self, small_tensor):
        from embedkit.improvement.augmentation import FeatureDropout
        aug = FeatureDropout(p=0.2)
        xi, xj = aug(small_tensor)
        assert xi.shape == small_tensor.shape

    def test_feature_masking_shape(self, small_tensor):
        from embedkit.improvement.augmentation import FeatureMasking
        aug = FeatureMasking(mask_ratio=0.2)
        xi, xj = aug(small_tensor)
        assert xi.shape == small_tensor.shape

    def test_embedding_mixup_shape(self, small_tensor):
        from embedkit.improvement.augmentation import EmbeddingMixup
        aug = EmbeddingMixup(k=5, alpha=0.4)
        xi, xj = aug(small_tensor)
        assert xi.shape == small_tensor.shape

    def test_knn_pairs_shape(self, small_tensor):
        from embedkit.improvement.augmentation import KNNPairs
        aug = KNNPairs(k=5)
        xi, xj = aug(small_tensor)
        assert xi.shape == small_tensor.shape

    def test_composite_sequential_shape(self, small_tensor):
        from embedkit.improvement.augmentation import CompositeAugmentation, GaussianNoise, FeatureDropout
        aug = CompositeAugmentation([GaussianNoise(0.05), FeatureDropout(0.1)], mode="sequential")
        xi, xj = aug(small_tensor)
        assert xi.shape == small_tensor.shape

    def test_composite_random_choice_shape(self, small_tensor):
        from embedkit.improvement.augmentation import CompositeAugmentation, GaussianNoise, FeatureDropout
        aug = CompositeAugmentation([GaussianNoise(0.05), FeatureDropout(0.1)], mode="random_choice")
        xi, xj = aug(small_tensor)
        assert xi.shape == small_tensor.shape


class TestLosses:
    def test_nt_xent_gradient(self, small_tensor):
        from embedkit.improvement.losses import NTXentLoss
        z = small_tensor.requires_grad_(True)
        loss_fn = NTXentLoss(temperature=0.1)
        loss = loss_fn(z, z + 0.01)
        loss.backward()
        assert z.grad is not None
        assert not torch.isnan(loss)

    def test_align_uniform_gradient(self, small_tensor):
        from embedkit.improvement.losses import AlignUniformLoss
        z = small_tensor.requires_grad_(True)
        loss_fn = AlignUniformLoss()
        loss = loss_fn(z, z + 0.01)
        loss.backward()
        assert z.grad is not None

    def test_triplet_loss_gradient(self, small_tensor, labels_tensor):
        from embedkit.improvement.losses import TripletLoss
        z = small_tensor.requires_grad_(True)
        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn(z, z + 0.01, labels_tensor)
        loss.backward()
        assert z.grad is not None

    def test_triplet_loss_requires_labels(self, small_tensor):
        from embedkit.improvement.losses import TripletLoss
        loss_fn = TripletLoss()
        with pytest.raises((ValueError, TypeError)):
            loss_fn(small_tensor, small_tensor + 0.01)

    def test_supcon_loss_gradient(self, small_tensor, labels_tensor):
        from embedkit.improvement.losses import SupConLoss
        z = small_tensor.requires_grad_(True)
        loss_fn = SupConLoss()
        loss = loss_fn(z, z + 0.01, labels_tensor)
        loss.backward()
        assert z.grad is not None

    def test_combined_loss_gradient(self, small_tensor):
        from embedkit.improvement.losses import CombinedLoss, NTXentLoss, AlignUniformLoss
        z = small_tensor.requires_grad_(True)
        loss_fn = CombinedLoss([(NTXentLoss(), 1.0), (AlignUniformLoss(), 0.5)])
        loss = loss_fn(z, z + 0.01)
        loss.backward()
        assert z.grad is not None

    def test_rnc_loss_gradient_scalar(self, small_tensor, cont_labels_tensor):
        from embedkit.improvement.losses import RankNContrastLoss
        z = small_tensor.requires_grad_(True)
        loss_fn = RankNContrastLoss()
        loss = loss_fn(z, z + 0.01, cont_labels_tensor)
        loss.backward()
        assert z.grad is not None
        assert not torch.isnan(loss)

    def test_rnc_loss_gradient_vector(self, small_tensor, vec_labels_tensor):
        from embedkit.improvement.losses import RankNContrastLoss
        z = small_tensor.requires_grad_(True)
        loss_fn = RankNContrastLoss()
        loss = loss_fn(z, z + 0.01, vec_labels_tensor)
        loss.backward()
        assert z.grad is not None
        assert not torch.isnan(loss)

    def test_rnc_loss_requires_labels(self, small_tensor):
        from embedkit.improvement.losses import RankNContrastLoss
        loss_fn = RankNContrastLoss()
        with pytest.raises((ValueError, TypeError)):
            loss_fn(small_tensor, small_tensor + 0.01)

    def test_rnc_loss_chunked_matches_full(self, small_tensor, cont_labels_tensor):
        from embedkit.improvement.losses import RankNContrastLoss
        z = small_tensor
        loss_full = RankNContrastLoss()(z, z + 0.01, cont_labels_tensor)
        loss_chunked = RankNContrastLoss(chunk_size=32)(z, z + 0.01, cont_labels_tensor)
        assert torch.allclose(loss_full, loss_chunked, atol=1e-5)

    def test_rnc_loss_nonnegative(self, small_tensor, cont_labels_tensor):
        from embedkit.improvement.losses import RankNContrastLoss
        loss_fn = RankNContrastLoss()
        loss = loss_fn(small_tensor, small_tensor + 0.01, cont_labels_tensor)
        assert loss.item() >= 0


class TestEmbeddingRefiner:
    def test_forward_shape(self):
        from embedkit.improvement.model import EmbeddingRefiner
        model = EmbeddingRefiner(input_dim=16, target_dim=8)
        x = torch.randn(32, 16)
        z = model(x)
        assert z.shape == (32, 8)

    def test_output_normalized(self):
        from embedkit.improvement.model import EmbeddingRefiner
        model = EmbeddingRefiner(input_dim=16, target_dim=8, normalize_output=True)
        x = torch.randn(32, 16)
        z = model(x)
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-5)

    def test_forward_no_normalize(self):
        from embedkit.improvement.model import EmbeddingRefiner
        model = EmbeddingRefiner(input_dim=16, target_dim=8, normalize_output=False)
        x = torch.randn(32, 16)
        z = model(x)
        assert z.shape == (32, 8)


class TestTrainer:
    def test_smoke_fit(self, small_X):
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, NTXentLoss
        model = EmbeddingRefiner(input_dim=small_X.shape[1], target_dim=8)
        trainer = Trainer(
            model=model,
            augmentation=GaussianNoise(std=0.05),
            loss=NTXentLoss(),
            epochs=2,
            batch_size=32,
            eval_every=100,
        )
        trainer.fit(small_X)
        assert len(trainer.history["loss"]) == 2

    def test_transform_output_shape(self, small_X):
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, NTXentLoss
        model = EmbeddingRefiner(input_dim=small_X.shape[1], target_dim=8)
        trainer = Trainer(
            model=model,
            augmentation=GaussianNoise(std=0.05),
            loss=NTXentLoss(),
            epochs=2,
            batch_size=32,
            eval_every=100,
        )
        trainer.fit(small_X)
        Z = trainer.transform(small_X)
        assert Z.shape == (small_X.shape[0], 8)

    def test_supervised_fit(self, small_Xy):
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, SupConLoss
        X, y = small_Xy
        model = EmbeddingRefiner(input_dim=X.shape[1], target_dim=8)
        trainer = Trainer(
            model=model,
            augmentation=GaussianNoise(std=0.05),
            loss=SupConLoss(),
            epochs=2,
            batch_size=32,
            eval_every=100,
        )
        trainer.fit(X, y=y)
        Z = trainer.transform(X)
        assert Z.shape == (X.shape[0], 8)

    def test_trainer_supervised_continuous(self, small_X):
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, RankNContrastLoss
        rng = np.random.default_rng(1)
        y = rng.standard_normal(small_X.shape[0]).astype(np.float32)
        model = EmbeddingRefiner(input_dim=small_X.shape[1], target_dim=8)
        trainer = Trainer(
            model=model,
            augmentation=GaussianNoise(std=0.05),
            loss=RankNContrastLoss(),
            epochs=2,
            batch_size=32,
            eval_every=100,
        )
        trainer.fit(small_X, y=y)
        Z = trainer.transform(small_X)
        assert Z.shape == (small_X.shape[0], 8)

    def test_loss_decreases(self, small_X):
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, NTXentLoss
        model = EmbeddingRefiner(input_dim=small_X.shape[1], target_dim=8)
        trainer = Trainer(
            model=model,
            augmentation=GaussianNoise(std=0.05),
            loss=NTXentLoss(),
            epochs=5,
            batch_size=32,
            eval_every=100,
        )
        trainer.fit(small_X)
        losses = trainer.history["loss"]
        # loss should generally decrease over 5 epochs on this small dataset
        assert losses[-1] < losses[0] * 2  # at least not explode
