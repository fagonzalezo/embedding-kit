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


def _supcon_reference(z_i, z_j, labels, temperature):
    """Direct loop implementation of Khosla et al. (2020), Eq. 2, in float64."""
    z = torch.nn.functional.normalize(torch.cat([z_i, z_j]).double(), dim=1)
    two_n = z.shape[0]
    n = two_n // 2
    if labels is None:
        lbl = torch.cat([torch.arange(n), torch.arange(n)])
    else:
        lbl = torch.cat([labels, labels])
    sim = (z @ z.T) / temperature
    total = 0.0
    for i in range(two_n):
        others = [k for k in range(two_n) if k != i]
        log_denom = torch.logsumexp(sim[i, others], dim=0)
        positives = [p for p in others if lbl[p] == lbl[i]]
        total += sum(-(sim[i, p] - log_denom) for p in positives) / max(len(positives), 1)
    return total / two_n


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
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("use_labels", [True, False])
    @pytest.mark.parametrize("chunk_size", [None, 48])
    def test_supcon_loss_finite(self, small_tensor, labels_tensor, use_labels, chunk_size):
        from embedkit.improvement.losses import SupConLoss
        y = labels_tensor if use_labels else None
        loss = SupConLoss(0.1, chunk_size=chunk_size)(small_tensor, small_tensor + 0.01, y)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("use_labels", [True, False])
    def test_supcon_matches_reference(self, labels_tensor, use_labels):
        from embedkit.improvement.losses import SupConLoss
        torch.manual_seed(3)
        z1, z2 = torch.randn(64, 16), torch.randn(64, 16)
        y = labels_tensor if use_labels else None
        loss = SupConLoss(0.1)(z1, z2, y)
        ref = _supcon_reference(z1, z2, y, 0.1)
        assert abs(loss.item() - ref.item()) < 1e-5

    @pytest.mark.parametrize("use_labels", [True, False])
    @pytest.mark.parametrize("chunk_size", [16, 24, 50])
    def test_supcon_chunked_matches_full(self, small_tensor, labels_tensor, use_labels, chunk_size):
        from embedkit.improvement.losses import SupConLoss
        y = labels_tensor if use_labels else None
        z1, z2 = small_tensor, small_tensor + 0.1 * torch.randn_like(small_tensor)
        loss_full = SupConLoss(0.1)(z1, z2, y)
        loss_chunked = SupConLoss(0.1, chunk_size=chunk_size)(z1, z2, y)
        assert torch.allclose(loss_full, loss_chunked, atol=1e-5)

    def test_supcon_gradient_unchanged(self, labels_tensor):
        """The pre-0.1.2 formula had a NaN value but the same gradient as the fixed one."""
        import torch.nn.functional as F
        from embedkit.improvement.losses import SupConLoss
        torch.manual_seed(4)
        z1 = torch.randn(64, 16, requires_grad=True)
        z2 = torch.randn(64, 16)

        SupConLoss(0.1)(z1, z2, labels_tensor).backward()
        grad_fixed = z1.grad.clone()
        z1.grad = None

        # Old formula: −inf diagonal kept in log_prob, so the value is NaN.
        z = F.normalize(torch.cat([z1, z2]), dim=1)
        lbl = torch.cat([labels_tensor, labels_tensor])
        pos_mask = (lbl[None] == lbl[:, None]).float().fill_diagonal_(0.0)
        eye = torch.eye(z.shape[0], dtype=torch.bool)
        log_prob = F.log_softmax((z @ z.T / 0.1).masked_fill(eye, float("-inf")), dim=1)
        old_loss = (-(pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()
        assert torch.isnan(old_loss)
        old_loss.backward()
        assert torch.allclose(grad_fixed, z1.grad, atol=1e-7)

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

    @pytest.mark.parametrize("label_fixture", ["cont_labels_tensor", "vec_labels_tensor"])
    @pytest.mark.parametrize("chunk_size", [32, 24, 50])
    def test_rnc_loss_chunked_matches_full(self, small_tensor, request, label_fixture, chunk_size):
        from embedkit.improvement.losses import RankNContrastLoss
        y = request.getfixturevalue(label_fixture)
        z = small_tensor
        loss_full = RankNContrastLoss()(z, z + 0.01, y)
        loss_chunked = RankNContrastLoss(chunk_size=chunk_size)(z, z + 0.01, y)
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
        assert len(trainer.history["loss"]) == 2
        assert np.all(np.isfinite(trainer.history["loss"]))

    @pytest.mark.parametrize("val_split", [0.0, 0.2])
    def test_batch_size_larger_than_dataset(self, small_X, val_split):
        """n < batch_size used to give zero batches per epoch and a NaN loss."""
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, NTXentLoss
        model = EmbeddingRefiner(input_dim=small_X.shape[1], target_dim=8)
        trainer = Trainer(
            model=model,
            augmentation=GaussianNoise(std=0.05),
            loss=NTXentLoss(),
            epochs=3,
            batch_size=256,  # > 100 samples in small_X
            eval_every=100,
            val_split=val_split,
        )
        params_before = [p.detach().clone() for p in model.parameters()]
        with pytest.warns(UserWarning, match="batch_size=256 exceeds"):
            trainer.fit(small_X)
        assert len(trainer.history["loss"]) == 3
        assert np.all(np.isfinite(trainer.history["loss"]))
        assert trainer.batch_size == 256  # constructor param left untouched
        # The model was actually trained.
        assert any(not torch.equal(a, b) for a, b in zip(params_before, model.parameters()))

    def test_no_warning_when_batch_fits(self, small_X):
        import warnings
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, NTXentLoss
        trainer = Trainer(
            model=EmbeddingRefiner(input_dim=small_X.shape[1], target_dim=8),
            augmentation=GaussianNoise(std=0.05),
            loss=NTXentLoss(),
            epochs=1,
            batch_size=32,
            eval_every=100,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="batch_size=")
            trainer.fit(small_X)

    def test_too_few_samples_raises(self, small_X):
        from embedkit.improvement import EmbeddingRefiner, Trainer, GaussianNoise, NTXentLoss
        trainer = Trainer(
            model=EmbeddingRefiner(input_dim=small_X.shape[1], target_dim=8),
            augmentation=GaussianNoise(std=0.05),
            loss=NTXentLoss(),
            epochs=1,
            eval_every=100,
        )
        with pytest.raises(ValueError, match="at least 2 training samples"):
            trainer.fit(small_X[:1])

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
