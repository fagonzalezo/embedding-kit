"""WP 5.2: positives and negatives that use the input-space neighbourhoods."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from embedkit.improvement.augmentation import IdentityPair, KNNPairs
from embedkit.improvement.losses import AlignUniformLoss, CombinedLoss, NTXentLoss
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer
from embedkit.utils.neighbors import knn


def _reference_ntxent(z_i, z_j, t, mask=None, decoupled=False):
    """Direct per-row formula over the 2N x 2N similarity matrix."""
    n = len(z_i)
    z = F.normalize(torch.cat([z_i, z_j]), dim=1).double()
    s = (z @ z.T / t).numpy()
    total = 0.0
    for r in range(2 * n):
        p = r + n if r < n else r - n
        neg = [c for c in range(2 * n) if c not in (r, p) and not (mask is not None and mask[r, c])]
        den = [s[r, c] for c in neg] + ([] if decoupled else [s[r, p]])
        total += np.log(np.exp(den).sum()) - s[r, p]
    return total / (2 * n)


def _views(n=12, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)


# ---------------------------------------------------------------------------------- KNNPairs
def test_mutual_pairs_are_mutual():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 5)), rng.normal(0, 0.05, (5, 5))]).astype(np.float32)
    aug = KNNPairs(k=5, mode="mutual")
    aug.precompute(X)
    _, nbr = knn(X, 5)
    aug.set_rng(None, torch.Generator().manual_seed(0))
    idx = torch.arange(len(X))
    for _ in range(5):
        _, x_j = aug(torch.from_numpy(X), idx)
        j = np.array([np.flatnonzero((X == row).all(axis=1))[0] for row in x_j.numpy()])
        for i, jj in enumerate(j):
            mutual = [c for c in nbr[i] if i in nbr[c]]
            assert jj in (mutual or [nbr[i, 0]])


def test_mutual_fallback_is_nearest_neighbour():
    nbr = np.array([[1, 2], [2, 3], [3, 1], [1, 2]])  # 0 has no mutual neighbour
    allowed = KNNPairs._allowed_mask(nbr)
    np.testing.assert_array_equal(allowed[0], [True, False])
    np.testing.assert_array_equal(allowed[1], [True, True])  # 1<->2, 1<->3 mutual


def test_mutual_batch_path():
    x = torch.randn(20, 4)
    aug = KNNPairs(k=3, mode="mutual")
    a, b = aug(x)
    assert a.shape == b.shape == x.shape


def test_knn_mode_validated():
    with pytest.raises(ValueError, match="mode"):
        KNNPairs(mode="shared")


# ------------------------------------------------------------------------------------ NT-Xent
@pytest.mark.parametrize("chunk", [None, 5])
def test_plain_ntxent_matches_formula(chunk):
    z_i, z_j = _views()
    got = NTXentLoss(temperature=0.2, chunk_size=chunk)(z_i, z_j)
    assert got.item() == pytest.approx(_reference_ntxent(z_i, z_j, 0.2), rel=1e-5)


@pytest.mark.parametrize("chunk", [None, 7])
@pytest.mark.parametrize("decoupled", [False, True])
def test_neighbor_mask_matches_formula(chunk, decoupled):
    n = 12
    z_i, z_j = _views(n)
    rng = np.random.default_rng(1)
    nbr = np.stack([rng.choice(np.delete(np.arange(40), i), 3, replace=False) for i in range(40)])
    indices = torch.from_numpy(rng.choice(40, n, replace=False))
    loss = NTXentLoss(temperature=0.3, chunk_size=chunk, neighbor_mask=nbr, decoupled=decoupled)
    got = loss(z_i, z_j, indices=indices)
    idx = indices.numpy()
    M = np.array([[a != b and (idx[b] in nbr[idx[a]] or idx[a] in nbr[idx[b]])
                   for b in range(n)] for a in range(n)])
    assert M.any()
    mask = np.block([[M, M], [M, M]])
    expected = _reference_ntxent(z_i, z_j, 0.3, mask, decoupled)
    assert got.item() == pytest.approx(expected, rel=1e-5)


def test_decoupled_matches_formula():
    z_i, z_j = _views()
    got = NTXentLoss(temperature=0.5, decoupled=True)(z_i, z_j)
    assert got.item() == pytest.approx(_reference_ntxent(z_i, z_j, 0.5, decoupled=True), rel=1e-5)


def test_all_negatives_masked():
    n = 4
    z_i, z_j = _views(n)
    nbr = np.array([[j for j in range(n) if j != i] for i in range(n)])
    idx = torch.arange(n)
    assert NTXentLoss(neighbor_mask=nbr)(z_i, z_j, indices=idx).item() == pytest.approx(0.0)
    # Decoupled keeps the negatives of a row that would have none, so the loss stays finite.
    dcl = NTXentLoss(neighbor_mask=nbr, decoupled=True)(z_i, z_j, indices=idx)
    assert dcl.item() == pytest.approx(NTXentLoss(decoupled=True)(z_i, z_j).item(), rel=1e-6)


def test_mask_needs_indices():
    z_i, z_j = _views()
    loss = NTXentLoss(neighbor_mask=np.zeros((12, 2), dtype=int))
    assert loss(z_i, z_j).item() == pytest.approx(NTXentLoss()(z_i, z_j).item(), rel=1e-6)
    assert loss.uses_indices and not NTXentLoss().uses_indices


def test_neighbor_mask_precompute_and_gradient():
    X = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    loss = NTXentLoss(neighbor_mask=4)
    loss.precompute(X)
    np.testing.assert_array_equal(loss.neighbor_index, knn(X, 4)[1])
    z_i = torch.randn(16, 8, requires_grad=True)
    out = loss(z_i, torch.randn(16, 8), indices=torch.arange(16))
    out.backward()
    assert torch.isfinite(z_i.grad).all()


def test_combined_forwards_indices_and_precompute():
    X = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    nt = NTXentLoss(neighbor_mask=4)
    combo = CombinedLoss([(nt, 1.0), (AlignUniformLoss(), 0.5)])
    assert combo.uses_indices
    combo.precompute(X)
    assert nt.neighbor_index is not None
    z_i, z_j = _views(10, 8)
    idx = torch.arange(10)
    expected = nt(z_i, z_j, indices=idx) + 0.5 * AlignUniformLoss()(z_i, z_j)
    assert combo(z_i, z_j, indices=idx).item() == pytest.approx(expected.item(), rel=1e-6)


# ------------------------------------------------------------------------------- integration
@pytest.mark.parametrize("aug", [KNNPairs(k=5, mode="mutual"), IdentityPair()])
def test_trainer_with_wp52_options(aug):
    X = np.random.default_rng(0).normal(size=(200, 16)).astype(np.float32)
    loss = CombinedLoss([(NTXentLoss(neighbor_mask=5, decoupled=True), 1.0),
                         (AlignUniformLoss(), 0.5)])
    model = EmbeddingRefiner(16, 8, architecture="residual")
    t = Trainer(model, aug, loss, epochs=3, batch_size=64, eval_every=100, device="cpu",
                random_state=0)
    t.fit(X)
    assert np.isfinite(t.history["loss"]).all()
    assert loss.loss_fns[0].neighbor_index.shape == (200, 5)


def test_identity_pair():
    x = torch.randn(5, 3)
    a, b = IdentityPair()(x)
    assert a is x and b is x
