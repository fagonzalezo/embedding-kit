"""WP 5.3: Matryoshka output."""

import numpy as np
import pytest
import torch

from embedkit.improvement.losses import (
    AlignUniformLoss,
    MatryoshkaLoss,
    NTXentLoss,
    SupConLoss,
)


def _views(n=16, d=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)


def test_sum_over_prefixes():
    z_i, z_j = _views()
    base = NTXentLoss(temperature=0.2)
    loss = MatryoshkaLoss(base, dims=(8, 16, 32, 64))
    assert loss.prefixes(40) == [(8, 1.0), (16, 1.0), (32, 1.0), (40, 1.0)]
    expected = sum(base(z_i[:, :m], z_j[:, :m]) for m in (8, 16, 32, 40))
    assert loss(z_i, z_j).item() == pytest.approx(expected.item(), rel=1e-6)


def test_full_dimension_not_double_counted_and_weights():
    z_i, z_j = _views(d=32)
    base = AlignUniformLoss()
    loss = MatryoshkaLoss(base, dims=(16, 32), weights=(0.5, 2.0))
    assert loss.prefixes(32) == [(16, 0.5), (32, 2.0)]
    expected = 0.5 * base(z_i[:, :16], z_j[:, :16]) + 2.0 * base(z_i, z_j)
    assert loss(z_i, z_j).item() == pytest.approx(expected.item(), rel=1e-6)


def test_forwards_labels_and_indices():
    z_i, z_j = _views()
    y = torch.arange(16) % 4
    sup = MatryoshkaLoss(SupConLoss(), dims=(8,))
    assert sup.uses_labels
    expected = SupConLoss()(z_i[:, :8], z_j[:, :8], y) + SupConLoss()(z_i, z_j, y)
    assert sup(z_i, z_j, y).item() == pytest.approx(expected.item(), rel=1e-6)

    nbr = np.stack([np.roll(np.arange(16), -s)[:3] for s in range(1, 17)])
    nt = NTXentLoss(neighbor_mask=nbr)
    mat = MatryoshkaLoss(nt, dims=(8,))
    assert mat.uses_indices
    idx = torch.arange(16)
    expected = nt(z_i[:, :8], z_j[:, :8], indices=idx) + nt(z_i, z_j, indices=idx)
    assert mat(z_i, z_j, indices=idx).item() == pytest.approx(expected.item(), rel=1e-6)


@pytest.mark.parametrize("dims", [(), (0, 8), (16, 8), (8, 8)])
def test_dims_validated(dims):
    with pytest.raises(ValueError, match="dims"):
        MatryoshkaLoss(NTXentLoss(), dims=dims)


def test_embedkit_truncated_transform(small_X):
    from embedkit import EmbedKit

    ek = EmbedKit(method="refiner", target_dim=8, loss=MatryoshkaLoss(NTXentLoss(), dims=(2, 4)),
                  epochs=2,
                  eval_every=100, device="cpu", id_methods=["TwoNN"]).fit(small_X)
    Z = ek.transform(small_X)
    Z4 = ek.transform(small_X, dim=4)
    assert Z4.shape == (len(small_X), 4)
    np.testing.assert_allclose(np.linalg.norm(Z4, axis=1), 1.0, atol=1e-5)
    ref = Z[:, :4] / np.linalg.norm(Z[:, :4], axis=1, keepdims=True)
    np.testing.assert_allclose(Z4, ref, atol=1e-6)
    np.testing.assert_allclose(ek.transform(small_X, dim=8), Z, atol=1e-6)
    Zt = ek.transform(torch.from_numpy(small_X), dim=4)
    assert isinstance(Zt, torch.Tensor)
    np.testing.assert_allclose(Zt.numpy(), Z4, atol=1e-6)
    with pytest.raises(ValueError, match="dim"):
        ek.transform(small_X, dim=9)


def test_embedkit_nested_dims_auto(small_X):
    from embedkit import EmbedKit

    kw = dict(method="refiner", epochs=1, eval_every=100, device="cpu", id_methods=["TwoNN"])
    ek = EmbedKit(target_dim=200, nested_dims="auto", **kw)
    assert ek._resolve_nested_dims(200) == (32, 64, 128)
    assert ek._resolve_nested_dims(32) == (32,)
    ek = EmbedKit(target_dim=8, nested_dims=(2, 4), **kw).fit(small_X)
    loss = ek._trainer.loss
    assert isinstance(loss, MatryoshkaLoss) and loss.dims == (2, 4)
    assert ek.nested_dims_ == (2, 4) and ek._config["nested_dims"] == [2, 4]
    assert ek.transform(small_X, dim=2).shape == (len(small_X), 2)
    plain = EmbedKit(target_dim=8, **kw).fit(small_X)
    assert not isinstance(plain._trainer.loss, MatryoshkaLoss) and plain.nested_dims_ is None


def test_embedkit_nested_dims_validation(small_X):
    from embedkit import EmbedKit

    kw = dict(method="refiner", epochs=1, eval_every=100, device="cpu", id_methods=["TwoNN"])
    with pytest.raises(ValueError, match="exceed"):
        EmbedKit(target_dim=8, nested_dims=(4, 16), **kw).fit(small_X)
    with pytest.raises(ValueError, match="'auto'"):
        EmbedKit(target_dim=8, nested_dims="all", **kw).fit(small_X)
    with pytest.raises(ValueError, match="not both"):
        EmbedKit(target_dim=8, nested_dims=(4,), loss=MatryoshkaLoss(NTXentLoss(), dims=(2,)),
                 **kw).fit(small_X)
