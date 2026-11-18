"""Tests for the training-free transforms in ``embedkit.baselines`` (WP 2.2)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from embedkit.baselines import (
    AllButTheTop,
    CenterNormalize,
    ConceptorNegation,
    Identity,
    PCAReduce,
    RandomProjection,
    Standardize,
    ZCAWhiten,
)

N, D = 300, 20


def make_data(n=N, d=D, seed=0, offset=3.0):
    """Anisotropic, off-centre, well-conditioned Gaussian data."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) / np.sqrt(d) + np.diag(np.linspace(3.0, 0.5, d))
    mu = offset * rng.normal(size=d)
    return (rng.normal(size=(n, d)) @ A + mu).astype(np.float32)


@pytest.fixture(scope="module")
def X():
    return make_data(seed=0)


@pytest.fixture(scope="module")
def XB():
    return make_data(n=150, seed=1, offset=-2.0)


FACTORIES = {
    "identity": lambda: Identity(),
    "center_normalize": lambda: CenterNormalize(),
    "standardize": lambda: Standardize(),
    "pca": lambda: PCAReduce(5),
    "pca_whiten": lambda: PCAReduce(5, whiten=True),
    "zca": lambda: ZCAWhiten(),
    "abtt": lambda: AllButTheTop(2),
    "abtt_reduce": lambda: AllButTheTop(2, n_components=8),
    "conceptor": lambda: ConceptorNegation(),
    "random_projection": lambda: RandomProjection(6),
}
EXPECTED_OUT = {
    "identity": D, "center_normalize": D, "standardize": D, "pca": 5, "pca_whiten": 5,
    "zca": D, "abtt": D, "abtt_reduce": 8, "conceptor": D, "random_projection": 6,
}
# classes whose output depends on the training data (random projection and identity do not)
DATA_DEPENDENT = [k for k in FACTORIES if k not in ("identity", "random_projection")]


@pytest.fixture(params=list(FACTORIES))
def name(request):
    return request.param


# ----------------------------------------------------------------------------- generic contract


def test_shapes_dtype_and_n_features_out(name, X, XB):
    t = FACTORIES[name]()
    assert t.fit(X) is t
    assert t.n_features_out_ == EXPECTED_OUT[name]
    assert t.n_features_in_ == D
    for A in (X, XB):
        Z = t.transform(A)
        assert isinstance(Z, np.ndarray)
        assert Z.shape == (A.shape[0], EXPECTED_OUT[name])
        assert Z.dtype == np.float32
        assert np.isfinite(Z).all()


def test_fit_transform_matches_fit_then_transform(name, X):
    Z1 = FACTORIES[name]().fit_transform(X)
    Z2 = FACTORIES[name]().fit(X).transform(X)
    np.testing.assert_allclose(Z1, Z2, atol=0)


def test_y_is_accepted_and_ignored(name, X):
    y = np.arange(N) % 3
    Z1 = FACTORIES[name]().fit(X, y).transform(X)
    Z2 = FACTORIES[name]().fit(X).transform(X)
    np.testing.assert_allclose(Z1, Z2, atol=0)


def test_torch_input_round_trip(name, X, XB):
    Z_np = FACTORIES[name]().fit(X).transform(XB)
    t = FACTORIES[name]().fit(torch.from_numpy(X))
    Z_t = t.transform(torch.from_numpy(XB))
    assert isinstance(Z_t, np.ndarray)
    assert Z_t.dtype == np.float32
    np.testing.assert_allclose(Z_t, Z_np, atol=1e-6)
    # a float64 tensor with requires_grad is accepted too
    t2 = FACTORIES[name]().fit(torch.tensor(X, dtype=torch.float64, requires_grad=True))
    assert t2.transform(torch.from_numpy(XB)).shape == Z_np.shape


def test_transform_before_fit_raises(name, X):
    with pytest.raises(NotFittedError):
        FACTORIES[name]().transform(X)


def test_wrong_column_count_raises(name, X):
    t = FACTORIES[name]().fit(X)
    with pytest.raises(ValueError, match="features"):
        t.transform(X[:, :-1])
    with pytest.raises(ValueError, match="features"):
        t.transform(np.hstack([X, X[:, :1]]))


def test_constant_column_gives_no_nan(name):
    X = make_data(seed=3)
    X[:, 4] = 7.0
    t = FACTORIES[name]().fit(X)
    Z = t.transform(X)
    assert np.isfinite(Z).all()
    X_new = make_data(n=40, seed=4)
    X_new[:, 4] = 7.0
    assert np.isfinite(t.transform(X_new)).all()


def test_get_params_round_trip(name):
    t = FACTORIES[name]()
    params = t.get_params()
    assert isinstance(params, dict)
    clone = type(t)(**params)
    assert clone.get_params() == params


def test_get_params_contents():
    assert Identity().get_params() == {}
    assert PCAReduce(7, whiten=True, random_state=3).get_params() == {
        "n_components": 7, "whiten": True, "random_state": 3}
    assert ZCAWhiten(eps=1e-3).get_params() == {"eps": 1e-3}
    assert AllButTheTop(2, n_components=9).get_params() == {"n_remove": 2, "n_components": 9}
    assert ConceptorNegation(alpha=1.5).get_params() == {"alpha": 1.5}
    assert RandomProjection(4, random_state=9).get_params() == {
        "n_components": 4, "random_state": 9}


def test_only_statistics_are_stored(name, X):
    t = FACTORIES[name]().fit(X)
    for key, val in vars(t).items():
        if isinstance(val, np.ndarray):
            assert val.shape[0] != N, f"{key} looks like a copy of the training data"
    fitted = [k for k in vars(t) if k.endswith("_")]
    assert "n_features_in_" in fitted and "n_features_out_" in fitted


def test_transform_does_not_change_fitted_state(name, X, XB):
    t = FACTORIES[name]().fit(X)
    before = t.transform(X)
    t.transform(XB)
    np.testing.assert_array_equal(t.transform(X), before)


@pytest.mark.parametrize("name", DATA_DEPENDENT)
def test_transform_differs_from_refitting_on_new_data(name, X, XB):
    fitted_on_a = FACTORIES[name]().fit(X).transform(XB)
    fitted_on_b = FACTORIES[name]().fit(XB).transform(XB)
    assert not np.allclose(fitted_on_a, fitted_on_b, atol=1e-3)


def test_input_validation_errors():
    with pytest.raises(ValueError):
        Identity().fit(np.zeros(5))  # 1-D
    bad = np.ones((10, 3), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN|finite"):
        Standardize().fit(bad)
    with pytest.raises(ValueError):
        PCAReduce(0).fit(make_data())
    with pytest.raises(ValueError):
        RandomProjection(0).fit(make_data())
    with pytest.raises(ValueError):
        ZCAWhiten(eps=-1.0).fit(make_data())
    with pytest.raises(ValueError):
        ConceptorNegation(alpha=0.0).fit(make_data())
    with pytest.raises(ValueError):
        AllButTheTop(-1).fit(make_data())
    with pytest.raises(ValueError):
        AllButTheTop(1, n_components=0).fit(make_data())
    with pytest.raises(ValueError):
        PCAReduce(3).fit(make_data(n=1))


# ----------------------------------------------------------------------------- Identity


def test_identity_returns_input_values(X):
    Z = Identity().fit_transform(X)
    np.testing.assert_array_equal(Z, X)
    assert Z is not X


# ----------------------------------------------------------------------------- CenterNormalize


def test_center_normalize_uses_training_mean(X, XB):
    t = CenterNormalize().fit(X)
    mu = X.astype(np.float64).mean(0)
    np.testing.assert_allclose(t.mean_, mu, atol=1e-12)
    Z = t.transform(XB)
    C = XB.astype(np.float64) - mu
    expected = C / np.linalg.norm(C, axis=1, keepdims=True)
    np.testing.assert_allclose(Z, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-6)


def test_center_normalize_zero_row_stays_zero():
    X = np.array([[1, 2], [3, 4], [2, 3]], dtype=np.float32)  # mean is exactly (2, 3)
    Z = CenterNormalize().fit(X).transform(X)
    np.testing.assert_array_equal(Z[2], [0.0, 0.0])
    assert np.isfinite(Z).all()


# ----------------------------------------------------------------------------- Standardize


def test_standardize_uses_training_statistics(X, XB):
    t = Standardize().fit(X)
    X64 = X.astype(np.float64)
    np.testing.assert_allclose(t.mean_, X64.mean(0), atol=1e-12)
    np.testing.assert_allclose(t.scale_, X64.std(0), atol=1e-12)
    np.testing.assert_allclose(t.transform(XB), (XB - X64.mean(0)) / X64.std(0), atol=1e-5)
    Z = t.transform(X)
    np.testing.assert_allclose(Z.mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(Z.std(0), 1.0, atol=1e-5)


def test_standardize_constant_column_is_centred_not_scaled():
    X = make_data(seed=5)
    X[:, 2] = 7.0
    t = Standardize().fit(X)
    assert t.scale_[2] == 1.0
    np.testing.assert_array_equal(t.transform(X)[:, 2], 0.0)
    new = X[:10].copy()
    new[:, 2] = 9.0  # unseen value in a column that was constant in training
    np.testing.assert_allclose(t.transform(new)[:, 2], 2.0, atol=1e-6)


# ----------------------------------------------------------------------------- PCAReduce


def test_pca_matches_sklearn_up_to_sign(X, XB):
    ours = PCAReduce(6).fit(X)
    ref = PCA(6, random_state=0).fit(X)
    np.testing.assert_allclose(ours.mean_, ref.mean_, atol=1e-5)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-4)
    np.testing.assert_allclose(np.abs(ours.transform(XB)), np.abs(ref.transform(XB)), atol=1e-3)
    sign = np.sign(np.sum(ours.components_ * ref.components_, axis=1))
    np.testing.assert_allclose(ours.transform(XB), ref.transform(XB) * sign, atol=1e-3)


def test_pca_whiten_matches_sklearn_up_to_sign(X, XB):
    ours = PCAReduce(6, whiten=True).fit(X)
    ref = PCA(6, whiten=True, random_state=0).fit(X)
    np.testing.assert_allclose(np.abs(ours.transform(XB)), np.abs(ref.transform(XB)), atol=1e-3)


def test_pca_uses_training_statistics(X, XB):
    t = PCAReduce(5).fit(X)
    mu = X.astype(np.float64).mean(0)
    manual = (XB.astype(np.float64) - mu) @ t.components_.T
    np.testing.assert_allclose(t.transform(XB), manual, atol=1e-5)
    # the training components are orthonormal
    np.testing.assert_allclose(t.components_ @ t.components_.T, np.eye(5), atol=1e-10)


def test_pca_whitened_training_data_has_identity_covariance(X):
    Z = PCAReduce(10, whiten=True).fit_transform(X)
    np.testing.assert_allclose(np.cov(Z.astype(np.float64).T), np.eye(10), atol=1e-3)
    Zfull = PCAReduce(D, whiten=True).fit_transform(X)
    np.testing.assert_allclose(np.cov(Zfull.astype(np.float64).T), np.eye(D), atol=1e-3)


def test_pca_unwhitened_covariance_is_diagonal_with_eigenvalues(X):
    t = PCAReduce(5).fit(X)
    Z = t.transform(X).astype(np.float64)
    np.testing.assert_allclose(np.cov(Z.T), np.diag(t.explained_variance_), rtol=1e-3, atol=1e-3)


def test_pca_sign_convention_is_deterministic(X):
    a = PCAReduce(5).fit(X)
    b = PCAReduce(5, random_state=123).fit(X[::-1].copy())
    np.testing.assert_allclose(a.components_, b.components_, atol=1e-6)
    # largest-magnitude loading of each component is positive
    idx = np.abs(a.components_).argmax(axis=1)
    assert (a.components_[np.arange(5), idx] > 0).all()


def test_pca_is_capped_at_rank_with_warning_on_digits():
    Xd = load_digits().data.astype(np.float32)
    rank = int(np.linalg.matrix_rank(Xd.astype(np.float64) - Xd.astype(np.float64).mean(0)))
    assert rank < Xd.shape[1]  # digits has constant pixels
    for whiten in (False, True):
        t = PCAReduce(Xd.shape[1], whiten=whiten)
        with pytest.warns(UserWarning, match="rank"):
            Z = t.fit_transform(Xd)
        assert t.n_components_ == rank
        assert t.n_features_out_ == rank
        assert Z.shape == (Xd.shape[0], rank)
        assert np.isfinite(Z).all()
        assert t.components_.shape == (rank, Xd.shape[1])


def test_pca_whitened_digits_has_identity_covariance():
    Xd = load_digits().data.astype(np.float32)
    with pytest.warns(UserWarning):
        Z = PCAReduce(64, whiten=True).fit_transform(Xd)
    np.testing.assert_allclose(np.cov(Z.astype(np.float64).T), np.eye(Z.shape[1]), atol=1e-3)


def test_pca_no_warning_when_not_capped(X):
    t = PCAReduce(5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t.fit(X)
    assert t.n_components_ == 5


def test_pca_caps_at_number_of_samples():
    X = make_data(n=8, d=20, seed=6)
    t = PCAReduce(15)
    with pytest.warns(UserWarning, match="rank"):
        t.fit(X)
    assert t.n_components_ == 7  # rank of 8 centred points
    assert np.isfinite(t.transform(X)).all()


# ----------------------------------------------------------------------------- ZCAWhiten


def test_zca_training_data_has_identity_covariance(X):
    Z = ZCAWhiten().fit_transform(X)
    assert Z.shape == X.shape
    np.testing.assert_allclose(np.cov(Z.astype(np.float64).T), np.eye(D), atol=1e-3)
    np.testing.assert_allclose(Z.mean(0), 0.0, atol=1e-4)


def test_zca_matrix_is_symmetric_and_uses_training_statistics(X, XB):
    t = ZCAWhiten(eps=1e-5).fit(X)
    np.testing.assert_allclose(t.whitening_, t.whitening_.T, atol=1e-10)
    X64 = X.astype(np.float64)
    mu = X64.mean(0)
    lam, U = np.linalg.eigh(np.cov(X64.T))
    W = U @ np.diag(1.0 / np.sqrt(lam + 1e-5)) @ U.T
    np.testing.assert_allclose(t.transform(XB), (XB.astype(np.float64) - mu) @ W, atol=1e-4)


def test_zca_rank_deficient_data_is_finite():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(200, 5))
    X = (latent @ rng.normal(size=(5, 12))).astype(np.float32)  # rank 5 in 12 dimensions
    t = ZCAWhiten(eps=1e-5)
    Z = t.fit_transform(X)
    assert np.isfinite(Z).all()
    # training data is whitened on its 5-dimensional support and has no variance elsewhere
    ev = np.linalg.eigvalsh(np.cov(Z.astype(np.float64).T))
    np.testing.assert_allclose(np.sort(ev)[-5:], 1.0, atol=5e-3)
    assert np.sort(ev)[0] < 1e-3
    # new data stays finite
    assert np.isfinite(t.transform(rng.normal(size=(10, 12)).astype(np.float32))).all()


# ----------------------------------------------------------------------------- AllButTheTop


def _top_directions(X, k):
    X64 = X.astype(np.float64)
    _, _, Vt = np.linalg.svd(X64 - X64.mean(0), full_matrices=False)
    return Vt[:k]


def test_abtt_removes_top_directions_from_train_and_new_data(X, XB):
    t = AllButTheTop(3).fit(X)
    assert t.components_.shape == (3, D)
    for A in (X, XB):
        Z = t.transform(A).astype(np.float64)
        centred_norm = np.linalg.norm(A.astype(np.float64) - t.mean_)
        proj = Z @ t.components_.T
        assert np.linalg.norm(proj) / centred_norm < 1e-6


def test_abtt_components_are_top_principal_directions(X):
    t = AllButTheTop(3).fit(X)
    ref = _top_directions(X, 3)
    np.testing.assert_allclose(np.abs(t.components_), np.abs(ref), atol=1e-6)


def test_abtt_matches_manual_computation_with_training_statistics(X, XB):
    t = AllButTheTop(2).fit(X)
    U = _top_directions(X, 2)
    mu = X.astype(np.float64).mean(0)
    C = XB.astype(np.float64) - mu
    manual = C - (C @ U.T) @ U
    np.testing.assert_allclose(t.transform(XB), manual, atol=1e-5)


def test_abtt_zero_components_is_centring(X):
    t = AllButTheTop(0).fit(X)
    np.testing.assert_allclose(t.transform(X), X - X.astype(np.float64).mean(0), atol=1e-5)


def test_abtt_n_remove_is_capped_at_rank_with_warning():
    rng = np.random.default_rng(8)
    X = (rng.normal(size=(100, 3)) @ rng.normal(size=(3, 10))).astype(np.float32)
    t = AllButTheTop(6)
    with pytest.warns(UserWarning, match="rank"):
        Z = t.fit_transform(X)
    assert np.isfinite(Z).all()
    assert t.components_.shape[0] == 3


def _ppa(X, D_):
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[:D_]
    return X - (X @ U.T) @ U


def test_abtt_reduce_follows_raunak_algorithm_order(X, XB):
    """Raunak et al. (2019), Algorithm 2: PPA, then PCA to N dimensions, then PPA again."""
    n_remove, n_comp = 2, 8
    t = AllButTheTop(n_remove, n_components=n_comp).fit(X)
    # reference computed independently on the training data
    X64 = X.astype(np.float64)
    P1 = _ppa(X64, n_remove)
    _, _, Vt = np.linalg.svd(P1 - P1.mean(0), full_matrices=False)
    R = (P1 - P1.mean(0)) @ Vt[:n_comp].T
    expected_train = _ppa(R, n_remove)
    got = t.transform(X).astype(np.float64)
    assert got.shape == (N, n_comp)
    # components are defined up to sign, so compare the invariant Gram matrices
    np.testing.assert_allclose(got @ got.T, expected_train @ expected_train.T, atol=1e-3)
    # the final output has no mass on the removed directions of the last post-processing step
    for A in (X, XB):
        Z = t.transform(A).astype(np.float64)
        assert np.abs(Z @ t.post_components_.T).max() < 1e-4
    assert t.post_components_.shape == (n_remove, n_comp)
    assert t.components_.shape == (n_remove, D)


def test_abtt_reduce_uses_training_statistics(X, XB):
    t = AllButTheTop(2, n_components=8).fit(X)
    X64 = XB.astype(np.float64)
    c1 = X64 - t.mean_
    s1 = c1 - (c1 @ t.components_.T) @ t.components_
    s2 = (s1 - t.pca_mean_) @ t.pca_components_.T
    c3 = s2 - t.post_mean_
    s3 = c3 - (c3 @ t.post_components_.T) @ t.post_components_
    np.testing.assert_allclose(t.transform(XB), s3, atol=1e-5)


def test_abtt_reduce_caps_n_components_at_rank_with_warning():
    rng = np.random.default_rng(9)
    X = (rng.normal(size=(100, 4)) @ rng.normal(size=(4, 12))).astype(np.float32)
    t = AllButTheTop(1, n_components=10)
    with pytest.warns(UserWarning, match="rank"):
        Z = t.fit_transform(X)
    assert np.isfinite(Z).all()
    assert t.n_features_out_ == Z.shape[1] <= 3


# ----------------------------------------------------------------------------- ConceptorNegation


def _direct_conceptor(X, alpha):
    X64 = X.astype(np.float64)
    R = X64.T @ X64 / X64.shape[0]  # uncentred correlation matrix, as in Liu et al. (2019)
    C = R @ np.linalg.inv(R + alpha ** -2 * np.eye(R.shape[0]))
    return np.eye(R.shape[0]) - C


@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0])
def test_conceptor_matches_direct_formula(X, XB, alpha):
    t = ConceptorNegation(alpha=alpha).fit(X)
    M = _direct_conceptor(X, alpha)
    np.testing.assert_allclose(t.transform(XB), XB.astype(np.float64) @ M.T, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(t.transform(X), X.astype(np.float64) @ M.T, atol=1e-4, rtol=1e-4)


def test_conceptor_default_alpha_is_two():
    assert ConceptorNegation().alpha == 2.0


def test_conceptor_does_not_centre():
    """The paper's Algorithm 2 uses R = E[x x^T] of the raw vectors; no mean is subtracted."""
    rng = np.random.default_rng(10)
    X = (rng.normal(size=(200, 6)) + 5.0).astype(np.float32)
    t = ConceptorNegation(alpha=2.0).fit(X)
    assert not hasattr(t, "mean_")
    np.testing.assert_allclose(t.transform(X), X.astype(np.float64) @ _direct_conceptor(X, 2.0).T,
                               atol=1e-4, rtol=1e-4)


def test_conceptor_shrinks_dominant_direction_more_than_weak_one():
    rng = np.random.default_rng(11)
    scales = np.array([10.0, 3.0, 1.0, 0.3, 0.1])
    X = (rng.normal(size=(2000, 5)) * scales).astype(np.float32)
    t = ConceptorNegation(alpha=2.0).fit(X)
    Z = t.transform(X)
    ratio = Z.std(0) / X.std(0)  # axes are (nearly) the principal directions
    assert ratio[0] < ratio[2] < ratio[4]
    assert ratio[0] < 0.1
    assert ratio[4] > 0.9
    # the negated conceptor is a symmetric contraction with eigenvalues in (0, 1]
    ev = np.linalg.eigvalsh(t.negation_)
    assert ev.min() > 0 and ev.max() <= 1.0 + 1e-12


# ----------------------------------------------------------------------------- RandomProjection


def test_random_projection_is_deterministic_given_seed(X, XB):
    a = RandomProjection(6, random_state=5).fit(X).transform(XB)
    b = RandomProjection(6, random_state=5).fit(XB).transform(XB)  # data does not matter
    c = RandomProjection(6, random_state=6).fit(X).transform(XB)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_random_projection_matrix_scaling():
    t = RandomProjection(50, random_state=0).fit(np.zeros((5, 4000), dtype=np.float32) + 1)
    assert t.components_.shape == (50, 4000)
    assert abs(t.components_.std() - 1 / np.sqrt(50)) < 0.002
    assert abs(t.components_.mean()) < 0.002


def test_random_projection_preserves_pairwise_distances():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(100, 400)).astype(np.float32)
    Z = RandomProjection(200, random_state=0).fit_transform(X)
    iu = np.triu_indices(100, k=1)

    def pdist(A):
        A = A.astype(np.float64)
        sq = (A ** 2).sum(1)
        return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * A @ A.T, 0))[iu]

    ratio = pdist(Z) / pdist(X)
    assert abs(ratio.mean() - 1.0) < 0.02
    assert ratio.min() > 0.7 and ratio.max() < 1.3


def test_random_projection_can_increase_dimension(X):
    Z = RandomProjection(50, random_state=0).fit_transform(X)
    assert Z.shape == (N, 50)
