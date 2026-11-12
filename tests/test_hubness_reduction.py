"""WP 5.5: hubness reduction as neighbour search."""

import numpy as np
import pytest
from scipy.stats import norm

from embedkit import hubness_reduction as H
from embedkit.evaluation import evaluate_transform
from embedkit.evaluation.metrics import k_skewness, knn_accuracy, recall_at_k


def _data(n_ref=40, n_q=15, d=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_ref, d)), rng.normal(size=(n_q, d))


def _base(A, B, metric):
    if metric == "cosine":
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
        return 1 - A @ B.T, A, B
    return np.linalg.norm(A[:, None] - B[None], axis=2), A, B


def _knn_rows(D, k, exclude=None):
    D = D.copy()
    if exclude is not None:
        D[np.arange(len(D)), exclude] = np.inf
    return np.sort(D, axis=1)[:, :k], np.argsort(D, axis=1)[:, :k]


def _reference_secondary(name, R, Q, metric, k):
    """Direct formulas on the full matrices; Q=None means the reference set itself."""
    self_q = Q is None
    D_rr, Rp, _ = _base(R, R, metric)
    excl = np.arange(len(R))
    D_qr, Qp, _ = _base(R if self_q else Q, R, metric)
    q_excl = excl if self_q else None
    if name == "local_scaling":
        s_r = _knn_rows(D_rr, k, excl)[0][:, -1]
        s_q = _knn_rows(D_qr, k, q_excl)[0][:, -1]
        S = 1 - np.exp(-D_qr**2 / (s_q[:, None] * s_r[None]))
    elif name == "mutual_proximity":
        off = ~np.eye(len(R), dtype=bool)
        mu_r = np.array([D_rr[i, off[i]].mean() for i in range(len(R))])
        sd_r = np.array([D_rr[i, off[i]].std() for i in range(len(R))])
        mu_q, sd_q = (mu_r, sd_r) if self_q else (D_qr.mean(1), D_qr.std(1))
        S = 1 - norm.sf(D_qr, mu_q[:, None], sd_q[:, None]) * norm.sf(D_qr, mu_r, sd_r)
    elif name == "csls":
        r_r = _knn_rows(D_rr, k, excl)[0].mean(1)
        r_q = _knn_rows(D_qr, k, q_excl)[0].mean(1)
        S = 2 * D_qr - r_q[:, None] - r_r[None]
    else:  # dissimlocal
        c_r = Rp[_knn_rows(D_rr, k, excl)[1]].mean(1)
        c_q = Rp[_knn_rows(D_qr, k, q_excl)[1]].mean(1)
        sq = ((Qp[:, None] - Rp[None]) ** 2).sum(2)
        S = sq - ((Qp - c_q) ** 2).sum(1)[:, None] - ((Rp - c_r) ** 2).sum(1)[None]
    if self_q:
        S[excl, excl] = np.inf
    return S


def _make(name, metric, k):
    cls = H.HUBNESS_REDUCTIONS[name]
    if name == "mutual_proximity":
        return cls(metric=metric, sample_size=None)
    return cls(k=k, metric=metric)


@pytest.mark.parametrize("name", sorted(H.HUBNESS_REDUCTIONS))
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("self_query", [False, True])
def test_matches_direct_formula(name, metric, self_query):
    R, Q = _data()
    S = _reference_secondary(name, R, None if self_query else Q, metric, k=4)
    r = _make(name, metric, 4).fit(R)
    dist, idx = r.kneighbors(None if self_query else Q, n_neighbors=7)
    np.testing.assert_array_equal(idx, np.argsort(S, axis=1, kind="stable")[:, :7])
    np.testing.assert_allclose(dist, np.sort(S, axis=1)[:, :7], rtol=1e-6, atol=1e-9)
    if self_query:
        assert not (idx == np.arange(len(R))[:, None]).any()


@pytest.mark.parametrize("name", sorted(H.HUBNESS_REDUCTIONS))
def test_chunking_does_not_change_result(name, monkeypatch):
    R, Q = _data(n_ref=60, n_q=25)
    full = _make(name, "euclidean", 5).fit(R)
    want = full.kneighbors(Q, 6), full.kneighbors(None, 6)
    monkeypatch.setattr(H, "_CHUNK_ENTRIES", 70)  # one row per chunk
    small = _make(name, "euclidean", 5).fit(R)
    got = small.kneighbors(Q, 6), small.kneighbors(None, 6)
    for (dw, iw), (dg, ig) in zip(want, got):
        np.testing.assert_array_equal(ig, iw)
        np.testing.assert_allclose(dg, dw, rtol=1e-10)


def test_csls_cosine_is_the_original_score():
    R, Q = _data()
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    C_rr, C_qr = Rn @ Rn.T, Qn @ Rn.T
    np.fill_diagonal(C_rr, -np.inf)
    r_r = -np.sort(-C_rr, axis=1)[:, :10].mean(1)
    r_q = -np.sort(-C_qr, axis=1)[:, :10].mean(1)
    score = 2 * C_qr - r_q[:, None] - r_r[None]
    dist, idx = H.CSLS(k=10).fit(R).kneighbors(Q, 5)
    np.testing.assert_array_equal(idx, np.argsort(-score, axis=1)[:, :5])
    np.testing.assert_allclose(dist, -np.sort(score, axis=1)[:, ::-1][:, :5], atol=1e-9)


def test_mutual_proximity_sample():
    R, Q = _data(n_ref=300)
    mp = H.MutualProximity(sample_size=50, random_state=3).fit(R)
    assert len(mp.sample_) == 50 and np.all(np.diff(mp.sample_) > 0)
    D, _, _ = _base(R, R[mp.sample_], "euclidean")
    i_in, i_out = mp.sample_[0], np.setdiff1d(np.arange(300), mp.sample_)[0]
    assert mp.mean_[i_out] == pytest.approx(D[i_out].mean())
    assert mp.mean_[i_in] == pytest.approx(np.delete(D[i_in], 0).mean())  # self left out
    same = H.MutualProximity(sample_size=50, random_state=3).fit(R).kneighbors(Q, 5)
    np.testing.assert_array_equal(same[1], mp.kneighbors(Q, 5)[1])
    # Far pairs do not saturate: the ranking key stays finite and ordered.
    dist, _ = mp.kneighbors(Q, 299)
    assert np.all(np.diff(dist, axis=1) >= 0)


def test_duplicates():
    R, _ = _data()
    R2 = np.vstack([R, R[:1]])  # row 40 duplicates row 0
    _, idx = H.LocalScaling(k=3).fit(R2).kneighbors(None, 3)
    assert idx[0, 0] == 40 and idx[40, 0] == 0
    _, idx = H.CSLS(k=3).fit(R2).kneighbors(R2[:1], 2)  # a query is a new point
    assert set(idx[0]) == {0, 40}


@pytest.mark.parametrize("name", sorted(H.HUBNESS_REDUCTIONS))
def test_reduces_hubness(name):
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(1000, 100))  # high-dimensional i.i.d. data are hubby
    base = k_skewness(Z, k=10, metric="euclidean")
    reduced = k_skewness(Z, k=10, metric="euclidean", hubness_reduction=_make(name, "cosine", 10))
    assert base > 1.0
    assert reduced < 0.5 * base


class _Plain(H._HubnessReduction):
    """The base distance itself: the reduced code paths must reproduce the plain ones."""

    def _fit_stats(self, R):
        pass

    def _secondary(self, D, Q, rows, self_query):
        return D.copy()


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_metric_hooks_reproduce_plain_search(metric):
    rng = np.random.default_rng(1)
    Ztr, Zte = rng.normal(size=(120, 8)), rng.normal(size=(50, 8))
    ytr, yte = rng.integers(0, 4, 120), rng.integers(0, 4, 50)
    plain = _Plain()
    a = knn_accuracy(Ztr, ytr, Zte, yte, k=5, metric=metric, return_per_point=True)
    b = knn_accuracy(Ztr, ytr, Zte, yte, k=5, metric=metric, return_per_point=True,
                     hubness_reduction=plain)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    for exclude_self, Q in [(False, Zte), (True, Ztr)]:
        yq = yte if Q is Zte else ytr
        for rel in [(yq, ytr), yq[:, None] == ytr[None]]:
            a = recall_at_k(Q, Ztr, rel, k=3, metric=metric, exclude_self=exclude_self,
                            return_per_point=True)
            b = recall_at_k(Q, Ztr, rel, k=3, metric=metric, exclude_self=exclude_self,
                            return_per_point=True, hubness_reduction=plain)
            assert a[0] == pytest.approx(b[0])
            np.testing.assert_array_equal(a[1], b[1])
    assert k_skewness(Ztr, 10, metric) == pytest.approx(
        k_skewness(Ztr, 10, metric, hubness_reduction=plain))


def test_knn_vote_ties_go_to_smallest_class():
    class _Fixed(_Plain):
        def _secondary(self, D, Q, rows, self_query):
            return np.tile(np.arange(D.shape[1], dtype=float), (len(D), 1))

    Ztr = np.random.default_rng(0).normal(size=(4, 3))
    ytr = np.array([2, 1, 1, 2])  # neighbours 0, 1 under _Fixed: one vote each
    acc = knn_accuracy(Ztr, ytr, Ztr[:1], np.array([1]), k=2, hubness_reduction=_Fixed())
    assert acc == 1.0


def test_evaluate_transform_hook():
    from embedkit.baselines import Identity

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 30)) * 2
    ytr, yte = rng.integers(0, 3, 300), rng.integers(0, 3, 100)
    Xtr = centers[ytr] + rng.normal(size=(300, 30))
    Xte = centers[yte] + rng.normal(size=(100, 30))
    metrics = ["knn_accuracy", "recall_at_10", "k_skewness", "linear_probe"]
    plain = evaluate_transform(lambda s: Identity(), Xtr, Xte, ytr, yte, seeds=(0,),
                               metrics=metrics).set_index("metric")
    mp = evaluate_transform(lambda s: Identity(), Xtr, Xte, ytr, yte, seeds=(0,),
                            metrics=metrics,
                            hubness_reduction=H.MutualProximity()).set_index("metric")
    assert mp.loc["linear_probe", "value"] == plain.loc["linear_probe", "value"]
    assert mp.loc["k_skewness", "value"] != plain.loc["k_skewness", "value"]
    assert mp.loc["knn_accuracy", "params"]["hubness_reduction"].startswith("MutualProximity(")
    assert "hubness_reduction" not in plain.loc["knn_accuracy", "params"]


def test_clone_and_params():
    r = H.LocalScaling(k=7, metric="cosine")
    c = r.clone(metric="euclidean")
    assert type(c) is H.LocalScaling and c.get_params() == {"k": 7, "metric": "euclidean"}
    assert repr(r) == "LocalScaling(k=7, metric='cosine')"
    assert not hasattr(c, "reference_")


def test_validation():
    R, Q = _data()
    with pytest.raises(ValueError, match="metric"):
        H.CSLS(metric="manhattan")
    with pytest.raises(ValueError, match="k must be"):
        H.LocalScaling(k=40).fit(R)
    with pytest.raises(ValueError, match="sample_size"):
        H.MutualProximity(sample_size=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        H.CSLS().kneighbors(Q)
    r = H.DisSimLocal().fit(R)
    with pytest.raises(ValueError, match="n_neighbors"):
        r.kneighbors(None, 40)
    assert r.kneighbors(Q, 40)[1].shape == (15, 40)  # a query may use every reference row
    with pytest.raises(ValueError, match="features"):
        r.kneighbors(Q[:, :3])
    with pytest.raises(ValueError, match="2 reference rows"):
        H.CSLS(k=1).fit(R[:1])


def test_tensor_input():
    torch = pytest.importorskip("torch")
    R, Q = _data()
    a = H.CSLS(k=4).fit(R).kneighbors(Q, 3)
    b = H.CSLS(k=4).fit(torch.from_numpy(R)).kneighbors(torch.from_numpy(Q), 3)
    np.testing.assert_array_equal(a[1], b[1])
