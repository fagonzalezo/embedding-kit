"""Known-answer tests for embedkit.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from embedkit.evaluation import metrics as M


def blobs(n_per=40, n_classes=3, d=8, sep=20.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, d)) * sep
    X = np.concatenate([c + rng.normal(size=(n_per, d)) for c in centers])
    y = np.repeat(np.arange(n_classes), n_per)
    return X, y


def split(X, y, seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * frac)
    a, b = idx[:cut], idx[cut:]
    return X[a], y[a], X[b], y[b]


# ---------------------------------------------------------------- knn_accuracy
class TestKnnAccuracy:
    def test_separated_blobs(self):
        Xtr, ytr, Xte, yte = split(*blobs())
        assert M.knn_accuracy(Xtr, ytr, Xte, yte, k=5, metric="euclidean") == 1.0

    def test_independent_labels_near_chance(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(1200, 10))
        y = rng.integers(0, 4, size=1200)
        Xtr, ytr, Xte, yte = split(X, y)
        acc = M.knn_accuracy(Xtr, ytr, Xte, yte, k=5)
        assert 0.18 < acc < 0.34

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_matches_brute_force_loop(self, metric):
        rng = np.random.default_rng(2)
        Xtr = rng.normal(size=(60, 5))
        Xte = rng.normal(size=(25, 5))
        ytr = rng.integers(0, 3, size=60)
        yte = rng.integers(0, 3, size=25)
        k = 5
        hits = 0
        for x, t in zip(Xte, yte):
            if metric == "cosine":
                d = 1 - (Xtr @ x) / (np.linalg.norm(Xtr, axis=1) * np.linalg.norm(x))
            else:
                d = np.linalg.norm(Xtr - x, axis=1)
            nn = np.argsort(d)[:k]
            votes = np.bincount(ytr[nn], minlength=3)
            hits += int(np.argmax(votes) == t)  # ties go to the lowest class code
        ref = hits / len(yte)
        assert M.knn_accuracy(Xtr, ytr, Xte, yte, k=k, metric=metric) == pytest.approx(ref)

    def test_string_labels_and_neighbours_from_train_only(self):
        Xtr = np.array([[0.0, 0], [0.1, 0], [10, 10], [10.1, 10]])
        ytr = ["a", "a", "b", "b"]
        # The test point of class "b" sits exactly on a test duplicate of itself; only train counts
        Xte = np.array([[0.05, 0.0], [10.0, 10.05]])
        yte = ["a", "b"]
        assert M.knn_accuracy(Xtr, ytr, Xte, yte, k=2, metric="euclidean") == 1.0
        assert M.knn_accuracy(Xtr, ytr, Xte, ["b", "a"], k=2, metric="euclidean") == 0.0

    def test_unseen_test_label_counts_as_error(self):
        Xtr = np.array([[0.0, 0], [0.1, 0], [10, 10], [10.1, 10]])
        ytr = ["a", "a", "b", "b"]
        Xte = np.array([[0.0, 0.1], [10.0, 10.1]])
        assert M.knn_accuracy(Xtr, ytr, Xte, ["a", "zzz"], k=1, metric="euclidean") == 0.5

    def test_returns_float_and_accepts_torch(self):
        import torch

        Xtr, ytr, Xte, yte = split(*blobs())
        acc = M.knn_accuracy(torch.tensor(Xtr), ytr, torch.tensor(Xte), yte)
        assert isinstance(acc, float)

    def test_rejects_continuous_labels(self):
        Xtr, ytr, Xte, yte = split(*blobs())
        with pytest.raises(ValueError):
            M.knn_accuracy(Xtr, ytr.astype(float) + 0.5, Xte, yte.astype(float) + 0.5)


# ---------------------------------------------------------------- linear_probe
class TestLinearProbe:
    def test_separable_classes(self):
        Xtr, ytr, Xte, yte = split(*blobs(sep=10.0))
        assert M.linear_probe(Xtr, ytr, Xte, yte) > 0.98

    def test_random_labels_near_chance(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(1000, 10))
        y = rng.integers(0, 4, size=1000)
        Xtr, ytr, Xte, yte = split(X, y)
        assert M.linear_probe(Xtr, ytr, Xte, yte) < 0.36

    def test_linear_target_r2(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(600, 6))
        y = X @ rng.normal(size=6) + 0.01 * rng.normal(size=600)
        Xtr, ytr, Xte, yte = split(X, y)
        assert M.linear_probe(Xtr, ytr, Xte, yte) > 0.99

    def test_independent_target_r2_near_zero(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(600, 6))
        y = rng.normal(size=600)
        Xtr, ytr, Xte, yte = split(X, y)
        assert abs(M.linear_probe(Xtr, ytr, Xte, yte)) < 0.1

    def test_multi_output_regression_is_mean_r2(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(600, 6))
        W = rng.normal(size=(6, 2))
        Y = X @ W + 0.01 * rng.normal(size=(600, 2))
        Xtr, Ytr, Xte, Yte = split(X, Y)
        r2 = M.linear_probe(Xtr, Ytr, Xte, Yte, task="auto")
        assert r2 > 0.99

    def test_task_override_and_validation(self):
        Xtr, ytr, Xte, yte = split(*blobs(sep=10.0))
        assert M.linear_probe(Xtr, ytr, Xte, yte, task="classification") > 0.98
        with pytest.raises(ValueError):
            M.linear_probe(Xtr, ytr, Xte, yte, task="nope")
        with pytest.raises(ValueError):
            M.linear_probe(Xtr, ytr.astype(float) + 0.5, Xte, yte.astype(float) + 0.5,
                           task="classification")

    def test_string_labels_and_scale_invariance(self):
        Xtr, ytr, Xte, yte = split(*blobs(sep=10.0))
        names = np.array(["a", "b", "c"])
        a = M.linear_probe(Xtr, names[ytr], Xte, names[yte])
        b = M.linear_probe(Xtr * 1000.0, names[ytr], Xte * 1000.0, names[yte])
        assert a == pytest.approx(b, abs=0.02)


# ---------------------------------------------------------------- recall_at_k
class TestRecallAtK:
    def test_matches_loop_reference(self):
        rng = np.random.default_rng(7)
        Q = rng.normal(size=(30, 4))
        G = rng.normal(size=(50, 4))
        yq = rng.integers(0, 6, size=30)
        yg = rng.integers(0, 6, size=50)
        k = 5
        hits, counted = 0, 0
        for q, lab in zip(Q, yq):
            rel = yg == lab
            if not rel.any():
                continue
            counted += 1
            d = 1 - (G @ q) / (np.linalg.norm(G, axis=1) * np.linalg.norm(q))
            hits += int(rel[np.argsort(d)[:k]].any())
        got = M.recall_at_k(Q, G, (yq, yg), k=k)
        assert got == pytest.approx(hits / counted)
        # boolean relevance gives the same answer
        rel_mat = yq[:, None] == yg[None, :]
        assert M.recall_at_k(Q, G, rel_mat, k=k) == pytest.approx(hits / counted)

    def test_identical_copy_in_gallery_gives_one(self):
        rng = np.random.default_rng(8)
        G = rng.normal(size=(40, 6))
        Q = G[:15]
        rel = np.zeros((15, 40), dtype=bool)
        rel[np.arange(15), np.arange(15)] = True
        assert M.recall_at_k(Q, G, rel, k=1) == 1.0

    def test_queries_without_relevant_items_are_excluded(self):
        G = np.array([[1.0, 0], [0, 1.0]])
        Q = np.array([[1.0, 0.1], [0.1, 1.0], [1.0, 1.0]])
        yq = np.array([0, 1, 99])
        yg = np.array([0, 1])
        assert M.recall_at_k(Q, G, (yq, yg), k=1) == 1.0

    def test_no_relevant_anywhere_raises(self):
        G = np.eye(3)
        with pytest.raises(ValueError):
            M.recall_at_k(G, G, np.zeros((3, 3), dtype=bool), k=1)

    def test_exclude_self(self):
        rng = np.random.default_rng(9)
        Z = rng.normal(size=(40, 5))
        y = rng.integers(0, 3, size=40)
        # without exclusion every query finds itself at rank 1
        assert M.recall_at_k(Z, Z, (y, y), k=1) == 1.0
        k = 3
        hits, counted = 0, 0
        for i in range(40):
            rel = y == y[i]
            rel[i] = False
            if not rel.any():
                continue
            counted += 1
            d = 1 - (Z @ Z[i]) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(Z[i]))
            d[i] = np.inf
            hits += int(rel[np.argsort(d)[:k]].any())
        got = M.recall_at_k(Z, Z, (y, y), k=k, exclude_self=True)
        assert got == pytest.approx(hits / counted)

    def test_k_larger_than_gallery_and_bad_shapes(self):
        G = np.eye(3)
        rel = np.ones((3, 3), dtype=bool)
        assert M.recall_at_k(G, G, rel, k=10) == 1.0
        with pytest.raises(ValueError):
            M.recall_at_k(G, G, np.ones((2, 3), dtype=bool), k=1)


# ------------------------------------------------- trustworthiness / continuity
def _rotation(d, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q


class TestTrustworthinessContinuity:
    @pytest.mark.parametrize("fn", ["trustworthiness", "continuity"])
    def test_identity_and_rotation_are_one(self, fn):
        X = np.random.default_rng(10).normal(size=(200, 6))
        f = getattr(M, fn)
        assert f(X, X, k=10) == pytest.approx(1.0)
        assert f(X, X @ _rotation(6), k=10) == pytest.approx(1.0)

    @pytest.mark.parametrize("fn", ["trustworthiness", "continuity"])
    def test_shuffled_rows_are_worse(self, fn):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(200, 6))
        Z = X[rng.permutation(200)]
        value = getattr(M, fn)(X, Z, k=10)
        assert value < 0.65  # about 0.5 for unrelated spaces

    def test_matches_scikit_learn_and_continuity_swaps_roles(self):
        from sklearn.manifold import trustworthiness as sk_trust

        rng = np.random.default_rng(12)
        X = rng.normal(size=(150, 8))
        Z = X[:, :3] + 0.3 * rng.normal(size=(150, 3))
        assert M.trustworthiness(X, Z, k=7) == pytest.approx(sk_trust(X, Z, n_neighbors=7))
        assert M.continuity(X, Z, k=7) == pytest.approx(sk_trust(Z, X, n_neighbors=7))

    def test_subsampling_is_deterministic_and_aligned(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(300, 6))
        Z = X @ _rotation(6)
        a = M.trustworthiness(X, Z, k=5, max_samples=100, random_state=3)
        b = M.trustworthiness(X, Z, k=5, max_samples=100, random_state=3)
        assert a == b
        assert a == pytest.approx(1.0)  # rows of X and Z subsampled together

    def test_accepts_torch_and_returns_float(self):
        import torch

        X = np.random.default_rng(14).normal(size=(100, 4))
        v = M.trustworthiness(torch.tensor(X), torch.tensor(X), k=5)
        assert isinstance(v, float)

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError):
            M.trustworthiness(np.zeros((10, 2)), np.zeros((9, 2)))


# ----------------------------------------------------------------- knn_overlap
class TestKnnOverlap:
    def test_identity_is_one(self):
        X = np.random.default_rng(20).normal(size=(120, 5))
        assert M.knn_overlap(X, X, k=10, metric_x="euclidean", metric_z="euclidean") == 1.0
        assert M.knn_overlap(X, X, k=10, metric_x="cosine", metric_z="cosine") == 1.0

    def test_independent_spaces_near_k_over_n(self):
        rng = np.random.default_rng(21)
        n, k = 600, 10
        X, Z = rng.normal(size=(n, 6)), rng.normal(size=(n, 6))
        v = M.knn_overlap(X, Z, k=k, metric_x="euclidean", metric_z="euclidean")
        assert abs(v - k / (n - 1)) < 0.015

    def test_matches_loop_reference_with_mixed_metrics(self):
        rng = np.random.default_rng(22)
        X, Z = rng.normal(size=(60, 5)), rng.normal(size=(60, 3))
        k = 4
        total = 0.0
        for i in range(60):
            dx = np.linalg.norm(X - X[i], axis=1)
            dx[i] = np.inf
            dz = 1 - (Z @ Z[i]) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(Z[i]))
            dz[i] = np.inf
            total += len(set(np.argsort(dx)[:k]) & set(np.argsort(dz)[:k])) / k
        got = M.knn_overlap(X, Z, k=k, metric_x="euclidean", metric_z="cosine")
        assert got == pytest.approx(total / 60)

    def test_subsample_is_deterministic(self):
        X = np.random.default_rng(23).normal(size=(300, 5))
        a = M.knn_overlap(X, X, k=5, max_samples=80, random_state=1,
                          metric_x="euclidean", metric_z="euclidean")
        assert a == 1.0

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError):
            M.knn_overlap(np.zeros((10, 2)), np.zeros((9, 2)), k=2)


# ------------------------------------------------------------------ linear_cka
def _cka_reference(X, Z):
    X = X - X.mean(0)
    Z = Z - Z.mean(0)
    num = np.linalg.norm(X.T @ Z, "fro") ** 2
    return num / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Z.T @ Z, "fro"))


class TestLinearCKA:
    def test_rotation_and_scaling_are_one(self):
        X = np.random.default_rng(30).normal(size=(300, 7)) @ np.diag(np.arange(1, 8.0))
        assert M.linear_cka(X, X @ _rotation(7)) == pytest.approx(1.0)
        assert M.linear_cka(X, 3.0 * X) == pytest.approx(1.0)
        assert M.linear_cka(X, X) == pytest.approx(1.0)

    def test_independent_high_n_is_near_zero(self):
        rng = np.random.default_rng(31)
        assert M.linear_cka(rng.normal(size=(5000, 5)), rng.normal(size=(5000, 5))) < 0.01

    def test_invariant_to_isotropic_scaling_of_one_argument(self):
        rng = np.random.default_rng(32)
        X, Z = rng.normal(size=(100, 6)), rng.normal(size=(100, 4))
        assert M.linear_cka(X, 7.5 * Z) == pytest.approx(M.linear_cka(X, Z))

    def test_matches_direct_formula_both_shapes(self):
        rng = np.random.default_rng(33)
        # n > d (feature-space form is cheaper) and d > n (Gram form is cheaper)
        for n, p, q in [(200, 6, 4), (30, 100, 80), (40, 50, 5)]:
            X = rng.normal(size=(n, p))
            Z = X[:, : min(p, 3)] @ rng.normal(size=(min(p, 3), q)) + 0.5 * rng.normal(size=(n, q))
            assert M.linear_cka(X, Z) == pytest.approx(_cka_reference(X, Z), rel=1e-6)

    def test_centring_matters(self):
        rng = np.random.default_rng(34)
        X = rng.normal(size=(200, 4))
        Z = rng.normal(size=(200, 4)) + 100.0  # a huge shift must not change the result
        assert M.linear_cka(X, Z) == pytest.approx(M.linear_cka(X, Z - 100.0))

    def test_constant_input_gives_nan(self):
        X = np.random.default_rng(35).normal(size=(50, 3))
        assert np.isnan(M.linear_cka(X, np.ones((50, 3))))

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError):
            M.linear_cka(np.zeros((10, 2)), np.zeros((9, 2)))


# -------------------------------------------------------------- effective_rank
def _orthonormal_factors(n, d, r, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, r)))
    V, _ = np.linalg.qr(rng.normal(size=(d, r)))
    return U, V


class TestEffectiveRank:
    @pytest.mark.parametrize("r", [1, 2, 5, 9])
    def test_equal_singular_values_give_r(self, r):
        U, V = _orthonormal_factors(80, 30, r)
        Z = 4.0 * U @ V.T
        assert M.effective_rank(Z) == pytest.approx(float(r))

    def test_known_unequal_spectrum(self):
        U, V = _orthonormal_factors(50, 20, 3)
        s = np.array([1.0, 1.0, 2.0])
        Z = (U * s) @ V.T
        p = s / s.sum()
        assert M.effective_rank(Z) == pytest.approx(float(np.exp(-(p * np.log(p)).sum())))

    def test_center_option(self):
        U, V = _orthonormal_factors(60, 12, 4, seed=1)
        # columns of U are not centred; build exactly centred factors
        Uc = U - U.mean(0)
        Uc, _ = np.linalg.qr(Uc)
        Zc = 2.0 * Uc @ V.T
        offset = 50.0 * np.ones((60, 1)) @ np.ones((1, 12))
        Z = Zc + offset
        assert M.effective_rank(Zc, center=False) == pytest.approx(4.0)
        assert M.effective_rank(Z, center=True) == pytest.approx(4.0)
        assert M.effective_rank(Z, center=False) < 4.0  # the dominant offset direction

    def test_rank_one_and_all_zero(self):
        v = np.random.default_rng(40).normal(size=(1, 9))
        Z = np.arange(1, 21, dtype=float)[:, None] @ v
        assert M.effective_rank(Z) == pytest.approx(1.0)
        assert np.isnan(M.effective_rank(np.zeros((10, 4))))
        assert np.isnan(M.effective_rank(np.ones((10, 4)), center=True))

    def test_accepts_torch_and_returns_float(self):
        import torch

        assert isinstance(M.effective_rank(torch.randn(20, 5)), float)


# ------------------------------------------------------------------ k_skewness
class TestKSkewness:
    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_equals_hubness_analyzer(self, metric):
        from embedkit.analysis.hubness import HubnessAnalyzer

        Z = np.random.default_rng(50).normal(size=(300, 40)).astype(np.float32)
        ref = HubnessAnalyzer(k=7, metric=metric).fit(Z).k_skewness
        assert M.k_skewness(Z, k=7, metric=metric) == pytest.approx(ref)

    def test_shared_helpers(self):
        from embedkit.analysis import hubness

        idx = np.array([[1, 2], [0, 2], [0, 1], [0, 2]])
        N = hubness.k_occurrence(idx, 4)
        assert N.tolist() == [3, 2, 3, 0]
        assert hubness.k_occurrence_skewness(N) == pytest.approx(
            float(np.mean((N - N.mean()) ** 3) / N.std() ** 3), rel=1e-6
        )

    def test_hubby_data_has_larger_skew_than_uniform_grid(self):
        rng = np.random.default_rng(51)
        high_d = rng.normal(size=(500, 200))
        ring = np.stack([np.cos(np.linspace(0, 6.28, 200, endpoint=False)),
                         np.sin(np.linspace(0, 6.28, 200, endpoint=False))], axis=1)
        assert M.k_skewness(high_d, k=10) > M.k_skewness(ring, k=10) + 0.2
        assert M.k_skewness(ring, k=10) == pytest.approx(0.0, abs=1e-3)


# ------------------------------------------------------------------------ nc1
def _nc1_reference(Z, y):
    classes = np.unique(y)
    C = len(classes)
    means = np.stack([Z[y == c].mean(0) for c in classes])
    mu_g = means.mean(0)
    d = Z.shape[1]
    Sw = np.zeros((d, d))
    for c, m in zip(classes, means):
        for z in Z[y == c]:
            Sw += np.outer(z - m, z - m)
    Sw /= len(Z)
    Sb = sum(np.outer(m - mu_g, m - mu_g) for m in means) / C
    return float(np.trace(Sw @ np.linalg.pinv(Sb)) / C)


class TestNC1:
    def test_hand_computed_one_dimensional_case(self):
        # class means -3 and 3, within-class values mean +- 1: Sw = 1, Sb = 9, NC1 = 1 / 9 / 2
        Z = np.array([[-4.0], [-2.0], [2.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        assert M.nc1(Z, y) == pytest.approx(1.0 / 18.0)

    def test_matches_loop_reference(self):
        rng = np.random.default_rng(60)
        y = rng.integers(0, 4, size=120)
        Z = rng.normal(size=(120, 6)) + 2.0 * rng.normal(size=(4, 6))[y]
        assert M.nc1(Z, y) == pytest.approx(_nc1_reference(Z, y), rel=1e-6)

    def test_matches_reference_when_d_exceeds_classes_and_strings(self):
        rng = np.random.default_rng(61)
        y = rng.integers(0, 3, size=90)
        Z = rng.normal(size=(90, 40)) + 3.0 * rng.normal(size=(3, 40))[y]
        names = np.array(["x", "y", "z"])[y]
        assert M.nc1(Z, names) == pytest.approx(_nc1_reference(Z, y), rel=1e-6)

    def test_collapsed_is_near_zero_and_overlap_is_larger(self):
        rng = np.random.default_rng(62)
        y = np.repeat(np.arange(5), 40)
        centers = 5.0 * rng.normal(size=(5, 10))
        collapsed = centers[y] + 1e-4 * rng.normal(size=(200, 10))
        overlapping = centers[y] + 3.0 * rng.normal(size=(200, 10))
        assert M.nc1(collapsed, y) < 1e-6
        assert M.nc1(overlapping, y) > 100 * M.nc1(collapsed, y)

    def test_scale_invariant(self):
        rng = np.random.default_rng(63)
        y = rng.integers(0, 3, size=60)
        Z = rng.normal(size=(60, 5)) + 2.0 * rng.normal(size=(3, 5))[y]
        assert M.nc1(10.0 * Z, y) == pytest.approx(M.nc1(Z, y), rel=1e-6)

    def test_errors(self):
        Z = np.random.default_rng(64).normal(size=(20, 3))
        with pytest.raises(ValueError):
            M.nc1(Z, np.random.default_rng(0).normal(size=20))  # continuous
        with pytest.raises(ValueError):
            M.nc1(Z, np.zeros(20, dtype=int))  # one class
        with pytest.raises(ValueError):
            M.nc1(Z, np.zeros(19, dtype=int))


# ------------------------------------------------------------------- silhouette
class TestSilhouette:
    def test_tight_separated_clusters_near_one(self):
        X, y = blobs(n_per=30, n_classes=4, d=8, sep=50.0)
        assert M.silhouette(X, y, metric="cosine") > 0.9
        assert M.silhouette(X, y, metric="euclidean") > 0.9

    def test_random_labels_near_zero(self):
        rng = np.random.default_rng(70)
        X = rng.normal(size=(300, 10))
        y = rng.integers(0, 3, size=300)
        assert abs(M.silhouette(X, y)) < 0.1

    def test_matches_scikit_learn_and_subsampling_is_deterministic(self):
        from sklearn.metrics import silhouette_score

        X, y = blobs(n_per=50, n_classes=3, d=5, sep=3.0)
        assert M.silhouette(X, y, metric="euclidean") == pytest.approx(
            silhouette_score(X, y, metric="euclidean")
        )
        a = M.silhouette(X, y, metric="euclidean", max_samples=60, random_state=2)
        b = M.silhouette(X, y, metric="euclidean", max_samples=60, random_state=2)
        assert a == b and isinstance(a, float)

    def test_string_labels(self):
        X, y = blobs(n_per=20, n_classes=2, d=4, sep=50.0)
        assert M.silhouette(X, np.array(["u", "v"])[y], metric="euclidean") > 0.9


# ------------------------------------------------------------- return_per_point
class TestReturnPerPoint:
    def test_knn_accuracy(self):
        rng = np.random.default_rng(80)
        X = rng.normal(size=(300, 5))
        y = rng.integers(0, 3, size=300)
        Xtr, ytr, Xte, yte = split(X, y)
        value = M.knn_accuracy(Xtr, ytr, Xte, yte, k=5)
        v2, per_point = M.knn_accuracy(Xtr, ytr, Xte, yte, k=5, return_per_point=True)
        assert v2 == value and isinstance(v2, float)
        assert per_point.shape == (len(yte),)
        assert set(np.unique(per_point)) <= {0.0, 1.0}
        assert per_point.mean() == pytest.approx(value)

    def test_linear_probe_classification_and_regression(self):
        Xtr, ytr, Xte, yte = split(*blobs(sep=3.0))
        value = M.linear_probe(Xtr, ytr, Xte, yte)
        v2, per_point = M.linear_probe(Xtr, ytr, Xte, yte, return_per_point=True)
        assert v2 == value
        assert per_point.shape == (len(yte),)
        assert per_point.mean() == pytest.approx(value)

        rng = np.random.default_rng(81)
        X = rng.normal(size=(200, 4))
        y = X @ rng.normal(size=4)
        Xtr, ytr, Xte, yte = split(X, y)
        v3, pp = M.linear_probe(Xtr, ytr, Xte, yte, return_per_point=True)
        assert pp is None
        assert v3 == M.linear_probe(Xtr, ytr, Xte, yte)

    def test_recall_at_k(self):
        rng = np.random.default_rng(82)
        Q, G = rng.normal(size=(40, 4)), rng.normal(size=(60, 4))
        yq, yg = rng.integers(0, 12, size=40), rng.integers(0, 12, size=60)
        value = M.recall_at_k(Q, G, (yq, yg), k=3)
        v2, per_point = M.recall_at_k(Q, G, (yq, yg), k=3, return_per_point=True)
        assert v2 == value
        # only queries with a relevant gallery item are present
        n_valid = int(np.isin(yq, yg).sum())
        assert per_point.shape == (n_valid,)
        assert per_point.mean() == pytest.approx(value)


def test_linear_probe_fixed_c_skips_cross_validation(monkeypatch):
    """C fixes the regularisation (the review pilot's probe used C = 1)."""
    import sklearn.model_selection as ms

    from embedkit.evaluation.metrics import linear_probe

    rng = np.random.default_rng(0)
    Z = rng.standard_normal((120, 5))
    y = (Z[:, 0] > 0).astype(int)

    def forbidden(*a, **k):
        raise AssertionError("cross-validation ran")

    monkeypatch.setattr(ms, "cross_val_score", forbidden)
    acc = linear_probe(Z[:80], y[:80], Z[80:], y[80:], C=1.0, max_iter=3000)
    assert acc > 0.9


def test_linear_probe_does_not_amplify_rounding_residue():
    """A ~1e-17 residue proportional to the label must not become a usable feature."""
    from embedkit.evaluation.metrics import linear_probe

    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    noise = rng.standard_normal((n, 5))  # uninformative features
    leak = (y * 2.0 - 1.0) * 1e-17  # the label, at rounding scale
    Z = np.column_stack([noise, leak])
    acc = linear_probe(Z[:300], y[:300], Z[300:], y[300:], C=1.0)
    assert acc < 0.7
