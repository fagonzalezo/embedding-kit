"""Tests for embedkit.improvement.losses.

Every loss is compared with an independent, deliberately naive reference
implementation (explicit Python loops, float64) on values and, where it
matters, on gradients.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from embedkit.improvement.losses import (
    AlignUniformLoss,
    BaseLoss,
    CombinedLoss,
    NTXentLoss,
    RankNContrastLoss,
    SupConLoss,
    TripletLoss,
)

# ---------------------------------------------------------------------------
# Reference implementations (naive, float64)
# ---------------------------------------------------------------------------


def triplet_reference(z_i, z_j, labels, margin, mining, distance):
    z = torch.cat([z_i, z_j]).double()
    if distance == "cosine":
        z = F.normalize(z, dim=1)
    lbl = torch.cat([labels, labels]).tolist()
    n = z.shape[0]

    def dist(a, b):
        if distance == "cosine":
            return 1.0 - (z[a] * z[b]).sum()
        return ((z[a] - z[b]) ** 2).sum().sqrt()

    terms = []
    for i in range(n):
        pos = [dist(i, p) for p in range(n) if p != i and lbl[p] == lbl[i]]
        neg = [dist(i, k) for k in range(n) if lbl[k] != lbl[i]]
        if not pos or not neg:
            continue
        if mining == "hard":
            d_pos = max(pos, key=lambda t: t.item())
            d_neg = min(neg, key=lambda t: t.item())
        elif mining == "semi-hard":
            d_pos = sum(pos) / len(pos)
            harder = [d for d in neg if d.item() > d_pos.item()]
            d_neg = min(harder or neg, key=lambda t: t.item())
        else:
            d_pos = sum(pos) / len(pos)
            d_neg = sum(neg) / len(neg)
        terms.append(torch.clamp(d_pos - d_neg + margin, min=0.0))
    if not terms:
        return z.sum() * 0.0
    return sum(terms) / len(terms)


def ntxent_reference(z_i, z_j, temperature):
    z = F.normalize(torch.cat([z_i, z_j]).double(), dim=1)
    two_n = z.shape[0]
    n = two_n // 2
    sim = z @ z.T / temperature
    total = 0.0
    for i in range(two_n):
        others = [k for k in range(two_n) if k != i]
        pos = (i + n) % two_n
        total = total - (sim[i, pos] - torch.logsumexp(sim[i, others], dim=0))
    return total / two_n


def supcon_reference(z_i, z_j, labels, temperature):
    z = F.normalize(torch.cat([z_i, z_j]).double(), dim=1)
    two_n = z.shape[0]
    lbl = torch.cat([labels, labels]).tolist()
    sim = z @ z.T / temperature
    total = 0.0
    for i in range(two_n):
        others = [k for k in range(two_n) if k != i]
        log_denom = torch.logsumexp(sim[i, others], dim=0)
        pos = [p for p in others if lbl[p] == lbl[i]]
        total = total + sum(-(sim[i, p] - log_denom) for p in pos) / max(len(pos), 1)
    return total / two_n


def rnc_reference(z_i, z_j, labels, temperature):
    """Zha et al. (2023), Eq. 2: S_ij = {k != i : d(y_i, y_k) >= d(y_i, y_j)} (ties included)."""
    z = F.normalize(torch.cat([z_i, z_j]).double(), dim=1)
    lbl = torch.cat([labels, labels]).double()
    lbl = lbl[:, None] if lbl.dim() == 1 else lbl
    two_n = z.shape[0]
    rows = lbl.tolist()
    d = [
        [math.sqrt(sum((a - b) ** 2 for a, b in zip(rows[i], rows[k]))) for k in range(two_n)]
        for i in range(two_n)
    ]
    sim = z @ z.T / temperature
    total = 0.0
    count = 0
    for i in range(two_n):
        for j in range(two_n):
            if j == i:
                continue
            members = [k for k in range(two_n) if k != i and d[i][k] >= d[i][j]]
            total = total + torch.logsumexp(sim[i, members], dim=0) - sim[i, j]
            count += 1
    return total / count


# ---------------------------------------------------------------------------
# Triplet
# ---------------------------------------------------------------------------

MININGS = ["hard", "semi-hard", "easy"]
DISTANCES = ["euclidean", "cosine"]


def _triplet_data(dtype=torch.float64, n=14, d=5, classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    z1 = torch.randn(n, d, generator=g, dtype=dtype)
    z2 = z1 + 0.3 * torch.randn(n, d, generator=g, dtype=dtype)
    y = torch.randint(0, classes, (n,), generator=g)
    return z1, z2, y


class TestTriplet:
    @pytest.mark.parametrize("distance", DISTANCES)
    @pytest.mark.parametrize("mining", MININGS)
    def test_value_matches_reference(self, mining, distance):
        z1, z2, y = _triplet_data()
        loss = TripletLoss(0.5, mining, distance)(z1, z2, y)
        ref = triplet_reference(z1, z2, y, 0.5, mining, distance)
        assert abs(loss.item() - ref.item()) < 1e-8
        assert loss.item() > 0  # the comparison is not vacuous

    @pytest.mark.parametrize("distance", DISTANCES)
    @pytest.mark.parametrize("mining", MININGS)
    def test_gradient_matches_reference(self, mining, distance):
        z1, z2, y = _triplet_data()
        z1 = z1.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(TripletLoss(0.5, mining, distance)(z1, z2, y), z1)
        (g_ref,) = torch.autograd.grad(triplet_reference(z1, z2, y, 0.5, mining, distance), z1)
        assert torch.allclose(g, g_ref, atol=1e-8)
        assert g.abs().sum() > 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("distance", DISTANCES)
    @pytest.mark.parametrize("mining", MININGS)
    def test_gradients_finite_with_duplicate_embeddings(self, mining, distance, dtype):
        z1, _, y = _triplet_data(dtype=dtype, n=12)
        z1[3] = z1[0]  # duplicates inside a view
        z1[7] = z1[0]
        z1 = z1.clone().requires_grad_(True)
        z2 = z1.detach().clone()  # every embedding also duplicates its other view
        loss = TripletLoss(0.5, mining, distance)(z1, z2, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(z1.grad).all()

    @pytest.mark.parametrize("mining", MININGS)
    def test_duplicate_embeddings_value_close_to_reference(self, mining):
        z1, _, y = _triplet_data(n=12)
        z1[3] = z1[0]
        loss = TripletLoss(0.5, mining)(z1, z1.clone(), y)
        ref = triplet_reference(z1, z1.clone(), y, 0.5, mining, "euclidean")
        assert abs(loss.item() - ref.item()) < 1e-4  # sqrt(eps) floor on zero distances

    def test_gradients_finite_float32_random_batch(self):
        z1, z2, y = _triplet_data(dtype=torch.float32, n=64, d=16, classes=4)
        z1 = z1.clone().requires_grad_(True)
        TripletLoss(0.5, "hard")(z1, z2, y).backward()
        assert torch.isfinite(z1.grad).all()

    @pytest.mark.parametrize("mining", MININGS)
    def test_no_valid_anchor_returns_zero_that_keeps_graph(self, mining):
        z1 = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
        y = torch.zeros(6, dtype=torch.long)  # a single class: no negatives anywhere
        loss = TripletLoss(0.5, mining)(z1, z1.detach() + 0.1, y)
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()
        assert torch.isfinite(z1.grad).all()

    @pytest.mark.parametrize("mining", MININGS)
    def test_two_class_value_matches_reference(self, mining):
        z1, z2, y = _triplet_data(n=10, classes=2, seed=5)
        loss = TripletLoss(0.5, mining)(z1, z2, y)
        ref = triplet_reference(z1, z2, y, 0.5, mining, "euclidean")
        assert abs(loss.item() - ref.item()) < 1e-8

    def test_requires_labels(self):
        with pytest.raises(ValueError):
            TripletLoss()(torch.randn(4, 3), torch.randn(4, 3))


# ---------------------------------------------------------------------------
# NT-Xent
# ---------------------------------------------------------------------------

N_NT = 50


class TestNTXent:
    def test_value_matches_reference(self):
        torch.manual_seed(0)
        z1, z2 = torch.randn(20, 8, dtype=torch.float64), torch.randn(20, 8, dtype=torch.float64)
        loss = NTXentLoss(0.2)(z1, z2)
        assert abs(loss.item() - ntxent_reference(z1, z2, 0.2).item()) < 1e-9

    @pytest.mark.parametrize(
        "chunk_size", [1, 7, N_NT - 1, N_NT, N_NT + 1, 2 * N_NT - 1, 2 * N_NT]
    )
    def test_chunked_equals_full_and_reference(self, chunk_size):
        torch.manual_seed(1)
        z1 = torch.randn(N_NT, 8, dtype=torch.float64, requires_grad=True)
        z2 = torch.randn(N_NT, 8, dtype=torch.float64)
        full = NTXentLoss(0.1)(z1, z2)
        chunked = NTXentLoss(0.1, chunk_size=chunk_size)(z1, z2)
        assert abs(full.item() - chunked.item()) < 1e-6
        assert abs(chunked.item() - ntxent_reference(z1, z2, 0.1).item()) < 1e-6
        (g_full,) = torch.autograd.grad(full, z1)
        (g_chunk,) = torch.autograd.grad(chunked, z1)
        assert torch.allclose(g_full, g_chunk, atol=1e-8)

    def test_chunked_float32_gradient_finite(self):
        z1 = torch.randn(30, 8, requires_grad=True)
        NTXentLoss(0.1, chunk_size=7)(z1, torch.randn(30, 8)).backward()
        assert torch.isfinite(z1.grad).all()


# ---------------------------------------------------------------------------
# Rank-N-Contrast
# ---------------------------------------------------------------------------

B_RNC = 16  # 2N = 32 > 25: the range where torch.cdist switches to a matrix-product formula


def _rnc_labels(kind, b=B_RNC, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "continuous":
        return torch.randn(b, generator=g, dtype=torch.float64)
    if kind == "three_level":
        return torch.randint(0, 3, (b,), generator=g)
    if kind == "binary_vector":
        return torch.randint(0, 2, (b, 6), generator=g).double()
    if kind == "one_hot":
        return F.one_hot(torch.randint(0, 5, (b,), generator=g), 5).double()
    if kind == "continuous_13d":
        return torch.randn(b, 13, generator=g, dtype=torch.float64)
    raise ValueError(kind)


LABEL_KINDS = ["continuous", "three_level", "binary_vector", "one_hot"]


def _rnc_embeddings(b=B_RNC, seed=1):
    g = torch.Generator().manual_seed(seed)
    z1 = torch.randn(b, 8, generator=g, dtype=torch.float64)
    z2 = torch.randn(b, 8, generator=g, dtype=torch.float64)
    return z1, z2


class TestRankNContrast:
    @pytest.mark.parametrize("kind", LABEL_KINDS)
    def test_value_matches_reference(self, kind):
        z1, z2 = _rnc_embeddings()
        y = _rnc_labels(kind)
        loss = RankNContrastLoss(0.1)(z1, z2, y)
        assert abs(loss.item() - rnc_reference(z1, z2, y, 0.1).item()) < 1e-6

    @pytest.mark.parametrize("kind", LABEL_KINDS)
    def test_gradient_matches_reference(self, kind):
        z1, z2 = _rnc_embeddings()
        y = _rnc_labels(kind)
        z1 = z1.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(RankNContrastLoss(0.1)(z1, z2, y), z1)
        (g_ref,) = torch.autograd.grad(rnc_reference(z1, z2, y, 0.1), z1)
        assert torch.allclose(g, g_ref, atol=1e-6)

    @pytest.mark.parametrize("kind", LABEL_KINDS)
    def test_row_permutation_invariance(self, kind):
        z1, z2 = _rnc_embeddings()
        y = _rnc_labels(kind)
        base = RankNContrastLoss(0.1)(z1, z2, y)
        for seed in range(3):
            p = torch.randperm(B_RNC, generator=torch.Generator().manual_seed(seed))
            permuted = RankNContrastLoss(0.1)(z1[p], z2[p], y[p])
            assert abs(base.item() - permuted.item()) < 1e-9

    def test_distinct_continuous_labels_match_reference(self):
        """No ties other than the two views of each sample."""
        z1, z2 = _rnc_embeddings()
        y = _rnc_labels("continuous", seed=7)
        loss = RankNContrastLoss(0.2)(z1, z2, y)
        assert abs(loss.item() - rnc_reference(z1, z2, y, 0.2).item()) < 1e-6

    def test_exact_ties_survive_with_more_than_25_rows(self):
        """Identical label vectors tie exactly (torch.cdist's default mode does not give that)."""
        from embedkit.improvement.losses.rnc import _label_distances

        torch.manual_seed(0)
        y = torch.randn(64, 13)
        lbl = torch.cat([y, y])  # 128 rows, rows i and i + 64 identical
        D = _label_distances(lbl)
        idx = torch.arange(64)
        assert torch.equal(D[idx, idx + 64], torch.zeros(64))
        # Both views of a sample are at exactly the same distance from every other row.
        assert torch.equal(D[:, :64], D[:, 64:])

    def test_value_matches_reference_with_13d_float_labels_over_25_rows(self):
        z1, z2 = _rnc_embeddings(b=20)
        y = _rnc_labels("continuous_13d", b=20, seed=4)
        loss = RankNContrastLoss(0.1)(z1, z2, y.float())
        assert abs(loss.item() - rnc_reference(z1, z2, y.float(), 0.1).item()) < 1e-5

    @pytest.mark.parametrize("chunk_size", [1, 5, 7, 31, 32, 33])
    @pytest.mark.parametrize("kind", ["three_level", "one_hot", "continuous"])
    def test_chunked_equals_full_and_reference_with_ties(self, kind, chunk_size):
        z1, z2 = _rnc_embeddings()
        y = _rnc_labels(kind)
        z1 = z1.clone().requires_grad_(True)
        full = RankNContrastLoss(0.1)(z1, z2, y)
        chunked = RankNContrastLoss(0.1, chunk_size=chunk_size)(z1, z2, y)
        assert abs(full.item() - chunked.item()) < 1e-9
        assert abs(chunked.item() - rnc_reference(z1, z2, y, 0.1).item()) < 1e-6
        (g_full,) = torch.autograd.grad(full, z1)
        (g_chunk,) = torch.autograd.grad(chunked, z1)
        assert torch.allclose(g_full, g_chunk, atol=1e-8)

    def test_requires_labels(self):
        with pytest.raises(ValueError):
            RankNContrastLoss()(torch.randn(4, 3), torch.randn(4, 3))

    def test_docstring_states_tie_semantics(self):
        assert "tie" in RankNContrastLoss.__doc__.lower()


# ---------------------------------------------------------------------------
# SupCon
# ---------------------------------------------------------------------------


class TestSupCon:
    @pytest.mark.parametrize("chunk_size", [None, 7])
    def test_value_matches_loop_reference_1d_labels(self, chunk_size):
        g = torch.Generator().manual_seed(0)
        z1 = torch.randn(20, 8, generator=g, dtype=torch.float64)
        z2 = torch.randn(20, 8, generator=g, dtype=torch.float64)
        y = torch.randint(0, 4, (20,), generator=g)
        loss = SupConLoss(0.1, chunk_size=chunk_size)(z1, z2, y)
        assert abs(loss.item() - supcon_reference(z1, z2, y, 0.1).item()) < 1e-9

    def test_column_vector_labels_equal_1d_labels(self):
        z1, z2 = torch.randn(10, 4), torch.randn(10, 4)
        y = torch.randint(0, 3, (10,))
        a = SupConLoss(0.1)(z1, z2, y)
        b = SupConLoss(0.1)(z1, z2, y[:, None])
        assert torch.equal(a, b)

    @pytest.mark.parametrize("chunk_size", [None, 5])
    def test_multi_column_labels_raise(self, chunk_size):
        z1, z2 = torch.randn(10, 4), torch.randn(10, 4)
        y = torch.randint(0, 2, (10, 3))
        with pytest.raises(ValueError, match="1-D"):
            SupConLoss(0.1, chunk_size=chunk_size)(z1, z2, y)

    def test_label_length_mismatch_raises(self):
        z1, z2 = torch.randn(10, 4), torch.randn(10, 4)
        with pytest.raises(ValueError):
            SupConLoss(0.1)(z1, z2, torch.randint(0, 2, (9,)))


# ---------------------------------------------------------------------------
# CombinedLoss and uses_labels
# ---------------------------------------------------------------------------


class _Spy(BaseLoss):
    def __init__(self, uses_labels):
        super().__init__()
        self.uses_labels = uses_labels
        self.received = "unset"

    def forward(self, z_i, z_j, labels=None):
        self.received = labels
        return (z_i - z_j).pow(2).mean()


class _LabelBoom(BaseLoss):
    """A supervised loss that raises ValueError whenever it gets labels."""

    uses_labels = True

    def forward(self, z_i, z_j, labels=None):
        if labels is not None:
            raise ValueError("bad label shape")
        return z_i.sum() * 0.0


class TestCombined:
    def test_uses_labels_declared_on_every_loss(self):
        assert BaseLoss.uses_labels is False
        for cls in (SupConLoss, TripletLoss, RankNContrastLoss):
            assert cls.uses_labels is True, cls
        for cls in (NTXentLoss, AlignUniformLoss):
            assert cls.uses_labels is False, cls

    def test_combined_uses_labels_is_any_of_members(self):
        assert CombinedLoss([(NTXentLoss(), 1.0), (AlignUniformLoss(), 1.0)]).uses_labels is False
        assert CombinedLoss([(NTXentLoss(), 1.0), (SupConLoss(), 1.0)]).uses_labels is True
        nested = CombinedLoss([(CombinedLoss([(TripletLoss(), 1.0)]), 1.0), (NTXentLoss(), 1.0)])
        assert nested.uses_labels is True

    def test_labels_reach_only_losses_that_use_them(self):
        sup, unsup = _Spy(True), _Spy(False)
        z1, z2 = torch.randn(6, 3), torch.randn(6, 3)
        y = torch.arange(6)
        CombinedLoss([(sup, 1.0), (unsup, 1.0)])(z1, z2, y)
        assert sup.received is y
        assert unsup.received is None

    def test_value_is_weighted_sum(self):
        torch.manual_seed(0)
        z1, z2 = torch.randn(12, 5), torch.randn(12, 5)
        y = torch.randint(0, 3, (12,))
        a, b = SupConLoss(0.1), NTXentLoss(0.1)
        combined = CombinedLoss([(a, 1.0), (b, 0.5)])(z1, z2, y)
        assert torch.allclose(combined, a(z1, z2, y) + 0.5 * b(z1, z2), atol=1e-6)

    def test_error_inside_supervised_loss_propagates(self):
        z1, z2 = torch.randn(6, 3), torch.randn(6, 3)
        with pytest.raises(ValueError, match="bad label shape"):
            CombinedLoss([(NTXentLoss(), 1.0), (_LabelBoom(), 1.0)])(z1, z2, torch.arange(6))

    def test_wrong_label_shape_in_supcon_propagates(self):
        z1, z2 = torch.randn(6, 3), torch.randn(6, 3)
        y = torch.randint(0, 2, (6, 2))
        with pytest.raises(ValueError):
            CombinedLoss([(NTXentLoss(), 1.0), (SupConLoss(), 1.0)])(z1, z2, y)

    def test_supervised_loss_without_labels_still_raises(self):
        z1, z2 = torch.randn(6, 3), torch.randn(6, 3)
        with pytest.raises(ValueError, match="requires labels"):
            CombinedLoss([(TripletLoss(), 1.0)])(z1, z2, None)

    def test_gradient_flows(self):
        z1 = torch.randn(8, 4, requires_grad=True)
        y = torch.randint(0, 2, (8,))
        loss_fn = CombinedLoss([(SupConLoss(), 1.0), (AlignUniformLoss(), 0.5)])
        loss_fn(z1, z1.detach() + 0.1, y).backward()
        assert torch.isfinite(z1.grad).all()


def test_align_uniform_matches_loop_reference():
    """Uniformity term uses 2 - 2 a.b for unit vectors; check against an explicit loop."""
    import math

    import torch.nn.functional as F

    from embedkit.improvement.losses import AlignUniformLoss

    torch.manual_seed(0)
    z1, z2 = torch.randn(9, 5, dtype=torch.float64), torch.randn(9, 5, dtype=torch.float64)
    a, b = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    align = sum(float((a[i] - b[i]).norm() ** 2) for i in range(9)) / 9
    z = torch.cat([a, b])
    pots = [math.exp(-2.0 * float((z[i] - z[j]).norm() ** 2))
            for i in range(18) for j in range(i + 1, 18)]
    expected = align + math.log(sum(pots) / len(pots))
    assert AlignUniformLoss()(z1, z2).item() == pytest.approx(expected, abs=1e-9)
