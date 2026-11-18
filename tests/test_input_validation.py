"""WP 7.3: input validation at the public entry points (review E9, defects 10 and 17).

NaN or infinite values raise; constant columns and exact duplicate rows warn once per user call,
with counts; float64 and tensors are accepted; ``transform`` checks values and width.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import torch

from embedkit import EmbedKit, EmbedKitAnalyzer
from embedkit.analysis.geometry import (
    DistanceConcentration,
    IsotropyAnalyzer,
    NeighborConsistency,
    UniformityScore,
)
from embedkit.analysis.hubness import HubnessAnalyzer
from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator
from embedkit.analysis.kernel import KernelDiagnostics
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
from embedkit.hubness_reduction import CSLS
from embedkit.improvement import EmbeddingRefiner, GaussianNoise, NTXentLoss, Trainer
from embedkit.utils.validation import (
    count_constant_columns,
    count_duplicate_rows,
    trusted_input,
    validate_embeddings,
)

N, D = 60, 8
_PATTERN = re.compile(r"constant column|duplicate row")


def clean(dtype=np.float32, n=N, d=D, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(dtype)


def with_constants(dtype=np.float32) -> np.ndarray:
    X = clean(dtype)
    X[:, 2] = 1.0
    X[:, 5] = 0.0
    return X


def with_duplicates(dtype=np.float32) -> np.ndarray:
    X = clean(dtype)
    X[10] = X[0]
    X[11] = X[0]
    X[20] = X[3]
    return X


def structure_warnings(record) -> list[str]:
    return [str(w.message) for w in record if _PATTERN.search(str(w.message))]


def make_trainer(d=D, epochs=1) -> Trainer:
    return Trainer(
        model=EmbeddingRefiner(input_dim=d, target_dim=4),
        augmentation=GaussianNoise(std=0.05),
        loss=NTXentLoss(),
        epochs=epochs,
        eval_every=100,
        random_state=0,
    )


def fit_analyzer(X):
    return EmbedKitAnalyzer(
        id_methods=["TwoNN"], calibrate=False, id_bootstrap=0, id_scale_curve=False
    ).fit(X)


def fit_embedkit(X):
    return EmbedKit(target_dim=4, id_methods=["TwoNN"]).fit(X)


def fit_embedkit_refiner(X):
    return EmbedKit(method="refiner", target_dim=4, epochs=1, eval_every=100,
                    id_methods=["TwoNN"]).fit(X)


def fit_embedkit_transform(X):
    return EmbedKit(target_dim=4, id_methods=["TwoNN"]).fit_transform(X)


def fit_trainer(X):
    return make_trainer().fit(X)


# Entry points that take X and consume it (the ones that scan for structure and warn).
FIT_ENTRY_POINTS = {
    "EmbedKitAnalyzer": fit_analyzer,
    "IntrinsicDimensionEstimator": lambda X: IntrinsicDimensionEstimator(methods=["TwoNN"]).fit(X),
    "HubnessAnalyzer": lambda X: HubnessAnalyzer(k=5).fit(X),
    "DistanceConcentration": lambda X: DistanceConcentration().fit(X),
    "IsotropyAnalyzer": lambda X: IsotropyAnalyzer().fit(X),
    "NeighborConsistency": lambda X: NeighborConsistency(k=5, n_perturbations=2).fit(X),
    "UniformityScore": lambda X: UniformityScore().fit(X),
    "KernelDiagnostics": lambda X: KernelDiagnostics().fit(X),
    "EmbedKit.fit": fit_embedkit,
    "EmbedKit.fit(refiner)": fit_embedkit_refiner,
    "EmbedKit.fit_transform": fit_embedkit_transform,
    "Trainer.fit": fit_trainer,
    "Identity": lambda X: Identity().fit(X),
    "CenterNormalize": lambda X: CenterNormalize().fit(X),
    "Standardize": lambda X: Standardize().fit(X),
    "PCAReduce": lambda X: PCAReduce(3).fit(X),
    "ZCAWhiten": lambda X: ZCAWhiten().fit(X),
    "AllButTheTop": lambda X: AllButTheTop(1).fit(X),
    "ConceptorNegation": lambda X: ConceptorNegation().fit(X),
    "RandomProjection": lambda X: RandomProjection(3, random_state=0).fit(X),
    "CSLS": lambda X: CSLS(k=5).fit(X),
}
# CSLS does not warn: its fit runs inside evaluation loops (see hubness_reduction._prepare).
WARNING_ENTRY_POINTS = {k: v for k, v in FIT_ENTRY_POINTS.items() if k != "CSLS"}


# --------------------------------------------------------------------------- the validator


class TestValidator:
    def test_returns_float32_2d(self):
        out = validate_embeddings(clean(np.float64))
        assert out.dtype == np.float32 and out.shape == (N, D)

    def test_float32_is_not_copied(self):
        X = clean()
        assert validate_embeddings(X) is X

    def test_dtype_float64_is_kept(self):
        X = clean(np.float64)
        assert validate_embeddings(X, dtype=np.float64) is X

    def test_integers_and_lists(self):
        assert validate_embeddings([[1, 2], [3, 4]]).dtype == np.float32

    def test_one_dimensional_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_embeddings(np.ones(10))

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="at least 2 samples, got 1"):
            validate_embeddings(clean(n=1))
        with pytest.raises(ValueError, match="at least 5 samples"):
            validate_embeddings(clean(n=4), min_samples=5)

    def test_no_columns_raises(self):
        with pytest.raises(ValueError, match="no columns"):
            validate_embeddings(np.empty((5, 0)))

    def test_nan_and_inf_counts_in_message(self):
        X = clean()
        X[0, 0] = np.nan
        X[0, 1] = np.nan
        X[3, 2] = np.inf
        X[7, 4] = -np.inf
        pattern = r"4 non-finite entries \(2 NaN, 2 infinite\) in 3 of 60 rows"
        with pytest.raises(ValueError, match=pattern):
            validate_embeddings(X)

    def test_float64_overflow_to_float32_raises(self):
        X = clean(np.float64)
        X[1, 1] = 1e300
        with pytest.raises(ValueError, match="non-finite"):
            validate_embeddings(X)
        assert np.isfinite(validate_embeddings(X, dtype=np.float64)).all()

    def test_n_features_mismatch_raises(self):
        with pytest.raises(ValueError, match="features"):
            validate_embeddings(clean(), n_features=D + 1)
        validate_embeddings(clean(), n_features=D)

    def test_warn_false_is_silent(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            validate_embeddings(with_duplicates(), warn=False)
            validate_embeddings(with_constants(), warn=False)
        assert not structure_warnings(rec)

    def test_trusted_input_skips_scans_but_not_shape(self):
        X = clean()
        X[0, 0] = np.nan
        with trusted_input():
            assert validate_embeddings(X).shape == X.shape
            with pytest.raises(ValueError, match="2-D"):
                validate_embeddings(np.ones(4))
        with pytest.raises(ValueError, match="NaN"):  # the flag is reset on exit
            validate_embeddings(X)

    def test_trusted_input_resets_after_an_exception(self):
        with pytest.raises(RuntimeError), trusted_input():
            raise RuntimeError
        X = clean()
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            validate_embeddings(X)

    def test_constant_and_duplicate_messages(self):
        X = with_constants()
        X[10] = X[0]
        X[20] = X[3]
        X[30] = X[3]
        with pytest.warns(UserWarning) as rec:
            validate_embeddings(X)
        messages = structure_warnings(rec)
        assert len(messages) == 2
        assert any("2 constant column(s) of 8" in m for m in messages)
        assert any("3 duplicate row(s) of 60" in m for m in messages)

    def test_warning_points_at_the_caller(self):
        for fit in (IsotropyAnalyzer().fit, Standardize().fit, fit_analyzer):
            with pytest.warns(UserWarning) as rec:
                fit(with_constants())
            assert structure_warnings(rec)
            assert [w.filename for w in rec if _PATTERN.search(str(w.message))] == [__file__]

    def test_clean_data_is_silent(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            validate_embeddings(clean())
        assert not rec

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("levels,width", [(2, 1), (4, 3), (3, 8), (1000, 2)])
    def test_duplicate_count_matches_np_unique(self, dtype, levels, width):
        # small integer grids: structured bit patterns, many repeated rows
        rng = np.random.default_rng(1)
        base = rng.integers(0, levels, size=(1500, width)).astype(dtype)
        expected = len(base) - len(np.unique(base, axis=0))
        assert count_duplicate_rows(base) == expected
        assert count_duplicate_rows(clean(dtype)) == 0

    def test_duplicate_count_treats_negative_zero_as_zero(self):
        X = np.array([[0.0, 1.0], [-0.0, 1.0], [2.0, 1.0]], dtype=np.float32)
        assert count_duplicate_rows(X) == 1

    def test_duplicate_count_many_blocks(self):
        # more rows than one hashing block holds, duplicates in different blocks
        X = clean(n=5000, d=4)
        X[4000] = X[10]
        X[4500] = X[10]
        assert count_duplicate_rows(X) == 2

    def test_constant_columns_count(self):
        assert count_constant_columns(with_constants()) == 2
        assert count_constant_columns(clean()) == 0

    def test_torch_inputs(self):
        X = torch.from_numpy(clean())
        assert validate_embeddings(X).dtype == np.float32
        assert validate_embeddings(X.double(), dtype=np.float64).dtype == np.float64
        assert validate_embeddings(X.bfloat16()).shape == (N, D)
        assert validate_embeddings(X.clone().requires_grad_(True)).shape == (N, D)
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_embeddings(X)


# ------------------------------------------------------------------------ entry points


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("name", list(FIT_ENTRY_POINTS))
def test_fit_rejects_non_finite(name, bad):
    X = clean()
    X[4, 3] = bad
    with pytest.raises(ValueError, match="1 non-finite entries"):
        FIT_ENTRY_POINTS[name](X)


@pytest.mark.parametrize("name", list(FIT_ENTRY_POINTS))
def test_fit_rejects_non_finite_torch(name):
    X = torch.from_numpy(clean())
    X[4, 3] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite"):
        FIT_ENTRY_POINTS[name](X)


@pytest.mark.parametrize("name", list(WARNING_ENTRY_POINTS))
def test_constant_columns_warn_once(name):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        WARNING_ENTRY_POINTS[name](with_constants())
    messages = structure_warnings(rec)
    assert len(messages) == 1, messages
    assert "2 constant column(s) of 8" in messages[0]


@pytest.mark.parametrize("name", list(WARNING_ENTRY_POINTS))
def test_duplicate_rows_warn_once(name):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        WARNING_ENTRY_POINTS[name](with_duplicates())
    messages = structure_warnings(rec)
    assert len(messages) == 1, messages
    assert "3 duplicate row(s) of 60" in messages[0]


@pytest.mark.parametrize("name", list(FIT_ENTRY_POINTS))
def test_clean_data_gives_no_structure_warning(name):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        FIT_ENTRY_POINTS[name](clean())
    assert not structure_warnings(rec)


def test_each_call_warns_again():
    X = with_constants()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        analyzer = EmbedKitAnalyzer(id_methods=["TwoNN"], calibrate=False, id_bootstrap=0,
                                    id_scale_curve=False)
        analyzer.fit(X)
        analyzer.fit(X)
    assert len(structure_warnings(rec)) == 2


def test_failed_fit_does_not_leave_the_trusted_flag_on():
    X = clean()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_embedkit(X)
    with pytest.raises(ValueError, match="NaN"):
        IsotropyAnalyzer().fit(X)


def test_analyzer_still_reports_removed_duplicates():
    # the intrinsic-dimension step keeps counting the duplicates it removes (D14)
    with pytest.warns(UserWarning, match="duplicate row"):
        report = fit_analyzer(with_duplicates())
    assert report.intrinsic_dim.n_duplicates_removed == 3


def test_too_few_rows():
    with pytest.raises(ValueError, match="at least 5 samples"):
        fit_analyzer(clean(n=4))
    with pytest.raises(ValueError, match="at least 5 samples"):
        fit_embedkit(clean(n=4))
    with pytest.raises(ValueError, match="at least 2 samples"):
        IsotropyAnalyzer().fit(clean(n=1))
    with pytest.raises(ValueError, match="at least 2 samples"):
        PCAReduce(1).fit(clean(n=1))


# ------------------------------------------------------------------------------ transform


BASELINES = [
    Identity, CenterNormalize, Standardize, lambda: PCAReduce(3), ZCAWhiten,
    lambda: AllButTheTop(1), ConceptorNegation, lambda: RandomProjection(3, random_state=0),
]


@pytest.fixture(scope="module")
def fitted_trainer():
    return make_trainer().fit(clean())


@pytest.fixture(scope="module")
def fitted_embedkit():
    return EmbedKit(target_dim=4, id_methods=["TwoNN"]).fit(clean())


@pytest.mark.parametrize("factory", BASELINES)
def test_baseline_transform_checks_values_and_width(factory):
    t = factory().fit(clean())
    bad = clean()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        t.transform(bad)
    with pytest.raises(ValueError, match="features"):
        t.transform(clean(d=D - 1))


@pytest.mark.parametrize("which", ["trainer", "embedkit"])
def test_transform_checks_values_and_width(which, fitted_trainer, fitted_embedkit):
    model = fitted_trainer if which == "trainer" else fitted_embedkit
    bad = clean()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.transform(bad)
    with pytest.raises(ValueError, match="features"):
        model.transform(clean(d=D + 2))
    with pytest.raises(ValueError, match="2-D"):
        model.transform(np.ones(D, dtype=np.float32))


@pytest.mark.parametrize("which", ["trainer", "embedkit"])
def test_transform_does_not_warn_about_structure(which, fitted_trainer, fitted_embedkit):
    model = fitted_trainer if which == "trainer" else fitted_embedkit
    X = with_duplicates()
    X[:, 2] = 1.0
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model.transform(X)
    assert not structure_warnings(rec)


@pytest.mark.parametrize("factory", BASELINES)
def test_baseline_transform_does_not_warn_about_structure(factory):
    t = factory().fit(clean())
    X = with_duplicates()
    X[:, 2] = 1.0
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        t.transform(X)
    assert not structure_warnings(rec)


def test_hubness_reduction_rejects_non_finite():
    X = clean()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        CSLS(k=5).fit(X)
    r = CSLS(k=5).fit(clean())
    with pytest.raises(ValueError, match="NaN or infinite"):
        r.kneighbors(X, 3)


# ----------------------------------------------------------------- float64 and tensors


def float64_data() -> np.ndarray:
    # not representable in float32, so that the two dtypes really differ
    return np.random.default_rng(3).standard_normal((N, D)) * (1 + 1e-9)


@pytest.mark.parametrize("name", list(FIT_ENTRY_POINTS))
def test_float64_input_works_on_every_fit_path(name):
    FIT_ENTRY_POINTS[name](float64_data())


def test_float64_analysis_equals_float32_analysis():
    X64 = float64_data()
    r64 = fit_analyzer(X64)
    r32 = fit_analyzer(X64.astype(np.float32))
    assert r64.suggested_target_dim == r32.suggested_target_dim
    assert r64.k == r32.k
    assert r64.hubness.k_skewness == pytest.approx(r32.hubness.k_skewness, abs=1e-9)
    assert r64.geometry.isotropy.isotropy_score == pytest.approx(
        r32.geometry.isotropy.isotropy_score, abs=1e-9)
    assert r64.kernel.sigma == pytest.approx(r32.kernel.sigma, abs=1e-9)


@pytest.mark.parametrize("make", [IsotropyAnalyzer, UniformityScore, DistanceConcentration])
def test_float64_standalone_analyzers_equal_float32(make):
    X64 = float64_data()
    a, b = make().fit(X64).to_dict(), make().fit(X64.astype(np.float32)).to_dict()
    assert a == b


@pytest.mark.parametrize("factory", BASELINES)
def test_float64_baselines_match_float32(factory):
    X64 = float64_data()
    Z64 = factory().fit(X64).transform(X64)
    Z32 = factory().fit(X64.astype(np.float32)).transform(X64.astype(np.float32))
    assert Z64.dtype == np.float32
    # the training data differ by float32 rounding (6e-8 relative), amplified by whitening
    np.testing.assert_allclose(Z64, Z32, rtol=1e-3, atol=1e-3)


def test_float64_trainer_matches_float32():
    X64 = float64_data()
    X32 = X64.astype(np.float32)
    Z64 = make_trainer().fit(X64).transform(X64)
    Z32 = make_trainer().fit(X32).transform(X32)
    assert Z64.dtype == np.float32
    np.testing.assert_allclose(Z64, Z32, atol=1e-6)


def test_float64_embedkit_matches_float32():
    X64 = float64_data()
    X32 = X64.astype(np.float32)
    Z64 = EmbedKit(target_dim=4, id_methods=["TwoNN"]).fit_transform(X64)
    Z32 = EmbedKit(target_dim=4, id_methods=["TwoNN"]).fit_transform(X32)
    np.testing.assert_allclose(Z64, Z32, atol=1e-6)


def test_float64_hubness_reduction_keeps_precision():
    X64 = float64_data()
    r = CSLS(k=5).fit(X64)
    assert r.reference_.dtype == np.float64


def test_torch_tensors_on_every_path():
    X = torch.from_numpy(clean())
    for fit in FIT_ENTRY_POINTS.values():
        fit(X)  # accepted
    out = make_trainer().fit(X).transform(X)
    assert isinstance(out, torch.Tensor) and out.shape == (N, 4)
    ek = EmbedKit(target_dim=4, id_methods=["TwoNN"]).fit(X.double())
    assert isinstance(ek.transform(X.double()), torch.Tensor)
    np.testing.assert_allclose(Standardize().fit(X).transform(X),
                               Standardize().fit(X.numpy()).transform(X.numpy()), atol=1e-6)
