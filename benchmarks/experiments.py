"""Method grids for the benchmark experiments (``docs/dev/CRITICAL_REVIEW.md`` section 5).

A method is a :class:`Method`: a name, a factory ``make(seed, dataset, device) -> transform``
(an unfitted object with ``fit`` and ``transform``), and flags. Names encode the configuration,
for example ``pca_whiten-32`` or ``embedkit-d64``, and are stable: result files are keyed on them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from embedkit import baselines as B

DIMS = (16, 32, 64, 128, 256)


@dataclass(frozen=True)
class Method:
    name: str
    make: Callable  # (seed, dataset, device) -> unfitted transform
    deterministic: bool = True  # one seed is enough
    fit_with_labels: bool = False
    family: str = ""  # for grouping in reports, e.g. "pca", "embedkit"
    params: dict = field(default_factory=dict)
    hubness_reduction: object = None  # neighbour search for the kNN-based metrics (WP 5.5)
    metrics: tuple | None = None  # metrics for this method (default: the run's --metrics)
    transductive: bool = False  # fit on train + test rows, without labels (E7)


def _dims_for(d: int) -> list[int]:
    """Output dimensions from DIMS that do not exceed the input dimension."""
    return [k for k in DIMS if k <= d]


def e1_methods(dataset) -> list[Method]:
    """E1: training-free baselines and EmbedKit configurations at matched dimensions."""
    d = dataset.X_train.shape[1]
    labelled = dataset.y_train is not None and dataset.meta.get("task") == "classification"
    m: list[Method] = [
        Method("raw", lambda s, ds, dev: B.Identity(), family="raw"),
        Method("center_l2", lambda s, ds, dev: B.CenterNormalize(), family="center_l2"),
        Method("standardize", lambda s, ds, dev: B.Standardize(), family="standardize"),
        Method("conceptor", lambda s, ds, dev: B.ConceptorNegation(alpha=2.0),
               family="conceptor", params=dict(alpha=2.0)),
    ]
    for k in _dims_for(d):
        m += [
            Method(f"pca-{k}", lambda s, ds, dev, k=k: B.PCAReduce(k), family="pca",
                   params=dict(n_components=k)),
            Method(f"pca_whiten-{k}", lambda s, ds, dev, k=k: B.PCAReduce(k, whiten=True),
                   family="pca_whiten", params=dict(n_components=k)),
            Method(f"random_projection-{k}",
                   lambda s, ds, dev, k=k: B.RandomProjection(k, random_state=s),
                   deterministic=False, family="random_projection", params=dict(n_components=k)),
        ]
    m.append(Method("zca", lambda s, ds, dev: B.ZCAWhiten(), family="zca"))
    for D in sorted({1, 3, max(1, d // 100)}):
        m.append(Method(f"abtt-{D}", lambda s, ds, dev, D=D: B.AllButTheTop(D),
                        family="abtt", params=dict(n_remove=D)))
    for k in _dims_for(d):
        # Raunak et al. (2019): PPA(7) -> PCA(k) -> PPA(7)
        m.append(Method(f"abtt_reduce-{k}",
                        lambda s, ds, dev, k=k: B.AllButTheTop(7, n_components=k),
                        family="abtt_reduce", params=dict(n_remove=7, n_components=k)))

    from embedkit import EmbedKit

    m.append(Method("embedkit",
                    lambda s, ds, dev: EmbedKit(method="refiner", random_state=s, device=dev),
                    deterministic=False, family="embedkit"))
    for k in (32, 64, 128):
        if k < d:
            m.append(Method(
                f"embedkit-d{k}",
                lambda s, ds, dev, k=k: EmbedKit(method="refiner", target_dim=k, random_state=s,
                                            device=dev),
                deterministic=False, family="embedkit_fixed_d", params=dict(target_dim=k),
            ))
    # WP 5.5: hubness reduction on the raw embedding, a retrieval-time method. It changes the
    # neighbour search, not the coordinates, so only the kNN-based metrics are scored.
    from embedkit.hubness_reduction import CSLS, DisSimLocal, LocalScaling, MutualProximity

    hub_metrics = (("knn_accuracy", "knn_accuracy_euclidean", "recall_at_10", "k_skewness")
                   if labelled else ("k_skewness",))
    for short, red in [("ls", LocalScaling()), ("mp", MutualProximity()), ("csls", CSLS()),
                       ("dsl", DisSimLocal())]:
        m.append(Method(f"raw+{short}", lambda s, ds, dev: B.Identity(), family=f"hub_{short}",
                        params=dict(hubness_reduction=repr(red)), hubness_reduction=red,
                        metrics=hub_metrics))
    if labelled:
        m.append(Method(
            "embedkit_supervised",
            lambda s, ds, dev: EmbedKit(mode="supervised", method="refiner", random_state=s,
                                        device=dev),
            deterministic=False, fit_with_labels=True, family="embedkit_supervised",
        ))
    return m


E6_DIMS = (32, 64, 128)
E6_ARCHITECTURES = ("mlp", "residual", "linear")
MATRYOSHKA_MAX_DIM = 256


class _Prefix:
    """A fitted ``EmbedKit`` read at the prefix ``dim`` of its output (WP 5.3)."""

    def __init__(self, ek, dim: int):
        self.ek, self.dim = ek, dim

    def fit(self, X, y=None):
        self.ek.fit(X, y)  # labels reach supervised models (E8); None for self-supervised ones
        return self

    def transform(self, X):
        return self.ek.transform(X, dim=self.dim)

    def get_params(self) -> dict:
        return dict(dim=self.dim, target_dim=self.ek.target_dim,
                    architecture=self.ek.architecture)


def _default_loss():
    from embedkit.improvement.losses import AlignUniformLoss, CombinedLoss, NTXentLoss

    return CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)])


def _e6_stage2(dims, d) -> list[Method]:
    """WP 5.2 and 5.3 arms: one factor changed from the accepted WP 5.1 default (``linear``).

    Fixed d only: no family selected the auto d in stage 1. The dropout-positive arm needs
    dropout inside the refiner, which ``linear`` does not have, so it runs on ``residual`` and
    is compared with ``e6_residual``.
    """
    from embedkit import EmbedKit
    from embedkit.improvement.augmentation import IdentityPair, KNNPairs
    from embedkit.improvement.losses import (
        AlignUniformLoss,
        CombinedLoss,
        MatryoshkaLoss,
        NTXentLoss,
    )

    def loss_with(**nt):
        return CombinedLoss([(NTXentLoss(temperature=0.07, **nt), 1.0),
                             (AlignUniformLoss(), 0.5)])

    arms = {
        "mutual": ("linear", lambda: dict(augmentation=KNNPairs(k=5, mode="mutual"))),
        "nbrmask": ("linear", lambda: dict(loss=loss_with(neighbor_mask=5))),
        "decoupled": ("linear", lambda: dict(loss=loss_with(decoupled=True))),
        "dropout": ("residual", lambda: dict(augmentation=IdentityPair())),
    }
    m: list[Method] = []
    for arm, (arch, kwargs) in arms.items():
        for k in dims:
            m.append(Method(
                f"{arm}-d{k}",
                lambda s, ds, dev, k=k, arch=arch, kwargs=kwargs: EmbedKit(
                    method="refiner", target_dim=k, architecture=arch, random_state=s,
                    device=dev, **kwargs()),
                deterministic=False, family=f"e6_{arm}",
                params=dict(architecture=arch, target_dim=k, arm=arm),
            ))
    # WP 5.3: one model trained at min(256, d) with nested prefixes, read at each d.
    top = min(MATRYOSHKA_MAX_DIM, d)
    nested = tuple(k for k in dims if k < top)
    for k in dims:
        m.append(Method(
            f"matryoshka-d{k}",
            lambda s, ds, dev, k=k: _Prefix(EmbedKit(
                method="refiner", target_dim=top, architecture="linear", random_state=s, device=dev,
                loss=MatryoshkaLoss(_default_loss(), dims=nested)), k),
            deterministic=False, family="e6_matryoshka",
            params=dict(architecture="linear", target_dim=top, dim=k, nested=list(nested)),
        ))
    return m


def e6_methods(dataset) -> list[Method]:
    """E6: one-factor ablation of the refiner (review section 5; phase 5 of the plan).

    Every arm is self-supervised ``EmbedKit`` with the current defaults (no early stopping) and
    one factor changed, at the default (auto) d and at fixed d in ``E6_DIMS``; each family is
    selected on validation as in E1. Stage WP 5.1: the architecture (``mlp`` is the previous
    default). Raw, PCA and PCA-whitening at the same d are the reference for the gate.
    Stage WP 5.2/5.3: see :func:`_e6_stage2`.
    """
    from embedkit import EmbedKit

    d = dataset.X_train.shape[1]
    dims = [k for k in E6_DIMS if k < d]
    m: list[Method] = [Method("raw", lambda s, ds, dev: B.Identity(), family="raw")]
    for k in dims:
        m += [
            Method(f"pca-{k}", lambda s, ds, dev, k=k: B.PCAReduce(k), family="pca",
                   params=dict(n_components=k)),
            Method(f"pca_whiten-{k}", lambda s, ds, dev, k=k: B.PCAReduce(k, whiten=True),
                   family="pca_whiten", params=dict(n_components=k)),
        ]
    for arch in E6_ARCHITECTURES:
        for k in ["auto", *dims]:
            name = arch if k == "auto" else f"{arch}-d{k}"
            m.append(Method(
                name,
                lambda s, ds, dev, k=k, arch=arch: EmbedKit(
                    method="refiner", target_dim=k, architecture=arch, random_state=s,
                    device=dev),
                deterministic=False, family=f"e6_{arch}",
                params=dict(architecture=arch, target_dim=k),
            ))
    return m + _e6_stage2(dims, d)


E7_SIZES = (200, 500, 1000, 5000, 20000, 100000)
# The d selected on validation for the default refiner (e6_linear) in E6 stage 1.
E7_DIM = {"banking77_mpnet": 64, "cifar100_dinov2": 128, "digits": 32, "pets_resnet50": 64,
          "stl10_clip": 128, "stsb_mpnet": 128, "synthetic_anisotropic": 128}


class _Subsample:
    """Fit ``transform`` on ``n`` training rows drawn with ``seed``; transform every row (E7).

    The kNN gallery and the probe's training set stay the whole (transformed) training split; only
    the data the transform is fitted on shrink.
    """

    def __init__(self, transform, n: int, seed: int):
        self.transform_, self.n, self.seed = transform, n, seed

    def fit(self, X, y=None):
        rows = np.random.default_rng(self.seed).choice(len(X), size=min(self.n, len(X)),
                                                       replace=False)
        self.transform_.fit(X[np.sort(rows)])
        return self

    def transform(self, X):
        return self.transform_.transform(X)

    def get_params(self) -> dict:
        get = getattr(self.transform_, "get_params", None)
        return {**(get() if callable(get) else {}), "n_fit": self.n}


def e7_methods(dataset) -> list[Method]:
    """E7: learning curves over the number of rows the transform is fitted on, and inductive
    against transductive fitting (review section 5).

    The current ``EmbedKit`` default and PCA / PCA-whitening, all at the d that ``e6_linear``
    selected on validation (:data:`E7_DIM`). Sizes from :data:`E7_SIZES` below the training size,
    plus the whole training split (``-all``); the ``-transductive`` methods are fitted on train +
    test rows without labels.
    """
    from embedkit import EmbedKit

    n_train, d_in = dataset.X_train.shape
    k = min(E7_DIM.get(dataset.name, 64), d_in)
    makers = {
        "embedkit": (lambda s, dev: EmbedKit(method="refiner", target_dim=k, random_state=s,
                                            device=dev), False),
        "pca": (lambda s, dev: B.PCAReduce(k), True),
        "pca_whiten": (lambda s, dev: B.PCAReduce(k, whiten=True), True),
    }
    m: list[Method] = [Method("raw", lambda s, ds, dev: B.Identity(), family="raw")]
    for name, (make, deterministic) in makers.items():
        for n in [n for n in E7_SIZES if n < n_train]:
            m.append(Method(
                f"{name}-n{n}",
                lambda s, ds, dev, make=make, n=n: _Subsample(make(s, dev), n, s),
                deterministic=False, family=f"e7_{name}", params=dict(target_dim=k, n_fit=n),
            ))
        m.append(Method(f"{name}-all", lambda s, ds, dev, make=make: make(s, dev),
                        deterministic=deterministic, family=f"e7_{name}",
                        params=dict(target_dim=k, n_fit=n_train)))
        m.append(Method(f"{name}-transductive", lambda s, ds, dev, make=make: make(s, dev),
                        deterministic=deterministic, family=f"e7_{name}_transductive",
                        params=dict(target_dim=k), transductive=True))
    return m


class _LDA:
    """Linear discriminant analysis as a supervised transform (E8 baseline): ``min(C - 1, d)``
    discriminant directions, shrinkage covariance (Ledoit-Wolf), so it also works when a class
    has fewer rows than dimensions."""

    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        y = np.asarray(y)
        k = min(len(np.unique(y)) - 1, X.shape[1])
        self.lda_ = LinearDiscriminantAnalysis(n_components=k, solver="eigen",
                                               shrinkage="auto").fit(X, y)
        return self

    def transform(self, X):
        return self.lda_.transform(X).astype(np.float32)

    def get_params(self) -> dict:
        return dict(solver="eigen", shrinkage="auto")


class _FitLabels:
    """Fit ``transform`` with substituted labels; evaluation keeps the true ones (E8).

    ``kind="coarse"``: each fine label replaced by its superclass (``mapping``, fine -> coarse).
    ``kind="noise"``: a fraction ``rate`` of the labels replaced by a different class drawn
    uniformly (fixed by ``seed``). The kNN gallery and the probe use the clean labels, so the
    result measures what the representation learned, not how a noisy classifier scores.
    """

    def __init__(self, transform, kind: str, mapping=None, rate: float = 0.0, seed: int = 0):
        self.transform_, self.kind, self.mapping, self.rate, self.seed = (
            transform, kind, mapping, rate, seed)

    def fit(self, X, y):
        y = np.asarray(y)
        if self.kind == "coarse":
            y_fit = np.asarray([self.mapping[int(v)] for v in y])
        else:
            rng = np.random.default_rng(self.seed)
            classes = np.unique(y)
            y_fit = y.copy()
            flip = np.flatnonzero(rng.random(len(y)) < self.rate)
            for i in flip:
                y_fit[i] = rng.choice(classes[classes != y[i]])
        self.transform_.fit(X, y_fit)
        return self

    def transform(self, X):
        return self.transform_.transform(X)

    def get_params(self) -> dict:
        get = getattr(self.transform_, "get_params", None)
        return {**(get() if callable(get) else {}), "fit_labels": self.kind, "rate": self.rate}


E8_NOISE = (0.1, 0.2, 0.4)
E8_SPREAD = (0.25, 0.5)
E8_CLASS_METRICS = ("knn_accuracy", "knn_accuracy_euclidean", "linear_probe", "nc1")


def _e8_dim(d_in: int) -> int:
    return min(128, d_in // 2)


def e8_methods(dataset) -> list[Method]:
    """E8: supervised and regression modes (phase 6 of the plan, review section 5).

    Every refiner arm is ``EmbedKit`` in the dataset's mode at one d (``min(128, D // 2)``) with
    one factor changed from the phase 6 default (``supcon`` / ``rnc``). Classification: the
    previous default (``prev``: MLP, SupCon + 0.5 AlignUniform, no early stopping, as in E1), the
    uniformity term, the MLP, no early stopping, the spread term (Chen et al. 2022), nested
    output; LDA and raw as baselines; label noise (``-noise<p>``) and, on CIFAR-100, fitting on
    the 20 superclasses (``-coarse``), both scored with the clean fine labels. Label-efficiency
    variants (``<name>__lpc<k>``) run a reduced grid. Regression: the 0.1.2 RnC with the tie bug
    (``rnc_old``), no early stopping, Gaussian-noise views instead of kNN pairs, the MLP; raw and
    PCA as baselines. Both modes also run the residual adapter (``supcon_res``, ``rnc_res``:
    the PCA-initialised linear map plus a zero-initialised MLP), added after the first E8 results
    showed the plain MLP ahead in regression, to tell the non-linearity from the PCA start.
    A second round adds both to the label-efficiency grid, runs the regression architectures on
    small training sets (``<name>__n<k>``), and combines the regression MLP with Gaussian-noise
    views (``rnc_mlp_noise``) and with no early stopping (``rnc_mlp_noes``).
    """
    from embedkit import EmbedKit
    from embedkit.improvement.augmentation import GaussianNoise, KNNPairs
    from embedkit.improvement.losses import (
        AlignUniformLoss,
        CombinedLoss,
        SupConLoss,
    )

    d_in = dataset.X_train.shape[1]
    k = _e8_dim(d_in)
    task = dataset.meta.get("task")
    m: list[Method] = [Method("raw", lambda s, ds, dev: B.Identity(), family="raw",
                              metrics=E8_CLASS_METRICS if task == "classification" else None)]

    def refiner(name, family, mode, wrap=None, **kw):
        # Pin the defaults E8 ran with (before phase 6 made them depend on the mode), so that every
        # arm is "the 0.4 default plus its own change" whatever the library default is now.
        kw = {"architecture": "linear", "augmentation": lambda: KNNPairs(k=5), **kw}

        def make(s, ds, dev):
            extra = {key: (v() if callable(v) else v) for key, v in kw.items()}
            t = EmbedKit(mode=mode, method="refiner", target_dim=extra.pop("target_dim", k),
                         random_state=s, device=dev, **extra)
            return wrap(t, s) if wrap else t
        return Method(name, make, deterministic=False, fit_with_labels=True, family=family,
                      params=dict(mode=mode, target_dim=k, **{
                          key: (repr(v()) if callable(v) else v) for key, v in kw.items()}),
                      metrics=E8_CLASS_METRICS if mode == "supervised" else None)

    if task == "regression":
        from legacy_losses import RankNContrastLoss012

        m.append(Method(f"pca-{k}", lambda s, ds, dev: B.PCAReduce(k), family="pca"))
        m += [
            refiner("rnc", "rnc", "regression"),
            refiner("rnc_old", "rnc_old", "regression", loss=lambda: RankNContrastLoss012()),
            refiner("rnc_noes", "rnc_noes", "regression", early_stopping_patience=None),
            refiner("rnc_noise", "rnc_noise", "regression",
                    augmentation=lambda: GaussianNoise(std=0.05)),
            refiner("rnc_mlp", "rnc_mlp", "regression", architecture="mlp"),
            refiner("rnc_res", "rnc_res", "regression", architecture="residual"),
        ]
        if "train_rows" in dataset.meta:  # small training sets: architecture only
            return [x for x in m if x.name in ("raw", f"pca-{k}", "rnc", "rnc_mlp", "rnc_res")]
        # The MLP with each of the other one-factor changes that won (a second round).
        m += [refiner("rnc_mlp_noise", "rnc_mlp_noise", "regression", architecture="mlp",
                      augmentation=lambda: GaussianNoise(std=0.05)),
              refiner("rnc_mlp_noes", "rnc_mlp_noes", "regression", architecture="mlp",
                      early_stopping_patience=None)]
        return m
    if task != "classification":
        raise ValueError(f"E8 needs a classification or regression set, got {task!r}")

    lda = Method("lda", lambda s, ds, dev: _LDA(), fit_with_labels=True, family="lda",
                 metrics=E8_CLASS_METRICS)
    supcon = refiner("supcon", "supcon", "supervised")
    prev = refiner("prev", "prev", "supervised", architecture="mlp", early_stopping_patience=None,
                   loss=lambda: CombinedLoss([(SupConLoss(0.07), 1.0), (AlignUniformLoss(), 0.5)]))
    noes = refiner("supcon_noes", "supcon_noes", "supervised", early_stopping_patience=None)
    if "labels_per_class" in dataset.meta:  # label efficiency: a reduced grid
        # LDA needs more rows than classes.
        return m + ([lda] if dataset.meta["labels_per_class"] >= 2 else []) + [
            supcon, prev, noes,
            refiner("supcon_mlp", "supcon_mlp", "supervised", architecture="mlp"),
            refiner("supcon_res", "supcon_res", "supervised", architecture="residual")]

    m += [lda, supcon, prev, noes,
          refiner("supcon_au", "supcon_au", "supervised",
                  loss=lambda: CombinedLoss([(SupConLoss(0.07), 1.0), (AlignUniformLoss(), 0.5)])),
          refiner("supcon_mlp", "supcon_mlp", "supervised", architecture="mlp"),
          refiner("supcon_res", "supcon_res", "supervised", architecture="residual")]
    for w in E8_SPREAD:
        m.append(refiner(f"spread{w}", "supcon_spread", "supervised",
                         loss=lambda w=w: SupConLoss(0.07, spread_weight=w)))
    # Nested output: one model at min(256, D), read at k (against supcon at k).
    top = min(MATRYOSHKA_MAX_DIM, d_in)
    if top > k:
        m.append(Method(
            f"nested-d{k}",
            lambda s, ds, dev: _Prefix(EmbedKit(mode="supervised", method="refiner",
                                                target_dim=top, nested_dims=(k,),
                                                random_state=s, device=dev), k),
            deterministic=False, fit_with_labels=True, family="supcon_nested",
            params=dict(target_dim=top, dim=k), metrics=E8_CLASS_METRICS))

    def wrapped(base, suffix, wrap):
        return Method(f"{base.name}-{suffix}", lambda s, ds, dev: wrap(base.make(s, ds, dev), s),
                      deterministic=base.deterministic, fit_with_labels=True,
                      family=f"{base.family}-{suffix.rstrip('0123456789')}",
                      params={**base.params, "fit_labels": suffix}, metrics=E8_CLASS_METRICS)

    for rate in E8_NOISE:
        tag = f"noise{int(rate * 100)}"
        for base in (supcon, lda):
            m.append(wrapped(base, tag, lambda t, s, rate=rate: _FitLabels(t, "noise", rate=rate,
                                                                           seed=s)))
    if "y_extra_coarse_train" in dataset.meta:
        fine, coarse = dataset.y_train, dataset.meta["y_extra_coarse_train"]
        mapping = {int(f): int(c) for f, c in zip(fine, coarse)}
        spread = next(x for x in m if x.name == f"spread{E8_SPREAD[-1]}")
        au = next(x for x in m if x.name == "supcon_au")
        for base in (supcon, spread, au, lda):
            m.append(wrapped(base, "coarse", lambda t, s: _FitLabels(t, "coarse", mapping)))
    return m


E5_DIM = 128
E5_METRICS = ("knn_accuracy", "knn_accuracy_euclidean", "linear_probe_c1")


def e5_methods(dataset) -> list[Method]:
    """E5: the training-free baselines of phase 2 as repairs of a damaged embedding.

    Run on ``<name>`` and its ``<name>__dmg_<kind><level>`` variants (``bench_datasets.damage``).
    Fixed settings, no selection (run with ``--stages test``): PCA, whitening and the reduced
    all-but-the-top at ``E5_DIM`` dimensions, all-but-the-top with 1, 3 and d / 100 components.
    Only kNN accuracy (cosine and Euclidean) and the C = 1 probe are scored, which keeps a
    CIFAR-100 run under a few minutes (E1's trustworthiness and continuity dominated its cost).
    """
    d = dataset.X_train.shape[1]
    k = min(E5_DIM, d)
    m = [
        Method("raw", lambda s, ds, dev: B.Identity(), family="raw"),
        Method("center_l2", lambda s, ds, dev: B.CenterNormalize(), family="center_l2"),
        Method("standardize", lambda s, ds, dev: B.Standardize(), family="standardize"),
        Method("zca", lambda s, ds, dev: B.ZCAWhiten(), family="zca"),
        Method("conceptor", lambda s, ds, dev: B.ConceptorNegation(alpha=2.0),
               family="conceptor", params=dict(alpha=2.0)),
        Method(f"pca-{k}", lambda s, ds, dev: B.PCAReduce(k), family="pca",
               params=dict(n_components=k)),
        Method(f"pca_whiten-{k}", lambda s, ds, dev: B.PCAReduce(k, whiten=True),
               family="pca_whiten", params=dict(n_components=k)),
        Method(f"abtt_reduce-{k}", lambda s, ds, dev: B.AllButTheTop(7, n_components=k),
               family="abtt_reduce", params=dict(n_remove=7, n_components=k)),
    ]
    for D in sorted({1, 3, max(1, d // 100)}):
        m.append(Method(f"abtt-{D}", lambda s, ds, dev, D=D: B.AllButTheTop(D),
                        family="abtt", params=dict(n_remove=D)))
    from embedkit.hubness_reduction import CSLS, MutualProximity

    for short, red in [("mp", MutualProximity()), ("csls", CSLS())]:
        m.append(Method(f"raw+{short}", lambda s, ds, dev: B.Identity(), family=f"hub_{short}",
                        params=dict(hubness_reduction=repr(red)), hubness_reduction=red,
                        metrics=("knn_accuracy", "knn_accuracy_euclidean")))
    return [x if x.metrics else _with_metrics(x, E5_METRICS) for x in m]


def _with_metrics(method: Method, metrics: tuple) -> Method:
    from dataclasses import replace

    return replace(method, metrics=metrics)

E4_AUTO_SEED = 42  # EmbedKit's default random_state, which its analysis step uses


class _AutoPCA:
    """PCA at the dimension ``EmbedKit`` would pick on the same rows (WP 7.1).

    ``fit`` runs the analysis step of :meth:`embedkit.EmbedKit.fit` (same ``EmbedKitAnalyzer``
    settings, ``id_methods`` and ``random_state`` defaults), takes ``suggested_target_dim`` as
    ``k`` and fits ``PCAReduce(k, whiten=whiten)``. ``k`` is in ``get_params()``, so it lands in
    the result JSON (the last fit of each stage).
    """

    def __init__(self, whiten: bool = False, random_state: int | None = E4_AUTO_SEED):
        self.whiten, self.random_state = whiten, random_state

    def fit(self, X, y=None):
        from embedkit.analysis.report import EmbedKitAnalyzer

        X = np.asarray(X)
        report = EmbedKitAnalyzer(random_state=self.random_state, calibrate=False,
                                  id_bootstrap=0, id_scale_curve=False).fit(X)
        self.k_ = int(report.suggested_target_dim)
        self.pca_ = B.PCAReduce(self.k_, whiten=self.whiten).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)

    def get_params(self) -> dict:
        return dict(whiten=self.whiten, k=getattr(self, "k_", None),
                    n_components=getattr(getattr(self, "pca_", None), "n_components_", None))


def e4_methods(dataset) -> list[Method]:
    """E4 (WP 7.1): training-free interventions on the natural data sets, both stages.

    E1's training-free grid (E1's ``DIMS`` up to ``d``) with the hubness reductions, plus
    ``pca-auto`` and ``pca_whiten-auto`` at the dimension ``EmbedKit`` itself would choose
    (:class:`_AutoPCA`; their own families, so that selection within ``pca`` stays over the
    fixed grid). No refiner, no random projection, no labels. Scored on cosine kNN, Euclidean
    kNN and the C = 1 probe; the hubness reductions only on the two kNN metrics.
    """
    d = dataset.X_train.shape[1]
    m: list[Method] = [
        Method("raw", lambda s, ds, dev: B.Identity(), family="raw"),
        Method("center_l2", lambda s, ds, dev: B.CenterNormalize(), family="center_l2"),
        Method("standardize", lambda s, ds, dev: B.Standardize(), family="standardize"),
        Method("zca", lambda s, ds, dev: B.ZCAWhiten(), family="zca"),
        Method("conceptor", lambda s, ds, dev: B.ConceptorNegation(alpha=2.0),
               family="conceptor", params=dict(alpha=2.0)),
    ]
    for k in _dims_for(d):
        m += [
            Method(f"pca-{k}", lambda s, ds, dev, k=k: B.PCAReduce(k), family="pca",
                   params=dict(n_components=k)),
            Method(f"pca_whiten-{k}", lambda s, ds, dev, k=k: B.PCAReduce(k, whiten=True),
                   family="pca_whiten", params=dict(n_components=k)),
        ]
    m += [
        Method("pca-auto", lambda s, ds, dev: _AutoPCA(), family="pca_auto"),
        Method("pca_whiten-auto", lambda s, ds, dev: _AutoPCA(whiten=True),
               family="pca_whiten_auto"),
    ]
    for D in sorted({1, 3, max(1, d // 100)}):
        m.append(Method(f"abtt-{D}", lambda s, ds, dev, D=D: B.AllButTheTop(D),
                        family="abtt", params=dict(n_remove=D)))
    for k in _dims_for(d):
        m.append(Method(f"abtt_reduce-{k}",
                        lambda s, ds, dev, k=k: B.AllButTheTop(7, n_components=k),
                        family="abtt_reduce", params=dict(n_remove=7, n_components=k)))
    from embedkit.hubness_reduction import CSLS, DisSimLocal, LocalScaling, MutualProximity

    m = [_with_metrics(x, E5_METRICS) for x in m]
    for short, red in [("ls", LocalScaling()), ("mp", MutualProximity()), ("csls", CSLS()),
                       ("dsl", DisSimLocal())]:
        m.append(Method(f"raw+{short}", lambda s, ds, dev: B.Identity(), family=f"hub_{short}",
                        params=dict(hubness_reduction=repr(red)), hubness_reduction=red,
                        metrics=("knn_accuracy", "knn_accuracy_euclidean")))
    return m


# --------------------------------------------------------------------------------------------
# E10 (WP 8.2 and WP 8.3): the default selection on the E4 sets
# --------------------------------------------------------------------------------------------

E10_CAPS = (5_000, 10_000, 20_000)  # selection_max_rows values below the default that bind
E10_DEFAULT_CAP = 50_000            # EmbedKit's default selection_max_rows (WP 8.2)
E10_REGRESSION_METRICS = ("knn_r2",)  # WP 8.3 regression sets: kNN R^2 (cosine, k = 10)


class _Selected:
    """``EmbedKit`` with ``method="auto"`` that records its choice in ``get_params``.

    ``selection_report`` is the candidate table as a JSON string (the result record keeps only
    scalars and lists of scalars).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y=None):
        from embedkit import EmbedKit

        self.ek_ = EmbedKit(**self.kwargs).fit(X, y)
        return self

    def transform(self, X):
        return self.ek_.transform(X)

    def get_params(self) -> dict:
        ek = getattr(self, "ek_", None)
        if ek is None:
            return {}
        t = ek.selection_report_
        return dict(selection_max_rows=ek.selection_max_rows, selected=ek.selected_,
                    mode=ek.mode_,
                    n_selection_rows=None if t is None else int(t.attrs["n_selection_rows"]),
                    selection_report=None if t is None else t.to_json(orient="records"))


class _PcaL2:
    """PCA to ``k`` followed by L2 normalisation: the ``pca-k`` candidate of the selection."""

    def __init__(self, k: int):
        self.k = k

    def fit(self, X, y=None):
        self.pca_ = B.PCAReduce(self.k).fit(X)
        return self

    def transform(self, X):
        Z = self.pca_.transform(X)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        return (Z / np.where(norms > 1e-12, norms, 1.0)).astype(np.float32)

    def get_params(self) -> dict:
        return dict(k=self.k)


def e10_methods(dataset) -> list[Method]:
    """E10: ``EmbedKit()`` with labels (the default selection) and each of its candidates.

    - ``select``: the default, ``selection_max_rows=50_000``.
    - ``select-all``: ``selection_max_rows=None`` (the 0.6.0 selection), only where the cap
      binds (more than 50,000 training rows); elsewhere ``select`` is the same configuration.
    - ``select-cap<c>`` for ``c`` in ``E10_CAPS`` below the number of training rows (WP 8.2:
      does a subsample choose as well as all the rows?).
    - The candidates on their own, each fitted on all the training rows, for the best choice
      in hindsight (WP 8.3): ``identity``, ``center_l2``, ``pca_l2-<k>`` (64, 128, 256 below d)
      and ``refiner`` (the supervised refiner at ``min(128, d // 2)``, ``method="refiner"``).

    Classification sets (E4, and the WP 8.3 extra sets E4X) are scored on ``E5_METRICS``.
    Regression sets (WP 8.3: E8's four) run in ``mode="regression"`` (the selection and the
    refiner) and are scored on ``E10_REGRESSION_METRICS`` (kNN R^2, cosine, k = 10).
    """
    from embedkit import EmbedKit

    task = dataset.meta.get("task")
    if dataset.y_train is None or task not in ("classification", "regression"):
        return []
    regression = task == "regression"
    n, d = dataset.X_train.shape
    sel = dict(fit_with_labels=True, deterministic=False, family="select")
    mode = dict(mode="regression") if regression else {}

    def select(cap=E10_DEFAULT_CAP):
        return lambda s, ds, dev: _Selected(random_state=s, device=dev, selection_max_rows=cap,
                                            **mode)

    m = [Method("select", select(), params=dict(selection_max_rows=E10_DEFAULT_CAP), **sel)]
    if n > E10_DEFAULT_CAP:
        m.append(Method("select-all", select(None), params=dict(selection_max_rows=None),
                        **sel))
    for cap in E10_CAPS:
        if cap < n:
            m.append(Method(f"select-cap{cap}", select(cap),
                            params=dict(selection_max_rows=cap), **sel))
    m += [
        Method("identity", lambda s, ds, dev: B.Identity(), family="candidate"),
        Method("center_l2", lambda s, ds, dev: B.CenterNormalize(), family="candidate"),
    ]
    for k in (64, 128, 256):
        if k < d:
            m.append(Method(f"pca_l2-{k}", lambda s, ds, dev, k=k: _PcaL2(k),
                            family="candidate", params=dict(k=k)))
    t = min(128, max(1, d // 2))
    m.append(Method(
        "refiner",
        lambda s, ds, dev: EmbedKit(mode="regression" if regression else "supervised",
                                    method="refiner", target_dim=t,
                                    random_state=s, device=dev),
        deterministic=False, fit_with_labels=True, family="candidate",
        params=dict(target_dim=t)))
    return [_with_metrics(x, E10_REGRESSION_METRICS if regression else E5_METRICS) for x in m]


EXPERIMENTS: dict[str, Callable] = {"e1": e1_methods, "e4": e4_methods, "e5": e5_methods,
                                    "e6": e6_methods, "e7": e7_methods, "e8": e8_methods,
                                    "e10": e10_methods}
