"""Method grids for the benchmark experiments (``docs/CRITICAL_REVIEW.md`` section 5).

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
            lambda s, ds, dev: EmbedKit(mode="supervised", random_state=s, device=dev),
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
        self.ek.fit(X)
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


EXPERIMENTS: dict[str, Callable] = {"e1": e1_methods, "e6": e6_methods, "e7": e7_methods}
