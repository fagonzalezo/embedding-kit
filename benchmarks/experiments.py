"""Method grids for the benchmark experiments (``docs/CRITICAL_REVIEW.md`` section 5).

A method is a :class:`Method`: a name, a factory ``make(seed, dataset, device) -> transform``
(an unfitted object with ``fit`` and ``transform``), and flags. Names encode the configuration,
for example ``pca_whiten-32`` or ``embedkit-d64``, and are stable: result files are keyed on them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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

    m.append(Method("embedkit", lambda s, ds, dev: EmbedKit(random_state=s, device=dev),
                    deterministic=False, family="embedkit"))
    for k in (32, 64, 128):
        if k < d:
            m.append(Method(
                f"embedkit-d{k}",
                lambda s, ds, dev, k=k: EmbedKit(target_dim=k, random_state=s, device=dev),
                deterministic=False, family="embedkit_fixed_d", params=dict(target_dim=k),
            ))
    if labelled:
        m.append(Method(
            "embedkit_supervised",
            lambda s, ds, dev: EmbedKit(mode="supervised", random_state=s, device=dev),
            deterministic=False, fit_with_labels=True, family="embedkit_supervised",
        ))
    return m


E6_DIMS = (32, 64, 128)
E6_ARCHITECTURES = ("mlp", "residual", "linear")


def e6_methods(dataset) -> list[Method]:
    """E6: one-factor ablation of the refiner (review section 5; phase 5 of the plan).

    Every arm is self-supervised ``EmbedKit`` with the current defaults (no early stopping) and
    one factor changed, at the default (auto) d and at fixed d in ``E6_DIMS``; each family is
    selected on validation as in E1. Stage WP 5.1: the architecture (``mlp`` is the previous
    default). Raw, PCA and PCA-whitening at the same d are the reference for the gate.
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
                    target_dim=k, architecture=arch, random_state=s, device=dev),
                deterministic=False, family=f"e6_{arch}",
                params=dict(architecture=arch, target_dim=k),
            ))
    return m


EXPERIMENTS: dict[str, Callable] = {"e1": e1_methods, "e6": e6_methods}
