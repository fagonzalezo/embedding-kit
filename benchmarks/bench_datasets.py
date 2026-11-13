"""Benchmark datasets: loaders that return (X_train, y_train, X_test, y_test, meta).

Datasets and cached features live outside the repository, under a data root given by
``--data-root`` or the environment variable ``EMBEDKIT_BENCH_DATA`` (default
``benchmarks/data/``, git-ignored). Loaders never download anything. Features of the large sets
are produced on the cluster by ``benchmarks/extract_features.py``; a missing file raises
:class:`DatasetUnavailableError` with the command that creates it.

Labels are ``None`` for unlabelled sets (the ANN-Benchmarks vectors). STS sets carry their
test sentence pairs and gold scores in ``meta["test_pairs"]``, and a validation set of sentences
with pairs in ``meta["val_split"] = (X_val, pairs)``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO / "benchmarks" / "data"
ENV_VAR = "EMBEDKIT_BENCH_DATA"


class DatasetUnavailableError(FileNotFoundError):
    """The dataset's files are not under the data root."""


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray | None
    X_test: np.ndarray
    y_test: np.ndarray | None
    meta: dict = field(default_factory=dict)

    def as_tuple(self):
        return self.X_train, self.y_train, self.X_test, self.y_test, self.meta


def data_root(path: str | os.PathLike | None = None) -> Path:
    """The data root: the argument, else ``$EMBEDKIT_BENCH_DATA``, else ``benchmarks/data``."""
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser() if env else DEFAULT_DATA_ROOT


# ---------------------------------------------------------------------------------------------
# Small sets, available everywhere
# ---------------------------------------------------------------------------------------------


def _pets_resnet50(root: Path) -> Dataset:
    candidates = [root / "pets_resnet50.npz", REPO / "examples" / "data" / "pets_resnet50.npz"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise DatasetUnavailableError(
            "pets_resnet50: no pets_resnet50.npz under the data root or examples/data/. "
            "Create it with `python examples/04_image_classification.py` (it downloads Oxford "
            "Pets and a ResNet50) and copy it to the data root."
        )
    d = np.load(path)
    return Dataset(
        "pets_resnet50",
        d["X_train"].astype(np.float32), d["y_train"].astype(np.int64),
        d["X_test"].astype(np.float32), d["y_test"].astype(np.int64),
        dict(source=str(path), encoder="torchvision ResNet50 (ImageNet)", task="classification",
             n_classes=37, learned_embedding=True),
    )


def _digits(root: Path) -> Dataset:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)  # shipped with scikit-learn; no download
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    return Dataset(
        "digits", Xtr.astype(np.float32), ytr.astype(np.int64), Xte.astype(np.float32),
        yte.astype(np.int64),
        dict(source="sklearn.datasets.load_digits", task="classification", n_classes=10,
             learned_embedding=False, note="raw 8x8 pixels, not a learned embedding"),
    )


def make_synthetic_anisotropic(
    n=6000, n_classes=10, latent=16, D=512, sep=1.0, hidden=128, mean_scale=20.0,
    nuis_scale=2.0, n_nuis=3, seed=0,
):
    """The review pilot's synthetic set (``docs/CRITICAL_REVIEW.md`` section 3.7).

    10 Gaussian classes on a 16-d latent, through a fixed random 2-layer tanh MLP to 512-d,
    plus a large shared mean vector and ``n_nuis`` high-variance class-independent nuisance
    directions. Built to resemble common-direction anisotropy.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(n_classes, latent))
    y = np.repeat(np.arange(n_classes), n // n_classes)
    Z = centers[y] + rng.normal(0, 1.0, size=(n, latent))
    W1 = rng.normal(0, 1 / np.sqrt(latent), size=(latent, hidden))
    b1 = rng.normal(0, 0.1, hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), size=(hidden, D))
    b2 = rng.normal(0, 0.1, D)
    X = np.tanh(np.tanh(Z @ W1 + b1) @ W2 + b2)
    dev_norm = np.linalg.norm(X - X.mean(0), axis=1).mean()
    mu = rng.normal(size=D)
    mu = mu / np.linalg.norm(mu) * mean_scale * dev_norm
    Q, _ = np.linalg.qr(rng.normal(size=(D, n_nuis)))
    coef = rng.normal(0, nuis_scale * dev_norm / np.sqrt(n_nuis), size=(n, n_nuis))
    X = X + mu + coef @ Q.T
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    info = dict(sep=sep, mean_scale=mean_scale, nuis_scale=nuis_scale, n_nuis=n_nuis,
                dev_norm=float(dev_norm))
    return Xtr.astype(np.float32), ytr, Xte.astype(np.float32), yte, info


def make_synthetic_regression(n=6000, latent=8, D=256, hidden=128, nuis_scale=2.0, n_nuis=3,
                              noise=0.1, levels=None, seed=0):
    """A continuous target on a latent space seen through a random tanh MLP (E8, phase 6).

    ``u ~ N(0, I_8)``; the target is ``sin(a.u) + 0.5 (b.u)^2`` (``a``, ``b`` random unit vectors),
    standardised, plus Gaussian noise of standard deviation ``noise``. The embedding is the same
    construction as :func:`make_synthetic_anisotropic`: a fixed 2-layer tanh MLP to ``D`` dimensions
    plus ``n_nuis`` high-variance nuisance directions that do not depend on ``u``. The target is
    non-linear in the embedding, so a ridge probe on the raw rows cannot fit it exactly. Calibration
    (continuous target, test R^2): 10-NN regression on the latent ``u`` 0.77, ridge on the embedding
    0.70, 10-NN on the embedding 0.43. With ``levels``, the target is quantised to that many equally
    populated levels (quantiles of the training targets, values ``0 .. levels - 1``), on both
    splits: heavy ties, the case of ``docs/BUG-rnc-ties.md``. Half the rows are train, half test.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, latent))
    a, b = (v / np.linalg.norm(v) for v in rng.standard_normal((2, latent)))
    y = np.sin(u @ a) + 0.5 * (u @ b) ** 2
    y = (y - y.mean()) / y.std() + noise * rng.standard_normal(n)
    W1 = rng.normal(0, 1 / np.sqrt(latent), size=(latent, hidden))
    W2 = rng.normal(0, 1 / np.sqrt(hidden), size=(hidden, D))
    X = np.tanh(np.tanh(u @ W1 + rng.normal(0, 0.1, hidden)) @ W2 + rng.normal(0, 0.1, D))
    dev_norm = np.linalg.norm(X - X.mean(0), axis=1).mean()
    Q, _ = np.linalg.qr(rng.normal(size=(D, n_nuis)))
    X = X + rng.normal(0, nuis_scale * dev_norm / np.sqrt(n_nuis), size=(n, n_nuis)) @ Q.T
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed)
    if levels is not None:
        edges = np.quantile(ytr, np.linspace(0, 1, levels + 1)[1:-1])
        ytr, yte = np.digitize(ytr, edges).astype(float), np.digitize(yte, edges).astype(float)
    info = dict(latent=latent, nuis_scale=nuis_scale, n_nuis=n_nuis, noise=noise,
                levels=levels or 0)
    return Xtr.astype(np.float32), ytr.astype(np.float64), Xte.astype(np.float32), \
        yte.astype(np.float64), info


def _synthetic_regression(levels):
    def load(root: Path) -> Dataset:
        name = "synthetic_regression" + ("_ties" if levels else "")
        Xtr, ytr, Xte, yte, info = make_synthetic_regression(levels=levels)
        return Dataset(name, Xtr, ytr, Xte, yte,
                       dict(source="benchmarks.bench_datasets.make_synthetic_regression",
                            task="regression", learned_embedding=False, **info))

    return load


def label_subset(ds: Dataset, per_class: int, seed: int = 0) -> Dataset:
    """Keep ``per_class`` labelled training rows of every class (all of a smaller class).

    The label-efficiency variants of E8 (``<name>__lpc<k>``): the transform, the kNN gallery
    and the probe all see only these rows; the test split is unchanged.
    """
    rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(ds.y_train == c))[:per_class]
        for c in np.unique(ds.y_train)]))
    meta = dict(ds.meta, labels_per_class=per_class, base_dataset=ds.name,
                n_train_full=len(ds.X_train))
    for key in [k for k in meta if k.startswith("y_extra_") and k.endswith("_train")]:
        meta[key] = meta[key][keep]
    return Dataset(f"{ds.name}__lpc{per_class}", ds.X_train[keep], ds.y_train[keep], ds.X_test,
                   ds.y_test, meta)


def _synthetic_anisotropic(root: Path) -> Dataset:
    # The pilot ran make_synthetic(sep=1.0, nuis_scale=2.0).
    Xtr, ytr, Xte, yte, info = make_synthetic_anisotropic(sep=1.0, nuis_scale=2.0)
    return Dataset(
        "synthetic_anisotropic", Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64),
        dict(source="benchmarks.datasets.make_synthetic_anisotropic", task="classification",
             n_classes=10, learned_embedding=False, **info),
    )


# ---------------------------------------------------------------------------------------------
# Cached features, produced on the cluster by benchmarks/extract_features.py
# ---------------------------------------------------------------------------------------------


def _cached(name: str, task: str) -> Callable[[Path], Dataset]:
    def load(root: Path) -> Dataset:
        path = root / "features" / f"{name}.npz"
        if not path.exists():
            raise DatasetUnavailableError(
                f"{name}: {path} does not exist. Create it on the cluster with "
                f"`python benchmarks/extract_features.py --dataset {name} --data-root {root}` "
                "(see benchmarks/cluster/extract.slurm.sh). Large sets are not downloaded to "
                "laptops."
            )
        d = np.load(path, allow_pickle=False)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update(source=str(path), task=task)
        y_tr = d["y_train"] if "y_train" in d.files else None
        y_te = d["y_test"] if "y_test" in d.files else None
        if task == "sts":
            meta["test_pairs"] = (
                d["pairs_a"].astype(np.int64), d["pairs_b"].astype(np.int64),
                d["scores"].astype(np.float64),
            )
            meta["val_split"] = (
                d["X_val"].astype(np.float32),
                (d["val_pairs_a"].astype(np.int64), d["val_pairs_b"].astype(np.int64),
                 d["val_scores"].astype(np.float64)),
            )
        for key in d.files:  # extra label sets, e.g. CIFAR-100 superclasses
            if key.startswith("y_extra_"):
                meta[key] = d[key]
        return Dataset(name, d["X_train"].astype(np.float32), y_tr,
                       d["X_test"].astype(np.float32), y_te, meta)

    return load


LOADERS: dict[str, Callable[[Path], Dataset]] = {
    "pets_resnet50": _pets_resnet50,
    "digits": _digits,
    "synthetic_anisotropic": _synthetic_anisotropic,
    # images (cluster)
    "cifar100_dinov2": _cached("cifar100_dinov2", "classification"),
    "stl10_clip": _cached("stl10_clip", "classification"),
    # text (cluster)
    "banking77_mpnet": _cached("banking77_mpnet", "classification"),
    "stsb_mpnet": _cached("stsb_mpnet", "sts"),
    # regression (E8): synthetic (local), Yelp stars and protein fluorescence (cluster)
    "synthetic_regression": _synthetic_regression(None),
    "synthetic_regression_ties": _synthetic_regression(5),
    "yelp_mpnet": _cached("yelp_mpnet", "regression"),
    "fluorescence_esm2": _cached("fluorescence_esm2", "regression"),
    # large unlabelled vector sets from ANN-Benchmarks (cluster)
    "glove100": _cached("glove100", "unlabelled"),
    "sift128": _cached("sift128", "unlabelled"),
    "gist960": _cached("gist960", "unlabelled"),
}

LOCAL = ("pets_resnet50", "digits", "synthetic_anisotropic", "synthetic_regression",
         "synthetic_regression_ties")


def load(name: str, root: str | os.PathLike | None = None) -> Dataset:
    """Load a benchmark dataset by name. Raises :class:`DatasetUnavailableError` if it is absent.

    ``<name>__lpc<k>`` is the classification set ``<name>`` with ``k`` labelled training rows
    per class (:func:`label_subset`).
    """
    base, sep, variant = name.partition("__")
    if sep:
        if not variant.startswith("lpc") or not variant[3:].isdigit():
            raise KeyError(f"Unknown dataset variant {variant!r} in {name!r} (expected lpc<k>)")
        ds = load(base, root)
        if ds.meta.get("task") != "classification":
            raise ValueError(f"{name}: label subsets need a classification set")
        return label_subset(ds, int(variant[3:]))
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(LOADERS)}")
    return LOADERS[name](data_root(root))
