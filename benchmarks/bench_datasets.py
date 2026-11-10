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
    # large unlabelled vector sets from ANN-Benchmarks (cluster)
    "glove100": _cached("glove100", "unlabelled"),
    "sift128": _cached("sift128", "unlabelled"),
    "gist960": _cached("gist960", "unlabelled"),
}

LOCAL = ("pets_resnet50", "digits", "synthetic_anisotropic")


def load(name: str, root: str | os.PathLike | None = None) -> Dataset:
    """Load a benchmark dataset by name. Raises :class:`DatasetUnavailableError` if it is absent."""
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(LOADERS)}")
    return LOADERS[name](data_root(root))
