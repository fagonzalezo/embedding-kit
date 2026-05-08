"""Ablation study for unsupervised embedding refinement on Oxford-IIIT Pets.

Requires the feature cache produced by 04_image_classification.py:
    examples/data/pets_resnet50.npz

Six configs are compared against the raw ResNet50 baseline, each varying one
axis at a time (target_dim, augmentation, loss, input normalisation).

Run:
    conda run -n pytorch python examples/04b_unsup_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}\n")

# ---------------------------------------------------------------------------
# Load cached features
# ---------------------------------------------------------------------------
CACHE = Path(__file__).parent / "data" / "pets_resnet50.npz"
if not CACHE.exists():
    raise FileNotFoundError(
        f"Cache not found at {CACHE}.\n"
        "Run examples/04_image_classification.py first to extract features."
    )

d = np.load(CACHE)
X_train, y_train = d["X_train"], d["y_train"]
X_test,  y_test  = d["X_test"],  d["y_test"]
print(f"Train: {X_train.shape}  Test: {X_test.shape}  Classes: {len(np.unique(y_train))}\n")

# ---------------------------------------------------------------------------
# EmbedKit imports
# ---------------------------------------------------------------------------
from embedkit.improvement import (
    AlignUniformLoss,
    CombinedLoss,
    CompositeAugmentation,
    EmbeddingMixup,
    EmbeddingRefiner,
    FeatureDropout,
    GaussianNoise,
    KNNPairs,
    NTXentLoss,
    Trainer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-8, None)


def run_config(
    name: str,
    target_dim: int,
    aug,
    loss_fn,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    epochs: int = 80,
) -> float:
    print(f"  [{name}] target_dim={target_dim}", flush=True)
    model = EmbeddingRefiner(input_dim=X_tr.shape[1], target_dim=target_dim, hidden_dim=512, n_layers=2)
    trainer = Trainer(
        model=model,
        augmentation=aug,
        loss=loss_fn,
        epochs=epochs,
        batch_size=128,
        optimizer="adam",
        lr=3e-4,
        scheduler="cosine",
        eval_every=epochs,          # only log once per config to keep output clean
        device=device,
        random_state=42,
    )
    trainer.fit(X_tr)
    Xtr_r = trainer.transform(X_tr)
    Xte_r = trainer.transform(X_te)
    acc = KNeighborsClassifier(n_neighbors=5).fit(Xtr_r, y_train).score(Xte_r, y_test)
    print(f"           acc={acc:.3f}", flush=True)
    return acc


# ---------------------------------------------------------------------------
# Raw baseline
# ---------------------------------------------------------------------------
acc_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).score(X_test, y_test)
print(f"Raw ResNet50 kNN accuracy (k=5): {acc_raw:.3f}\n")

# ---------------------------------------------------------------------------
# Ablation configs
# (name, target_dim, aug, loss, use_l2_norm)
# ---------------------------------------------------------------------------
configs: list[tuple[str, int, object, object, bool]] = [
    (
        "auto_baseline",
        24,
        EmbeddingMixup(k=30, alpha=0.4),
        NTXentLoss(temperature=0.07),
        False,
    ),
    (
        "dim128",
        128,
        EmbeddingMixup(k=30, alpha=0.4),
        NTXentLoss(temperature=0.07),
        False,
    ),
    (
        "noise_dropout",
        128,
        CompositeAugmentation([GaussianNoise(std=0.05), FeatureDropout(p=0.1)]),
        NTXentLoss(temperature=0.07),
        False,
    ),
    (
        "knn_pairs",
        128,
        KNNPairs(k=5),
        NTXentLoss(temperature=0.07),
        False,
    ),
    (
        "knn_alignuni",
        128,
        KNNPairs(k=5),
        CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)]),
        False,
    ),
    (
        "l2norm_knn_alignuni",
        128,
        KNNPairs(k=5),
        CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)]),
        True,
    ),
]

# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------
print("Running ablation (6 configs × 80 epochs) …\n")
results: list[tuple[str, float]] = []

for name, target_dim, aug, loss_fn, use_l2 in configs:
    Xtr = l2_normalize(X_train) if use_l2 else X_train
    Xte = l2_normalize(X_test)  if use_l2 else X_test
    acc = run_config(name, target_dim, aug, loss_fn, Xtr, Xte)
    results.append((name, acc))

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
print()
print("=" * 58)
print("Ablation results — kNN (k=5) accuracy")
print("=" * 58)
print(f"{'Config':<24} {'Acc':>6}  {'Δ raw':>7}  {'Δ auto':>7}")
print("-" * 58)
acc_auto = next(a for n, a in results if n == "auto_baseline")
print(f"{'raw':24} {acc_raw:.3f}  {'—':>7}  {'—':>7}")
for name, acc in results:
    delta_raw  = (acc - acc_raw) * 100
    delta_auto = (acc - acc_auto) * 100
    print(f"{name:<24} {acc:.3f}  {delta_raw:>+6.1f}%  {delta_auto:>+6.1f}%")
print("=" * 58)
