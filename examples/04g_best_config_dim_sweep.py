"""Best unsupervised config × multi-method ID target dims.

Takes the best config found across ablations 1-4:
  KNNPairs(k=5) + NTXentLoss(τ=0.07) + AlignUniformLoss(0.5×), 80 epochs

Sweeps target_dim values derived from the ID ablation (04f):
  24  — current auto-config (TwoNN=16.15 × 1.5)
  22  — 1.5 × median consensus (14.51)
  29  — 1.5 × mean consensus  (19.10)
  57  — 1.5 × max consensus   (lPCA=38.00)
  128 — empirically best from ablation 1-4 (reference ceiling)

Run:
    conda run -n pytorch python examples/04g_best_config_dim_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}\n")

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

from embedkit.improvement import (
    AlignUniformLoss,
    CombinedLoss,
    EmbeddingRefiner,
    KNNPairs,
    NTXentLoss,
    Trainer,
)

acc_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).score(X_test, y_test)
print(f"Raw ResNet50 kNN accuracy (k=5): {acc_raw:.3f}\n")

# target_dim → (label, source)
TARGET_DIMS = [
    (24,  "TwoNN=16.15 × 1.5        (current auto)"),
    (22,  "median=14.51 × 1.5"),
    (29,  "mean=19.10 × 1.5"),
    (57,  "max/lPCA=38.00 × 1.5"),
    (128, "empirical best (ablations 1-4)"),
]

print("Running best config at each target_dim …\n")
results: list[tuple[int, str, float]] = []

for target_dim, label in TARGET_DIMS:
    print(f"  target_dim={target_dim:<4}  ({label})", flush=True)
    trainer = Trainer(
        model=EmbeddingRefiner(input_dim=2048, target_dim=target_dim, hidden_dim=512, n_layers=2),
        augmentation=KNNPairs(k=5),
        loss=CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)]),
        epochs=80,
        batch_size=128,
        optimizer="adam",
        lr=3e-4,
        scheduler="cosine",
        eval_every=80,
        device=device,
        random_state=42,
    )
    trainer.fit(X_train)
    Xtr_r = trainer.transform(X_train)
    Xte_r = trainer.transform(X_test)
    acc = KNeighborsClassifier(n_neighbors=5).fit(Xtr_r, y_train).score(Xte_r, y_test)
    print(f"             acc={acc:.3f}", flush=True)
    results.append((target_dim, label, acc))

print()
print("=" * 70)
print("Best config × target_dim sweep — kNN (k=5) accuracy")
print("=" * 70)
print(f"{'target_dim':>10}  {'Acc':>6}  {'Δ raw':>7}  {'Source'}")
print("-" * 70)
print(f"{'raw':>10}  {acc_raw:.3f}  {'—':>7}  ResNet50 baseline")
for target_dim, label, acc in results:
    delta = (acc - acc_raw) * 100
    print(f"{target_dim:>10}  {acc:.3f}  {delta:>+6.1f}%  {label}")
print("=" * 70)
