"""Ablation 2: global KNN precomputation and training-duration sweep.

Hypothesis from ablation 1: KNNPairs(k=5) was the best augmentation but
underperformed because it fell back to batch-level neighbours under a shuffled
DataLoader. This script tests global precomputed neighbours after fixing the
KNNPairs + Trainer plumbing.

Configs:
  knn_local_80        — batch-level kNN, 80 epochs  (reproduction of ablation 1 best)
  knn_global_80       — global precomputed kNN, 80 epochs
  knn_global_200      — global precomputed kNN, 200 epochs
  knn_alignuni_local_80  — batch kNN + AlignUniform, 80 epochs (ablation 1 best overall)
  knn_alignuni_global_80 — global kNN + AlignUniform, 80 epochs
  knn_alignuni_global_200 — global kNN + AlignUniform, 200 epochs

Requires: examples/data/pets_resnet50.npz (run 04_image_classification.py first)
Run:
    conda run -n pytorch python examples/04c_unsup_ablation2.py
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


def run_config(name: str, aug, loss_fn, epochs: int) -> float:
    print(f"  [{name}] epochs={epochs}", flush=True)
    model = EmbeddingRefiner(input_dim=2048, target_dim=128, hidden_dim=512, n_layers=2)
    trainer = Trainer(
        model=model,
        augmentation=aug,
        loss=loss_fn,
        epochs=epochs,
        batch_size=128,
        optimizer="adam",
        lr=3e-4,
        scheduler="cosine",
        eval_every=epochs,
        device=device,
        random_state=42,
    )
    trainer.fit(X_train)
    Xtr_r = trainer.transform(X_train)
    Xte_r = trainer.transform(X_test)
    acc = KNeighborsClassifier(n_neighbors=5).fit(Xtr_r, y_train).score(Xte_r, y_test)
    print(f"           acc={acc:.3f}", flush=True)
    return acc


ntxent   = lambda: NTXentLoss(temperature=0.07)
alignuni = lambda: CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)])

configs = [
    # (name, aug_factory, loss_factory, epochs)
    ("knn_local_80",           lambda: KNNPairs(k=5),   ntxent,   80),
    ("knn_global_80",          lambda: KNNPairs(k=5),   ntxent,   80),
    ("knn_global_200",         lambda: KNNPairs(k=5),   ntxent,  200),
    ("knn_alignuni_local_80",  lambda: KNNPairs(k=5),   alignuni,  80),
    ("knn_alignuni_global_80", lambda: KNNPairs(k=5),   alignuni,  80),
    ("knn_alignuni_global_200",lambda: KNNPairs(k=5),   alignuni, 200),
]

# The "global" configs require precompute; "local" configs skip it.
# We force local configs to NOT have a precomputed index by setting neighbor_index=None
# explicitly and calling precompute only for global configs.
LOCAL  = {"knn_local_80", "knn_alignuni_local_80"}

print("Running ablation 2 (6 configs) …\n")
results: list[tuple[str, float]] = []

for name, aug_fn, loss_fn, epochs in configs:
    aug = aug_fn()
    if name not in LOCAL:
        # precompute=True path: build the global index before creating the Trainer
        print(f"  Precomputing global kNN for [{name}] …", flush=True)
        aug.precompute(X_train)
    loss = loss_fn()
    acc = run_config(name, aug, loss, epochs)
    results.append((name, acc))

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("Ablation 2 results — kNN (k=5) accuracy")
print("=" * 62)
print(f"{'Config':<28} {'Acc':>6}  {'Δ raw':>7}  {'Δ local80':>9}")
print("-" * 62)
acc_local80 = next(a for n, a in results if n == "knn_local_80")
print(f"{'raw':28} {acc_raw:.3f}  {'—':>7}  {'—':>9}")
for name, acc in results:
    delta_raw   = (acc - acc_raw) * 100
    delta_local = (acc - acc_local80) * 100
    print(f"{name:<28} {acc:.3f}  {delta_raw:>+6.1f}%  {delta_local:>+8.1f}%")
print("=" * 62)
