"""Ablation 3: unsupervised early stopping via held-out validation split.

Tests whether monitoring uniformity on a held-out val partition (val_split=0.1)
with early stopping can find a better stopping point than fixed epochs,
recovering the accuracy lost when training past the optimum (seen at 200 epochs).

Configs (all use knn_alignuni_local, the ablation-2 best, target_dim=128):
  fixed_80       — 80 epochs,  no early stopping  (ablation-2 baseline)
  fixed_200      — 200 epochs, no early stopping  (ablation-2 worst; overfits)
  es_p5_e10      — up to 200 epochs, patience=5,  eval_every=10
  es_p10_e10     — up to 200 epochs, patience=10, eval_every=10
  es_p5_e5       — up to 200 epochs, patience=5,  eval_every=5  (finer granularity)

Run:
    conda run -n pytorch python examples/04d_unsup_ablation3.py
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


def make_trainer(epochs, val_split=0.0, patience=None, eval_every=10) -> Trainer:
    return Trainer(
        model=EmbeddingRefiner(input_dim=2048, target_dim=128, hidden_dim=512, n_layers=2),
        augmentation=KNNPairs(k=5),
        loss=CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)]),
        epochs=epochs,
        batch_size=128,
        optimizer="adam",
        lr=3e-4,
        scheduler="cosine",
        eval_every=eval_every,
        eval_metrics=["uniformity"],
        early_stopping_patience=patience,
        monitor="uniformity",
        val_split=val_split,
        device=device,
        random_state=42,
    )


configs = [
    # (name, epochs, val_split, patience, eval_every)
    ("fixed_80",    80,  0.0, None,  80),
    ("fixed_200",  200,  0.0, None, 200),
    ("es_p5_e10",  200,  0.1,    5,  10),
    ("es_p10_e10", 200,  0.1,   10,  10),
    ("es_p5_e5",   200,  0.1,    5,   5),
]

print("Running ablation 3 (5 configs) …\n")
results: list[tuple[str, float, int]] = []

for name, epochs, val_split, patience, eval_every in configs:
    print(f"  [{name}] epochs≤{epochs}  val_split={val_split}  patience={patience}  eval_every={eval_every}", flush=True)
    trainer = make_trainer(epochs, val_split, patience, eval_every)
    trainer.fit(X_train)
    stopped = trainer.stopped_epoch_
    Xtr_r = trainer.transform(X_train)
    Xte_r = trainer.transform(X_test)
    acc = KNeighborsClassifier(n_neighbors=5).fit(Xtr_r, y_train).score(Xte_r, y_test)
    print(f"           stopped={stopped:>3}  acc={acc:.3f}", flush=True)
    results.append((name, acc, stopped))

print()
print("=" * 66)
print("Ablation 3 results — kNN (k=5) accuracy")
print("=" * 66)
print(f"{'Config':<14} {'Stopped':>7}  {'Acc':>6}  {'Δ raw':>7}  {'Δ fixed80':>9}")
print("-" * 66)
acc_fixed80 = next(a for n, a, _ in results if n == "fixed_80")
print(f"{'raw':14} {'—':>7}  {acc_raw:.3f}  {'—':>7}  {'—':>9}")
for name, acc, stopped in results:
    delta_raw = (acc - acc_raw) * 100
    delta_f80 = (acc - acc_fixed80) * 100
    print(f"{name:<14} {stopped:>7}  {acc:.3f}  {delta_raw:>+6.1f}%  {delta_f80:>+8.1f}%")
print("=" * 66)
