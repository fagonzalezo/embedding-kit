"""Ablation 4: k_skewness as early-stopping monitor.

Uniformity (ablation 3) proved a poor signal because it is nearly monotonic
under NTXent+AlignUniform. k_skewness measures hubness — the number of times
a point appears as a neighbour of others — which should rise when the
contrastive loss over-pushes geometry and creates hub concentration. If it has
an inflection point it can serve as a better unsupervised proxy for kNN accuracy.

Lower k_skewness = fewer hubs = better, so the existing patience logic
(stop when score stops improving) applies without change.

Configs (all use knn_alignuni_local, target_dim=128, val_split=0.1):
  fixed_80           — 80 epochs,  no ES         (overall best so far)
  fixed_200          — 200 epochs, no ES         (known to overfit)
  es_unif_p5_e5      — uniformity monitor, p=5, e=5   (ablation-3 best ES)
  es_kskew_p5_e10    — k_skewness monitor, p=5, e=10
  es_kskew_p5_e5     — k_skewness monitor, p=5, e=5   (finer granularity)
  es_kskew_p10_e5    — k_skewness monitor, p=10, e=5  (more patience)

Run:
    conda run -n pytorch python examples/04e_unsup_ablation4.py
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


def make_trainer(epochs, val_split=0.0, patience=None, eval_every=10, monitor="uniformity") -> Trainer:
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
        eval_metrics=[monitor],
        early_stopping_patience=patience,
        monitor=monitor,
        val_split=val_split,
        device=device,
        random_state=42,
    )


# (name, epochs, val_split, patience, eval_every, monitor)
configs = [
    ("fixed_80",        80,  0.0, None,  80, "uniformity"),
    ("fixed_200",      200,  0.0, None, 200, "uniformity"),
    ("es_unif_p5_e5",  200,  0.1,    5,   5, "uniformity"),
    ("es_kskew_p5_e10",200,  0.1,    5,  10, "k_skewness"),
    ("es_kskew_p5_e5", 200,  0.1,    5,   5, "k_skewness"),
    ("es_kskew_p10_e5",200,  0.1,   10,   5, "k_skewness"),
]

print("Running ablation 4 (6 configs) …\n")
results: list[tuple[str, float, int]] = []

for name, epochs, val_split, patience, eval_every, monitor in configs:
    print(f"  [{name}] monitor={monitor}  patience={patience}  eval_every={eval_every}", flush=True)
    trainer = make_trainer(epochs, val_split, patience, eval_every, monitor)
    trainer.fit(X_train)
    stopped = trainer.stopped_epoch_
    Xtr_r = trainer.transform(X_train)
    Xte_r = trainer.transform(X_test)
    acc = KNeighborsClassifier(n_neighbors=5).fit(Xtr_r, y_train).score(Xte_r, y_test)
    print(f"           stopped={stopped:>3}  acc={acc:.3f}", flush=True)
    results.append((name, acc, stopped))

print()
print("=" * 68)
print("Ablation 4 results — kNN (k=5) accuracy")
print("=" * 68)
print(f"{'Config':<18} {'Stopped':>7}  {'Acc':>6}  {'Δ raw':>7}  {'Δ fixed80':>9}")
print("-" * 68)
acc_fixed80 = next(a for n, a, _ in results if n == "fixed_80")
print(f"{'raw':18} {'—':>7}  {acc_raw:.3f}  {'—':>7}  {'—':>9}")
for name, acc, stopped in results:
    delta_raw = (acc - acc_raw) * 100
    delta_f80 = (acc - acc_fixed80) * 100
    print(f"{name:<18} {stopped:>7}  {acc:.3f}  {delta_raw:>+6.1f}%  {delta_f80:>+8.1f}%")
print("=" * 68)
