"""Example 4: Embedding refinement for image classification (Oxford-IIIT Pets).

Backbone: ResNet50 (ImageNet weights), 2048-D penultimate features.
Dataset:  Oxford-IIIT Pets — 37 fine-grained classes, ~3.7K train / ~3.7K test.

Stages compared with kNN (k=5) accuracy:
  1. Raw ResNet50 features
  2. Unsupervised refinement   (EmbedKit self-supervised mode)
  3. Supervised refinement     (SupConLoss + AlignUniformLoss via low-level API)

Embeddings are extracted once and cached to examples/data/pets_resnet50.npz.

Install:
    pip install -e ".[examples]"   # or: uv sync --extra examples
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
CACHE = Path(__file__).parent / "data" / "pets_resnet50.npz"


def extract_or_load(cache_path: Path):
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        d = np.load(cache_path)
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    print("Cache not found — downloading Pets and extracting ResNet50 features ...")
    print("(This may take several minutes on first run.)\n")

    try:
        from torchvision import datasets, models, transforms
    except ImportError as e:
        raise ImportError(
            "torchvision is required for feature extraction.\n"
            "Install with: pip install -e '.[examples]'"
        ) from e

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_root = cache_path.parent
    data_root.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.OxfordIIITPet(root=data_root, split="trainval", download=True, transform=transform)
    test_ds  = datasets.OxfordIIITPet(root=data_root, split="test",     download=True, transform=transform)

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    def run_extraction(dataset, desc):
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=(device != "cpu"))
        features, labels = [], []
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(loader):
                feats = backbone(imgs.to(device))
                features.append(feats.cpu().numpy())
                labels.append(lbls.numpy())
                if (i + 1) % 10 == 0:
                    print(f"  {desc}: {(i + 1) * loader.batch_size}/{len(dataset)}", end="\r")
        print()
        return np.concatenate(features).astype(np.float32), np.concatenate(labels)

    X_train, y_train = run_extraction(train_ds, "train")
    X_test,  y_test  = run_extraction(test_ds,  "test ")

    np.savez_compressed(cache_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Saved embeddings to {cache_path}\n")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Load / extract
# ---------------------------------------------------------------------------
X_train, y_train, X_test, y_test = extract_or_load(CACHE)
print(f"Train: {X_train.shape}  Test: {X_test.shape}  Classes: {len(np.unique(y_train))}\n")

# ---------------------------------------------------------------------------
# EmbedKit imports (after optional torchvision dep check)
# ---------------------------------------------------------------------------
from embedkit import EmbedKit, EmbedKitAnalyzer
from embedkit.improvement import (
    AlignUniformLoss,
    CompositeAugmentation,
    CombinedLoss,
    EmbeddingRefiner,
    FeatureDropout,
    GaussianNoise,
    SupConLoss,
    Trainer,
)

# ---------------------------------------------------------------------------
# 1. Raw baseline
# ---------------------------------------------------------------------------
print("=" * 60)
print("Stage 1 — Raw ResNet50 features")
print("=" * 60)
EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X_train).print_summary()
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
acc_raw = knn_raw.score(X_test, y_test)
print(f"Raw kNN accuracy (k=5): {acc_raw:.3f}\n")

# ---------------------------------------------------------------------------
# 2. Unsupervised refinement
# ---------------------------------------------------------------------------
print("=" * 60)
print("Stage 2 — Unsupervised refinement (self-supervised)")
print("=" * 60)
ek = EmbedKit(mode="self_supervised", target_dim="auto", epochs=80, eval_every=20, device=device)
X_train_u = ek.fit_transform(X_train)
X_test_u  = ek.transform(X_test)

EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X_train_u).print_summary()
knn_u = KNeighborsClassifier(n_neighbors=5).fit(X_train_u, y_train)
acc_u = knn_u.score(X_test_u, y_test)
print(f"Unsupervised refined kNN accuracy (k=5): {acc_u:.3f}\n")

# ---------------------------------------------------------------------------
# 3. Supervised refinement
# ---------------------------------------------------------------------------
print("=" * 60)
print("Stage 3 — Supervised refinement (SupCon + AlignUniform)")
print("=" * 60)
aug     = CompositeAugmentation([GaussianNoise(std=0.05), FeatureDropout(p=0.1)])
loss_fn = CombinedLoss([(SupConLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.3)])
model   = EmbeddingRefiner(input_dim=2048, target_dim=128, hidden_dim=512, n_layers=2)
trainer = Trainer(
    model=model,
    augmentation=aug,
    loss=loss_fn,
    epochs=80,
    batch_size=128,
    optimizer="adam",
    lr=3e-4,
    scheduler="cosine",
    eval_every=20,
    eval_metrics=["uniformity", "k_skewness"],
    device=device,
    random_state=42,
)
trainer.fit(X_train, y=y_train)

X_train_s = trainer.transform(X_train)
X_test_s  = trainer.transform(X_test)

EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X_train_s).print_summary()
knn_s = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
acc_s = knn_s.score(X_test_s, y_test)
print(f"Supervised refined kNN accuracy (k=5): {acc_s:.3f}\n")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Stage':<30} {'Acc':>6}  {'Delta':>7}")
print("-" * 46)
print(f"{'Raw ResNet50':<30} {acc_raw:>6.3f}  {'—':>7}")
print(f"{'Unsupervised refinement':<30} {acc_u:>6.3f}  {(acc_u - acc_raw)*100:>+6.1f}%")
print(f"{'Supervised refinement':<30} {acc_s:>6.3f}  {(acc_s - acc_raw)*100:>+6.1f}%")
