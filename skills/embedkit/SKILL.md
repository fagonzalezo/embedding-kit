---
name: embedkit
description: >
  Guide ML practitioners through analyzing, improving, and integrating embedding spaces
  using the EmbedKit library. Use this skill whenever the user has embedding arrays
  (NumPy/PyTorch), wants to assess embedding quality, diagnose geometric pathologies
  (hubness, anisotropy, distance concentration, intrinsic dimensionality mismatch),
  choose improvement strategies, apply contrastive learning to refine embeddings, or
  plug EmbedKit into their own ML workflows (HuggingFace, PyTorch training loops,
  scikit-learn pipelines, RAG systems, retrieval stacks). Trigger on: "embeddings",
  "hubness", "intrinsic dimension", "isotropy", "embedding space", "refine embeddings",
  "EmbedKit", sentence/image/word embedding quality, kNN quality, representation
  learning diagnostics, retrieval quality, semantic similarity, embedding geometry.
---

# EmbedKit Skill

EmbedKit is a Python library at `embedkit/` (project root) that diagnoses geometric
pathologies in embedding spaces and learns improved embeddings via contrastive training.

## Installation

### uv

```bash
uv add git+https://github.com/fagonzalezo/embedding-kit.git
```

### pip

```bash
pip install git+https://github.com/fagonzalezo/embedding-kit.git
```

---

## Quick Start

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

X = np.load("embeddings.npy")   # shape (n_samples, n_features)

# ── Analysis only ──────────────────────────────────────────
report = EmbedKitAnalyzer(k=10).fit(X)
report.print_summary()          # human-readable overview
df = report.to_dataframe()      # pandas DataFrame of all metrics

# ── Full self-supervised refinement (auto-configured) ──────
X_refined = EmbedKit(mode="self_supervised", epochs=200).fit_transform(X)

# ── Supervised refinement with class labels ────────────────
y = np.load("labels.npy")       # integer class indices
X_refined = EmbedKit(mode="supervised", epochs=100).fit_transform(X, y=y)
```

---

## Analysis API

```python
from embedkit import EmbedKitAnalyzer

analyzer = EmbedKitAnalyzer(
    k=10,                                    # kNN size; auto = sqrt(n)/2 if None
    id_methods=["TwoNN", "MLE", "lPCA", "MOM"],  # ID estimators (default)
    metric="euclidean",                       # distance metric
    random_state=42,
)
report = analyzer.fit(X, y=None)    # y enables kernel_alignment
```

`EmbedKitAnalyzer.fit()` runs all diagnostics in one pass and returns an
`EmbedKitReport` with these sub-results:

| Attribute | Type | What it holds |
|---|---|---|
| `report.intrinsic_dim` | `IntrinsicDimensionResult` | Per-method ID estimates, consensus, uncertainty |
| `report.hubness` | `HubnessResult` | k-occurrence distribution, skewness, hub/antihub ratios |
| `report.geometry` | `GeometryBundle` | Distance concentration, isotropy, neighbor consistency, uniformity |
| `report.kernel` | `KernelDiagnosticsResult` | RBF kernel rank, spectral gap, condition number |
| `report.severity` | `"low"/"medium"/"high"` | Overall pathology severity |
| `report.recommendations` | `list[str]` | Actionable suggestions |
| `report.suggested_k` | `int` | Recommended kNN size |
| `report.suggested_sigma` | `float` | Recommended RBF sigma |
| `report.suggested_target_dim` | `int` | Recommended output dimension (≈ 1.5 × ID) |

---

## Reading the Report

```python
report.print_summary()           # full text overview
report.to_dataframe()            # pandas table for logging / CSV export

# Access sub-results directly
print(report.hubness.k_skewness)
print(report.geometry.isotropy.participation_ratio)
print(report.intrinsic_dim.consensus)
print(report.recommendations)
print(report.severity)           # "low" | "medium" | "high"
```

For detailed interpretation of every metric and its healthy ranges, read
`references/metric_guide.md`.

---

## Pathology → Improvement Quick Reference

| Detected pathology | Metric signal | Recommended improvement |
|---|---|---|
| Severe hubness | `k_skewness > 5` | `AlignUniformLoss` + `KNNPairs` |
| Moderate hubness | `k_skewness 2–5` | `NTXentLoss` + `KNNPairs` |
| High anisotropy | `participation_ratio < 0.1 × D` | `AlignUniformLoss` or PCA whitening first |
| Distance collapse | `concentration_ratio > 0.8` | Reduce `target_dim` aggressively |
| Sparse manifold | `ID/D < 0.1` | Strong bottleneck (low `target_dim`) |
| Low neighbor stability | `mean_consistency < 0.5` | `KNNPairs` (auto-config default) |
| Poor spread / clusters | `uniformity > -1.0` | `AlignUniformLoss` |
| Supervised, class labels available | any severity | `mode="supervised"`, `SupConLoss` |
| Supervised, continuous/regression labels | any severity | `mode="supervised"`, `RankNContrastLoss` |
| Sparse / NLP embeddings | high dim, sparse values | `FeatureDropout` augmentation |
| Locally smooth, pretrained | low severity | `EmbeddingMixup` |

When `augmentation="auto"` and `loss="auto"` (the defaults), `EmbedKit` auto-configures
`KNNPairs(k=5)` + `CombinedLoss(NTXent + 0.5·AlignUniform)` for self-supervised mode,
and `KNNPairs(k=5)` + `CombinedLoss(SupCon + 0.5·AlignUniform)` for supervised mode
(see `api.py:_resolve_loss` and `_resolve_augmentation`).

---

## Improvement API

### High-level (recommended)

```python
from embedkit import EmbedKit

ek = EmbedKit(
    mode="self_supervised",     # or "supervised"
    augmentation="auto",        # or pass an augmentation object
    loss="auto",                # or pass a loss object
    target_dim="auto",          # or int; auto = 1.5 × intrinsic dim
    hidden_dim=256,
    n_layers=2,
    epochs=200,
    batch_size=256,
    lr=3e-4,
    scheduler="cosine",         # "cosine" | "step" | "plateau" | None
    early_stopping_patience=20, # None to disable
    device="cuda",              # or "cpu", or None (auto-detect)
    random_state=42,
)
ek.fit(X, y=y)                  # y=None for self-supervised
X_refined = ek.transform(X)

# or in one call:
X_refined = ek.fit_transform(X, y=y)

# inspect what was auto-chosen:
ek.analysis_report.print_summary()
print(ek._config)
```

### Low-level (full control)

```python
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer
from embedkit.improvement.augmentation import (
    GaussianNoise, FeatureDropout, EmbeddingMixup,
    KNNPairs, FeatureMasking, CompositeAugmentation,
)
from embedkit.improvement.losses import (
    NTXentLoss, AlignUniformLoss, TripletLoss, SupConLoss,
    RankNContrastLoss, CombinedLoss,
)

model = EmbeddingRefiner(
    input_dim=768,
    target_dim=64,
    hidden_dim=256,
    n_layers=2,
    normalize=True,       # L2-normalize output; required for NTXentLoss
)

aug = CompositeAugmentation(
    [GaussianNoise(std=0.05, adaptive=True), FeatureDropout(p=0.1)],
    mode="sequential",    # or "random_choice"
)

loss_fn = CombinedLoss([
    (NTXentLoss(temperature=0.07), 1.0),
    (AlignUniformLoss(alpha=2, t=2), 0.5),
])

trainer = Trainer(
    model=model,
    augmentation=aug,
    loss=loss_fn,
    epochs=200,
    batch_size=256,
    optimizer="adam",     # "adam" | "sgd" | "lars"
    lr=3e-4,
    weight_decay=1e-4,
    scheduler="cosine",
    eval_every=10,        # run EmbedKitAnalyzer on refined embeddings every N epochs
    early_stopping_patience=20,
    device="cuda",
    random_state=42,
)
trainer.fit(X)            # self-supervised; trainer.fit(X, y=y) for supervised
X_refined = trainer.transform(X)
```

#### Augmentation guide

| Class | Best for |
|---|---|
| `GaussianNoise(std, adaptive)` | Robust baseline for most situations |
| `FeatureDropout(p)` | Sparse / over-complete NLP embeddings |
| `EmbeddingMixup(alpha, k)` | Locally smooth, trusted embeddings |
| `KNNPairs(k, hard_negatives)` | Auto-config default; pretrained embeddings with good local structure |
| `FeatureMasking(mask_ratio)` | Structured, correlated embeddings |
| `CompositeAugmentation(augs, mode)` | High-severity or combination strategies |

#### Loss guide

| Class | Mode | Best for |
|---|---|---|
| `NTXentLoss(temperature)` | self-supervised | Default self-supervised choice |
| `AlignUniformLoss(alpha, t)` | self-supervised | Explicit hubness / uniformity control |
| `SupConLoss(temperature)` | supervised | Class labels available |
| `TripletLoss(margin, mining)` | supervised | Complex class boundaries |
| `RankNContrastLoss(temperature, chunk_size)` | supervised | Continuous/regression labels; ranks neighbors by label distance |
| `CombinedLoss([(loss, weight)])` | any | Add uniformity regularization to any base loss |

---

## Visualization

```python
from embedkit.visualization.plots import AnalysisPlotter
from embedkit.visualization.embedding_viz import EmbeddingVisualizer

# ── Analysis report plots ──────────────────────────────────
plotter = AnalysisPlotter(report)
fig = plotter.plot_eigenvalue_spectrum()   # isotropy / PCA spectrum
fig = plotter.plot_distance_histogram()    # pairwise distance distribution
fig = plotter.plot_k_occurrence()          # hubness k-occurrence distribution
fig = plotter.plot_kernel_spectrum()       # RBF kernel eigenvalue spectrum
fig.savefig("spectrum.png")

# ── Before / after 2-D projection ─────────────────────────
viz = EmbeddingVisualizer(method="umap")   # or method="pca"
fig = viz.plot_comparison(X, X_refined, labels=y)
fig.savefig("comparison.png")
```

---

## Persistence

```python
# Save everything (model weights, config, analysis report)
ek.save("my_embedkit_model/")

# Load back
from embedkit import EmbedKit
ek2 = EmbedKit.load("my_embedkit_model/")
X_refined = ek2.transform(X_new)

# Low-level: save / load just the bundle
from embedkit.utils.io import save_bundle, load_bundle
save_bundle("outdir/", model, config_dict, report)
bundle = load_bundle("outdir/")
# bundle keys: "state_dict", "config", "report"
```

---

## ML Workflow Integration

EmbedKit slots into common ML stacks. For detailed patterns with code, read
`references/workflow_guide.md`. Topics covered:

- **HuggingFace transformers** — extract embeddings, run EmbedKit, feed refined
  embeddings to a classifier or retriever
- **PyTorch training loops** — use `EmbeddingRefiner` as a projection head; hook
  `Trainer` eval into your existing loop
- **scikit-learn pipelines** — wrap `EmbedKit` as a `TransformerMixin` for use in
  `Pipeline`
- **RAG / vector store retrieval** — diagnose hub contamination, refine, re-index
- **Production inference** — batch transform with `trainer.transform(X)`; export
  `EmbeddingRefiner` to ONNX

---

## Common Pitfalls

**Don't L2-normalize before analysis.** Run `EmbedKitAnalyzer` on the raw embeddings.
Normalizing first collapses distance information and makes the hubness / isotropy
metrics unreliable.

**`NTXentLoss` requires normalized embeddings.** `EmbeddingRefiner` defaults to
`normalize=True` for this reason. If you swap in a different loss (e.g., `TripletLoss`
with Euclidean distance), pass `normalize=False`.

**Temperature matters.** `NTXentLoss` and `SupConLoss` are sensitive to temperature.
The default `0.07` works well for most cases; go lower (colder) if representations
collapse early in training.

**FAISS-backed kNN.** `embedkit.utils.neighbors.knn()` uses FAISS by default (mandatory dep) with an LRU result cache. If FAISS fails to import at runtime it falls back to sklearn and emits a `RuntimeWarning` — check your `faiss-cpu` install if you see it.

**`target_dim="auto"` sets dim to ≈ 1.5 × intrinsic dimension.** If intrinsic
dimension estimates are unreliable (small n or noisy data), override with an explicit
integer.

**Batch size and in-batch negatives.** `NTXentLoss` uses in-batch negatives — larger
batches give a stronger learning signal. Try `batch_size=512` or higher on GPU.
