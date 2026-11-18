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

# ── Self-supervised: PCA at the suggested dim + L2 norm (no training) ──
X_reduced = EmbedKit(mode="self_supervised").fit_transform(X)
# opt-in contrastive refiner:
X_refined = EmbedKit(mode="self_supervised", method="refiner", epochs=100).fit_transform(X)

# ── Supervised refinement with class labels ────────────────
y = np.load("labels.npy")       # integer class indices
X_refined = EmbedKit(mode="supervised", epochs=100).fit_transform(X, y=y)

# ── Regression targets (Rank-N-Contrast) ───────────────────
X_refined = EmbedKit(mode="regression", epochs=100).fit_transform(X, y=targets)
```

---

## Analysis API

```python
from embedkit import EmbedKitAnalyzer

analyzer = EmbedKitAnalyzer(
    k=10,                                    # kNN size; auto = max(5, round(sqrt(n)/2)) if None
    id_methods=["TwoNN", "MLE", "lPCA", "MOM"],  # ID estimators (default)
    metric="euclidean",                       # "euclidean" or "cosine" (normalised rows)
    calibrate=True,                           # compare 5 statistics with a null model (z-scores)
    random_state=42,
)
report = analyzer.fit(X, y=None)    # y enables centered_kernel_alignment
```

`EmbedKitAnalyzer.fit()` runs all diagnostics in one pass and returns an
`EmbedKitReport` with these sub-results:

| Attribute | Type | What it holds |
|---|---|---|
| `report.intrinsic_dim` | `IntrinsicDimensionResult` | Per-method ID estimates with bootstrap intervals, TwoNN scale curve, reference estimate |
| `report.hubness` | `HubnessResult` | k-occurrence distribution, skewness, hub/antihub ratios |
| `report.geometry` | `GeometryBundle` | Distance concentration, isotropy, neighbor consistency, uniformity |
| `report.kernel` | `KernelDiagnosticsResult` | Centred RBF kernel: effective rank, spectral gap, `top_k_eigen_ratio` (`condition_number` is a deprecated alias), kernel alignment |
| `report.calibration` | `dict[str, CalibratedStatistic]` | k-skewness, antihub ratio, distance CV, neighbour consistency, uniformity against a null model: `observed`, `null_mean`, `null_std`, `z_score`; `None` if `calibrate=False` |
| `report.calibration_k` | `int` | The `k` used by the calibration |
| `report.n_outside_null` | `int` | Calibrated statistics with \|z\| > `z_threshold` (3); replaces `severity` |
| `report.properties` | `list[str]` | Descriptive statements (spectrum, mean direction, uniformity vs null); anisotropy is a property, not a recommendation |
| `report.recommendations` | `list[str]` | Actions, Euclidean metric only: more hubness or more antihubs than the null (the two rules E4 kept) |
| `report.suggested_k` | `int` | Recommended kNN size |
| `report.suggested_sigma` | `float` | Recommended RBF sigma |
| `report.suggested_target_dim` | `int` | Recommended output dimension (≈ 1.5 × TwoNN ID) |

---

## Reading the Report

```python
report.print_summary()           # full text overview
report.to_dataframe()            # pandas table for logging / CSV export

# Access sub-results directly
print(report.hubness.k_skewness)
print(report.geometry.isotropy.participation_ratio)
print(report.intrinsic_dim.estimates)    # per method; no consensus
print(report.calibration["k_skewness"].z_score)   # against the null model
print(report.recommendations)
print(report.n_outside_null)     # calibrated statistics with |z| > 3
```

For detailed interpretation of every metric and its healthy ranges, read
`references/metric_guide.md`.

---

## Pathology → Improvement Quick Reference

> Experiment E4 (WP 7.1) tested the report's rules on 35 natural classification sets
> (`benchmarks/results/e4/REPORT.md`). Only the first two rows below are backed by it, and only
> for Euclidean kNN; the report recommends nothing else. Distance concentration, neighbour
> consistency, low ID/D, the mean direction, the top components, anisotropy and dominant
> coordinates did not predict which training-free fix helps, so their statistics are reported
> without an action. PCA to `suggested_target_dim` lost accuracy on 32 of the 35 sets. The
> remaining rows are untested suggestions.

| Detected pathology | Metric signal | Recommended improvement |
|---|---|---|
| Hubness (Euclidean) | `calibration["k_skewness"].z_score > 3` with `metric="euclidean"` | Hubness-reducing search (`embedkit.hubness_reduction.MutualProximity`, `CSLS`) |
| Antihubs (Euclidean) | `calibration["antihub_ratio"].z_score > 3` with `metric="euclidean"` | Same |
| Supervised, class labels available | labels given | `mode="supervised"`, `SupConLoss` |
| Supervised, continuous/regression labels | targets given | `mode="regression"`, `RankNContrastLoss` |
| Sparse / NLP embeddings | high dim, sparse values | `FeatureDropout` augmentation |
| Locally smooth, pretrained | `n_outside_null == 0` | `EmbeddingMixup` |

With `method="auto"`, self-supervised mode trains nothing (PCA at `target_dim`, L2-normalised); the
trained refiner is `method="refiner"`, and the default for supervised and regression modes. With
`augmentation="auto"`, `loss="auto"` and `architecture="auto"`, the refiner uses:

| Mode | Architecture | Augmentation | Loss |
|---|---|---|---|
| `self_supervised` | `linear` | `KNNPairs(k=5)` | `NTXentLoss(0.07)` + 0.5 `AlignUniformLoss` |
| `supervised` | `residual` | `KNNPairs(k=5)` | `SupConLoss(0.07)` |
| `regression` | `mlp` | `GaussianNoise(0.05)` | `RankNContrastLoss()` |

(see `api.py:_resolve_loss`, `_resolve_augmentation`, `_resolve_architecture`). Setting
`augmentation`, `loss`, `nested_dims` or a non-linear `architecture` with `method="pca"` raises.

---

## Improvement API

### High-level (recommended)

```python
from embedkit import EmbedKit

ek = EmbedKit(
    mode="self_supervised",     # or "supervised" | "regression"
    method="auto",              # "auto" | "pca" | "refiner"; auto = pca without labels
    augmentation="auto",        # or pass an augmentation object (method="refiner" only)
    loss="auto",                # or pass a loss object
    target_dim="auto",          # or int; auto = 1.5 × TwoNN intrinsic dim
    hidden_dim=256,
    n_layers=2,
    epochs=200,
    batch_size=256,
    lr=3e-4,
    scheduler="cosine",         # "cosine" | "step" | "plateau" | None
    early_stopping_patience="auto",  # int, None to disable; auto = off without labels
    device="cuda",              # or "cpu", or None (auto-detect)
    random_state=42,
)
ek.fit(X, y=y)                  # y=None for self-supervised; targets for regression
X_refined = ek.transform(X)

# or in one call:
X_refined = ek.fit_transform(X, y=y)

# inspect what was auto-chosen:
ek.analysis_report.print_summary()   # EmbedKit runs the report uncalibrated
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
    normalize_output=True,  # L2-normalize output; required for NTXentLoss
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
    optimizer="adam",     # "adam" | "sgd"
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
| `GaussianNoise(std, adaptive, relative)` | Robust baseline for most situations |
| `FeatureDropout(p)` | Sparse / over-complete NLP embeddings |
| `EmbeddingMixup(alpha, k)` | Locally smooth, trusted embeddings |
| `KNNPairs(k, neighbor_index, metric)` | Auto-config default; pretrained embeddings with good local structure |
| `FeatureMasking(mask_ratio)` | Structured, correlated embeddings |
| `CompositeAugmentation(augs, mode)` | Combination strategies |

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
# bundle keys: "state_dict", "config", "report_dict"
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

**Don't L2-normalize before analysis.** Run `EmbedKitAnalyzer` on the raw embeddings. If you
search by cosine, pass `metric="cosine"`: the analyzer normalises the rows itself and every
statistic then describes the normalised rows. Rows already normalised upstream are calibrated
against a null normalised to the same norm, which is reliable only for flat spectra.

**`NTXentLoss` requires normalized embeddings.** `EmbeddingRefiner` defaults to
`normalize_output=True` for this reason. If you swap in a different loss (e.g., `TripletLoss`
with Euclidean distance), pass `normalize_output=False`.

**Temperature matters.** `NTXentLoss` and `SupConLoss` are sensitive to temperature.
The default `0.07` works well for most cases; go lower (colder) if representations
collapse early in training.

**kNN backend.** `embedkit.utils.neighbors.knn()` runs an exact scikit-learn search by default and caches nothing. FAISS is optional (`pip install embedkit[faiss]`); select it with `backend="faiss"` or `EMBEDKIT_KNN_BACKEND=faiss`. Both return true distances (Euclidean, or `1 - cos` for cosine), ascending, without the query point.

**`target_dim="auto"` sets dim to ≈ 1.5 × intrinsic dimension.** If intrinsic
dimension estimates are unreliable (small n or noisy data), override with an explicit
integer.

**Batch size and in-batch negatives.** `NTXentLoss` uses in-batch negatives — larger
batches give a stronger learning signal. Try `batch_size=512` or higher on GPU.
