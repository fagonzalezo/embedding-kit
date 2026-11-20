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

EmbedKit is a Python library at `embedkit/` (project root) that describes the geometry of
embedding spaces, compares transforms against training-free baselines on held-out data, and
trains a contrastive refiner. The benchmarks mostly found that diagnosing an embedding does not
tell you how to fix it, so the high-level `EmbedKit` chooses its output by validation, not by
diagnosis. Say so to the user: do not present a diagnostic reading as a reason to transform.

## What the benchmarks show (use when advising)

- The self-supervised refiner never beat PCA or PCA-whitening (E1, E6, E7:
  `docs/benchmarks/BENCHMARK_E1_E2.md`, `docs/benchmarks/BENCHMARK_E6_E7.md`) and cost about 100 times more.
- On 35 natural classification sets (E4, `benchmarks/results/e4/REPORT.md`), no training-free fix
  beat the raw embedding significantly on more than 16 sets (centring + L2, Euclidean kNN: 16 up,
  1 down, median +0.3 points). PCA to the report's `suggested_target_dim` lost accuracy on 32 of 35;
  ZCA whitening on 34 of 35 (cosine kNN).
- Of 9 diagnostic rules, only two predicted a fix, both for Euclidean kNN: hubness and antihubs.
  The fix is a hubness-reducing neighbour search. Retested on 60 sets (E10), the hubness rule fired
  on 5 and the fix helped on all 5; the antihub rule fired on 22 and helped on 19. The evidence for
  hubness is still small, and no multiplicity correction was applied.
- With class labels (E8, `docs/benchmarks/BENCHMARK_SUPERVISED.md`) the supervised refiner had higher kNN
  accuracy than the raw embeddings on 4 of 6 sets and lower on none (+0.003 to +0.021 on real data,
  +0.185 on a synthetic set). With 5 labels per class LDA was ahead of it by 0.04 to 0.08.
  Regression: the MLP refiner beat raw and PCA on 4 of 4 sets, but with 1,000 rows or fewer no
  refiner reliably beat raw.
- The default selection with labels (E10, `docs/benchmarks/BENCHMARK_SELECTION.md`), on 60 classification and
  4 regression sets: never significantly worse than the input, significantly better on 43 (+0.8 to
  +15.1 points of cosine kNN accuracy), at most 1.2 points below the best candidate in hindsight.
  It chooses by kNN; a linear probe on its output was worse on 5 sets (0.2 to 0.8 points).

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

# ── No labels: centred and L2-normalised, nothing trained ──
Z = EmbedKit().fit_transform(X)

# ── Labels: validated selection, Identity unless significantly beaten ──
y = np.load("labels.npy")       # class labels
ek = EmbedKit(epochs=50).fit(X, y)
Z = ek.transform(X)
print(ek.selected_)             # "identity", "center_l2", "pca-128", "refiner-128", ...
print(ek.selection_report_)     # one row per candidate: score, gain over Identity, 95% interval

# ── Continuous targets (Rank-N-Contrast refiner is a candidate) ──
ek = EmbedKit(mode="regression", epochs=50).fit(X, targets)

# ── One fixed configuration, no selection ──────────────────
Z = EmbedKit(strategy="fixed", target_dim=64).fit_transform(X)                  # PCA to 64, L2
Z = EmbedKit(method="refiner", target_dim=64, epochs=100).fit_transform(X)      # contrastive refiner
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
| `report.calibration` | `dict[str, CalibratedStatistic]` | k-skewness, antihub ratio, distance CV, neighbour consistency, uniformity against a null model: `observed`, `null_mean`, `null_std`, `z_score`, `z_interval`, `uncalibrated`; `None` if `calibrate=False` |
| `report.calibration_k` | `int` | The `k` used by the calibration |
| `report.n_outside_null` | `int` | Calibrated statistics (not flagged `uncalibrated`) with \|z\| > `z_threshold` (3); replaces `severity` |
| `report.properties` | `list[str]` | Descriptive statements (spectrum, mean direction, uniformity vs null); anisotropy is a property, not a recommendation |
| `report.recommendations` | `list[str]` | Actions, Euclidean metric only: more hubness or more antihubs than the null (the two rules E4 kept). Otherwise "No recommendation: ..." |
| `report.k` | `int` | The kNN size the report used (replaces `suggested_k`, removed in 0.7.0) |
| `report.kernel.sigma` | `float` | The RBF bandwidth of the kernel diagnostics (replaces `suggested_sigma`, removed in 0.7.0) |
| `report.suggested_target_dim` | `int` | ≈ 1.5 × TwoNN ID. Descriptive: PCA to it lost accuracy on 32 of 35 E4 sets. Only `EmbedKit(strategy="fixed", target_dim="auto")` reads it |

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

> Experiment E4 tested the report's rules on 35 natural classification sets
> (`benchmarks/results/e4/REPORT.md`). Only the first two rows below are backed by it, and only
> for Euclidean kNN; the report recommends nothing else. Distance concentration, neighbour
> consistency, low ID/D, the mean direction, the top components, anisotropy and dominant
> coordinates did not predict which training-free fix helps, so their statistics are reported
> without an action. PCA to `suggested_target_dim` lost accuracy on 32 of the 35 sets. The
> rows with labels follow E8 (and with few labels per class, LDA or the raw embeddings can be
> better). The last two rows were not benchmarked; do not present them as findings.

| Detected pathology | Metric signal | Recommended improvement |
|---|---|---|
| Hubness (Euclidean) | `calibration["k_skewness"].z_score > 3` with `metric="euclidean"` | Hubness-reducing search (`embedkit.hubness_reduction.MutualProximity`, `CSLS`) |
| Antihubs (Euclidean) | `calibration["antihub_ratio"].z_score > 3` with `metric="euclidean"` | Same |
| Class labels available | labels given | `EmbedKit().fit(X, y)` (selection; the supervised refiner is a candidate, `SupConLoss`) |
| Continuous targets available | targets given | `EmbedKit(mode="regression").fit(X, y)` (`RankNContrastLoss`; needs more than 1,000 rows) |
| Sparse / NLP embeddings | high dim, sparse values | `FeatureDropout` augmentation (not benchmarked) |
| Locally smooth, pretrained | `n_outside_null == 0` | `EmbeddingMixup` (not benchmarked) |

`strategy="select"` (the default) without labels returns `CenterNormalize` (centred, L2); with
labels it fits `Identity`, `CenterNormalize`, PCA at 64/128/256 (+ L2) and, in supervised and
regression mode, the refiner on 80% of the rows, scores each by 10-NN accuracy (or R^2) on the
other 20%, and keeps `Identity` unless a candidate's paired-bootstrap interval of the gain
(95%, 2,000 resamples) is above 0. `strategy="fixed"` (implied by any `method` other than `"auto"`)
builds one configuration: `method="pca"` (the self-supervised `"auto"`; nothing trained) or
`method="refiner"` (the default for supervised and regression modes). With `augmentation="auto"`,
`loss="auto"` and `architecture="auto"`, the refiner uses:

| Mode | Architecture | Augmentation | Loss |
|---|---|---|---|
| `self_supervised` | `linear` | `KNNPairs(k=5)` | `NTXentLoss(0.07)` + 0.5 `AlignUniformLoss` |
| `supervised` | `residual` | `KNNPairs(k=5)` | `SupConLoss(0.07)` |
| `regression` | `mlp` | `GaussianNoise(0.05)` | `RankNContrastLoss()` |

(see `api.py:_resolve_loss`, `_resolve_augmentation`, `_resolve_architecture`). Setting
`augmentation`, `loss`, `nested_dims` or a non-linear `architecture` when no refiner is fitted or
compared (`method="pca"`, or `strategy="select"` without labels or in `mode="self_supervised"`)
raises `ValueError`.

---

## Improvement API

### High-level (recommended)

```python
from embedkit import EmbedKit

ek = EmbedKit(
    mode="auto",                # "auto" (self-supervised without labels, supervised with them)
                                # | "self_supervised" | "supervised" | "regression"
    strategy="select",          # "select" (validated choice) | "fixed" (build `method`)
    method="auto",              # "auto" | "pca" | "refiner"; any value but "auto" implies "fixed"
    augmentation="auto",        # or an augmentation object (refiner only)
    loss="auto",                # or a loss object (refiner only)
    target_dim="auto",          # select: PCA/refiner dim (auto: PCA 64/128/256, refiner min(128, d // 2));
                                # fixed: auto = report.suggested_target_dim, but pass an int
    val_fraction=0.2,           # select: rows held out to score the candidates
    selection_metric="cosine",  # select: "cosine" | "euclidean" kNN
    selection_max_rows=50_000,  # select: candidates fitted and scored on at most this many rows
                                # (stratified subsample); the winner is refitted on all rows;
                                # None = all rows (0.6.0)
    epochs=100,
    batch_size=256,
    lr=3e-4,
    scheduler="cosine",         # "cosine" | "step" | "plateau" | None
    early_stopping_patience="auto",  # int, None to disable; auto = off without labels, 10 with
    device=None,                # "cuda", "mps", "cpu", or None (auto-detect)
    random_state=42,
)
ek.fit(X, y=y)                  # y=None: CenterNormalize; class labels or (mode="regression") targets
Z = ek.transform(X)             # transform(X, dim=m) keeps the first m coordinates, renormalised

# or in one call:
Z = ek.fit_transform(X, y=y)

# inspect what was chosen:
print(ek.selected_)             # winner under "select"; None for a fixed configuration
print(ek.selection_report_)     # DataFrame: candidate, dim, score, diff_vs_identity,
                                # diff_lo, diff_hi, beats_identity, selected
print(ek.strategy_, ek.mode_, ek.method_)
# ek.analysis_report is None under "select"; a fixed configuration runs the report
# uncalibrated (it reads only suggested_target_dim)
```

Rules to remember: continuous targets need `mode="regression"` (with `mode="auto"` non-integer
float labels raise); `mode="self_supervised"` with labels only uses them to choose among
training-free candidates; with fewer than 20 rows to fit or to score (about 100 rows or fewer at the
default `val_fraction`) selection warns and returns `Identity`; `fit` validates `X` (NaN or
infinite values raise, constant columns and duplicate rows warn).

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
    eval_every=10,        # evaluate the monitors in `eval_metrics` every N epochs
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

#### Training-free baselines and evaluation

```python
from embedkit.baselines import Identity, CenterNormalize, PCAReduce   # also Standardize, ZCAWhiten,
from embedkit.evaluation import compare, evaluate_transform          # AllButTheTop, ConceptorNegation,
                                                                      # RandomProjection
splits = dict(X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te, seeds=[0])
res = [evaluate_transform(lambda s: Identity(), method="raw", **splits),
       evaluate_transform(lambda s: PCAReduce(64), method="pca-64", **splits)]
print(compare(res, reference="raw"))    # mean, sd, and the difference with a paired bootstrap interval
```

Always compare a proposed transform with `Identity` (and PCA) on held-out data; a win needs the
95% interval to exclude 0. Hubness-reducing neighbour search (`embedkit.hubness_reduction`:
`LocalScaling`, `MutualProximity`, `CSLS`, `DisSimLocal`) is a secondary distance fitted on a
gallery, not a transform: `MutualProximity(metric="euclidean").fit(Z_gallery).kneighbors(Z_query,
n_neighbors=10)`.

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
# Save everything: config.json, plus model.pt (PCA or refiner output) or arrays.npz (the mean of
# a CenterNormalize result; Identity saves an empty one), and report.json if there is a report
ek.save("my_embedkit_model/")

# Load back
from embedkit import EmbedKit
ek2 = EmbedKit.load("my_embedkit_model/")
X_refined = ek2.transform(X_new)

# Low-level: save / load just the bundle
from embedkit.utils.io import save_bundle, load_bundle
save_bundle("outdir/", model, config_dict, report)   # model=None and arrays={...} for training-free
bundle = load_bundle("outdir/")
# bundle keys: "state_dict" (None without model.pt), "config", "report_dict", "arrays"
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
against a null normalised to the same norm, which is reliable only for flat spectra: the report
marks the antihub ratio, `distance_cv` and uniformity of such rows as uncalibrated
(`calibration[name].uncalibrated`) and draws no recommendation from them.

**`NTXentLoss` requires normalized embeddings.** `EmbeddingRefiner` defaults to
`normalize_output=True` for this reason. If you swap in a different loss (e.g., `TripletLoss`
with Euclidean distance), pass `normalize_output=False`.

**Temperature matters.** `NTXentLoss` and `SupConLoss` are sensitive to temperature.
The default `0.07` works well for most cases; go lower (colder) if representations
collapse early in training.

**kNN backend.** `embedkit.utils.neighbors.knn()` runs an exact scikit-learn search by default and caches nothing. FAISS is optional (`pip install embedkit[faiss]`); select it with `backend="faiss"` or `EMBEDKIT_KNN_BACKEND=faiss`. Both return true distances (Euclidean, or `1 - cos` for cosine), ascending, without the query point.

**`target_dim="auto"` under `strategy="fixed"` is ≈ 1.5 × intrinsic dimension.** No benchmark
supports that factor and PCA to it lost accuracy on 32 of 35 E4 sets. Pass an explicit integer, or
leave the default `strategy="select"`, which compares PCA at 64, 128 and 256 with the input.

**Do not read a diagnostic as a reason to transform.** Only two rules (hubness and antihubs, Euclidean
kNN) predicted a fix in E4, and the first fired on 3 sets. Whitening (ZCA, PCA-whitening) hurt on most
sets. Test any proposed transform on held-out data against `Identity`.

**Small data.** With 1,000 rows or fewer, or 1 to 5 labels per class, no refiner reliably beat the
raw embeddings (E8). With labels, `EmbedKit` selection falls back to `Identity` below about 100 rows.

**Batch size and in-batch negatives.** `NTXentLoss` uses in-batch negatives — larger
batches give a stronger learning signal. Try `batch_size=512` or higher on GPU.
