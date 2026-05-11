# Embedding Kit

A Python library for analyzing and improving ML embedding spaces. Embedding Kit has two pillars:

- **Analysis** — a suite of geometry-aware metrics that diagnose problems in an embedding space (hubness, distance collapse, anisotropy, intrinsic dimensionality mismatch, etc.) and produce actionable recommendations.
- **Improvement** — a contrastive learning pipeline that trains a small MLP projector to produce geometrically healthier embeddings, guided by the analysis report.

## Features

### Analysis
- **Intrinsic dimension** estimation with 7 estimators: `TwoNN`, `MLE`, `lPCA`, `DANCo`, `CorrInt`, `MOM`, `FisherS`
- **Hubness analysis** — k-occurrence distribution, skewness, hub/antihub ratios
- **Distance concentration** — detects when pairwise distances collapse to a narrow range (curse of dimensionality)
- **Isotropy** — eigenvalue spectrum of the covariance matrix, participation ratio
- **Neighbor consistency** — stability of the kNN graph across perturbations
- **Uniformity score** — how well embeddings fill the unit hypersphere
- **Kernel diagnostics** — RBF kernel rank, spectral gap, and condition number

All estimators follow the sklearn `fit` / `transform` API and accept `np.ndarray` or `torch.Tensor`.

### Improvement
- **`EmbeddingRefiner`** — a PyTorch `nn.Module` (MLP projector with optional L2 normalization)
- **Augmentations**: `GaussianNoise`, `FeatureDropout`, `EmbeddingMixup`, `KNNPairs`, `FeatureMasking`, `CompositeAugmentation`
- **Losses**: `NTXentLoss`, `AlignUniformLoss`, `SupConLoss`, `TripletLoss`, `RankNContrastLoss`, `CombinedLoss`
- **Diagnostics-driven auto-config** — `EmbedKit` reads the analysis report and automatically selects the best loss function, augmentation strategy, and target dimension
- **Self-supervised and supervised** training modes

### Visualization
- `AnalysisPlotter` — eigenvalue spectrum, distance histogram, k-occurrence distribution, kernel spectrum
- `EmbeddingVisualizer` — before/after 2-D UMAP or PCA comparison with class labels

### Ecosystem integration
- Sklearn-compatible (`fit` / `transform`, usable in `Pipeline`)
- Accepts `np.ndarray` and `torch.Tensor` inputs
- FAISS-backed kNN for fast geometry diagnostics on large datasets
- Model save/load with `EmbedKit.save()` / `EmbedKit.load()`

## Installation

### uv

```bash
uv add git+https://github.com/fagonzalezo/embedding-kit.git

# With torchvision (required for image-classification examples):
uv add "embedkit[examples] @ git+https://github.com/fagonzalezo/embedding-kit.git"
```

### pip

```bash
pip install git+https://github.com/fagonzalezo/embedding-kit.git

# With torchvision (required for image-classification examples):
pip install "embedkit[examples] @ git+https://github.com/fagonzalezo/embedding-kit.git"
```

### Optional extras

| Extra | Installs | When to use |
|---|---|---|
| `examples` | `torchvision` | Running image-classification examples (e.g., `examples/04_image_classification.*`) |

## Quickstart

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

X = np.load("embeddings.npy")   # shape (n_samples, n_features)

# Analyze geometry — produces a report with metrics and recommendations
report = EmbedKitAnalyzer(k=10).fit(X)
report.print_summary()
df = report.to_dataframe()      # pandas DataFrame for logging / CSV export

# Self-supervised refinement (auto-configured from analysis report)
X_refined = EmbedKit(mode="self_supervised", epochs=200).fit_transform(X)

# Supervised refinement with class labels
y = np.load("labels.npy")
X_refined = EmbedKit(mode="supervised", epochs=100).fit_transform(X, y=y)
```

See [`examples/`](https://github.com/fagonzalezo/embedding-kit/tree/main/examples) for full usage patterns including low-level API, custom augmentations, and workflow integrations.

## Documentation

| Document | Description |
|----------|-------------|
| [Concepts Manual](https://github.com/fagonzalezo/embedding-kit/blob/main/docs/manual.md) | Full guide to analysis metrics, the improvement pipeline, and visualization |
| [Metric Guide](https://github.com/fagonzalezo/embedding-kit/blob/main/skills/embedkit/references/metric_guide.md) | Per-metric healthy ranges, pathology thresholds, geometric meaning, and downstream consequences |
| [Workflow Guide](https://github.com/fagonzalezo/embedding-kit/blob/main/skills/embedkit/references/workflow_guide.md) | Integration patterns for HuggingFace, PyTorch training loops, scikit-learn pipelines, RAG / vector stores, and ONNX export |
| [Examples](https://github.com/fagonzalezo/embedding-kit/tree/main/examples) | Runnable notebooks |

## Claude Code Skill

EmbedKit ships a [Claude Code](https://claude.ai/code) skill at `skills/embedkit/`. Once installed, Claude automatically applies deep EmbedKit knowledge whenever you ask about embeddings, hubness, isotropy, retrieval quality, or related topics — no manual prompting required.

### Install the skill

**Option 1 — Copy from a cloned repo (global install):**
```bash
git clone https://github.com/fagonzalezo/embedding-kit.git
cp -r embedkit/skills/embedkit ~/.claude/skills/
```

**Option 2 — Symlink (keeps the skill in sync with the repo):**
```bash
git clone https://github.com/fagonzalezo/embedding-kit.git
ln -s "$(pwd)/embedkit/skills/embedkit" ~/.claude/skills/embedkit
```

**Option 3 — Project-level (already works inside this repo):**
The skill is already present at `skills/embedkit/` and Claude Code picks it up automatically when working within the repository.

### What the skill provides

- Quick-start API reference for both the analysis and improvement pipelines
- A pathology → improvement decision table (which loss and augmentation to use for each detected geometric problem)
- Visualization and persistence guides
- Common pitfall warnings (e.g., don't L2-normalize before analysis, temperature sensitivity for NTXentLoss)
- Pointers to the [Metric Guide](https://github.com/fagonzalezo/embedding-kit/blob/main/skills/embedkit/references/metric_guide.md) and [Workflow Guide](https://github.com/fagonzalezo/embedding-kit/blob/main/skills/embedkit/references/workflow_guide.md)
