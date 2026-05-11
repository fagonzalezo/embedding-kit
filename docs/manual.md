# EmbedKit Concepts Manual

EmbedKit is a Python library for analyzing and improving ML embedding spaces. It has two pillars:

- **Analysis** — a suite of geometry-aware metrics that diagnose problems in an embedding space (hubness, distance collapse, anisotropy, etc.) and produce actionable recommendations.
- **Improvement** — a contrastive learning pipeline that trains a small MLP projector to produce geometrically healthier embeddings, guided by the analysis report.

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

X = np.load("embeddings.npy")          # (n_samples, n_features)

# analyze only
report = EmbedKitAnalyzer().fit(X)
report.print_summary()

# analyze + improve in one call
X_refined = EmbedKit(epochs=100).fit_transform(X)
```

---

## Part 1 — Analysis

### 1.1 Intrinsic Dimension

The *intrinsic dimension* (ID) of a dataset is the minimum number of coordinates needed to describe its points without significant information loss. A sentence-embedding model may output 768-dimensional vectors, but those vectors may lie on a manifold whose true dimensionality is closer to 20.

**Why it matters.** Knowing the ID sets a principled lower bound for `target_dim`: projecting below the ID loses information, projecting far above it wastes capacity and amplifies the curse of dimensionality. A large gap between ID and ambient dimension D also signals that the embedding space is poorly utilized.

EmbedKit supports seven estimators, each making different geometric assumptions:

| Method | Approach |
|--------|----------|
| `TwoNN` | Ratio of the two nearest-neighbor distances; fast and robust (Facco et al., 2017) |
| `MLE` | Maximum-likelihood over local neighborhood distances (Levina & Bickel, 2005) |
| `lPCA` | Dimensionality from local PCA scree plots |
| `DANCo` | Combines distance and angular statistics (Ceruti et al., 2012) |
| `CorrInt` | Correlation integral scaling (Grassberger & Procaccia, 1983) |
| `MOM` | Method of moments on nearest-neighbor distances (Amsaleg et al., 2015) |
| `FisherS` | Separability-based estimate using Fisher information (Albergante et al., 2019) |

**Class:** `IntrinsicDimensionEstimator(methods, aggregate, random_state)`

**Result fields:**
- `estimates` — per-method ID values.
- `consensus` — aggregate across methods (mean by default).
- `uncertainty` — standard deviation of per-method estimates.
- `local_estimates` — per-point local ID for methods that support it (TwoNN, MLE, lPCA).

**Ranges & interpretation:**
- `consensus`: positive float in `[1, D]`. A value close to D means the embedding fills the ambient space. Values much smaller than D reveal an under-utilized, low-dimensional manifold. The library flags `ID/D < 0.1` as a strong signal for dimensionality reduction and `ID/D < 0.3` as a moderate signal.
- `uncertainty`: `[0, ∞)`. Near 0 means all methods agree. Large values (> 5) suggest ambiguous geometry; prefer a conservative `target_dim` in that case.

```python
from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

result = IntrinsicDimensionEstimator(methods=["TwoNN", "MLE", "lPCA"]).fit(X)
print(f"Consensus ID: {result.consensus:.1f}  ± {result.uncertainty:.1f}")
print(f"Per-method:   {result.estimates}")
```

---

### 1.2 Hubness

*Hubness* is a curse-of-dimensionality phenomenon in which a handful of points — called **hubs** — repeatedly appear in the k-nearest-neighbor lists of many other points, while **antihubs** never appear as anyone's neighbor at all (Radovanović et al., 2010). This imbalance emerges from the geometry of high-dimensional spaces and is independent of the specific data distribution.

**Why it matters.** Hubness degrades any kNN-based task: retrieval, classification, clustering, and anomaly detection all use nearest-neighbor graphs. Hubs inflate recall for a few points and effectively hide antihubs from retrieval entirely.

**Class:** `HubnessAnalyzer(k, hub_threshold, metric)`

**Result fields:** `k_skewness`, `robinhood_index`, `antihub_ratio`, `hub_ratio`, `k_occurrence`, `hubs`, `antihubs`, `hub_contamination`

**Ranges & interpretation:**

| Metric | Range | Good | Concerning |
|--------|-------|------|------------|
| `k_skewness` | (−∞, ∞) | near 0 | > 2 (moderate), > 5 (severe) |
| `robinhood_index` | [0, 1] | near 0 | > 0.3 |
| `antihub_ratio` | [0, 1] | < 0.05 | > 0.3 |
| `hub_ratio` | [0, 1] | < 0.05 | > 0.1 |
| `hub_contamination` | [0, 1] | near 0 | > 0.3 |

- `k_skewness`: skewness of the k-occurrence distribution N_k (how many times each point is a kNN of someone else). Gaussian distributions have zero skewness; positive skew means a fat tail of highly popular hubs.
- `robinhood_index`: Gini-like inequality measure over N_k values. 0 = every point is equally popular; near 1 = a few points dominate the entire neighbor graph.
- `antihub_ratio`: fraction of points with N_k = 0 (never retrieved). Above 0.3, a significant portion of your dataset is effectively invisible to similarity search.
- `hub_ratio`: fraction of points classified as hubs (N_k > `hub_threshold` × mean N_k). Should be small.
- `hub_contamination`: fraction of a hub's own kNN that are also hubs. High values indicate that hubs cluster together, further compounding retrieval skew.

```python
from embedkit.analysis.hubness import HubnessAnalyzer

result = HubnessAnalyzer(k=10).fit(X)
print(f"k-skewness:     {result.k_skewness:.3f}")
print(f"Robin Hood idx: {result.robinhood_index:.3f}")
print(f"Antihub ratio:  {result.antihub_ratio:.3f}")
```

---

### 1.3 Distance Concentration

In high-dimensional spaces, pairwise Euclidean distances tend to concentrate near their mean — the ratio `(d_max − d_min) / d_mean` shrinks toward zero as dimensionality grows (Beyer et al., 1999). When this happens, all points look equally (dis)similar and nearest-neighbor queries become meaningless.

**Class:** `DistanceConcentration(subsample, n_bins, random_state)`

**Result fields:** `relative_contrast`, `concentration_ratio`, `distance_histogram`, `bin_edges`

**Ranges & interpretation:**

- `relative_contrast` = `(d_max − d_min) / d_mean`: `[0, ∞)`. **Larger is better.** Near 0 means the distance histogram is a narrow spike; high-dimensional random data can push this below 0.1. Values above 1.0 indicate a healthy spread of distances.
- `concentration_ratio` = `d_min / d_max`: `[0, 1]`. Near 0 = healthy (minimum distances are much smaller than maximum). Near 1 = severe concentration (every pair of points is nearly the same distance apart). The library issues a warning above 0.8.

```python
from embedkit.analysis.geometry import DistanceConcentration

result = DistanceConcentration().fit(X)
print(f"Relative contrast:  {result.relative_contrast:.3f}")
print(f"Concentration ratio:{result.concentration_ratio:.3f}")
```

---

### 1.4 Isotropy

An *isotropic* embedding space uses all directions equally — variance is spread evenly across every dimension. An *anisotropic* space concentrates most variance in a small number of principal directions, leaving the rest unused.

**Why it matters.** Anisotropy often co-occurs with hubness (a few dominant directions create hub structure). It also wastes capacity: if 700 of 768 dimensions carry negligible variance, those dimensions add noise but no signal. Studies of language model embeddings have shown that strong anisotropy is common in pre-trained representations and degrades downstream tasks (Mu et al., 2018).

**Class:** `IsotropyAnalyzer()`

**Result fields:** `isotropy_score`, `participation_ratio`, `effective_rank`, `eigenvalue_spectrum`, `explained_variance_ratio`

**Ranges & interpretation:**

- `isotropy_score` = `effective_rank / D`: `[0, 1]` (clipped). **1 = perfectly isotropic** (all directions used equally); near 0 = the space has effectively collapsed to a single direction.
- `participation_ratio` = `(Σλᵢ)² / Σλᵢ²`: `[1, D]`. Intuitively, the number of eigenvalue-weighted dimensions that contribute to the space. Near D = isotropic. The library flags `PR < 0.1 × D` as highly anisotropic and `PR < 0.3 × D` as moderately anisotropic.
- `effective_rank` = `exp(H)` where H is the entropy of the normalized eigenvalue spectrum: `[1, D]`. Measures how many "effective" principal components are needed to describe the space.

```python
from embedkit.analysis.geometry import IsotropyAnalyzer

result = IsotropyAnalyzer().fit(X)
print(f"Isotropy score:    {result.isotropy_score:.3f}")
print(f"Participation ratio: {result.participation_ratio:.1f} / {X.shape[1]}")
print(f"Effective rank:    {result.effective_rank:.1f}")
```

---

### 1.5 Neighbor Consistency

*Neighbor consistency* measures how stable the k-NN graph is under small Gaussian perturbations. For each point, a perturbed copy of the dataset is generated with noise scaled to `noise_std × std(X)`, and the fraction of original neighbors that survive in the perturbed graph is recorded.

**Why it matters.** An unstable kNN graph means that tiny changes in input (e.g., measurement noise, model updates) can completely reorganize the neighborhood structure. Downstream tasks that rely on kNN will be unreliable in such spaces.

**Class:** `NeighborConsistency(k, n_perturbations, noise_std, metric, random_state)`

**Result fields:** `mean_consistency`, `std_consistency`, `per_perturbation`

**Ranges & interpretation:**
- `mean_consistency`: `[0, 1]`. Fraction of original k-neighbors that are preserved after perturbation, averaged over all points and all perturbation trials. Near 1 = robust neighborhood structure. Values below 0.5 indicate that the embedding lacks local stability; kNN-based tasks will be fragile.

```python
from embedkit.analysis.geometry import NeighborConsistency

result = NeighborConsistency(k=10, n_perturbations=5).fit(X)
print(f"Mean consistency: {result.mean_consistency:.3f}  ± {result.std_consistency:.3f}")
```

---

### 1.6 Uniformity

The *uniformity* metric (Wang & Isola, 2020) measures how evenly normalized embeddings are distributed on the unit hypersphere. A uniform distribution means the space is well-utilized; a non-uniform one means embeddings cluster in pockets.

**Why it matters.** Representation collapse — where all embeddings converge to the same point — shows up as extreme non-uniformity. Uniformity is also directly optimized by `AlignUniformLoss`.

**Class:** `UniformityScore(t, subsample, random_state)`

**Result field:** `uniformity`

**Ranges & interpretation:**
- `uniformity` = `log E[exp(−t ‖zᵢ − zⱼ‖²)]` for L2-normalized vectors: always **negative** (the expectation is strictly less than 1 for any non-degenerate distribution). **More negative = more uniform** (better coverage of the sphere). Near 0 means embeddings are tightly clustered or collapsed. The library flags values above −1.0 as poor uniformity.

```python
from embedkit.analysis.geometry import UniformityScore

result = UniformityScore(t=2.0).fit(X)
print(f"Uniformity: {result.uniformity:.4f}")
# good: around -3 to -5; poor: above -1
```

---

### 1.7 Kernel Diagnostics

The RBF (Gaussian) kernel matrix `K(xᵢ, xⱼ) = exp(−‖xᵢ − xⱼ‖² / 2σ²)` encodes a non-linear similarity structure. Analyzing its eigenspectrum reveals whether the embeddings are well-separated and numerically stable for kernel-based methods (SVMs, kernel PCA, MMD tests).

**Class:** `KernelDiagnostics(sigma, n_components, subsample, random_state)`

When `sigma="median"` (default), σ is set to the median pairwise distance — a standard heuristic.

**Result fields:** `effective_rank`, `spectral_gap`, `condition_number`, `row_sum_skewness`, `eigenvalues`, `kernel_alignment`, `sigma`

**Ranges & interpretation:**

| Metric | Range | Good | Concerning |
|--------|-------|------|------------|
| `effective_rank` | [1, subsample_size] | high | low (kernel dominated by 1 component) |
| `spectral_gap` | [0, 1] | large | near 0 (no clear cluster structure) |
| `condition_number` | [1, ∞) | near 1 | > 1e6 (numerically unstable) |
| `kernel_alignment` | [0, 1] or None | near 1 | near 0 (kernel unrelated to labels) |

- `effective_rank`: measures how many eigenvalues carry non-trivial weight. High effective rank = the kernel captures diverse structure in the data.
- `spectral_gap`: ratio between adjacent leading eigenvalues. A large gap after the first k eigenvalues indicates k well-separated clusters that the kernel can discriminate.
- `condition_number`: ratio of largest to smallest eigenvalue. Large values indicate numerical ill-conditioning that will destabilize kernel solvers and gradient computations.
- `kernel_alignment`: populated only when labels `y` are passed to `fit(X, y)`. Measures alignment between the data kernel matrix and the ideal label kernel (Cristianini et al., 2001); near 1 means the RBF geometry reflects the class structure.

```python
from embedkit.analysis.kernel import KernelDiagnostics

result = KernelDiagnostics(sigma="median").fit(X)
print(f"Effective rank:   {result.effective_rank:.1f}")
print(f"Spectral gap:     {result.spectral_gap:.3f}")
print(f"Condition number: {result.condition_number:.2e}")
```

---

### 1.8 Unified Analysis Report

`EmbedKitAnalyzer` runs all seven analyzers in a single call and assembles the results into an `EmbedKitReport`. The report also computes a **severity level** and a list of **recommendations** that guide the improvement module.

**Severity scoring.** Each problematic metric adds points to a cumulative score (hubness, ID/D ratio, anisotropy, distance concentration, neighbor consistency, and uniformity each contribute 1–2 points). Score ≥ 4 → `"high"`, ≥ 2 → `"medium"`, else `"low"`.

**Auto-configuration outputs.** The report exposes three suggested hyperparameters consumed by `EmbedKit`:

| Field | Meaning |
|-------|---------|
| `suggested_k` | kNN neighborhood size, set to `max(5, √n / 2)` |
| `suggested_sigma` | RBF bandwidth from median-distance heuristic |
| `suggested_target_dim` | `clip(1.5 × ID, ID, D)` — a dimensionality target that preserves the manifold with headroom |

**Class:** `EmbedKitAnalyzer(k, id_methods, metric, random_state)`

```python
from embedkit import EmbedKitAnalyzer

report = EmbedKitAnalyzer(k=10, id_methods=["TwoNN", "MLE", "lPCA"]).fit(X)
report.print_summary()

# access individual sub-results
print(report.hubness.k_skewness)
print(report.geometry.isotropy.isotropy_score)
print(report.severity)            # "low" | "medium" | "high"
print(report.recommendations)     # list of strings
print(report.suggested_target_dim)

# tabular view
df = report.to_dataframe()
```

---

## Part 2 — Improvement

### 2.1 EmbeddingRefiner

`EmbeddingRefiner` is a small feed-forward network that maps original embeddings of dimension D to a geometrically improved space of dimension `target_dim`. It is the learnable component trained by the contrastive objective.

**Architecture:** `n_layers` repetitions of `Linear → BatchNorm → ReLU → Dropout`, followed by a final `Linear(hidden_dim, target_dim)`. When `normalize_output=True` (default), the output is L2-normalized to the unit sphere, which is required by cosine-similarity-based losses.

**Class:** `EmbeddingRefiner(input_dim, target_dim, hidden_dim=256, n_layers=2, dropout=0.1, normalize_output=True)`

```python
from embedkit.improvement.model import EmbeddingRefiner
import torch

model = EmbeddingRefiner(input_dim=128, target_dim=32)
z = model(torch.randn(64, 128))   # (64, 32), L2-normalized
```

---

### 2.2 Augmentations

Contrastive learning requires *positive pairs*: two different views of the same embedding that should be close in the refined space. Each augmentation in EmbedKit transforms a batch tensor `x` into a tuple `(x_i, x_j)` — two perturbed versions of the same batch. The loss then pulls `z_i = f(x_i)` toward `z_j = f(x_j)` while pushing them away from other points in the batch.

**Base class:** `BaseAugmentation.__call__(x: Tensor) → (Tensor, Tensor)`

---

#### GaussianNoise

Adds independent isotropic Gaussian noise to each embedding, producing two noisy views.

When `adaptive=True`, the noise standard deviation is scaled per-point by the mean distance to its k-nearest neighbors. This adapts the perturbation magnitude to the local density: tight clusters get smaller noise, sparse regions get larger noise.

**Parameters:** `std=0.05`, `adaptive=False`, `k=10`

```python
from embedkit.improvement.augmentation import GaussianNoise

aug = GaussianNoise(std=0.05, adaptive=True)
x_i, x_j = aug(x_batch)
```

---

#### FeatureDropout

Independently zeroes out random dimensions in each view. Two independent masks are sampled, so `x_i` and `x_j` lose different dimensions. This forces the model to learn redundant representations that do not depend on any single feature.

**Parameters:** `p=0.1` (probability of dropping each dimension)

```python
from embedkit.improvement.augmentation import FeatureDropout

aug = FeatureDropout(p=0.15)
x_i, x_j = aug(x_batch)
```

---

#### EmbeddingMixup

Interpolates each embedding with a randomly selected k-nearest neighbor using a Beta-distributed mixing coefficient λ (Zhang et al., 2018):

```
x_aug = λ · x + (1 − λ) · x_neighbor
```

Because the neighbor lies on the same manifold, the interpolated point is geometrically meaningful — unlike random noise in ambient space. Two independent mixing coefficients are drawn, producing two views.

**Parameters:** `k=10`, `alpha=0.4` (Beta distribution concentration)

```python
from embedkit.improvement.augmentation import EmbeddingMixup

aug = EmbeddingMixup(k=10, alpha=0.4)
x_i, x_j = aug(x_batch)
```

---

#### KNNPairs

Instead of perturbing `x` to create a view, `KNNPairs` uses an actual nearest neighbor as the positive partner. The anchor `x` and its closest neighbor `x_pos` form the pair `(x, x_pos)`. This augmentation carries the strongest geometric signal: the network explicitly learns that a point and its neighbors should map to similar representations.

**Parameters:** `k=10`, `hard_negatives=False`

```python
from embedkit.improvement.augmentation import KNNPairs

aug = KNNPairs(k=10)
anchor, positive = aug(x_batch)
```

---

#### FeatureMasking

Masks contiguous blocks of dimensions to zero, inspired by CutOut (DeVries & Taylor, 2017). Two independently sampled masks are applied to produce the pair. This forces the network to build representations that are robust to missing sub-sequences of features — useful for structured embeddings (e.g., token-level representations concatenated in a fixed order).

**Parameters:** `mask_ratio=0.15`, `block_size=1`

```python
from embedkit.improvement.augmentation import FeatureMasking

aug = FeatureMasking(mask_ratio=0.2, block_size=4)
x_i, x_j = aug(x_batch)
```

---

#### CompositeAugmentation

Chains or randomly selects from multiple augmentations.

- `mode="sequential"` — applies every augmentation in order, threading the views through each one. The final pair has accumulated all transformations.
- `mode="random_choice"` — at each training step, picks one augmentation at random from the list.

```python
from embedkit.improvement.augmentation import CompositeAugmentation, GaussianNoise, FeatureDropout

aug = CompositeAugmentation(
    [GaussianNoise(std=0.05), FeatureDropout(p=0.1)],
    mode="sequential",
)
x_i, x_j = aug(x_batch)
```

---

### 2.3 Loss Functions

All losses inherit from `BaseLoss(nn.Module)` and implement:

```python
forward(z_i: Tensor, z_j: Tensor, labels: Tensor | None = None) -> Tensor
```

where `z_i` and `z_j` are the projected views of the positive pair (typically L2-normalized).

---

#### NTXentLoss — Normalized Temperature-Scaled Cross-Entropy

The SimCLR / InfoNCE objective (van den Oord et al., 2018; Chen et al., 2020). Given a batch of N positive pairs, the 2N representations are assembled into a matrix. For each anchor, the corresponding view is the positive example and all other 2N − 2 representations are in-batch negatives. The loss is a softmax cross-entropy over cosine similarities scaled by temperature τ.

Lower τ concentrates the gradient on hard negatives; higher τ produces a softer distribution.

**Parameters:** `temperature=0.07`

**Use when:** self-supervised learning with no labels available.

```python
from embedkit.improvement.losses import NTXentLoss

loss_fn = NTXentLoss(temperature=0.07)
loss = loss_fn(z_i, z_j)
```

---

#### AlignUniformLoss

Directly optimizes the two properties identified by Wang & Isola (2020) as sufficient for a good representation space:

- **Alignment** — positive pairs should map to nearby points: `E[‖zᵢ − zⱼ‖^α]`.
- **Uniformity** — all representations should spread uniformly over the sphere: `log E[exp(−t ‖zᵢ − zⱼ‖²)]`.

The loss is the sum of both terms. The uniformity term is the same quantity measured by `UniformityScore`, so this loss directly optimizes for one of the analysis metrics.

**Parameters:** `alpha=2.0`, `t=2.0`

**Use when:** uniformity analysis shows poor coverage of the sphere (score > −1.0), or as a complement to NTXentLoss via `CombinedLoss`.

```python
from embedkit.improvement.losses import AlignUniformLoss

loss_fn = AlignUniformLoss(alpha=2.0, t=2.0)
loss = loss_fn(z_i, z_j)
```

---

#### TripletLoss

Classic metric learning objective (Schroff et al., 2015). Given an anchor, a positive (same class), and a negative (different class), the loss penalizes configurations where `d(anchor, positive) − d(anchor, negative) + margin > 0`.

Supports three mining strategies:
- `"hard"` — use the hardest positive (farthest same-class point) and hardest negative (closest different-class point). Aggressive; risks training instability.
- `"semi-hard"` — use the closest negative that is still farther than the positive. Balances difficulty and stability.
- `"easy"` — average over all positives and negatives. Stable but slow convergence.

**Parameters:** `margin=0.5`, `mining="hard"`, `distance="euclidean"` | `"cosine"`

**Requires labels.**

```python
from embedkit.improvement.losses import TripletLoss

loss_fn = TripletLoss(margin=0.5, mining="semi-hard")
loss = loss_fn(z_i, z_j, labels=y_batch)
```

---

#### SupConLoss — Supervised Contrastive Loss

Extension of NTXentLoss to the supervised setting (Khosla et al., 2020). Rather than treating only the augmented pair as the sole positive, all samples from the same class are positives for each anchor. This allows the loss to use multiple positives per anchor, providing a richer learning signal when labels are available.

Falls back to the NTXentLoss behavior if no labels are provided.

**Parameters:** `temperature=0.07`

**Requires labels** for full benefit.

```python
from embedkit.improvement.losses import SupConLoss

loss_fn = SupConLoss(temperature=0.07)
loss = loss_fn(z_i, z_j, labels=y_batch)
```

---

#### RankNContrastLoss — Rank-N-Contrast for Regression

Supervised contrastive loss for continuous or multi-dimensional labels (Zha et al., 2023). Rather than defining positives by class membership, it imposes a ranking constraint: for each anchor, samples *closer* in label space must produce *higher* cosine similarity in the embedding space than samples farther away. The ranking is distribution-free — no kernel bandwidth to tune.

Supports both scalar regression targets (shape `(n,)`) and vector targets (shape `(n, k)` for multi-output regression).

**Parameters:** `temperature=0.07`, `chunk_size=None`

- `chunk_size`: when set, the (2N × 2N) similarity matrix is computed in row-blocks of this size, reducing peak GPU memory at the cost of slightly more compute.

**Requires labels** (raises `ValueError` otherwise).

**Use when:** supervised refinement with continuous targets such as regression scores, ratings, or multi-dimensional attribute vectors — where class-boundary losses (`SupConLoss`, `TripletLoss`) cannot apply.

```python
from embedkit.improvement.losses import RankNContrastLoss

loss_fn = RankNContrastLoss(temperature=0.07)
loss = loss_fn(z_i, z_j, labels=y_batch)   # y_batch: (N,) or (N, k)

# memory-efficient variant for large batches:
loss_fn = RankNContrastLoss(temperature=0.07, chunk_size=128)
```

---

#### CombinedLoss

Weighted sum of multiple loss functions. Useful for combining complementary objectives, e.g., a classification-aware loss (SupConLoss) with a geometry-aware term (AlignUniformLoss).

```python
from embedkit.improvement.losses import CombinedLoss, NTXentLoss, AlignUniformLoss

loss_fn = CombinedLoss([
    (NTXentLoss(temperature=0.07), 1.0),
    (AlignUniformLoss(), 0.5),
])
loss = loss_fn(z_i, z_j)
```

---

### 2.4 Trainer

`Trainer` orchestrates the full contrastive training loop. It handles batching, optimization, learning rate scheduling, periodic evaluation against analysis metrics, early stopping, and loss history tracking.

**Class:** `Trainer(model, augmentation, loss, epochs, batch_size, optimizer, lr, weight_decay, scheduler, warmup_epochs, eval_every, eval_metrics, early_stopping_patience, monitor, device, random_state)`

**Key parameters:**

| Parameter | Options | Notes |
|-----------|---------|-------|
| `optimizer` | `"adam"`, `"sgd"`, `"lars"` | Adam is a safe default |
| `scheduler` | `"cosine"`, `"step"`, `"plateau"`, `None` | Cosine annealing with warmup is recommended |
| `eval_metrics` | `["uniformity", "isotropy", "k_skewness"]` | Metrics tracked during training |
| `monitor` | any `eval_metrics` key | Metric watched for early stopping |
| `early_stopping_patience` | int or `None` | Stops after N evaluations without improvement |

**Workflow:**

```python
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.augmentation import GaussianNoise
from embedkit.improvement.losses import NTXentLoss
from embedkit.improvement.trainer import Trainer

model = EmbeddingRefiner(input_dim=128, target_dim=32)
aug   = GaussianNoise(std=0.05, adaptive=True)
loss  = NTXentLoss(temperature=0.07)

trainer = Trainer(
    model=model,
    augmentation=aug,
    loss=loss,
    epochs=200,
    batch_size=256,
    optimizer="adam",
    lr=3e-4,
    scheduler="cosine",
    eval_every=10,
    eval_metrics=["uniformity", "k_skewness"],
    early_stopping_patience=5,
)

trainer.fit(X)              # self-supervised
# trainer.fit(X, y=labels)  # supervised

X_refined = trainer.transform(X)   # returns np.ndarray
print(trainer.history)             # {"loss": [...], "uniformity": [...], ...}
```

---

### 2.5 High-Level EmbedKit API

`EmbedKit` is the one-stop end-to-end pipeline: it analyzes the input embeddings, auto-configures the augmentation, loss function, and `target_dim` based on the analysis report, builds the model, and trains it — all in a single `fit()` call.

**Auto-configuration logic:**

| Condition | Effect |
|-----------|--------|
| `augmentation == "auto"` | Uses `EmbeddingMixup(k=report.suggested_k, alpha=0.4)` regardless of severity |
| `mode == "self_supervised"` | Base loss is `NTXentLoss` |
| `mode == "supervised"` | Base loss is `SupConLoss` |
| `k_skewness > 5` | Adds `AlignUniformLoss` (weight 0.5) via `CombinedLoss` |
| `target_dim == "auto"` | Uses `report.suggested_target_dim` = `clip(1.5 × ID, ID, D)` |

**Class:** `EmbedKit(mode, augmentation, loss, target_dim, hidden_dim, n_layers, epochs, batch_size, optimizer, lr, scheduler, eval_every, early_stopping_patience, device, id_methods, random_state)`

```python
from embedkit import EmbedKit

# self-supervised (no labels needed)
ek = EmbedKit(mode="self_supervised", epochs=100, target_dim="auto")
X_refined = ek.fit_transform(X)

# supervised
ek = EmbedKit(mode="supervised", epochs=100)
X_refined = ek.fit_transform(X, y=labels)

# inspect what was auto-configured
print(ek.analysis_report.severity)
print(ek._config)

# save and reload
ek.save("my_model/")
ek2 = EmbedKit.load("my_model/")
X_new_refined = ek2.transform(X_new)
```

---

## Part 3 — Visualization

### AnalysisPlotter

Renders analysis metrics for a fitted `EmbedKitReport`.

```python
from embedkit.visualization.plots import AnalysisPlotter

plotter = AnalysisPlotter(report)
fig = plotter.plot_full_report()       # grid of all sub-plots
fig = plotter.plot_eigenvalue_spectrum()
fig = plotter.plot_distance_histogram()
fig = plotter.plot_k_occurrence()
fig = plotter.plot_kernel_spectrum()
fig = plotter.plot_id_local_map(X)     # spatial map of local ID estimates
fig = plotter.plot_hubness_map(X)      # spatial map of hub scores
```

### EmbeddingVisualizer

Projects embeddings to 2-D using PCA, UMAP, or t-SNE for visual inspection.

```python
from embedkit.visualization.embedding_viz import EmbeddingVisualizer

viz = EmbeddingVisualizer(method="pca", random_state=42)
fig = viz.plot_comparison(X, X_refined, labels=y)  # before vs. after
fig = viz.plot_training_trajectory(trainer.history)  # loss + metrics over epochs
fig = viz.plot_knn_graph(X, k=10, highlight_hubs=True)
```

---

## References

Albergante, L., Bac, J., & Zinovyev, A. (2019). Estimating the effective dimension of large biological datasets using Fisher separability analysis. In *2019 International Joint Conference on Neural Networks (IJCNN)*. IEEE.

Amsaleg, L., Chelly, O., Furon, T., Girard, S., Houle, M. E., Kawarabayashi, K., & Nett, M. (2015). Estimating local intrinsic dimensionality. In *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 29–38). ACM.

Beyer, K. S., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? In *Proceedings of the 7th International Conference on Database Theory (ICDT)* (pp. 217–235). Springer.

Ceruti, C., Bassis, S., Rozza, A., Lombardi, G., Casiraghi, E., & Campadelli, P. (2012). DANCo: An intrinsic dimensionality estimator exploiting angle and norm concentration. In *Proceedings of the 21st International Conference on Pattern Recognition (ICPR 2012)* (pp. 1045–1048). IEEE.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119.

Cristianini, N., Shawe-Taylor, J., Elisseeff, A., & Kandola, J. S. (2001). On kernel-target alignment. In *Advances in Neural Information Processing Systems* (Vol. 14, pp. 367–373). MIT Press.

DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with Cutout. *arXiv preprint arXiv:1708.04552*.

Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7, 12140.

Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of strange attractors. *Physica D: Nonlinear Phenomena*, 9(1–2), 189–208.

Khosla, P., Tian, Y., Wang, X., Liu, C., Isola, P., & others (2020). Supervised contrastive learning. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 18661–18673). Curran Associates.

Levina, E., & Bickel, P. J. (2005). Maximum likelihood estimation of intrinsic dimension. In *Advances in Neural Information Processing Systems* (Vol. 17, pp. 777–784). MIT Press.

Mu, J., Bhat, S., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. In *International Conference on Learning Representations (ICLR 2018)*.

Radovanović, M., Nanopoulos, A., & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research*, 11, 2487–2531.

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015)* (pp. 815–823). IEEE.

van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

Wang, T., & Isola, P. (2020). Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119.

Zha, K., Cao, P., Son, J., Yang, Y., & Katabi, D. (2023). Rank-N-Contrast: Learning continuous representations for regression. In *Advances in Neural Information Processing Systems* (Vol. 36). Curran Associates.

Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. In *International Conference on Learning Representations (ICLR 2018)*.
