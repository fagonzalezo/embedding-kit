# EmbedKit Metric Guide

Detailed interpretation of every metric in `EmbedKitReport`, including healthy ranges,
pathology thresholds, geometric meaning, and the downstream consequences.

---

## Intrinsic Dimension (`report.intrinsic_dim`)

| Attribute | Type | Meaning |
|---|---|---|
| `consensus` | float | Aggregated ID across methods (mean of non-NaN estimates) |
| `uncertainty` | float | Std across methods; high value means estimates disagree |
| `estimates` | dict[str, float] | Per-method estimates |
| `local_estimates` | dict[str, ndarray] | Per-point estimates where available |

**Methods available:** `TwoNN`, `MLE`, `lPCA`, `DANCo`, `CorrInt`, `MOM`, `FisherS`
(pass `id_methods=[...]` to `EmbedKitAnalyzer` to select a subset).

### Key ratio: ID / ambient dimension (D)

| ID/D ratio | Interpretation | Recommended action |
|---|---|---|
| `< 0.1` | Very sparse manifold — ambient dim far exceeds data complexity | Strong bottleneck: set `target_dim` ≈ 2 × ID |
| `0.1 – 0.3` | Moderate sparseness | Mild reduction: `target_dim` ≈ 1.5 × ID (default) |
| `0.3 – 0.7` | Reasonable use of dimensions | No aggressive reduction needed |
| `> 0.7` | Dimensions mostly used | Reduction may hurt; keep `target_dim` close to D |

**Severity contribution:** `ID/D < 0.1` → +2 (high); `0.1 – 0.3` → +1 (medium).

---

## Hubness (`report.hubness`)

Hubness measures how unevenly points appear as k-nearest neighbors of other points.
In high-dimensional spaces, a few "hub" points dominate the kNN graph regardless of
semantic similarity, breaking retrieval and kNN-based classifiers.

| Attribute | Healthy range | Pathological | Meaning |
|---|---|---|---|
| `k_skewness` | `< 2` | `> 5` (severe), `2–5` (moderate) | Skewness of the k-occurrence distribution |
| `robinhood_index` | `< 0.2` | `> 0.4` | Gini-like inequality; 0 = uniform, 1 = monopoly |
| `antihub_ratio` | `< 0.1` | `> 0.3` | Fraction of points never retrieved as neighbors |
| `hub_ratio` | `< 0.05` | `> 0.1` | Fraction of points occurring > 2k times |
| `hub_contamination` | `< 0.2` | `> 0.5` | Fraction of kNN slots occupied by hubs |
| `k_occurrence` | ndarray | — | Raw per-point occurrence counts |

### What hubness does to downstream tasks

- **Retrieval / RAG**: Hub documents are returned for almost every query, drowning out
  relevant results.
- **kNN classification**: Decision boundaries are dominated by hub labels.
- **Clustering**: Hubs act as artificial attractors, distorting cluster assignments.

### Fix

- Apply `AlignUniformLoss` (uniformity term penalizes hub formation directly).
- Use `CompositeAugmentation` for high-severity cases.
- The auto-configured `EmbedKit` adds `AlignUniformLoss` whenever `k_skewness > 5`.

**Severity contribution:** `k_skewness > 5` → +2; `2–5` → +1; `antihub_ratio > 0.3` → +1.

---

## Geometry (`report.geometry`)

### Distance Concentration (`report.geometry.distance_concentration`)

| Attribute | Healthy | Pathological | Meaning |
|---|---|---|---|
| `relative_contrast` | `> 1.0` | `< 0.5` | `(d_max − d_min) / d_min` — how spread out distances are |
| `concentration_ratio` | `< 0.4` | `> 0.8` | `std(D) / mean(D)` — coefficient of variation of pairwise distances |
| `distance_histogram` | ndarray | — | Histogram of sampled pairwise distances |
| `bin_edges` | ndarray | — | Bin edges for the histogram |

**Effect:** When all points are nearly equidistant (`concentration_ratio > 0.8`),
nearest-neighbor search becomes meaningless — all distances are indistinguishable.
This is the curse of dimensionality in its most acute form.

**Fix:** Aggressive dimensionality reduction (`target_dim` ← 2–4 × ID).

**Severity contribution:** `concentration_ratio > 0.8` → +2.

---

### Isotropy (`report.geometry.isotropy`)

| Attribute | Healthy | Pathological | Meaning |
|---|---|---|---|
| `participation_ratio` | close to D | `< 0.1 × D` (severe), `0.1–0.3 × D` (moderate) | `(Σλ)² / Σλ²` — effective number of active PCA dimensions |
| `effective_rank` | close to D | — | Entropy-based rank estimate |
| `isotropy_score` | close to 1.0 | `< 0.3` | Normalized [0, 1] isotropy |
| `eigenvalue_spectrum` | ndarray | — | All PCA eigenvalues (use `AnalysisPlotter.plot_eigenvalue_spectrum()`) |
| `explained_variance_ratio` | ndarray | — | Per-component variance ratios |

**Effect:** Anisotropic embeddings concentrate information in a few directions.
Cosine similarity becomes unreliable because most variation lies in a low-rank subspace.

**Fix:** `AlignUniformLoss` pushes the representation toward the unit hypersphere.
PCA whitening before training can help as a preprocessing step.

**Severity contribution:** `PR < 0.1 × D` → +2; `0.1–0.3 × D` → +1.

---

### Neighbor Consistency (`report.geometry.neighbor_consistency`)

| Attribute | Healthy | Pathological | Meaning |
|---|---|---|---|
| `mean_consistency` | `> 0.7` | `< 0.5` | Avg fraction of kNN preserved under small Gaussian perturbation |
| `std_consistency` | low | high | Variability in consistency across perturbation trials |
| `per_perturbation` | ndarray | — | Per-trial consistency scores |

**Effect:** Low consistency means small changes in input values cause large changes
in the kNN graph. The local structure is fragile — models trained on it are not robust.

**Fix:** `GaussianNoise(adaptive=True)` or `EmbeddingMixup` augmentations teach the
model to produce consistent local neighborhoods.

**Severity contribution:** `mean_consistency < 0.5` → +1.

---

### Uniformity (`report.geometry.uniformity`)

| Attribute | Healthy | Pathological | Meaning |
|---|---|---|---|
| `uniformity` | `< −2.0` | `> −1.0` | `log E[exp(−t·||z_i − z_j||²)]` — Wang & Isola uniformity |

Lower (more negative) is better. A score near 0 means all embeddings are collapsed
to the same point; a score around −3 to −5 indicates well-distributed embeddings.

**Effect:** Poorly uniform embeddings form dense clusters in embedding space, which
hurts retrieval diversity and model calibration.

**Fix:** `AlignUniformLoss` directly optimizes this term.

**Severity contribution:** `uniformity > −1.0` → +1.

---

## Kernel Diagnostics (`report.kernel`)

Applies an RBF kernel `K(x_i, x_j) = exp(−||x_i − x_j||² / (2σ²))` and analyzes
its spectrum. Sigma is chosen by the median heuristic from the report's
`suggested_sigma`.

| Attribute | Meaning |
|---|---|
| `sigma` | RBF bandwidth used |
| `effective_rank` | Entropy-based effective rank of the kernel matrix |
| `spectral_gap` | `λ₁ / λ₂` — dominance of top eigenvalue |
| `condition_number` | `λ_max / λ_min` — numerical stability indicator |
| `row_sum_skewness` | Continuous analog of hubness in kernel space |
| `kernel_alignment` | Alignment with the label kernel (only if `y` passed to `fit`) |
| `eigenvalues` | Full kernel eigenvalue array |

### Interpretation

| Signal | Meaning |
|---|---|
| `effective_rank` close to n | Kernel sees all points as unique; good diversity |
| Low `effective_rank` (< 0.1 × n) | Kernel has degenerate structure; embeddings collapse |
| `spectral_gap > 5` | Top eigenvector dominates — strong global cluster |
| `condition_number > 1e6` | Numerically ill-conditioned; may cause instability |
| `kernel_alignment > 0.3` | Good label alignment — supervised training should be effective |
| `kernel_alignment < 0.05` | Label structure not visible in current geometry — embeddings may need improvement before supervised training |

---

## Severity Scoring

`EmbedKitAnalyzer` accumulates a score from individual checks:

| Score | Severity |
|---|---|
| 0–1 | `"low"` — minor or no issues |
| 2–3 | `"medium"` — some pathologies present |
| ≥ 4 | `"high"` — multiple severe pathologies |

The score contributions from each check are listed per-metric above. `EmbedKit` auto-config
always uses `EmbeddingMixup(k=report.suggested_k, alpha=0.4)` regardless of severity. Severity
still escalates the *loss* (adding `AlignUniformLoss` when `k_skewness > 5`).

---

## Summary: Pathology Checklist

When reviewing a report, scan these questions in order:

1. **Hubness** — Is `k_skewness > 2`? Is `antihub_ratio > 0.3`?
2. **Dimensionality** — Is `ID/D < 0.3`? Is `consensus_id` reliable (`uncertainty < 5`)?
3. **Isotropy** — Is `participation_ratio < 0.3 × D`? Is `isotropy_score < 0.5`?
4. **Distance collapse** — Is `concentration_ratio > 0.8`?
5. **Stability** — Is `mean_consistency < 0.7`?
6. **Spread** — Is `uniformity > −1.5`?
7. **Label structure** — If labels are available, is `kernel_alignment > 0.1`?

If most answers are "no", severity will be `"low"` and simple refinement (or no
refinement at all) is appropriate.
