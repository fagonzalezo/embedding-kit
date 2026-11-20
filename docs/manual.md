# EmbedKit Concepts Manual

EmbedKit is a Python library for analyzing and improving ML embedding spaces. It has two pillars:

- **Analysis** — a suite of geometry-aware metrics that describe an embedding space (intrinsic dimension, hubness, distance concentration, anisotropy, kernel spectrum), each judged against a null model. The report makes a recommendation only for the two rules that experiment E4 validated (more hubness and more antihubs than the null, under Euclidean distance).
- **Improvement** — a transform chosen by validation: the input centred and L2-normalised without labels; with labels, the best of the input, centring, PCA and a supervised contrastive refiner, kept only if it is significantly better than the input on held-out rows. A contrastive refiner, training-free baselines and hubness-reducing neighbour search are also available directly.

The benchmarks behind these defaults, and where the library does not help, are summarised in the
README (`docs/benchmarks/` and `benchmarks/results/e4/REPORT.md` have the tables).

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

X = np.load("embeddings.npy")          # (n_samples, n_features)

# analyze only
report = EmbedKitAnalyzer().fit(X)
report.print_summary()

# improve in one call: without labels, centred and L2-normalised
X_refined = EmbedKit().fit_transform(X)
```

---

## Part 1 — Analysis

### 1.1 Intrinsic Dimension

The *intrinsic dimension* (ID) of a dataset is the minimum number of coordinates needed to describe its points without significant information loss. A sentence-embedding model may output 768-dimensional vectors, but those vectors may lie on a manifold whose true dimensionality is closer to 20.

**Why it matters.** The ID describes how many directions the data uses locally, and projecting below it loses information. It does not say that a reduction helps: in E4, PCA to the report's `suggested_target_dim` (1.5 × the TwoNN estimate) lost kNN accuracy significantly on 32 of 35 natural sets (median -5.1 points under cosine). Treat ID and the ID / D ratio as description.

EmbedKit supports eight estimators, each making different geometric assumptions. They measure different things and are reported separately; nothing combines them:

| Method | Approach |
|--------|----------|
| `TwoNN` | Ratio of the two nearest-neighbor distances; fast and robust (Facco et al., 2017) |
| `MLE` | Maximum-likelihood over local neighborhood distances (Levina & Bickel, 2005) |
| `lPCA` | Dimensionality from local PCA scree plots |
| `DANCo` | Combines distance and angular statistics (Ceruti et al., 2012) |
| `CorrInt` | Correlation integral scaling (Grassberger & Procaccia, 1983) |
| `MOM` | Method of moments on nearest-neighbor distances (Amsaleg et al., 2015) |
| `FisherS` | Separability-based estimate using Fisher information (Albergante et al., 2019) |
| `ESS` | Expected simplex skewness (Johnsson et al., 2015); the most accurate at high ID in our tests, but about 100 s per fit at n = 5,000 |

**Class:** `IntrinsicDimensionEstimator(methods, n_max, n_bootstrap, scale_curve, random_state)`

**Result fields:**
- `estimates` — per-method ID values (NaN when a method failed, with a warning).
- `standard_errors`, `intervals` — bootstrap standard error (`n_bootstrap = 20` resamples of the rows; repeated rows are dropped, as in `fit`) and the interval `estimate ± 1.96 se`. Against the spread over independent draws of spheres and cubes, these standard errors were within about 30% of the true sampling error. `ESS` and `DANCo` are too slow to resample and get NaN.
- `between_method_spread` — standard deviation of the estimates across methods (formerly `uncertainty`). It is not a sampling uncertainty.
- `scale_curve`, `scale_max_deviation` — TwoNN on nested subsamples of n, n/2, n/4 and n/8 rows (fewer rows probe larger scales), and the largest relative deviation of a level from the full-sample value. Noise inflates the ID at short scales, so a curve that rises with n points to noise (Denti et al., 2022). No threshold is applied: a boolean plateau flag (within 10%) flipped between random subsets of one dataset in E3, and even clean manifolds drift by about 8% at ID 20.
- `reference_method`, `reference_estimate` — the one estimate the report uses for `suggested_target_dim`: TwoNN when requested and successful, otherwise the first method in `methods` that succeeded.
- `local_estimates` — per-point local ID for methods that support it (TwoNN, MLE, lPCA).
- `n_duplicates_removed` — exact duplicate rows dropped before estimation (duplicates make TwoNN divide by zero and MLE return 0).

Methods run one after another; running them in threads made results vary between runs with the same `random_state`.

**Ranges & interpretation:**
- **The intervals cover sampling error only, not bias.** The k-NN estimators (TwoNN, MLE, MOM) read low at high ID: on a 20-sphere with n = 5,000 they give 15 to 17. `lPCA` counts linear dimensions, so it returns d + 1 for a d-sphere and is exact only for flat manifolds.
- The report makes no recommendation from ID/D. The 0.1 and 0.3 rule was removed after experiment E4: it fired on all 35 natural classification sets, and PCA to `suggested_target_dim` lost kNN accuracy on 32 of them (median -5 points under cosine).

```python
from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

result = IntrinsicDimensionEstimator(methods=["TwoNN", "MLE", "lPCA"]).fit(X)
for m, est in result.estimates.items():
    print(m, est, result.intervals[m])
print(result.scale_curve, result.scale_max_deviation)
```

---

### 1.2 Hubness

*Hubness* is a curse-of-dimensionality phenomenon in which a handful of points — called **hubs** — repeatedly appear in the k-nearest-neighbor lists of many other points, while **antihubs** never appear as anyone's neighbor at all (Radovanović et al., 2010). This imbalance emerges from the geometry of high-dimensional spaces and is independent of the specific data distribution.

**Why it matters.** Hubness can degrade kNN-based tasks: retrieval, classification, clustering and anomaly detection all use nearest-neighbor graphs. Hubs fill neighbour lists and hide antihubs from retrieval. On the pretrained embeddings of E1 to E4, though, hubness reduction changed kNN accuracy by little (best +0.005 on the E1 sets; E4: significantly better than raw on 14 of 35 sets under Euclidean kNN, worse on 3, median +0.4 points). It is the one defect for which the report makes a recommendation (Euclidean only, below).

**Class:** `HubnessAnalyzer(k, hub_threshold, metric)`

**Result fields:** `k_skewness`, `robinhood_index`, `antihub_ratio`, `hub_ratio`, `k_occurrence`, `hubs`, `antihubs`, `hub_contamination`

**Ranges & interpretation.** The ranges below are rough guides for reading a number. The report does not use them: it compares `k_skewness` and `antihub_ratio` with a null model.

| Metric | Range | Good | Concerning |
|--------|-------|------|------------|
| `k_skewness` | (−∞, ∞) | near 0 | > 2 (moderate), > 5 (severe) |
| `robinhood_index` | [0, 1] | near 0 | > 0.3 |
| `antihub_ratio` | [0, 1] | depends on n, d, k | above the null (see below) |
| `hub_ratio` | [0, 1] | < 0.05 | > 0.1 |
| `hub_contamination` | [0, 1] | near 0 | > 0.3 |

- `k_skewness`: skewness of the k-occurrence distribution N_k (how many times each point is a kNN of someone else). Gaussian distributions have zero skewness; positive skew means a fat tail of highly popular hubs.
- `robinhood_index`: Gini-like inequality measure over N_k values. 0 = every point is equally popular; near 1 = a few points dominate the entire neighbor graph.
- `antihub_ratio`: fraction of points with N_k = 0 (never retrieved). It depends strongly on n, d, k and the row norms (an isotropic Gaussian at d = 1,024 has 0.21 to 0.39 under Euclidean distance and none under cosine), so the report judges it against the null model instead of a fixed threshold.
- `hub_ratio`: fraction of points classified as hubs (N_k > `hub_threshold` × mean N_k). Should be small.
- `hub_contamination`: fraction of a hub's own kNN that are also hubs. High values indicate that hubs cluster together, further compounding retrieval skew.

```python
from embedkit.analysis.hubness import HubnessAnalyzer

result = HubnessAnalyzer(k=10).fit(X)
print(f"k-skewness:     {result.k_skewness:.3f}")
print(f"Robin Hood idx: {result.robinhood_index:.3f}")
print(f"Antihub ratio:  {result.antihub_ratio:.3f}")
```

**The recommendation.** When `report.calibration["k_skewness"]` or `["antihub_ratio"]` has z above `z_threshold` under `metric="euclidean"`, the report recommends a hubness-reducing neighbour search (`embedkit.hubness_reduction`, section 2.7). In E4 the rule for k-skewness fired on only 3 of 35 sets (a significant gain on all 3; Spearman 0.36, 95% interval 0.13 to 0.65 across source data sets), and the one for antihubs on 8 (a significant gain on 5; 0.32, 0.05 to 0.58). Retested on 60 sets with 5 more encoders (E10, `docs/benchmarks/BENCHMARK_SELECTION.md`), the k-skewness rule fired on 5 (a significant gain on all 5; 0.48, 0.30 to 0.71) and the antihub rule on 22 (19; 0.53, 0.29 to 0.74). Under cosine neither predicted a gain, so the report recommends nothing there. The 23 primary tests of E4 had no multiplicity correction; about one false pass is expected by chance.

---

### 1.3 Distance Concentration

In high-dimensional spaces, pairwise Euclidean distances tend to concentrate near their mean — the ratio `(d_max − d_min) / d_mean` shrinks toward zero as dimensionality grows (Beyer et al., 1999). When this happens, all points look equally (dis)similar and nearest-neighbor queries become meaningless.

**Class:** `DistanceConcentration(subsample, n_bins, random_state)`

**Result fields:** `distance_cv`, `relative_contrast_per_query`, `relative_contrast`, `concentration_ratio`, `distance_histogram`, `bin_edges`

**Ranges & interpretation:**

- `distance_cv` = standard deviation over mean of the pairwise distances: `[0, ∞)`. **Larger is better.** For iid Gaussian data it is about `1 / sqrt(2d)`. The report compares it with the null model (a z-score in `report.calibration`), not with a fixed value, because its scale depends on d and the spectrum; it makes no recommendation from it, since in E4 the gain of PCA or hubness reduction fell as the signal rose (Spearman -0.50). Robust to a single duplicated pair.
- `relative_contrast_per_query` = median over query points of `(d_max − d_min) / d_min`, with zero distances excluded.
- `relative_contrast` = `(d_max − d_min) / d_mean`: `[0, ∞)`. **Larger is better.** Near 0 means the distance histogram is a narrow spike; high-dimensional random data can push this below 0.1. Values above 1.0 indicate a healthy spread of distances.
- `concentration_ratio` = `d_min / d_max`: `[0, 1]`. Near 0 = healthy (minimum distances are much smaller than maximum). Near 1 = severe concentration (every pair of points is nearly the same distance apart). One duplicated pair sets it to 0, so the report no longer uses it (since 0.1.3).

```python
from embedkit.analysis.geometry import DistanceConcentration

result = DistanceConcentration().fit(X)
print(f"Relative contrast:  {result.relative_contrast:.3f}")
print(f"Distance CV:        {result.distance_cv:.4f}")
```

---

### 1.4 Isotropy

An *isotropic* embedding space uses all directions equally — variance is spread evenly across every dimension. An *anisotropic* space concentrates most variance in a small number of principal directions, leaving the rest unused.

**Why it matters.** Anisotropy often co-occurs with hubness (a few dominant directions create hub structure), and removing the mean and top components helped some language-model tasks (Mu et al., 2018). It is a property, not a defect: strict isotropy is at odds with cluster structure, and whitening can hurt classification. The report therefore describes these statistics in `report.properties` and makes no recommendation from them. Experiment E4 tested the review's hypotheses (a dominant mean direction, dominant top components, anisotropy, rogue coordinates) against their fixes (mean removal, all-but-the-top, ZCA and PCA whitening, standardising): none predicted its fix, and ZCA lost accuracy on 34 of 35 sets under cosine kNN and on all 35 under Euclidean kNN.

**Class:** `IsotropyAnalyzer(n_components=512, n_max=5000, random_state=42)`

**Result fields:** `isotropy_score`, `participation_ratio`, `participation_ratio_corrected`, `participation_ratio_normalized`, `effective_rank`, `isoscore`, `top_k_variance_share`, `top_k_coordinate_share`, `spectrum_alpha`, `mean_vector_ratio`, `mean_cosine`, `eigenvalue_spectrum_top`, `explained_variance_ratio`

All statistics are computed exactly, from the covariance or the Gram matrix of at most `n_max` rows, at any dimension. `eigenvalue_spectrum_top` holds the top `n_components` eigenvalues, for plotting (`eigenvalue_spectrum` is a deprecated alias).

**Ranges & interpretation:**

- `isotropy_score` = `effective_rank / min(n − 1, D)`: `[0, 1]` (clipped). **1 = perfectly isotropic** (all directions used equally); near 0 = the space has effectively collapsed to a single direction.
- `participation_ratio` = `(Σλᵢ)² / Σλᵢ²`: `[1, D]`. Intuitively, the number of eigenvalue-weighted dimensions that contribute to the space. Near D = isotropic. The sample value is biased downward by about `1 / (1 + D/n)`, so an isotropic Gaussian with n = 3,000 gives 0.75·D at D = 1,000 and 0.55·D at D = 2,500.
- `participation_ratio_corrected`: the same ratio with an unbiased estimate of `tr(Σ²)` (exact for Gaussian data), clipped to `[1, D]`. `participation_ratio_normalized` = corrected / D, in `[0, 1]`: the summary of isotropy that does not depend on n.
- `effective_rank` = `exp(H)` where H is the entropy of the normalized eigenvalue spectrum: `[1, D]`. Measures how many "effective" principal components are needed to describe the space. Like `isotropy_score`, it is a plug-in value, biased low when n is not much larger than D (0.66 of the maximum for iid data at n = 3,000, D = 2,500).
- `isoscore`: IsoScore (Rudman et al., 2022), `[0, 1]`, computed as published. Its six steps reduce to `(participation_ratio − 1) / (D − 1)`, so it carries the plug-in bias: iid data at n = 300, D = 512 score 0.37, not 1.
- `top_k_variance_share`: share of the variance in the top 1, 3 and 10 principal components. `top_k_coordinate_share`: the same for the coordinates with the largest variance; a few dominant coordinates are the "rogue dimensions" of Timkey and van Schijndel (2021).
- `spectrum_alpha`: power-law decay exponent of the eigenvalues, `λᵢ ∝ i^−α` (Agrawal et al., 2022), fitted on a log-log scale over the top half of the non-zero spectrum (the finite-sample tail bends down). It recovers α = 1 and 2 within 0.06 from n = 300 to 5,000; a flat spectrum gives 0.1 to 0.35.
- `mean_vector_ratio` = `‖mean(x)‖ / mean(‖x‖)` and `mean_cosine` (average cosine over all pairs of distinct rows), both uncentred: how much of each vector is a shared offset.

```python
from embedkit.analysis.geometry import IsotropyAnalyzer

result = IsotropyAnalyzer().fit(X)
print(f"Isotropy score:    {result.isotropy_score:.3f}")
print(f"Participation ratio: {result.participation_ratio:.1f} / {X.shape[1]}")
print(f"Effective rank:    {result.effective_rank:.1f}")
```

---

### 1.5 Neighbor Consistency

*Neighbor consistency* measures how stable the k-NN graph is under small Gaussian perturbations. For each point, a perturbed copy of the dataset is generated with noise of standard deviation `noise_std × s`, where `s` is the RMS per-coordinate standard deviation `sqrt(mean_j var(X[:, j]))` (translation-invariant; before 0.5 it was the standard deviation over all entries, which also counted the spread of the coordinate means), and the fraction of original neighbors that survive in the perturbed graph is recorded.

**Why it matters.** An unstable kNN graph means that tiny changes in input (e.g., measurement noise, model updates) can completely reorganize the neighborhood structure. Downstream tasks that rely on kNN will be unreliable in such spaces.

**Class:** `NeighborConsistency(k, n_perturbations, noise_std, metric, random_state)`

**Result fields:** `mean_consistency`, `std_consistency`, `per_perturbation`

**Ranges & interpretation:**
- `mean_consistency`: `[0, 1]`. Fraction of original k-neighbors that are preserved after perturbation, averaged over all points and all perturbation trials. Near 1 = robust neighborhood structure. Low values mean the neighbour lists change under small noise. The report compares it with the null model but makes no recommendation from it: in E4 it did not predict the gain of PCA (Spearman -0.35 and -0.02 for the two metrics, intervals including 0).

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
- `uniformity` = `log E[exp(−t ‖zᵢ − zⱼ‖²)]` for L2-normalized vectors: always **negative** (the expectation is strictly less than 1 for any non-degenerate distribution). **More negative = more uniform** (better coverage of the sphere). Near 0 means embeddings are tightly clustered or collapsed. The report compares it with the null model and states the result in `report.properties` (clustered directions are a property, not a defect); there is no fixed threshold.

```python
from embedkit.analysis.geometry import UniformityScore

result = UniformityScore(t=2.0).fit(X)
print(f"Uniformity: {result.uniformity:.4f}")
# rough reading: around -3 to -5 is spread out; above -1 is clustered
```

---

### 1.7 Kernel Diagnostics

The RBF (Gaussian) kernel matrix `K(xᵢ, xⱼ) = exp(−‖xᵢ − xⱼ‖² / 2σ²)` encodes a non-linear similarity structure. Analyzing its eigenspectrum reveals whether the embeddings are well-separated and numerically stable for kernel-based methods (SVMs, kernel PCA, MMD tests).

**Class:** `KernelDiagnostics(sigma, n_components, subsample, random_state)`

When `sigma="median"` (default), σ is set to the median pairwise distance — a standard heuristic.

**Result fields:** `effective_rank`, `spectral_gap`, `top_k_eigen_ratio`, `row_sum_skewness`, `eigenvalues`, `centered_kernel_alignment`, `sigma`

The spectrum statistics come from the centred kernel `Kc = HKH` (`H = I − 11ᵀ/n`). The uncentred RBF kernel has a large eigenvalue along the constant direction, which says only that the kernel is positive; before 0.5 it dominated the spectral gap.

| Metric | Range | Meaning |
|--------|-------|---------|
| `effective_rank` | [1, subsample − 1] | `tr(Kc)² / tr(Kc²)`: how many directions of the kernel feature space carry the variance. From traces, over the whole spectrum. |
| `spectral_gap` | [1, ∞) | `λ₁ / λ₂` of `Kc`. Large when one direction dominates, for example two well-separated clusters. |
| `top_k_eigen_ratio` | [1, ∞) | `λ₁ / λ_k`, with k the number of eigenvalues kept (`n_components`, default 50). Formerly `condition_number`, which is a deprecated alias. |
| `row_sum_skewness` | (−∞, ∞) | Skewness of the row sums of the uncentred kernel: a continuous analogue of hubness. |
| `centered_kernel_alignment` | [−1, 1] or None | See below. |

The effective rank depends on σ: a small bandwidth makes every point its own direction. With `sigma="median"` it is comparable between data sets of the same size. Before 0.5 `effective_rank` was the entropy rank of the top 50 eigenvalues of the uncentred kernel, capped at 50.
- `centered_kernel_alignment`: populated only when labels `y` are passed to `fit(X, y)`. Centred alignment (both kernels centred, `HKH`) between the data kernel and a label kernel (Cristianini et al., 2001): the indicator kernel for class labels (integers, strings, booleans), an RBF kernel with a median-heuristic bandwidth for continuous labels (1-D floats or 2-D arrays). About 0 for labels independent of the data; near 1 means the RBF geometry reflects the labels. `kernel_alignment` is a deprecated alias (before 0.1.3 it was uncentred, and mostly measured class balance).

```python
from embedkit.analysis.kernel import KernelDiagnostics

result = KernelDiagnostics(sigma="median").fit(X)
print(f"Effective rank:   {result.effective_rank:.1f}")
print(f"Spectral gap:     {result.spectral_gap:.3f}")
print(f"lambda_1/lambda_k: {result.top_k_eigen_ratio:.3g}")
```

---

### 1.8 Unified Analysis Report

`EmbedKitAnalyzer` runs all seven analyzers in a single call and assembles the results into an `EmbedKitReport`. The report also holds a null-model calibration, a list of **properties** (descriptive statements about the spectrum, the dominant coordinates, the mean direction and, when calibrated, uniformity against the null; no thresholds) and a list of **recommendations** (actions). Only rules that experiment E4 found to predict the gain of their fix on 35 natural classification sets make a recommendation: more hubness or more antihubs than the null, under `metric="euclidean"`, each pointing to a hubness-reducing neighbour search (`embedkit.hubness_reduction`). E4 tested 9 rules; 7 failed. Distance concentration, neighbour consistency and ID/D are still computed and calibrated but raise no action; nor does anything under cosine (`benchmarks/results/e4/REPORT.md`). When no rule fires the report says "No recommendation: no statistic with a validated rule is beyond its null range.", which does not mean no fix can help, only that the report cannot tell which one would. Everything else in the report is descriptive.

**Null-model calibration** (`calibrate=True`, the default). k-skewness, the antihub ratio, `distance_cv`, neighbour consistency and uniformity depend on n, d, k and the covariance spectrum, so a fixed threshold means something different on every dataset. The analyzer recomputes these five statistics on `n_null_draws` (default 20) data sets drawn from a null model of the same size as a subsample of `n_calibration` rows (default 2,000), with the same metric. The k of the calibration (`report.calibration_k`) is `k` when given, and otherwise the default rule applied to the calibration rows (22 for 2,000 rows), so that the z-scores do not change with the size of the data. Rows that arrive with a common norm (normalised upstream) are compared with null draws normalised to the same norm. `report.calibration[name]` holds the observed value, the null mean and standard deviation, the z-score, `z_interval` (a 95% interval for z that reflects the limited number of null draws: with 20 draws, a z between 3 and 4 can land on either side of 3 when the analysis is rerun, and a recommendation whose interval reaches below the threshold says so; raise `n_null_draws` to settle such a reading) and `uncalibrated` (`None`, or the reason the null is not calibrated for that statistic here; `.calibrated` is the boolean). `report.n_outside_null` counts the calibrated statistics with `|z| > z_threshold` (default 3); it replaces the old `severity`. The default `null_model="gaussian"` is a Gaussian with the data's mean (shrunk so that its squared norm is unbiased) and a covariance whose trace and `tr(Σ²)` match unbiased estimates from the data (`embedkit.analysis.null`); `"permuted"` permutes each column independently. The recommendations come from the z-scores (k-skewness and antihub ratio, Euclidean only), and with `calibrate=False` none is made. The null model is not reliable everywhere: on data drawn from the null itself, some statistics fall outside ±3 too often in some settings (`docs/benchmarks/BENCHMARK_DIAGNOSTICS.md`). The report marks those statistics as uncalibrated, with the reason in `uncalibrated`:

- rows normalised upstream (all rows with the same norm): the antihub ratio, `distance_cv` and uniformity, because with an uneven spectrum the null on the sphere is more isotropic than the data;
- `metric="cosine"`: k-skewness and the antihub ratio;
- `metric="euclidean"` with more than 256 dimensions: neighbour consistency.

An uncalibrated statistic keeps its z-score, but raises no recommendation and is not counted in `n_outside_null`. In practice the only recommendation this removes is the antihub one on rows normalised upstream. One known weak spot is not flagged: `distance_cv` under cosine on data with a large common mean, whose z-scores occasionally land just beyond ±3 on null data. Calibration costs `n_null_draws + 1` evaluations of the five statistics: about 50 s for 3,680 × 2,048 ResNet features on a laptop.

**Derived values.**

| Field | Meaning |
|-------|---------|
| `k` | The neighbourhood size the report used: the `k` given, else `max(5, round(√min(n, n_max) / 2))`. Replaces `suggested_k` |
| `kernel.sigma` | The RBF bandwidth of the kernel diagnostics (median pairwise distance). Replaces `suggested_sigma` |
| `suggested_target_dim` | `clip(1.5 × ID, max(4, ID), D)` with ID the reference estimate (TwoNN by default). No benchmark supports the factor 1.5, and PCA to it lost accuracy on 32 of 35 E4 sets. Only `EmbedKit(method="pca" or "refiner", target_dim="auto")` reads it |

`report.suggested_k` and `report.suggested_sigma`, deprecated in 0.6.0, were removed in 0.7.0.

**Class:** `EmbedKitAnalyzer(k, id_methods, metric, random_state, calibrate, null_model, n_null_draws, n_calibration, z_threshold, id_bootstrap, id_scale_curve)`

**Metric.** `metric` is `"euclidean"` (default) or `"cosine"`; anything else raises `ValueError`. Every analyzer honours it. With `"cosine"` the rows are L2-normalised once and every number in the report describes the normalised rows, the geometry a cosine search sees: the kNN graph, distance concentration and the kernel use chord distances on the sphere (monotone in cosine distance), intrinsic dimension is estimated on the sphere, and the isotropy statistics describe the normalised rows. The calibration null is the exception: it is fitted to the rows as given and its draws are normalised before the statistics are computed, so the null mean and spread depend on the row norms. Rows with zero norm have no direction and raise `ValueError`.

```python
from embedkit import EmbedKitAnalyzer

report = EmbedKitAnalyzer(k=10, id_methods=["TwoNN", "MLE", "lPCA"]).fit(X)
report.print_summary()

# access individual sub-results
print(report.hubness.k_skewness)
print(report.geometry.isotropy.isotropy_score)
print(report.calibration["k_skewness"].z_score)   # against the null model
print(report.n_outside_null)      # statistics with |z| > 3
print(report.properties)          # descriptive sentences, no thresholds
print(report.recommendations)     # list of strings; at most the two validated rules
print(report.k, report.kernel.sigma, report.suggested_target_dim)

# tabular view
df = report.to_dataframe()
```

---

## Part 2 — Improvement

### 2.1 EmbeddingRefiner

`EmbeddingRefiner` is a small feed-forward network that maps original embeddings of dimension D to a space of dimension `target_dim`. It is the learnable component trained by the contrastive objective. Whether the trained map beats PCA depends on the mode: it did not without labels (E1, E6, E7), and it did with labels (E8; section 2.5).

**Architecture:** `n_layers` repetitions of `Linear → BatchNorm → ReLU → Dropout`, followed by a final `Linear(hidden_dim, target_dim)`. When `normalize_output=True` (default), the output is L2-normalized to the unit sphere, which is required by cosine-similarity-based losses.

**Residual and linear variants** (`architecture="residual"` or `"linear"`): `z = W x̃ + g(x̃)` (or `z = W x̃`), where `x̃` is the input standardised with fixed per-feature statistics and `g` is the MLP above with its output layer initialised to zero. `Trainer.fit` calls `model.initialize_from_data(X_train)`, which sets the statistics and initialises `W` from the top `target_dim` principal directions, so that before training the output equals `PCAReduce(target_dim)` followed by L2 normalisation. Training starts from PCA instead of a random map. The self-supervised refiner uses `"linear"` since benchmark E6 (`docs/benchmarks/BENCHMARK_E6_E7.md`): both variants beat `"mlp"` on 5 of the 6 datasets with an interval and lost on none, and `"linear"` was at least as good as `"residual"` on 5 of 6; the supervised refiner uses `"residual"` and the regression one `"mlp"` (E8, section 2.5). `EmbeddingRefiner` itself defaults to `"linear"`; bundles saved without an `architecture` entry load as `"mlp"`.

**Class:** `EmbeddingRefiner(input_dim, target_dim, hidden_dim=256, n_layers=2, dropout=0.1, normalize_output=True, architecture="linear")`

`hidden_dim`, `n_layers` and `dropout` configure the MLP; `architecture="linear"` ignores them. `Trainer.fit` initialises a linear or residual refiner at PCA; a custom training loop must call `initialize_from_data(X_train)` itself (training without it warns once).

```python
from embedkit.improvement.model import EmbeddingRefiner
import torch

X_train = torch.randn(500, 128)
model = EmbeddingRefiner(input_dim=128, target_dim=32)   # architecture="linear"
model.initialize_from_data(X_train.numpy())               # start at PCA
z = model(X_train[:64])   # (64, 32), L2-normalized
```

---

### 2.2 Augmentations

Contrastive learning requires *positive pairs*: two different views of the same embedding that should be close in the refined space. Each augmentation in EmbedKit transforms a batch tensor `x` into a tuple `(x_i, x_j)` — two perturbed versions of the same batch. The loss then pulls `z_i = f(x_i)` toward `z_j = f(x_j)` while pushing them away from other points in the batch.

**Base class:** `BaseAugmentation.__call__(x: Tensor) → (Tensor, Tensor)`

---

#### GaussianNoise

Adds independent Gaussian noise to each embedding, producing two noisy views.

With `relative=True` (the default since 0.1.3), `std` is a multiple of the per-dimension standard deviation of the training data, computed by `precompute` (the trainer calls it); the expected noise norm is then about `std` times the norm of the centred data, whatever the embedding's scale. `relative=False` gives the old absolute standard deviation.

When `adaptive=True`, the noise standard deviation is scaled per-point by the mean distance to its k-nearest neighbors. This adapts the perturbation magnitude to the local density: tight clusters get smaller noise, sparse regions get larger noise.

**Parameters:** `std=0.05`, `adaptive=False`, `k=10`, `relative=True`

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

Instead of perturbing `x` to create a view, `KNNPairs` uses an actual nearest neighbor as the positive partner. The anchor `x` and one of its `k` nearest neighbors, drawn at random, form the pair `(x, x_pos)`. When the trainer calls `precompute`, neighbors come from the whole training set; otherwise they are searched within the batch. This augmentation carries the strongest geometric signal: the network explicitly learns that a point and its neighbors should map to similar representations.

`KNNPairs` is a *pair* augmentation (`kind = "pair"`). In a sequential `CompositeAugmentation` it must come first, and at most one pair augmentation is allowed; view augmentations placed after it perturb both the anchor and the neighbor.

**Parameters:** `k=10`, `neighbor_index=None`, `metric="euclidean"`, `mode="all"`

With `mode="mutual"` only mutual neighbours are used (`j` is among the `k` neighbours of `i` and `i` among those of `j`). This removes most of the cross-cluster pairs that hubs create. A point with no mutual neighbour is paired with its nearest neighbour.

```python
from embedkit.improvement.augmentation import KNNPairs

aug = KNNPairs(k=10)
anchor, positive = aug(x_batch)
```

---

#### IdentityPair

Returns `(x, x)`. Because the two views pass through the refiner separately in training mode, they differ only by its dropout masks: the SimCSE positives of Gao et al. (2021). It is a *pair* augmentation, and view augmentations may follow it in a `CompositeAugmentation`.

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

```text
forward(z_i: Tensor, z_j: Tensor, labels: Tensor | None = None) -> Tensor
```

where `z_i` and `z_j` are the projected views of the positive pair (typically L2-normalized).

---

#### NTXentLoss — Normalized Temperature-Scaled Cross-Entropy

The SimCLR / InfoNCE objective (van den Oord et al., 2018; Chen et al., 2020). Given a batch of N positive pairs, the 2N representations are assembled into a matrix. For each anchor, the corresponding view is the positive example and all other 2N − 2 representations are in-batch negatives. The loss is a softmax cross-entropy over cosine similarities scaled by temperature τ.

Lower τ concentrates the gradient on hard negatives; higher τ produces a softer distribution.

**Parameters:** `temperature=0.07`, `chunk_size=None`, `neighbor_mask=None`, `neighbor_metric="euclidean"`, `decoupled=False`

- `neighbor_mask=k` (or a precomputed `(n_train, k)` neighbour-index array) removes likely false negatives. Two rows of a batch are not negatives for each other when one is among the `k` nearest input-space neighbours of the other. The trainer computes the neighbours (`precompute`) and passes the batch's global row indices (`forward(..., indices=...)`). Without indices, no mask is applied.
- `decoupled=True` leaves the positive out of the denominator (Yeh et al. 2022).

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

**Use when:** you want an explicit uniformity term, for example as a complement to NTXentLoss via `CombinedLoss` (the self-supervised default pairs them at weights 1 and 0.5). In supervised mode E8 did not accept adding it to SupCon (`supcon_au` against SupCon alone: 1 win, 4 ties, 1 loss on kNN accuracy; 0 wins, 3 ties, 3 losses on the linear probe).

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

**Ties.** Candidates at exactly the same label distance from an anchor share one denominator, which holds the whole tie group and every farther candidate (as in the authors' code). Label distances are computed exactly, so identical labels always tie, and the loss does not depend on the order of the rows in the batch. To treat nearby continuous values as tied, round the labels before training. Versions before 0.1.3 handled ties incorrectly.

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

#### MatryoshkaLoss

The sum of a base loss over nested prefixes of the output, `z[:, :m]` for each `m` in `dims` (Kusupati et al. 2022). The full output is always included. Every prefix is trained to be an embedding on its own, so one model serves several output dimensions: `EmbedKit.transform(X, dim=m)` keeps the first `m` coordinates and renormalises. Labels, batch indices and `precompute` are forwarded to the base loss.

**Parameters:** `base_loss`, `dims=(16, 32, 64, 128)`, `weights=None` (1 for every prefix)

```python
from embedkit import EmbedKit
from embedkit.improvement.losses import MatryoshkaLoss, NTXentLoss

ek = EmbedKit(method="refiner", target_dim=128,
              loss=MatryoshkaLoss(NTXentLoss(), dims=(16, 32, 64)))
ek.fit(X)
Z32 = ek.transform(X, dim=32)   # first 32 coordinates, renormalised
```

`EmbedKit(method="refiner", nested_dims=...)` does the wrapping for the default loss: `"auto"` nests the powers of two from 32 below `target_dim`, a sequence gives the dims. In E6 (`docs/benchmarks/BENCHMARK_E6_E7.md`) the prefixes of one model at 256 matched `linear` models trained separately at 32, 64 and 128, at about 1.5 times the training cost of one of them. Without a refiner, PCA prefixes are nested by construction (`EmbedKit(method="pca", target_dim=128).fit(X).transform(X, dim=32)`).

```python
from embedkit import EmbedKit

ek = EmbedKit(method="refiner", target_dim=256, nested_dims="auto").fit(X)   # prefixes 32, 64, 128, 256
Z64 = ek.transform(X, dim=64)
```

---

### 2.4 Trainer

`Trainer` orchestrates the full contrastive training loop. It handles batching, optimization, learning rate scheduling, periodic evaluation against analysis metrics, early stopping, and loss history tracking.

**Class:** `Trainer(model, augmentation, loss, epochs, batch_size, optimizer, lr, weight_decay, scheduler, warmup_epochs, eval_every, eval_metrics, eval_subsample, eval_k, eval_metric, early_stopping_patience, monitor, restore_best, val_split, device, random_state)`

**Key parameters:**

| Parameter | Options | Notes |
|-----------|---------|-------|
| `optimizer` | `"adam"`, `"sgd"` | Adam is a safe default |
| `scheduler` | `"cosine"`, `"step"`, `"plateau"`, `None` | Cosine is stepped per batch, step per epoch, plateau per evaluation |
| `warmup_epochs` | int, default 0 | Linear learning-rate warm-up before the scheduler |
| `eval_metrics` | names in `trainer.MONITORS`: `"uniformity"`, `"isotropy"`, `"k_skewness"` | Metrics tracked during training; the monitor is always added |
| `eval_k`, `eval_metric` | default 10, `"cosine"` | Neighbourhood used by `k_skewness` |
| `monitor` | a name in `MONITORS` | Metric watched for early stopping; its direction (min or max) comes from the registry |
| `early_stopping_patience` | int or `None` | Stops after N **evaluations** (not epochs) without improvement |
| `restore_best` | bool, default `True` | With early stopping, reload the best evaluation's weights at the end (`best_score_`, `best_epoch_`) |
| `device` | `None`, `"cuda"`, `"mps"`, `"cpu"` | `None` picks CUDA, then MPS, then CPU |

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

`EmbedKit` fits one transform on the input embeddings and applies it with `transform`. One
parameter, `method`, says what it builds:

| `method` | Builds |
|---|---|
| `"auto"` (default) | Chosen for you: centre + L2 without labels; with labels, the candidate that validates best (below) |
| `"center_l2"` | The input centred on the training mean and scaled to unit L2 norm, with or without labels |
| `"pca"` | The top `target_dim` principal directions followed by L2 normalisation; nothing is trained |
| `"refiner"` | A trained `EmbeddingRefiner`, configured by `mode` (below) |

It is sklearn-style: `fit(X, y=None)`, `transform(X, dim=None)`, `fit_transform`, `save` and
`load`. Up to 0.7.0 a second parameter, `strategy`, switched between choosing and building; it is
deprecated (see "Fixed configurations" below).

#### What the default does, and why

**Without labels** (`EmbedKit().fit(X)`), the output is the input centred on the training mean and
scaled to unit L2 norm (`CenterNormalize`); nothing is trained and no analysis report is computed
(`analysis_report` is `None`). The reasons:

- No label-free score can say whether a transform helps. `knn_overlap`, which ranked refiner runs
  against each other in E2, gives the unchanged input the top score (`docs/benchmarks/BENCHMARK_E1_E2.md`).
- In E4, on 35 natural classification sets, no training-free fix beat the raw embedding
  significantly on more than 16 sets. Centring plus L2 normalisation was the most consistent small
  gain: Euclidean kNN 16 sets up and 1 down, median +0.3 points; cosine kNN 5 up and 4 down
  (`benchmarks/results/e4/REPORT.md`). Under a cosine search it changes nothing beyond removing the mean.
- PCA to `suggested_target_dim`, the 0.5.0 default, lost accuracy significantly on 32 of the 35 sets.
- The self-supervised refiner never beat PCA in E1, E6 and E7 and cost about 100 times more.

**With labels** (`EmbedKit().fit(X, y)`), `fit` selects among candidates:

| Step | What happens |
|---|---|
| Split | `val_fraction` (default 0.2) of the rows is held out, stratified by class when every class has at least 2 rows. The rest fits the candidates. |
| Candidates | `Identity` (the input as given); `CenterNormalize`; PCA at 64, 128 and 256 followed by L2 normalisation (only those below the input dimension and the data rank; the given `target_dim` instead, if any); in `"supervised"` and `"regression"` mode, the refiner of that mode at `target_dim` (`"auto"`: `min(128, d // 2)`, as in E8). |
| Criterion | 10-NN accuracy for class labels, 10-NN regression R^2 for continuous targets, on the held-out rows, with `selection_metric` (`"cosine"` by default, or `"euclidean"`). `knn_overlap` is not used. |
| Significance rule | A candidate replaces `Identity` only when its paired bootstrap interval of the per-row gain over `Identity` (95%, 2,000 resamples of the held-out rows) lies above 0. Among the candidates that pass, the best held-out score wins. |
| Result | `Identity` when none passes, so the output is never worse than the input by this criterion. The winner is refitted on all rows. |
| Large inputs | With more than `selection_max_rows` rows (default 50,000), the split, the candidates and the scores use a random subsample of that many rows, stratified by class; only the winner is fitted on all the rows. `None` selects on all the rows (the 0.6.0 behaviour). |

Most gains in E4 were under one point, so the interval keeps noise from choosing a transform. In
E10 (`docs/benchmarks/BENCHMARK_SELECTION.md`), on 60 natural classification sets and 4 regression sets, the
selection was significantly worse than the input on none and better on 43 (+0.8 to +15.1 points
of cosine kNN accuracy on the classification sets). Against the candidate that was best on the
test set in hindsight it lost nothing on 50 sets and at most 1.2 points. It chooses by kNN: a
linear probe on the output was worse than on the input on 5 sets (by 0.2 to 0.8 points), so check
your own task if it is not a neighbour search.

After `fit`: `selected_` names the winner (`"identity"`, `"center_l2"`, `"pca-128"`,
`"refiner-128"`, ...) and `selection_report_` is a DataFrame with one row per candidate:

| Column | Meaning |
|---|---|
| `candidate`, `dim` | Name and output dimension |
| `score` | Held-out 10-NN accuracy or R^2 |
| `diff_vs_identity`, `diff_lo`, `diff_hi` | Mean per-row gain over `Identity` and its 95% bootstrap interval |
| `beats_identity` | `diff_lo > 0` |
| `selected` | The winner |

`selection_report_.attrs` holds the criterion, `k`, the metric, the number of fit and held-out rows,
the number of resamples, the number of input rows (`n_rows`) and the number of rows the
selection used (`n_selection_rows`). With fewer than 20 rows to fit or to score (about 100 rows or fewer
at `val_fraction=0.2`), `fit` warns and returns `Identity`; pass `method="center_l2"`, `"pca"` or
`"refiner"` to build one anyway.

**Modes.** `mode="auto"` (default) is `"self_supervised"` without labels and `"supervised"` with
them. Continuous targets need `mode="regression"`: with `mode="auto"`, float labels that are not
whole numbers raise. With `mode="self_supervised"` and labels, selection uses the labels to choose among the
training-free candidates only (no refiner is fitted). `mode="supervised"` and `"regression"`
need `y`. `augmentation`, `loss`, `nested_dims` or a non-linear `architecture` configure the
refiner, so setting them when no refiner is fitted or compared raises `ValueError`.

#### Fixed configurations

`method="center_l2"`, `"pca"` or `"refiner"` builds that one configuration, with no comparison.
`selected_` names it as the selection would (`"center_l2"`, `"pca-64"`, `"refiner-128"`) and
`selection_report_` is `None`.

- `"pca"` is the `linear` refiner at its initialisation, so `transform`, `dim=`, `save` and `load`
  behave as for a trained model.
- For `"pca"` and `"refiner"`, `target_dim="auto"` is the report's `suggested_target_dim`,
  `clip(1.5 × ID, ID, D)`: `fit` runs the analysis report (uncalibrated, without the ID bootstrap and
  scale curve) on all rows to get it, and keeps it in `analysis_report`. No benchmark supports the
  factor 1.5, and PCA to this dimension lost accuracy on 32 of 35 E4 sets, so pass an explicit
  `target_dim`.
- `strategy` is deprecated and goes in 0.9.0. `strategy="select"` is `method="auto"`;
  `strategy="fixed"` with `method="auto"` is `method="pca"` without labels and `method="refiner"`
  with them. `strategy_` is deprecated too: `selection_report_` is `None` unless a selection ran.

Why PCA rather than a refiner without labels: in E1, E6 and E7 (`docs/benchmarks/BENCHMARK_E1_E2.md`,
`docs/benchmarks/BENCHMARK_E6_E7.md`) the self-supervised refiner never beat the better of PCA and PCA-whitening
chosen on validation (with all training rows: 1 win, 4 ties, 1 loss over 6 datasets), lost to them
when fitted on fewer than a few thousand rows, and cost about 100 times more. Plain PCA rather than
PCA-whitening because choosing between the two needs labels, and on real data plain PCA was never
more than 0.004 behind whitening, while whitening fell up to 0.020 behind PCA. PCA prefixes are
nested by construction, so `transform(X, dim=m)` is PCA at `m`.

**Refiner configuration** (`method="refiner"`, or the refiner candidate of the selection):

| Setting (`"auto"`) | `mode="self_supervised"` | `mode="supervised"` | `mode="regression"` |
|---|---|---|---|
| `architecture` | `"linear"` | `"residual"` | `"mlp"` |
| `augmentation` | `KNNPairs(k=5)` | `KNNPairs(k=5)` | `GaussianNoise(std=0.05)` |
| `loss` | `NTXentLoss(0.07)` + 0.5 `AlignUniformLoss` | `SupConLoss(0.07)` | `RankNContrastLoss()` |
| `monitor` | `k_skewness` (logged only) | `val_knn_accuracy` | `val_ridge_r2` |
| `early_stopping_patience` | `None` | 10 (10% of rows held out, best weights restored) | 10 (as supervised) |

After `fit`, `architecture_`, `monitor_`, `early_stopping_patience_` and `nested_dims_` hold the
resolved values. Without labels the refiner trains for a fixed budget of `epochs` (default 100)
and does not stop early: in E2 no label-free statistic tracked downstream accuracy within a run.
`early_stopping_patience` counts evaluations, every `eval_every` epochs; with early stopping
`val_split` defaults to 0.1 of the rows, held out from training, and without it to 0.

Why these defaults: benchmark E8 (`docs/benchmarks/BENCHMARK_SUPERVISED.md`). With class labels, the residual
adapter (a PCA-initialised linear map plus a zero-initialised MLP) matched the plain MLP on full
training sets and did not collapse with 1-5 labels per class as the MLP did. Against the raw
embeddings it had higher kNN accuracy on 4 of 6 datasets and lower on none, by +0.003 to +0.021 on
real data; against LDA it won on 3 of 6 and lost on none. With 5 labels per class LDA was ahead of
every refiner by 0.04 to 0.08, and with 1 label per class the refiner could fall below the raw
embeddings. For regression the MLP with noise views beat the linear refiner, the raw embeddings and
PCA on all 4 datasets; with 1000 training rows or fewer no refiner reliably beat the raw embeddings,
and a fixed regression refiner warns.

**Class:** `EmbedKit(mode="auto", method="auto", augmentation="auto", loss="auto", target_dim="auto", hidden_dim=256, n_layers=2, architecture="auto", epochs=100, batch_size=256, optimizer="adam", lr=3e-4, weight_decay=1e-4, scheduler="cosine", eval_every=5, early_stopping_patience="auto", monitor="auto", val_split=None, device=None, id_methods=None, random_state=42, nested_dims=None, val_fraction=0.2, selection_metric="cosine", selection_max_rows=50000, strategy=None)` (`strategy` deprecated)

`nested_dims=None` (default), `"auto"` or a sequence trains nested prefixes of a refiner, readable with
`transform(X, dim=m)`; see `MatryoshkaLoss` above.

Attributes after `fit`: `mode_`, `method_`, `selected_` (what was built: `"identity"`,
`"center_l2"`, `"pca-<k>"` or `"refiner-<k>"`), `selection_report_` (`None` unless
`method="auto"` selected with labels), `analysis_report` (`None` for `"auto"` and `"center_l2"`),
and for a refiner `architecture_`, `monitor_`, `early_stopping_patience_`, `nested_dims_`.
`strategy_` is deprecated.

`X` is validated once at `fit` (at least 5 rows; NaN or infinite values raise; constant columns and
duplicate rows warn). `transform(X, dim=m)` keeps the first `m` output coordinates and renormalises
(for `CenterNormalize` and refiner outputs, which are unit-norm); it is meant for PCA and nested
models. `transform` checks the feature count against `fit`.

```python
import numpy as np
from embedkit import EmbedKit

# X: (n, d) embeddings, y: class labels or continuous targets

# Without labels: centred and L2-normalised, nothing trained
ek = EmbedKit()
Z = ek.fit_transform(X)
print(ek.selected_)                         # "center_l2"

# With labels: selection with Identity as the default
ek = EmbedKit(epochs=50).fit(X, y)
Z = ek.transform(X)
print(ek.selected_)                         # "identity", "center_l2", "pca-128", "refiner-128", ...
print(ek.selection_report_[["candidate", "score", "diff_lo", "diff_hi", "selected"]])

# Labels used only to choose among training-free candidates
ek = EmbedKit(mode="self_supervised").fit(X, y)

# Continuous targets: 10-NN R^2 as the criterion, Rank-N-Contrast refiner as a candidate
ek = EmbedKit(mode="regression", epochs=50).fit(X, y.astype(float))

# One fixed configuration
ek = EmbedKit(method="pca", target_dim=64)                # PCA to 64, L2
Z = ek.fit_transform(X)
Z16 = ek.transform(X, dim=16)                             # PCA at 16, renormalised
ek = EmbedKit(method="refiner", target_dim=64, epochs=50)  # contrastive refiner, no labels
Z = ek.fit_transform(X)
ek = EmbedKit(method="refiner", target_dim=32, epochs=50)
Z = ek.fit_transform(X, y=y)                              # SupCon on the residual adapter

# What a fixed configuration auto-configured
print(ek.analysis_report.suggested_target_dim, ek.architecture_)
```

#### Saving and loading

```python
ek.save("my_model/")
ek2 = EmbedKit.load("my_model/")            # device=None: CUDA, then MPS, then CPU
Z_new = ek2.transform(X_new)
```

`save` writes `config.json` (mode, method, input dimension, `selected`, and for a model
its dimensions and architecture) plus the state: `model.pt` for PCA and refiner outputs, and
`arrays.npz` for an `Identity` or `CenterNormalize` result (the training mean; no `model.pt`). It
writes `report.json` only when `analysis_report` exists. `load` restores `method_` and `selected_`;
`selection_report_` is not saved. Bundles written by earlier versions load as before.

### 2.6 Training-free baselines

`embedkit.baselines` holds the transforms every learned method should be compared with. They share
one interface: `fit(X, y=None)` (labels are accepted and ignored), `transform(X)`,
`fit_transform(X, y=None)` and `get_params()`. After `fit`, `n_features_in_`, `n_features_out_` and
the learned statistics are attributes with a trailing underscore; no training rows are stored.
They compute in float64 and return float32, and `transform` raises `NotFittedError` before `fit`
and `ValueError` for a different number of columns.

| Class | Transform |
|---|---|
| `Identity()` | The input unchanged: the "raw embedding" row of a table |
| `CenterNormalize()` | Subtract the training mean, scale each row to unit L2 norm |
| `Standardize()` | Per-column z-score with the training mean and population standard deviation |
| `PCAReduce(n_components, whiten=False)` | Top principal components of the training data, optionally divided by the root eigenvalues. `n_components` is capped at the data rank, with a warning |
| `ZCAWhiten(eps=1e-5)` | Full-rank whitening that stays in the input coordinates |
| `AllButTheTop(n_remove, n_components=None)` | Remove the mean and the top `n_remove` principal directions (Mu & Viswanath, 2018); with `n_components`, the reduce-then-clean variant of Raunak et al. (2019) |
| `ConceptorNegation(alpha=2.0)` | Conceptor negation of Liu et al. (2019) |
| `RandomProjection(n_components)` | Gaussian random projection |

E4 scored these on the 35 natural sets (`benchmarks/results/e4/REPORT.md`). Number of sets on which
the transform was significantly better than the raw embedding, and significantly worse, out of 35
(PCA, PCA-whitening, all-but-the-top and the hubness reductions are the member of the family chosen
on validation; `pca-auto` is PCA to the report's `suggested_target_dim`):

| Transform | cosine kNN up / down | Euclidean kNN up / down |
|---|---|---|
| `CenterNormalize` | 5 / 4 | 16 / 1 |
| `Standardize` | 8 / 6 | 6 / 7 |
| `PCAReduce` | 7 / 5 | 6 / 1 |
| PCA to `suggested_target_dim` | 0 / 32 | 0 / 28 |
| `PCAReduce(whiten=True)` | 3 / 26 | 2 / 24 |
| `ZCAWhiten` | 1 / 34 | 0 / 35 |
| `AllButTheTop` | 5 / 9 | 0 / 15 |
| Hubness reductions (section 2.7) | 10 / 6 | 14 / 3 |

Whitening hurt on most sets, and nothing was better than raw on a majority. A transform that is
fitted on labelled data can still pay off: that is what the selection of section 2.5 tests on held-out
rows.

```python
from embedkit.baselines import CenterNormalize, PCAReduce

cn = CenterNormalize().fit(X_train)
Z_test = cn.transform(X_test)
pca = PCAReduce(32).fit(X_train)
Z_train = pca.transform(X_train)          # (n_train, 32), float32
```

---

### 2.7 Hubness Reduction

`embedkit.hubness_reduction` holds four secondary distances for k-nearest-neighbour search (Feldbauer & Flexer, 2019). They change the distances between points, not their coordinates, so they are not transforms: each is fitted on a reference set (a gallery, or the training split of a kNN classifier) and answers neighbour queries against it.

```python
from embedkit.hubness_reduction import CSLS, DisSimLocal, LocalScaling, MutualProximity

mp = MutualProximity(metric="cosine").fit(Z_train)
dist, idx = mp.kneighbors(Z_test, n_neighbors=10)  # neighbours among the Z_train rows
dist, idx = mp.kneighbors(n_neighbors=10)          # reference rows among each other, self excluded
```

All four start from a base distance `d` (`metric="euclidean"`, or `"cosine"` for `1 - cos`):

| Class | Secondary distance | Statistic |
|---|---|---|
| `LocalScaling(k=5)` (Schnitzer et al., 2012) | `1 - exp(-d² / (s_x s_y))` | `s_x`: distance to the `k`-th nearest reference point |
| `MutualProximity(sample_size=2000)` (Schnitzer et al., 2012; Gaussian variant) | `1 - P(D_x > d) P(D_y > d)` | `D_x`: a normal fitted to the distances from `x` to a sample of the reference set |
| `CSLS(k=10)` (Conneau et al., 2018) | `2d - r_x - r_y` | `r_x`: mean distance to the `k` nearest reference points; with the cosine metric this ranks as the original `2 cos - r_x - r_y` |
| `DisSimLocal(k=5)` (Hara et al., 2016) | `‖x - y‖² - ‖x - c_x‖² - ‖y - c_y‖²` | `c_x`: centroid of the `k` nearest reference points; with the cosine metric it runs on unit vectors |

They are the one fix the report recommends (Euclidean metric, when k-skewness or the antihub ratio is above the null). The evidence is modest: on the E1 sets the best change against raw cosine kNN was +0.005, and in E4 the best reduction was significantly better than raw on 14 of 35 sets under Euclidean kNN and worse on 3 (median +0.4 points). Check them on your own held-out data with `evaluate_transform(..., hubness_reduction=...)`.

The search is exact and chunked to bound memory. A reference row is never its own neighbour when the reference set is queried; a query is a new point, so a duplicate of a reference row is returned.

The kNN-based evaluation metrics take a `hubness_reduction` argument: `knn_accuracy`, `recall_at_k` and `k_skewness` in `embedkit.evaluation.metrics`, and `evaluate_transform(..., hubness_reduction=CSLS())`, which applies it to `knn_accuracy`, `knn_accuracy_euclidean`, `recall_at_10` and `k_skewness` (each with its own base metric) and leaves the other metrics unchanged.

---

## Part 3 — Evaluation and input checks

### 3.1 Metrics

`embedkit.evaluation.metrics` holds pure functions on arrays, each returning a float. Neighbours,
fits and standardisation use the training split only.

| Function | Scores |
|---|---|
| `knn_accuracy(Z_train, y_train, Z_test, y_test, k=5, metric="cosine")` | kNN majority-vote accuracy |
| `knn_regression_r2(...)` | kNN regression R^2 |
| `linear_probe(...)` | Logistic regression (classes) or ridge (continuous); regularisation chosen by cross-validation on the training split, or fixed with `C=` |
| `recall_at_k`, `trustworthiness`, `continuity`, `knn_overlap` | Neighbour preservation between an input and its transform |
| `linear_cka`, `effective_rank`, `k_skewness`, `nc1`, `silhouette` | Geometry of one embedding |

`knn_accuracy`, `linear_probe` and `recall_at_k` accept `return_per_point=True` and then also return the
per-test-point outcomes, which `compare` bootstraps. Note that `knn_overlap` rates an unchanged input
as perfect, so it cannot tell whether a transform helps.

### 3.2 Evaluation protocol

`evaluate_transform(make_transform, X_train, X_test, y_train, y_test, seeds=...)` fits a fresh transform
per seed on the training split (`make_transform(seed)` returns an unfitted object with `fit` and
`transform`; `transductive=True` fits on both splits, recorded; `fit_with_labels=True` passes
`y_train` to `fit`), scores the test split, and returns a tidy DataFrame with one row per seed and
metric, plus fit and transform times, the output dimension and the parameters. `compare(results,
reference="raw")` gives mean and sd over seeds per method and metric, and the difference to the reference with a
paired bootstrap interval over test points (2,000 resamples, 95%) where per-point outcomes exist.
The metrics come from the registry `embedkit.evaluation.METRICS`.

```python
from embedkit import EmbedKit
from embedkit.baselines import Identity, PCAReduce
from embedkit.evaluation import compare, evaluate_transform

splits = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, seeds=[0])
results = [
    evaluate_transform(lambda s: Identity(), method="raw", **splits),
    evaluate_transform(lambda s: PCAReduce(32), method="pca-32", **splits),
    evaluate_transform(lambda s: EmbedKit(epochs=20, random_state=s), method="embedkit",
                       fit_with_labels=True, **splits),
]
print(compare(results, reference="raw"))
```

A difference counts as a win or loss only when its interval excludes 0. The benchmarks in
`benchmarks/` use this protocol (one JSON file per data set and method, selection on a validation
split, the test split scored once).

### 3.3 Input validation

`embedkit.utils.validation.validate_embeddings(X, *, name, min_samples, n_features, warn, dtype)` is
the one check of an embedding matrix, called at the public entry points (the analyzers,
`EmbedKitAnalyzer.fit`, `EmbedKit.fit`, `Trainer.fit`, the baselines' `fit` and the hubness
reductions). It takes an array or tensor and returns a 2-D array. It raises `ValueError` for input
that is not 2-D, too few rows, a wrong feature count, and NaN or infinite values (the message gives
the number of entries and rows). It issues one `UserWarning` for constant columns and one for exact
duplicate rows, each with its count. `transform` calls check shape and values but not the two
warnings. Calls inside another entry point skip the scans, so each warning appears once per user
call. float64 input is rounded to float32 by the analyzers, `EmbedKit` and `Trainer`; the baselines and
hubness reductions keep float64.

### 3.4 Nearest neighbours

`embedkit.utils.neighbors.knn(X, k, metric=...)` is the shared exact kNN search. It uses scikit-learn
by default; FAISS is optional (`pip install embedkit[faiss]`, then `backend="faiss"` or the
environment variable `EMBEDKIT_KNN_BACKEND=faiss`). Every backend returns true distances
(Euclidean, or `1 - cos`), ascending, without the query point; the input is not modified. `k`
outside `1 ... n - 1` raises.

### 3.5 Time and memory

Measured in experiment E9 (`benchmarks/e9.py`, `benchmarks/results/e9/REPORT.md`) on 16 CPUs of a
cluster node (no GPU), with synthetic float32 data (10 classes, power-law spectrum). Each entry is
the wall time of one `fit` in a fresh process and the peak resident memory of that process (GB =
1e9 bytes); the peak includes the interpreter and torch (about 0.6 GB) and the input itself
(16.4 GB at n = 1e6, d = 4,096).

`EmbedKitAnalyzer().fit(X)` (the defaults: calibration with 20 null draws, ID bootstrap and scale
curve):

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 36 s / 0.8 GB | 37 s / 0.9 GB | 1.3 min / 1.2 GB |
| 10,000 | 57 s / 1.0 GB | 1.0 min / 1.1 GB | 2.9 min / 2.0 GB |
| 100,000 | 1.1 min / 1.1 GB | 1.5 min / 2.2 GB | 3.1 min / 7.1 GB |
| 1,000,000 | 56 s / 2.6 GB | 1.6 min / 17.0 GB | 4.1 min / 66.1 GB |

`EmbedKitAnalyzer(calibrate=False, id_bootstrap=0, id_scale_curve=False).fit(X)`:

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 4.6 s / 0.8 GB | 6.3 s / 0.9 GB | 6.7 s / 1.1 GB |
| 10,000 | 6.7 s / 1.0 GB | 7.1 s / 1.1 GB | 27 s / 1.9 GB |
| 100,000 | 5.9 s / 1.1 GB | 10.5 s / 2.2 GB | 41 s / 7.1 GB |
| 1,000,000 | 7.9 s / 2.6 GB | 17.4 s / 17.0 GB | 1.2 min / 66.1 GB |

The analyzer's time hardly grows with n: the neighbour statistics use a subsample of `n_max`
(20,000) rows, the ID estimators 5,000 and the calibration 2,000. Its memory does grow, to about
four times the input at d = 4,096, because the input is copied (float32 conversion, centring)
before it is subsampled. Calibration accounts for most of the time at small n.

`EmbedKit().fit(X)` without labels (centre and L2-normalise):

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | < 0.1 s / 0.6 GB | < 0.1 s / 0.6 GB | < 0.1 s / 0.7 GB |
| 10,000 | < 0.1 s / 0.6 GB | 0.1 s / 0.7 GB | 0.3 s / 1.1 GB |
| 50,000 | 0.1 s / 0.7 GB | 0.3 s / 1.0 GB | 1.1 s / 2.3 GB |
| 100,000 | 0.1 s / 0.7 GB | 0.6 s / 1.2 GB | 2.2 s / 3.1 GB |
| 1,000,000 | 1.1 s / 1.2 GB | 5.9 s / 4.9 GB | 22 s / 17.8 GB |

Memory stays close to the input itself: at 1e6 × 4,096 the peak is about 1.4 GB above the
input, the interpreter included.

`EmbedKit().fit(X, y)` with class labels (selection, with the supervised refiner as a candidate):

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 10 s / 1.0 GB | 11 s / 1.0 GB | 15 s / 1.2 GB |
| 10,000 | 28 s / 1.2 GB | 37 s / 1.4 GB | 2.2 min / 3.4 GB |
| 50,000 | 1.9 min / 2.8 GB | 2.6 min / 3.6 GB | 6.4 min / 11.0 GB |
| 100,000 | 1.9 min / 2.8 GB | 2.6 min / 3.8 GB | 6.4 min / 11.8 GB |
| 1,000,000 | 1.9 min / 3.3 GB | 2.7 min / 7.5 GB | 6.7 min / 26.5 GB |

Above `selection_max_rows` rows (50,000 by default) the candidates are fitted and scored on a
stratified subsample of that size, so the selection's time, and its memory beyond the input,
stop growing with n: about 2.2 GB at d up to 1,024 and 9.5 GB at d = 4,096. In these
measurements `Identity` won every time, so nothing was refitted. When the refiner or PCA wins,
it is fitted again on all the rows, and that refit does grow with n: the refiner trains on every
row, and PCA makes a float64 copy of the input. Before 0.7.0 the selection always used all the
rows; at a million rows it took 1.8 to 9.7 hours on the same CPUs.

On a GPU (Quadro RTX 8000, 20 CPUs), at n = 1e6:

| | 128 | 1,024 | 4,096 |
|---|---|---|---|
| `selection_max_rows=50_000` (default) | 1.4 min / 4.0 GB | 2.1 min / 8.2 GB | 5.0 min / 26.6 GB |
| `selection_max_rows=None` (all rows) | 2.8 h / 10.5 GB | 4.2 h / 46.1 GB | over 5 h; not completed |

Selecting on a subsample costs little in quality. On 64 datasets, selecting on 10,000 or 20,000
rows picked a transform as good as selecting on all of them wherever the dataset was larger than
the cap (40 and 11 datasets); with 5,000 rows the choice was significantly worse on 6 of 47
datasets (1.7 to 6.2 points of kNN accuracy), so do not set the cap much below 10,000. Details: `benchmarks/results/e9b/REPORT.md`, `benchmarks/results/e9b_gpu/REPORT.md` and
`benchmarks/results/e10/REPORT.md`.

---

## Part 4 — Visualization

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
fig = viz.plot_knn_graph(X, k=10, highlight_hubs=report.hubness.hubs)   # indices of the points to colour
```

---

## References

Agrawal, K. K., Mondal, A. K., Ghosh, A., & Richards, B. A. (2022). α-ReQ: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. In *Advances in Neural Information Processing Systems* (Vol. 35).

Albergante, L., Bac, J., & Zinovyev, A. (2019). Estimating the effective dimension of large biological datasets using Fisher separability analysis. In *2019 International Joint Conference on Neural Networks (IJCNN)*. IEEE.

Amsaleg, L., Chelly, O., Furon, T., Girard, S., Houle, M. E., Kawarabayashi, K., & Nett, M. (2015). Estimating local intrinsic dimensionality. In *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 29–38). ACM.

Beyer, K. S., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? In *Proceedings of the 7th International Conference on Database Theory (ICDT)* (pp. 217–235). Springer.

Ceruti, C., Bassis, S., Rozza, A., Lombardi, G., Casiraghi, E., & Campadelli, P. (2012). DANCo: An intrinsic dimensionality estimator exploiting angle and norm concentration. In *Proceedings of the 21st International Conference on Pattern Recognition (ICPR 2012)* (pp. 1045–1048). IEEE.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119.

Conneau, A., Lample, G., Ranzato, M., Denoyer, L., & Jégou, H. (2018). Word translation without parallel data. In *International Conference on Learning Representations (ICLR 2018)*.

Cristianini, N., Shawe-Taylor, J., Elisseeff, A., & Kandola, J. S. (2001). On kernel-target alignment. In *Advances in Neural Information Processing Systems* (Vol. 14, pp. 367–373). MIT Press.

Denti, F., Doimo, D., Laio, A., & Mira, A. (2022). The generalized ratios intrinsic dimension estimator. *Scientific Reports*, 12, 20005.

DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with Cutout. *arXiv preprint arXiv:1708.04552*.

Feldbauer, R., & Flexer, A. (2019). A comprehensive empirical comparison of hubness reduction in high-dimensional spaces. *Knowledge and Information Systems*, 59, 137–166.

Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7, 12140.

Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple contrastive learning of sentence embeddings. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)* (pp. 6894–6910).

Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of strange attractors. *Physica D: Nonlinear Phenomena*, 9(1–2), 189–208.

Hara, K., Suzuki, I., Kobayashi, K., Fukumizu, K., & Radovanović, M. (2016). Flattening the density gradient for eliminating spatial centrality to reduce hubness. In *Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016)* (pp. 1659–1665).

Johnsson, K., Soneson, C., & Fontes, M. (2015). Low bias local intrinsic dimension estimation from expected simplex skewness. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 37(1), 196–202.

Khosla, P., Tian, Y., Wang, X., Liu, C., Isola, P., & others (2020). Supervised contrastive learning. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 18661–18673). Curran Associates.

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022). Matryoshka representation learning. In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 30233–30249).

Levina, E., & Bickel, P. J. (2005). Maximum likelihood estimation of intrinsic dimension. In *Advances in Neural Information Processing Systems* (Vol. 17, pp. 777–784). MIT Press.

Liu, T., Ungar, L., & Sedoc, J. (2019). Unsupervised post-processing of word vectors via conceptor negation. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33).

Mu, J., Bhat, S., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. In *International Conference on Learning Representations (ICLR 2018)*.

Raunak, V., Gupta, V., & Metze, F. (2019). Effective dimensionality reduction for word embeddings. In *Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP)*.

Radovanović, M., Nanopoulos, A., & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research*, 11, 2487–2531.

Rudman, W., Gillman, N., Rayne, T., & Eickhoff, C. (2022). IsoScore: Measuring the uniformity of embedding space utilization. In *Findings of the Association for Computational Linguistics: ACL 2022*.

Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). Local and global scaling reduce hubs in space. *Journal of Machine Learning Research*, 13, 2871–2902.

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015)* (pp. 815–823). IEEE.

van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

Timkey, W., & van Schijndel, M. (2021). All bark and no bite: Rogue dimensions in transformer language models obscure representational quality. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

Wang, T., & Isola, P. (2020). Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119.

Yeh, C.-H., Hong, C.-Y., Hsu, Y.-C., Liu, T.-L., Chen, Y., & LeCun, Y. (2022). Decoupled contrastive learning. In *European Conference on Computer Vision (ECCV 2022)* (pp. 668–684). Springer.

Zha, K., Cao, P., Son, J., Yang, Y., & Katabi, D. (2023). Rank-N-Contrast: Learning continuous representations for regression. In *Advances in Neural Information Processing Systems* (Vol. 36). Curran Associates.

Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. In *International Conference on Learning Representations (ICLR 2018)*.
