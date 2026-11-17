# EmbedKit Metric Guide

Detailed interpretation of every metric in `EmbedKitReport`, including rough reading ranges,
which statistics the report judges, geometric meaning, and the downstream consequences.

Since 0.5.0 the report does not apply fixed pathology thresholds, with one exception (the
ID / D ratio). Hubness, antihubs, distance concentration, neighbour consistency and uniformity are
compared with a null model (see "Calibration" below), and recommendations about them come only from
z-scores. The ranges in the tables below are rough guides for reading a number; the report does not
use them.

---

## Intrinsic Dimension (`report.intrinsic_dim`)

| Attribute | Type | Meaning |
|---|---|---|
| `estimates` | dict[str, float] | Per-method estimates (NaN when a method failed). The methods measure different things and are not combined |
| `standard_errors` | dict[str, float] | Bootstrap standard error per method (NaN for `DANCo`, `ESS`, failed methods, or `id_bootstrap=0`) |
| `intervals` | dict[str, tuple] | `estimate ± 1.96 se`; sampling error only, no bias |
| `between_method_spread` | float | Std of the estimates across methods (formerly `uncertainty`); not a sampling uncertainty |
| `scale_curve` | dict[int, float] | TwoNN on n, n/2, n/4, n/8 rows, `{n_rows: estimate}` |
| `scale_max_deviation` | float | Largest relative deviation of a level from the full-sample TwoNN; no threshold is applied. A curve that rises with n points to noise. Clean manifolds drift by about 8% at ID 20 |
| `reference_method`, `reference_estimate` | str, float | The one estimate the report uses: TwoNN if requested and successful, else the first successful method |
| `local_estimates` | dict[str, ndarray] | Per-point estimates (TwoNN, MLE, lPCA) |
| `n_duplicates_removed` | int | Exact duplicate rows dropped before estimation |

**Methods available:** `TwoNN`, `MLE`, `lPCA`, `DANCo`, `CorrInt`, `MOM`, `FisherS`, `ESS`
(pass `id_methods=[...]` to `EmbedKitAnalyzer`; default `TwoNN`, `MLE`, `lPCA`, `MOM`). `ESS` and
`DANCo` take about 100 s per fit at n = 5,000.

The k-NN estimators (TwoNN, MLE, MOM) read low at high ID (15 to 17 on a 20-sphere, n = 5,000);
`lPCA` returns d + 1 for a d-sphere. The intervals do not cover these biases.

### Key ratio: reference ID / ambient dimension (D)

| ID/D ratio | Report | `suggested_target_dim` |
|---|---|---|
| `< 0.1` | Recommendation: very sparse, strong dimensionality reduction advised | `clip(1.5 × ID, max(4, ID), D)` |
| `0.1 – 0.3` | Recommendation: moderate, reduction may help | same |
| `≥ 0.3` | No recommendation | same |

No benchmark supports the factor 1.5. If the estimate is unreliable (small n, noisy data), pass an
explicit `target_dim`.

---

## Hubness (`report.hubness`)

Hubness measures how unevenly points appear as k-nearest neighbors of other points.
In high-dimensional spaces, a few "hub" points dominate the kNN graph regardless of
semantic similarity, breaking retrieval and kNN-based classifiers.

| Attribute | Judged by the report | Rough reading | Meaning |
|---|---|---|---|
| `k_skewness` | yes: z > 3 gives a recommendation | near 0 | Skewness of the k-occurrence distribution |
| `antihub_ratio` | yes: z > 3, Euclidean metric only | depends on n, d, k, norms | Fraction of points never retrieved as neighbors (isotropic Gaussian at d = 1,024: 0.21 to 0.39 Euclidean, none cosine) |
| `robinhood_index` | no | `> 0.3` concerning | Gini-like inequality; 0 = uniform, 1 = monopoly |
| `hub_ratio` | no | `> 0.1` concerning | Fraction of points with `N_k > hub_threshold × mean N_k` (default 2) |
| `hub_contamination` | no | `> 0.3` concerning | Fraction of the hubs' own neighbors that are also hubs |
| `k_occurrence`, `hubs`, `antihubs` | no | | Raw per-point counts and index arrays |

Under `metric="cosine"` the antihub z-score is reported in `report.calibration` but raises no
recommendation: the antihub ratio of Gaussian data is not calibrated there.

### What hubness does to downstream tasks

- **Retrieval / RAG**: Hub documents are returned for almost every query, drowning out
  relevant results.
- **kNN classification**: Decision boundaries are dominated by hub labels.
- **Clustering**: Hubs act as artificial attractors, distorting cluster assignments.

### Fix

- The report's recommendation: a hubness-reducing neighbour search such as
  `embedkit.hubness_reduction.MutualProximity` (also `LocalScaling`, `CSLS`, `DisSimLocal`).
- Training with `AlignUniformLoss` (uniformity term) is a possible remedy but not benchmarked for
  hubness (E4 pending).

---

## Geometry (`report.geometry`)

### Distance Concentration (`report.geometry.distance_concentration`)

| Attribute | Rough reading | Meaning |
|---|---|---|
| `distance_cv` | larger is better; about `1/sqrt(2d)` for iid Gaussian | `std(D) / mean(D)` of pairwise distances. The statistic the report judges: z < -3 gives a recommendation |
| `relative_contrast` | `> 1.0` healthy, `< 0.1` narrow spike | `(d_max − d_min) / d_mean` |
| `relative_contrast_per_query` | larger is better | Median over query points of `(d_max − d_min) / d_min`, zero distances excluded |
| `concentration_ratio` | not used | `d_min / d_max`; one duplicated pair sets it to 0 |
| `distance_histogram`, `bin_edges` | | Histogram of sampled pairwise distances |

**Effect:** When all points are nearly equidistant, nearest-neighbor search becomes
meaningless — all distances are indistinguishable. This is the curse of dimensionality in its
most acute form.

**Fix (not benchmarked):** a lower-dimensional output (`target_dim`).

---

### Isotropy (`report.geometry.isotropy`)

Described in `report.properties` (spectrum, dominant coordinates, mean direction); the report makes
no recommendation from these. Anisotropy is a property, not a defect: strict isotropy is at odds
with cluster structure, and whitening can hurt classification.

| Attribute | Meaning |
|---|---|
| `participation_ratio` | `(Σλ)² / Σλ²`, in `[1, D]`; plug-in value, biased low by about `1 / (1 + D/n)` |
| `participation_ratio_corrected`, `participation_ratio_normalized` | Unbiased-`tr(Σ²)` version (Gaussian model) and its ratio to D, in `[0, 1]`; the n-robust summary |
| `effective_rank` | `exp(entropy)` of the normalised eigenvalues; plug-in, biased low when n is not much larger than D |
| `isotropy_score` | `effective_rank / min(n − 1, D)`, in `[0, 1]`; same bias |
| `isoscore` | IsoScore (Rudman et al.), `(PR − 1) / (D − 1)`; biased low (iid data at n = 300, D = 512 score 0.37) |
| `top_k_variance_share` | Share of variance in the top 1, 3, 10 principal components |
| `top_k_coordinate_share` | The same for the 1, 3, 10 highest-variance coordinates (rogue dimensions) |
| `spectrum_alpha` | Power-law decay exponent of the eigenvalues, `λ_i ∝ i^−α` |
| `mean_vector_ratio`, `mean_cosine` | Shared offset: `‖mean x‖ / mean ‖x‖`, and average pairwise cosine (uncentred) |
| `eigenvalue_spectrum_top`, `explained_variance_ratio` | Top `n_components` eigenvalues, for plotting (`AnalysisPlotter.plot_eigenvalue_spectrum()`); `eigenvalue_spectrum` is a deprecated alias |

**Effect:** Anisotropic embeddings concentrate variance in a few directions, and a shared mean
vector inflates cosine similarity between unrelated rows. Whether `AlignUniformLoss`, whitening or
mean removal helps is an open question (E4).

---

### Neighbor Consistency (`report.geometry.neighbor_consistency`)

| Attribute | Rough reading | Meaning |
|---|---|---|
| `mean_consistency` | near 1 robust, `< 0.5` fragile | Avg fraction of kNN preserved under small Gaussian perturbation (`0.01 ×` the RMS per-coordinate sd). Judged by the report: z < -3 gives a recommendation |
| `std_consistency` | low | Variability in consistency across perturbation trials |
| `per_perturbation` | | Per-trial consistency scores |

**Effect:** Low consistency means small changes in input values cause large changes
in the kNN graph. The local structure is fragile — models trained on it are not robust.

**Fix (not benchmarked):** `GaussianNoise(adaptive=True)` or `EmbeddingMixup` augmentations.

---

### Uniformity (`report.geometry.uniformity`)

| Attribute | Rough reading | Meaning |
|---|---|---|
| `uniformity` | `< −2.0` spread, `> −1.0` clustered | `log E[exp(−t·||z_i − z_j||²)]` on L2-normalised rows — Wang & Isola uniformity |

Lower (more negative) is more uniform. A score near 0 means embeddings are collapsed to nearly
the same point; a score around −3 to −5 indicates well-distributed embeddings. The report
compares it with the null and states the direction in `report.properties` ("more clustered than"
or "more spread than" the null); it makes no recommendation.

**Effect:** Poorly uniform embeddings form dense clusters in embedding space, which
hurts retrieval diversity and model calibration. `AlignUniformLoss` optimizes this term directly.

---

## Kernel Diagnostics (`report.kernel`)

Applies an RBF kernel `K(x_i, x_j) = exp(−||x_i − x_j||² / (2σ²))` on a subsample and analyzes
the spectrum of the centred kernel `Kc = HKH`. Sigma is the median pairwise distance
(`suggested_sigma` in the report).

| Attribute | Meaning |
|---|---|
| `sigma` | RBF bandwidth used |
| `effective_rank` | `tr(Kc)² / tr(Kc²)`: trace-ratio rank of the centred kernel, in `[1, subsample − 1]` |
| `spectral_gap` | `λ₁ / λ₂` of `Kc`, in `[1, ∞)` |
| `top_k_eigen_ratio` | `λ₁ / λ_k` of `Kc`, with k the number of eigenvalues kept (`n_components`, 50). `condition_number` is a deprecated alias; it is not a condition number |
| `row_sum_skewness` | Skewness of the uncentred kernel's row sums; continuous analog of hubness |
| `centered_kernel_alignment` | Centred alignment with the label kernel, in `[-1, 1]` (only if `y` passed to `fit`). `kernel_alignment` is a deprecated alias |
| `eigenvalues` | Top `n_components` eigenvalues of `Kc` |

### Interpretation

| Signal | Meaning |
|---|---|
| Large `effective_rank` | Variance spread over many kernel directions; depends on σ (small σ makes every point its own direction), so compare data sets of the same size with `sigma="median"` |
| Small `effective_rank` | Variance concentrated in a few directions of the kernel feature space |
| Large `spectral_gap` (say > 5) | One direction dominates, for example two well-separated clusters |
| Large `top_k_eigen_ratio` | Steep eigenvalue decay |
| `centered_kernel_alignment` near 0 | Labels independent of the RBF geometry |
| `centered_kernel_alignment` near 1 | The RBF geometry reflects the labels |

No thresholds are applied to any kernel statistic.

---

## Calibration (`report.calibration`, `report.n_outside_null`)

`EmbedKitAnalyzer(calibrate=True)` (the default) compares five statistics (`k_skewness`,
`antihub_ratio`, `distance_cv`, `neighbor_consistency`, `uniformity`) with their distribution
under a null model (`null_model="gaussian"`: a Gaussian with the data's mean and covariance
spectrum; or `"permuted"`: each column permuted). The comparison runs on a subsample of
`n_calibration` rows (2,000) against `n_null_draws` (20) null data sets.

- `report.calibration[name]` holds `observed`, `null_mean`, `null_std`, `z_score`.
- `report.calibration_k` is the `k` used: `k` if given, else `max(5, round(sqrt(rows)/2))` on the
  calibration rows, so z-scores do not change with the size of `X`.
- `report.n_outside_null` counts statistics with `|z| > z_threshold` (3). It replaces the removed
  `severity`.
- Recommendations: `k_skewness` z > 3 (more hubness), `antihub_ratio` z > 3 (Euclidean only),
  `distance_cv` z < -3 (more concentrated), `neighbor_consistency` z < -3 (less stable). Plus the
  ID / D rule. With `calibrate=False` the report makes no recommendation about these statistics.
- A large |z| means "unlike a Gaussian with the same spectrum", not necessarily "defective".
- Rows with a common norm (normalised upstream) are compared with null draws normalised to that
  norm; this calibrates flat-spectrum rows only (a power-law spectrum on the sphere gives
  spurious z of +8 to +11).
- The high-level `EmbedKit` runs the report uncalibrated and reads only `suggested_target_dim`.

---

## Summary: Reading Checklist

When reviewing a report, scan these in order:

1. **Calibration** — Is `report.n_outside_null > 0`? Which statistics, in which direction?
   Read `report.recommendations`.
2. **Hubness** — `calibration["k_skewness"].z_score` and (Euclidean) `["antihub_ratio"]`.
3. **Dimensionality** — Is `reference_estimate / D < 0.3`? Do the per-method `estimates` and
   `intervals` agree, and is `scale_max_deviation` small?
4. **Distance concentration** — `calibration["distance_cv"].z_score` below -3?
5. **Stability** — `calibration["neighbor_consistency"].z_score` below -3?
6. **Properties** — read `report.properties` (spectrum, mean direction, uniformity against the
   null) as description, not as defects.
7. **Label structure** — If labels were passed, is `centered_kernel_alignment` well above 0?

If nothing is beyond its null range and the ID / D ratio is at least 0.3, the report says so
("No recommendation") and simple refinement (or none) is appropriate.
