# Changelog

## Unreleased

### Changed (WP 7.2: selection-based configuration)
- `EmbedKit` has `strategy="select"` as the default (`"fixed"` is the 0.5.0 behaviour; any
  explicit `method` implies it). Without labels it returns the input centred and L2-normalised
  (`CenterNormalize`): no label-free score can tell whether a transform helps (`knn_overlap`
  gives the unchanged input the top score), and in E4 PCA to `suggested_target_dim`, the 0.5.0
  default, lost accuracy on 32 of 35 natural sets, while centring + L2 was the most consistent
  small gain on Euclidean kNN (16 up, 1 down).
- With labels, `fit(X, y)` holds out 20% of the rows (`val_fraction`, stratified), fits
  `Identity`, `CenterNormalize`, PCA at 64, 128 and 256 below d (L2-normalised) and, in
  supervised and regression mode, the refiner of the mode at `min(128, d // 2)`, scores each by
  10-NN accuracy (classes) or 10-NN regression R^2 on the held-out rows (`selection_metric`,
  cosine by default), and keeps `Identity` unless a candidate beats it with a paired bootstrap
  interval (95%, 2,000 resamples) above 0. The winner is refitted on all rows. New attributes:
  `selected_`, `selection_report_` (the table), `strategy_`, `mode_`. Fewer than 20 rows to fit
  or to score: `Identity` with a warning.
- `mode="auto"` is the new default: self-supervised without labels, supervised with them;
  continuous targets need `mode="regression"`.
- Under `strategy="select"` no analysis report is computed (`analysis_report` is `None`); no
  decision reads it.
- `EmbedKitReport.suggested_k` is now `report.k` and `suggested_sigma` is `report.kernel.sigma`;
  the old names warn (`DeprecationWarning`) and go in 0.7.0. Nothing in the library read them.
- Bundles of an `Identity` or `CenterNormalize` result store `arrays.npz` (no `model.pt`);
  `save_bundle` takes `model=None` and `arrays=`.

### Changed (WP 7.1: experiment E4)
- The report's recommendations are cut to the rules E4 validated. E4 ran the report and every
  training-free fix (centring, standardising, ZCA, conceptors, PCA and PCA-whitening at E1's
  dimensions and at `suggested_target_dim`, ABTT, the four hubness reductions) on 35 natural
  classification sets (10 source data sets, 3 or 4 encoders each). A rule stays when the gain of
  its fix rises with its signal (Spearman interval over 2,000 bootstrap resamples of source data
  sets excluding 0) and the fix beats the raw embedding significantly on at least half of the
  sets where the rule fires (criteria fixed before the runs; `benchmarks/results/e4/REPORT.md`).
- Kept, under `metric="euclidean"` only: k-skewness above the null (Spearman 0.36, 95% interval
  0.13 to 0.65; significant Euclidean kNN gain on 3 of 3 firing sets) and the antihub ratio
  above the null (0.32, 0.05 to 0.58; 5 of 8). Both now name a hubness-reducing neighbour search.
- Removed: hubness under cosine (Spearman 0.05, -0.24 to 0.39), distance concentration (-0.50
  with PCA on Euclidean kNN: the gain falls as the signal rises), neighbour consistency, and the
  fixed ID / D rule (0.1 and 0.3), which fired on all 35 sets while PCA to
  `suggested_target_dim` lost accuracy on 32 of them (median -5.1 points on cosine kNN). The
  statistics stay in `report.calibration` and `report.intrinsic_dim`. The review's hypotheses
  about the mean direction, the top components, anisotropy and dominant coordinates (H1 to H4)
  did not predict their fixes either, so `report.properties` stays descriptive, with no
  thresholds.
- No-action message: "No recommendation: no statistic with a validated rule is beyond its null
  range."

### Added (WP 7.1)
- E4 harness: `<data>_<encoder>` feature builders for 31 new sets (`extract_features.py`),
  `e4_methods` and `run.py --experiment e4`, `benchmarks/e4.py`, `benchmarks/e4_data.py`,
  `benchmarks/report_e4.py`, `benchmarks/cluster/e4.slurm.sh`; results in
  `benchmarks/results/e4/` (report and summary tables; per-run files stay on Occidata).

### Added (WP 7.3)
- `embedkit.utils.validation.validate_embeddings(X, *, name, min_samples, n_features, warn,
  dtype)`: the one check of an embedding matrix at the public entry points (ndarray or tensor in,
  2-D ndarray out). It raises `ValueError` for non-2-D input, too few rows, a wrong feature count
  and NaN or infinite values (the message gives the number of entries and rows), and issues one
  `UserWarning` for constant columns and one for exact duplicate rows, each with its count.
  Duplicates are found by hashing rows (`count_duplicate_rows`), about 4 times faster than a void-view
  `np.unique` at n = 250,000, d = 4,096 (1.4 s against 5.3 s).
- Called from `EmbedKitAnalyzer.fit`, `IntrinsicDimensionEstimator`, `HubnessAnalyzer`,
  `DistanceConcentration`, `IsotropyAnalyzer`, `NeighborConsistency`, `UniformityScore`,
  `KernelDiagnostics`, `EmbedKit.fit` / `fit_transform`, `Trainer.fit` and every baseline's `fit`.
  Calls that run inside another entry point (the analyzers inside `EmbedKitAnalyzer.fit`, the
  trainer inside `EmbedKit.fit`, the monitors during training) skip the scans, so each warning
  appears once per user call.

### Changed (WP 7.3)
- NaN or infinite values now raise in the analyzers, `EmbedKit`, `Trainer` and the hubness
  reductions (`embedkit.hubness_reduction`), which used to pass them on. `Trainer.transform`,
  `EmbedKit.transform` and the hubness reductions' queries check values and shape;
  `Trainer.transform` and `EmbedKit.transform` also check the feature count against the model.
- Constant columns and duplicate rows in the training matrix now warn (`transform` does not).
- float64 input: unchanged for the analyzers, `EmbedKit` and `Trainer` (rounded to float32, as
  `_to_numpy` always did); the baselines and hubness reductions, which compute in float64, now
  keep float64 input as given instead of passing it through float32. Results for float32 input
  are unchanged.

## 0.5.0 (unreleased)

Phase 6 of `docs/IMPLEMENTATION_PLAN.md`: supervised and regression modes. Phase 4, WP 4.1:
null-calibrated statistics in the report; WP 4.2: intrinsic dimension per estimator; WP 4.3:
isotropy and spectrum statistics, reported as properties; WP 4.4: one metric in every analyzer
and the centred kernel spectrum; WP 4.5: benchmarks E3 and E5 (`docs/BENCHMARK_DIAGNOSTICS.md`)
and the calibration fixes they led to.

### Breaking changes since 0.4.0 (summary; details below)
- `EmbedKitReport.severity` is removed; use `n_outside_null` and `calibration` (WP 4.1).
- `IntrinsicDimensionResult.consensus` and the `aggregate` argument are removed, and
  `uncertainty` is renamed `between_method_spread` (WP 4.2).
- `EmbedKitAnalyzer(metric=...)` accepts only `"euclidean"` and `"cosine"` (WP 4.4).
- Recommendations: anisotropy statements moved to `report.properties` (WP 4.3); the fixed
  hubness, concentration, consistency, uniformity (WP 4.1) and antihub (WP 4.5) thresholds are
  replaced by z-scores against a null model, so `calibrate=False` makes none of them.
- `KernelDiagnosticsResult.effective_rank`, `spectral_gap` and `eigenvalues` come from the centred
  kernel; `condition_number` is a deprecated alias of `top_k_eigen_ratio` (WP 4.4).
- Supervised and regression `EmbedKit` defaults changed (architecture, loss, early stopping; phase
  6).

### Changed (WP 4.5)
- **The Gaussian null's mean is shrunk** so that its squared norm is unbiased
  (`null.mean_shrinkage`): centred on the raw sample mean, zero-mean data gave uniformity z of -6
  to -19. On flat Gaussians uniformity is now inside +-3 in 98% of E3 cells (was 31%).
- **Calibration `k` no longer depends on the size of `X`.** It is `k` when given, otherwise
  `max(5, round(sqrt(rows) / 2))` on the `n_calibration` rows (`report.calibration_k`; 22 for
  2,000 rows). Reusing the report's `k` (71 at 20,000 rows) raised the neighbour-consistency z on
  the same 2,000 rows from 9 to 23.
- **Breaking: the fixed antihub rule (`antihub_ratio > 0.3`) is removed.** The antihub ratio is a
  fifth calibrated statistic on the same kNN graph, with its null sd floored at one count
  (`1 / rows`); the recommendation is made when its z exceeds `z_threshold`, under
  `metric="euclidean"` only. Under cosine the z-score is reported but raises nothing: it gave a
  false recommendation in 44% of spiked-spectrum Gaussian cells. The fixed rule fired on 31% of
  isotropic Gaussian cells (now 1%).
- **`scale_plateau` and `plateau_tolerance` (added in WP 4.2, never released) are replaced by
  `scale_max_deviation`**, the largest relative deviation of the TwoNN scale curve from the
  full-sample value, with no threshold. The boolean flipped between random subsets of one dataset.
- **Rows with a common norm** (relative spread under 1e-3, e.g. normalised upstream) are compared
  with null draws normalised to that norm, under either metric. Before, `distance_cv` z reached
  -35 and the concentration recommendation fired on every such data set. Normalised rows with an
  anisotropic (power-law) spectrum remain uncalibrated (`distance_cv` and uniformity z of +8 to
  +11); see the `EmbedKitAnalyzer` docstring.
- `NeighborConsistency` scales its perturbations by the RMS per-coordinate standard deviation,
  `sqrt(mean_j var(X[:, j]))`, instead of `np.std(X)` over all entries, which also counted the
  spread of the coordinate means: adding a common vector to digits moved the calibrated
  neighbour-consistency z from 14.9 to 33.5. The statistic is now translation-invariant.

### Added (WP 4.5)
- `EmbedKitReport.calibration_k`; `antihub_ratio` in `report.calibration`;
  `embedkit.analysis.null.mean_shrinkage` and `embedkit.analysis.report.common_norm`.
- Benchmarks E3 (`benchmarks/e3.py`: known-ID manifolds, Gaussian null families, subsampling
  stability of six real embeddings) and E5 (`benchmarks/e5.py` and `run.py --experiment e5`:
  six kinds of controlled damage via `bench_datasets.damage`, detection and training-free
  repair), with `report_e3.py`, `report_e5.py`, Slurm scripts and
  `docs/BENCHMARK_DIAGNOSTICS.md`. The phase 4 exit criterion holds: on all six real data sets
  the report reads the same at 2,000 and 20,000 rows.

### Changed (WP 4.4)
- **Breaking: `EmbedKitAnalyzer(metric=...)` accepts only `"euclidean"` and `"cosine"`**, and
  every analyzer honours it (defect 28: distance concentration, uniformity, the kernel and
  intrinsic dimension used to stay Euclidean). With `"cosine"` the rows are L2-normalised once
  and the whole report, isotropy included, describes the normalised rows; rows with zero norm
  raise `ValueError`. The calibration null is fitted to the rows as given and its draws are
  normalised (a null fitted to the normalised rows came out too isotropic: z up to 9 on
  Gaussian data). Other metrics raise `ValueError`; `HubnessAnalyzer` and
  `NeighborConsistency` still accept any scikit-learn metric on their own.
- **Kernel spectrum from the centred kernel** (defect 29). `KernelDiagnosticsResult.effective_rank`
  is now `tr(Kc)^2 / tr(Kc^2)` of the centred kernel, over the whole spectrum (it was the
  entropy rank of the top 50 eigenvalues of the uncentred kernel, capped at 50; on the Pets
  ResNet features it is 70). `spectral_gap` and `eigenvalues` are of the centred kernel, so the
  constant direction no longer dominates the gap. `row_sum_skewness` keeps the uncentred kernel.
- `KernelDiagnosticsResult.condition_number` is renamed `top_k_eigen_ratio` (`lambda_1 / lambda_k`,
  not a condition number); `condition_number` remains as a deprecated alias.
- `knn(metric="cosine")` with the scikit-learn backend searches the L2-normalised rows with the
  Euclidean reduction (same neighbours and distances; calibration on 300 x 32 rows went from
  58 s to 2 s). Rows with zero norm keep the old path.

### Added (WP 4.4)
- `EmbedKitReport.metric`, in `to_dataframe()` and `print_summary()`.
- `embedkit.analysis.report.l2_normalize` and `METRICS`.

### Changed (WP 4.3)
- **Anisotropy and clustered directions are properties, not defects.** The participation-ratio
  recommendations ("highly anisotropic ... consider whitening") and the calibrated uniformity
  statement leave `report.recommendations` for the new `report.properties`: descriptive
  sentences about the spectrum, the dominant coordinates, the mean direction and (when
  calibrated) uniformity against the null, with no thresholds. Which of them predict that an
  intervention helps is E4's question (WP 7.1).
- With no recommendation, the report says "No recommendation: ..." instead of "Embedding geometry
  looks healthy".

### Added (WP 4.3)
- `IsotropyResult`: `isoscore` (Rudman et al. 2022, as published; it reduces to
  `(PR - 1) / (d - 1)` and has the plug-in bias), `top_k_variance_share` and
  `top_k_coordinate_share` for k in 1, 3, 10 (principal components; coordinates, the "rogue
  dimensions" of Timkey and van Schijndel 2021), `spectrum_alpha` (power-law exponent of the top
  half of the spectrum; recovers 1 and 2 within 0.06), `mean_vector_ratio` and `mean_cosine`
  (uncentred). All in `to_dataframe()` and `print_summary()`.

### Changed (WP 4.2)
- **Breaking: `IntrinsicDimensionResult.consensus` and the `aggregate` argument are removed.** The
  estimators measure different things and are no longer combined (review 3.3). `uncertainty` is
  renamed `between_method_spread`, which says what it is.
- **`suggested_target_dim` is `1.5 x` the TwoNN estimate** instead of the median across
  estimators (the user's choice; no benchmark supports either). If `id_methods` omits TwoNN or it
  fails, the first method of `id_methods` that succeeded is used (`reference_method`). The ID / D
  recommendation uses the same number. The auto d moves slightly: Pets 22 to 24, digits 12 to 13.
- `print_summary()` shows each estimator with its interval, and the scale curve.
- `EmbedKit.fit` skips the ID bootstrap and scale curve (it reads only `suggested_target_dim`).

### Added (WP 4.2)
- Bootstrap standard errors and intervals per estimator (`IntrinsicDimensionEstimator(n_bootstrap=20)`,
  `standard_errors`, `intervals`). They cover sampling error only: on spheres and cubes they match
  the spread over independent draws (ratios 0.95-1.19), but the k-NN estimators read low at high
  ID (a 20-sphere gives 15-17 at n = 5,000), which no interval covers. `ESS` and `DANCo` (50-110 s
  per fit) are not resampled.
- `scale_curve` and `scale_plateau`: TwoNN on nested subsamples of n, n/2, n/4 and n/8 rows, and
  whether all levels are within `plateau_tolerance` (10%) of the full-sample value.
- `ESS` (expected simplex skewness) among the supported methods; about 100 s per fit at n = 5,000.
- `EmbedKitAnalyzer(id_bootstrap=20, id_scale_curve=True)`.

### Changed (WP 4.1)
- **Breaking: `EmbedKitReport.severity` is removed.** It was a sum of fixed-threshold crossings
  with no reference distribution (review 3.2). `report.n_outside_null` counts the calibrated
  statistics with `|z| > z_threshold` (default 3), or is `None` without calibration.
- **The recommendations about k-skewness, `distance_cv`, neighbour consistency and uniformity
  come from z-scores against a null model**, not from the constants (k-skewness > 2 or 5,
  `distance_cv < 0.02`, consistency < 0.5, uniformity > -1). Each is made only in its direction
  (more hubness, more concentrated distances, a less stable graph, more clustered directions than
  the null). With `calibrate=False` none is made. The antihub, ID/D and participation-ratio rules
  are unchanged until WP 4.2 and 4.3.
- `EmbedKit.fit` runs the report with `calibrate=False`: it reads only `suggested_target_dim`.
- `DistanceConcentration` and `UniformityScore` compute pairwise distances from one BLAS product
  in float64 instead of `scipy` `cdist` / `pdist`, 15-20 times faster at 2,000 x 2,048.
  Distances agree with `cdist` to 1e-13 and identical rows still give exact zeros; uniformity,
  now computed in float64 instead of float32, agrees to about 1e-8.

### Added (WP 4.1)
- `embedkit.analysis.null`: `spectrum_matched_gaussian` (a Gaussian with the data's mean and
  covariance `alpha C + beta I`, where the trace and `tr(Sigma^2)` match unbiased estimates;
  the sample covariance itself makes iid data look more isotropic than their own null, with
  distance-CV z-scores down to -19), `column_permuted`, `null_distribution`, `calibrate` and
  `CalibratedStatistic` (observed, null mean and sd, z-score).
- `EmbedKitAnalyzer(calibrate=True, null_model="gaussian", n_null_draws=20, n_calibration=2000,
  z_threshold=3.0)` and `report.calibration`, also in `to_dataframe()` and `print_summary()`.
  It costs `n_null_draws + 1` evaluations of the four statistics on `n_calibration` rows (about
  50 s at 3,680 x 2,048).

### Changed
- **Supervised `EmbedKit` trains the residual adapter** (`architecture="residual"`: the
  PCA-initialised linear map plus a zero-initialised MLP). In E8 it ties the plain MLP on full
  training sets and, unlike it, does not collapse with 1-5 labels per class
  (`docs/BENCHMARK_SUPERVISED.md`).
- **Regression `EmbedKit` trains the plain MLP on Gaussian-noise views**
  (`architecture="mlp"`, `GaussianNoise(0.05)`, relative to the per-feature scale). In E8 it beats
  the linear RnC refiner, the raw embeddings and PCA on all 4 regression sets. `fit` warns with
  1000 rows or fewer, where no refiner reliably beat the raw embeddings.
- `architecture` and `augmentation` default to `"auto"`: residual / MLP / linear and `KNNPairs(5)`
  / `GaussianNoise(0.05)` / `KNNPairs(5)` for supervised / regression / self-supervised. The
  resolved architecture is in `architecture_` after `fit`.
- **Supervised `EmbedKit` drops the uniformity term**: the default loss is `SupConLoss(0.07)` alone
  (was SupCon + 0.5 `AlignUniformLoss`), which spread each class against SupCon's pull (review 3.4).
- **Supervised `EmbedKit` stops on a held-out score**: `monitor="auto"` and
  `early_stopping_patience="auto"` (new defaults) resolve to `val_knn_accuracy` with patience 10
  and 10% of the rows held out, restoring the best weights. Self-supervised mode keeps no early
  stopping. Pass `early_stopping_patience=None` to train on every row for the full budget.

### Added
- `EmbedKit(mode="regression")`: `RankNContrastLoss`, targets cast to float, and the new monitor
  `val_ridge_r2` (R^2 of a ridge probe from the training subsample to the validation rows).
- `SupConLoss(spread_weight=w)`: `(1 - w) * SupCon + w * spread`, the class-conditional InfoNCE term
  of Chen et al. (2022) that keeps within-class structure (WP 6.2).
- Trainer monitors `train_nc1` and `val_nc1` (neural-collapse NC1 on the training subsample and the
  validation rows); supervised `EmbedKit` logs both in `history`. `val_knn_accuracy` now rejects
  continuous labels with a clear message.
- `knn_regression_r2` and the `knn_r2` protocol metric (10-NN regression); the regression ridge probe
  returns per-point terms `1 - e_i^2 / var(y_test)` (mean = R^2), so `compare` bootstraps it.
- Benchmark E8 (`run.py --experiment e8`, `report_e8.py`, `cluster/e8.slurm.sh`,
  `docs/BENCHMARK_SUPERVISED.md`): supervised arms
  against the new default and the previous one, LDA, label noise, superclass fitting, label
  efficiency (`<name>__lpc<k>`), and regression arms (the 0.1.2 RnC in `legacy_losses.py`) on
  `synthetic_regression[_ties]`, `yelp_mpnet` and `fluorescence_esm2`; residual and MLP arms in
  both modes, regression on 200 and 1000 training rows (`<name>__n<k>`), and the chosen defaults
  against the baselines in the report. `e8.slurm.sh` takes `TASKS=<array in env.sh>` and
  `<dataset>@<method>,...` entries.

### Fixed
- `EmbedKit.load(directory, device=None)`: the device for `transform` can be given. A bundle fitted
  on CPU and reloaded on Apple MPS (the default device where available) gave outputs about 1e-4
  apart, which failed the save/load round-trip test on the macOS CI runner with Python 3.10 since
  WP 5.1.
- Early stopping with ties: of evaluations with the same best monitor value, the latest one's
  weights are restored (was the first). On a small validation split kNN accuracy often plateaus, and
  restoring the first plateau epoch kept the refiner near its initialisation. Ties still do not
  reset the patience.
- `EmbedKit(mode="supervised" | "regression", early_stopping_patience=None)` no longer fails: with
  nothing held out, `monitor="auto"` falls back to `k_skewness`.

## 0.4.0 (unreleased)

Phases 3 and 5 of `docs/IMPLEMENTATION_PLAN.md`: benchmarks E1 and E2 and the decision gate
(`docs/BENCHMARK_E1_E2.md`), then the refiner ablation E6 and the learning curves E7
(`docs/BENCHMARK_E6_E7.md`).

### Changed
- **`EmbedKit(mode="self_supervised")` now returns PCA, not a trained refiner** (gate outcome C of the
  plan). New parameter `method="auto" | "pca" | "refiner"`: `"auto"` is `"pca"` without labels and
  `"refiner"` in supervised mode. `"pca"` is the top `target_dim` principal directions followed by L2
  normalisation, built as the `linear` refiner at its initialisation (no training), so `transform`,
  `dim=`, `save` and `load` are unchanged. In E1, E6 and E7 the self-supervised refiner never beat
  the better of PCA and PCA-whitening, lost to them when fitted on fewer than a few thousand rows,
  and cost about 100 times more. Pass `method="refiner"` for the trained refiner; setting
  `augmentation`, `loss`, `nested_dims` or a non-linear `architecture` with `"pca"` raises.
  `config.json` records the method; bundles without it load as `"refiner"`.
- **`EmbedKit` no longer stops early by default** (`early_stopping_patience=None`, was 10): it trains
  for the fixed `epochs` budget. In E2 no label-free monitor tracked test accuracy within a run, and
  the validation contrastive loss moved against it. `val_split` now defaults to `None`: 0.1 of the
  rows are held out only when early stopping is enabled, so by default the refiner trains on every row.
  Pass `early_stopping_patience=10` to restore the 0.2.0 behaviour.
- **`EmbedKit` uses `architecture="linear"` by default** (was the MLP): a linear map initialised at PCA
  and trained with the contrastive loss. In E6 (`benchmarks/results/e6/REPORT.md`) it beat the MLP on
  5 of 6 datasets with an interval and lost on none; the residual variant did as well against the MLP
  but lost to `"linear"` on 3 datasets and won on 1. Pass `architecture="mlp"` for the 0.2.0 model.
  `EmbeddingRefiner` has the same default. A linear or residual refiner trained without
  `initialize_from_data` (a custom loop; `Trainer.fit` calls it) warns once, as it then starts from a
  random map instead of PCA.

### Added
- `EmbeddingRefiner(architecture="residual")` (WP 5.1): `z = normalize(W x̃ + g(x̃))`, with `x̃` the
  input standardised by fixed statistics, `W` initialised from the top principal directions and the
  MLP `g` with a zero-initialised output layer, so training starts exactly at PCA + L2 normalisation.
  `architecture="linear"` keeps only `W`. `Trainer.fit` calls `model.initialize_from_data` on the
  training rows when the model defines it. `EmbedKit(architecture=...)` passes it through and stores it
  in `config.json`; `EmbedKit.load` restores it (bundles without the entry load as `"mlp"`).
- WP 5.2 options:
  - `KNNPairs(mode="mutual")` draws positives only among mutual neighbours, with the nearest neighbour
    as a fallback.
  - `NTXentLoss(neighbor_mask=k)` stops input-space neighbours from acting as negatives for each other.
  - `NTXentLoss(decoupled=True)` is the decoupled contrastive loss.
  - `IdentityPair` gives dropout-only positives.
- `MatryoshkaLoss(base_loss, dims)` (WP 5.3): the base loss summed over nested prefixes of the output.
  `EmbedKit.transform(X, dim=m)` and `fit_transform(..., dim=m)` keep the first `m` coordinates and
  renormalise. `EmbedKit(nested_dims="auto" | dims)` wraps the loss in it (`"auto"`: powers of two
  from 32 below the output dimension) and records the dims in `config.json`. In E6 the prefixes of one
  model at d = 256 matched `linear` models trained at 32, 64 and 128 (one loss in 16 comparisons).
- `embedkit.hubness_reduction` (WP 5.5): `LocalScaling`, `MutualProximity` (Gaussian), `CSLS` and
  `DisSimLocal` as secondary distances for exact kNN search, with `fit(X_reference)` and
  `kneighbors(X=None, n_neighbors=5)`. `knn_accuracy`, `recall_at_k`, `k_skewness` and
  `evaluate_transform` take a `hubness_reduction` argument.
- Losses have optional `precompute(X)` and `uses_indices` hooks, mirroring the augmentations.
  `Trainer` and `CombinedLoss` call `precompute` with the training rows, and pass the batch's global
  row indices to losses that declare `uses_indices`.
- `NTXentLoss` computes its full and chunked forms in one code path (same values).

### Benchmarks
- `benchmarks/run.py --experiment e6`: the phase 5 ablation grid (architecture arms at auto and fixed
  d, with raw, PCA and PCA-whitening as references); `benchmarks/report_e6.py` applies the phase 5
  acceptance rule; `--shard I/N` splits a dataset's grid between jobs. Stage WP 5.1 results are in
  `benchmarks/results/e6/`. Stage WP 5.2/5.3 adds the arms `mutual`, `nbrmask`, `decoupled`,
  `dropout` (on `residual`, which has dropout) and `matryoshka` (one model at min(256, d) read at each
  prefix), at fixed d. Results in `docs/BENCHMARK_E6_E7.md`: no WP 5.2 arm is accepted, WP 5.3 is
  non-inferior. `report_e6.py` adds the per-d prefix table and defaults to `--reference e6_linear`.
- E1 gains the hubness-reduction methods `raw+ls`, `raw+mp`, `raw+csls` and `raw+dsl`, scored on the
  kNN-based metrics only (`Method.hubness_reduction`, `Method.metrics`);
  `benchmarks/cluster/e1_hub.slurm.sh` runs them on CPU nodes. They leave kNN accuracy unchanged on the
  E1 sets (-0.028 to +0.005 against raw).
- `benchmarks/run.py --experiment e7` and `benchmarks/report_e7.py`: learning curves of the default
  refiner, PCA and PCA-whitening over the number of rows fitted (200 to 100k, capped at the training
  size), and inductive against transductive fitting (`Method.transductive`).
- `benchmarks/e2.py` and `benchmarks/report_e2.py`: experiment E2, every trainer monitor logged
  next to test kNN and probe accuracy over 30 refiner configurations per dataset.
- `benchmarks/report_gate.py`: the phase 3 gate on E1 results.
- Results in `benchmarks/results/e1/` and `benchmarks/results/e2/`.

## 0.2.0 (unreleased)

Phase 2 of `docs/IMPLEMENTATION_PLAN.md`: any transform, trained or not, can be scored with one call on
held-out data, against training-free baselines, with a benchmark harness that runs unchanged on a laptop
and on the cluster.

### Evaluation metrics (WP 2.1)
- New module `embedkit.evaluation.metrics`: pure functions on arrays, each returning a float.
  `knn_accuracy` and `linear_probe` (neighbours and fits use the training split only), `recall_at_k`,
  `trustworthiness`, `continuity`, `knn_overlap`, `linear_cka`, `effective_rank` (RankMe, full
  singular-value spectrum), `k_skewness`, `nc1` (neural-collapse variability) and `silhouette`.
  Subsampled metrics use a fixed seed.
- `linear_probe` standardises on the training split and chooses its regularisation by cross-validation
  on the training split (logistic regression for classes, ridge for continuous targets); `C=` fixes it
  instead (`C=1.0, max_iter=3000` reproduces the review pilot's probe).
- `knn_accuracy`, `linear_probe` and `recall_at_k` accept `return_per_point=True` and also return the
  per-test-point outcomes.
- `embedkit.analysis.hubness` exposes `k_occurrence` and `k_occurrence_skewness`, shared by
  `HubnessAnalyzer` and `k_skewness`. `HubnessAnalyzer` results are unchanged.

### Training-free baselines (WP 2.2)
- New module `embedkit.baselines`: `Identity`, `CenterNormalize`, `Standardize`, `PCAReduce`,
  `ZCAWhiten`, `AllButTheTop` (Mu & Viswanath 2018; with `n_components`, the reduce-then-clean
  algorithm of Raunak et al. 2019), `ConceptorNegation` (Liu et al. 2019, uncentred correlation matrix,
  `alpha=2`) and `RandomProjection`. They share `fit` / `transform` / `fit_transform` / `get_params`,
  accept NumPy arrays or tensors, store only statistics of the data passed to `fit`, and compute in
  float64 with float32 output.
- `PCAReduce` caps `n_components` at the rank of the centred training data, with a warning, so
  whitening never divides by a zero eigenvalue. Component signs are deterministic.

### Evaluation protocol (WP 2.3)
- `embedkit.evaluation.evaluate_transform(make_transform, X_train, X_test, y_train, y_test, seeds=...)`
  fits a fresh transform per seed on the training split (or on both splits with `transductive=True`,
  recorded), scores the test split, and returns a tidy DataFrame with one row per seed and metric,
  fit and transform times, the output dimension and the transform's parameters. `fit_with_labels=True`
  for supervised methods. `test_pairs=` adds `sts_spearman` for sentence-similarity sets. Metrics come
  from the registry `embedkit.evaluation.METRICS`; label metrics are skipped without labels.
- `embedkit.evaluation.compare(results, reference="raw")`: mean ± sd over seeds per method and metric,
  and the difference to the reference with a paired bootstrap interval over test points where
  per-point outcomes exist.

### Trainer monitors (WP 2.4)
- New monitors in `trainer.MONITORS`: `knn_overlap` (max), `effective_rank` (max), `val_loss` (min; the
  training loss on the validation split with fixed augmentation draws) and `val_knn_accuracy` (max;
  5-NN accuracy of the validation rows against the training rows). With labels and `val_split > 0`,
  the validation labels are kept for it. Monitors are called as `fn(Z, **context)` with the input rows,
  labels and reference pool; custom monitors should accept `**_`. The default monitor is unchanged.
- `Trainer.evaluate(X, y=None, X_ref=None, y_ref=None)` accepts the labels the new monitors need.

### Benchmark harness (WP 2.5)
- `benchmarks/`: dataset loaders that never download (`bench_datasets.py`), method grids
  (`experiments.py`), a resumable runner that writes one JSON file per dataset and method with a
  validation stage for selection and a test stage (`run.py`), a Markdown report that selects each
  family's configuration on validation (`report.py`), a cluster-only feature extractor
  (`extract_features.py`: CIFAR-100/DINOv2, STL-10/CLIP, Banking77 and STS-B with all-mpnet-base-v2,
  ANN-Benchmarks GloVe/SIFT/GIST), and Slurm jobs for OCCIDATA (`benchmarks/cluster/`).

## 0.1.3 (2026-11-10)

This release fixes defects D1 to D27 of `docs/CRITICAL_REVIEW.md`: every defect rated A or B, and the
C-rated ones scheduled for phase 1 of `docs/IMPLEMENTATION_PLAN.md` (D28 to D30 come later). Defect numbers refer to section 2 of that review. Several fixes change results or
the public API; the **Breaking** items list what callers must change. Each fix has a regression test
that fails on 0.1.2.

### Nearest neighbours (D1, D2, D12, D15, D16, D17)
- **The kNN cache returned another array's neighbours (D1).** The cache key was the array's memory
  address, so a freed array's address could be reused by a new array of the same shape, and in-place
  edits were not seen. The cache is removed. `NeighborConsistency` reused earlier rounds' neighbours
  because of it, which understated `std_consistency` (D2). Any analysis that called `knn` on several
  arrays of the same shape in one process could be affected.
- **Breaking: FAISS is optional (D12).** `faiss-cpu` moved from the dependencies to the `faiss` extra
  (`pip install embedkit[faiss]`). `backend="auto"` now means scikit-learn; use `backend="faiss"` or
  the environment variable `EMBEDKIT_KNN_BACKEND=faiss` to opt in. In some environments importing
  FAISS next to PyTorch aborted the process (`OMP: Error #15`) or crashed inside the search. The
  "falling back to sklearn" warning is gone.
- **`knn` has one contract for every backend (D15, D16, D17).** Distances are true Euclidean
  distances (FAISS returned squared ones) or cosine distances `1 − cos` (FAISS returned inner products,
  sorted descending), ascending in each row. The caller's array is no longer L2-normalised in place by
  the FAISS cosine path. The query point is removed by index, so with duplicate rows a point is no
  longer listed as its own neighbour. `GaussianNoise(adaptive=True)` therefore scales noise the same
  way on both backends.
- **Breaking:** `clear_cache()` does nothing and emits a `DeprecationWarning`; it will be removed in
  the next release. `knn` raises `ValueError` for `k` outside `1 … n − 1` and for an unknown backend.
- `HubnessAnalyzer.fit` and `NeighborConsistency.fit` accept `neighbor_index=` (a precomputed graph).
  `EmbedKitAnalyzer` computes the kNN graph once and passes it to both.

### Losses (D3, D4, D20, D21, D27)
- **`TripletLoss` gradients were NaN (D3)** for every mining mode with the default Euclidean distance:
  the square root of the zero diagonal has an infinite derivative. The loss value was finite, so
  nothing warned. Distances now use `sqrt(d² + 1e-12)` with the diagonal masked, and gradients are
  finite even for duplicate embeddings. The per-anchor Python loop is replaced by tensor operations
  with the same mining semantics. **Breaking:** an unknown `mining` value raises `ValueError` instead of
  behaving as `"easy"`.
- **Chunked `NTXentLoss` crashed (D4)** whenever `chunk_size` did not divide 2N, and when it ran it
  averaged per-chunk means, overweighting a short last chunk (D20). It now equals the unchunked loss for
  every `chunk_size`.
- **`RankNContrastLoss` mishandled ties in label distance (D21).** Each candidate's denominator started
  at its own position in an arbitrary sort order instead of at the start of its tie group, so the value
  and gradient differed from Rank-N-Contrast (Zha et al., 2023) and depended on the order of the rows
  in the batch. Ties always occur (both views of a sample share a label) and are pervasive with
  discrete, binned or one-hot labels, where the gradient was up to about 35° off. Label distances are
  now computed exactly (`torch.cdist` is inexact above 25 rows) and candidates at equal label distance
  share one denominator. **Embeddings trained with `RankNContrastLoss` on discrete or repeated labels
  optimised a different, batch-order-dependent objective and may need retraining.**
- **`CombinedLoss` no longer swallows errors (D27).** A `TypeError` or `ValueError` inside a member
  used to trigger a silent retry without labels, turning a shape error in a supervised loss into the
  unsupervised form. Losses now declare `uses_labels` (`SupConLoss`, `TripletLoss`,
  `RankNContrastLoss`: `True`; `NTXentLoss`, `AlignUniformLoss`: `False`) and `CombinedLoss` passes
  labels only to those that use them. **Breaking:** custom `BaseLoss` subclasses that need labels
  inside a `CombinedLoss` must set `uses_labels = True`.
- **Breaking:** `SupConLoss` accepts labels of shape `(N,)` or `(N, 1)` and raises `ValueError`
  otherwise. Vector-valued labels used to produce a silently three-dimensional positive mask.
- `AlignUniformLoss` computes squared distances between unit vectors as `2 − 2 a·b` instead of
  `torch.cdist`, whose backward is not implemented on Apple MPS. The value is unchanged.

### Trainer (D5, D8, D11, D18, D19, D22, D23)
- **`scheduler="step"` decayed the learning rate per batch (D5)**, so it reached about 1e-51 within a
  few epochs and training effectively stopped. It is now stepped once per epoch (×0.1 every
  `epochs // 3` epochs). `"cosine"` is still stepped per batch; `"plateau"` at each evaluation.
- **Early stopping always minimised the monitor (D18)**, so with `monitor="isotropy"` it stopped on
  improvement. A registry, `embedkit.improvement.trainer.MONITORS`, gives each monitor its direction
  (`uniformity` min, `isotropy` max, `k_skewness` min). Early stopping, best-weight tracking and
  `ReduceLROnPlateau` read it. **Breaking:** an unknown `monitor` or `eval_metrics` name raises
  `ValueError` when the `Trainer` is built.
- **The best weights were not restored (D19).** After patience ran out, the model was the one from
  `patience × eval_every` epochs after the best point. New `restore_best=True` (default) reloads the
  best evaluation's weights at the end of `fit` when early stopping is enabled. New attributes
  `best_score_` and `best_epoch_`; `history` gains `"lr"` (per epoch) and `"eval_epoch"`.
- **Evaluation errors were swallowed (D23)**, which silently disabled early stopping. They now
  propagate. Monitors use the new `eval_k` (default 10) and `eval_metric` (default `"cosine"`, since the
  refiner's output is L2-normalised) arguments instead of `HubnessAnalyzer`'s Euclidean defaults.
- **Labels (D8):** `Trainer.fit` accepts Python lists and string labels. New helper
  `embedkit.utils.validation.encode_labels`: strings and booleans become class codes, integers stay
  class labels with their values, 1-D floats and 2-D arrays are continuous. A label count that does
  not match `X` raises `ValueError`.
- **Device (D11):** with `device=None` the trainer selects CUDA, then Apple MPS, then CPU. If the loss
  does not run on MPS it falls back to CPU with a warning. All built-in losses run on MPS.
- **Breaking: warm-up and LARS (D22).** `warmup_epochs` was stored and never used. It is now a linear
  warm-up, and its default changes from 10 to 0 so existing runs keep their learning-rate curve.
  `optimizer="lars"` (SGD with a 10× learning rate, not LARS) is removed and raises `ValueError`.
- `early_stopping_patience` counts evaluations, not epochs (now documented). The evaluation subsample
  is also drawn when `random_state=None`.

### Augmentations (D9, D24, D25, D26)
- **Composition order changed results silently (D9).** In a sequential `CompositeAugmentation`,
  `KNNPairs` discarded the incoming second view, so `[GaussianNoise, KNNPairs]` produced a noisy anchor
  and a raw neighbour. Augmentations now declare `kind`: `"pair"` (`KNNPairs`, `EmbeddingMixup`) or
  `"view"` (`GaussianNoise`, `FeatureDropout`, `FeatureMasking`). **Breaking:** a sequential composite
  raises `ValueError` if it has more than one pair augmentation or a pair augmentation that is not
  first. View augmentations are applied once to each view with independent draws, so
  `[KNNPairs, GaussianNoise]` gives a noised anchor and a noised neighbour. `random_choice` mode uses
  torch's RNG instead of Python's `random`.
- **`KNNPairs` (D24):** the in-batch path always took the nearest neighbour; it now samples one of the
  k neighbours at random, as the precomputed path does, with `k` capped at batch size − 1. Both paths
  use torch's RNG instead of the global `np.random`. New `metric` argument (`"euclidean"` or
  `"cosine"`). **Breaking:** the unused `hard_negatives` argument is removed, so `neighbor_index` is
  now the second positional parameter.
- **`FeatureMasking` (D25)** drew one mask for the whole batch. It now draws an independent mask per
  row and per view, masking exactly `max(1, round(d · mask_ratio))` distinct features per row with
  `block_size=1`. Blocks are documented as meaningful only for ordered features.
- **Breaking: `GaussianNoise` is relative by default (D26).** `std=0.05` was an absolute scale: for
  unit-norm 768-d embeddings the noise vector was 1.4 times as long as the signal. With the new
  default `relative=True`, `std` multiplies the per-dimension standard deviation of the training data
  (computed by the new `precompute`, which the trainer calls; the current batch is used otherwise).
  Pass `relative=False` for the old behaviour. Adaptive noise ignores `relative`.
- `EmbeddingMixup` and adaptive `GaussianNoise` cap their in-batch `k` at batch size − 1 instead of
  failing on small batches.

### Analysis (D6, D7, D8, D10, D14)
- **The participation ratio was capped above 2,000 dimensions (D6).** Only the top 512 eigenvalues
  were computed, so an isotropic Gaussian with d = 2,500 was reported as moderately anisotropic while
  its `isotropy_score` was 0.98. `IsotropyAnalyzer` now computes every statistic exactly from the
  covariance or Gram matrix of at most `n_max=5000` rows (new `random_state`; `d_max` is ignored).
  New fields: `participation_ratio_corrected` (the ratio with an unbiased estimate of `tr(Σ²)`, exact
  for Gaussian data) and `participation_ratio_normalized` (corrected / d, in [0, 1]). The report
  thresholds the normalised value (0.1 and 0.3, as before relative to d), so the recommendation no
  longer depends on n/d for isotropic data. `isotropy_score` divides the effective rank by
  `min(n − 1, d)` instead of 512. **Breaking:** `eigenvalue_spectrum` is renamed
  `eigenvalue_spectrum_top` (plotting only); the old name is a deprecated alias for one release.
  `explained_variance_ratio` covers the same top eigenvalues, relative to the total variance.
- **Kernel alignment was not centred (D7)**, so it mostly measured class balance: 0.70 for 2 and 0.32
  for 10 classes independent of X. It is now centred kernel alignment (`HKH`), about 0 for independent
  labels. Class labels use the indicator kernel; continuous labels (1-D floats or 2-D arrays) use an
  RBF kernel with a median-heuristic bandwidth instead of float equality. **Breaking:** the field is
  renamed `centered_kernel_alignment`; `kernel_alignment` is a deprecated alias for one release that
  returns the centred value.
- **Labels (D8):** `KernelDiagnostics.fit` accepts string labels and Python lists of any length.
- **Distance concentration (D10):** new `distance_cv` (standard deviation over mean of pairwise
  distances) and `relative_contrast_per_query`. The report uses `distance_cv < 0.02` instead of
  `concentration_ratio > 0.8`, which one duplicated pair set to 0. The threshold is provisional until
  the null calibration of phase 4; for iid Gaussian data it fires from about d = 1,250 (the old rule
  fired from about d = 1,000). Old fields are kept.
- **Intrinsic dimension was not reproducible (D14).** The methods ran in a thread pool, which made
  TwoNN vary between runs with the same `random_state` (8.89, then NaN, on scikit-learn digits). They
  now run sequentially. Exact duplicate rows are removed first (new field `n_duplicates_removed`):
  they made TwoNN divide by zero and MLE return 0. A non-finite or non-positive estimate is now a
  failure (warning, NaN, excluded from the consensus), and `random_state` reaches every estimator that
  accepts it. `EmbedKit(target_dim="auto")` can choose a different dimension than in 0.1.2.

### Reproducibility (D13, D24; WP 0.2)
- **`random_state` did not make a run reproducible (D13).** `EmbedKit` built the refiner, and drew its
  initial weights, before `Trainer.fit` seeded torch; four identical `EmbedKit(random_state=0)` fits
  stopped at four different epochs. `EmbedKit` now draws the initial weights from `random_state`, and
  two fits with the same `random_state` return identical arrays on CPU (GPU and MPS kernels are not
  guaranteed to be deterministic).
- **`Trainer.fit` no longer reseeds the global NumPy and torch state.** It used to call
  `np.random.seed` and `torch.manual_seed`, a side effect on the caller's process. The data order, the
  augmentation draws and dropout now come from generators derived from `random_state`, and the global
  CPU, CUDA and MPS random states are restored when `fit` returns.
- New `BaseAugmentation.set_rng(np_rng, torch_gen)`, called by `Trainer.fit`; every built-in
  augmentation draws only from these generators. `EmbeddingMixup` draws its mixing weights with NumPy's
  `beta` from that generator.
- The MPS capability probe no longer consumes random numbers.

### Tooling (WP 0.3)
- GitHub Actions workflow (`.github/workflows/ci.yml`): `ruff check embedkit tests` and `pytest` on
  Ubuntu and macOS with Python 3.10 and 3.12, plus a job that runs the kNN tests with
  `EMBEDKIT_KNN_BACKEND=faiss`.
- Ruff ignores the naming rules for math-style names (`X`, `Z`, `F`); the package and tests pass
  `ruff check`.

### Other
- The version is 0.1.3 in both `pyproject.toml` and `embedkit.__version__`.
- `EmbeddingVisualizer.plot_training_trajectory` plots monitors against `history["eval_epoch"]` and
  skips the per-epoch `"lr"` series.

## 0.1.2 (2026-11-08)

### Fixed
- **`SupConLoss` returned NaN for every input** (both the full and the chunked paths, with and without labels).
  The −inf diagonal of `log_prob` was multiplied by the zero diagonal of the positive mask, and
  `0 * (−inf) = NaN`. Loss values from versions ≤ 0.1.1 were NaN, so `Trainer.history["loss"]` was NaN for any
  run that used `SupConLoss`, alone or inside `CombinedLoss` (including `EmbedKit(mode="supervised")` with
  `loss="auto"`). **The gradient was correct, so embeddings trained with earlier versions are unaffected.**
- **Chunked `SupConLoss` and `RankNContrastLoss` did not match the full computation** when `chunk_size` did not
  divide `2N`: they averaged the per-chunk means, which overweighted the shorter last chunk. Both now average
  over all rows, matching the unchunked loss.
- **`Trainer` trained nothing and logged NaN when `batch_size` exceeded the number of training samples**
  (default `batch_size=256`, so any dataset under ~256 rows). With `drop_last=True` the loader yielded zero
  batches per epoch. The batch size is now capped at the number of training samples, with a `UserWarning`;
  fewer than 2 training samples raises `ValueError`.
- **Running the analysis silenced all Python warnings in the user's process.** Importing `skdim` calls
  `warnings.filterwarnings("ignore")`; EmbedKit now imports it inside `warnings.catch_warnings()` so the global
  filters are restored.
- `embedkit.__version__` now matches the package version (it reported `0.1.0`).
