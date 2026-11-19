# Migrating from EmbedKit 0.1.2 to 0.6.0

This note is for code written against EmbedKit 0.1.2 (analysis reports, `EmbedKit(...)`, `Trainer`,
losses, augmentations, saved bundles). It lists every breaking change, the version that made it and
what to do. `CHANGELOG.md` has the reasons and the measurements; `docs/manual.md` describes the
current API.

Version path: 0.1.2 -> 0.1.3 -> 0.2.0 -> 0.4.0 -> 0.5.0 -> 0.6.0. There is no 0.3.0. The "Since" column
below says which release introduced each change. Releases 0.2.0 and 0.4.0 are mostly additions
(`embedkit.evaluation`, `embedkit.baselines`, `embedkit.hubness_reduction`, `MatryoshkaLoss`, the
`residual` and `linear` refiners); only their breaking parts are listed.

Quick triage:

- You only call `EmbedKit(...).fit_transform(X)`: the output changes (self-supervised mode now
  returns PCA). See section 4.
- You read `report.severity`, `report.intrinsic_dim.consensus` or `kernel.condition_number`: sections
  1 to 3.
- You build `Trainer`, losses and augmentations yourself: section 5.
- You rely on FAISS being installed with the package: section 6.

## 1. Analysis report (`EmbedKitAnalyzer`, `EmbedKitReport`)

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.5.0 | `report.severity` (`"low"`, `"medium"`, `"high"`) | Removed. `report.n_outside_null` is the number of calibrated statistics with `abs(z) > z_threshold` (default 3), or `None` when `calibrate=False`. Per statistic: `report.calibration[name].z_score` | Replace severity tests with `n_outside_null` or with the z-scores. There is no mapping to the three levels |
| 0.5.0 | Fixed-threshold recommendations: k-skewness above 2 or 5, `antihub_ratio > 0.3`, `distance_cv < 0.02` (0.1.3; `concentration_ratio > 0.8` in 0.1.2), neighbour consistency below 0.5, uniformity above -1 | Made from z-scores against a null model, one direction per statistic (more hubness, more concentrated distances, a less stable graph, more clustered directions than the null). The antihub one is made under `metric="euclidean"` only. With `calibrate=False` none of them is made | Keep `calibrate=True` (the default) if you use these recommendations. Code that matches the old wording must be rewritten |
| 0.5.0 | Participation-ratio recommendations ("highly anisotropic ... consider whitening") | Gone from `recommendations`. `report.properties` holds descriptive sentences about the spectrum, dominant coordinates, mean direction and (when calibrated) uniformity, with no thresholds | Read `report.properties` for them. Anisotropy is not treated as a defect |
| 0.5.0 | "No major issues detected. Embedding geometry looks healthy." | "No recommendation: no statistic is beyond its null range or a fixed rule." The ID/D rule (0.1 and 0.3) is the only fixed rule left | Update any string match |
| 0.6.0 | 0.5.0 recommendations for more concentrated distances, a less stable kNN graph, hubness under cosine, and ID/D below 0.1 or 0.3 | Removed: in experiment E4 none predicted the gain of its fix on 35 natural classification sets. Only two remain, under `metric="euclidean"`: more hubness and more antihubs than the null, both naming a hubness-reducing search. The statistics and z-scores are unchanged in `report.calibration` and `report.intrinsic_dim`. The no-action message is now "No recommendation: no statistic with a validated rule is beyond its null range." | Read the z-scores yourself if you relied on those rules; update string matches |
| 0.6.0 | `report.suggested_k`, `report.suggested_sigma` | `report.k` (the neighbourhood size the report used) and `report.kernel.sigma`. The old names still work with a `DeprecationWarning` and go in 0.7.0; `to_dataframe()` has `Summary/k` and no `Summary/suggested_k` or `Summary/suggested_sigma` rows | Rename |
| 0.5.0 | `EmbedKitAnalyzer(metric=...)` took any string, and only some analyzers used it | Only `"euclidean"` and `"cosine"`; anything else raises `ValueError`. Every analyzer honours it. With `"cosine"` the rows are L2-normalised once and the whole report, isotropy included, describes the normalised rows; a zero-norm row raises `ValueError` | Pass `"euclidean"` or `"cosine"`. `HubnessAnalyzer` and `NeighborConsistency` on their own still accept any scikit-learn metric |
| 0.5.0 | `suggested_target_dim` from the median across ID estimators | `1.5 x` the TwoNN estimate (same clipping to `[max(4, ID), d]`). If TwoNN is not in `id_methods` or fails, the first method that succeeded (`reference_method`) | Pin `target_dim` explicitly if you need the old value |
| 0.5.0 | `fit` ran the analyzers only | By default also runs the null calibration (`n_null_draws + 1` evaluations of five statistics on `n_calibration` rows) and the ID bootstrap and scale curve. New arguments: `calibrate`, `null_model`, `n_null_draws`, `n_calibration`, `z_threshold`, `id_bootstrap`, `id_scale_curve` | `calibrate=False, id_bootstrap=0, id_scale_curve=False` skips the extra work (this is what `EmbedKit.fit` uses) |
| 0.5.0 | `report.to_dataframe()` rows `ID/consensus`, `ID/uncertainty`, `Kernel/condition_number`, `Summary/severity` | `ID/<method>`, `ID/<method>_se`, `ID/between_method_spread`, `ID/reference_method`, `ID/scale_max_deviation`, `Kernel/top_k_eigen_ratio`, `Kernel/centered_kernel_alignment`, `Summary/n_outside_null`, `Summary/metric`, calibration rows, and the new isotropy rows | Update any filter on the removed row names |
| 0.1.3 | `EmbedKitAnalyzer(d_max=...)` capped the isotropy spectrum | Accepted and ignored: isotropy is computed exactly from at most `n_max` rows | Nothing to do |

`k` is unchanged: `k=None` still means `max(5, round(sqrt(min(n, n_max)) / 2))`. The calibration uses
its own `k` (`report.calibration_k`, from the calibration rows only).

Before and after:

```python
# 0.1.2
report = EmbedKitAnalyzer().fit(X)
print(report.severity, report.intrinsic_dim.consensus)
for rec in report.recommendations:
    print(rec)

# 0.5.0
report = EmbedKitAnalyzer().fit(X)
print(report.n_outside_null, report.intrinsic_dim.reference_estimate)   # count; target-dim ID
for name, c in report.calibration.items():                              # None if calibrate=False
    print(name, round(c.z_score, 1))
for rec in report.recommendations:
    print(rec)
report.print_summary()                                                  # also lists report.properties
```

## 2. Intrinsic dimension (`IntrinsicDimensionEstimator`, `report.intrinsic_dim`)

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.5.0 | `result.consensus` (median, mean, min or max of the estimates) | Removed. The estimators measure different things and are not combined. `result.estimates` holds one value per method; `result.reference_estimate` and `result.reference_method` give the one the report uses (TwoNN when available) | Use `estimates["TwoNN"]`, or aggregate yourself (below) |
| 0.5.0 | `IntrinsicDimensionEstimator(aggregate="median")` | Argument removed (passing it raises `TypeError`) | Delete it |
| 0.5.0 | `result.uncertainty` (standard deviation across methods) | Renamed `between_method_spread`. It is not a sampling uncertainty. `standard_errors` and `intervals` (bootstrap, per estimator) cover sampling error only | Rename |
| 0.1.3 | Methods ran in a thread pool (TwoNN varied between runs with the same seed) | Sequential. Exact duplicate rows are removed first (`n_duplicates_removed`). A non-finite or non-positive estimate is a failure: warning, `NaN`, left out of the spread. `random_state` reaches every estimator that accepts it | Expect `NaN` entries in `estimates` instead of a zero or infinite value |
| 0.5.0 | Not available | `IntrinsicDimensionEstimator(n_bootstrap=20, scale_curve=True)` by default, so `fit` is slower; `ESS` is a supported method (about 100 s per fit at n = 5,000) | `n_bootstrap=0, scale_curve=False` for the old cost |

`scale_plateau` and `plateau_tolerance` appeared during the 0.5.0 development and were replaced by
`scale_max_deviation` (the largest relative deviation of the TwoNN scale curve from the full-sample
value, no threshold) before the release. They are not in any release, so 0.1.2 code never used them.

```python
# 0.1.2
id_hat = report.intrinsic_dim.consensus            # median of the estimators

# 0.5.0
ids = report.intrinsic_dim
twonn = ids.estimates["TwoNN"]
lo, hi = ids.intervals["TwoNN"]                    # sampling error only
median_id = float(np.nanmedian(list(ids.estimates.values())))   # the 0.1.2 consensus, if you want it
spread = ids.between_method_spread                 # was `uncertainty`
```

## 3. Hubness, geometry and kernel results

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.1.3 | `IsotropyResult.eigenvalue_spectrum` | Renamed `eigenvalue_spectrum_top` (the top `n_components`, for plotting only). The old name is a deprecated alias | Rename |
| 0.1.3 | Participation ratio from the top 512 eigenvalues only; `isotropy_score = effective_rank / 512` | Every statistic is exact, from the covariance or Gram matrix of at most `n_max=5000` rows (new `random_state`). `isotropy_score` divides by `min(n - 1, d)`. New `participation_ratio_corrected` and `participation_ratio_normalized` (corrected / d, in [0, 1]) | Use the normalised value to compare data sets of different n/d |
| 0.5.0 | Isotropy fields only | New on `IsotropyResult`: `isoscore`, `top_k_variance_share`, `top_k_coordinate_share` (k in 1, 3, 10), `spectrum_alpha`, `mean_vector_ratio`, `mean_cosine` | Additive |
| 0.1.3 | `KernelDiagnosticsResult.kernel_alignment` (uncentred: mostly class balance, 0.70 for two classes whatever X) | Renamed `centered_kernel_alignment` (`HKH`, about 0 for independent labels). Class labels use the indicator kernel; continuous labels (1-D floats, 2-D arrays) an RBF kernel with a median-heuristic bandwidth. Strings and Python lists are accepted as labels. `kernel_alignment` is a deprecated alias that returns the centred value | Rename; do not compare old and new values |
| 0.5.0 | `KernelDiagnosticsResult.condition_number` | Renamed `top_k_eigen_ratio` (`lambda_1 / lambda_k`; it was never a condition number). `condition_number` is a deprecated alias | Rename |
| 0.5.0 | `effective_rank`: entropy rank of the top 50 eigenvalues of the uncentred kernel (at most 50) | `tr(Kc)^2 / tr(Kc^2)` of the centred kernel over the whole spectrum. `spectral_gap` and `eigenvalues` are also of the centred kernel; `row_sum_skewness` keeps the uncentred kernel | Re-baseline any threshold on these |
| 0.1.3 | `DistanceConcentration`: `relative_contrast`, `concentration_ratio` | These remain. New `distance_cv` and `relative_contrast_per_query`. The report no longer thresholds `concentration_ratio` | Use `distance_cv` |
| 0.1.3 | `HubnessAnalyzer.fit(X)`, `NeighborConsistency.fit(X)` | Also accept `neighbor_index=` (a precomputed kNN graph). `HubnessAnalyzer` values are unchanged except for the kNN fixes in section 6 | Optional |

```python
# 0.1.2
kr = report.kernel
kr.condition_number, kr.kernel_alignment, report.geometry.isotropy.eigenvalue_spectrum

# 0.5.0
kr = report.kernel
kr.top_k_eigen_ratio, kr.centered_kernel_alignment, report.geometry.isotropy.eigenvalue_spectrum_top
```

## 4. High-level `EmbedKit`

All defaults below change what `EmbedKit(...).fit(X)` returns. Rows are for 0.1.2 code that relied on
the defaults.

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.6.0 | `EmbedKit().fit(X)` built one configuration | New `strategy="select"` (default). Without labels: the input centred and L2-normalised (`CenterNormalize`); no analysis report. With labels: `Identity`, `CenterNormalize`, PCA at 64/128/256 (+ L2) and, in supervised or regression mode, the refiner at `min(128, d // 2)` are fitted on 80% of the rows and scored by 10-NN accuracy (cosine) or 10-NN regression R^2 on the rest; a candidate replaces `Identity` only with a paired-bootstrap interval above 0; the winner is refitted on all rows. `selected_`, `selection_report_` and `strategy_` record it | `strategy="fixed"` (or any explicit `method`) for the 0.5.0 behaviour |
| 0.6.0 | `mode="self_supervised"` was the default, so `fit(X, y)` ignored `y` except for the report | `mode="auto"`: `"self_supervised"` without labels, `"supervised"` with them; continuous targets raise unless `mode="regression"` | Pass `mode="self_supervised"` to keep labels out of training (they are then used only to select among training-free candidates) |
| 0.6.0 | Bundles held `model.pt` | An `Identity` or `CenterNormalize` result saves `arrays.npz` instead (the training mean), and `config.json` has `strategy` and `selected` | Older bundles load as before |
| 0.4.0 | `mode="self_supervised"` trained a refiner (MLP, `KNNPairs(5)`, NT-Xent + 0.5 `AlignUniformLoss`) | New `method="auto"` is `"pca"` without labels: the top `target_dim` principal directions followed by L2 normalisation, no training. `transform`, `dim=`, `save` and `load` work as before. In E1, E6 and E7 the refiner never beat the better of PCA and PCA-whitening | `method="refiner"` trains the refiner. With `method="pca"`, setting `augmentation`, `loss`, `nested_dims` or a non-linear `architecture` raises `ValueError` |
| 0.4.0, 0.5.0 | The refiner was an MLP | `architecture="auto"`: `"linear"` (PCA-initialised linear map) in self-supervised mode, `"residual"` in supervised mode, `"mlp"` in regression mode. The resolved value is `architecture_` after `fit` | `architecture="mlp"` for the 0.1.2 model |
| 0.4.0 | `early_stopping_patience=10`, `val_split=0.1`, `monitor="k_skewness"` | Defaults `"auto"`, `None` and `"auto"`. Self-supervised: no early stopping, all rows used for training, `k_skewness` only logged. Supervised and regression: patience 10, 10% of rows held out, best weights restored | Pass `early_stopping_patience=10, val_split=0.1, monitor="k_skewness"` for the 0.1.2 behaviour |
| 0.5.0 | `mode="supervised"`: `SupConLoss(0.07)` + 0.5 `AlignUniformLoss`, MLP, `k_skewness` monitor | `SupConLoss(0.07)` alone on the residual adapter, monitored by `val_knn_accuracy` | See the snippet below |
| 0.5.0 | No regression mode | `mode="regression"`: `RankNContrastLoss` on an MLP with `GaussianNoise(0.05)` views, monitored by `val_ridge_r2`; warns with 1000 rows or fewer | Additive |
| 0.5.0 | `analysis_report` held the full report | With `strategy="fixed"`, `fit` runs the report with `calibrate=False, id_bootstrap=0, id_scale_curve=False`, since it reads only `suggested_target_dim`; with `strategy="select"` (the default since WP 7.2) it runs none and `analysis_report` is `None`. `analysis_report.n_outside_null` and `.calibration` are `None`, and the calibrated recommendations are absent | Run `EmbedKitAnalyzer().fit(X)` yourself for the full report |
| 0.1.3, 0.5.0 | `target_dim="auto"` | Can differ: 0.1.3 changed the ID estimates (duplicates removed, sequential runs); 0.5.0 uses 1.5 x TwoNN (Pets 22 to 24, digits 12 to 13) | Pass `target_dim=` to pin it |
| 0.1.3 | `random_state` did not reproduce a run | Two fits with the same `random_state` return identical arrays on CPU. The initial weights come from `random_state`, so numbers differ from 0.1.2 | Re-run baselines; do not expect 0.1.2 numbers |
| 0.4.0 | `config.json` had `mode`, `target_dim`, `hidden_dim`, `n_layers`, `input_dim` and the training settings | Also `method`, `architecture`, `nested_dims` | Bundles from 0.1.2 still load (section 6) |

Reproducing the 0.1.2 behaviour:

```python
from embedkit import EmbedKit
from embedkit.improvement.losses import AlignUniformLoss, CombinedLoss, SupConLoss

# 0.1.2: self-supervised refiner (MLP, early stopping on k_skewness)
ek = EmbedKit(method="refiner", architecture="mlp",
              early_stopping_patience=10, monitor="k_skewness", val_split=0.1)
Z = ek.fit_transform(X)

# 0.1.2: supervised, SupCon + 0.5 AlignUniform on the MLP
ek = EmbedKit(mode="supervised", method="refiner", architecture="mlp",
              loss=CombinedLoss([(SupConLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)]),
              early_stopping_patience=10, monitor="k_skewness", val_split=0.1)
Z = ek.fit_transform(X, y)

# 0.5.0 defaults: PCA without labels, residual adapter with SupCon and a held-out kNN score with labels
Z = EmbedKit(strategy="fixed").fit_transform(X)
Z = EmbedKit(mode="supervised", strategy="fixed").fit_transform(X, y)

# 0.6.0 defaults: centre + L2 without labels, validated selection with labels
Z = EmbedKit().fit_transform(X)
ek = EmbedKit().fit(X, y)
Z = ek.transform(X)
print(ek.selected_, ek.selection_report_)
```

## 5. `Trainer`, `EmbeddingRefiner`, losses and augmentations

### Refiner and trainer

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.4.0 | `EmbeddingRefiner(input_dim, target_dim)` was a 2-layer MLP | Default `architecture="linear"` (also `"mlp"` and `"residual"`). `Trainer.fit` calls `initialize_from_data`; a linear or residual refiner trained in a custom loop without it warns and starts from a random map | `EmbeddingRefiner(..., architecture="mlp")`. State dicts of 0.1.2 refiners load into it |
| 0.1.3 | Early stopping always minimised the monitor (`monitor="isotropy"` stopped on improvement) | `trainer.MONITORS` gives each monitor its direction | Re-check runs that used `isotropy` |
| 0.1.3 | An unknown `monitor` or `eval_metrics` name was silently ignored; evaluation errors became a warning and disabled early stopping | `ValueError` when the `Trainer` is built; evaluation errors propagate | Fix names |
| 0.1.3 | After patience ran out the model was `patience x eval_every` epochs past the best point | `restore_best=True` (default) reloads the best weights when early stopping is enabled; `best_score_`, `best_epoch_`; `history` gains `"lr"` and `"eval_epoch"` | `restore_best=False` for the old behaviour |
| 0.1.3 | `scheduler="step"` decayed per batch (training stalled) | Per epoch, x0.1 every `epochs // 3` epochs | Re-tune if you used it |
| 0.1.3 | `warmup_epochs=10` was stored and unused; `optimizer="lars"` was SGD with a 10x rate | `warmup_epochs` is a real linear warm-up and defaults to 0; `optimizer="lars"` raises `ValueError` | Use `"sgd"` or `"adam"` |
| 0.1.3 | Monitors used `HubnessAnalyzer`'s Euclidean, k=10 defaults | New `eval_k=10`, `eval_metric="cosine"` | Set `eval_metric="euclidean"` to match |
| 0.1.3 | `Trainer.fit` called `np.random.seed` and `torch.manual_seed` | It no longer touches the global RNGs; data order, augmentation draws and dropout come from generators derived from `random_state`, and global states are restored | Seed the global RNGs yourself if other code relied on it |
| 0.1.3 | `device=None` meant CUDA or CPU | CUDA, then Apple MPS, then CPU (falls back to CPU with a warning if the loss cannot run on MPS) | Pass `device="cpu"` to keep CPU |
| 0.1.3 | Labels had to be arrays | Lists and string labels are accepted (`encode_labels`); a length mismatch raises `ValueError` | Additive |
| 0.2.0 | Monitors were fixed functions | Monitors are called as `fn(Z, **context)` | A custom entry in `MONITORS` must accept `**_` |

### Losses

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.1.3 | `TripletLoss`: NaN gradients with Euclidean distance; unknown `mining` behaved as `"easy"` | Finite gradients; unknown `mining` raises `ValueError` | Fix the argument |
| 0.1.3 | `RankNContrastLoss` mishandled ties in label distance (batch-order dependent) | Follows Rank-N-Contrast: tied candidates share one denominator; label distances exact | Retrain embeddings made with discrete or repeated labels |
| 0.1.3 | `CombinedLoss` retried a failing member without labels | Errors propagate. Losses declare `uses_labels` (`True` for `SupConLoss`, `TripletLoss`, `RankNContrastLoss`) and only those receive labels | A custom `BaseLoss` that needs labels inside `CombinedLoss` must set `uses_labels = True` |
| 0.1.3 | `SupConLoss` accepted any label shape | Labels of shape `(N,)` or `(N, 1)`; otherwise `ValueError` | Squeeze or encode vector labels |
| 0.1.3 | Chunked `NTXentLoss` crashed when `chunk_size` did not divide 2N and averaged chunk means | Equals the unchunked loss for every `chunk_size` | None |

### Augmentations

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.1.3 | `GaussianNoise(std=0.05)` was absolute | `relative=True` by default: `std` multiplies the per-dimension standard deviation of the training data. Adaptive noise ignores `relative` | `GaussianNoise(std, relative=False)` for the old scale |
| 0.1.3 | `KNNPairs(k, hard_negatives, neighbor_index)` | `hard_negatives` removed; signature is `KNNPairs(k, neighbor_index, metric, mode)` | Drop `hard_negatives`; pass `neighbor_index` by keyword |
| 0.1.3 | Sequential `CompositeAugmentation` let `KNNPairs` discard the incoming view | Augmentations declare `kind` (`"pair"`: `KNNPairs`, `EmbeddingMixup`; `"view"`: the rest, the default). A sequential composite needs at most one pair augmentation, first; otherwise `ValueError`. View augmentations act on both views with independent draws | Reorder, e.g. `[KNNPairs(5), GaussianNoise(0.05)]`. A custom pair augmentation sets `kind = "pair"` |
| 0.1.3 | `FeatureMasking` drew one mask per batch; in-batch `KNNPairs` always took the nearest neighbour | One mask per row and view; one of the k neighbours sampled at random. Augmentations draw from generators set by `Trainer.fit` (`set_rng`) | Custom augmentations should draw with `self._randn`, `self._rand`, `self._randint` to follow `random_state` |

```python
# 0.1.2: absolute noise, MLP refiner
aug = GaussianNoise(std=0.05)
refiner = EmbeddingRefiner(input_dim=D, target_dim=d)

# 0.5.0: the same model and scale
aug = GaussianNoise(std=0.05, relative=False)
refiner = EmbeddingRefiner(input_dim=D, target_dim=d, architecture="mlp")
Trainer(model=refiner, augmentation=aug, loss=NTXentLoss(), epochs=100).fit(X)
```

## 6. Neighbours, I/O, saved bundles and dependencies

| Since | 0.1.2 | Now | What to do |
|---|---|---|---|
| 0.1.3 | `faiss-cpu` was a mandatory dependency and `backend="auto"` used it | An optional extra: `pip install embedkit[faiss]`. `backend="auto"` is scikit-learn. Opt in with `backend="faiss"` or `EMBEDKIT_KNN_BACKEND=faiss`. The "falling back to sklearn" warning is gone | Install the extra if you want FAISS |
| 0.1.3 | FAISS path returned squared Euclidean distances, or inner products sorted descending for cosine, and L2-normalised the caller's array in place | One contract: true Euclidean or cosine (`1 - cos`) distances, ascending, input never modified | Remove any squaring or sign conversion you added |
| 0.1.3 | `knn` cached results by array address and dropped column 0 as "self" | No cache; the query point is removed by index, so duplicate rows are not their own neighbours. `knn` raises `ValueError` for `k` outside `1 ... n - 1` and for an unknown backend | None |
| 0.1.3 | `clear_cache()` emptied the cache | A no-op that emits `DeprecationWarning` | Delete the call |
| 0.5.0 | `knn(metric="cosine")` with scikit-learn used the cosine metric directly | Searches the L2-normalised rows with the Euclidean reduction (same neighbours and distances, much faster); zero-norm rows keep the old path | None |
| 0.4.0 | `EmbedKit.load` assumed an MLP | Reads `method` and `architecture` from `config.json`; a bundle without them loads as `"refiner"` and `"mlp"`. `EmbedKit.load(directory, device=None)` (0.5.0) | 0.1.2 bundles load unchanged. Bundles saved by 0.5.0 do not load in 0.1.2 |

`pyproject.toml` otherwise keeps the same requirements as 0.1.2 (Python 3.10 or later, scikit-learn,
scikit-dimension, torch, umap-learn and the rest); only `faiss-cpu` moved to an extra.

Input checks (0.6.0): every public `fit` (the analyzers, `EmbedKitAnalyzer`, `EmbedKit`, `Trainer`,
the baselines, the hubness reductions) raises `ValueError` for NaN or infinite values, which
0.5.0 passed on, and warns once for constant columns and once for exact duplicate rows.
`transform` and the hubness reductions' queries check shape and values without the warnings. Clean
or drop such rows before the call (`embedkit.utils.validation.validate_embeddings` runs the same
check).

## 7. Deprecated aliases still accepted in 0.6.0

| Alias | Replacement | Warning | Removal |
|---|---|---|---|
| `IsotropyResult.eigenvalue_spectrum` | `eigenvalue_spectrum_top` | `DeprecationWarning` | "One release" (0.1.3 changelog); still present, no date set |
| `KernelDiagnosticsResult.kernel_alignment` | `centered_kernel_alignment` | `DeprecationWarning` | Same |
| `KernelDiagnosticsResult.condition_number` | `top_k_eigen_ratio` | `DeprecationWarning` | No date |
| `embedkit.utils.neighbors.clear_cache()` | none (delete) | `DeprecationWarning` | "Next release" (0.1.3 changelog); still present |

`DeprecationWarning` is hidden by default outside `__main__`; run with `python -W error::DeprecationWarning`
or under pytest to find the calls.

## 8. Results will differ

Same inputs and same code, different numbers. None of these raises an error.

- **kNN graph (0.1.3).** The old cache could return another array's neighbours, and a duplicated row
  could be listed as its own neighbour. Hubness, neighbour consistency (`std_consistency` was
  understated) and `GaussianNoise(adaptive=True)` change. Distance scales differ between the old
  FAISS and scikit-learn paths.
- **Participation ratio and isotropy score (0.1.3).** Exact for any d (0.1.2 used only the top 512
  eigenvalues, which misled above 2,000 dimensions); `isotropy_score` uses a different denominator.
- **Kernel (0.1.3, 0.5.0).** Alignment is centred; `effective_rank`, `spectral_gap` and `eigenvalues`
  come from the centred kernel.
- **Intrinsic dimension (0.1.3, 0.5.0).** Duplicates removed, sequential runs, no consensus, and
  `suggested_target_dim` from TwoNN.
- **Neighbour consistency (0.5.0).** Perturbations scale with the RMS per-coordinate standard
  deviation, so the statistic is translation-invariant.
- **Recommendations and `properties` (0.5.0).** Calibrated, so the same data can produce different
  recommendations. Normalised rows with a power-law spectrum remain uncalibrated (see the
  `EmbedKitAnalyzer` docstring).
- **Uniformity and distance concentration (0.5.0).** Pairwise distances in float64 through one BLAS
  product; agreement with 0.1.2 is about 1e-8 for uniformity.
- **Training (0.1.3).** `RankNContrastLoss` on tied labels, relative `GaussianNoise`, per-row
  `FeatureMasking`, `scheduler="step"`, best-weight restoration, early-stopping direction, seeding,
  and MPS selection change trained embeddings. A run with a fixed `random_state` will not reproduce
  0.1.2 numbers.
- **`EmbedKit` defaults (0.4.0, 0.5.0, 0.6.0).** Section 4. In 0.5.0 the self-supervised output
  was PCA; in 0.6.0 it is the input centred and L2-normalised, and with labels the output is the
  input itself unless a candidate wins on held-out rows. Any downstream result built on it changes.
- **float64 input to the baselines and hubness reductions (0.6.0).** Kept in float64 instead of
  being rounded to float32 first. Results for float32 input are unchanged.
