# Changelog

## Unreleased

Phase 3 of `docs/IMPLEMENTATION_PLAN.md`: benchmarks E1 and E2 and the decision gate
(`docs/BENCHMARK_E1_E2.md`).

### Changed
- **`EmbedKit` no longer stops early by default** (`early_stopping_patience=None`, was 10): it trains
  for the fixed `epochs` budget. In E2 no label-free monitor tracked test accuracy within a run, and
  the validation contrastive loss moved against it. `val_split` now defaults to `None`: 0.1 of the
  rows are held out only when early stopping is enabled, so by default the refiner trains on every row.
  Pass `early_stopping_patience=10` to restore the 0.2.0 behaviour.

### Added
- `EmbeddingRefiner(architecture="residual")` (WP 5.1): `z = normalize(W x̃ + g(x̃))`, with `x̃` the
  input standardised by fixed statistics, `W` initialised from the top principal directions and the
  MLP `g` with a zero-initialised output layer, so training starts exactly at PCA + L2 normalisation.
  `architecture="linear"` keeps only `W`. `Trainer.fit` calls `model.initialize_from_data` on the
  training rows when the model defines it. `EmbedKit(architecture=...)` passes it through and stores it
  in `config.json`; `EmbedKit.load` restores it. The default stays `"mlp"` until E6 accepts the change.

### Benchmarks
- `benchmarks/run.py --experiment e6`: the phase 5 ablation grid (architecture arms at auto and fixed
  d, with raw, PCA and PCA-whitening as references); `benchmarks/report_e6.py` applies the phase 5
  acceptance rule; `--shard I/N` splits a dataset's grid between jobs.
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
