# EmbedKit implementation plan

**Written:** 2026-11-10, against version 0.1.2 (commit `4d2a513`).
**Source:** `docs/CRITICAL_REVIEW.md`. Defect numbers (D1 to D30), section numbers and experiment names (E1 to E9)
in this plan refer to that document. Read its sections 1 and 2 before starting.

This plan is written to be picked up in a later session without the conversation that produced it.

---

## How to use this plan

- The phases are ordered by dependency. Phases 0 to 3 are fixed. Phase 3 ends in a decision gate, and what happens
  in phases 5 and 6 depends on its outcome.
- Each phase is split into work packages (WP). One work package is one branch and one reviewable change.
- A work package is done when its acceptance criteria hold and the whole test suite passes.
- Every bug fix lands together with a test that fails on the old code. Write the test first and watch it fail.
- Update `CHANGELOG.md` in every work package that changes behaviour.

### Environment facts a new session needs

- Run Python with `conda run -n pytorch python ...` from the repository root. The project's `.venv` is broken
  (SciPy does not import).
- **FAISS crashes in that environment** (`OMP: Error #15`, or a segfault inside `index.search`). Until WP 0.1 is
  done, run tests with:

  ```bash
  conda run -n pytorch python -c "import sys; sys.modules['faiss']=None; import pytest; sys.exit(pytest.main(['tests','-q']))"
  ```

  This gives 96 passed on commit `4d2a513`.
- `conda run` buffers output until the process ends.
- `examples/data/pets_resnet50.npz` (ResNet50 features of Oxford Pets, 3,680 train / 3,669 test, 37 classes) exists
  locally and is git-ignored. `examples/04_image_classification.py` regenerates it.
- Scripts that reproduce the review's findings are in `benchmarks/review_2026_11/`. They are the starting point for
  the regression tests in phase 1.
- `docs/BUG-rnc-ties.md` holds a reproduction, a reference implementation and a proposed fix for D21.

### Decisions confirmed by the user (2026-11-10)

The user was asked about each of these and confirmed the choice shown. Several work packages depend on them.

| Decision | Choice | Reason |
|---|---|---|
| kNN backend default | scikit-learn. FAISS becomes an opt-in extra. | The default install must not crash. Exact FAISS search gives no accuracy benefit, only speed. |
| kNN result cache | Removed. Callers that need to reuse a graph pass it explicitly. | A cache keyed on a memory address cannot be made safe, and a content hash costs about as much as the search at these sizes. |
| Breaking changes | Allowed. The version is 0.1.x and the known user is the author. Each one is listed in the changelog. | Several defects are in the public contract (return values of `knn`, augmentation order). |
| Fixed thresholds in the report | Replaced by null-calibrated statistics in phase 4. Left untouched before that. | Changing them twice is wasted work. |
| New losses and augmentations | None before the phase 3 gate. | The pilot shows the refiner does not yet beat PCA. New objectives cannot be judged without the harness. |
| Benchmark data | Image sets, text sets and large unlabelled vector sets are all approved. | Three datasets are too few to support general claims. |
| Where large data and compute live | The user's remote GPU cluster (CUDA), already used in other projects. No large dataset is downloaded to the local Mac. | The user's instruction. See "Local machine and cluster" below. |
| Continuous integration | GitHub Actions. | The repository has a GitHub remote, and a macOS job catches platform failures such as the FAISS crash. |

### Local machine and cluster

- **Local Mac:** development, the unit tests, and small experiments only. That means the Pets features already on
  disk, scikit-learn digits, the synthetic sets, and small text tasks (under about 100 MB). Do not download CIFAR,
  STL-10, ANN-Benchmarks files or encoder weights here.
- **Remote cluster:** every large dataset, feature extraction with pretrained encoders, and the phase 3, 5 and 6
  training grids.
- The benchmark harness (WP 2.5) must therefore run unchanged in both places: a `--data-root` argument (or the
  `EMBEDKIT_BENCH_DATA` environment variable) for where datasets and cached features live, a `--device` argument,
  resumable runs, and results written as small JSON files that can be copied back and committed.
- **Not known to this plan:** how to reach the cluster (host, scheduler, storage paths, how the conda environment is
  set up there). Ask the user at the start of the session that first needs it, which is WP 2.5.

---

## Phase overview

| Phase | Goal | Depends on | Effort | Release |
|---|---|---|---|---|
| 0 | The test suite runs on a stock install and results are reproducible | none | 1 to 2 days | none |
| 1 | Every defect rated A or B is fixed, with a regression test | 0 | 4 to 6 days | 0.1.3 |
| 2 | Evaluation module, training-free baselines, benchmark harness | 1 | 4 to 5 days | 0.2.0 |
| 3 | Experiments E1 and E2, then the decision gate | 2 | 3 to 4 days plus compute | none |
| 4 | Diagnostics that are calibrated and consistent | 1 (2 for tests) | 5 to 7 days | 0.3.0 |
| 5 | Refiner improvements, each judged by the harness | 3 (gate) | 1 to 2 weeks | 0.4.0 |
| 6 | Supervised and regression modes | 2, WP 1.2 | 4 to 6 days | with 0.4.0 |
| 7 | Auto-configuration that selects among candidates; documentation | 3, 4, 5 | 4 to 5 days | 0.5.0 |

Phases 4 and 6 do not depend on the gate and can run in parallel with phase 3's compute.

---

## Phase 0: a working baseline

**Goal:** `pytest tests` passes on a clean macOS install with no workaround, and two runs with the same
`random_state` give the same result.

### WP 0.1 Make FAISS optional (D12)

- `pyproject.toml`: move `faiss-cpu` from `dependencies` to a new extra, `faiss = ["faiss-cpu>=1.7"]`.
- `embedkit/utils/neighbors.py`:
  - `backend="auto"` resolves to scikit-learn.
  - `backend="faiss"` uses FAISS and raises a clear `ImportError` if it is missing.
  - An environment variable `EMBEDKIT_KNN_BACKEND` overrides `"auto"`.
  - Remove the "falling back to sklearn" warning.
- Do not attempt to detect the libomp crash at runtime. It cannot be caught.
- **Acceptance:** the plain `pytest tests -q` command passes in the `pytorch` conda env without touching
  `sys.modules`.

### WP 0.2 Reproducibility (D13)

- `embedkit/api.py`: seed torch before `EmbeddingRefiner` is constructed.
- `embedkit/improvement/trainer.py`:
  - Stop calling `np.random.seed` and `torch.manual_seed` on the global state. Create a `torch.Generator` and a
    `np.random.Generator` from `random_state`.
  - Pass the torch generator to the `DataLoader`.
  - Give augmentations access to a generator. Add an optional `set_rng(np_rng, torch_gen)` method on
    `BaseAugmentation`, called by `Trainer.fit`. `KNNPairs` must stop using `np.random` (D24).
  - Dropout inside the model still uses the global torch state. Seed it locally with `torch.random.fork_rng` around
    the training loop, so the caller's state is restored afterwards.
- **Acceptance:**
  - New test: two `EmbedKit(random_state=0).fit_transform(X)` calls return identical arrays on CPU.
  - New test: the same for the low-level `Trainer` when the model is built under the same seed.
  - New test: `np.random.get_state()` is unchanged by `Trainer.fit`.

### WP 0.3 Continuous integration

- Add a GitHub Actions workflow: Python 3.10 and 3.12, Ubuntu and macOS, `pip install -e .[dev]`, `ruff check`,
  `pytest`.
- A second job installs the `faiss` extra on Ubuntu only and runs `tests/test_utils.py` with
  `EMBEDKIT_KNN_BACKEND=faiss`.
- **Acceptance:** both jobs are green on the phase 0 branch.

### WP 0.4 Stale documentation (D30)

- `CLAUDE.md` is git-ignored, so this is a local edit: remove the "design phase" statement and the claim that
  hubness wraps `scikit-hubness`.
- `README.md`: state that FAISS is optional and how to enable it.

---

## Phase 1: correctness

**Goal:** every defect rated A or B in the review is fixed. Release as 0.1.3.

**Testing rule for this phase:** each loss is compared with an independent, deliberately naive reference
implementation written in the test file (explicit loops are fine). Each analysis statistic is checked on data with
a known answer. "A gradient exists" is not an acceptance test.

### WP 1.1 Nearest neighbours (D1, D2, D15, D16, D17)

File: `embedkit/utils/neighbors.py`. Contract after this work package:

- `knn(X, k, metric="euclidean", backend="auto")` returns `(distances, indices)`, both of shape `(n, k)`.
- Distances are true distances for every backend: Euclidean distance (not squared), and cosine distance
  `1 − cos` for `metric="cosine"`. They are sorted in ascending order.
- The input array is never modified. Copy before `faiss.normalize_L2`.
- The query point is removed by index. Search for `k + 1`, drop the entry equal to the row index when it is
  present, otherwise drop the last entry.
- There is no cache. Delete `_cache`, `_MAX_CACHE` and `clear_cache`. Keep a `clear_cache` that does nothing and
  emits a `DeprecationWarning`, for one release.
- Callers that recompute the same graph get an explicit argument instead:
  - `HubnessAnalyzer.fit(X, neighbor_index=None)`.
  - `EmbedKitAnalyzer` computes the graph once and passes it to the analyzers that need the unperturbed graph.
- Review every caller for the changed distance semantics: `GaussianNoise._compute_radii` (`noise.py:27`),
  `visualization/embedding_viz.py`.

Tests:
- Results equal a brute-force NumPy reference for Euclidean and cosine, on random data and on data with exact
  duplicate rows.
- No row contains its own index, including with duplicates.
- `X` is byte-identical before and after a cosine query.
- Two different arrays of the same shape, created and freed in a loop of 50 iterations, always get their own
  neighbours (the D1 reproduction in `benchmarks/review_2026_11/verify.py`).
- `NeighborConsistency(...).fit(X).per_perturbation` has no repeated values for `n_perturbations=5` (D2).
- The FAISS and scikit-learn backends agree to 1e-4 (runs only in the FAISS CI job).

### WP 1.2 Losses (D3, D4, D20, D21, D27)

- `triplet.py` (D3): compute `sqrt(d2 + eps)` with the diagonal masked out before the square root, or use
  `torch.cdist`. Replace the per-anchor Python loop with masked tensor operations.
  - Test: gradients are finite for all three mining modes and both distances, including a batch that contains
    duplicate embeddings. Value equals a loop reference.
- `nt_xent.py` (D4, D20): delete the unused `torch.where` block. Collect per-row losses with
  `reduction="none"` and average over all rows.
  - Test: chunked equals full to 1e-6 for `chunk_size` in {1, 7, N − 1, N, N + 1, 2N − 1, 2N} with N = 50. Value
    equals a loop reference.
- `rnc.py` (D21): apply the fix in `docs/BUG-rnc-ties.md` (exact label distances, denominators that start at the
  first position of each tie group).
  - Tests, taken from that document: value and gradient equal `rnc_reference` for continuous, three-level,
    binary-vector and one-hot labels; the loss is invariant to a permutation of the batch rows.
- `combined.py` (D27): remove the `try / except`. Decide whether a loss receives labels from an explicit class
  attribute, `uses_labels: bool`, declared on each `BaseLoss` subclass.
  - Test: a shape error inside a supervised loss propagates.
- `supcon.py`: add a reference-value test for vector-valued labels, or raise a clear error if they are not
  supported. Today the mask silently becomes three-dimensional.

### WP 1.3 Trainer (D5, D8, D11, D18, D19, D22, D23)

File: `embedkit/improvement/trainer.py`.

- **Schedulers (D5):** step `cosine` per batch as now. Step `step` once per epoch. Keep `plateau` on evaluation.
- **Warm-up (D22):** implement linear warm-up over `warmup_epochs` with `SequentialLR`, and change its default to 0
  so that existing runs keep their learning-rate curve.
- **LARS (D22):** remove the `"lars"` option. It was never LARS.
- **Monitor direction (D18):** introduce a registry, `MONITORS: dict[str, tuple[Callable, Literal["min", "max"]]]`.
  Early stopping and `ReduceLROnPlateau` read the direction from it. An unknown monitor name raises at
  construction time.
- **Best weights (D19):** keep an in-memory copy of the best `state_dict`. Add `restore_best: bool = True`. Expose
  `best_epoch_` and `best_score_`.
- **Evaluation errors (D23):** remove the blanket `except`. Let errors propagate.
- **Monitor settings (D23):** compute monitor statistics with the trainer's metric. Add `eval_k` and `eval_metric`
  arguments, with cosine as the default because the output is L2-normalised.
- **Labels (D8):** add `embedkit/utils/validation.py::encode_labels(y)`. It accepts lists, string arrays, integer
  arrays, float vectors and 2-D float arrays, and returns a tensor plus a `kind` of `"class"` or `"continuous"`.
  Use it in `Trainer.fit` and in `KernelDiagnostics`.
- **Device (D11):** select `cuda`, then `mps`, then `cpu`. Check that every loss runs on MPS; fall back to CPU with
  a warning for any operation that does not. The cluster runs use CUDA, so also run the loss tests on a CUDA device
  there once before phase 3.
- Document that `early_stopping_patience` counts evaluations, not epochs.

Tests:
- `StepLR` with 30 epochs and `step_size=10` ends at `lr × 1e-3`.
- With a monitor in `max` mode, training does not stop while the score rises.
- After early stopping with `restore_best=True`, the monitor recomputed on the returned model equals
  `best_score_`.
- String labels and Python lists work in `Trainer.fit` and `KernelDiagnostics.fit`.

### WP 1.4 Augmentations (D9, D24, D25, D26)

- **Composition contract (D9):** give `BaseAugmentation` a class attribute `kind`, either `"pair"` (produces the
  second view from another sample: `KNNPairs`, `EmbeddingMixup`) or `"view"` (perturbs each view independently:
  noise, dropout, masking). `CompositeAugmentation(mode="sequential")` allows at most one `pair` augmentation,
  requires it to be first, raises `ValueError` otherwise, and applies each `view` augmentation once to each view.
- **`KNNPairs` (D24):** remove the unused `hard_negatives` argument. In the in-batch path, sample one of the k
  neighbours at random, as the precomputed path does. Add a `metric` argument.
- **`FeatureMasking` (D25):** draw an independent mask per row. Keep `block_size` and document that blocks are
  only meaningful for ordered features.
- **`GaussianNoise` (D26):** add `relative: bool = True`. When set, `std` multiplies the per-dimension standard
  deviation of the training data, computed in `precompute`. Changing the default is a breaking change; list it in
  the changelog.

Tests:
- `[GaussianNoise, KNNPairs]` raises. `[KNNPairs, GaussianNoise]` gives two noisy views.
- For unit-norm 768-d data, the default `GaussianNoise` adds noise whose norm is about `std` times the norm of the
  centred signal.

### WP 1.5 Analysis fixes that cannot wait for phase 4 (D6, D7, D8, D10, D14)

- **Isotropy (D6):** compute the participation ratio from traces, which is exact at any dimension:
  `PR = tr(C)² / tr(C²)`, with `tr(C) = ‖Xc‖²_F / (n − 1)` and `tr(C²) = ‖Xc Xcᵀ‖²_F / (n − 1)²`. Use the Gram
  matrix when n < d and the covariance otherwise, on a subsample of at most 5,000 rows. Report
  `participation_ratio_normalized = PR / min(n − 1, d)` and make the report use it. Keep the truncated spectrum for
  plotting only, and name it accordingly.
- **Kernel alignment (D7):** centre both kernels (`HKH`). For `kind == "continuous"` labels use an RBF label kernel
  with a median-heuristic bandwidth. Rename the field to `centered_kernel_alignment`, and keep the old name as a
  deprecated alias for one release.
- **Distance concentration (D10):** add `distance_cv` (standard deviation over mean of pairwise distances) and
  `relative_contrast_per_query` (median over queries of `(d_max − d_min) / d_min`, with zero distances excluded).
  Make the report use `distance_cv`. Keep the old fields.
- **Intrinsic dimension (D14):** remove exact duplicate rows before estimation and record how many were removed.
  Make sure the estimators receive `random_state`. Treat a non-finite or non-positive estimate as a failure.

Tests:
- Isotropic Gaussian, n = 3,000: `participation_ratio_normalized` is within 0.05 of the same value at d = 1,000 and
  d = 2,500, and the report gives no anisotropy recommendation at either.
- Labels independent of X: centred alignment is below 0.05 in absolute value, for 2 and for 10 classes.
- Labels that are a function of the cluster: centred alignment is above 0.5.
- `distance_cv` changes by less than 1% when one row is duplicated.
- On scikit-learn digits, two analyzer runs with the same `random_state` give identical ID estimates with no NaN.

### Phase 1 exit criteria

- All tests pass in CI on both operating systems.
- `benchmarks/review_2026_11/verify.py` and `verify2.py`, updated for the new API, show none of the defect
  behaviours.
- Version 0.1.3 is tagged, with a changelog entry per defect.

---

## Phase 2: evaluation and baselines

**Goal:** any transform, trained or not, can be scored with one call on held-out data. Release as 0.2.0.

### WP 2.1 Metrics

New module `embedkit/evaluation/metrics.py`. Pure functions on NumPy arrays, each with a known-answer test.

| Function | Notes |
|---|---|
| `knn_accuracy(Z_train, y_train, Z_test, y_test, k=5, metric="cosine")` | Neighbours come from the training set only. |
| `linear_probe(Z_train, y_train, Z_test, y_test, task="auto")` | Logistic regression for classes, ridge for continuous targets. Inputs standardised on the training set. Regularisation chosen by internal cross-validation on the training set. |
| `recall_at_k(Z_query, Z_gallery, relevance, k=10)` | For retrieval data. |
| `trustworthiness(X, Z, k=10)`, `continuity(X, Z, k=10)` | Wrap scikit-learn. Subsample to at most 5,000 points with a fixed seed. |
| `knn_overlap(X, Z, k=10)` | Mean fraction of shared neighbours between the two spaces. |
| `linear_cka(X, Z)` | Centred. |
| `effective_rank(Z)` | Exponential of the entropy of the normalised singular values, on the full spectrum. |
| `k_skewness(Z, k=10, metric="cosine")` | Shares code with `HubnessAnalyzer`. |
| `nc1(Z, y)` | Trace of within-class covariance times the pseudo-inverse of between-class covariance, divided by the number of classes. |
| `silhouette(Z, y)` | Wraps scikit-learn, subsampled. |

### WP 2.2 Training-free transforms

New module `embedkit/baselines.py`. Each class has `fit(X, y=None)`, `transform(X)` and `fit_transform`, and stores
only statistics of the data passed to `fit`.

- `Identity`
- `CenterNormalize` (subtract the mean, then L2-normalise)
- `Standardize`
- `PCAReduce(n_components, whiten=False)`, with `n_components` capped at the rank of the training data
- `ZCAWhiten(eps)`
- `AllButTheTop(n_remove)` (Mu et al. 2018), and `AllButTheTop(n_remove, n_components)` for the reduce-then-clean
  variant (Raunak et al. 2019)
- `ConceptorNegation(alpha)` (Liu et al. 2019)
- `RandomProjection(n_components, random_state)`

Tests: output shapes; whitened training data has identity covariance to 1e-3; `AllButTheTop` output has zero
projection on the removed directions; `transform` on new data uses the training statistics.

### WP 2.3 Protocol

New module `embedkit/evaluation/protocol.py`.

- `evaluate_transform(make_transform, X_train, X_test, y_train=None, y_test=None, seeds=(0, 1, 2, 3, 4), metrics=...)`
  - `make_transform(seed)` returns an unfitted object with `fit` and `transform`. `EmbedKit` already satisfies
    this.
  - It fits on the training split only, transforms both splits, computes the requested metrics, records fit and
    transform time, and returns a tidy `pandas.DataFrame` with one row per seed and metric.
  - `transductive=True` fits on the concatenation of both splits without labels. The flag is recorded in the
    output.
- `compare(results, reference="raw")` returns mean ± sd per method, and the paired difference to the reference with
  a bootstrap interval over test points.

### WP 2.4 Trainer integration

- Register the label-free metrics as monitors in the `MONITORS` registry from WP 1.3: `knn_overlap` (max),
  `effective_rank` (max), `val_loss` (min, the training loss on the validation split with augmentation), and the
  existing three.
- When labels are given and `val_split > 0`, carry `y` into the validation split and register `val_knn_accuracy`
  (max).
- Do not change the default monitor yet. Phase 3 decides it.

### WP 2.5 Benchmark harness

New top-level directory `benchmarks/` (not part of the installed package).

- `benchmarks/datasets.py`: loaders that return `(X_train, y_train, X_test, y_test, meta)` and cache features under
  `benchmarks/data/` (git-ignored).
  - `pets_resnet50` (exists locally)
  - `digits` (scikit-learn; marked as not a learned embedding)
  - `synthetic_anisotropic` (port from `benchmarks/review_2026_11/pilot/datasets.py`)
  - `cifar100_dinov2` and `stl10_clip`: write the feature-extraction script and run it on the remote cluster.
    Do not download these to the local Mac.
  - Two text sets, embedded with a sentence-transformers model: one classification task (for example Banking77 or
    20 Newsgroups) and one STS task (STS-B). The extraction also runs on the cluster.
  - `glove100`, `sift128`, `gist960` from ANN-Benchmarks, unlabelled, for phase 4 calibration and phase 7 scaling.
    Cluster only.
  - Every loader takes the data root from `--data-root` / `EMBEDKIT_BENCH_DATA` and fails with a clear message when
    a dataset is absent. It never downloads implicitly.
- `benchmarks/run.py --experiment e1 --datasets ... --seeds 5 --device cuda`: writes one JSON file per dataset and
  method under `benchmarks/results/<experiment>/`, and skips combinations that already exist.
- A job-submission template for the cluster's scheduler, once the user has described it.
- `benchmarks/report.py`: turns the JSON files into Markdown tables.

### Phase 2 exit criteria

- `benchmarks/run.py --experiment e1 --datasets pets_resnet50 digits synthetic_anisotropic --seeds 3` reproduces
  the baseline rows of the pilot table in review section 3.7 to within 0.005. The `EmbedKit` rows will differ
  because phase 1 changes its behaviour; record the new values.
- Version 0.2.0 is tagged.

---

## Phase 3: experiments E1 and E2, and the decision gate

**Goal:** know whether the self-supervised refiner improves on training-free baselines, and which label-free
statistic can select and stop a run.

### WP 3.1 Run E1

- Runs on the remote cluster. Copy the result JSON files back into `benchmarks/results/` and commit them.
- Follow review section 5, E1. At least five datasets: Pets, the synthetic set, one more image set, two text sets.
  Five seeds.
- Methods: every class in `embedkit.baselines` over d ∈ {16, 32, 64, 128, 256} where it applies, parametric UMAP if
  `umap-learn`'s parametric mode is usable in the environment, `EmbedKit` default, `EmbedKit` with fixed d ∈
  {32, 64, 128}, and `EmbedKit` supervised.
- **Selection rule:** choose every hyperparameter, including the baseline's d and `n_remove`, on a validation part
  of the training split. Report the test value of the selected setting. The pilot did not do this.

### WP 3.2 Run E2

- Follow review section 5, E2. On three datasets, about 30 refiner configurations each, log every monitor in the
  registry next to test kNN accuracy and linear-probe accuracy every 5 epochs.
- Report the Spearman correlation within runs and across runs, per monitor.

### WP 3.3 Write up

- `docs/BENCHMARK_E1_E2.md`: tables, the selection rule, compute used, and a plain statement of the outcome.
- Add a note at the top of `docs/UNSUPERVISED_REFINEMENT_REPORT.md` that its conclusions rest on one seed and on
  selection by test accuracy, with a link to the new document.

### Decision gate

Compare the best self-supervised `EmbedKit` configuration, selected on validation data, with the best training-free
baseline at the same output dimension. Call a dataset a "win" when the paired difference in the primary metric
(kNN accuracy for labelled sets, STS correlation or recall@10 otherwise) is positive and its interval excludes 0.

| Outcome | Meaning | What follows |
|---|---|---|
| **A.** Wins on at least half the datasets | The refiner adds value. | Phase 5 as written. |
| **B.** No wins, but within the seed spread of the baseline at lower dimension, with trustworthiness at least equal to PCA | The refiner is a compression method. | Phase 5 limited to WP 5.1, 5.2 and 5.3. Reposition the self-supervised mode as compression in the README. |
| **C.** Loses to baselines on most datasets | The self-supervised mode as designed does not help. | Phase 5 limited to WP 5.1 and one two-day attempt at WP 5.2. If that fails, `EmbedKit(mode="self_supervised")` returns the best baseline, and development effort moves to phases 4, 6 and 7. |

Independent of the outcome:
- Set the default monitor to the statistic with the best within-run correlation in E2. If none exceeds a median ρ
  of 0.5 in the right direction, disable early stopping by default and use a fixed epoch budget.
- This is the user's decision to confirm. Present the tables and a recommendation; do not proceed to phase 5
  without that confirmation.

---

## Phase 4: diagnostics

**Goal:** every number in the report is consistent across n and d, comparable against a reference, and honours the
requested metric. Release as 0.3.0. This phase does not depend on the gate.

### WP 4.1 Null models and calibrated statistics (review 3.2)

- New module `embedkit/analysis/null.py`:
  - `spectrum_matched_gaussian(X, n, rng)`: a Gaussian sample with the covariance spectrum of X.
  - `column_permuted(X, rng)`: each column permuted independently, which keeps the marginals and removes
    dependence.
  - `null_distribution(stat_fn, X, model, n_draws=20)`.
- `EmbedKitAnalyzer(calibrate=True)` reports, for k-skewness, `distance_cv`, uniformity and neighbour consistency:
  the observed value, the null mean and standard deviation, and a z-score. This costs about 20 extra kNN searches
  on the subsample; expose `n_null_draws`.
- `_assess_severity_and_recommendations` uses z-scores (for example above 3) in place of the constants. Remove the
  `severity` sum or redefine it as the count of statistics outside the null range, and document which.

Tests: for data drawn from the null model itself, the z-scores are within ±3 in at least 95% of 40 repetitions,
across n ∈ {300, 3,000} and d ∈ {32, 512}.

### WP 4.2 Intrinsic dimension (review 3.3)

- Report estimators separately. Remove `consensus` from the printed summary and stop using a median across methods.
- Add ID as a function of scale: TwoNN on nested subsamples (n, n/2, n/4, n/8), or GRIDE through the `dadapy`
  package as an optional extra. Report the curve and whether a plateau exists.
- Add bootstrap intervals (20 resamples) for each estimator, and rename the current `uncertainty` field to
  `between_method_spread`.
- Add `skdim.id.ESS` to the supported methods.

Tests: on `skdim.datasets` hyperspheres and hypercubes of known dimension 5, 10 and 20, embedded linearly in 256
dimensions with n = 5,000, the reported interval of at least one default estimator contains the true value. This is
also the first part of experiment E3.

### WP 4.3 Isotropy and spectrum (review 4.6)

- Add `isoscore` (Rudman et al. 2022), `mean_vector_ratio` (norm of the mean over the mean norm),
  `mean_cosine` (average cosine similarity of random pairs, uncentred), `top_k_variance_share` for k ∈ {1, 3, 10},
  and `spectrum_alpha` (power-law decay exponent of the eigenvalues, fitted on a log-log scale).
- Describe these in the report as properties. Remove wording that calls anisotropy or clustering a defect.

### WP 4.4 One metric everywhere (D28) and kernel spectrum (D29)

- `EmbedKitAnalyzer(metric=...)` reaches every analyzer. For `"cosine"`, L2-normalise once and use Euclidean
  routines where they are equivalent.
- `KernelDiagnostics`: centre the kernel before the eigendecomposition. Report effective rank from
  `tr(K)² / tr(K²)`, which needs no truncation. Rename `condition_number` to `top_k_eigen_ratio` or remove it.

### WP 4.5 Run E3 and E5

- E3: the calibration sweeps in review section 5 (known-ID manifolds, Gaussian families, subsampling stability).
  Add them as `benchmarks/run.py --experiment e3`.
- E5: controlled damage of a good embedding (review section 5). Use the baselines from phase 2 as the repair
  methods.
- Write `docs/BENCHMARK_DIAGNOSTICS.md` with a table per statistic: bias, variance, dependence on n.

### Phase 4 exit criteria

- No statistic in the report changes its qualitative reading when the same dataset is subsampled from 20,000 to
  2,000 rows.
- The isotropic-Gaussian and independent-label tests from WP 1.5 still pass.

---

## Phase 5: refiner improvements (scope set by the gate)

**Rule for this phase:** each work package is accepted only if `benchmarks/run.py --experiment e6` shows a gain
over the previous default on at least half the datasets, by the gate's definition of a win. Otherwise the feature
stays available but is not made the default, or it is dropped. Add one thing at a time.

### WP 5.1 Residual adapter and input standardisation (review 4.5, items 1 and 2)

- `EmbeddingRefiner(architecture="residual")`: `z = normalize(W x̃ + g(x̃))`.
  - `x̃` is the input standardised by a non-trainable layer whose statistics are set in `Trainer.fit`.
  - `W` is a linear map initialised from the top `target_dim` principal directions of the training data.
  - `g` is the current MLP with its last layer initialised to zero.
- `Trainer.fit` calls a new `model.initialize_from_data(X_train)` when the model defines it.
- Save and load the new buffers in `utils/io.py` and `EmbedKit.load`. Store the architecture in `config.json`.
- **Acceptance:** before any training step, `transform` equals `PCAReduce(target_dim)` followed by L2
  normalisation, to 1e-4. This makes PCA the starting point of every run.

### WP 5.2 Positives and negatives that use the diagnostics (review 3.4, 4.4)

- `KNNPairs(mode="mutual")`: keep only mutual neighbours, and fall back to the plain nearest neighbour for points
  with none.
- `NTXentLoss(neighbor_mask=...)`: exclude the anchor's k nearest neighbours in the input space from the
  denominator. This needs the batch's global indices, so extend `BaseLoss.forward` with an optional `indices`
  argument and pass it from the trainer.
- `NTXentLoss(decoupled=True)` (Yeh et al. 2022): remove the positive from the denominator.
- Dropout positives: a new `IdentityPair` augmentation returning `(x, x)`, so that the two views differ only by the
  refiner's own dropout (Gao et al. 2021).

### WP 5.3 Matryoshka output (review 3.3, 4.5 item 3)

- `MatryoshkaLoss(base_loss, dims=(16, 32, 64, 128))`: the sum of the base loss on each prefix of the output.
- `EmbedKit.transform(X, dim=None)` truncates and renormalises.
- When this is accepted, `target_dim="auto"` means "train at min(256, d) with nested prefixes", and
  `suggested_target_dim` becomes advice about where to truncate.

### WP 5.4 Structure preservation and covariance regularisation (gate outcome A only)

- `NeighborPreservationLoss`: a penalty on the difference between input-space and output-space cosine similarities
  over each anchor's k neighbours.
- `VICRegLoss` (Bardes et al. 2022), usable alone or inside `CombinedLoss`.
- `MarginalResample` augmentation (SCARF; Bahri et al. 2022), applied in the PCA basis fitted in WP 5.1.

### WP 5.5 Hubness reduction as transforms (any gate outcome)

- New module `embedkit/hubness_reduction.py` with `MutualProximity`, `LocalScaling`, `CSLS` and `DisSimLocal`.
  These change distances, not coordinates, so their interface is
  `fit(X_reference)` and `kneighbors(X_query, k)`.
- Add them to E1 as retrieval-time methods, scored by kNN accuracy and k-skewness.

### WP 5.6 Run E6 and E7

- E6: the one-factor ablation in review section 5, restricted to the features accepted above.
- E7: learning curves over training-set size, and inductive against transductive fitting.
- Append the results to `docs/BENCHMARK_E1_E2.md`. Release as 0.4.0.

---

## Phase 6: supervised and regression modes

**Goal:** the mode where the pilot showed real gains is correct, has sensible defaults and is measured. This phase
needs phase 2 and WP 1.2, and does not depend on the gate.

### WP 6.1 Defaults

- `EmbedKit(mode="supervised")`: remove `AlignUniformLoss` from the default loss (review 3.4). Use
  `val_knn_accuracy` as the monitor, in `max` mode, with `restore_best=True`.
- Add `mode="regression"`, which selects `RankNContrastLoss` and a ridge-probe monitor. Today a user has to build
  this by hand.

### WP 6.2 Class collapse

- Report `nc1` on the training and validation splits in `Trainer.history`.
- Add `SupConLoss(spread_weight=...)`: a class-conditional InfoNCE term among same-class samples, which keeps
  within-class structure (Chen et al. 2022).

### WP 6.3 Run E8

- Classification: SupCon, SupCon with uniformity (the old default), SupCon with spread, against a linear probe and
  LDA on raw embeddings. Report kNN accuracy, probe accuracy, NC1, and coarse-to-fine transfer where the dataset
  has a label hierarchy (CIFAR-100 superclasses).
- Label efficiency: 1, 5, 20 and 100 labels per class. Label noise: 0%, 10%, 20%, 40%.
- Regression: the fixed RnC against the 0.1.2 RnC and against ridge regression on raw embeddings, on targets with
  and without ties. Add ConR or SupReMix only if RnC fails to beat ridge.
- Rerun the downstream experiment arm named in `docs/BUG-rnc-ties.md` with the fixed loss. That document states this as
  an untested hypothesis.
- Write `docs/BENCHMARK_SUPERVISED.md`.

---

## Phase 7: auto-configuration and documentation

**Goal:** `EmbedKit()` with defaults does something defensible, and the documentation says what was measured.
Release as 0.5.0.

### WP 7.1 Run E4

- For every benchmark dataset, compute the calibrated report from phase 4 and the gain of every intervention
  (baselines, hubness reduction, refiner). Regress gain on the diagnostics.
- Keep a recommendation in the report only if the data support it. Each kept recommendation names a class in the
  library.

### WP 7.2 Selection-based auto-configuration (review 4.7)

- `EmbedKit(strategy="select")`, which becomes the default:
  1. Split off a validation part of the training data.
  2. Build candidates: `Identity`, the baselines suggested by WP 7.1's rules, and one or two refiner
     configurations.
  3. Score each on the validation part with the criterion chosen in phase 3 (validation kNN accuracy when labels
     exist, otherwise the label-free statistic that E2 supported).
  4. Refit the winner on all the training data and expose `selected_`, plus a `selection_report_` table.
- `Identity` is always a candidate, so the default can no longer return something worse than the input by its own
  criterion.
- Remove `suggested_k` and `suggested_sigma` from the report unless something consumes them by this point.
- Move the analysis inside the training part of the split, so that validation rows no longer influence the
  configuration.

### WP 7.3 Robustness and scaling (E9)

- Input validation at the public entry points: NaN or infinite values raise; constant columns and duplicate rows
  give a warning with counts; float64 is converted.
- Time and memory of `EmbedKitAnalyzer.fit` and `EmbedKit.fit` for n ∈ {1e3, 1e4, 1e5, 1e6} and d ∈ {128, 1,024,
  4,096}. Record the table in the manual.

### WP 7.4 Documentation

- `README.md`: lead with what the benchmarks show, including where the library does not help.
- `docs/manual.md` and `skills/embedkit/`: update for every API change in phases 1 to 7. Both currently describe
  the 0.1.x behaviour.
- A migration note listing each breaking change from 0.1.2.

---

## Tracking

| WP | Title | Status |
|---|---|---|
| 0.1 | FAISS optional | done (2026-11-10, with WP 1.1) |
| 0.2 | Reproducibility | done (2026-11-10) |
| 0.3 | Continuous integration | workflow written (2026-11-10); green run pending a push |
| 0.4 | Stale documentation | done (2026-11-10) |
| 1.1 | Nearest neighbours | done (2026-11-10) |
| 1.2 | Losses | done (2026-11-10) |
| 1.3 | Trainer | done (2026-11-10); CUDA run of the loss tests on the cluster still to do |
| 1.4 | Augmentations | done (2026-11-10) |
| 1.5 | Analysis fixes | done (2026-11-10); see deviation note below |
| 2.1 | Metrics | not started |
| 2.2 | Training-free transforms | not started |
| 2.3 | Protocol | not started |
| 2.4 | Trainer integration | not started |
| 2.5 | Benchmark harness | not started |
| 3.1 | Run E1 | not started |
| 3.2 | Run E2 | not started |
| 3.3 | Write up and gate | not started |
| 4.1 | Null models | not started |
| 4.2 | Intrinsic dimension | not started |
| 4.3 | Isotropy and spectrum | not started |
| 4.4 | Metric consistency, kernel spectrum | not started |
| 4.5 | Run E3 and E5 | not started |
| 5.1 | Residual adapter | waits for gate |
| 5.2 | Positives and negatives | waits for gate |
| 5.3 | Matryoshka output | waits for gate |
| 5.4 | Structure preservation, VICReg | waits for gate |
| 5.5 | Hubness reduction | not started |
| 5.6 | Run E6 and E7 | waits for gate |
| 6.1 | Supervised defaults | not started |
| 6.2 | Class collapse | not started |
| 6.3 | Run E8 | not started |
| 7.1 | Run E4 | not started |
| 7.2 | Selection-based auto-configuration | not started |
| 7.3 | Robustness and scaling | not started |
| 7.4 | Documentation | not started |

### Phase 1 notes (2026-11-10)

- **Deviation in WP 1.5 (confirmed by the user, 2026-11-10).** `PR / min(n − 1, d)` cannot meet the WP 1.5 acceptance test (within 0.05 at
  d = 1,000 and 2,500, n = 3,000): the plug-in participation ratio is biased by about `1 / (1 + d/n)`,
  giving 0.75 and 0.55. `participation_ratio` stays the plug-in value. A new
  `participation_ratio_corrected` uses the unbiased Gaussian (Wishart) estimate of `tr(Σ²)`, and
  `participation_ratio_normalized = corrected / d` is what the report thresholds. `isotropy_score` and
  `effective_rank` still carry the finite-sample bias (0.85 and 0.66 in the same setting); phase 4 (WP 4.1,
  4.3) should revisit them with the null models.
- **`SupConLoss` with vector labels** raises `ValueError` (confirmed by the user). Multi-label
  supervision, if wanted, belongs to phase 6.
- **Provisional `distance_cv` threshold:** 0.02 (fires for iid Gaussians from about d = 1,250 at
  n = 2,000; the old `concentration_ratio > 0.8` rule fired from about d = 1,000). Replace in WP 4.1.
- **Still open from phase 1 exit criteria:** a green CI run on both operating systems (needs a push);
  the loss tests have not been run on CUDA.
- Phase 0 was completed after phase 1 on the same branch (WP 0.2 to 0.4). The CI workflow has not
  run yet: it needs the branch pushed.
- `skills/embedkit/references/metric_guide.md` still describes 0.1.2 fields and thresholds (WP 7.4).

## Open points for the next session

All design decisions in this plan are confirmed. Two practical points remain:

1. **Cluster access details** (host, scheduler, storage paths, environment setup). Needed from WP 2.5 onwards.
2. **The phase 3 gate.** The outcome is the user's call. Present the E1 and E2 tables with a recommendation and
   wait for confirmation before starting phase 5.
