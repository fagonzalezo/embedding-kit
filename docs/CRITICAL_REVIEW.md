# EmbedKit: critical review, improvement roadmap and experiment plan

**Reviewed:** version 0.1.2, commit `4d2a513`, on 2026-11-10.
**Scope:** all of `embedkit/` (about 2,500 lines), the tests, and `docs/UNSUPERVISED_REFINEMENT_REPORT.md`.

How to read the evidence labels in this document:

- **Measured** means I ran code against this commit and saw the behaviour. The scripts are listed in the appendix.
- **By reading** means the conclusion comes from reading the source and was not executed.
- Literature references were collected by a separate survey with web verification. A dagger (†) marks a reference
  whose venue or one other field could not be confirmed. Claims about what a paper found are the paper's own.

---

## 1. Summary

EmbedKit has a clean, modular shape, and the idea of driving refinement from diagnostics is a good one. The current
implementation does not yet deliver on it, for four reasons.

1. **There is no evidence that the refiner beats doing nothing, and none that it beats cheap baselines.**
   The only study in the repository (Oxford Pets, ResNet50) reports +0.4 pp kNN accuracy over raw features for the
   best unsupervised configuration. That is one dataset and one seed, the gain is smaller than one standard error
   (about 0.5 pp at n = 3,669), and the configuration was chosen by looking at test accuracy. PCA, whitening and
   all-but-the-top were never run. A pilot run for this review (section 3.7) did run them: on three datasets and
   three seeds, the default self-supervised refiner never beat the best training-free baseline, and it was below
   raw embeddings on kNN accuracy in all three.
2. **The "diagnostics-driven" auto-configuration uses one number.** `EmbedKit.fit` reads only
   `suggested_target_dim` from the report. `suggested_k`, `suggested_sigma`, `severity`, the hubness results and
   the isotropy results are computed and then ignored (measured, `api.py:173-191`).
3. **Several diagnostics are miscalibrated or internally inconsistent.** Thresholds are fixed constants although
   the statistics depend on n, d and k. An isotropic Gaussian is reported as anisotropic above 2,000 dimensions.
   Kernel alignment is 0.70 for labels that are independent of the data.
4. **There are correctness bugs that affect results silently.** The kNN cache returns stale neighbours, `TripletLoss`
   produces NaN gradients, chunked `NTXentLoss` crashes, and `RankNContrastLoss` mishandles ties (already
   documented in `docs/BUG-rnc-ties.md`).

The most valuable next step is not a new loss. It is an evaluation harness with training-free baselines and
held-out downstream metrics, so that every later change can be judged. Section 6 gives the order I recommend.

---

## 2. Correctness defects

Severity: **A** changes results silently, **B** crashes or blocks a documented feature, **C** is misleading or
wasteful.

### 2.1 Confirmed by running the code

| # | Sev | Where | Defect | Evidence |
|---|---|---|---|---|
| 1 | A | `utils/neighbors.py:28` | The kNN cache key is the array's memory address, shape, dtype, k and metric. The cache does not hold a reference to the array, so a freed array's address is reused and a different array gets the old neighbours. In-place edits also return the old result. | Fresh random arrays of the same shape received another array's neighbours in 11 of 50 and 20 of 50 trials in two runs. After `A[:] = new_values`, `knn(A)` returned the old indices. |
| 2 | A | `analysis/geometry.py:155-156` | A consequence of defect 1: `NeighborConsistency` allocates a new perturbed array per round, the address repeats, and later rounds reuse an earlier round's neighbours. `std_consistency` is therefore understated. | `per_perturbation` = `[0.5265, 0.5185, 0.5227, 0.5185, 0.5227]`: rounds 4 and 5 repeat rounds 2 and 3 exactly. |
| 3 | A | `improvement/losses/triplet.py:49` | `d2.clamp(min=0).sqrt()` has an infinite derivative at 0, and the diagonal is always 0. The diagonal is not selected by the loss, so its upstream gradient is 0, and 0 × ∞ = NaN. | Gradient with respect to the input is NaN for `mining="hard"`, `"semi-hard"` and `"easy"` with the default Euclidean distance. The loss value is finite, so nothing warns. The existing test only checks that a gradient exists. |
| 4 | B | `improvement/losses/nt_xent.py:52-56` | A leftover `torch.where` (its result is unused) builds tensors of mismatched length whenever a chunk does not end exactly at row N. | N = 50: `chunk_size` 30, 64 and 99 raise `RuntimeError: The size of tensor a ... must match`. Only `chunk_size=50` works. |
| 5 | A | `improvement/trainer.py:161-162, 242` | `StepLR(step_size=epochs // 3)` is stepped once per batch, not per epoch. | 30 epochs, 16 batches per epoch, lr 1e-3: final lr is 1e-51 instead of 1e-6. Training effectively stops after the first few epochs. |
| 6 | A | `analysis/geometry.py:84-90, 107` and `report.py:225-236` | Above `d_max` = 2,000 dimensions only the top 512 eigenvalues are computed. The participation ratio is then capped at 512, while the report compares it with 0.1·d and 0.3·d. `isotropy_score` divides by 512 instead of d. | Isotropic Gaussian, n = 3,000. At d = 1,000: PR/d = 0.75, score 0.85. At d = 2,500: PR/d = 0.196, which triggers "moderate anisotropy", while the score is 0.98. The two outputs contradict each other. |
| 7 | A | `analysis/kernel.py:89, 130-135` | Kernel alignment is not centred, so it mostly measures class balance. A 1-D continuous target is compared by float equality. | Labels drawn independently of X: alignment 0.70 with 2 classes and 0.32 with 10. A centred version would give about 0. |
| 8 | B | `analysis/kernel.py:55, 86`, `improvement/trainer.py:99` | Labels must be numeric arrays. | String labels raise `ValueError` in `KernelDiagnostics` and `TypeError` in `Trainer.fit`. A Python list of labels raises `TypeError` once n exceeds the subsample size. |
| 9 | A | `improvement/augmentation/__init__.py:41-43` with `knn.py:48` | In a sequential `CompositeAugmentation`, `KNNPairs` discards the incoming second view and returns a raw dataset row. Any augmentation placed before it has no effect on view j. | `[GaussianNoise(1.0), KNNPairs]`: view j is an exact dataset row, view i is noisy. The result depends on the order of the list, with no warning. |
| 10 | C | `analysis/geometry.py:62-63`, `report.py:239-246` | `concentration_ratio` is the global minimum pairwise distance over the global maximum. One duplicate pair sets it to 0. | iid Gaussians give 0.14 (d = 16), 0.64 (d = 256), 0.86 (d = 2,048). After copying one row onto another: 0.0. |
| 11 | C | `improvement/trainer.py:56` | Device selection ignores Apple MPS. | On this machine MPS is available and the trainer selects CPU. |
| 12 | B | environment | In the `pytorch` conda environment FAISS aborts with `OMP: Error #15` (two copies of libomp) or segfaults inside `index.search`. FAISS is a mandatory dependency and the default backend. The project's `.venv` cannot import SciPy. | `pytest tests` passes 4 tests and then segfaults in `faiss ... search`. With FAISS disabled, 96 pass. All measurements here used the scikit-learn backend, forced with `sys.modules["faiss"] = None`. |
| 13 | A | `api.py:103` with `improvement/trainer.py:61-63` | `random_state` does not make a run reproducible. The refiner's weights are drawn when the model is built, before `Trainer.fit` seeds torch. | Four identical `EmbedKit(random_state=0)` fits stopped at epochs 60, 70, 55 and 90. Seeding torch before building the model made two runs identical. |
| 14 | A | `analysis/intrinsic_dim.py` | Intrinsic-dimension estimates are unstable on data with duplicate points, and the failure changes the auto-selected dimension. | On scikit-learn digits, TwoNN divided by zero and returned 8.21 in one run and NaN in two others with the same `random_state`; MLE returned 0.0 once. `target_dim` came out as 9 or 11. |

### 2.2 Found by reading (not executed)

| # | Sev | Where | Defect |
|---|---|---|---|
| 15 | A | `utils/neighbors.py:72-77` | For `metric="cosine"`, `np.ascontiguousarray` returns the caller's own array when it is already contiguous float32, and `faiss.normalize_L2` then normalises it in place. The user's data is modified. The returned "distances" are inner products, sorted in descending order. |
| 16 | A | `utils/neighbors.py:75, 82` | `IndexFlatL2` returns squared distances. The scikit-learn fallback returns distances. `GaussianNoise(adaptive=True)` (`noise.py:27`) therefore scales noise differently depending on the backend. |
| 17 | A | `utils/neighbors.py:67, 83` | Dropping the first column assumes it is the query point. With duplicate rows the query can appear later, so a point becomes its own neighbour. |
| 18 | A | `improvement/trainer.py:178` | Early stopping always minimises the monitor. With `monitor="isotropy"` (higher is better) it stops on improvement. |
| 19 | A | `improvement/trainer.py:176-185` | The best weights are not saved or restored. After patience runs out, the model is the one from `patience × eval_every` epochs after the best point (50 epochs with the `EmbedKit` defaults). |
| 20 | A | `improvement/losses/nt_xent.py:61` | The chunked path averages per-chunk means, which overweights a short last chunk. Version 0.1.2 fixed this in `SupConLoss` and `RankNContrastLoss` but not here. It is hidden behind defect 4. |
| 21 | A | `improvement/losses/rnc.py` | Ties in label distance. See `docs/BUG-rnc-ties.md`, which has a reproduction and a proposed fix. |
| 22 | C | `improvement/trainer.py:49, 227-232` | `warmup_epochs` is stored and never used. `optimizer="lars"` is SGD with a 10× learning rate, not LARS. |
| 23 | C | `improvement/trainer.py:216` | The early-stopping hubness uses `HubnessAnalyzer()` defaults (k = 10, Euclidean) on the validation split. The refiner's output is L2-normalised and trained with cosine similarity. Any exception in evaluation is swallowed (`:217`), which silently disables early stopping. |
| 24 | C | `improvement/augmentation/knn.py:15, 55` | `hard_negatives` is accepted and unused. The in-batch path computes k neighbours and always takes the first, so the positive is deterministic. Random choices use the global `np.random` state, which `Trainer.fit` reseeds (`trainer.py:63`), a side effect on the caller's process. |
| 25 | C | `improvement/augmentation/cutout.py:19-21` | `FeatureMasking` draws one mask for the whole batch, and masks contiguous blocks. Embedding dimensions have no spatial order, so blocks have no meaning. |
| 26 | C | `improvement/augmentation/noise.py:13` | `GaussianNoise(std=0.05)` is an absolute scale. For unit-norm 768-d embeddings the per-coordinate standard deviation is 0.036, so the noise vector is 1.4 times as long as the signal (measured). |
| 27 | C | `improvement/losses/combined.py:24-27` | `except (TypeError, ValueError)` retries a loss without labels. A genuine shape or dtype error in a supervised loss becomes a silent switch to the unsupervised form. |
| 28 | C | `analysis/report.py:143-154` | `EmbedKitAnalyzer(metric=...)` reaches only hubness and neighbour consistency. Distance concentration, uniformity, the kernel and intrinsic dimension stay Euclidean. |
| 29 | C | `analysis/kernel.py:67-77` | Effective rank and condition number are computed from the top 50 eigenvalues of an uncentred kernel. The effective rank is capped at 50, the "condition number" is λ₁/λ₅₀, and the spectral gap mostly reflects the constant direction. |
| 30 | C | docs | `CLAUDE.md` says the project is in the design phase and that hubness wraps `scikit-hubness`. Neither is true: the code exists and `scikit-hubness` is not a dependency. |

---

## 3. Design and methodology critique

### 3.1 The library optimises and monitors proxies, not tasks

- The trainer can monitor `uniformity`, `isotropy` and `k_skewness`. None is a downstream measure. There is no kNN
  accuracy, linear probe, retrieval recall or neighbourhood-preservation metric anywhere in the package.
- The literature does not support treating these proxies as targets:
  - Isotropy calibration gives no consistent gain across models and tasks (Ding et al. 2022).
  - Whitening helps sentence similarity and hurts classification, sometimes by a large margin (Forooghi et al.
    2024†).
  - Strict isotropy is at odds with cluster structure (Mickus et al. 2024†), and lowering isotropy helped in some
    fine-tuning experiments (Rudman and Eickhoff 2024).
  - Uniformity trades off against tolerance of semantically similar points (Wang and Liu 2021).
- The report's wording goes further than the evidence. "Poor uniformity: embeddings are clustered rather than
  spread" (`report.py:258-263`) describes what a good class-structured embedding looks like.
- The repository's own study shows the problem: uniformity improved monotonically for 200 epochs while accuracy
  peaked near epoch 80 (report, section 4).

### 3.2 Thresholds are constants, the statistics are not

- k-skewness for iid Gaussian data in 64 dimensions at k = 10 (measured, 8 seeds each):

  | n | mean k-skewness | sd across seeds |
  |---|---|---|
  | 100 | 1.72 | 0.43 |
  | 368 | 3.00 | 0.71 |
  | 1,000 | 4.48 | 0.77 |
  | 5,000 | 5.96 | 0.87 |

  The same distribution moves from "no issue" to "high hubness (> 5)" as n grows. The analyzer also sets
  k = √n / 2 (`report.py:133`), which changes the statistic again.
- The early-stopping monitor is k-skewness on a 10% validation split. For the Pets study that is 368 points, where
  the seed-to-seed standard deviation is about 0.7. A monitor this noisy will trigger patience by chance.
- `severity` is a sum of threshold crossings. It has no reference distribution and no link to any outcome.
- **Recommendation:** report each statistic against a null model with the same n, d and k (for example a Gaussian
  with the data's covariance spectrum, or column-permuted data), as a z-score or percentile.

### 3.3 Intrinsic dimension and the target dimension

- The consensus is a median over estimators that measure different things. In the Pets study TwoNN gave 16.2,
  MLE 12.9, MOM 9.4 and lPCA 38.0. The `uncertainty` field is the spread across methods, not a sampling
  uncertainty.
- Intrinsic dimension depends on scale and is inflated by noise at short scales (Denti et al. 2022). Estimating on
  a 5,000-point subsample (`intrinsic_dim.py:41-43`) changes the scale probed, so the estimate depends on n.
- `suggested_target_dim = 1.5 × ID` has no stated justification. The manifold dimension is a lower bound for a
  faithful embedding, not a recommendation, and a contrastive objective on the sphere may need more.
- The study in the repository found accuracy flat between 22 and 128 dimensions. That argues for removing the
  choice: train once with nested prefix losses (Matryoshka; Kusupati et al. 2022) and let the user truncate.

### 3.4 What the self-supervised refiner can and cannot learn

- With `KNNPairs`, positives are the input space's own nearest neighbours. The refiner is then a parametric
  neighbour embedding: it is trained to reproduce the input kNN graph in fewer dimensions. This is the same family
  as parametric UMAP (Sainburg et al. 2021) and contrastive neighbour embeddings (Damrich et al. 2023). It cannot
  add information that the input geometry lacks, so "no better than raw kNN" is the expected result, not a tuning
  failure.
- The positives inherit the defects that the analysis module measures. Hubs are over-represented among neighbours,
  so they are over-represented among positives. The diagnostics are not used to correct this (for instance with
  mutual-kNN or hubness-reduced neighbours).
- Other neighbours of the anchor, and same-class points, are used as negatives. At τ = 0.07 the loss concentrates
  on the hardest negatives (Wang and Liu 2021), which here are mostly false negatives.
- In supervised mode the default adds `AlignUniformLoss` to `SupConLoss` (`api.py:183-191`). The uniformity term
  pushes same-class points apart while SupCon pulls them together. SupCon alone tends to collapse each class to a
  point, which removes within-class structure (Graf et al. 2021; Chen et al. 2022).
- Feature-wise augmentation is the mechanism identified as a cause of dimensional collapse in contrastive learning
  (Jing et al. 2022). The effective rank of the output is not monitored during training.

### 3.5 Architecture and training

- The refiner is a plain MLP from random initialisation with no skip connection (`model.py`). It must relearn the
  input structure before it can improve on it. Published adapters for frozen embeddings use a residual form
  (Search-Adaptor, Yoon et al. 2024; Matryoshka-Adaptor, Yoon et al. 2024†).
- Inputs are not standardised. Embedding scales differ by orders of magnitude between models, and fixed-scale
  augmentations depend on that scale (defect 26).
- The output is always L2-normalised, which discards the norm. The norm carries information in some models.
- The analysis in `EmbedKit.fit` runs on all of X before the validation split is taken (`api.py:77-81`), so the
  held-out rows influence the configuration. Labels passed to the analyzer do too.
- kNN positives and every fitted statistic are transductive. `transform` on new data works, but nothing in the
  library evaluates out-of-sample behaviour.

### 3.6 Testing

- 96 test cases exist (they pass with FAISS disabled), and almost all check shapes, types or "a gradient exists". Three of the four loss bugs found so
  far (SupCon NaN, RnC ties, triplet NaN) passed that kind of test.
- No test compares a loss with an independent reference, checks invariance to batch order, or checks a diagnostic
  on data with a known answer.

---

### 3.7 Pilot comparison with training-free baselines (measured)

A small pilot was run for this review. It is three datasets and three seeds, so it indicates a direction and does
not settle the question. Experiment E1 in section 5 is the full version.

- **Protocol:** every method was fitted on the training split and applied to the test split. kNN accuracy uses
  k = 5 with neighbours from the training split. The linear probe is logistic regression on standardised inputs.
  Trustworthiness is against the raw input at k = 10. `EmbedKit` rows are mean ± sd over `random_state` 0, 1, 2.
  Because of defect 13, that spread also contains uncontrolled initialisation noise.
- **Datasets:**
  - Pets: the cached ResNet50 features in `examples/data/pets_resnet50.npz` (2,048-d, 3,680 train / 3,669 test,
    37 classes).
  - Digits: scikit-learn `load_digits` (64-d raw pixels, not a learned embedding; 898 / 899).
  - Synthetic: 10 Gaussian classes on a 16-d latent, passed through a fixed random tanh MLP to 512-d, plus a large
    shared mean vector and 3 high-variance nuisance directions that are independent of the class (3,000 / 3,000).
    This set was built to resemble common-direction anisotropy, so it favours all-but-the-top with 3 components.

| Method | Pets kNN | Pets probe | Digits kNN | Digits probe | Synthetic kNN | Synthetic probe |
|---|---|---|---|---|---|---|
| Raw | 0.898 | 0.916 | **0.984** | 0.963 | 0.744 | 0.933 |
| Standardise | 0.829 | 0.916 | 0.963 | 0.963 | 0.789 | 0.933 |
| PCA, best d | **0.907** (d = 64) | 0.914 | **0.984** (d = 64) | 0.934 | 0.748 (d = 64) | 0.930 |
| PCA-whitening, best d | 0.893 (d = 32) | 0.910 | 0.971 (d = 16) | 0.944 | 0.879 (d = 32) | 0.936 |
| All-but-the-top, best D | 0.897 (D = 1) | 0.915 | 0.974 (D = 1) | 0.928 | **0.919** (D = 3) | 0.931 |
| `EmbedKit` default | 0.893 ± 0.003 | 0.906 ± 0.004 | 0.966 ± 0.006 | 0.931 ± 0.004 | 0.733 ± 0.008 | 0.869 ± 0.007 |
| `EmbedKit`, d = 64, 80 epochs, no early stopping | 0.898 ± 0.004 | 0.905 ± 0.002 | 0.976 ± 0.003 | 0.966 ± 0.002 | 0.764 ± 0.004 | 0.900 ± 0.006 |
| `EmbedKit` supervised (uses labels) | 0.918 ± 0.002 | 0.920 ± 0.003 | 0.977 ± 0.004 | 0.976 ± 0.003 | 0.928 ± 0.002 | 0.933 ± 0.003 |

kNN is Euclidean, and each probe value is for the same d or D as the kNN value beside it. "Best d" and "best D"
were picked by test kNN accuracy, which flatters those rows slightly; the
gaps on the synthetic set are far larger than that effect, and those on Pets and digits are not.

What the numbers show:

- **The default self-supervised refiner did not beat raw embeddings on kNN accuracy on any of the three datasets,**
  and it lowered linear-probe accuracy on all three (by 1.0, 3.2 and 6.4 points).
- **Fixing d = 64 and disabling early stopping was better than the default everywhere.** It matched raw on Pets,
  and beat raw kNN by 2 points on the synthetic set, where standardisation alone gave 4.5 points and
  all-but-the-top gave 17.5.
- **The refiner did what its objective asks.** It produced the lowest k-skewness on every dataset (0.25 to 0.55,
  against 0.6 to 1.9 for raw) and kept trustworthiness at 0.987 or above. Lower hubness did not come with higher
  accuracy.
- **Supervised mode is where the library adds value.** It was the best method on Pets (+1.1 points kNN over the
  best baseline) and on the synthetic set. On digits it was below raw kNN, and on the synthetic set the label-free
  all-but-the-top came within 0.9 points of it.
- **Cost:** the baselines fit in under 0.3 s. `EmbedKit` took 15 to 35 s on Pets and the synthetic set.

Does the early-stopping monitor track accuracy? On the synthetic set, the default configuration was trained for
100 epochs without stopping, and held-out k-skewness and uniformity were logged next to test kNN accuracy every
5 epochs:

| Seed | Spearman ρ (k-skewness, accuracy) | Spearman ρ (uniformity, accuracy) | Epoch of lowest k-skewness | Epoch of best accuracy |
|---|---|---|---|---|
| 0 | +0.41 | −0.86 | 10 | 80 |
| 1 | −0.37 | −0.84 | 50 | 95 |
| 2 | +0.77 | −0.95 | 5 | 95 |

- k-skewness is minimised, so it would need a negative correlation to be a useful monitor. The sign changes
  between seeds.
- Uniformity correlates with accuracy here, but both simply move with the epoch count, so it carries no stopping
  information. This matches the Pets study in the repository.
- In all three default runs on Pets, the monitor was at its lowest at the first evaluation (epoch 5), so training
  stopped at epoch 55, the earliest point the patience setting allows. With no restoration of the best weights
  (defect 19), early stopping in this configuration acts as a fixed 55-epoch budget.

---

## 4. Suggested improvements

Effort: S is under a day, M is a few days, L is a week or more.

### 4.1 Fix what is broken (all S unless noted)

1. Replace the kNN cache. Either remove it, or key it on a content hash and hold a reference to the array. Make the
   backends agree: return true distances, never modify the input, and remove the query by index, not by position.
2. `TripletLoss`: compute distances with an epsilon inside the square root, or work with squared distances.
   Vectorise the per-anchor Python loop.
3. `NTXentLoss`: delete the dead `torch.where`, and average over rows.
4. `RankNContrastLoss`: apply the fix in `docs/BUG-rnc-ties.md`.
5. `Trainer`: seed before the model is built (or re-initialise the model inside `fit`), step epoch-based schedulers per epoch, add a `mode` (min or max) per monitor, restore the best
   weights, implement or remove `warmup_epochs` and `lars`, select MPS when available, stop swallowing evaluation
   errors, and encode labels (strings, lists) once at entry.
6. `CompositeAugmentation`: define the contract so that pair-producing augmentations (`KNNPairs`, mixup) run first
   and view-wise ones are applied to both views afterwards. Raise on an invalid order.
7. Isotropy: compute the participation ratio from traces (`tr(C)² / tr(C²)` needs no eigendecomposition, and
   `tr(C²)` is the squared Frobenius norm of the Gram matrix), so the value is exact at any d. Normalise by
   min(n − 1, d).
8. Kernel: centre both kernels (CKA; Cortes et al. 2012, Kornblith et al. 2019). Use a label kernel suited to the
   target type, with an RBF kernel on y for regression.
9. Make FAISS optional (M). The default should work on a stock macOS install. Add a CI job that runs the tests.
10. Update `CLAUDE.md` and the README to match the code.

### 4.2 Build the evaluation layer first (M)

A new `embedkit/evaluation/` module, used by the trainer, the examples and the benchmarks:

- **Label-free:** trustworthiness and continuity (Venna and Kaski 2001†), kNN-set overlap with the input (LCMC;
  Chen and Buja 2009†), CKA against the input, effective rank / RankMe (Garrido et al. 2023).
- **Label-aware:** kNN accuracy, linear probe, retrieval recall@k, silhouette, and NC1 (within-class over
  between-class variability; Papyan et al. 2020) to detect class collapse.
- **Protocol helper:** `evaluate(method, X_train, X_test, y_train, y_test, seeds)` that fits on the training split
  only and reports mean and standard deviation over seeds.
- **Early stopping:** allow any of these as the monitor, with a direction. When labels exist, validation kNN
  accuracy is the obvious default. Without labels, a held-out contrastive loss (as in SCARF; Bahri et al. 2022) or
  kNN overlap with the input is better founded than k-skewness.

### 4.3 Training-free baselines as first-class transforms (S)

`embedkit/improvement/baselines.py`, each with `fit` / `transform`:

- centring, standardisation, L2 normalisation
- PCA to d, PCA whitening and ZCA (Su et al. 2021)
- all-but-the-top (Mu et al. 2018), and the reduce-then-clean variant (Raunak et al. 2019)
- conceptor negation (Liu et al. 2019)
- Gaussian random projection, as the floor for dimensionality reduction

These are the comparison every refiner result needs. They may also be the right answer for many users, and the
auto-configuration should be allowed to return one of them.

### 4.4 Hubness reduction (S to M)

The report recommends "hubness-aware distance weighting" and the library offers none.

- Mutual proximity and local scaling (Schnitzer et al. 2012), CSLS (Conneau et al. 2018), DisSimLocal (Hara et al.
  2016), localised centring (Hara et al. 2015).
- A large comparison found that dimensionality reduction does not reduce hubness unless the target dimension falls
  below the intrinsic dimension (Feldbauer and Flexer 2019). That is relevant to how the refiner's effect on
  hubness should be read.
- noHub (Trosten et al. 2023) is the closest published method to this refiner: a learned map that trades local
  similarity preservation against uniformity on the sphere. It is the natural learned baseline.
- Use the reduced neighbour graph inside `KNNPairs` (mutual neighbours, or neighbours under mutual proximity) so
  that the diagnostics change the training signal.

### 4.5 Refiner and objective (S to M each)

In the order I would try them:

1. **Residual adapter.** `z = normalize(W x + g(x))` with `g` initialised to zero and `W` initialised from PCA. At
   step 0 the refiner equals the PCA baseline, so training can only be judged against it.
2. **Input standardisation layer** fitted on the training split, and augmentation scales expressed relative to the
   data (per-dimension standard deviation, or kNN radius).
3. **Matryoshka prefix losses** (Kusupati et al. 2022), which make `suggested_target_dim` a truncation choice made
   after training.
4. **Structure-preservation term**, for example a penalty on the change in kNN similarities, so that the refiner
   cannot buy uniformity by destroying neighbourhoods. This is the local-similarity term in noHub.
5. **Neighbour-aware negatives.** Mask the anchor's k nearest neighbours out of the denominator, or use the
   debiased loss (Chuang et al. 2020). Decoupled contrastive loss (Yeh et al. 2022) is a one-line variant that is
   reported to help at small batch sizes.
6. **Dropout positives** (SimCSE; Gao et al. 2021): two forward passes through the refiner's own dropout, with no
   input augmentation. The refiner already has dropout, so this costs nothing.
7. **Marginal-resampling corruption** (SCARF; Bahri et al. 2022) in place of zero-masking. Apply it in a PCA basis,
   because raw embedding coordinates are arbitrary rotations.
8. **VICReg or Barlow Twins** (Bardes et al. 2022; Zbontar et al. 2021†) as a covariance regulariser inside
   `CombinedLoss`. Contrastive and non-contrastive objectives are closely related under tuning (Garrido et al.
   2023), so one well-tuned option is worth more than five untuned ones.
9. **Supervised mode:** drop the uniformity term by default, add a within-class spread term (Chen et al. 2022),
   and report NC1. For regression, fix RnC first, then compare with ConR (Keramati et al. 2024) and SupReMix
   (Wu et al. 2023).
10. **Pseudo-labels** from clustering with SupCon, as the existing report proposes. Evaluate it against the
    baselines before adopting it.

A caution from metric learning: once tuning budgets were equalised, a decade of reported loss improvements was
marginal (Musgrave et al. 2020). Add objectives only when the harness in 4.2 shows a gain over the baselines.

### 4.6 Diagnostics (S to M)

- **Null-calibrated statistics** (section 3.2), which replace the fixed thresholds.
- **Intrinsic dimension as a function of scale** (GRIDE; Denti et al. 2022), per-point local ID (Amsaleg et al.
  2015), and bootstrap intervals. Report estimators separately. Do not take a median across them.
- **Isotropy done properly:** IsoScore (Rudman et al. 2022), the norm of the mean vector relative to the mean norm,
  and the share of variance in the top few dimensions ("rogue dimensions"; Timkey and van Schijndel 2021†).
  Describe these as properties, not defects.
- **Spectrum:** effective rank on the full spectrum, and the power-law decay exponent α (Agrawal et al. 2022).
- **Distance concentration:** the coefficient of variation of pairwise distances, and the per-query relative
  contrast of Beyer et al., in place of the min/max ratio.
- **Metric consistency:** one `metric` argument that every analyzer honours.
- **Recommendations that name an action in the library**, for example "k-skewness is at the 99th percentile of the
  null: try `MutualProximity` or `AllButTheTop(3)`", each backed by the benchmark in section 5.

### 4.7 Auto-configuration

Either make it real or make it honest.

- Real: choose among candidates (raw, each baseline, a few refiner settings) by a label-free validation criterion
  measured on a held-out split, and return the winner. The candidates include "do nothing".
- Honest: rename the current behaviour to what it is (a default configuration plus an ID-based dimension), and
  remove `suggested_k` and `suggested_sigma` until something consumes them.

---

## 5. Experiments and explorations

Common protocol for all of them:

- Fit every transform on a training split only. Report on a held-out split.
- At least 5 seeds for anything trained. Report mean ± sd, and paired differences against raw and against PCA at
  the same output dimension.
- Choose hyperparameters on a validation split, never on the test split.
- Record wall-clock time. A method that needs minutes must beat one that needs milliseconds.

### E1. Baseline table (do this first)

- **Question:** does the refiner beat raw embeddings and the training-free baselines at equal dimension?
- **Methods:** raw; centre + L2; standardise; PCA and PCA-whitening at d ∈ {16, 32, 64, 128, 256};
  all-but-the-top with 1, 3 and d/100 components; random projection; parametric UMAP; `EmbedKit` default;
  `EmbedKit` with fixed d; supervised `EmbedKit` as the labelled reference.
- **Metrics:** kNN accuracy (Euclidean and cosine), linear probe, recall@10, trustworthiness, continuity,
  k-skewness, effective rank, fit time.
- **Decision:** if no refiner configuration beats PCA or whitening at the same d by more than the seed spread on at
  least half the datasets, the self-supervised mode should be repositioned as compression, not improvement.

### E2. Do the proxies predict downstream quality?

- **Question:** which label-free statistic, if any, tracks downstream accuracy during training and across
  configurations?
- **Design:** log test kNN accuracy and linear-probe accuracy every few epochs next to uniformity, k-skewness,
  isotropy, effective rank, held-out contrastive loss and kNN overlap with the input. Do this over a grid of
  roughly 30 configurations per dataset.
- **Analysis:** Spearman correlation within a run (is it a stopping signal?) and across runs (is it a selection
  signal?). RankMe and LiDAR were proposed for exactly this in image self-supervision (Garrido et al. 2023; Thilak
  et al. 2024) and are untested for frozen embeddings.
- **Decision:** the default monitor becomes whichever statistic has the best within-run correlation, and the
  claim in the existing report that k-skewness is a principled stopping rule is kept or withdrawn.

### E3. Calibration of the diagnostics on data with known answers

- **Intrinsic dimension:** manifolds with known ID from `skdim.datasets` (hyperspheres, hypercubes, Swiss roll,
  nonlinear manifolds), embedded in 64 to 2,048 dimensions by linear and nonlinear maps. Sweep n from 500 to
  50,000 and additive noise from 0 to 20% of the data scale. Report bias and variance for each estimator. The
  reference framework is Campadelli et al. (2015).
- **Hubness, isotropy, concentration:** iid Gaussians with a controlled covariance spectrum (flat, power-law with
  several exponents, a few dominant directions, a large shared mean). Sweep n, d and k. This yields the null
  tables that section 3.2 asks for, and regression tests for defects 6 and 10.
- **Subsampling stability:** every statistic at n ∈ {1k, 2k, 5k, 20k} drawn from the same dataset. A diagnostic
  that moves with n needs a correction or a warning.

### E4. Does a diagnosis predict which fix helps?

- **Question:** is the report useful for choosing an intervention? This tests the library's central premise.
- **Design:** for each dataset, compute the report, then apply every intervention from E1 and section 4.4. Regress
  the gain of each intervention on the diagnostics (k-skewness percentile, ID/d, top-component variance share,
  mean-vector norm).
- **Expected patterns to confirm or refute:** all-but-the-top helps when the mean vector and top components
  dominate; hubness reduction helps when null-calibrated skewness is high; whitening helps similarity tasks and
  hurts classification.
- **Decision:** recommendations that the data support stay in the report. The rest are removed.

### E5. Controlled degradation and recovery

- **Design:** start from a good embedding and damage it in known ways: add a large common vector, inflate a few
  nuisance directions, apply a random anisotropic linear map, add isotropic noise, pad with noise dimensions, mix
  in a second domain with a different mean.
- **Measure:** does the analyzer detect each kind of damage, and which method recovers the original downstream
  accuracy?
- **Value:** unlike real data, the correct answer is known, so both the diagnostics and the refiner can be scored.

### E6. Ablation of the refiner

One factor at a time against the E1 protocol, on three datasets of different modality:

- architecture: plain MLP, residual with PCA initialisation, linear only
- positives: Gaussian noise (relative scale), dropout passes, SCARF corruption, in-batch kNN, global kNN, mutual
  kNN, k ∈ {1, 5, 15, 50}
- negatives: all, neighbour-masked, debiased, decoupled
- temperature: 0.05, 0.1, 0.2, 0.5, and a cool-down schedule
- loss: NT-Xent, NT-Xent + uniformity, VICReg, Barlow Twins, NT-Xent + structure-preservation
- output: fixed d against Matryoshka prefixes, normalised against unnormalised
- input: raw, standardised, whitened
- batch size: 64, 256, 1,024

### E7. Sample size and transduction

- **Learning curves:** n_train ∈ {200, 500, 1k, 5k, 20k, 100k}. Small MLPs on small data are where a learned
  refiner is most likely to lose to PCA.
- **Inductive against transductive:** fit on train and evaluate on test, against fit on train + test (without
  labels) and evaluate on test. Report both, labelled as such. Whitening, all-but-the-top, mutual proximity and
  kNN positives all benefit from seeing the test points.
- **Distribution shift:** fit on one domain, apply to another (for example two text corpora embedded by the same
  model).

### E8. Supervised and regression modes

- **Classification:** SupCon against SupCon + uniformity (current default), SupCon + within-class spread, and a
  linear probe or LDA on raw embeddings as the baseline. Report NC1 and fine-grained transfer (train on coarse
  labels, evaluate on fine labels) to expose class collapse.
- **Label efficiency:** 1, 5, 20 and 100 labels per class, and label noise of 0 to 40%.
- **Regression:** fixed RnC against the current RnC, ConR, SupReMix, and ridge regression on raw embeddings. Use
  targets with heavy ties and without, which is the case `docs/BUG-rnc-ties.md` identifies.

### E9. Scaling and robustness

- Time and peak memory of `EmbedKitAnalyzer` and `EmbedKit.fit` for n up to 1M and d up to 4,096.
- Robustness: exact duplicates, constant columns, NaN or infinite values, n < d, float64 input, very small n.
  Several current code paths assume none of these occur (defects 10 and 17).

### Datasets

| Modality | Source | Labels | Use |
|---|---|---|---|
| Image | CIFAR-100, STL-10, Oxford Pets through a frozen ResNet50, DINOv2 or CLIP encoder | classes | E1, E2, E6, E8 |
| Text | MTEB / BEIR tasks embedded locally with two or three sentence-transformers models (Muennighoff et al. 2023; Thakur et al. 2021) | classes, STS scores, retrieval judgements | E1, E2, E4, E7 |
| Text, large | Cohere Wikipedia embeddings; DBpedia entities with OpenAI embeddings (Hugging Face) | none | E3 stability, E9 |
| Vectors | ANN-Benchmarks: GloVe-100, SIFT-128, GIST-960 (Aumüller et al. 2019) | none | E3, E4 (hubness), E9 |
| Tabular | OpenML-CC18 | classes | E6 (corruption augmentations) |
| Molecular, protein | MoleculeNet with a ChemBERTa-type encoder; ESM-2 on a fluorescence task (to be computed locally; not verified as ready-made sets) | regression | E8 |
| Histopathology | the survival-analysis embeddings already used with this library in a downstream project | survival | E8 |
| Synthetic | `skdim.datasets`, Gaussian families from E3, damaged embeddings from E5 | known | E3, E5 |

---

## 6. Recommended order of work

1. **Fix the defects in 2.1 and 2.2 that are rated A**, each with a regression test that compares against a
   reference or a known answer. Make the test suite run on a clean macOS install.
2. **Build the evaluation module (4.2) and the baselines (4.3).**
3. **Run E1 and E2** on three datasets. This decides what the library should claim.
4. **Run E3** and replace fixed thresholds with null-calibrated statistics.
5. **Then, and only if E1 leaves room:** residual adapter, Matryoshka prefixes, neighbour-aware negatives and
   hubness-reduced positives, each judged by E6.
6. **Run E4** to decide which recommendations the report keeps.

---

## References

Post-processing and isotropy
- Mu, Bhat, Viswanath. All-but-the-Top. ICLR 2018. arXiv:1702.01417
- Raunak, Gupta, Metze. Effective Dimensionality Reduction for Word Embeddings. RepL4NLP 2019
- Su, Cao, Liu, Ou. Whitening Sentence Representations. arXiv:2103.15316
- Huang et al. WhiteningBERT. Findings of EMNLP 2021. arXiv:2104.01767
- Liu, Ungar, Sedoc. Conceptor Negation. AAAI 2019. arXiv:1811.11001
- Ding et al. On Isotropy Calibration of Transformers. Insights workshop 2022. arXiv:2109.13304
- Forooghi, Sadeghi, Lu. Whitening Not Recommended for Classification Tasks in LLMs. arXiv:2407.12886 †
- Cai, Huang, Bian, Church. Isotropy in the Contextual Embedding Space. ICLR 2021
- Mickus et al. Isotropy, Clusters, and Classifiers. arXiv:2402.03191 †
- Timkey, van Schijndel. All Bark and No Bite. arXiv:2109.04404 †
- Rudman, Gillman, Rayne, Eickhoff. IsoScore. Findings of ACL 2022. arXiv:2108.07344
- Rudman, Eickhoff. Stable Anisotropic Regularization. ICLR 2024. arXiv:2305.19358

Hubness
- Radovanović, Nanopoulos, Ivanović. Hubs in Space. JMLR 2010
- Schnitzer, Flexer, Schedl, Widmer. Local and Global Scaling Reduce Hubs in Space. JMLR 2012
- Conneau et al. Word Translation Without Parallel Data. ICLR 2018. arXiv:1710.04087
- Hara et al. Localized Centering. AAAI 2015
- Hara et al. Flattening the Density Gradient (DisSimLocal). AAAI 2016
- Feldbauer, Flexer. A comprehensive empirical comparison of hubness reduction. KAIS 2019
- Bogolin et al. Querybank Normalisation. CVPR 2022. arXiv:2112.12777
- Trosten et al. Hubs and Hyperspheres (noHub). CVPR 2023. arXiv:2303.09352

Self-supervised objectives and representation quality
- Wang, Isola. Alignment and Uniformity. ICML 2020. arXiv:2005.10242
- Wang, Liu. Understanding the Behaviour of Contrastive Loss. CVPR 2021. arXiv:2012.09740
- Zbontar et al. Barlow Twins. arXiv:2103.03230 †
- Bardes, Ponce, LeCun. VICReg. ICLR 2022. arXiv:2105.04906
- Garrido et al. On the Duality Between Contrastive and Non-contrastive SSL. ICLR 2023. arXiv:2206.02574
- Jing, Vincent, LeCun, Tian. Understanding Dimensional Collapse. ICLR 2022. arXiv:2110.09348
- Garrido et al. RankMe. ICML 2023. arXiv:2210.02885
- Agrawal et al. α-ReQ. NeurIPS 2022
- Thilak et al. LiDAR. ICLR 2024. arXiv:2312.04000
- Chuang et al. Debiased Contrastive Learning. NeurIPS 2020. arXiv:2007.00224
- Robinson et al. Contrastive Learning with Hard Negative Samples. ICLR 2021. arXiv:2010.04592
- Yeh et al. Decoupled Contrastive Learning. ECCV 2022. arXiv:2110.06848
- Kusupati et al. Matryoshka Representation Learning. NeurIPS 2022. arXiv:2205.13147
- Gao, Yao, Chen. SimCSE. EMNLP 2021. arXiv:2104.08821
- Damrich, Böhm, Hamprecht, Kobak. From t-SNE to UMAP with contrastive learning. ICLR 2023. arXiv:2206.01816
- Sainburg, McInnes, Gentner. Parametric UMAP. Neural Computation 2021. arXiv:2009.12981
- Bahri et al. SCARF. ICLR 2022. arXiv:2106.15147

Supervised, regression and adapters
- Khosla et al. Supervised Contrastive Learning. NeurIPS 2020. arXiv:2004.11362
- Graf et al. Dissecting Supervised Contrastive Learning. ICML 2021. arXiv:2102.08817
- Chen et al. Perfectly Balanced. ICML 2022. arXiv:2204.07596
- Papyan, Han, Donoho. Prevalence of Neural Collapse. PNAS 2020. arXiv:2008.08186
- Musgrave, Belongie, Lim. A Metric Learning Reality Check. ECCV 2020. arXiv:2003.08505
- Zha et al. Rank-N-Contrast. NeurIPS 2023. arXiv:2210.01189
- Keramati, Meng, Evans. ConR. ICLR 2024. arXiv:2309.06651
- Wu et al. SupReMix. arXiv:2309.16633
- Yoon, Chen, Arık, Pfister. Search-Adaptor. ACL 2024. arXiv:2310.08750
- Yoon, Sinha, Arık, Pfister. Matryoshka-Adaptor. arXiv:2407.20243 †

Diagnostics and evaluation
- Facco et al. TwoNN. Scientific Reports 2017. arXiv:1803.06992
- Denti, Doimo, Laio, Mira. GRIDE. Scientific Reports 2022
- Amsaleg et al. Estimating Local Intrinsic Dimensionality. KDD 2015
- Campadelli et al. Intrinsic Dimension Estimation: Relevant Techniques and a Benchmark Framework. Math. Probl. Eng. 2015
- Venna, Kaski. Neighborhood preservation in nonlinear projection methods. ICANN 2001 †
- Chen, Buja. Local Multidimensional Scaling (LCMC). JASA 2009 †
- Kornblith et al. Similarity of Neural Network Representations Revisited (CKA). ICML 2019. arXiv:1905.00414
- Cortes, Mohri, Rostamizadeh. Algorithms for Learning Kernels Based on Centered Alignment. JMLR 2012
- Muennighoff et al. MTEB. EACL 2023. arXiv:2210.07316
- Thakur et al. BEIR. NeurIPS 2021 Datasets and Benchmarks. arXiv:2104.08663
- Aumüller, Bernhardsson, Faithfull. ANN-Benchmarks. Information Systems 2019. arXiv:1807.05614

---

## Appendix: how the measurements were made

- Environment: conda env `pytorch` (Python 3.12, torch 2.10, faiss 1.13.2), macOS, CPU.
- FAISS could not be used in this environment (defect 12). The scripts force the scikit-learn kNN backend. Defects
  15 and 16 concern the FAISS path and are therefore listed as "by reading".
- Scripts for the defect checks: `benchmarks/review_2026_11/verify.py`, `verify2.py` and `verify_faiss.py` (the
  last one needs a working FAISS and could not be run here).
- Pilot experiment (section 3.7): scripts, logs and full per-run tables are in `benchmarks/review_2026_11/pilot/`.
  Open `results_tables.md` there first. The kNN cache was disabled for these runs (`_MAX_CACHE = 0`).
- Run them from the repository root with `conda run -n pytorch python <script>`.
- The phased plan that follows from this review is in `docs/IMPLEMENTATION_PLAN.md`.
