# Benchmarks E1 and E2: does the self-supervised refiner help, and can a label-free statistic tell?

Phase 3 of `docs/IMPLEMENTATION_PLAN.md`, following experiments E1 and E2 of `docs/CRITICAL_REVIEW.md`
(section 5). Run on 2026-11-11 on the OCCIDATA cluster.

## Outcome

- **E1. The self-supervised refiner did not beat a training-free baseline on any of the seven datasets.**
  The refiner configuration chosen on validation was compared with the best training-free baseline at
  the same output dimension. It tied on three datasets (Pets, Banking77, CIFAR-100) and lost on three
  (digits, STL-10, synthetic). On STS-B it scored 0.036 lower, with no interval available.
  - On five of the seven datasets, PCA or PCA-whitening at half the dimension or less matched or beat
    the refiner.
  - Its trustworthiness was below PCA's at the same dimension on six of the seven.
  - So it is neither an improvement nor a better compressor.
- **E1, supervised mode.** The supervised refiner is the strongest method on CIFAR-100, Pets and the
  synthetic set, with intervals above 0. It is unaffected by the gate.
- **E2. No label-free statistic tracks test accuracy over the epochs of a run.** No monitor has a median
  within-run Spearman correlation above 0.5 on more than one of four datasets.
  - The held-out contrastive loss is anti-correlated with test accuracy on all four: accuracy falls
    while the loss falls.
  - The cost of not stopping is small. Training to the last epoch loses 0.005 to 0.012 test kNN
    accuracy, on average, against each run's best epoch on the real datasets.
- **E2, choosing between configurations.** `knn_overlap` (kNN overlap with the input) is a good
  label-free signal on the three real datasets: across-run ρ is 0.72 to 0.79, and its first choice is
  at most 0.002 below the best. It fails on the synthetic set (ρ = −0.03).
  - Uniformity, the trainer's default monitor, ranks the Pets configurations backwards (ρ = −0.60).
  - The k-skewness stopping rule claimed in `docs/UNSUPERVISED_REFINEMENT_REPORT.md` is not supported:
    median within-run ρ is between −0.03 and +0.50.
- **Gate decision (confirmed by the user, 2026-11-11): outcome C, with WP 5.3 added.**
  - Phase 5 consists of:
    - WP 5.1 (residual adapter initialised at PCA);
    - one two-day attempt at WP 5.2 (better positives and negatives);
    - WP 5.3 (Matryoshka output), kept because nested prefixes are useful downstream.
  - If WP 5.2 does not produce wins, `EmbedKit(mode="self_supervised")` returns the best training-free
    baseline.
  - Early stopping is now off by default: `EmbedKit` trains for a fixed epoch budget (see the changelog).

## Datasets

| Name | Embedding | Classes | Train / test | d |
|---|---|---|---|---|
| `pets_resnet50` | Oxford-IIIT Pets, torchvision ResNet50 (ImageNet) | 37 | 3,680 / 3,669 | 2,048 |
| `cifar100_dinov2` | CIFAR-100, DINOv2-base CLS token | 100 | 50,000 / 10,000 | 768 |
| `stl10_clip` | STL-10, CLIP ViT-B/32 image features | 10 | 5,000 / 8,000 | 512 |
| `banking77_mpnet` | Banking77, all-mpnet-base-v2 mean pooling, unnormalised | 77 | 9,993 / 3,076 | 768 |
| `stsb_mpnet` | STS-B sentences, all-mpnet-base-v2, unnormalised | (pairs) | 1,500 val / 1,379 test pairs | 768 |
| `digits` | scikit-learn 8×8 digits, raw pixels (not a learned embedding) | 10 | 898 / 899 | 64 |
| `synthetic_anisotropic` | 16-d latent classes through a random tanh network, plus a large shared mean and nuisance directions | 10 | 3,000 / 3,000 | 512 |

The cluster extracted the features. Model and dataset revisions are recorded in each result file
(`dataset_meta`).

## Protocol

- **Selection rule.** Every method is fitted twice:
  - on the training split minus a stratified 20% validation part, and scored on that part;
  - on the whole training split, and scored on the test split.
  - Every hyperparameter is chosen on the validation score of the primary metric, never on test:
    output dimension, number of removed components, and which fixed d for `EmbedKit`. The pilot in
    `docs/UNSUPERVISED_REFINEMENT_REPORT.md` chose on test accuracy.
  - The primary metric is kNN accuracy (cosine, k = 5) for labelled sets and Spearman correlation for
    STS-B.
- **Seeds.** Five for stochastic methods (`EmbedKit`, random projection); one for deterministic ones.
- **Intervals.** Paired bootstrap over test points (2,000 resamples, 95%) on per-point outcomes,
  averaged over seeds (`embedkit.evaluation.compare`). STS correlation has no per-point outcome, so
  it has no interval.
- **Gate.** A dataset is a "win" when the refiner minus the baseline is above 0 and its interval
  excludes 0. The baseline is the best on validation among PCA, PCA-whitening, random projection and
  all-but-the-top-then-reduce at the same d.
- **E1 methods.**
  - Raw; centre + L2; standardise; ZCA; conceptor negation.
  - All-but-the-top with 1, 3 and d/100 components.
  - PCA, PCA-whitening, random projection, and all-but-the-top-then-reduce, each at d ∈ {16, 32, 64,
    128, 256}.
  - `EmbedKit()` with the 0.2.0 defaults (auto d, early stopping on k-skewness with patience 10).
  - `EmbedKit(target_dim=d)` for d ∈ {32, 64, 128}, and supervised `EmbedKit` as the labelled
    reference.
  - Parametric UMAP was not run: its parametric mode needs TensorFlow, which the cluster container
    does not have.
- **E2 design.**
  - Four classification sets, with 30 refiner configurations each:
    - `c00` is the default recipe (NT-Xent at τ = 0.07 plus 0.5 × align-uniform, kNN-5 pairs,
      d = 64, lr 3e-4);
    - the other 29 are a fixed random draw from loss (NT-Xent + AU, NT-Xent, align-uniform),
      temperature (0.07, 0.2, 0.5), augmentation (kNN-5 or kNN-20 pairs, Gaussian noise, feature
      dropout, mixup), d (32, 64, 128) and lr (1e-4, 3e-4, 1e-3).
  - Training: 100 epochs, no early stopping, seed 0.
  - Every 5 epochs, two kinds of value are logged:
    - every trainer monitor, computed on the trainer's 10% held-out split as during training:
      uniformity, isotropy, k-skewness, kNN overlap, effective rank (RankMe), held-out loss and
      training loss, plus labelled validation kNN accuracy as a reference;
    - test kNN accuracy and a logistic-regression probe (C = 1), on at most 10,000 training and
      5,000 test rows (stratified subsample; only CIFAR-100 and Banking77 are subsampled).
  - Correlations are Spearman, with each monitor signed so that positive means "better monitor, higher
    accuracy".

## E1 results

Test value of the primary metric. Each family's configuration is selected on validation; ± is the
standard deviation over seeds. Full tables: `benchmarks/results/e1/REPORT.md`. These two tables
come from `benchmarks/results/e1/GATE.md`.

| Dataset | Raw | Best training-free | `EmbedKit` default (auto d) | `EmbedKit` fixed d | `EmbedKit` supervised |
|---|---|---|---|---|---|
| Pets | 0.901 | 0.901 (raw) | 0.894 ± 0.003 (d 22) | 0.900 ± 0.003 (d 64) | **0.915 ± 0.004** (d 22) |
| CIFAR-100 | 0.879 | 0.882 (ABTT-3) | 0.858 ± 0.003 (d 28) | 0.863 ± 0.001 (d 64) | **0.888 ± 0.008** (d 28) |
| STL-10 | 0.983 | 0.976 (ABTT-5) | 0.978 ± 0.001 (d 24) | 0.979 ± 0.001 (d 128) | 0.985 ± 0.001 (d 24) |
| Banking77 | **0.931** | 0.928 (conceptor) | 0.899 ± 0.007 (d 12) | 0.917 ± 0.003 (d 64) | 0.930 ± 0.004 (d 12) |
| STS-B (Spearman) | 0.834 | **0.839** (centre + L2) | 0.781 ± 0.008 (d 26) | 0.797 ± 0.004 (d 128) | n/a |
| digits | **0.981** | 0.972 (centre + L2) | 0.946 ± 0.023 (d 11) | 0.966 ± 0.011 (d 32) | 0.973 ± 0.016 (d 11) |
| synthetic | 0.746 | 0.913 (ABTT-3) | 0.720 ± 0.010 (d 15) | 0.744 ± 0.010 (d 128) | **0.921 ± 0.004** (d 15) |

The best training-free method is chosen on validation, so on Pets it is raw itself, and on STL-10,
Banking77 and digits its test value is below raw.

### The gate

| Dataset | `EmbedKit` (val-selected) | Baseline at the same d | Difference [95% CI] | Verdict | Best baseline at a lower d | Trustworthiness, `EmbedKit` vs PCA |
|---|---|---|---|---|---|---|
| Pets | d64: 0.900 | PCA-64: 0.901 | −0.001 [−0.006, +0.005] | tie | PCA-32: 0.902 | 0.990 vs 0.997 |
| CIFAR-100 | d64: 0.863 | PCA-whitening-64: 0.866 | −0.003 [−0.008, +0.002] | tie | PCA-whitening-32: 0.831 | 0.996 vs 0.993 |
| Banking77 | d64: 0.917 | PCA-64: 0.919 | −0.002 [−0.008, +0.004] | tie | PCA-32: 0.908 | 0.997 vs 1.000 |
| STL-10 | d128: 0.979 | PCA-128: 0.984 | −0.005 [−0.007, −0.003] | loss | PCA-32: 0.984 | 0.995 vs 1.000 |
| digits | d32: 0.966 | PCA-32: 0.976 | −0.010 [−0.017, −0.002] | loss | PCA-16: 0.972 | 0.995 vs 1.000 |
| synthetic | d128: 0.744 | PCA-whitening-128: 0.910 | −0.165 [−0.180, −0.151] | loss | PCA-whitening-32: 0.892 | 0.998 vs 1.000 |
| STS-B | d128: 0.797 | PCA-128: 0.833 | −0.036 | no interval | PCA-whitening-64: 0.818 | 0.978 vs 0.985 |

**Result: no wins, 3 ties, 3 losses, and one shortfall without an interval.**
- **Outcome A** ("wins on at least half") does not hold.
- **Outcome B** (a compression method: within the seed spread of a lower-d baseline, and at least as
  trustworthy as PCA) does not hold. Trustworthiness is below PCA's on six of seven datasets. On Pets,
  STL-10, digits, synthetic and STS-B, PCA or PCA-whitening at a lower d matches or beats the refiner.
- **Outcome C** applies.
- One fixed-d configuration beats its same-d baseline: CIFAR-100 at d = 32 (+0.028 over
  PCA-whitening-32). Validation did not select it, so it does not count.
- The default `EmbedKit` (auto d = 11 to 28) has no baseline at its dimension, because baselines use
  d ∈ {16, 32, …}. It scores below the fixed-d refiner on every dataset.

## E2 results

Full tables, including the linear probe: `benchmarks/results/e2/REPORT.md`.

| | Pets | CIFAR-100 | Banking77 | synthetic |
|---|---|---|---|---|
| Raw test kNN | 0.901 | 0.859 | 0.931 | 0.746 |
| Test kNN at the last epoch: median (best) of 30 configurations | 0.893 (0.908) | 0.815 (0.847) | 0.897 (0.921) | 0.683 (0.803) |

CIFAR-100 and Banking77 are scored on the subsample, so raw differs slightly from E1.

### Within a run: a stopping signal?

Median over 30 runs of the Spearman ρ between the monitor and test kNN accuracy over the 20
evaluations. Each cell also gives the mean regret of stopping at the monitor's best epoch: the
run's best test kNN minus the test kNN at that epoch.

| Monitor | Pets | CIFAR-100 | Banking77 | synthetic |
|---|---|---|---|---|
| uniformity | −0.22 / 0.012 | +0.42 / 0.008 | **+0.55** / 0.005 | +0.16 / 0.031 |
| isotropy | −0.22 / 0.013 | **+0.51** / 0.005 | +0.47 / 0.003 | +0.26 / 0.021 |
| k-skewness | −0.03 / 0.009 | +0.50 / 0.009 | +0.16 / 0.005 | +0.07 / 0.034 |
| kNN overlap | +0.19 / 0.005 | −0.17 / 0.007 | +0.02 / 0.005 | +0.08 / 0.023 |
| effective rank (RankMe) | +0.07 / 0.009 | +0.02 / 0.009 | −0.21 / 0.006 | −0.17 / 0.025 |
| held-out loss | −0.59 / 0.015 | −0.38 / 0.018 | −0.64 / 0.009 | −0.37 / 0.040 |
| training loss | −0.14 / 0.012 | +0.46 / 0.010 | +0.54 / 0.005 | +0.17 / 0.032 |
| validation kNN accuracy (labels) | +0.20 / 0.003 | +0.52 / 0.003 | +0.42 / 0.002 | +0.49 / 0.006 |
| no stopping (last epoch) | — / 0.012 | — / 0.010 | — / 0.005 | — / 0.033 |

- No label-free monitor reaches a median ρ of 0.5 on more than one dataset. By the plan's rule,
  early stopping is disabled by default.
- The regrets show why the decision costs little:
  - On the real datasets, stopping at the best of any label-free monitor saves at most 0.007 over
    training to the end.
  - Stopping at the held-out contrastive loss's best epoch would lose more than not stopping at all.
- Even the labelled validation accuracy reaches a median ρ of only 0.2 to 0.5. Most accuracy curves
  are nearly flat after the first evaluations, so within-run correlation is dominated by noise.
- The synthetic set is the exception: some runs collapse after their peak (largest last-epoch
  regret 0.33).

### Across runs: a selection signal?

Spearman ρ over the 30 configurations between the monitor and test kNN, both at the last epoch.
Each cell also gives the selection regret: the best configuration's test kNN minus that of the
configuration the monitor ranks first.

| Monitor | Pets | CIFAR-100 | Banking77 | synthetic |
|---|---|---|---|---|
| uniformity | −0.60 / 0.120 | −0.13 / 0.220 | −0.17 / 0.081 | +0.73 / 0.079 |
| isotropy | −0.15 / 0.004 | +0.10 / 0.101 | −0.04 / 0.004 | +0.62 / 0.087 |
| k-skewness | −0.34 / 0.027 | +0.68 / 0.006 | +0.28 / 0.003 | +0.00 / 0.122 |
| kNN overlap | **+0.72 / 0.000** | **+0.79 / 0.001** | **+0.74 / 0.002** | −0.03 / 0.135 |
| effective rank (RankMe) | −0.19 / 0.120 | +0.13 / 0.220 | +0.20 / 0.081 | +0.49 / 0.035 |
| held-out loss | −0.58 / 0.078 | −0.33 / 0.073 | −0.43 / 0.035 | +0.11 / 0.011 |
| validation kNN accuracy (labels) | +0.87 / 0.006 | +0.95 / 0.002 | +0.89 / 0.001 | +0.95 / 0.000 |

On the real embeddings, kNN overlap with the input chooses well. It amounts to "change the
neighbourhoods as little as possible", which is consistent with E1: the raw embedding is hard to
beat. On the synthetic set, the input neighbourhoods are polluted by the nuisance directions, so
preserving them is the wrong goal there. Phase 7's selection-based auto-configuration (WP 7.2) can
use kNN overlap as its label-free criterion. Labelled validation accuracy is better wherever labels
exist.

## Limitations

- **One seed.** E2 uses one seed per configuration. E1 has five seeds, but the bootstrap intervals
  are over test points, not over training sets.
- **Subsampled downstream scoring.** E2 scores downstream accuracy on a subsample of at most 10,000
  training and 5,000 test rows, with a fixed-C probe. The correlations rest on these cheaper scores.
- **Early stopping in E1.** E1's `EmbedKit` runs used the 0.2.0 defaults: early stopping on
  k-skewness, with 10% of the rows held out. In E2, stopping at k-skewness's best epoch lost 0.005
  to 0.009 on the real sets, about as much as training to the end. So the new default (no
  stopping, every row used) is not expected to turn the ties into wins. The losses on synthetic and
  STS-B are far larger than this margin. E6 (WP 5.6) reruns the default with the new settings.
- **Limited grid.** E2's configurations are drawn from the existing refiner. Phase 5 changes the
  architecture (residual adapter initialised at PCA), so its runs should log the same monitors.
- **Dimensions without baselines.** The baselines do not cover the default `EmbedKit` dimensions
  (11 to 28).

## Compute

Quadro RTX 8000 nodes (`r9ng-8000-*`), one GPU and 8 CPUs per array task, at most two tasks at a
time.

| Run | Slurm array | Sum of per-method times | Wall time per dataset |
|---|---|---|---|
| E1 | 3077382 (7 tasks) | 11.9 GPU-hours | CIFAR-100 8 h 12 min; Banking77 70 min; the others 13 to 32 min |
| E2 | 3077451 (4 tasks) | 2.9 GPU-hours | CIFAR-100 1 h 47 min; Banking77 38 min; Pets 17 min; synthetic 11 min |

The OpenBLAS thread count mattered more than the GPU. With 8 BLAS threads, the linear probe's
small L-BFGS fits ran 10 to 20 times slower than with 1 to 4. The jobs now use half the allocated
CPUs.

## Reproducing

```bash
# on the cluster (see benchmarks/cluster/): one array task per dataset
sbatch benchmarks/cluster/run.slurm.sh      # E1
sbatch benchmarks/cluster/e2.slurm.sh       # E2
# anywhere, from the committed result files
python benchmarks/report.py --experiment e1 --out benchmarks/results/e1/REPORT.md
python benchmarks/report_gate.py --out benchmarks/results/e1/GATE.md
python benchmarks/report_e2.py --out benchmarks/results/e2/REPORT.md
```

Every result file records the commit and whether the code had local changes. All E1 and E2 files
are from clean checkouts: `a20c252` or `6ef1a3b` for E1, which differ only in the cluster
environment script, and `bcba00f` for E2.
