# Benchmark E8: supervised and regression modes

E8 chooses the defaults of `EmbedKit` with labels: `mode="supervised"` (classes) and
`mode="regression"` (continuous targets). It starts from the defaults fixed just before it, called
the *starting default* below: SupCon alone (or Rank-N-Contrast for regression) on the linear
refiner, with `KNNPairs(5)` views and early stopping on a held-out score. Each arm changes one factor,
and the arms are compared with LDA, PCA and the raw embeddings, with few labels, with label noise and
with few training rows. Run on 2026-11-13 and 2026-11-14 on the OCCIDATA cluster, 5 seeds per trained
arm. The generated tables are in `benchmarks/results/e8/REPORT.md`; this document gives the outcome
and the decisions taken from it.

What changed in the library as a result (0.5.0): supervised `EmbedKit` trains the residual adapter,
and regression `EmbedKit` trains the MLP on Gaussian-noise views. Since 0.6.0, `EmbedKit` with labels
also compares the refiner with the input, centring and PCA on held-out rows and keeps the input
unless one of them is significantly better (`docs/benchmarks/BENCHMARK_SELECTION.md`).

## Outcome

- **Supervised default: SupCon alone on the residual adapter** (`architecture="residual"`: the
  PCA-initialised linear map plus a zero-initialised MLP), with `KNNPairs(5)` views, held-out kNN
  accuracy as the early-stopping monitor and patience 10.
  - On the 6 full classification sets it beats the linear starting default on 2 and ties on 4
    (kNN accuracy). The acceptance rule (wins on at least half the datasets) is not met. The plain MLP
    meets it (3 wins, 3 ties) and ties the residual adapter on 5 of 6 sets.
  - The rule was set aside because of the label-efficiency sets, run as a follow-up for this
    decision. With 1 to 100 labels per class (12 sets) the MLP loses to the linear default on 6
    sets: -0.082 on CIFAR-100 at 1 label per class, -0.074 at 5, -0.039 on STL-10 at 1. The
    residual adapter loses on 2 (-0.016 and -0.009, CIFAR-100 at 1 and 5) and wins on 6, by up to
    +0.050 (Banking77, 5 labels per class). Over all 18 sets: residual 8 wins, 8 ties, 2 losses;
    MLP 7, 5, 6.
  - Against training-free baselines the new default wins on kNN accuracy on 4 of 6 sets against
    the raw embeddings and on 3 of 6 against LDA, and loses on none. The gains are small on real
    data: +0.003 to +0.021 against raw, +0.002 to +0.013 against LDA. The large one is the
    synthetic anisotropic set (+0.185 against raw).
- **Regression default: Rank-N-Contrast on the plain MLP with relative Gaussian-noise views**
  (`architecture="mlp"`, `GaussianNoise(0.05)`), held-out ridge R^2 as the monitor.
  - The MLP beats the linear starting default (RnC) on ridge R^2 on all 4 sets, by +0.010 (Yelp) to
    +0.266 (fluorescence). Noise views on top beat the MLP alone on 3 of 4 and tie on Yelp.
  - Against the raw embeddings and PCA at the same dimension the new default wins on all 4 sets,
    on both ridge R^2 (+0.023 to +0.574 against raw) and 10-NN R^2.
  - **With few training rows no refiner is reliable.** With 200 or 1000 training rows the MLP
    loses to the raw embeddings on 6 of 8 sets. `EmbedKit(mode="regression")` therefore warns
    when it is fitted on 1000 rows or fewer.
- **Labels make the refiner useful; LDA remains a strong baseline.** With every training row the
  supervised refiner is at most 0.02 above LDA on real data. With 5 labels per class LDA is ahead
  of the best refiner by 0.04 to 0.08. With 1 label per class (where LDA cannot run) the linear
  refiner falls below the raw embeddings on Banking77 and STL-10.
- **Rejected changes** (none meets the rule): the uniformity term (`supcon_au`), the spread term
  (`SupConLoss(spread_weight)`), nested training, and turning off early stopping, in
  classification. In regression, no early stopping on the MLP adds little (+0.003 to +0.019) and
  loses on Yelp (-0.016).
- **Class collapse is real, and the spread term can undo it.** Fitted on CIFAR-100's 20
  superclasses and scored on the 100 fine classes, SupCon falls to 0.645 kNN accuracy (raw 0.879).
  With uniformity it reaches 0.804, and with the spread term at 0.5, 0.869. Trained on the fine
  labels themselves, the spread term ties SupCon on all 6 sets, so it is a documented option for
  label hierarchies, not a default.

## Datasets

| Set | Target | Train / test | Embedding |
|---|---|---|---|
| `banking77_mpnet` | 77 intents | 9,993 / 3,076 | all-mpnet-base-v2 |
| `cifar100_dinov2` | 100 classes (20 superclasses) | 50,000 / 10,000 | DINOv2-base |
| `digits` | 10 classes | 898 / 899 | raw 8×8 pixels |
| `pets_resnet50` | 37 breeds | 3,680 / 3,669 | ResNet-50 (ImageNet) |
| `stl10_clip` | 10 classes | 5,000 / 8,000 | CLIP ViT-B/32 |
| `synthetic_anisotropic` | 10 classes | 3,000 / 3,000 | random tanh network, nuisance directions |
| `synthetic_regression` | continuous | 3,000 / 3,000 | 8-d latent through a random tanh network, nuisance directions |
| `synthetic_regression_ties` | the same target on 5 levels | 3,000 / 3,000 | as above |
| `yelp_mpnet` | review stars | 20,000 / 5,000 | all-mpnet-base-v2 |
| `fluorescence_esm2` | TAPE GFP log-fluorescence | TAPE splits, about 21k / 27k | ESM-2 35M, mean over residues |

The classification sets are those of E1 (`docs/benchmarks/BENCHMARK_E1_E2.md`) minus STS-B. Fluorescence test
sequences carry more mutations than the training ones, so it is an extrapolation task; ridge on
raw ESM-2 features scores below zero there.

Derived sets:

- `<name>__lpc<k>`: k labelled training rows per class (k = 1, 5, 20, 100) on CIFAR-100,
  Banking77 and STL-10. Everything is restricted to these rows, gallery and probe included.
- `<name>__n<k>`: k random training rows (k = 200, 1000) on the 4 regression sets.
- Label noise: 10, 20 and 40% of the fit labels replaced at random. CIFAR-100 superclass fitting
  uses the 20 coarse labels. Both are scored with the clean fine labels.

## Protocol

As in E1 and E6: 5 seeds per trained arm, the test split scored once per seed, paired bootstrap
over test points against a reference. A win or loss needs the 95% interval to exclude 0. The
output dimension is min(128, D // 2). Each arm changes one factor from the starting default and
keeps the others as they were (linear architecture, `KNNPairs(5)` views; `benchmarks/experiments.py`
pins these, so the arms do not move with the library defaults).

- Classification: kNN accuracy (cosine, k = 5) first, then a logistic-regression probe and NC1.
- Regression: ridge R^2 (`linear_probe`) first, then 10-NN regression R^2 (`knn_r2`).

Arms:

| Arm | Change from `supcon` / `rnc` |
|---|---|
| `prev` | The 0.2 default: MLP, SupCon + 0.5 AlignUniform, no early stopping |
| `supcon_au` | SupCon + 0.5 AlignUniform |
| `supcon_mlp`, `rnc_mlp` | Plain MLP |
| `supcon_res`, `rnc_res` | Residual adapter |
| `supcon_noes`, `rnc_noes` | No early stopping |
| `spread{0.25,0.5}` | `SupConLoss(spread_weight=w)`, w chosen on validation |
| `nested-d<k>` | One model nested at min(256, D), read at k |
| `rnc_old` | The 0.1.2 RnC loss (`benchmarks/legacy_losses.py`) |
| `rnc_noise` | `GaussianNoise(0.05)` views instead of `KNNPairs(5)` |
| `rnc_mlp_noise`, `rnc_mlp_noes` | The MLP with noise views, or with no early stopping |

Baselines: raw embeddings and LDA (shrinkage) for classification, raw and PCA for regression.

## Classification

kNN accuracy (mean over seeds):

| Set | raw | LDA | linear (starting default) | MLP | residual (new default) |
|---|---|---|---|---|---|
| banking77_mpnet | 0.931 | 0.936 | 0.936 | 0.938 | 0.938 |
| cifar100_dinov2 | 0.879 | 0.895 | 0.896 | 0.900 | 0.900 |
| digits | 0.981 | 0.966 | 0.975 | 0.984 | 0.978 |
| pets_resnet50 | 0.901 | 0.919 | 0.920 | 0.919 | 0.921 |
| stl10_clip | 0.983 | 0.986 | 0.986 | 0.985 | 0.986 |
| synthetic_anisotropic | 0.746 | 0.919 | 0.916 | 0.936 | 0.931 |

Each arm against the linear default (wins / ties / losses on kNN accuracy; linear probe):

| Arm | kNN accuracy | Linear probe |
|---|---|---|
| `prev` | 3 / 3 / 0 | 3 / 2 / 1 |
| `supcon_mlp` | 3 / 3 / 0 | 2 / 4 / 0 |
| `supcon_res` | 2 / 4 / 0 | 1 / 5 / 0 |
| `supcon_noes` | 2 / 4 / 0 | 0 / 4 / 2 |
| `supcon_au` | 1 / 4 / 1 | 0 / 3 / 3 |
| spread | 0 / 6 / 0 | 2 / 4 / 0 |
| nested | 0 / 6 / 0 | 1 / 5 / 0 |

`prev` passes the rule too, but with no early stopping and the uniformity term it collapses with
few labels (0.101 on CIFAR-100 at 1 label per class, against 0.180 for the linear default).

### Label efficiency

kNN accuracy:

| Set | raw | LDA | linear | MLP | residual |
|---|---|---|---|---|---|
| banking77 lpc1 | 0.142 | | 0.118 | 0.140 | 0.134 |
| banking77 lpc5 | 0.698 | 0.808 | 0.711 | 0.737 | 0.761 |
| banking77 lpc20 | 0.856 | 0.884 | 0.877 | 0.882 | 0.886 |
| banking77 lpc100 | 0.928 | 0.936 | 0.931 | 0.933 | 0.935 |
| cifar100 lpc1 | 0.175 | | 0.180 | 0.098 | 0.164 |
| cifar100 lpc5 | 0.705 | 0.792 | 0.711 | 0.637 | 0.703 |
| cifar100 lpc20 | 0.809 | 0.843 | 0.824 | 0.801 | 0.827 |
| cifar100 lpc100 | 0.853 | 0.877 | 0.875 | 0.871 | 0.874 |
| stl10 lpc1 | 0.260 | | 0.211 | 0.173 | 0.229 |
| stl10 lpc5 | 0.905 | 0.962 | 0.898 | 0.908 | 0.919 |
| stl10 lpc20 | 0.961 | 0.981 | 0.976 | 0.973 | 0.976 |
| stl10 lpc100 | 0.977 | 0.984 | 0.983 | 0.982 | 0.983 |

With 1 label per class the raw embeddings are the safest choice: the linear refiner loses to them
on Banking77 and STL-10, the MLP on CIFAR-100 and STL-10. At 5 labels per class LDA is the better
transform on all three sets. At 20 it is still ahead on CIFAR-100 (by 0.013 over the best refiner)
and STL-10 (0.005) and level on Banking77. At 100 the refiners and LDA are level.

### Label noise

Run with the linear starting default, not the residual adapter. Up to 20% noise, SupCon does not
lose to the raw embeddings on any set (its worst difference is -0.006, a tie). At 40% it loses on
Pets (-0.056, where LDA ties at -0.008) and STL-10 (-0.005). LDA loses to raw on digits at every
noise level, and with clean labels too. Full tables: `REPORT.md`, per dataset.

## Regression

Ridge R^2 (mean over seeds):

| Set | raw | PCA | linear RnC (starting default) | MLP | residual | MLP + noise (new default) |
|---|---|---|---|---|---|---|
| fluorescence_esm2 | -0.033 | -0.122 | 0.136 | 0.402 | 0.420 | 0.541 |
| synthetic_regression | 0.702 | 0.584 | 0.579 | 0.843 | 0.761 | 0.903 |
| synthetic_regression_ties | 0.763 | 0.688 | 0.695 | 0.859 | 0.705 | 0.899 |
| yelp_mpnet | 0.676 | 0.630 | 0.688 | 0.698 | 0.698 | 0.699 |

10-NN R^2 of the new default: 0.478, 0.886, 0.900 and 0.693 (raw: -0.794, 0.383, 0.442, 0.519).

The linear RnC refiner lost to ridge on the raw embeddings on both synthetic sets; the MLP is what
makes RnC pay. The residual adapter, which starts at PCA, falls well short of the MLP on the
synthetic sets (0.761 and 0.705 against 0.843 and 0.859), where PCA itself scores below the raw
embeddings.

The 0.1.2 RnC (`rnc_old`, with the tie bug fixed in 0.1.3; D21 in `CHANGELOG.md`) differs from the
fixed loss by -0.007 to +0.002 in ridge R^2, so the fix matters less in these benchmarks than the
architecture does.

### Few training rows

Ridge R^2:

| Set | raw | PCA | linear RnC | MLP | residual |
|---|---|---|---|---|---|
| fluorescence n200 | -0.814 | -0.869 | -0.598 | -0.853 | -0.979 |
| fluorescence n1000 | -0.301 | -0.252 | -0.177 | -0.414 | -0.544 |
| synthetic n200 | 0.398 | 0.388 | 0.378 | 0.332 | 0.414 |
| synthetic n1000 | 0.658 | 0.558 | 0.527 | 0.765 | 0.614 |
| synthetic ties n200 | 0.568 | 0.455 | 0.484 | 0.431 | 0.550 |
| synthetic ties n1000 | 0.731 | 0.667 | 0.652 | 0.771 | 0.648 |
| yelp n200 | 0.530 | 0.463 | 0.375 | 0.372 | 0.471 |
| yelp n1000 | 0.598 | 0.572 | 0.560 | 0.570 | 0.580 |

Noise views were not run at these sizes. The warning threshold (1000 rows) is where the MLP first
beats raw, on the two synthetic sets only.

## Decisions

| Mode | Starting default | Now | Rule |
|---|---|---|---|
| `supervised` | linear, `KNNPairs(5)` | residual, `KNNPairs(5)` | not met (2 of 6); chosen for few-label robustness |
| `regression` | linear, `KNNPairs(5)` | MLP, `GaussianNoise(0.05)` | met (4 of 4, both changes) |
| `self_supervised` | PCA (`method="auto"`); linear refiner | unchanged | E6 |

`architecture="auto"` and `augmentation="auto"` resolve per mode, and both can be set explicitly.
SupCon alone, early stopping on a held-out score and patience 10 stay as in the starting default.

## Compute

| Job | What | GPU-hours |
|---|---|---|
| 3079012 | E8 main grid (15 tasks) | 23.0 |
| 3079280 | Residual arms | 1.7 |
| 3079354 | Follow-up: few labels, few rows, MLP combinations | 2.0 |

## Reproducing

```bash
# on the cluster (see benchmarks/cluster/)
sbatch e8.slurm.sh
TASKS=E8R_TASKS sbatch --array=0-2%2 e8.slurm.sh
TASKS=E8C_TASKS sbatch --array=0-5%2 e8.slurm.sh
# from the result files
python benchmarks/report_e8.py --out benchmarks/results/e8/REPORT.md
```

The per-run JSON files (about 200 MB) stay on the cluster
(`benchmarks/results/e8/` in the cluster clone); only `REPORT.md` is committed.
