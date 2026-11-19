# Benchmarks E3 and E5: how far the diagnostics can be trusted

WP 4.5 of `docs/IMPLEMENTATION_PLAN.md`, following experiments E3 and E5 of `docs/CRITICAL_REVIEW.md`
(section 5). Run on the OCCIDATA cluster on 2026-11-15 (first run) and 2026-11-16 (calibration parts
rerun after the fixes below). Full tables: `benchmarks/results/e3/REPORT.md` and
`benchmarks/results/e5/REPORT.md`; the first run's reports are in commit `7eca9b2`.

- **E3** (`benchmarks/e3.py`): data with known answers. `id`: 16 manifolds of known intrinsic dimension
  (ID 2 to 20), 1,110 cells over n, ambient D, noise and a nonlinear map, 5 seeds. `gauss`: 1,260
  Gaussian cells (flat, power-law `i^-0.5`, `i^-1`, `i^-2`, three spikes, a shared mean; n = 500,
  2,000, 10,000; d = 32, 256, 1,024; k = 5, 10, 30; Euclidean and cosine), plus the flat and `i^-1`
  Gaussians with every row L2-normalised (`_l2`). `subsample`: six real embeddings (CIFAR-100 DINOv2,
  Yelp MPNet, fluorescence ESM-2, GloVe-100, SIFT-128, GIST-960), five random subsets each at n = 1,000,
  2,000, 5,000 and 20,000.
- **E5** (`benchmarks/e5.py`, `run.py --experiment e5`): four good embeddings (CIFAR-100 DINOv2,
  Banking77 MPNet, STL-10 CLIP, Pets ResNet-50), each damaged in six ways at two levels, reported
  under both metrics, and repaired with the training-free baselines of phase 2.

## Outcome

- **The phase 4 exit criterion holds after the fixes.** On all six real datasets the report reads the
  same at 2,000 and at 20,000 rows (majority over five subsets of each recommendation type and of the
  sign of each calibrated z beyond +-3; the plateau flag was dropped from the reading after the first
  run). In the first run it failed on four of six. Two caveats. The z items now agree largely by
  design: at both sizes the calibration uses 2,000 rows with k = 22, so only the rows differ; for
  Yelp (exactly 20,000 rows) they do not even differ, the 2k subset being the 20k report's
  calibration sample, and its z-scores are identical. And single subsets are noisy: 12 of 30
  subsets at 2,000 rows differ from the 20,000-row majority on at least one item (22 of 330
  items), almost all k-skewness or antihub signs with z between about 2 and 4; Yelp's antihub z
  ranges from 0.7 to 5.2 across subsets.
- **On data drawn from the null family the calibrated statistics meet the WP 4.1 bar (95% inside
  +-3) under Euclidean distance except neighbour consistency on spiked spectra at d = 1,024.
  Under cosine the antihub ratio misses it (0.89 of cells inside +-3; spiked at d = 256 and 1,024,
  `i^-0.5` and `i^-1` at d = 1,024), as does k-skewness on `i^-0.5` at d = 1,024.** Rows normalised
  upstream with a power-law spectrum (`power1_l2`) are not modelled by the null.
- **The report's properties detect every kind of damage, its recommendations rarely name it.** In E5
  the expected property moved beyond the noise floor in 131 of 136 (cosine) and 130 of 136
  (Euclidean) cases. The misses: `scale_max_deviation` under noise on CIFAR-100 and STL-10 (and
  Pets at the higher level, Euclidean), and the kernel spectral gap for `domain1` on STL-10.
  The recommendations that a damage adds are mostly side effects (concentration after any damage on
  STL-10 CLIP under Euclidean; antihubs after a mean shift under cosine). Which properties predict that
  an intervention helps is E4's question (phase 7).
- **Intrinsic dimension:** the k-NN estimators (TwoNN, MLE, MOM) read 14 to 29% low at ID 20 even
  at n = 5,000 and noise-free. ESS is within 0.7 of the true ID on the 10- and 20-dimensional
  spheres and cubes but reads 34% and 58% high on two curved manifolds (M6, M8; centre cell), runs only up to
  n = 5,000 and costs minutes; DANCo is erratic (+3.8 on the 10-sphere, NaN on the 2-cube); lPCA
  counts linear dimensions and overshoots on curved manifolds. Bootstrap intervals
  describe sampling error well (se / sd 0.9 to 1.2) but cover the true ID in 1 to 33% of cells,
  because the bias is larger than the interval.

## Changes made because of the first run

All five are in commits `f4e5fa8` and `27b8421`; the decisions on items 2 to 5 are the user's
(2026-11-16).

1. **The Gaussian null's mean** is the sample mean shrunk so that its squared norm is unbiased.
   Centred on the raw sample mean, zero-mean data gave uniformity z of -6 to -19 (a WP 4.1 bug: its
   acceptance data had a large mean, which hides the sample mean's noise). Uniformity inside +-3
   went from 31% to 98% of flat cells.
2. **Calibration k** follows the default rule on the calibration rows (`report.calibration_k`, 22
   for 2,000 rows) instead of reusing the report's k (71 at 20,000 rows). Neighbour-consistency z on
   the same 2,000 Pets rows had grown from 9 (k = 11) to 23 (k = 71).
3. **The antihub ratio is calibrated** (a fifth statistic on the same kNN graph, null sd floored at
   one count); the fixed rule `antihub_ratio > 0.3` fired on 31% of flat Gaussian cells and flipped
   between 2,000 and 20,000 rows on GloVe. After the rerun its recommendation was restricted to
   Euclidean distance (the user's decision): under cosine the antihub z is reported but raises
   nothing, because of the false alarms below. The rerun's reports predate this restriction, so
   their cosine antihub recommendations would no longer be made; the subsample part (the exit
   criterion) is Euclidean and unaffected.
4. **`scale_max_deviation` replaces the plateau flag** (TwoNN on n, n/2, n/4, n/8 rows within 10%),
   which flipped between subsets of one dataset and separated noisy from clean manifolds only weakly.
5. **Rows with a common norm** are compared with null draws on the same sphere, under either metric.
   Unit rows against a null with varying norms gave `distance_cv` z down to -35 and a concentration
   recommendation in every `_l2` cell; now none.

A sphere-aware null for normalised rows with an anisotropic spectrum was tried and not adopted (see
Limitations).

## Per statistic: bias, variance, dependence on n

"Null behaviour" is the raw statistic on isotropic Gaussians (E3 null table, k = 10, Euclidean).
"Spread" is the median over the six real datasets of the relative sd over five subsets at n = 2,000.
"20k to 2k" is the median relative change of the subset median from n = 20,000 to n = 2,000.

| Statistic | Null behaviour (bias) | Spread at 2k | 20k to 2k | Reading |
|---|---|---|---|---|
| k-skewness | grows with n and d on structureless data: 2.8 (500 x 32) to 15.2 (10,000 x 1,024) | 0.11 | -0.18 | only against the null (calibrated) |
| antihub ratio | 0.10 to 0.39 under Euclidean, about 0 under cosine; falls as k grows | 0.18 | +2.18 | only against the null (calibrated) |
| hub ratio | about 0.11 to 0.14 | small | 0.00 | descriptive |
| `distance_cv` | about `1 / sqrt(2d)` (0.126 at d = 32, 0.022 at d = 1,024), flat in n | 0.015 | 0.00 | against the null |
| neighbour consistency | 0.976 to 0.982, nearly flat | < 0.001 | 0.00 | against the null (its z depends on k) |
| uniformity | -3.75 to -3.99, flat in n | 0.006 | 0.00 | against the null (a property) |
| participation ratio (corrected) | about 1.0 on isotropic data at any n | 0.018 | 0.00 | property |
| IsoScore | biased low when n is not much larger than d (0.33 at 500 x 1,024) | 0.019 | -0.02 | property |
| spectrum alpha | 0.05 to 0.33 on isotropic data (sampling spread of the eigenvalues) | 0.004 | +0.03 | property |
| top-1 / top-10 PC share | sampling spread inflates them at small n | 0.028 | 0.00 | property |
| mean vector ratio, mean cosine | `~ 1 / sqrt(n)` on zero-mean data | 0.003 | 0.00 | property |
| kernel effective rank, spectral gap | effective rank saturates near min(n, d); gap about 1.02 | 0.013, 0.031 | 0.00, +0.01 | property |
| TwoNN | far below d on Gaussians (78 to 91 at d = 256, 162 to 194 at d = 1,024) | 0.031 | -0.02 | reference ID |
| MLE, MOM | as TwoNN, more biased at high ID | 0.018, 0.012 | -0.08, -0.09 | ID |
| lPCA | counts linear dimensions | 0.032 | 0.00 | ID |
| `scale_max_deviation` | median 0.06 on clean manifolds, 0.23 at noise 0.2 | 0.43 | +0.58 | descriptive; too noisy for a threshold |
| `suggested_target_dim` | 1.5 x TwoNN | 0.030 | -0.02 | used by `EmbedKit` |

Raw k-skewness and the antihub ratio depend on n, d and k too much to be read without the null;
`distance_cv`, neighbour consistency and uniformity are stable across n but their scale depends on d
and the spectrum, which the null supplies.

## Calibrated statistics on null data

Share of cells with |z| < 3 by family (rerun; all families except `_l2` are drawn from the null):

| family | k-skewness | antihub ratio | `distance_cv` | neighbour consistency | uniformity |
|---|---|---|---|---|---|
| flat | 1.00 | 0.98 | 0.98 | 1.00 | 0.98 |
| flat_l2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| mean | 0.99 | 0.98 | 0.97 | 1.00 | 1.00 |
| power0.5 | 0.93 | 0.96 | 0.98 | 0.98 | 1.00 |
| power1 | 1.00 | 0.98 | 1.00 | 0.97 | 1.00 |
| power1_l2 | 0.96 | 0.51 | 0.00 | 0.97 | 0.00 |
| power2 | 0.99 | 0.99 | 1.00 | 0.99 | 1.00 |
| spiked | 0.99 | 0.87 | 1.00 | 0.84 | 1.00 |

False alarms (share of null cells where a calibrated recommendation fires) are at most 4% except:
antihubs under cosine on `spiked` (44%), `power0.5` (11%) and `power1_l2` (49%, both metrics);
hubness under cosine on `power0.5` (24%); neighbour consistency under Euclidean on `spiked` (21%).
The ID/D rules are not calibrated and fire on Gaussians by design (they describe the ratio, not a
defect).

## Intrinsic dimension

Bias at the centre cell (n = 5,000, D = 256, no noise, linear embedding), the WP 4.2 hand-over:

| manifold | DANCo | ESS | MLE | MOM | TwoNN | lPCA |
|---|---|---|---|---|---|---|
| cube10 | -0.01 | -0.65 | -1.69 | -1.94 | -1.37 | 0.00 |
| cube20 | 1.80 | -0.68 | -5.13 | -5.88 | -4.28 | 0.00 |
| sphere10 | 3.80 | 0.16 | -0.84 | -1.11 | -0.47 | 1.00 |
| sphere20 | -0.40 | 0.16 | -3.86 | -4.84 | -2.75 | 1.00 |

- Relative bias of the k-NN estimators shrinks with n (TwoNN -6% at 500 rows to -3% at 50,000; MLE
  -12% to -4%), does not depend on D or on a tanh map, and turns positive with noise (TwoNN +54% at
  noise 0.2).
- Bootstrap se against the sd over seeds: median 1.04 (MLE), 0.92 (MOM), 1.16 (TwoNN), 0.27 (lPCA,
  which is nearly deterministic). Coverage of the true ID: 6% (MLE), 1% (MOM), 33% (TwoNN), 38%
  (lPCA).
- DANCo returns NaN on the 2-cube at n = 5,000 (it works at n = 500).

## E5: damage and repair

Expected signatures were written before the runs (plan, WP 4.5). Detection used a noise floor from
five 80% row subsets of each clean set.

| damage | expected properties moved (cosine / Euclidean) | best repair (recovered within 1 point of clean raw kNN) |
|---|---|---|
| mean shift | 16/16, 16/16 | standardisation, ABTT or CSLS; recovered on all four |
| nuisance directions | 32/32, 32/32 | `abtt-3`; recovered on all four |
| anisotropic rescaling | 16/16, 16/16 | ZCA or CSLS; recovered except STL-10 at condition 10,000 |
| isotropic noise | 28/32, 27/32 | CSLS or PCA; recovered at the lower level on all four, at the higher level on none (PCA best) |
| zero padding | 24/24, 24/24 | several (CSLS, MP, PCA, PCA-whitening, standardisation); recovered except Banking77 at 4 x padding |
| domain offset | 15/16, 15/16 | ABTT (1 or 3 directions); recovered on all four |

## Limitations and open findings

**WP 8.4 (0.7.0).** The cases below are now marked in the report: a statistic whose null missed
the bar in the setting at hand carries `uncalibrated` (the reason), raises no recommendation and
is not counted in `n_outside_null` (`uncalibrated_statistics` in `embedkit/analysis/report.py`).
With the flags, every (family, metric, statistic) of the E3 null cells meets the bar except
`distance_cv` under cosine with a large mean (`mean` family, 42 of 45 cells, the three misses at
|z| 3.06 to 3.26), left unflagged (`benchmarks/check_e3_flags.py`,
`benchmarks/results/e3/FLAGS_WP84.md`). Each z now has a 95% interval over resamples of the null
draws (`z_interval`), and a recommendation whose interval reaches below the threshold says so.
DANCo's NaN on `cube2` was a bug in scikit-dimension 0.3.4 (the cosine of neighbour angles was
clipped at 1 but not at -1); EmbedKit clips both sides, and DANCo reads about 2.1 to 2.2 there.

- **Rows normalised upstream with an anisotropic spectrum are not calibrated.** The null is fitted to
  rows already on the sphere, and its draws, normalised again, are more isotropic than the data:
  `distance_cv` and uniformity z of +8 to +11 and a false antihub recommendation in about half of the
  `power1_l2` cells. A moment-matched sphere null was tried (branch `wip/sphere-null`) and failed the
  bar fixed before the test: one version fixed the power-law case and broke data with a strong mean,
  the other the reverse. (The branch is local, not pushed.) Documented in the `EmbedKitAnalyzer` docstring; revisit in phase 7 (the
  user's decision, 2026-11-16). Embeddings that arrive normalised (many sentence encoders) are
  affected. The four E5 sets are not normalised (their Euclidean and cosine reports differ); the
  subsample sets were not checked, which does not affect the exit criterion (row norms are a
  property of the dataset, so the same null applies at every n).
- **Spiked spectra at d = 1,024:** neighbour consistency under Euclidean (inside +-3 in 0 to 60% of
  seeds) and the antihub ratio under cosine (false alarm in 44% of cells) are not calibrated.
- **Near-threshold readings flip on 20 null draws.** A z between 3 and 4 is not a stable reading:
  per-subset flips concentrate on k-skewness and antihub signs. Calibration uses 20 draws by default
  (the user's decision, 2026-11-14).
- **Recommendations are weak damage detectors.** The properties carry the information; turning them
  into actions is phase 7 (E4).
