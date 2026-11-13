# Benchmarks E6 and E7: which refiner changes help, and the hubness-reduction methods

Phase 5 of `docs/IMPLEMENTATION_PLAN.md`, following experiments E6 and E7 of `docs/CRITICAL_REVIEW.md`
(section 5), with the gate outcome of `docs/BENCHMARK_E1_E2.md` (C, plus WP 5.3). Run on 2026-11-12 and
2026-11-13 on the OCCIDATA cluster. Datasets, splits, seeds, metrics and the paired bootstrap are those
of E1 (`docs/BENCHMARK_E1_E2.md`, sections Datasets and Protocol).

## Outcome

- **WP 5.1 accepted: the default refiner is a linear map initialised at PCA** (`architecture="linear"`).
  Against the previous MLP default it wins on 5 of 6 datasets with an interval and loses on none.
  The residual variant does as well against the MLP but loses to `linear` on 3 datasets and wins on 1.
- **WP 5.2 rejected: no positive or negative change wins against `linear` on half the datasets.**
  Mutual-neighbour positives, the neighbour mask and the decoupled loss each have 1 win, 3 ties and 2
  losses. Dropout-only positives (on the residual refiner) lose on 5 of 6 datasets, by up to 0.105.
- **WP 5.3 accepted (non-inferiority).** Prefixes of one nested model at d = 256 match `linear`
  models trained at 32, 64 and 128. Of 16 comparisons with an interval, one loses (CIFAR-100, d = 32,
  -0.003), one wins and the rest tie.
- **Against training-free baselines the refiner is level, not better.** The selected `linear`
  configuration has 1 win, 4 ties and 1 loss against the best of PCA and PCA-whitening at the same d.
  So gate outcome C stands. At d = 32, however, it beats plain PCA by 0.015 to 0.09 on four
  datasets, and no label-free rule tells PCA from PCA-whitening.
- **WP 5.5: hubness reduction does not raise kNN accuracy on these embeddings.** The best change
  against raw is +0.005 (mutual proximity, CIFAR-100). Mutual proximity loses on Pets and on the
  synthetic set.
- **Decision on the self-supervised default: pending E7** (confirmed by the user, 2026-11-13). The
  plan's rule for gate C says that `EmbedKit(mode="self_supervised")` returns the best training-free
  baseline when WP 5.2 fails. The previous point and the WP 5.3 result argue for keeping the refiner as a
  label-free compressor. E7 (learning curves and transduction against PCA) decides.

## E6: one-factor ablation of the refiner

Every arm is self-supervised `EmbedKit` with the defaults of the stage and one factor changed. Each
family is reduced to its configuration best on validation, over d in {32, 64, 128} below the input
dimension (stage 1 also had the auto d, which no family selected). The phase 5 rule accepts a change
when it wins against the previous default (difference above 0, 95% interval excluding 0) on at least
half of the datasets. STS-B has no interval (Spearman over pairs), so 6 datasets decide.

Full tables: `benchmarks/results/e6/REPORT.md` (against `e6_linear`, with the per-d prefix table) and
`REPORT_vs_residual.md` (the dropout arm against `e6_residual`).

### Stage 1 (WP 5.1): architecture

Job 3077519 at `16789d3`.

| Family | vs `mlp` (W / T / L) | vs best PCA at the same d (W / T / L) |
|---|---|---|
| `mlp` (0.2.0 default) | reference | 0 / 3 / 3 |
| `residual` | 5 / 1 / 0 | 2 / 2 / 2 |
| `linear` | 5 / 1 / 0 | 1 / 4 / 1 |

`residual` against `linear`: 1 win, 2 ties, 3 losses.

### Stage 2 (WP 5.2, 5.3): positives, negatives and nested output

Job 3077755 at `77bc902`. Arms changed one factor from `linear`: `mutual`
(`KNNPairs(k=5, mode="mutual")`), `nbrmask` (`NTXentLoss(neighbor_mask=5)`), `decoupled`
(`NTXentLoss(decoupled=True)`), `dropout` (`IdentityPair` on `residual`, the only refiner with
dropout) and `matryoshka` (one `linear` model at min(256, d) with `MatryoshkaLoss` over the smaller
dims, read at each prefix).

Paired difference in test kNN accuracy (STS-B: Spearman) of each family's selected configuration
against `e6_linear`; bold where the interval excludes 0.

| Dataset | mutual | nbrmask | decoupled | dropout (vs residual) |
|---|---|---|---|---|
| Banking77 | -0.003 | **+0.003** | -0.003 | **-0.048** |
| CIFAR-100 | +0.001 | -0.000 | +0.001 | **-0.068** |
| digits | +0.001 | +0.002 | +0.001 | **-0.009** |
| Pets | **-0.025** | **-0.004** | **-0.008** | **-0.105** |
| STL-10 | **-0.006** | **-0.002** | **-0.002** | **-0.028** |
| synthetic | **+0.006** | +0.002 | **+0.004** | **+0.036** |
| STS-B (no interval) | -0.010 | +0.001 | -0.008 | -0.029 |
| **W / T / L** | 1 / 3 / 2 | 1 / 3 / 2 | 1 / 3 / 2 | 1 / 0 / 5 |

Dropout-only positives fail because a linear projection with its views differing only by dropout in a
zero-initialised branch has nothing to align: the loss is driven by its uniformity term, which spreads
the points and erases class structure (CIFAR-100 linear probe 0.20 at d = 32). SimCSE works because it
fine-tunes a pretrained encoder for about one epoch; 100 epochs on a projection head are a different
regime.

**WP 5.3, prefix against a model trained at that d** (no selection, every d):

| Dataset | d = 32 | d = 64 | d = 128 |
|---|---|---|---|
| Banking77 | -0.001 | -0.000 | -0.001 |
| CIFAR-100 | **-0.003** | +0.001 | -0.000 |
| digits | -0.000 | | |
| Pets | -0.000 | -0.001 | +0.001 |
| STL-10 | **+0.001** | +0.000 | -0.000 |
| synthetic | +0.001 | -0.001 | -0.000 |
| STS-B (no interval) | +0.000 | -0.002 | -0.001 |

One dataset with a loss, so the non-inferiority rule (at most one) accepts it. Training the nested
model costs about 1.5 times a single `linear` run at d = 128 on CIFAR-100 (4627 s against 3136 s for
5 seeds, validation and test), and replaces one run per d.

### The refiner against PCA at low d

The gate compares with the better of PCA and PCA-whitening chosen on validation, which needs labels.
Against plain PCA at the same d, the nested `linear` prefixes (test kNN accuracy; STS-B Spearman):

| Dataset | d = 32 | d = 64 | d = 128 |
|---|---|---|---|
| Banking77 | **+0.015** | **+0.007** | -0.001 |
| CIFAR-100 | **+0.024** | **+0.008** | -0.001 |
| digits | -0.000 | | |
| Pets | **-0.008** | +0.003 | +0.004 |
| STL-10 | -0.000 | -0.001 | -0.001 |
| synthetic | **+0.091** | **+0.094** | **+0.095** |
| STS-B (no interval) | +0.059 | +0.019 | -0.005 |

On the synthetic set PCA-whitening is better still (0.892 at d = 32 against 0.839). The refiner sits
between PCA and PCA-whitening there, and at or above PCA on the real datasets.

## E1 addendum: hubness reduction (WP 5.5)

Job 3077756 (CPU nodes; 15 minutes on CIFAR-100, under a minute elsewhere). Secondary distances on the
raw embedding, exact search: local scaling (`raw+ls`, k = 5), mutual proximity (`raw+mp`, Gaussian,
2000-row sample), CSLS (`raw+csls`, k = 10, cosine) and DisSimLocal (`raw+dsl`, k = 5). Test cosine
kNN accuracy minus raw, paired; bold where the interval excludes 0. STS-B has no kNN metric.

| Dataset | raw | LS | MP | CSLS | DSL |
|---|---|---|---|---|---|
| Banking77 | 0.931 | -0.004 | -0.003 | +0.001 | -0.006 |
| CIFAR-100 | 0.879 | +0.002 | **+0.005** | +0.003 | -0.001 |
| digits | 0.981 | +0.001 | +0.002 | +0.004 | +0.006 |
| Pets | 0.901 | -0.002 | **-0.009** | -0.001 | +0.001 |
| STL-10 | 0.983 | +0.000 | +0.001 | +0.001 | +0.000 |
| synthetic | 0.746 | -0.004 | **-0.028** | -0.008 | **-0.014** |

These are the methods of Feldbauer and Flexer (2019) and Conneau et al. (2018). Their published gains
are on high-hubness data (high intrinsic dimension, k-skewness well above 1). The pretrained embeddings
here are not in that regime, and the methods leave accuracy unchanged. They stay available in
`embedkit.hubness_reduction` and the analyzer recommends them when hubness is high; no default changes.

## E7

Running (job 3078240, submitted 2026-11-13 01:05 CET). This section will hold the learning curves and
the transduction comparison.

## Compute

| Job | What | GPU-hours |
|---|---|---|
| 3077519 | E6 stage 1 | about 16 |
| 3077755 | E6 stage 2 | 18.0 |
| 3077756 | E1 hubness (CPU) | 0.3 CPU-node hours |

## Reproducing

```bash
# on the cluster (see benchmarks/cluster/)
sbatch e6.slurm.sh; sbatch e1_hub.slurm.sh; sbatch e7.slurm.sh
# anywhere, from the committed result files
python benchmarks/report_e6.py --out benchmarks/results/e6/REPORT.md
python benchmarks/report_e6.py --reference e6_residual --out benchmarks/results/e6/REPORT_vs_residual.md
python benchmarks/report.py --experiment e1 --out benchmarks/results/e1/REPORT.md
python benchmarks/report_e7.py --out benchmarks/results/e7/REPORT.md
```
