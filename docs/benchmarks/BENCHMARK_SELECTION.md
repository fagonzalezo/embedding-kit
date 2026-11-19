# Benchmark E10: the default selection

What `EmbedKit()` does with labels, measured on 64 data sets: does the validated selection ever
make an embedding worse, how often does it help, and what does it cost? The same campaign retests
the report's hubness recommendation on 25 more sets.

Run on 2026-11-18 and 2026-11-19 on the OCCIDATA cluster (one RTX 8000 GPU per run, one seed). The
acceptance criteria were fixed before the runs (development records: `docs/dev/IMPLEMENTATION_PLAN.md`). Generated tables:
[`benchmarks/results/e10/REPORT.md`](../../benchmarks/results/e10/REPORT.md) (selection) and
[`benchmarks/results/e4x/REPORT.md`](../../benchmarks/results/e4x/REPORT.md) (diagnostic rules).

## Data

- **E4's 35 classification sets:** 10 source data sets (AG News, Banking77, Emotion, MASSIVE,
  20 Newsgroups, CIFAR-10, CIFAR-100, Flowers-102, Oxford Pets, STL-10), each embedded by 3 or 4
  pretrained encoders (MiniLM, MPNet, BGE, E5; CLIP, DINOv2, ResNet-50).
- **25 more classification sets:** the same 10 sources with 5 encoders chosen for variety of
  architecture and training, before any of their statistics was computed: GTE, DistilRoBERTa and
  averaged GloVe vectors for text; ViT-B/16 and ConvNeXt-Tiny for images. Every one of them is
  in the analysis; none was screened out.
- **4 regression sets** from E8: two synthetic sets, Yelp review stars (MPNet) and protein
  fluorescence (ESM-2).

The splits are E4's and E8's. The test part is never seen by the selection, which holds out 20% of
the training rows for its own validation.

## Outcome

**The default never made an embedding significantly worse.** `EmbedKit()` was compared with the
unchanged input on held-out test points, by cosine 10-NN accuracy (10-NN R² on regression sets),
with a paired bootstrap 95% interval (2,000 resamples).

| sets | significantly worse | significantly better | gain where better |
|---|---|---|---|
| 35 E4 classification sets | 0 | 21 | +0.8 to +15.0 points |
| 25 additional classification sets | 0 | 18 | +0.9 to +15.1 points |
| 4 regression sets | 0 | 4 | R² +0.17 (Yelp) to +1.27 (fluorescence) |

- On the 39 classification sets where it helped, the median gain is +4.3 points of accuracy.
  The largest are on Flowers-102 (ViT +15.1, ResNet-50 +15.0, ConvNeXt +10.9), Emotion with GloVe
  (+14.6) and Pets with CLIP (+13.5). The smallest are on Banking77 (MPNet +0.8, GTE +0.9).
- On fluorescence the input's 10-NN R² is negative (−0.79); the selected refiner brings it to 0.48.

**What it chose.** The supervised refiner on 44 sets, the input unchanged on 19, and PCA to 256
dimensions on 1 (STL-10 with ViT).
- The input was kept on every AG News set, on 6 of 7 20 Newsgroups sets, on 4 of 5 STL-10 sets,
  and on Flowers-102 with DINOv2 and Banking77 with BGE. These are sets where the raw embedding is
  already strong for kNN (STL-10 and Flowers-102/DINOv2 at 0.977 to 0.996) or where no candidate is
  significantly better on the validation rows.
- The refiner was chosen on every CIFAR-10, CIFAR-100, Emotion, MASSIVE and Pets set and on all
  4 regression sets.

**How much it left on the table.** Against the candidate that is best on the test set in
hindsight (each fitted on all training rows), the regret is 0 on 50 of 64 sets and at most
1.2 points (20 Newsgroups with MPNet). Where it is not 0, the selection mostly kept the input
and another candidate (centring plus L2, PCA or the refiner) was slightly better in hindsight. On
STL-10/ViT it chose PCA and the input was better by 0.08 points.

**Other outcomes** (reported, no criterion attached):
- Euclidean 10-NN accuracy: better on 41 of the 60 classification sets, worse on none.
- Linear probe (logistic regression, C = 1): better on 21, **worse on 5**, by 0.2 to 0.8 points:
  Flowers-102/ConvNeXt and Pets with ConvNeXt, DINOv2 and ViT (refiner), and STL-10/ViT (PCA).
  The selection scores candidates by kNN, so a downstream linear classifier can lose a little.
  If that is your use, compare on your own task before replacing the input.

**Time.** From 9 s (Flowers-102, 2,040 training rows) to 6.6 min (CIFAR-100/ResNet-50, 50,000
rows), median 1.0 min, on one GPU, including the final refit on all rows.

**Selection on a subsample of rows.** `selection_max_rows` (default 50,000) caps the rows used to
choose; the winner is still refitted on every row. No set here has more than 50,000 training rows,
so the default cap never binds. Lowered caps were tested against selection on all rows:

| cap | sets where it binds | significantly worse | same choice |
|---|---|---|---|
| 5,000 | 47 | 6 (−1.7 to −6.2 points) | 38 |
| 10,000 | 40 | 0 | 39 |
| 20,000 | 11 | 0 | 11 |

At 5,000 rows the subsample sometimes cannot show that the refiner beats the input, and the input
is kept. From 10,000 rows on, the choice was not significantly worse than with all rows on any set.

## The hubness recommendation, retested on 60 sets

The report recommends a hubness-reducing search (`embedkit.hubness_reduction`) for Euclidean kNN
when k-skewness is more than 3 standard deviations above its null model. In E4 this rule passed its
criteria, but it fired on only 3 of 35 sets. The 25 additional sets were added to test it again,
with the same two conditions:
1. the Spearman correlation between the signal and the gain of the best hubness reduction over
   the sets, with a 95% interval from 2,000 bootstrap resamples of the source data sets, excludes 0;
2. where the rule fires, the gain is significant on at least half of the sets.

| rule | sets | Spearman (95% interval) | fires on | significant gain where it fires | kept |
|---|---|---|---|---|---|
| hubness, Euclidean kNN | 35 | 0.36 (0.13 to 0.65) | 3 | 3 | yes |
| hubness, Euclidean kNN | 60 | **0.48 (0.30 to 0.71)** | **5** | **5** | **yes** |
| antihubs, Euclidean kNN | 35 | 0.32 (0.05 to 0.58) | 8 | 5 | yes |
| antihubs, Euclidean kNN | 60 | 0.53 (0.29 to 0.74) | 22 | 19 | (reported) |
| hubness, cosine kNN | 57 | 0.005 (−0.29 to 0.42) | 2 | 0 | (reported) |

- **The hubness recommendation stays.** On 60 sets it fires on AG News/MiniLM, CIFAR-100/DINOv2,
  MASSIVE/MiniLM, MASSIVE/DistilRoBERTa and Banking77/GloVe. The best hubness reduction gained
  1.3 to 5.1 points of Euclidean kNN accuracy on each.
- The evidence is still small: 5 firing sets from 4 sources, one (Banking77/GloVe, z = 3.2) close
  to the threshold, and Emotion/GloVe just below it (z = 2.96). The 23 primary rule tests have no
  multiplicity correction.
- The antihub rule gets stronger support with the added sets.
- Under cosine kNN, k-skewness predicts nothing, as in E4. The report marks it as uncalibrated
  under cosine and draws no recommendation from it.

## Limits

- One seed per set. The intervals cover the test sampling, not the variation between training runs
  of the refiner.
- Averaged GloVe vectors give an all-zero row for a text with no word in the GloVe vocabulary
  (19 rows in Banking77, 56 in MASSIVE, 323 in 20 Newsgroups). The analyzer rejects zero rows under
  the cosine metric, since they have no direction, so the cosine rows of the rule table cover 57
  sets. The Euclidean report and `EmbedKit()` accept them.
- The regression comparison covers 4 sets, two of them synthetic.
