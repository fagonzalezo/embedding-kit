# EmbedKit

EmbedKit is a Python library for analyzing and changing ML embedding spaces. It has an analysis
side (intrinsic dimension, hubness, isotropy, distance concentration, kernel diagnostics, each
compared with a null model) and an improvement side (training-free baselines, hubness-reducing
neighbour search, and a contrastive refiner trained with or without labels). A benchmark harness
and an evaluation protocol with paired bootstrap intervals ship with it. Most benchmark results
below are negative, and the defaults were changed to match them.

## What the benchmarks show

All numbers below come from held-out splits with paired bootstrap intervals; "significant" means the
95% interval excludes 0. Sources are in `docs/` and `benchmarks/results/`. The data are mostly
pretrained text and image embeddings (E4: 35 sets from 10 source data sets and 3 or 4 encoders each);
digits (raw pixels) and synthetic sets are also in E1 to E8.

**Where the library does not help**

- **The self-supervised refiner did not beat PCA.** In E1 it won on none of 6 data sets against the
  best training-free baseline at the same dimension (3 ties, 3 losses; STS-B scored 0.036 lower,
  without an interval) ([`docs/BENCHMARK_E1_E2.md`](docs/BENCHMARK_E1_E2.md)). In E6 and E7, against
  the better of PCA and PCA-whitening, it had 1 win, 4 ties and 1 loss with every training row, and
  lost on 4 of 7 data sets when fitted on 200 to 1,000 rows (Banking77 -0.052 kNN accuracy,
  synthetic -0.251) ([`docs/BENCHMARK_E6_E7.md`](docs/BENCHMARK_E6_E7.md)). It costs about 100 times more than PCA
  ([`CHANGELOG.md`](CHANGELOG.md), 0.4.0). No label-free statistic tracked test accuracy during a
  run (E2).
- **No training-free fix beat the raw embedding on most data sets.** In E4, on 35 natural
  classification sets, nothing was significantly better than the raw embedding on more than 16 sets
  (centring plus L2 normalisation, Euclidean kNN: 16 up, 1 down, median +0.3 points). On cosine kNN
  the best count was 10 (hubness reduction; 6 sets down) ([`benchmarks/results/e4/REPORT.md`](benchmarks/results/e4/REPORT.md)).
- **PCA to the report's `suggested_target_dim` lost accuracy** on 32 of the 35 sets (cosine kNN,
  median -5.1 points). That was the 0.5.0 default. ZCA whitening lost on 34 of 35 (cosine kNN) and
  on all 35 (Euclidean kNN, median -21.5 points); PCA-whitening on 26 and 24.
- **Most diagnostic rules did not predict a fix.** E4 tested 9 rules (hubness, antihubs, distance
  concentration, neighbour consistency, ID/D, and four hypotheses about the mean direction, top
  components, anisotropy and dominant coordinates). Two passed the pre-set criteria: hubness (k-skewness
  above a null model) and antihubs, both for Euclidean kNN, both pointing to a hubness-reducing search.
  The hubness rule fires on only 3 of 35 sets, so that evidence is thin; the antihub rule fires on 8 and
  the fix helped significantly on 5. There was no correction for the 23 primary tests, where about one
  false pass is expected by chance. The report recommends nothing else.
- **Hubness reduction helps little under cosine.** On the E1 sets the best change against raw cosine
  kNN was +0.005 (mutual proximity, CIFAR-100); it lost 0.009 on Pets and 0.028 on the synthetic set
  (E1 addendum in `docs/BENCHMARK_E6_E7.md`). In E4 the best reduction was significantly better on 10
  of 35 sets under cosine kNN and 14 of 35 under Euclidean kNN (median +0.4 points), worse on 6 and 3.
- **The diagnostics have limits** ([`docs/BENCHMARK_DIAGNOSTICS.md`](docs/BENCHMARK_DIAGNOSTICS.md),
  E3 and E5). The k-NN intrinsic-dimension estimators read 14 to 29% low at ID 20, and their
  intervals cover sampling error only. Where the null model is not calibrated (rows normalised
  upstream, some statistics under cosine, neighbour consistency above d = 256) the report marks
  the statistic as uncalibrated and draws no recommendation from it.

**Where gains were measured: the supervised refiner**

- With class labels (E8, [`docs/BENCHMARK_SUPERVISED.md`](docs/BENCHMARK_SUPERVISED.md)), SupCon on a
  residual adapter had higher kNN accuracy than the raw embeddings on 4 of 6 sets and lower on none:
  CIFAR-100/DINOv2 0.879 to 0.900, Pets/ResNet-50 0.901 to 0.921, Banking77/MPNet 0.931 to 0.938,
  STL-10/CLIP 0.983 to 0.986, digits tied (0.981 against 0.978). The gain on real data is +0.003 to
  +0.021. On a synthetic set with large nuisance directions it is +0.185. Against LDA it won on 3 of
  6 sets (+0.002 to +0.013) and lost on none.
- Limits of that result: with 5 labels per class, LDA was ahead of the best refiner by 0.04 to 0.08; with
  1 label per class the refiner fell below raw on Banking77 and STL-10. Trained on 20 coarse labels
  and scored on the 100 fine CIFAR-100 classes, SupCon dropped to 0.645 (raw 0.879).
- With continuous targets, Rank-N-Contrast on an MLP beat raw and PCA on ridge R^2 on all 4 sets
  (+0.023 to +0.574 against raw). With 200 or 1,000 training rows it lost to raw on 6 of 8 sets.

Not yet measured: the selection procedure described below was not benchmarked on its own (it
reuses E4's findings). Time and memory for n up to 1e6 and d up to 4,096 are in the manual
(section 3.5, experiment E9): the analyzer takes minutes at any n; selection with labels trains
the refiner on CPU and took 1.8 to 9.7 hours at n = 1e6.

## What the defaults do

`EmbedKit()` chooses its output by validation, not by diagnosis (`strategy="select"`).

- **Without labels** it returns the input centred on the training mean and scaled to unit L2 norm
  (`CenterNormalize`). No label-free score can say that a transform helps (kNN overlap with the input
  ranks the unchanged input first), and centring plus L2 was the most consistent small gain in E4. For a
  cosine search it changes nothing beyond removing the mean.
- **With labels** it holds out 20% of the rows, fits `Identity`, `CenterNormalize`, PCA at 64, 128 and
  256 (below the input dimension, L2-normalised) and, in supervised or regression mode, the refiner, and
  scores each by 10-NN accuracy (classes) or 10-NN R^2 (continuous targets). A candidate replaces
  `Identity` only if a paired bootstrap interval of its gain on the held-out rows lies above 0. Otherwise
  the output is the input unchanged. The winner is refitted on all rows.
- `mode="auto"` is self-supervised without labels and supervised with them; continuous targets need
  `mode="regression"`. `strategy="fixed"` (or any `method` other than `"auto"`) builds one configuration
  as in 0.5.0: `method="pca"`, or the refiner with `method="refiner"`.

The 0.5.0 behaviour is in [`docs/MIGRATION.md`](docs/MIGRATION.md) with every other breaking change
since 0.1.2.

## Installation

Python 3.10 or later.

```bash
pip install git+https://github.com/fagonzalezo/embedding-kit.git
# or: uv add git+https://github.com/fagonzalezo/embedding-kit.git
```

| Extra | Installs | When to use |
|---|---|---|
| `examples` | `torchvision` | The image-classification example (`examples/04_image_classification.*`) |
| `faiss` | `faiss-cpu` | Faster exact kNN on large data: `knn(..., backend="faiss")` or `EMBEDKIT_KNN_BACKEND=faiss` |
| `dev` | pytest, ruff, mypy | Running the tests |

Install an extra with `pip install "embedkit[faiss] @ git+https://github.com/fagonzalezo/embedding-kit.git"`.
FAISS is not a default: next to PyTorch it can load two OpenMP runtimes and abort the process
(`OMP: Error #15`). The default scikit-learn backend returns the same exact neighbours.

## Quickstart

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

X = np.load("embeddings.npy")   # (n_samples, n_features)
y = np.load("labels.npy")       # class labels, only for the supervised parts

# Analysis: statistics, null-model z-scores, descriptive properties, recommendations
report = EmbedKitAnalyzer(k=10).fit(X)
report.print_summary()
print(report.recommendations)   # at most the two validated rules, Euclidean metric only
df = report.to_dataframe()

# Without labels: centred and L2-normalised, nothing trained
Z = EmbedKit().fit_transform(X)

# With labels: selection among candidates, Identity unless one is significantly better
ek = EmbedKit(epochs=50).fit(X, y)
Z = ek.transform(X)
print(ek.selected_)                  # "identity", "center_l2", "pca-128", "refiner-128", ...
print(ek.selection_report_)          # one row per candidate: score, gain over Identity, interval
```

`selection_report_` columns: `candidate`, `dim`, `score` (held-out), `diff_vs_identity`, `diff_lo`,
`diff_hi` (95% bootstrap interval), `beats_identity`, `selected`.

Fixed configurations, and a refiner trained without labels (E1, E6, E7: not better than PCA):

```python
from embedkit import EmbedKit

Z = EmbedKit(strategy="fixed", target_dim=64).fit_transform(X)                 # PCA to 64, L2
Z = EmbedKit(method="refiner", target_dim=64, epochs=100).fit_transform(X)     # contrastive refiner
Z = EmbedKit(mode="regression", epochs=50).fit_transform(X, y=y.astype(float))  # continuous targets
```

Hubness-reducing search, for Euclidean kNN when the report recommends it (a secondary distance, not a
transform: fit it on the gallery, then query):

```python
from embedkit.hubness_reduction import MutualProximity

Z_train, Z_test = X[:800], X[800:]
mp = MutualProximity(metric="euclidean").fit(Z_train)
dist, idx = mp.kneighbors(Z_test, n_neighbors=10)   # neighbours among the Z_train rows
```

Scoring a transform against baselines on held-out data, with paired bootstrap intervals:

```python
from embedkit.baselines import Identity, PCAReduce
from embedkit.evaluation import compare, evaluate_transform

splits = dict(X_train=X[:800], X_test=X[800:], y_train=y[:800], y_test=y[800:], seeds=[0])
res = [evaluate_transform(lambda s: Identity(), method="raw", **splits),
       evaluate_transform(lambda s: PCAReduce(32), method="pca-32", **splits)]
print(compare(res, reference="raw"))
```

More patterns (low-level `Trainer`, custom augmentations, pipelines) are in
[`examples/`](examples/).

## What is in the library

- **Analysis**: `IntrinsicDimensionEstimator` (TwoNN, MLE, lPCA, DANCo, CorrInt, MOM, FisherS, ESS, each
  reported with a bootstrap interval, not combined), `HubnessAnalyzer`, `DistanceConcentration`,
  `IsotropyAnalyzer`, `NeighborConsistency`, `UniformityScore`, `KernelDiagnostics`, null-model
  calibration, and `EmbedKitAnalyzer` to run them together.
- **Baselines** (`embedkit.baselines`): `Identity`, `CenterNormalize`, `Standardize`, `PCAReduce`,
  `ZCAWhiten`, `AllButTheTop`, `ConceptorNegation`, `RandomProjection`.
- **Hubness reduction** (`embedkit.hubness_reduction`): `LocalScaling`, `MutualProximity`, `CSLS`,
  `DisSimLocal`.
- **Refiner**: `EmbeddingRefiner` (linear, residual or MLP), `Trainer`, augmentations (`GaussianNoise`,
  `FeatureDropout`, `EmbeddingMixup`, `KNNPairs`, `FeatureMasking`, `IdentityPair`,
  `CompositeAugmentation`) and losses (`NTXentLoss`, `AlignUniformLoss`, `SupConLoss`, `TripletLoss`,
  `RankNContrastLoss`, `MatryoshkaLoss`, `CombinedLoss`).
- **Evaluation** (`embedkit.evaluation`): kNN accuracy, linear probe, recall@k, trustworthiness, CKA, NC1
  and others; `evaluate_transform` and `compare`.
- **Visualization**: `AnalysisPlotter`, `EmbeddingVisualizer`.
- Inputs: `np.ndarray` or `torch.Tensor`, validated once at each entry point (NaN or infinite values
  raise; constant columns and duplicate rows warn).

## Documentation

| Document | Description |
|---|---|
| [`docs/manual.md`](docs/manual.md) | Concepts manual: every analyzer, the improvement pipeline, the high-level API |
| [`docs/MIGRATION.md`](docs/MIGRATION.md) | Every breaking change since 0.1.2, with what to do |
| [`CHANGELOG.md`](CHANGELOG.md) | Changes with their reasons and measurements |
| [`skills/embedkit/references/metric_guide.md`](skills/embedkit/references/metric_guide.md) | Per-metric reading, and which statistics the report judges |
| [`skills/embedkit/references/workflow_guide.md`](skills/embedkit/references/workflow_guide.md) | HuggingFace, PyTorch loops, scikit-learn pipelines, retrieval, ONNX export |
| `docs/BENCHMARK_*.md` | Benchmark outcomes: [E1, E2](docs/BENCHMARK_E1_E2.md), [E6, E7](docs/BENCHMARK_E6_E7.md), [E8](docs/BENCHMARK_SUPERVISED.md), [E3, E5](docs/BENCHMARK_DIAGNOSTICS.md); E4 in [`benchmarks/results/e4/REPORT.md`](benchmarks/results/e4/REPORT.md) |

## Claude Code skill

`skills/embedkit/` is a [Claude Code](https://claude.ai/code) skill with the API reference and the
metric and workflow guides. To install it globally:

```bash
git clone https://github.com/fagonzalezo/embedding-kit.git
cp -r embedding-kit/skills/embedkit ~/.claude/skills/        # or ln -s to keep it in sync
```

Inside this repository Claude Code picks it up without installation.
