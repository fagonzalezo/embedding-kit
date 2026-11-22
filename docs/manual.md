# EmbedKit User Manual

EmbedKit improves embeddings you already have: the output of a pretrained text or image encoder,
for example. It fits a transform on them and applies it to new rows. The transforms are
centring and L2 normalisation, PCA, and a small network (the *refiner*) trained with your labels.
It also describes the geometry of an embedding space (intrinsic dimension, hubness, isotropy and
more), to help you understand and compare encoders.

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

X = np.load("embeddings.npy")          # (n_samples, n_features)
y = np.load("labels.npy")              # optional: class labels or continuous targets

# Without labels: centred and L2-normalised, nothing trained
Z = EmbedKit().fit_transform(X)

# With labels: the transform that beats the input on held-out rows, or the input itself
ek = EmbedKit().fit(X, y)              # mode="regression" for continuous targets
print(ek.selected_)                    # "identity", "center_l2", "pca-128", "refiner-128", ...
Z = ek.transform(X)

# Describe the space; no transform is chosen from this
report = EmbedKitAnalyzer().fit(X)
report.print_summary()
```

**Contents**

1. [Design philosophy](#1-design-philosophy)
2. [Which method for your situation](#2-which-method-for-your-situation)
3. [Improving embeddings with `EmbedKit`](#3-improving-embeddings-with-embedkit)
4. [Tuning a refiner on your data](#4-tuning-a-refiner-on-your-data)
5. [Training-free baselines](#5-training-free-baselines)
6. [Hubness-reducing neighbour search](#6-hubness-reducing-neighbour-search)
7. [Building a custom refiner](#7-building-a-custom-refiner)
8. [Evaluating on your own data](#8-evaluating-on-your-own-data)
9. [Diagnostics: describing an embedding space](#9-diagnostics-describing-an-embedding-space)
10. [Visualization](#10-visualization)
11. [Time and memory](#11-time-and-memory)
12. [Troubleshooting](#12-troubleshooting)

Upgrading from an earlier version: [`MIGRATION.md`](MIGRATION.md).

---

## 1. Design philosophy

**Fit on your data, apply to new data.** Every transform follows the scikit-learn pattern:
`fit` learns from training rows (a mean, principal directions, network weights), and `transform`
applies the same map to any rows with the same number of features, including queries and data
that arrive later. Fit on rows that represent the data you will transform, and never on your test
rows.

**A transform is judged by the task, on held-out rows.** An embedding is good or bad for a use:
nearest-neighbour search, a classifier, a regressor. EmbedKit therefore scores transforms by
what you do with the embedding (by default, 10-nearest-neighbour accuracy for classes and
10-nearest-neighbour R² for continuous targets) on rows the transform did not see.

**Keep the input unless a change wins with confidence.** With labels, `EmbedKit` compares the
unchanged input with a few candidates and replaces it only when a candidate is better on the
held-out rows *and* the gain is clearly above noise: a paired bootstrap interval of the gain
lies above 0. Otherwise it returns the input unchanged. Keeping your embedding is a valid
result: it means the data do not show that any candidate helps.

**Without labels, make a small safe change and train nothing.** Without labels there is no way to
measure whether a transform helps your task, so `EmbedKit` only centres the rows and scales them
to unit length. It does not reduce the dimension or train anything unless you ask for it.

**Diagnostics describe; they do not prescribe.** The analysis tools (section 9) tell you how an
embedding space is shaped and how it compares with random data of the same size. A statistic
outside its usual range does not mean that a fix will help, so no default reads the analysis.
The one recommendation the report makes is to check a hubness-reducing search when you search
with Euclidean distance.

**One parameter says what to build.** `method` is `"auto"` (let EmbedKit choose), or one of
`"center_l2"`, `"pca"`, `"refiner"` (build that one). After `fit`, `selected_` always names what
was built.

The benchmarks behind these defaults are summarised in the [README](../README.md) and written up
in [`docs/benchmarks/`](benchmarks/).

---

## 2. Which method for your situation

| Your situation | What to do | Section |
|---|---|---|
| No labels; you want a better embedding for search or clustering | `EmbedKit().fit_transform(X)` (centre + L2) | [3.2](#32-without-labels) |
| No labels; you need a smaller embedding | `EmbedKit(method="pca", target_dim=k)`, with `k` as large as your budget allows | [3.6](#36-output-dimension) |
| Class labels | `EmbedKit().fit(X, y)`, then read `selected_` and `selection_report_` | [3.3](#33-with-labels-the-validated-selection) |
| One continuous target per row | `EmbedKit(mode="regression").fit(X, y)` | [3.4](#34-modes) |
| Several continuous targets per row, `y` of shape `(n, m)` | The same, `mode="regression"`. Standardise target columns that are in different units; flatten higher-rank targets such as `(n, h, w)` to `(n, h*w)`; with many columns, consider reducing `y` first | [3.4](#34-modes) |
| Few labels per class (about 5 or fewer) | Run the default, and also compare with LDA, which is often better here | [4.2](#42-check-that-knn-is-your-criterion) |
| Labels coarser than the classes you care about | A refiner with `SupConLoss(spread_weight=0.5)` | [4.2](#42-check-that-knn-is-your-criterion) |
| The embedding feeds a linear classifier, not a neighbour search | Run the default, then confirm with a linear probe | [4.2](#42-check-that-knn-is-your-criterion) |
| You want to try other refiner settings | Tune on a held-out split, against the raw embedding | [4](#4-tuning-a-refiner-on-your-data) |
| You need several output sizes from one model | `transform(X, dim=m)` on PCA, or `nested_dims` on a refiner | [3.6](#36-output-dimension) |
| Nearest-neighbour search with Euclidean distance; some items are retrieved far too often | A hubness-reducing search | [6](#6-hubness-reducing-neighbour-search) |
| A retrieval system with queries and a document collection | Fit on the collection (or a training corpus), transform both queries and documents with the same object | [3.8](#38-applying-to-new-data-saving-and-loading) |
| Very many rows (hundreds of thousands or more) | The defaults already cap the selection at 50,000 rows; use a GPU for the refiner | [3.7](#37-large-data-devices-and-reproducibility), [11](#11-time-and-memory) |
| You want to understand or compare encoders, layers or data sets | `EmbedKitAnalyzer` | [9](#9-diagnostics-describing-an-embedding-space) |

---

## 3. Improving embeddings with `EmbedKit`

### 3.1 The four methods

`EmbedKit` fits one transform on the input embeddings and applies it with `transform`. The
parameter `method` says what it builds:

| `method` | Builds | Needs labels |
|---|---|---|
| `"auto"` (default) | Without labels, centre + L2. With labels, the candidate that wins a validated comparison (3.3), or the input unchanged | No |
| `"center_l2"` | The rows centred on the training mean and scaled to unit L2 norm | No |
| `"pca"` | The top `target_dim` principal directions, then L2 normalisation; nothing is trained | No |
| `"refiner"` | A trained `EmbeddingRefiner`, configured by `mode` (3.5) | Recommended |

The interface is scikit-learn's: `fit(X, y=None)`, `transform(X, dim=None)`,
`fit_transform(X, y=None, dim=None)`, plus `save` and `EmbedKit.load`. `X` is a NumPy array or a
PyTorch tensor of shape `(n_samples, n_features)`; `transform` returns the same kind.

After `fit`, `selected_` names what was built: `"identity"` (the input unchanged),
`"center_l2"`, `"pca-<k>"` or `"refiner-<k>"`, with `k` the output dimension.

### 3.2 Without labels

```python
from embedkit import EmbedKit

ek = EmbedKit()
Z = ek.fit_transform(X)
print(ek.selected_)          # "center_l2"
```

The output is the input minus the training mean, with every row scaled to unit length. Nothing
is trained and the dimension is unchanged. What it changes depends on how you compare
embeddings:

- **Euclidean distance:** distances now ignore row lengths and the shared offset that many
  encoders add to every vector. This is usually a small improvement.
- **Cosine similarity:** cosine already ignores length, so the only change is the removal of the
  mean direction.
- **Dot product:** on unit-length rows the dot product equals the cosine, so the output can go
  straight into a vector database that uses inner product.

If you need a smaller embedding, use `method="pca"` (3.6). A refiner trained without labels
(`method="refiner"` with no `y`) is available, but it is rarely better than PCA at the same
dimension and costs far more to fit; prefer PCA.

### 3.3 With labels: the validated selection

```python
ek = EmbedKit().fit(X_train, y_train)        # class labels
print(ek.selected_)
print(ek.selection_report_[["candidate", "dim", "score", "diff_vs_identity",
                            "diff_lo", "diff_hi", "selected"]])
Z_test = ek.transform(X_test)
```

With labels, `fit` runs a small, self-contained experiment:

| Step | What happens |
|---|---|
| Split | `val_fraction` (default 0.2) of the rows is held out, stratified by class when every class has at least 2 rows and there are fewer classes than held-out rows. The rest fits the candidates. |
| Candidates | `identity` (the input as given); `center_l2`; PCA at 64, 128 and 256 followed by L2 normalisation (only those below the input dimension and the rank of the data; the `target_dim` you pass replaces these three); and, with `mode="supervised"` (the default with class labels) or `"regression"`, a refiner at `target_dim` (`"auto"`: `min(128, d // 2)` for input dimension `d`). |
| Criterion | 10-NN accuracy for class labels, 10-NN regression R² for continuous targets, measured on the held-out rows with `selection_metric` (`"cosine"` by default, or `"euclidean"`). |
| Decision | A candidate can replace `identity` only when the 95% paired bootstrap interval of its per-row gain over `identity` (2,000 resamples of the held-out rows) lies above 0. Among the candidates that pass, the best score wins. If none passes, the result is `identity`. |
| Refit | The winner is fitted again on all the rows. |
| Large inputs | Above `selection_max_rows` rows (default 50,000), the split and the scores use a random subsample of that size, stratified by class; only the winner is fitted on all the rows. |

**Set `selection_metric` to how you will search.** If your application compares embeddings with
Euclidean distance, pass `selection_metric="euclidean"`, so the candidates are judged the way you
use them.

**Reading `selection_report_`.** One row per candidate:

| Column | Meaning |
|---|---|
| `candidate`, `dim` | Name and output dimension |
| `score` | Held-out 10-NN accuracy or R² |
| `diff_vs_identity`, `diff_lo`, `diff_hi` | Mean per-row gain over `identity` and its 95% bootstrap interval |
| `beats_identity` | `diff_lo > 0` |
| `selected` | The winner |

`selection_report_.attrs` records the criterion, `k`, the metric, the numbers of fit and held-out
rows, the number of resamples, the number of input rows (`n_rows`) and the number of rows the
selection used (`n_selection_rows`).

**What the outcome tells you:**

| Outcome | What it means | What to do next |
|---|---|---|
| A refiner was selected (`refiner-128`) | The refiner beats the input clearly | Use it. Tuning (section 4) may add a little |
| PCA or `center_l2` was selected | A training-free change is enough | Use it; nothing to train or tune |
| `identity` kept, every candidate's `diff_vs_identity` at or below 0 | Nothing helps on this data by this criterion | Keep your embedding |
| `identity` kept, a candidate has `diff_vs_identity` above 0 but `diff_lo` below 0 | Possibly a gain, but within noise | Hold out more labelled rows so the interval narrows, or tune that candidate (section 4) |

**Check the result on a test split.** The selection guarantees that the output is not worse than
the input *on its own held-out rows*. Before you rely on it, score it once on a test split that
played no part in the choice (section 8), with the measure you care about.

With fewer than 20 rows to fit or to score (about 100 labelled rows or fewer at the default
`val_fraction`), `fit` warns and returns `identity`. Pass `method="center_l2"`, `"pca"` or
`"refiner"` to build one anyway.

### 3.4 Modes

`mode` tells `EmbedKit` what the labels are.

| `mode` | Labels | Selection criterion | Refiner trained |
|---|---|---|---|
| `"auto"` (default) | None: self-supervised. Labels: class labels (`"supervised"`) | — | as below |
| `"supervised"` | Class labels: integers, strings or booleans, shape `(n,)` | 10-NN accuracy | SupCon on a residual adapter |
| `"regression"` | Continuous targets, shape `(n,)` or `(n, m)` | 10-NN R² (averaged over the `m` outputs) | Rank-N-Contrast on an MLP |
| `"self_supervised"` | None, or labels used only to choose among the training-free candidates | 10-NN accuracy | No refiner is compared |

```python
# Classes
ek = EmbedKit().fit(X, y)

# One continuous target
ek = EmbedKit(mode="regression").fit(X, y.astype(float))

# Several continuous targets per row: Y has shape (n, m)
ek = EmbedKit(mode="regression").fit(X, Y)

# Use labels to choose among centre + L2 and PCA only; train nothing
ek = EmbedKit(mode="self_supervised").fit(X, y)
```

In every mode other than `"regression"`, float labels that are not whole numbers raise an error
that asks for `mode="regression"`: EmbedKit does not guess that a float column is a regression
target.

**Preparing regression targets.**

- *Several targets per row.* Pass them as one array of shape `(n, m)`. The refiner learns from
  the Euclidean distance between target vectors, and the selection scores the R² averaged over
  the `m` outputs.
- *Different units.* Standardise the target columns first (for example with scikit-learn's
  `StandardScaler`). Otherwise the column with the largest values dominates both the training
  signal and the score.
- *Structured targets.* A target that is itself an array per row, of shape `(n, h, w)`, raises an
  error; flatten it with `Y.reshape(len(Y), -1)`. The distance between flattened targets is the
  same as between the originals.
- *Many target columns.* Distances between long target vectors become similar to each other,
  which weakens the refiner's training signal. If `m` is large, consider reducing `Y` (for
  example to its top principal components) and check the result on held-out rows.
- *Small data sets.* With 1,000 training rows or fewer, a regression refiner rarely beats the
  raw embedding; a fixed regression refiner warns. The selection still protects you, because the
  refiner must beat the input on held-out rows to be chosen.

### 3.5 Fixed configurations

`method="center_l2"`, `"pca"` or `"refiner"` builds that one configuration, with no comparison.
`selected_` names it as the selection would, and `selection_report_` is `None`. Use a fixed
configuration when you already know what you want, when you have too few labelled rows to
select, or inside your own evaluation loop (section 4).

```python
ek = EmbedKit(method="center_l2").fit(X)
ek = EmbedKit(method="pca", target_dim=64).fit(X)
ek = EmbedKit(method="refiner", target_dim=128).fit(X, y)
print(ek.selected_, ek.architecture_, ek.monitor_)    # refiner-128 residual val_knn_accuracy
```

**The refiner's defaults by mode.** These are the `"auto"` values; any of them can be set
explicitly.

| Setting | `"supervised"` | `"regression"` | `"self_supervised"` |
|---|---|---|---|
| `architecture` | `"residual"` | `"mlp"` | `"linear"` |
| `augmentation` | `KNNPairs(k=5)` | `GaussianNoise(std=0.05)` | `KNNPairs(k=5)` |
| `loss` | `SupConLoss(0.07)` | `RankNContrastLoss(0.07)` | `NTXentLoss(0.07)` + 0.5 `AlignUniformLoss` |
| `monitor` | `val_knn_accuracy` | `val_ridge_r2` | `k_skewness` (logged only) |
| `early_stopping_patience` | 10 evaluations | 10 evaluations | `None` (fixed budget) |

With labels, 10% of the training rows are held out (`val_split`) to watch the monitor, training
stops after 10 evaluations (every `eval_every = 5` epochs) without improvement, and the best
weights are restored. `epochs` (default 100) is then an upper bound. Without labels the refiner
trains for exactly `epochs` epochs: there is no label-free way to tell when to stop. After `fit`,
`architecture_`, `monitor_`, `early_stopping_patience_` and `nested_dims_` hold the resolved
values.

`augmentation`, `loss`, `nested_dims` and an `architecture` other than `"auto"` or `"linear"`
configure the refiner.
Setting them when no refiner is fitted or compared (for example with `mode="self_supervised"`
and `method="auto"`) raises a `ValueError`, so that a setting is never silently ignored.

### 3.6 Output dimension

`target_dim` sets the output dimension of `"pca"` and `"refiner"`.

- **Pass it explicitly.** With `method="pca"` or `"refiner"`, `target_dim="auto"` uses the
  analysis report's `suggested_target_dim` (1.5 times an intrinsic-dimension estimate,
  section 9.1). That value is often low, and reducing to it can cost accuracy.
- **Keep it as high as your budget allows.** Reducing the dimension saves memory and search
  time, but it can lose information. With labels, let the selection compare sizes (PCA at 64,
  128 and 256 are candidates by default); without labels you cannot measure the loss, so be
  conservative.
- **In the selection**, the refiner candidate uses `min(128, d // 2)` when `target_dim="auto"`;
  a `target_dim` you pass replaces both the PCA sizes and the refiner size.

**Several sizes from one model.** `transform(X, dim=m)` keeps the first `m` output coordinates
and renormalises them to unit length (the input is only truncated when `identity` was selected;
`dim` is meant for PCA and nested refiners).

- On PCA, the prefixes are the top `m` principal directions, so one PCA fit serves every size:

  ```python
  ek = EmbedKit(method="pca", target_dim=256).fit(X)
  Z64 = ek.transform(X, dim=64)               # PCA at 64, renormalised
  ```

- On a refiner, the prefixes are useful only if they were trained to be: pass `nested_dims`.
  `"auto"` trains the powers of two from 32 below `target_dim`, plus the full output; a sequence gives the sizes
  explicitly. It costs about 1.5 times one training and gives prefixes as good as separate
  models of those sizes.

  ```python
  ek = EmbedKit(method="refiner", target_dim=256, nested_dims="auto").fit(X, y)
  Z64 = ek.transform(X, dim=64)               # prefixes 32, 64, 128, 256 were trained
  ```

### 3.7 Large data, devices and reproducibility

- **Many rows.** The selection runs on at most `selection_max_rows` rows (default 50,000), so
  its time stops growing with `n`; only the winner is fitted on all the rows. Do not set the cap
  much below 10,000: the choice becomes unreliable. `selection_max_rows=None` selects on all the
  rows, which is much slower on large data. Section 11 has measured times.
- **Devices.** `device=None` uses CUDA if available, then Apple MPS, then the CPU. The refiner
  trains on that device, and PCA and refiner outputs are computed on it; centring runs in NumPy.
- **Reproducibility.** `random_state` (default 42) fixes the selection split, the bootstrap and
  the refiner's initialisation and data order. EmbedKit does not touch the global random state of
  NumPy or PyTorch. Results on GPU can differ slightly from CPU results.
- **Input checks.** `fit` needs at least 5 rows. NaN or infinite values raise an error; constant
  columns and duplicate rows give a warning. `transform` checks the number of features.

### 3.8 Applying to new data, saving and loading

Fit once, then transform every set of rows with the same fitted object:

```python
ek = EmbedKit().fit(X_train, y_train)
Z_train = ek.transform(X_train)
Z_test = ek.transform(X_test)
```

- **Retrieval.** Transform queries and the document collection with the same fitted object.
  Fit on the collection or on a training corpus, not on the queries you will evaluate.
- **Unit length.** The outputs of `center_l2`, PCA and refiners have unit length, so cosine
  similarity, dot product and Euclidean distance rank neighbours the same way. When `identity` is
  selected the rows come back unchanged; normalise them yourself if your system expects unit
  vectors.
- **When the data change.** A fitted transform encodes the training data (their mean, their
  principal directions, the labels the refiner learned). If new data come from a different
  source or encoder, fit again.

```python
ek.save("my_model/")
ek2 = EmbedKit.load("my_model/")            # device=None: CUDA, then MPS, then CPU
Z_new = ek2.transform(X_new)
```

`save` writes `config.json` (the configuration, `selected_` and, when a selection ran,
`selection_report_`) plus the fitted state: `model.pt` for PCA and refiners, `arrays.npz` for
`identity` and `center_l2` (the training mean). `load` restores `mode_`, `method_`, `selected_`
and `selection_report_`.

### 3.9 Parameter reference

`EmbedKit(mode="auto", method="auto", augmentation="auto", loss="auto", target_dim="auto",
hidden_dim=256, n_layers=2, architecture="auto", epochs=100, batch_size=256, optimizer="adam",
lr=3e-4, weight_decay=1e-4, scheduler="cosine", eval_every=5, early_stopping_patience="auto",
monitor="auto", val_split=None, device=None, id_methods=None, random_state=42,
nested_dims=None, val_fraction=0.2, selection_metric="cosine", selection_max_rows=50000)`

| Group | Parameters |
|---|---|
| What to build | `method`, `mode`, `target_dim` |
| Selection | `val_fraction`, `selection_metric`, `selection_max_rows` |
| Refiner model | `architecture` (`"residual"`, `"mlp"`, `"linear"`), `hidden_dim`, `n_layers`, `nested_dims` |
| Refiner training | `augmentation`, `loss` (objects from section 7), `epochs`, `batch_size`, `optimizer` (`"adam"`, `"sgd"`), `lr`, `weight_decay`, `scheduler` (`"cosine"`, `"step"`, `"plateau"`, `None`) |
| Early stopping | `monitor`, `early_stopping_patience` (in evaluations), `eval_every` (epochs), `val_split` (`None`: 0.1 with early stopping, else 0) |
| Other | `device`, `random_state`, `id_methods` (estimators for `target_dim="auto"`) |

Attributes after `fit`: `mode_`, `method_`, `selected_`, `selection_report_` (only when a
selection ran), `analysis_report` (only for `"pca"` or `"refiner"` with `target_dim="auto"`), and
for a refiner `architecture_`, `monitor_`, `early_stopping_patience_`, `nested_dims_`.

---

## 4. Tuning a refiner on your data

With labels, `EmbedKit().fit(X, y)` already compares one refiner with the raw embedding, centring
and PCA. Its training settings are general-purpose defaults, never tuned for a particular data
set. This section is for the next question: would a different refiner do better on your data?

### 4.1 Start from the default

Run the default and read its decision (3.3). A refiner is worth tuning when it was selected, or
when it was not selected but its `diff_vs_identity` is above 0. If every candidate is at or below
the input, tuning is unlikely to help.

### 4.2 Check that kNN is your criterion

The default scores candidates by 10-NN accuracy (or 10-NN R²). If you use the embedding
differently, judge the refiner the way you use it:

- **A linear classifier downstream.** A refiner can raise kNN accuracy and slightly lower the
  accuracy of a linear classifier, mostly on embeddings whose linear accuracy is already very
  high. Score candidates with `"linear_probe_c1"` as well as `"knn_accuracy"` (section 8).
- **Few labels.** With about 5 labels per class or fewer, linear discriminant analysis
  (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`)
  is often better than any refiner, and with 1 label per class a refiner can fall below the raw
  embedding. Compare with LDA, not only with the raw embedding.
- **Small regression sets.** With 1,000 training rows or fewer, a regression refiner rarely
  beats the raw embedding.
- **Labels coarser than the classes you care about** (trained on categories, used to find items
  of the same sub-category). `SupConLoss` pulls each class towards a single point and erases the
  structure inside it. `SupConLoss(spread_weight=0.5)` keeps that structure.
- **No labels.** Do not train a refiner; use centre + L2, or PCA for a smaller embedding.

### 4.3 What is worth tuning

| Setting | Default | Worth trying |
|---|---|---|
| `target_dim` | `min(128, d // 2)` in the selection | 64, 256 (below `d`) |
| `loss` | `SupConLoss(0.07)`; regression `RankNContrastLoss(0.07)` | temperature 0.1 to 0.2; `spread_weight` 0.25 to 0.5; `TripletLoss`; SupCon + 0.5 `AlignUniformLoss` |
| `architecture` | `"residual"`; regression `"mlp"` | `"mlp"`, `"linear"` |
| `lr`, `epochs` | `3e-4`, 100 | `1e-4` to `1e-3`; 50 to 200 (with early stopping, `epochs` is an upper bound) |
| `augmentation` | `KNNPairs(5)`; regression `GaussianNoise(0.05)` | `KNNPairs(k=10)`; `GaussianNoise(0.1)` |

The losses and augmentations are described in section 7. The residual architecture starts from
PCA and adds a small learned correction, so it is the safest with few labels; the MLP is more
flexible and needs more labels.

### 4.4 Tuning on a held-out split

Four rules keep the comparison honest:

1. **Three splits.** Fit on one part of the training rows, choose on another (the validation
   rows), and score the chosen configuration once on a test split that played no part in the
   choice. With many configurations, the winner's validation score is optimistic: part of its
   lead is luck. Only the test score estimates what you will get.
2. **One factor at a time**, starting from the default, so that a gain can be traced to its cause.
3. **Several seeds** for each trained configuration, so that a difference is not one lucky run.
4. **Against the raw embedding, with an interval.** Keep a configuration only if its paired
   interval against raw lies above 0; otherwise keep the raw embedding, as the default does.

`evaluate_transform` and `compare` (section 8) do the bookkeeping. The example compares the
default refiner with two output dimensions, four losses, another architecture and another
learning rate, with the raw embedding and PCA as references:

```python
from sklearn.model_selection import train_test_split

from embedkit import EmbedKit
from embedkit.baselines import Identity, PCAReduce
from embedkit.evaluation import compare, evaluate_transform
from embedkit.improvement.losses import AlignUniformLoss, CombinedLoss, SupConLoss, TripletLoss

# 1. Three splits. The test split (X_test, y_test) is used once, at the end.
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                              stratify=y_train, random_state=0)


def refiner(mode="supervised", target_dim=128, loss=None, **kw):
    """A make_transform for evaluate_transform: a new EmbedKit, and a new loss, per seed."""
    def make(seed):
        return EmbedKit(mode=mode, method="refiner", target_dim=target_dim,
                        loss=loss() if loss else "auto", random_state=seed, **kw)
    return make


# 2. The default refiner, then one factor changed at a time.
grid = {
    "default":        refiner(),                                  # SupCon(0.07), residual, 128
    "dim-64":         refiner(target_dim=64),
    "dim-256":        refiner(target_dim=256),                    # needs d > 256
    "supcon-t0.2":    refiner(loss=lambda: SupConLoss(temperature=0.2)),
    "supcon-spread":  refiner(loss=lambda: SupConLoss(temperature=0.07, spread_weight=0.25)),
    "supcon+uniform": refiner(loss=lambda: CombinedLoss([(SupConLoss(0.07), 1.0),
                                                         (AlignUniformLoss(), 0.5)])),
    "triplet":        refiner(loss=lambda: TripletLoss(margin=0.2, mining="semi-hard",
                                                       distance="cosine")),
    "mlp":            refiner(architecture="mlp"),
    "lr-1e-3":        refiner(lr=1e-3),
}

val = dict(X_train=X_fit, X_test=X_val, y_train=y_fit, y_test=y_val, metrics=["knn_accuracy"])
results = [evaluate_transform(lambda s: Identity(), method="raw", seeds=[0], **val),
           evaluate_transform(lambda s: PCAReduce(128), method="pca-128", seeds=[0], **val)]
for name, make in grid.items():
    results.append(evaluate_transform(make, method=name, seeds=[0, 1, 2],
                                      fit_with_labels=True, **val))
table = compare(results, reference="raw")
print(table[["method", "mean", "sd", "diff", "diff_lo", "diff_hi"]]
      .sort_values("mean", ascending=False).to_string(index=False))

# 3. Keep the best configuration only if its interval against raw is above 0 on the
#    validation rows; refit it on all the training rows and score it once on the test split.
passed = table[(table.method != "raw") & (table.diff_lo > 0)]
best = passed.sort_values("mean", ascending=False).method.iloc[0] if len(passed) else "raw"
test = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            metrics=["knn_accuracy", "linear_probe_c1"])
final = [evaluate_transform(lambda s: Identity(), method="raw", seeds=[0], **test)]
if best in grid:
    final.append(evaluate_transform(grid[best], method=best, seeds=[0], fit_with_labels=True,
                                    **test))
elif best != "raw":
    final.append(evaluate_transform(lambda s: PCAReduce(128), method=best, seeds=[0], **test))
print(best)
print(compare(final, reference="raw")[["method", "metric", "mean", "diff", "diff_lo", "diff_hi"]])
```

Reading the output:

- **The validation table** has one row per configuration: `mean` and `sd` over the seeds, and
  `diff` with its interval against raw. If every interval includes 0, step 3 keeps the raw
  embedding. That is a valid outcome: the data do not show that any of these refiners helps.
- **The test table** is the result to report. It scores the linear probe too, since a refiner can
  raise kNN accuracy and lower probe accuracy at the same time.
- **`knn_accuracy` in `evaluate_transform` uses k = 5**; the default selection uses k = 10. Use the
  metric that matches your downstream use (`"linear_probe_c1"`, `"knn_accuracy_euclidean"`,
  `"recall_at_10"`; section 8.1 lists them) and choose on that one.
- **Cost.** Nine configurations at three seeds are 27 refiner trainings, each about as long as
  the refiner candidate of one default `fit`. If you have more than 10,000 to 20,000 training
  rows, tune on a subsample of that size; the choice is just as good.

For continuous targets, change the mode, the loss and the metric; there are no classes to
stratify on. The same code works for targets of shape `(n, m)`:

```python
from embedkit.improvement.losses import RankNContrastLoss

X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
grid = {
    "default":   refiner(mode="regression"),               # RnC(0.07), MLP, GaussianNoise(0.05)
    "rnc-t0.2":  refiner(mode="regression", loss=lambda: RankNContrastLoss(temperature=0.2)),
    "residual":  refiner(mode="regression", architecture="residual"),
    "dim-64":    refiner(mode="regression", target_dim=64),
}
val = dict(X_train=X_fit, X_test=X_val, y_train=y_fit, y_test=y_val, metrics=["knn_r2"])
results = [evaluate_transform(lambda s: Identity(), method="raw", seeds=[0], **val)]
for name, make in grid.items():
    results.append(evaluate_transform(make, method=name, seeds=[0, 1, 2],
                                      fit_with_labels=True, **val))
print(compare(results, reference="raw")[["method", "mean", "diff", "diff_lo", "diff_hi"]])
```

**A shortcut for one candidate.** To try a single change without the grid, pass it to the default
selection. The refiner candidate then trains with your setting, and it still has to beat the raw
embedding on held-out rows to be picked:

```python
from embedkit.improvement.losses import TripletLoss

ek = EmbedKit(loss=TripletLoss(margin=0.2, mining="semi-hard", distance="cosine"),
              target_dim=64).fit(X_train, y_train)
print(ek.selected_, ek.selection_report_[["candidate", "score", "diff_lo", "diff_hi"]])
```

Here `target_dim=64` also replaces the PCA candidate sizes by 64 (3.3).

---

## 5. Training-free baselines

`embedkit.baselines` holds simple transforms that need no labels and no training. `EmbedKit`
uses `Identity`, `CenterNormalize` and PCA internally; the others are there for your own
comparisons (section 8) and for embeddings known to need a specific fix.

They share one interface: `fit(X, y=None)` (labels are accepted and ignored), `transform(X)`,
`fit_transform(X, y=None)` and `get_params()`. After `fit`, `n_features_in_`, `n_features_out_`
and the learned statistics are attributes with a trailing underscore; no training rows are
stored. They compute in float64 and return float32. `transform` raises `NotFittedError` before
`fit` and `ValueError` for a different number of columns.

| Class | Transform | When to use |
|---|---|---|
| `Identity()` | The input unchanged | The reference row of every comparison |
| `CenterNormalize()` | Subtract the training mean, scale each row to unit L2 norm | The safe default without labels |
| `Standardize()` | Per-column z-score with the training mean and standard deviation | When coordinates have very different scales; check on held-out data |
| `PCAReduce(n_components, whiten=False)` | Top principal components, optionally divided by the root eigenvalues. `n_components` is capped at the data rank, with a warning | A smaller embedding. Prefer `whiten=False`: whitening often hurts |
| `ZCAWhiten(eps=1e-5)` | Full-rank whitening that stays in the input coordinates | Rarely helpful for neighbour search; compare before using |
| `AllButTheTop(n_remove, n_components=None)` | Remove the mean and the top `n_remove` principal directions (Mu et al., 2018); with `n_components`, the reduce-then-clean variant of Raunak et al. (2019) | Word or sentence embeddings dominated by a few common directions; check on held-out data |
| `ConceptorNegation(alpha=2.0)` | Conceptor negation (Liu et al., 2019) | The same use as all-but-the-top, with a soft version of the removal |
| `RandomProjection(n_components)` | Gaussian random projection | A cheap reference for dimension reduction |

Most of these change neighbour-search accuracy little, in either direction, and whitening
(`ZCAWhiten`, `PCAReduce(whiten=True)`) usually lowers it. Treat every one as a candidate to
test on held-out data, not as a fix to apply.

```python
from embedkit.baselines import CenterNormalize, PCAReduce

cn = CenterNormalize().fit(X_train)
Z_test = cn.transform(X_test)
pca = PCAReduce(32).fit(X_train)
Z_train = pca.transform(X_train)          # (n_train, 32), float32
```

---

## 6. Hubness-reducing neighbour search

In high dimension, some points (*hubs*) appear among the nearest neighbours of many other points,
while others (*antihubs*) are nobody's neighbour (section 9.3). A search then keeps returning the
same few items. `embedkit.hubness_reduction` holds four *secondary distances* that correct this
by rescaling each distance with local statistics of the two points (Feldbauer & Flexer, 2019).

They change the distances between points, not their coordinates, so they are not transforms:
each is fitted on a reference set (a gallery, or the training split of a kNN classifier) and
answers neighbour queries against it.

```python
from embedkit.hubness_reduction import CSLS, DisSimLocal, LocalScaling, MutualProximity

mp = MutualProximity(metric="euclidean").fit(Z_train)
dist, idx = mp.kneighbors(Z_test, n_neighbors=10)  # neighbours among the Z_train rows
dist, idx = mp.kneighbors(n_neighbors=10)          # reference rows among each other, self excluded
```

All four start from a base distance `d` (`metric="euclidean"`, or `"cosine"` for `1 - cos`):

| Class | Secondary distance | Statistic |
|---|---|---|
| `LocalScaling(k=5, metric="euclidean")` (Schnitzer et al., 2012) | `1 - exp(-d² / (s_x s_y))` | `s_x`: distance to the `k`-th nearest reference point |
| `MutualProximity(metric="euclidean", sample_size=2000)` (Schnitzer et al., 2012; Gaussian variant) | `1 - P(D_x > d) P(D_y > d)` | `D_x`: a normal fitted to the distances from `x` to a sample of the reference set |
| `CSLS(k=10, metric="cosine")` (Conneau et al., 2018) | `2d - r_x - r_y` | `r_x`: mean distance to the `k` nearest reference points; with the cosine metric this ranks as the original `2 cos - r_x - r_y` |
| `DisSimLocal(k=5, metric="euclidean")` (Hara et al., 2016) | `‖x - y‖² - ‖x - c_x‖² - ‖y - c_y‖²` | `c_x`: centroid of the `k` nearest reference points; with the cosine metric it runs on unit vectors |

**When to use it.** When your system searches with Euclidean distance and the analysis report
recommends it (section 9.1: k-skewness or the antihub ratio above the null). The gains are
usually modest, often below one point of accuracy, and under cosine similarity they are smaller
still. Measure them on held-out data before you deploy:

```python
from embedkit.baselines import Identity
from embedkit.evaluation import compare, evaluate_transform

kw = dict(X_train=Z_train, X_test=Z_test, y_train=y_train, y_test=y_test, seeds=[0],
          metrics=["knn_accuracy_euclidean"])
results = [evaluate_transform(lambda s: Identity(), method="plain", **kw),
           evaluate_transform(lambda s: Identity(), method="mutual-proximity",
                              hubness_reduction=MutualProximity(), **kw)]
print(compare(results, reference="plain")[["method", "mean", "diff", "diff_lo", "diff_hi"]])
```

The search is exact and processed in chunks to bound memory. A reference row is never its own
neighbour when the reference set is queried; a query is a new point, so a duplicate of a
reference row is returned.

The kNN-based metrics accept a `hubness_reduction` argument: `knn_accuracy`, `recall_at_k` and
`k_skewness` in `embedkit.evaluation.metrics`, and `evaluate_transform(...,
hubness_reduction=...)`, which applies it to `knn_accuracy`, `knn_accuracy_euclidean`,
`recall_at_10` and `k_skewness` (each with its own base metric) and leaves the other metrics
unchanged.

---

## 7. Building a custom refiner

`EmbedKit(method="refiner")` assembles a refiner from parts you can also use directly: a model
(`EmbeddingRefiner`), an augmentation that makes pairs of related inputs, a contrastive loss, and
a `Trainer`. Use the parts directly when you need something `EmbedKit` does not expose, such as a
custom loss, a composite augmentation or full access to the training history. To change a single
part, it is simpler to pass it to `EmbedKit` (`augmentation=`, `loss=`, `architecture=`).

**How a refiner learns.** Each training step takes a batch of embeddings, makes two related
versions of it (two noisy copies, or each point and one of its neighbours), passes both through
the model, and applies a loss that pulls related outputs together and pushes unrelated ones
apart. With labels, "related" also means "same class" (SupCon) or "similar target"
(Rank-N-Contrast).

### 7.1 `EmbeddingRefiner`

A small network that maps embeddings of dimension D to `target_dim`, with L2-normalised output
by default.

| `architecture` | Map | When to use |
|---|---|---|
| `"linear"` | `z = W x̃`, `W` initialised at PCA | Without labels; the smallest model |
| `"residual"` | `z = W x̃ + g(x̃)`, `W` at PCA, `g` an MLP whose output layer starts at zero | With class labels; robust with few labels per class |
| `"mlp"` | `n_layers` × (`Linear → BatchNorm → ReLU → Dropout`), then `Linear(hidden_dim, target_dim)` | Regression; classification with many labels |

`x̃` is the input standardised with fixed per-feature statistics. The linear and residual
variants start from PCA: before training, their output equals `PCAReduce(target_dim)` followed
by L2 normalisation, so training refines PCA instead of starting from a random map.

**Class:** `EmbeddingRefiner(input_dim, target_dim, hidden_dim=256, n_layers=2, dropout=0.1,
normalize_output=True, architecture="linear")`. `hidden_dim`, `n_layers` and `dropout` configure
the MLP part; `"linear"` ignores them. `normalize_output=True` is required by the cosine-based
losses.

`Trainer.fit` initialises a linear or residual refiner at PCA; a custom training loop must call
`initialize_from_data(X_train)` itself (training without it warns once).

```python
import torch

from embedkit.improvement.model import EmbeddingRefiner

X_train = torch.randn(500, 128)
model = EmbeddingRefiner(input_dim=128, target_dim=32)   # architecture="linear"
model.initialize_from_data(X_train.numpy())               # start at PCA
z = model(X_train[:64])   # (64, 32), L2-normalised
```

### 7.2 Augmentations

An augmentation turns a batch `x` into a pair `(x_i, x_j)`: two versions of the same batch that
the loss should map close together. **Base class:** `BaseAugmentation.__call__(x) -> (Tensor,
Tensor)`.

| Augmentation | Pair | Use when |
|---|---|---|
| `KNNPairs` | Each point and one of its nearest neighbours | The default with class labels and without labels: neighbours carry the most signal |
| `GaussianNoise` | Two noisy copies | The default for regression; any time neighbours are unreliable |
| `FeatureDropout` | Two copies with different coordinates zeroed | Robustness to missing or noisy coordinates |
| `FeatureMasking` | Two copies with blocks of coordinates zeroed | Embeddings made by concatenating parts in a fixed order |
| `EmbeddingMixup` | Two interpolations with a neighbour | Smoother views along the data manifold |
| `IdentityPair` | `(x, x)`, differing only by the model's dropout | SimCSE-style training; combined with view augmentations |
| `CompositeAugmentation` | A chain of the above, or a random choice per step | Combining views |

#### KNNPairs

The anchor `x` and one of its `k` nearest neighbours, drawn at random, form the pair. When the
trainer calls `precompute` (it does), neighbours come from the whole training set; otherwise they
are searched within the batch.

**Parameters:** `k=10`, `neighbor_index=None` (a precomputed `(n_train, k)` index array),
`metric="euclidean"`, `mode="all"`. With `mode="mutual"` only mutual neighbours are used (`j` is
among the `k` neighbours of `i` and `i` among those of `j`), which removes most of the
cross-cluster pairs that hubs create; a point with no mutual neighbour is paired with its nearest
neighbour.

`KNNPairs` is a *pair* augmentation: in a sequential `CompositeAugmentation` it must come first,
and at most one pair augmentation is allowed. View augmentations placed after it perturb both
the anchor and the neighbour.

```python
from embedkit.improvement.augmentation import KNNPairs

aug = KNNPairs(k=10)
anchor, positive = aug(x_batch)
```

#### GaussianNoise

Adds independent Gaussian noise to each embedding. With `relative=True` (default), `std` is a
multiple of the per-dimension standard deviation of the training data (computed by
`precompute`), so the amount of noise does not depend on the scale of the embeddings;
`relative=False` uses `std` as an absolute standard deviation. With `adaptive=True` the noise
is also scaled per point by its mean distance to its `k` nearest neighbours: less noise in dense
regions, more in sparse ones.

**Parameters:** `std=0.05`, `adaptive=False`, `k=10`, `relative=True`

```python
from embedkit.improvement.augmentation import GaussianNoise

aug = GaussianNoise(std=0.05, adaptive=True)
x_i, x_j = aug(x_batch)
```

#### FeatureDropout

Zeroes random coordinates, with two independent masks, so the model cannot rely on any single
coordinate. **Parameters:** `p=0.1` (probability of dropping each coordinate).

```python
from embedkit.improvement.augmentation import FeatureDropout

aug = FeatureDropout(p=0.15)
x_i, x_j = aug(x_batch)
```

#### FeatureMasking

Zeroes contiguous blocks of coordinates, in the spirit of Cutout (DeVries & Taylor, 2017), with
two independent masks. Useful when the embedding is a concatenation of parts in a fixed order.
**Parameters:** `mask_ratio=0.15`, `block_size=1`.

```python
from embedkit.improvement.augmentation import FeatureMasking

aug = FeatureMasking(mask_ratio=0.2, block_size=4)
x_i, x_j = aug(x_batch)
```

#### EmbeddingMixup

Interpolates each embedding with one of its `k` nearest neighbours, `x_aug = λ x + (1 − λ)
x_neighbour`, with λ drawn from a Beta distribution (Zhang et al., 2018). Two independent draws
give the two views. **Parameters:** `k=10`, `alpha=0.4`.

```python
from embedkit.improvement.augmentation import EmbeddingMixup

aug = EmbeddingMixup(k=10, alpha=0.4)
x_i, x_j = aug(x_batch)
```

#### IdentityPair

Returns `(x, x)`. The two views pass through the refiner separately in training mode, so they
differ only by its dropout masks: the positives of SimCSE (Gao et al., 2021). It is a pair
augmentation; view augmentations may follow it in a `CompositeAugmentation`.

#### CompositeAugmentation

- `mode="sequential"` applies every augmentation in order, threading the views through each one.
- `mode="random_choice"` picks one augmentation at random at each training step.

```python
from embedkit.improvement.augmentation import CompositeAugmentation, FeatureDropout, GaussianNoise

aug = CompositeAugmentation([GaussianNoise(std=0.05), FeatureDropout(p=0.1)], mode="sequential")
x_i, x_j = aug(x_batch)
```

### 7.3 Losses

Every loss is a `BaseLoss` (a `torch.nn.Module`) with
`forward(z_i, z_j, labels=None) -> Tensor`, where `z_i` and `z_j` are the model outputs for the
two views.

| Loss | Labels | Use when |
|---|---|---|
| `SupConLoss` | Classes | The default with class labels |
| `RankNContrastLoss` | Continuous, `(n,)` or `(n, m)` | The default for regression |
| `TripletLoss` | Classes | An alternative to SupCon, especially with a margin-based downstream use |
| `NTXentLoss` | None | Self-supervised training |
| `AlignUniformLoss` | None | An extra term that spreads outputs over the sphere |
| `CombinedLoss` | As its members | A weighted sum of losses |
| `MatryoshkaLoss` | As its base loss | Training nested output sizes |

**Temperature.** The contrastive losses divide similarities by a temperature τ. A lower τ
concentrates the training on the hardest pairs; a higher τ treats pairs more evenly. 0.07 is the
default; 0.1 to 0.2 is worth trying (section 4.3).

#### SupConLoss

Supervised contrastive loss (Khosla et al., 2020). Every sample of the same class is a positive
for an anchor, not only its other view. Without labels it falls back to the NT-Xent behaviour.

**Parameters:** `temperature=0.07`, `chunk_size=None`, `spread_weight=0.0`.

- `spread_weight` (`w`, in [0, 1]) mixes in a class-conditional term (Chen et al., 2022):
  `(1 - w) SupCon + w spread`. The spread term pushes members of a class apart while SupCon pulls
  them together, so the structure inside each class survives. Use 0.25 to 0.5 when the labels are
  coarser than the distinctions you care about.
- `chunk_size` computes the similarity matrix in blocks of rows to reduce peak memory.

```python
from embedkit.improvement.losses import SupConLoss

loss_fn = SupConLoss(temperature=0.07)
loss = loss_fn(z_i, z_j, labels=y_batch)
```

#### RankNContrastLoss

Contrastive loss for continuous targets (Zha et al., 2023). For each anchor, samples closer in
target space must have a higher cosine similarity than samples farther away. It uses only the
ranking of target distances, so there is no bandwidth to tune. Targets can be scalars `(n,)` or
vectors `(n, m)`; distances between vectors are Euclidean (standardise columns in different
units).

Samples at exactly the same target distance from an anchor are treated as tied. To treat nearby
values as tied, round the targets before training.

**Parameters:** `temperature=0.07`, `chunk_size=None` (compute the `(2N × 2N)` similarity matrix
in blocks of rows to reduce peak GPU memory). **Requires labels** (raises `ValueError`
otherwise).

```python
from embedkit.improvement.losses import RankNContrastLoss

loss_fn = RankNContrastLoss(temperature=0.07)
loss = loss_fn(z_i, z_j, labels=y_batch)   # y_batch: (N,) or (N, m)

loss_fn = RankNContrastLoss(temperature=0.07, chunk_size=128)   # for large batches
```

#### TripletLoss

Metric-learning objective (Schroff et al., 2015): for an anchor, a positive (same class) and a
negative (another class), it penalises `d(anchor, positive) − d(anchor, negative) + margin > 0`.

Mining strategies:

- `"hard"`: the farthest positive and the closest negative. Aggressive; can be unstable.
- `"semi-hard"`: the closest negative that is still farther than the anchor's mean positive
  distance. A good balance.
- `"easy"`: the average over all positives and negatives. Stable but slow.

**Parameters:** `margin=0.5`, `mining="hard"`, `distance="euclidean"` or `"cosine"`.
**Requires labels.** With unit-length outputs, `distance="cosine"` and a margin around 0.2 are a
good start.

```python
from embedkit.improvement.losses import TripletLoss

loss_fn = TripletLoss(margin=0.2, mining="semi-hard", distance="cosine")
loss = loss_fn(z_i, z_j, labels=y_batch)
```

#### NTXentLoss

The SimCLR / InfoNCE objective (van den Oord et al., 2018; Chen et al., 2020). For each anchor,
its other view is the positive and the other 2N − 2 outputs of the batch are negatives.

**Parameters:** `temperature=0.07`, `chunk_size=None`, `neighbor_mask=None`,
`neighbor_metric="euclidean"`, `decoupled=False`.

- `neighbor_mask=k` (or a precomputed `(n_train, k)` index array) removes likely false
  negatives: two rows of a batch are not negatives for each other when one is among the `k`
  nearest input-space neighbours of the other. The trainer computes the neighbours and passes
  each batch's row indices; without indices no mask is applied.
- `decoupled=True` leaves the positive out of the denominator (Yeh et al., 2022).

```python
from embedkit.improvement.losses import NTXentLoss

loss_fn = NTXentLoss(temperature=0.07)
loss = loss_fn(z_i, z_j)
```

#### AlignUniformLoss

The two terms of Wang & Isola (2020): *alignment*, `E[‖z_i − z_j‖^α]`, pulls the views of a
pair together, and *uniformity*, `log E[exp(−t ‖z_i − z_j‖²)]`, spreads all outputs over the
sphere. The uniformity term is the quantity `UniformityScore` measures (section 9.7).

**Parameters:** `alpha=2.0`, `t=2.0`. Use it as a secondary term in a `CombinedLoss` (the
self-supervised default adds it to NT-Xent with weight 0.5). Added to SupCon it rarely helps;
validate it before keeping it.

```python
from embedkit.improvement.losses import AlignUniformLoss

loss_fn = AlignUniformLoss(alpha=2.0, t=2.0)
loss = loss_fn(z_i, z_j)
```

#### CombinedLoss

A weighted sum of losses. Labels are passed only to the members that use them.

```python
from embedkit.improvement.losses import AlignUniformLoss, CombinedLoss, SupConLoss

loss_fn = CombinedLoss([(SupConLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)])
loss = loss_fn(z_i, z_j, labels=y_batch)
```

#### MatryoshkaLoss

The sum of a base loss over nested prefixes of the output, `z[:, :m]` for each `m` in `dims`
(Kusupati et al., 2022); the full output is always included. Every prefix is trained to be an
embedding on its own, so one model serves several sizes through `transform(X, dim=m)`.

**Parameters:** `base_loss`, `dims=(16, 32, 64, 128)`, `weights=None` (1 for every prefix).
`EmbedKit(nested_dims=...)` wraps the default loss for you (section 3.6).

```python
from embedkit import EmbedKit
from embedkit.improvement.losses import MatryoshkaLoss, SupConLoss

ek = EmbedKit(method="refiner", target_dim=128,
              loss=MatryoshkaLoss(SupConLoss(), dims=(16, 32, 64)))
ek.fit(X, y)
Z32 = ek.transform(X, dim=32)   # first 32 coordinates, renormalised
```

#### Writing your own loss

Subclass `BaseLoss` and implement `forward(z_i, z_j, labels=None)`. Set the class attribute
`uses_labels = True` if `forward` needs the labels: `CombinedLoss` and `MatryoshkaLoss` pass
labels only to members that declare it. A custom loss can be passed to `EmbedKit(loss=...)` or
to `Trainer`.

### 7.4 `Trainer`

`Trainer` runs the training loop: batching, optimisation, learning-rate scheduling, periodic
evaluation, early stopping and the loss history.

**Class:** `Trainer(model, augmentation, loss, epochs=100, batch_size=256, optimizer="adam",
lr=3e-4, weight_decay=1e-4, scheduler="cosine", warmup_epochs=0, eval_every=10,
eval_metrics=None, eval_subsample=5000, eval_k=10, eval_metric="cosine",
early_stopping_patience=None, monitor="uniformity", restore_best=True, val_split=0.0,
device=None, random_state=42)`

| Parameter | Notes |
|---|---|
| `optimizer` | `"adam"` (a safe default) or `"sgd"` |
| `scheduler` | `"cosine"` (stepped per batch), `"step"` (per epoch), `"plateau"` (per evaluation) or `None` |
| `warmup_epochs` | Linear learning-rate warm-up before the scheduler |
| `eval_every` | Epochs between evaluations |
| `eval_metrics` | Names in `trainer.MONITORS`, recorded in `history`; the monitor is always added. Without labels: `"uniformity"`, `"isotropy"`, `"k_skewness"`, `"knn_overlap"`, `"effective_rank"`, `"val_loss"`. With labels: `"val_knn_accuracy"` (5-NN, classes), `"val_ridge_r2"` (continuous targets), `"val_nc1"`, `"train_nc1"` |
| `eval_k`, `eval_metric` | Neighbourhood used by `k_skewness` and `knn_overlap` |
| `monitor` | The metric watched for early stopping; its direction comes from the registry. The `val_*` monitors need `val_split > 0` |
| `val_split` | Fraction of the rows held out from training for the `val_*` monitors |
| `early_stopping_patience` | Stop after this many **evaluations** (not epochs) without improvement |
| `restore_best` | With early stopping, reload the best evaluation's weights at the end (`best_score_`, `best_epoch_`) |
| `device` | `None` picks CUDA, then MPS, then CPU |

**Stop early only on a labelled monitor.** Label-free statistics such as uniformity, isotropy or
k-skewness do not track downstream accuracy during training, so early stopping on them stops at
an arbitrary epoch. With labels, hold out rows and monitor `val_knn_accuracy` or `val_ridge_r2`;
without labels, train for a fixed number of epochs and log the label-free metrics for inspection
only.

The supervised refiner that `EmbedKit(method="refiner")` builds with class labels, written out:

```python
from embedkit.improvement.augmentation import KNNPairs
from embedkit.improvement.losses import SupConLoss
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer

model = EmbeddingRefiner(input_dim=X.shape[1], target_dim=32, architecture="residual")
trainer = Trainer(
    model=model,
    augmentation=KNNPairs(k=5),
    loss=SupConLoss(temperature=0.07),
    epochs=100,
    batch_size=256,
    lr=3e-4,
    scheduler="cosine",
    eval_every=5,
    monitor="val_knn_accuracy",    # 5-NN accuracy of the held-out rows
    val_split=0.1,
    early_stopping_patience=10,    # evaluations, not epochs
)
trainer.fit(X, y=y)                # initialises the residual map at PCA, then trains

X_refined = trainer.transform(X)   # np.ndarray, rows L2-normalised
print(trainer.best_epoch_, trainer.best_score_)
print(trainer.history.keys())      # "loss", "val_knn_accuracy", ...
```

`trainer.fit(X)` without `y` trains self-supervised. Pass a loss that needs labels only with
`y`.

---

## 8. Evaluating on your own data

Before you rely on a transform, score it on a test split that played no part in fitting or
choosing it, with the measure that matches your use. `embedkit.evaluation` provides the metrics
and a protocol that handles seeds and paired confidence intervals.

### 8.1 Choosing a metric

| You use the embedding for | Score it with |
|---|---|
| Nearest-neighbour classification or search, cosine | `"knn_accuracy"`, `"recall_at_10"` |
| The same, Euclidean distance | `"knn_accuracy_euclidean"` |
| A linear classifier | `"linear_probe_c1"` (fast) or `"linear_probe"` (tuned) |
| Regression from neighbours | `"knn_r2"` |
| Regression with a linear model | `"linear_probe"` (ridge for continuous targets) |
| Sentence similarity with gold scores | `"sts_spearman"` (pass `test_pairs=`) |
| Clustering | `"silhouette"`, `"nc1"` |

The registry `embedkit.evaluation.METRICS` holds these names:

| Name | Scores | Needs |
|---|---|---|
| `"knn_accuracy"`, `"knn_accuracy_euclidean"` | 5-NN accuracy, cosine or Euclidean | class labels |
| `"linear_probe"` | Logistic regression (classes) or ridge (continuous), regularisation chosen by cross-validation on the training split | labels |
| `"linear_probe_c1"` | Logistic regression with C = 1, no tuning | class labels |
| `"recall_at_10"` | Share of test rows with a same-class training row among their 10 nearest (cosine) | class labels |
| `"knn_r2"` | 10-NN regression R², cosine; averaged over outputs for `(n, m)` targets | continuous targets |
| `"nc1"`, `"silhouette"` | Class separation of the output | class labels |
| `"sts_spearman"` | Spearman correlation of pair similarities with gold scores | `test_pairs=` |
| `"trustworthiness"`, `"continuity"`, `"knn_overlap"`, `"linear_cka"` | How much of the input's structure the output keeps (the neighbour-based ones at k = 10) | nothing |
| `"effective_rank"`, `"k_skewness"` | Geometry of the output | nothing |

The label-free names describe what a transform did, but they cannot tell you whether it helps:
`knn_overlap`, for example, gives the unchanged input a perfect score. Choose with a labelled
metric that matches your use.

The same measures are available as functions in `embedkit.evaluation.metrics`, each returning a
float. Neighbours, fits and standardisation use the training split only.

| Function | Scores |
|---|---|
| `knn_accuracy(Z_train, y_train, Z_test, y_test, k=5, metric="cosine")` | kNN majority-vote accuracy |
| `knn_regression_r2(Z_train, y_train, Z_test, y_test, k=10, metric="cosine")` | kNN regression R²; targets `(n,)` or `(n, m)` |
| `linear_probe(Z_train, y_train, Z_test, y_test, task="auto")` | Logistic regression or ridge; regularisation by cross-validation, or fixed with `C=` |
| `recall_at_k(Z_query, Z_gallery, relevance, k=10)` | Share of queries with at least one relevant gallery row (same label, or a boolean matrix) among their `k` nearest |
| `trustworthiness`, `continuity`, `knn_overlap` | Neighbour preservation between an input and its transform |
| `linear_cka`, `effective_rank`, `k_skewness`, `nc1`, `silhouette` | Geometry of one embedding |

`knn_accuracy`, `knn_regression_r2`, `linear_probe` and `recall_at_k` accept
`return_per_point=True` and then also return the per-test-row outcomes, which `compare` uses for
its intervals.

### 8.2 The evaluation protocol

`evaluate_transform(make_transform, X_train, X_test, y_train, y_test, seeds=..., metrics=...)`
fits a fresh transform per seed on the training split, scores the test split, and returns a
table with one row per seed and metric, plus fit and transform times and the output dimension.

- `make_transform(seed)` returns an unfitted object with `fit` and `transform`: an `EmbedKit`, a
  baseline, or your own.
- `fit_with_labels=True` passes `y_train` to `fit` (needed for supervised refiners).
- `transductive=True` fits on both splits, and records that it did.
- `hubness_reduction=` applies a secondary distance to the kNN metrics (section 6).

`compare(results, reference="raw")` gives the mean and standard deviation over seeds for each
method and metric, and the difference to the reference with a paired bootstrap interval over
test rows (2,000 resamples, 95%) where per-row outcomes exist.

```python
from embedkit import EmbedKit
from embedkit.baselines import Identity, PCAReduce
from embedkit.evaluation import compare, evaluate_transform

splits = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, seeds=[0])
results = [
    evaluate_transform(lambda s: Identity(), method="raw", **splits),
    evaluate_transform(lambda s: PCAReduce(32), method="pca-32", **splits),
    evaluate_transform(lambda s: EmbedKit(random_state=s), method="embedkit",
                       fit_with_labels=True, **splits),
]
print(compare(results, reference="raw"))
```

**Reading the table.** Count a difference as a gain or a loss only when its interval excludes 0.
An interval that straddles 0 means the test split cannot tell the two apart; a larger test split
narrows it. For trained transforms, use several seeds (`seeds=[0, 1, 2]`) so that one lucky run
does not decide.

### 8.3 Input requirements

Every public entry point (`EmbedKit.fit`, the baselines, the analyzers, `Trainer.fit`, the
hubness reductions) checks its input with `embedkit.utils.validation.validate_embeddings`:

- **Errors:** input that is not 2-D, too few rows, a wrong number of features, NaN or infinite
  values (the message gives how many entries and rows).
- **Warnings** (from `EmbedKit.fit`, the baselines, the analyzers and `Trainer.fit`): constant
  columns and exact duplicate rows, each with its count. Duplicates are
  common when the same text or image appears twice; they do no harm to the transforms, but they
  inflate neighbour-based scores if the same item lands in both splits.

NumPy arrays and PyTorch tensors are both accepted. `EmbedKit`, `Trainer` and the analyzers work
in float32; the baselines and hubness reductions in float64.

### 8.4 Nearest-neighbour backend

`embedkit.utils.neighbors.knn(X, k, metric=...)` is the exact kNN search used throughout. It uses
scikit-learn by default. FAISS is an optional extra that is faster on large data
(`pip install "embedkit[faiss] @ git+https://github.com/fagonzalezo/embedding-kit.git"`, then
`backend="faiss"` or the environment variable `EMBEDKIT_KNN_BACKEND=faiss`). Every backend returns
true distances (Euclidean, or `1 - cos`), sorted, without the query point.

---

## 9. Diagnostics: describing an embedding space

The analyzers describe the geometry of an embedding space: how many directions it uses, how
uneven its neighbour graph is, how its distances and directions spread. Use them to:

- describe and compare encoders, layers or data sets;
- check for hubness before a Euclidean nearest-neighbour search (section 6);
- understand what a transform changed, next to the held-out scores of section 8.

**They do not choose a transform.** A statistic outside its usual range does not mean that a fix
will help, and no `EmbedKit` default reads the report. To decide whether a transform helps,
measure it (sections 3.3 and 8).

### 9.1 The report: `EmbedKitAnalyzer`

```python
from embedkit import EmbedKitAnalyzer

report = EmbedKitAnalyzer(k=10, id_methods=["TwoNN", "MLE", "lPCA"]).fit(X)
report.print_summary()

print(report.hubness.k_skewness)
print(report.geometry.isotropy.isotropy_score)
print(report.calibration["k_skewness"].z_score)   # against the null model
print(report.n_outside_null)      # calibrated statistics with |z| > 3
print(report.properties)          # descriptive sentences
print(report.recommendations)     # list of strings
print(report.k, report.kernel.sigma, report.suggested_target_dim)

df = report.to_dataframe()        # tabular view
```

`EmbedKitAnalyzer` runs the seven analyzers of sections 9.2 to 9.8 in one call and returns an
`EmbedKitReport` with:

- the results of the analyzers: `intrinsic_dim`, `hubness`, `geometry` (distance concentration,
  isotropy, neighbour consistency, uniformity) and `kernel`;
- `calibration`: five statistics compared with a null model, as z-scores (below);
- `properties`: descriptive sentences about the spectrum, the dominant coordinates, the mean
  direction and, when calibrated, uniformity against the null;
- `recommendations`: actions, at most two (below).

**Why compare with a null model.** Statistics such as k-skewness or the spread of distances
depend on the number of rows, the dimension, `k` and the variance spectrum, so the same value
means different things on different data. The report therefore recomputes five of them
(k-skewness, the antihub ratio, `distance_cv`, neighbour consistency and uniformity) on random
data of the same size and spectrum, and reports how far your data are from that reference as a
z-score. A z-score near 0 means "as for structureless data of this shape"; `|z| > 3` means
clearly different.

- **The null.** `null_model="gaussian"` (default) is a Gaussian with the data's mean and a
  covariance matched to the data's spectrum; `"permuted"` permutes each column independently.
  Rows normalised upstream (all of the same length) are compared with null draws normalised the
  same way.
- **The draws.** `n_null_draws` (default 20) data sets of `n_calibration` rows (default 2,000),
  with the same metric.
- **The result.** `report.calibration[name]` holds the observed value, the null mean and
  standard deviation, `z_score`, `z_interval` (a 95% interval that reflects the limited number
  of null draws) and `uncalibrated`. Raise `n_null_draws` if a z-score lies near a threshold
  and you need a stable reading.
- **The count.** `report.n_outside_null` counts the calibrated statistics with
  `|z| > z_threshold` (default 3).

**Where the null is not reliable.** In some settings, random data themselves fall outside ±3
too often, so the report flags the statistic as uncalibrated (`uncalibrated` gives the reason;
`.calibrated` is the boolean): the antihub ratio, `distance_cv` and uniformity on rows normalised
upstream; k-skewness and the antihub ratio under `metric="cosine"`; neighbour consistency under
`metric="euclidean"` above 256 dimensions. An uncalibrated statistic keeps its z-score but raises
no recommendation and is not counted in `n_outside_null`. `distance_cv` under cosine on data with
a large common mean can also land just beyond ±3 on random data; read it with care.

**Recommendations.** The report makes one kind of recommendation: when k-skewness or the antihub
ratio is above the null under `metric="euclidean"`, it suggests a hubness-reducing search
(section 6). No other statistic raises an action, and nothing does under cosine. When nothing
fires, the report says so. That does not mean no transform can help, only that the report cannot
tell which one would; the selection of section 3.3 can. With `calibrate=False` no recommendation
is made.

**Metric.** `metric="euclidean"` (default) or `"cosine"`. With `"cosine"` the rows are
L2-normalised once and every number describes the normalised rows, the geometry a cosine search
sees. Rows with zero length raise a `ValueError`. Analyse with the metric your system uses.

**Derived values.**

| Field | Meaning |
|---|---|
| `k` | The neighbourhood size used: the `k` given, else `max(5, round(√min(n, n_max) / 2))` |
| `calibration_k` | The `k` of the calibration: `k` when given, else the same rule applied to the calibration rows (22 for 2,000 rows) |
| `kernel.sigma` | The RBF bandwidth of the kernel diagnostics (median pairwise distance) |
| `suggested_target_dim` | `clip(round(1.5 × ID), max(4, ID), D)`, with ID the reference intrinsic-dimension estimate (section 9.2). A rough starting point, often too low for a reduction; read by `EmbedKit` only with `method="pca"` or `"refiner"` and `target_dim="auto"` |

**Class:** `EmbedKitAnalyzer(k=None, id_methods=None, metric="euclidean", n_max=20000,
random_state=42, calibrate=True, null_model="gaussian", n_null_draws=20, n_calibration=2000,
z_threshold=3.0, id_bootstrap=20, id_scale_curve=True)`. `id_methods=None` runs
`["TwoNN", "MLE", "lPCA", "MOM"]`; the neighbour statistics use at most `n_max` rows. For a
quick look, `calibrate=False, id_bootstrap=0, id_scale_curve=False` is several times faster
(section 11).

The sections below follow one template: what the statistic measures, the class and its result
fields, and how to read the numbers. The ranges given describe what a value means; they are not
pass/fail thresholds.

### 9.2 Intrinsic dimension

**What it measures.** The *intrinsic dimension* (ID) is the number of coordinates needed to
describe the data locally. A sentence encoder may output 768-dimensional vectors that lie close
to a surface of a few dozen dimensions. The ID describes the data; it is not a target for
dimension reduction, and reducing to it usually loses accuracy.

EmbedKit wraps eight estimators from scikit-dimension. They make different assumptions and
measure different things, so they are reported separately and never combined:

| Method | Approach |
|---|---|
| `TwoNN` | Ratio of the two nearest-neighbour distances; fast and robust (Facco et al., 2017) |
| `MLE` | Maximum likelihood over local neighbourhood distances (Levina & Bickel, 2005) |
| `lPCA` | Local PCA: counts linear dimensions |
| `DANCo` | Distance and angle statistics (Ceruti et al., 2012) |
| `CorrInt` | Correlation-integral scaling (Grassberger & Procaccia, 1983) |
| `MOM` | Method of moments on neighbour distances (Amsaleg et al., 2015) |
| `FisherS` | Fisher separability (Albergante et al., 2019) |
| `ESS` | Expected simplex skewness (Johnsson et al., 2015); accurate at high ID, but slow (about 100 s per fit at n = 5,000) |

**Class:** `IntrinsicDimensionEstimator(methods=None, n_max=5000, n_bootstrap=20,
scale_curve=True, random_state=42)`

**Result fields:**

- `estimates`: the ID per method (NaN, with a warning, when a method fails).
- `standard_errors`, `intervals`: bootstrap standard error and the interval
  `estimate ± 1.96 se`. `ESS` and `DANCo` are too slow to resample and get NaN.
- `between_method_spread`: standard deviation of the estimates across methods. It shows how much
  the methods disagree, not a sampling uncertainty.
- `scale_curve`, `scale_max_deviation`: TwoNN on nested subsamples of n, n/2, n/4 and n/8 rows
  (fewer rows probe larger scales), and the largest relative deviation from the full-sample value.
  Noise inflates the ID at short scales, so a curve that rises with n points to noise (Denti et
  al., 2022).
- `reference_method`, `reference_estimate`: the estimate the report uses for
  `suggested_target_dim` (TwoNN when available, otherwise the first method that succeeded).
- `local_estimates`: per-point ID for the methods that support it (TwoNN, MLE, lPCA).
- `n_duplicates_removed`: exact duplicate rows dropped before estimation.

**Reading the numbers.**

- The intervals cover sampling error only, not bias. The neighbour-based estimators (TwoNN, MLE,
  MOM) read low at high ID: on a 20-dimensional sphere with 5,000 points they give 15 to 17.
  `lPCA` counts linear dimensions, so it overestimates on curved data.
- Compare like with like: the same method, number of rows and preprocessing, across encoders or
  layers.

```python
from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator

result = IntrinsicDimensionEstimator(methods=["TwoNN", "MLE", "lPCA"]).fit(X)
for m, est in result.estimates.items():
    print(m, est, result.intervals[m])
print(result.scale_curve, result.scale_max_deviation)
```

### 9.3 Hubness

**What it measures.** *Hubness* is a high-dimensional effect in which a few points, the **hubs**,
appear in the nearest-neighbour lists of many other points, while **antihubs** appear in none
(Radovanović et al., 2010). Any method built on a neighbour graph (retrieval, kNN classification,
clustering, anomaly detection) is affected: hubs fill the result lists and antihubs are never
retrieved.

**What to do about it.** Hubness is the one property the report acts on: when k-skewness or the
antihub ratio is above the null under Euclidean distance, it suggests a hubness-reducing search
(section 6). Under cosine similarity hubness reduction helps little.

**Class:** `HubnessAnalyzer(k=10, hub_threshold=2.0, metric="euclidean", subsample=20000,
random_state=42)`. At most `subsample` rows are used; `fit(X, neighbor_index=...)` takes a
precomputed kNN graph.

**Result fields:**

| Field | Range | Meaning |
|---|---|---|
| `k_skewness` | (−∞, ∞) | Skewness of N_k, the number of times each point is among the `k` nearest neighbours of another. Near 0 when points are retrieved about equally often; positive with a tail of hubs |
| `robinhood_index` | [0, 1] | Inequality of N_k (the Robin Hood or Hoover index): the share of neighbour-list slots that would have to move to make every N_k equal. 0 when every point is retrieved equally often, near 1 when a few points take most of the neighbour lists |
| `antihub_ratio` | [0, 1] | Fraction of points with N_k = 0 (never retrieved) |
| `hub_ratio` | [0, 1] | Fraction of points with N_k > `hub_threshold` × mean N_k |
| `hub_contamination` | [0, 1] | Fraction of a hub's own neighbours that are also hubs |
| `k_occurrence`, `hubs`, `antihubs` | arrays | N_k per point, and the indices of the hubs and antihubs |

**Reading the numbers.** Raw values depend on n, d, k, the metric and the row lengths: random
Gaussian data at d = 1,024 already have an antihub ratio of 0.2 to 0.4 under Euclidean distance.
Compare with the null (`report.calibration`), or between embeddings of the same data at the same
`k`, not with fixed cut-offs.

```python
from embedkit.analysis.hubness import HubnessAnalyzer

result = HubnessAnalyzer(k=10).fit(X)
print(f"k-skewness:     {result.k_skewness:.3f}")
print(f"Robin Hood idx: {result.robinhood_index:.3f}")
print(f"Antihub ratio:  {result.antihub_ratio:.3f}")
```

### 9.4 Distance concentration

**What it measures.** In high dimension, pairwise distances tend to concentrate around their
mean: their spread shrinks relative to their size (Beyer et al., 1999), and the nearest and the
farthest point of a query become almost equally far away.

**Class:** `DistanceConcentration(subsample=2000, n_bins=50, random_state=42)`

**Result fields and reading:**

- `distance_cv`: standard deviation over mean of the pairwise distances, `[0, ∞)`. Small values
  mean concentrated distances. For random Gaussian data it is about `1 / sqrt(2d)`; because it
  depends on d and the spectrum, the report compares it with the null.
- `relative_contrast`: `(d_max − d_min) / d_mean`, `[0, ∞)`. Near 0 means the distance histogram
  is a narrow spike.
- `relative_contrast_per_query`: median over query points of `(d_max − d_min) / d_min`, with
  zero distances excluded.
- `concentration_ratio`: `d_min / d_max`, `[0, 1]`. A single duplicated pair sets it to 0, so
  prefer `distance_cv`.
- `distance_histogram`, `bin_edges`: the histogram of the pairwise distances, for plotting.

```python
from embedkit.analysis.geometry import DistanceConcentration

result = DistanceConcentration().fit(X)
print(f"Relative contrast:  {result.relative_contrast:.3f}")
print(f"Distance CV:        {result.distance_cv:.4f}")
```

### 9.5 Isotropy

**What it measures.** An *isotropic* space spreads its variance evenly over all directions; an
*anisotropic* one concentrates it in a few. Many encoders are strongly anisotropic, often with a
large shared mean direction or a few dominant coordinates. This is a property, not a defect:
cluster structure is itself anisotropic, and forcing isotropy (whitening) usually hurts
neighbour search. The report describes these statistics in `report.properties` and makes no
recommendation from them.

**Class:** `IsotropyAnalyzer(n_components=512, n_max=5000, random_state=42)`. All statistics are
computed exactly from at most `n_max` rows.

**Result fields:** `isotropy_score`, `participation_ratio`, `participation_ratio_corrected`,
`participation_ratio_normalized`, `effective_rank`, `isoscore`, `top_k_variance_share`,
`top_k_coordinate_share`, `spectrum_alpha`, `mean_vector_ratio`, `mean_cosine`,
`eigenvalue_spectrum_top` (the top `n_components` eigenvalues, for plotting),
`explained_variance_ratio`.

**Reading the numbers:**

- `participation_ratio_normalized` (in `[0, 1]`) is the summary of isotropy that does not depend
  on the number of rows: `participation_ratio_corrected / D`, where the corrected ratio uses an
  unbiased estimate of `tr(Σ²)`.
- `participation_ratio` = `(Σλᵢ)² / Σλᵢ²`, in `[1, D]`: roughly the number of directions that
  carry the variance. The plain value is biased low when n is not much larger than D (an
  isotropic Gaussian with n = 3,000 gives 0.75·D at D = 1,000).
- `effective_rank` = `exp(H)`, with H the entropy of the normalised eigenvalues, in `[1, D]`;
  `isotropy_score` = `effective_rank / min(n − 1, D)`, in `[0, 1]`. Both are biased low when n
  is not much larger than D.
- `isoscore`: IsoScore (Rudman et al., 2022), in `[0, 1]`, as published; it carries the same
  small-sample bias.
- `top_k_variance_share`: share of the variance in the top 1, 3 and 10 principal components.
  `top_k_coordinate_share`: the same for the coordinates with the largest variance; a few
  dominant coordinates are the "rogue dimensions" of Timkey & van Schijndel (2021).
- `spectrum_alpha`: power-law decay exponent of the eigenvalues, `λᵢ ∝ i^−α` (Agrawal et al.,
  2022), fitted over the top half of the spectrum. A flat spectrum gives 0.1 to 0.35.
- `mean_vector_ratio` = `‖mean(x)‖ / mean(‖x‖)` and `mean_cosine` (average cosine over pairs of
  rows): how much of each vector is a shared offset. A large value is what centring removes.

```python
from embedkit.analysis.geometry import IsotropyAnalyzer

result = IsotropyAnalyzer().fit(X)
print(f"Isotropy score:      {result.isotropy_score:.3f}")
print(f"Participation ratio: {result.participation_ratio:.1f} / {X.shape[1]}")
print(f"Effective rank:      {result.effective_rank:.1f}")
```

### 9.6 Neighbour consistency

**What it measures.** How stable the kNN graph is under small Gaussian perturbations. The data
are perturbed with noise of standard deviation `noise_std × s`, with `s` the root mean
per-coordinate standard deviation, and the fraction of each point's original neighbours that
survive is recorded. Low values mean that the neighbour lists change under small noise.

**Class:** `NeighborConsistency(k=10, n_perturbations=5, noise_std=0.01, metric="euclidean",
subsample=10000, random_state=42)`

**Result fields and reading:**

- `mean_consistency`, in `[0, 1]`: fraction of the original neighbours kept, averaged over points
  and perturbations. Near 1 means the neighbourhoods hardly move.
- `std_consistency`, `per_perturbation`: the spread over trials and the value of each trial.

```python
from embedkit.analysis.geometry import NeighborConsistency

result = NeighborConsistency(k=10, n_perturbations=5).fit(X)
print(f"Mean consistency: {result.mean_consistency:.3f}  ± {result.std_consistency:.3f}")
```

### 9.7 Uniformity

**What it measures.** How evenly the L2-normalised embeddings spread over the unit sphere (Wang &
Isola, 2020). Collapse, where all embeddings point in nearly the same direction, shows up as
values near 0. It is the uniformity term of `AlignUniformLoss`. Clustered directions are normal
for data with classes; the report states the result in `report.properties`.

**Class:** `UniformityScore(t=2.0, subsample=2000, random_state=42)`

**Result field and reading:** `uniformity` = `log E[exp(−t ‖zᵢ − zⱼ‖²)]` over pairs of distinct
normalised rows. It lies between about `−2t` (−4 at the default `t = 2`: rows spread uniformly in
high dimension) and 0 (all rows in the same direction).

```python
from embedkit.analysis.geometry import UniformityScore

result = UniformityScore(t=2.0).fit(X)
print(f"Uniformity: {result.uniformity:.4f}")   # about -4 (spread out) to 0 (collapsed)
```

### 9.8 Kernel diagnostics

**What it measures.** The RBF kernel `K(xᵢ, xⱼ) = exp(−‖xᵢ − xⱼ‖² / 2σ²)` describes a non-linear
similarity structure. Its eigenvalues show how many directions of the kernel feature space carry
the variance and whether one dominates (for example two well-separated clusters), which matters
to kernel methods (SVMs, kernel PCA, MMD tests). With labels, the kernel alignment measures how
closely the kernel geometry follows the labels.

**Class:** `KernelDiagnostics(sigma="median", n_components=50, subsample=2000, random_state=42)`.
With `sigma="median"` (default), σ is the median pairwise distance.

The spectrum statistics use the centred kernel `Kc = HKH` (`H = I − 11ᵀ/n`).

| Field | Range | Meaning |
|---|---|---|
| `effective_rank` | [1, subsample − 1] | `tr(Kc)² / tr(Kc²)`: how many kernel directions carry the variance |
| `spectral_gap` | [1, ∞) | `λ₁ / λ₂` of `Kc`: large when one direction dominates |
| `top_k_eigen_ratio` | [1, ∞) | `λ₁ / λ_k`, with k = `n_components` |
| `row_sum_skewness` | (−∞, ∞) | Skewness of the kernel's row sums: a continuous analogue of hubness |
| `centered_kernel_alignment` | [−1, 1], `None` or NaN | Only with `fit(X, y)` (NaN when the kernel or the labels are constant): centred alignment (Cristianini et al., 2001) between the data kernel and a label kernel (indicator kernel for classes, RBF for continuous targets). About 0 for labels unrelated to the data; near 1 when the kernel geometry reflects the labels |
| `eigenvalues`, `sigma` | | The kept eigenvalues and the bandwidth |

**Reading the numbers.** The effective rank depends on σ: a small bandwidth makes every point its
own direction. With `sigma="median"` it is comparable between data sets of the same size.

```python
from embedkit.analysis.kernel import KernelDiagnostics

result = KernelDiagnostics(sigma="median").fit(X)
print(f"Effective rank:    {result.effective_rank:.1f}")
print(f"Spectral gap:      {result.spectral_gap:.3f}")
print(f"lambda_1/lambda_k: {result.top_k_eigen_ratio:.3g}")
```

---

## 10. Visualization

Both classes return matplotlib `Figure` objects.

**`AnalysisPlotter`** renders a fitted `EmbedKitReport`:

```python
from embedkit.visualization.plots import AnalysisPlotter

plotter = AnalysisPlotter(report)
fig = plotter.plot_full_report()        # grid of all sub-plots
fig = plotter.plot_eigenvalue_spectrum()
fig = plotter.plot_distance_histogram()
fig = plotter.plot_k_occurrence()
fig = plotter.plot_kernel_spectrum()
fig = plotter.plot_id_local_map(X)      # histogram of per-point intrinsic dimension
fig = plotter.plot_hubness_map(X)       # k-occurrence per point, hubs and antihubs coloured
```

**`EmbeddingVisualizer`** projects embeddings to 2-D with PCA or UMAP (`method="pca"` or
`"umap"`, the default; UMAP needs the `umap-learn` package):

```python
from embedkit.visualization.embedding_viz import EmbeddingVisualizer

viz = EmbeddingVisualizer(method="pca", random_state=42)
fig = viz.plot_comparison(X, X_refined, labels=y)                   # before and after
fig = viz.plot_training_trajectory(trainer.history)                 # loss and metrics per epoch
fig = viz.plot_knn_graph(X, k=10, highlight_hubs=report.hubness.hubs)
```

A 2-D picture is a rough guide only: a projection can hide or invent structure, so judge a
transform by the held-out scores of section 8, not by how separated the clusters look. PCA is
the fastest and most faithful to global structure; UMAP shows local clusters better.

---

## 11. Time and memory

Measured on 16 CPUs of a cluster node (no GPU), with synthetic float32 data (10 classes). Each
entry is the wall time of one `fit` in a fresh process and its peak memory (GB = 1e9 bytes),
including the Python interpreter and PyTorch (about 0.6 GB) and the input itself (16.4 GB at
n = 1e6, d = 4,096).

`EmbedKit().fit(X)` without labels (centre + L2):

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | < 0.1 s / 0.6 GB | < 0.1 s / 0.6 GB | < 0.1 s / 0.7 GB |
| 10,000 | < 0.1 s / 0.6 GB | 0.1 s / 0.7 GB | 0.3 s / 1.1 GB |
| 100,000 | 0.1 s / 0.7 GB | 0.6 s / 1.2 GB | 2.2 s / 3.1 GB |
| 1,000,000 | 1.1 s / 1.2 GB | 5.9 s / 4.9 GB | 22 s / 17.8 GB |

`EmbedKit().fit(X, y)` with class labels (the selection, with the refiner as a candidate):

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 10 s / 1.0 GB | 11 s / 1.0 GB | 15 s / 1.2 GB |
| 10,000 | 28 s / 1.2 GB | 37 s / 1.4 GB | 2.2 min / 3.4 GB |
| 100,000 | 1.9 min / 2.8 GB | 2.6 min / 3.8 GB | 6.4 min / 11.8 GB |
| 1,000,000 | 1.9 min / 3.3 GB | 2.7 min / 7.5 GB | 6.7 min / 26.5 GB |

Above 50,000 rows the selection runs on a subsample, so its time stops growing with n. In these
runs the input was kept every time; when PCA or the refiner wins, it is fitted again on all the
rows, and that refit grows with n. On a GPU (Quadro RTX 8000) at n = 1e6, the selection took
1.4 min at d = 128, 2.1 min at d = 1,024 and 5.0 min at d = 4,096.

`EmbedKitAnalyzer().fit(X)` with the defaults:

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 36 s / 0.8 GB | 37 s / 0.9 GB | 1.3 min / 1.2 GB |
| 10,000 | 57 s / 1.0 GB | 1.0 min / 1.1 GB | 2.9 min / 2.0 GB |
| 100,000 | 1.1 min / 1.1 GB | 1.5 min / 2.2 GB | 3.1 min / 7.1 GB |
| 1,000,000 | 56 s / 2.6 GB | 1.6 min / 17.0 GB | 4.1 min / 66.1 GB |

`EmbedKitAnalyzer(calibrate=False, id_bootstrap=0, id_scale_curve=False).fit(X)`:

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 4.6 s / 0.8 GB | 6.3 s / 0.9 GB | 6.7 s / 1.1 GB |
| 10,000 | 6.7 s / 1.0 GB | 7.1 s / 1.1 GB | 27 s / 1.9 GB |
| 100,000 | 5.9 s / 1.1 GB | 10.5 s / 2.2 GB | 41 s / 7.1 GB |
| 1,000,000 | 7.9 s / 2.6 GB | 17.4 s / 17.0 GB | 1.2 min / 66.1 GB |

The analyzer's time hardly grows with n, because it works on subsamples (20,000 rows for the
neighbour statistics, 5,000 for intrinsic dimension, 2,000 for the calibration). Its memory does
grow, to about four times the input at d = 4,096, because the input is copied before it is
subsampled.

**Practical tips.**

- For a quick look at a large data set, run the analyzer with `calibrate=False, id_bootstrap=0,
  id_scale_curve=False`, or on a random sample of 20,000 rows.
- Keep `selection_max_rows` at its default or above 10,000. `selection_max_rows=None` on a
  million rows takes hours.
- Train refiners on a GPU when you have one; tuning grids (section 4) multiply the cost of one
  training by the number of configurations and seeds.
- At very large n × d, memory is dominated by the input and its copies; pass float32 arrays to
  avoid an extra conversion.

---

## 12. Troubleshooting

| Message or symptom | Cause | What to do |
|---|---|---|
| Warning "Too few rows to select a transform", result `identity` | Fewer than 20 rows to fit or to score in the selection | Use more labelled rows, raise `val_fraction`, or build a fixed configuration (`method="center_l2"`, `"pca"`, `"refiner"`) |
| `ValueError`: "y holds continuous values; pass mode='regression'" | Float labels with `mode="auto"` | Pass `mode="regression"`, or integer labels for classes |
| `ValueError`: regression targets must have shape `(n,)` or `(n, m)` | Targets of higher rank, such as `(n, h, w)` | `Y.reshape(len(Y), -1)` |
| `ValueError` naming `augmentation`, `loss`, `nested_dims` or `architecture` | A refiner setting where no refiner is fitted (for example `mode="self_supervised"` with `method="auto"`) | Remove the setting, or use a mode that compares a refiner |
| Warning "Regression refiner fitted on ... rows" | A fixed regression refiner on 1,000 rows or fewer | Prefer the selection (`method="auto"`), which keeps the input unless the refiner wins |
| Warning about constant columns or duplicate rows | Zero-variance coordinates; the same item embedded twice | Usually harmless; deduplicate before splitting so the same item is not in both train and test |
| `ValueError` about NaN or infinite values | Missing or corrupt embeddings | Remove or recompute those rows |
| `ValueError` on `transform` about the number of features | Rows from a different encoder or layer than at `fit` | Fit and transform with embeddings from the same encoder |
| `identity` selected although you expected a gain | No candidate beat the input clearly on the held-out rows | See the outcome table in 3.3; check `diff_vs_identity` and `diff_lo` |
| Different kNN accuracy in `evaluate_transform` and in `selection_report_` | The evaluation registry uses k = 5, the selection k = 10 | Compare numbers computed the same way |
| `selection_report_` is `None` | No selection ran: a fixed `method`, or no labels | Expected; the report exists only after `method="auto"` with labels |
| Crash with "OMP: Error #15" when using FAISS | Two OpenMP runtimes in one process | Use the default scikit-learn backend, or install FAISS and PyTorch from the same channel |
| Batch size warning ("exceeds the number of training samples") | Fewer training rows than `batch_size` | Harmless; one batch per epoch is used |

---
## References

Agrawal, K. K., Mondal, A. K., Ghosh, A., & Richards, B. A. (2022). α-ReQ: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. In *Advances in Neural Information Processing Systems* (Vol. 35).

Albergante, L., Bac, J., & Zinovyev, A. (2019). Estimating the effective dimension of large biological datasets using Fisher separability analysis. In *2019 International Joint Conference on Neural Networks (IJCNN)*. IEEE.

Amsaleg, L., Chelly, O., Furon, T., Girard, S., Houle, M. E., Kawarabayashi, K., & Nett, M. (2015). Estimating local intrinsic dimensionality. In *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 29–38). ACM.

Beyer, K. S., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? In *Proceedings of the 7th International Conference on Database Theory (ICDT)* (pp. 217–235). Springer.

Ceruti, C., Bassis, S., Rozza, A., Lombardi, G., Casiraghi, E., & Campadelli, P. (2012). DANCo: An intrinsic dimensionality estimator exploiting angle and norm concentration. In *Proceedings of the 21st International Conference on Pattern Recognition (ICPR 2012)* (pp. 1045–1048). IEEE.

Chen, M. F., Fu, D. Y., Narayan, A., Zhang, M., Song, Z., Fatahalian, K., & Ré, C. (2022). Perfectly balanced: Improving transfer and robustness of supervised contrastive learning. In *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)*, PMLR 162.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119.

Conneau, A., Lample, G., Ranzato, M., Denoyer, L., & Jégou, H. (2018). Word translation without parallel data. In *International Conference on Learning Representations (ICLR 2018)*.

Cristianini, N., Shawe-Taylor, J., Elisseeff, A., & Kandola, J. S. (2001). On kernel-target alignment. In *Advances in Neural Information Processing Systems* (Vol. 14, pp. 367–373). MIT Press.

Denti, F., Doimo, D., Laio, A., & Mira, A. (2022). The generalized ratios intrinsic dimension estimator. *Scientific Reports*, 12, 20005.

DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with Cutout. *arXiv preprint arXiv:1708.04552*.

Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7, 12140.

Feldbauer, R., & Flexer, A. (2019). A comprehensive empirical comparison of hubness reduction in high-dimensional spaces. *Knowledge and Information Systems*, 59, 137–166.

Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple contrastive learning of sentence embeddings. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)* (pp. 6894–6910).

Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of strange attractors. *Physica D: Nonlinear Phenomena*, 9(1–2), 189–208.

Hara, K., Suzuki, I., Kobayashi, K., Fukumizu, K., & Radovanović, M. (2016). Flattening the density gradient for eliminating spatial centrality to reduce hubness. In *Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016)* (pp. 1659–1665).

Johnsson, K., Soneson, C., & Fontes, M. (2015). Low bias local intrinsic dimension estimation from expected simplex skewness. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 37(1), 196–202.

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised contrastive learning. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 18661–18673). Curran Associates.

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022). Matryoshka representation learning. In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 30233–30249).

Levina, E., & Bickel, P. J. (2005). Maximum likelihood estimation of intrinsic dimension. In *Advances in Neural Information Processing Systems* (Vol. 17, pp. 777–784). MIT Press.

Liu, T., Ungar, L., & Sedoc, J. (2019). Unsupervised post-processing of word vectors via conceptor negation. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33).

Mu, J., Bhat, S., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. In *International Conference on Learning Representations (ICLR 2018)*.

Radovanović, M., Nanopoulos, A., & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research*, 11, 2487–2531.

Raunak, V., Gupta, V., & Metze, F. (2019). Effective dimensionality reduction for word embeddings. In *Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP)*.

Rudman, W., Gillman, N., Rayne, T., & Eickhoff, C. (2022). IsoScore: Measuring the uniformity of embedding space utilization. In *Findings of the Association for Computational Linguistics: ACL 2022*.

Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). Local and global scaling reduce hubs in space. *Journal of Machine Learning Research*, 13, 2871–2902.

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015)* (pp. 815–823). IEEE.

Timkey, W., & van Schijndel, M. (2021). All bark and no bite: Rogue dimensions in transformer language models obscure representational quality. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

Wang, T., & Isola, P. (2020). Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119.

Yeh, C.-H., Hong, C.-Y., Hsu, Y.-C., Liu, T.-L., Chen, Y., & LeCun, Y. (2022). Decoupled contrastive learning. In *European Conference on Computer Vision (ECCV 2022)* (pp. 668–684). Springer.

Zha, K., Cao, P., Son, J., Yang, Y., & Katabi, D. (2023). Rank-N-Contrast: Learning continuous representations for regression. In *Advances in Neural Information Processing Systems* (Vol. 36). Curran Associates.

Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. In *International Conference on Learning Representations (ICLR 2018)*.
