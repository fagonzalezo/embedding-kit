# EmbedKit ML Workflow Integration Guide

Patterns for integrating EmbedKit into real-world ML pipelines.

---

## Recommended workflow

EmbedKit chooses a transform by validation, not by diagnosis (`docs/manual.md`, "How EmbedKit
decides"):

1. **Start from the default.** `EmbedKit().fit_transform(X)` without labels returns the input
   centred and L2-normalised. With labels, `EmbedKit().fit(X, y)` selects among `Identity`,
   centring, PCA and (supervised or regression) the refiner on held-out rows and keeps `Identity`
   unless a candidate is significantly better. Inspect `selected_` and `selection_report_`.
2. **Check on your own task.** Fit on the training split, then score the test split against the raw
   embeddings with `evaluate_transform` and `compare` (paired bootstrap interval). Adopt a transform
   only if the interval excludes 0. The selection scores by kNN; if the embedding feeds a linear
   classifier, score `"linear_probe_c1"` too.
3. **Tune only with held-out data.** If the refiner came close but was not selected, tune it as in
   `docs/manual.md` section 1.2 (three splits, one factor at a time, several seeds).
4. **For Euclidean search, check hubness.** Run `EmbedKitAnalyzer(metric="euclidean")`; if
   `report.recommendations` names a hubness-reducing search, try `embedkit.hubness_reduction`
   (`MutualProximity`, `CSLS`) on the gallery and compare with plain search the same way. Every
   other reading of the report is description: E4 found that none predicted which fix helps.
5. **Persist** with `ek.save(...)`; fit on the corpus only and apply `transform` to queries.

---

## HuggingFace Transformers

The most common pattern: extract embeddings from a pretrained model, transform them with
EmbedKit, then use them for downstream tasks.

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from embedkit import EmbedKit, EmbedKitAnalyzer

# ── Step 1: Extract embeddings ─────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()

def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]

texts = ["sentence one", "sentence two", ...]   # your corpus
embeddings = []
with torch.no_grad():
    for batch in chunks(texts, size=64):
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        out = model(**enc)
        embeddings.append(mean_pool(out, enc["attention_mask"]).numpy())
X = np.concatenate(embeddings)   # (n, 384)

# ── Step 2: Transform ──────────────────────────────────────
ek = EmbedKit()                   # no labels: centred and L2-normalised
X_refined = ek.fit_transform(X)   # (n, 384)

# ── Step 3 (optional): Describe the space ─────────────────
report = EmbedKitAnalyzer(k=10, metric="cosine").fit(X)   # the metric you search with
report.print_summary()            # description; no transform is chosen from it

# ── Step 4: Use for downstream tasks ──────────────────────
# Classification, retrieval, clustering, etc.
```

**With class labels:**

```python
ek = EmbedKit(epochs=100, device="cuda").fit(X, y)   # y: class labels; selection on held-out rows
X_refined = ek.transform(X)
print(ek.selected_)               # "identity" when nothing was significantly better
print(ek.selection_report_)
```

---

## PyTorch Training Loop Integration

Use `EmbeddingRefiner` as a projection head on top of your encoder during contrastive
pre-training, or as a post-processing step after training.

### Option A: Post-processing (simplest)

Train your encoder as usual, extract embeddings at the end, then transform:

```python
# After encoder training:
X = extract_embeddings(encoder, dataloader)
X_refined = EmbedKit().fit_transform(X)   # or EmbedKit().fit(X, y) with labels
```

### Option B: EmbeddingRefiner as projection head

```python
import torch
import torch.nn as nn
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.losses import NTXentLoss
from embedkit.improvement.augmentation import GaussianNoise

encoder = MyEncoder(...)                          # your existing model
projector = EmbeddingRefiner(
    input_dim=encoder.output_dim,
    target_dim=64,
    hidden_dim=256,
    n_layers=2,
    architecture="mlp",          # the linear and residual variants start from PCA of fixed inputs
    normalize_output=True,
)

aug = GaussianNoise(std=0.05, adaptive=True)
loss_fn = NTXentLoss(temperature=0.07)

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(projector.parameters()),
    lr=3e-4, weight_decay=1e-4,
)

for epoch in range(epochs):
    for batch in dataloader:
        x = encoder(batch)                        # (B, D) embeddings, a torch.Tensor
        xi, xj = aug(x)
        zi, zj = projector(xi), projector(xj)
        loss = loss_fn(zi, zj)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Monitoring training with EmbedKit diagnostics

`Trainer` evaluates the monitors named in `eval_metrics` (`"uniformity"`, `"isotropy"`,
`"k_skewness"`, and with labels or a validation split also `"val_knn_accuracy"`, `"knn_overlap"`
and others in `trainer.MONITORS`) on the refined embeddings every `eval_every` epochs and stores
them in `trainer.history`, a dict of lists. In E2 no label-free monitor tracked test accuracy within
a run, so use them to look at training, not to stop it:

```python
from embedkit.improvement.trainer import Trainer

trainer = Trainer(
    model=projector,
    augmentation=aug,
    loss=loss_fn,
    epochs=200,
    eval_every=10,
    eval_metrics=["k_skewness", "uniformity"],
    device="cuda",
)
trainer.fit(X)

# Inspect training history: "loss" and "lr" per epoch, each monitor per evaluation
history = trainer.history
import pandas as pd
pd.DataFrame({"epoch": history["eval_epoch"], "k_skewness": history["k_skewness"],
              "uniformity": history["uniformity"]}).plot(x="epoch")
```

---

## scikit-learn Pipeline

Wrap `EmbedKit` as a sklearn transformer to slot it into a `Pipeline`:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from embedkit import EmbedKit


class EmbedKitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode="auto", epochs=100, target_dim="auto",
                 device=None, random_state=42):
        self.mode = mode
        self.epochs = epochs
        self.target_dim = target_dim
        self.device = device
        self.random_state = random_state
        self._ek = None

    def fit(self, X, y=None):
        self._ek = EmbedKit(
            mode=self.mode,
            epochs=self.epochs,
            target_dim=self.target_dim,
            device=self.device,
            random_state=self.random_state,
        )
        self._ek.fit(X, y=y)
        return self

    def transform(self, X):
        return self._ek.transform(X)


pipe = Pipeline([
    ("embedkit", EmbedKitTransformer(epochs=50)),   # mode="auto": selection with labels
    ("clf",      LogisticRegression(max_iter=500)),
])
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)
```

Inside a `Pipeline`, `fit` receives `y`, so `mode="auto"` selects with labels (and falls back to
`Identity` when nothing is significantly better); without `y` it returns the input centred and
L2-normalised. Fitting inside cross-validation keeps the selection hold-out inside each training fold.
The selection scores candidates by kNN, not by the downstream classifier: in E10 a linear probe on
its output was worse than on the input on 5 of 64 sets (0.2 to 0.8 points). Compare the pipeline
with one without `EmbedKitTransformer` on the same folds.

---

## RAG / Vector Store Retrieval

Hubness can degrade retrieval: hub documents appear in many neighbour lists. EmbedKit can measure
it before indexing, and for Euclidean search it can reduce it. The evidence for the fix is modest
(E4: significantly better than raw on 14 of 35 sets under Euclidean kNN and worse on 3, median
+0.4 points; on cosine search the report makes no recommendation), so compare on your own queries.

### Workflow

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

# ── 1. Load your corpus embeddings ────────────────────────
doc_embeddings = np.load("corpus_embeddings.npy")   # (n_docs, D)
query_embeddings = np.load("query_embeddings.npy")  # (n_queries, D)

# ── 2. Transform (fit on the corpus, apply to both) ────────
ek = EmbedKit()                                       # centred and L2-normalised
ek.fit(doc_embeddings)                                # fit on corpus
doc_refined   = ek.transform(doc_embeddings)          # (n_docs, D), unit-norm rows
query_refined = ek.transform(query_embeddings)        # (n_queries, D)

# ── 3. Index and query (cosine = inner product on unit-norm rows) ──
import faiss

index = faiss.IndexFlatIP(doc_refined.shape[1])
index.add(doc_refined.astype("float32"))
D, I = index.search(query_refined.astype("float32"), k=10)   # top-10 docs per query

# ── 4. Euclidean search only: check hubness on the corpus ──
report = EmbedKitAnalyzer(k=10, metric="euclidean").fit(doc_embeddings)
print(report.recommendations)     # names a hubness-reducing search when k-skewness or the
                                  # antihub ratio is above the null; otherwise "No recommendation"
from embedkit.hubness_reduction import MutualProximity
mp = MutualProximity(metric="euclidean").fit(doc_embeddings)
dist, idx = mp.kneighbors(query_embeddings, n_neighbors=10)   # neighbours among the corpus rows

# ── 5. Persist for production ──────────────────────────────
ek.save("rag_refiner/")
```

**Key insight:** Always fit EmbedKit on the document corpus only, not on queries.
Apply `transform()` to both at query time. The default does not change a cosine ranking beyond
removing the corpus mean, so it is cheap to try and easy to compare with the raw embeddings.
Compare either search with plain search on queries with known relevant documents
(`embedkit.evaluation.metrics.recall_at_k`) before switching. `faiss` is optional
(`pip install embedkit[faiss]`); any exact inner-product search works the same way.

### With a vector database (e.g., ChromaDB, Weaviate, Pinecone)

1. Extract all stored embeddings (or a representative sample of ≥ 5 000).
2. Fit `EmbedKit` on them (with labels, `selected_` may be `"identity"`, in which case there is
   nothing to re-index).
3. Re-embed everything with `transform()`.
4. Re-upload the refined embeddings under a new collection / index.
5. At query time, embed the query with the same sentence encoder, then call
   `ek.transform(query_embedding[None, :])` before sending to the database.

---

## Batch / Production Inference

### Offline batch transform

```python
# Load fitted model
from embedkit import EmbedKit
ek = EmbedKit.load("my_model/")     # device=None: CUDA, then MPS, then CPU

# Transform in batches to avoid OOM
batch_size = 2048
results = []
for i in range(0, len(X), batch_size):
    results.append(ek.transform(X[i : i + batch_size]))
X_refined = np.concatenate(results)
```

### ONNX export for low-latency serving

```python
import torch
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.utils.io import load_bundle

bundle = load_bundle("my_model/")
assert bundle["state_dict"] is not None   # Identity and CenterNormalize bundles have no model.pt
cfg = bundle["config"]
model = EmbeddingRefiner(
    input_dim=cfg["input_dim"],
    target_dim=cfg["target_dim"],
    hidden_dim=cfg["hidden_dim"],
    n_layers=cfg["n_layers"],
    architecture=cfg.get("architecture", "mlp"),
)
model.load_state_dict(bundle["state_dict"])
model.eval()

dummy = torch.randn(1, cfg["input_dim"])
torch.onnx.export(
    model, dummy, "embedkit_refiner.onnx",
    input_names=["embedding"],
    output_names=["refined"],
    dynamic_axes={"embedding": {0: "batch"}, "refined": {0: "batch"}},
    opset_version=17,
)
```

Load with `onnxruntime` for inference:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("embedkit_refiner.onnx")
X_refined = sess.run(["refined"], {"embedding": X.astype(np.float32)})[0]
```

---

## Tips for Common Embedding Sources

These are starting points, not benchmark results (E4 covered sentence encoders bge, e5, MiniLM and
MPNet and image encoders CLIP, DINOv2 and ResNet-50). Whether a reading such as anisotropy or hubness
means a transform will help is not predictable from the report: test on held-out data against the raw
embeddings.

| Source | Typical dim | What to expect | Suggested approach |
|---|---|---|---|
| BERT / RoBERTa (CLS) | 768 | Often anisotropic with a large mean direction (reported in the literature; descriptive here) | Default `EmbedKit()` (centred, L2); compare with raw |
| Sentence-Transformers | 384–768 | Usually already well-behaved | Default; with labels, `EmbedKit().fit(X, y)` and read `selected_` |
| OpenAI `text-embedding-*` | 1536–3072 | High dimension | PCA to a smaller size is a candidate of the selection (64, 128, 256); do not assume it helps (PCA to 1.5 × ID lost accuracy on 32 of 35 E4 sets) |
| CLIP vision | 512–1024 | Mean vector ratio around 0.8 on the E4 CLIP sets (descriptive) | Default; compare with raw |
| Word2Vec / GloVe | 100–300 | Low dimension | If the report recommends it (Euclidean search), `MutualProximity`; `AllButTheTop` was designed for word vectors but lost on most E4 sets |
| Random / untrained | any | No structure to recover | Unlikely to help: train the encoder first |
