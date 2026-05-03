# EmbedKit ML Workflow Integration Guide

Patterns for integrating EmbedKit into real-world ML pipelines.

---

## HuggingFace Transformers

The most common pattern: extract embeddings from a pretrained model, analyze them
with EmbedKit, refine them, then use them for downstream tasks.

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

texts = ["sentence one", "sentence two", ...]   # your corpus
embeddings = []
with torch.no_grad():
    for batch in chunks(texts, size=64):
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        out = model(**enc)
        embeddings.append(mean_pool(out, enc["attention_mask"]).numpy())
X = np.concatenate(embeddings)   # (n, 384)

# ── Step 2: Diagnose ───────────────────────────────────────
report = EmbedKitAnalyzer(k=10).fit(X)
report.print_summary()

# ── Step 3: Refine ─────────────────────────────────────────
ek = EmbedKit(mode="self_supervised", epochs=100, device="cuda")
X_refined = ek.fit_transform(X)   # (n, suggested_target_dim)

# ── Step 4: Use for downstream tasks ──────────────────────
# Classification, retrieval, clustering, etc.
```

**With class labels (supervised):**

```python
ek = EmbedKit(mode="supervised", epochs=100, device="cuda")
X_refined = ek.fit_transform(X, y=y)   # y: integer class indices
```

---

## PyTorch Training Loop Integration

Use `EmbeddingRefiner` as a projection head on top of your encoder during contrastive
pre-training, or as a post-processing step after training.

### Option A: Post-processing (simplest)

Train your encoder as usual, extract embeddings at the end, then refine:

```python
# After encoder training:
X = extract_embeddings(encoder, dataloader)
X_refined = EmbedKit(mode="self_supervised", epochs=50).fit_transform(X)
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
    normalize=True,
)

aug = GaussianNoise(std=0.05, adaptive=True)
loss_fn = NTXentLoss(temperature=0.07)

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(projector.parameters()),
    lr=3e-4, weight_decay=1e-4,
)

for epoch in range(epochs):
    for batch in dataloader:
        x = encoder(batch)                        # (B, D) embeddings
        x_t = torch.from_numpy(x.cpu().numpy()) if not isinstance(x, torch.Tensor) else x
        xi, xj = aug(x_t)
        zi, zj = projector(xi), projector(xj)
        loss = loss_fn(zi, zj)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Monitoring training with EmbedKit diagnostics

`Trainer` runs `EmbedKitAnalyzer` on the refined embeddings every `eval_every` epochs
and stores metrics in `trainer.history`:

```python
from embedkit.improvement.trainer import Trainer

trainer = Trainer(
    model=projector,
    augmentation=aug,
    loss=loss_fn,
    epochs=200,
    eval_every=10,
    early_stopping_patience=20,    # stop if no improvement for 20 evals
    device="cuda",
)
trainer.fit(X)

# Inspect training history
history = trainer.history   # list of dicts with "loss" and metric keys
import pandas as pd
df = pd.DataFrame(history)
df[["epoch", "loss", "hubness_k_skewness", "isotropy_score"]].plot(x="epoch")
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
    def __init__(self, mode="self_supervised", epochs=100, target_dim="auto",
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
    ("embedkit", EmbedKitTransformer(mode="supervised", epochs=50)),
    ("clf",      LogisticRegression(max_iter=500)),
])
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)
```

For unsupervised use, pass `mode="self_supervised"` and `y=None` to `fit`.

---

## RAG / Vector Store Retrieval

Hubness directly degrades retrieval quality — hub documents are returned for almost
every query. EmbedKit can diagnose and fix this before indexing.

### Workflow

```python
import numpy as np
from embedkit import EmbedKit, EmbedKitAnalyzer

# ── 1. Load your corpus embeddings ────────────────────────
doc_embeddings = np.load("corpus_embeddings.npy")   # (n_docs, D)
query_embeddings = np.load("query_embeddings.npy")  # (n_queries, D)

# ── 2. Diagnose ────────────────────────────────────────────
report = EmbedKitAnalyzer(k=10).fit(doc_embeddings)
report.print_summary()
# Look at: k_skewness, hub_contamination, antihub_ratio

# ── 3. Refine (train on corpus, apply to both) ─────────────
ek = EmbedKit(mode="self_supervised", epochs=100)
ek.fit(doc_embeddings)                                # train on corpus
doc_refined   = ek.transform(doc_embeddings)          # (n_docs, target_dim)
query_refined = ek.transform(query_embeddings)        # (n_queries, target_dim)

# ── 4. Re-index and query ──────────────────────────────────
import faiss

index = faiss.IndexFlatIP(doc_refined.shape[1])
# Normalize for cosine similarity
norms = np.linalg.norm(doc_refined, axis=1, keepdims=True)
index.add((doc_refined / norms).astype("float32"))

q_norm = query_refined / np.linalg.norm(query_refined, axis=1, keepdims=True)
D, I = index.search(q_norm.astype("float32"), k=10)   # top-10 docs per query

# ── 5. Persist the refiner for production ─────────────────
ek.save("rag_refiner/")
```

**Key insight:** Always train EmbedKit on the document corpus only, not on queries.
Apply `transform()` to both at query time.

### With a vector database (e.g., ChromaDB, Weaviate, Pinecone)

1. Extract all stored embeddings (or a representative sample of ≥ 5 000).
2. Fit `EmbedKit` on them.
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
ek = EmbedKit.load("my_model/")

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
model = EmbeddingRefiner(
    input_dim=bundle["config"]["input_dim"],
    target_dim=bundle["config"]["target_dim"],
    hidden_dim=bundle["config"]["hidden_dim"],
    n_layers=bundle["config"]["n_layers"],
)
model.load_state_dict(bundle["state_dict"])
model.eval()

dummy = torch.randn(1, bundle["config"]["input_dim"])
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

| Source | Typical dim | Common issue | Suggested approach |
|---|---|---|---|
| BERT / RoBERTa (CLS) | 768 | Anisotropy, moderate hubness | `mode="self_supervised"`, `AlignUniformLoss` |
| Sentence-Transformers | 384–768 | Generally healthy; mild hubness | `EmbeddingMixup`, `NTXentLoss` |
| OpenAI `text-embedding-*` | 1536–3072 | Distance collapse at high dim | Aggressive `target_dim` reduction |
| CLIP vision | 512–1024 | Anisotropy | `AlignUniformLoss` |
| Word2Vec / GloVe | 100–300 | Low intrinsic dim | Modest reduction, `KNNPairs` |
| Random / untrained | any | Everything | Unlikely to help — train encoder first |
