"""Example 3: Low-level supervised refinement with CombinedLoss.

Dataset: 8 classes in a 64-D space where only 8 dims carry class signal and
56 dims are high-variance noise (std=3 vs signal std=1). A random orthogonal
rotation mixes signal and noise, so raw kNN is mediocre. The supervised
refiner learns to project onto the discriminative subspace.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from embedkit.improvement import (
    EmbeddingRefiner,
    Trainer,
    CompositeAugmentation,
    GaussianNoise,
    FeatureDropout,
    SupConLoss,
    AlignUniformLoss,
    CombinedLoss,
)
from embedkit.analysis.report import EmbedKitAnalyzer

# --- Hard synthetic dataset --------------------------------------------------
rng = np.random.default_rng(0)
n_classes, n_informative, n_noise = 8, 8, 56
n_per_class = 75  # 600 total

means = rng.normal(scale=1.5, size=(n_classes, n_informative)).astype(np.float32)
X_info = np.vstack([
    means[c] + rng.normal(scale=1.0, size=(n_per_class, n_informative))
    for c in range(n_classes)
]).astype(np.float32)
y = np.repeat(np.arange(n_classes), n_per_class)

# Noise dims dominate the L2 metric
X_noise = rng.normal(scale=3.0, size=(n_classes * n_per_class, n_noise)).astype(np.float32)
X_full = np.concatenate([X_info, X_noise], axis=1)

# Random rotation: signal is no longer axis-aligned
Q, _ = np.linalg.qr(rng.normal(size=(64, 64)))
X = (X_full @ Q.astype(np.float32))
# -----------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# kNN accuracy on raw embeddings
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
acc_raw = knn_raw.score(X_test, y_test)
print(f"Raw kNN accuracy: {acc_raw:.3f}")

print("\n=== Raw embedding analysis ===")
EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X_train).print_summary()

# Build composite augmentation and combined loss
aug = CompositeAugmentation([GaussianNoise(std=0.05), FeatureDropout(p=0.1)])
loss_fn = CombinedLoss([(SupConLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.3)])

model = EmbeddingRefiner(input_dim=64, target_dim=16, hidden_dim=256, n_layers=2)
trainer = Trainer(
    model=model,
    augmentation=aug,
    loss=loss_fn,
    epochs=80,
    batch_size=128,
    optimizer="adam",
    lr=3e-4,
    scheduler="cosine",
    eval_every=20,
    eval_metrics=["uniformity", "k_skewness"],
    random_state=42,
)

print("\nTraining ...")
trainer.fit(X_train, y=y_train)

X_train_ref = trainer.transform(X_train)
X_test_ref = trainer.transform(X_test)

knn_ref = KNeighborsClassifier(n_neighbors=5).fit(X_train_ref, y_train)
acc_ref = knn_ref.score(X_test_ref, y_test)
print(f"Refined kNN accuracy: {acc_ref:.3f}")
print(f"Improvement: {(acc_ref - acc_raw) * 100:+.1f}%")

print("\n=== Refined embedding analysis ===")
EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X_train_ref).print_summary()
