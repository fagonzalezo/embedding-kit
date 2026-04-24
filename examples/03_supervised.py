"""Example 3: Low-level supervised refinement with CombinedLoss."""

import numpy as np
from sklearn.datasets import make_blobs
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

# Synthetic 10-class blobs in 64-D
X, y = make_blobs(n_samples=600, n_features=64, centers=10, random_state=0)
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# kNN accuracy on raw embeddings
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
acc_raw = knn_raw.score(X_test, y_test)
print(f"Raw kNN accuracy: {acc_raw:.3f}")

# Build composite augmentation and combined loss
aug = CompositeAugmentation([GaussianNoise(std=0.05), FeatureDropout(p=0.1)])
loss_fn = CombinedLoss([(SupConLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.3)])

model = EmbeddingRefiner(input_dim=64, target_dim=32, hidden_dim=256, n_layers=2)
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
report = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X_train_ref)
report.print_summary()
