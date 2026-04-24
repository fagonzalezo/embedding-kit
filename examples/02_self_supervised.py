"""Example 2: Self-supervised embedding refinement with EmbedKit."""

import numpy as np

from embedkit import EmbedKit, EmbedKitAnalyzer
from embedkit.visualization.embedding_viz import EmbeddingVisualizer

rng = np.random.default_rng(42)
n, d = 400, 64
scale = np.ones(d)
scale[:4] = 12.0
X = (rng.standard_normal((n, d)) * scale).astype(np.float32)

print("=== Before refinement ===")
report_before = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X)
report_before.print_summary()

print("\nRefining embedding (self-supervised) ...")
ek = EmbedKit(mode="self_supervised", epochs=50, eval_every=10, target_dim="auto")
X_refined = ek.fit_transform(X)
print(f"Refined shape: {X_refined.shape}")

print("\n=== After refinement ===")
report_after = EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(X_refined)
report_after.print_summary()

print(f"\nk_skewness:  {report_before.hubness.k_skewness:.3f} → {report_after.hubness.k_skewness:.3f}")
print(f"uniformity:  {report_before.geometry.uniformity.uniformity:.4f} → {report_after.geometry.uniformity.uniformity:.4f}")
print(f"severity:    {report_before.severity} → {report_after.severity}")

viz = EmbeddingVisualizer(method="pca", random_state=0)
fig = viz.plot_comparison(X, X_refined)
fig.savefig("before_after.png", dpi=120)
print("\nSaved before_after.png")

fig2 = viz.plot_training_trajectory(ek._trainer.history)
fig2.savefig("training_trajectory.png", dpi=120)
print("Saved training_trajectory.png")
