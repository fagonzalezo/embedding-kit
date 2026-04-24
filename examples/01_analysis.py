"""Example 1: Analyze embedding geometry with EmbedKitAnalyzer."""

import numpy as np
from sklearn.datasets import make_blobs

from embedkit import EmbedKitAnalyzer
from embedkit.visualization.plots import AnalysisPlotter

# Construct a deliberately hubbed high-dim embedding
rng = np.random.default_rng(42)
n, d = 500, 128
scale = np.ones(d)
scale[:5] = 15.0  # most variance in 5 dims → hubness
X = (rng.standard_normal((n, d)) * scale).astype(np.float32)

print(f"Dataset: {X.shape}")

report = EmbedKitAnalyzer(k=10, id_methods=["TwoNN", "MLE"]).fit(X)
report.print_summary()

print("\nDataFrame view (first 10 rows):")
print(report.to_dataframe().head(10).to_string(index=False))

plotter = AnalysisPlotter(report)
fig = plotter.plot_full_report()
fig.savefig("analysis_report.png", dpi=120)
print("\nSaved analysis_report.png")
