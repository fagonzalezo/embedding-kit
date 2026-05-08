"""Ablation 5: intrinsic-dimension estimator comparison on raw ResNet50 features.

Tests all seven skdim-backed estimators available in EmbedKit against the
participation_ratio and kernel effective_rank already in the analysis report,
to assess whether the current TwoNN-only consensus underestimates the useful
dimensionality of the Pets embedding space.

Run:
    conda run -n pytorch python examples/04f_id_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "data" / "pets_resnet50.npz"
if not CACHE.exists():
    raise FileNotFoundError(
        f"Cache not found at {CACHE}.\n"
        "Run examples/04_image_classification.py first to extract features."
    )

d = np.load(CACHE)
X_train = d["X_train"]
print(f"Train: {X_train.shape}\n")

from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator
from embedkit.analysis.geometry import IsotropyAnalyzer
from embedkit.analysis.kernel import KernelDiagnostics

# DANCo (O(n²) EM) and FisherS are prohibitively slow at n≈3700, d=2048.
# CorrInt requires a correlation-dimension scaling regime that rarely holds
# for high-d embeddings. The three remaining methods cover the main families:
# local-ratio (TwoNN), MLE (Levina-Bickel), and PCA-based (lPCA).
METHODS = ["TwoNN", "MLE", "lPCA", "MOM"]

# ---------------------------------------------------------------------------
# Run all estimators (one pass — aggregator only affects the consensus scalar)
# ---------------------------------------------------------------------------
print("Running ID estimators …\n", flush=True)
est = IntrinsicDimensionEstimator(methods=METHODS, aggregate="mean", random_state=42)
result = est.fit(X_train)

estimates = result.estimates   # dict[str, float]

# Compute the three consensus variants from the raw per-method values.
valid = {m: v for m, v in estimates.items() if not np.isnan(v)}
values = list(valid.values())
consensus_mean   = float(np.mean(values))
consensus_median = float(np.median(values))
consensus_max    = float(np.max(values))

# ---------------------------------------------------------------------------
# Reference metrics (already computed in the full analyzer; repeat here so the
# table is self-contained without the heavier full EmbedKitAnalyzer call)
# ---------------------------------------------------------------------------
print("Computing reference geometry metrics …\n")
iso_result    = IsotropyAnalyzer().fit(X_train)
participation_ratio = iso_result.participation_ratio

kernel_result = KernelDiagnostics(random_state=42).fit(X_train)
kernel_eff_rank = kernel_result.effective_rank

# ---------------------------------------------------------------------------
# Suggested target dims
# ---------------------------------------------------------------------------
def suggested_dim(consensus: float, D: int) -> int:
    return int(np.clip(round(1.5 * consensus), max(4, int(consensus)), D))

D = X_train.shape[1]
twonn_id = estimates.get("TwoNN", float("nan"))

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
print("=" * 60)
print("Intrinsic Dimension Estimator Comparison")
print(f"  Dataset: {X_train.shape[0]} samples × {D} dims")
print("=" * 60)
print(f"\n{'Method':<14} {'ID':>8}  {'×TwoNN':>8}  {'suggested_dim':>13}")
print("-" * 50)
for m in METHODS:
    v = estimates.get(m, float("nan"))
    ratio = v / twonn_id if (not np.isnan(v) and twonn_id > 0) else float("nan")
    sd = suggested_dim(v, D) if not np.isnan(v) else "—"
    ratio_str = f"{ratio:.2f}×" if not np.isnan(ratio) else "     —"
    print(f"{m:<14} {v:>8.2f}  {ratio_str:>8}  {sd:>13}")

print("-" * 50)
for label, cons in [("mean", consensus_mean), ("median", consensus_median), ("max", consensus_max)]:
    sd = suggested_dim(cons, D)
    ratio = cons / twonn_id if twonn_id > 0 else float("nan")
    ratio_str = f"{ratio:.2f}×" if not np.isnan(ratio) else "     —"
    print(f"{'[' + label + ']':<14} {cons:>8.2f}  {ratio_str:>8}  {sd:>13}")

print()
print(f"{'Reference metrics':}")
print(f"  participation_ratio (PCA-based)  : {participation_ratio:.2f}")
print(f"  kernel effective_rank            : {kernel_eff_rank:.2f}")
print(f"  current auto-config target_dim   : {suggested_dim(twonn_id, D)}  "
      f"(from TwoNN={twonn_id:.2f} × 1.5)")
print()

# ---------------------------------------------------------------------------
# Interpretation summary
# ---------------------------------------------------------------------------
spread = consensus_max - (min(v for v in values if not np.isnan(v)))
outliers = [m for m, v in valid.items() if v > 2 * twonn_id]
pr_ratio = participation_ratio / consensus_mean

print("=" * 60)
print("Interpretation")
print("=" * 60)
print(f"  Inter-method spread (max − min)  : {spread:.1f} dims")
if spread > 2 * twonn_id:
    print("  → LARGE spread: estimators disagree substantially; mean consensus is unreliable.")
elif spread > twonn_id:
    print("  → MODERATE spread: some estimators diverge meaningfully from TwoNN.")
else:
    print("  → SMALL spread: estimators broadly agree.")

if outliers:
    print(f"  Methods > 2× TwoNN              : {', '.join(outliers)}")
else:
    print("  No method reports > 2× TwoNN.")

print(f"  participation_ratio / mean ID    : {pr_ratio:.1f}×")
if pr_ratio > 3:
    print("  → participation_ratio is >3× the mean ID consensus: the 1.5× multiplier")
    print("    in suggested_target_dim likely underestimates discriminative dimensionality.")
elif pr_ratio > 1.5:
    print("  → participation_ratio exceeds ID consensus; multiplier may be too small.")
else:
    print("  → participation_ratio and ID consensus broadly consistent.")
print("=" * 60)
