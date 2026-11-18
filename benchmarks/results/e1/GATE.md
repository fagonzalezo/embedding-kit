# Phase 3 decision gate (E1)

For each dataset:

- **Headline:** test value of the primary metric for raw, the best training-free method, and each
  ``EmbedKit`` family, every one selected on validation as in ``report.py``.
- **Gate:** the self-supervised ``EmbedKit`` configuration best on validation (default or fixed d)
  against the training-free baseline best on validation among those with the same output
  dimension. Paired bootstrap interval over test points (``compare``). A win is a difference above
  0 whose interval excludes 0; a loss is an interval below 0. For context, the best training-free
  baseline at a lower dimension, and trustworthiness against PCA at the same d (outcome B's
  condition).

## Headline: test value of the primary metric, each family selected on validation

| Dataset | metric | raw | training-free | embedkit (default) | embedkit (fixed d) | embedkit supervised |
|---|---|---|---|---|---|---|
| banking77_mpnet | knn_accuracy | 0.931 (raw, d = 768) | 0.928 (conceptor, d = 768) | 0.899 ± 0.007 (embedkit, d = 12) | 0.917 ± 0.003 (embedkit-d64, d = 64) | 0.930 ± 0.004 (embedkit_supervised, d = 12) |
| cifar100_dinov2 | knn_accuracy | 0.879 (raw, d = 768) | 0.882 (abtt-3, d = 768) | 0.858 ± 0.003 (embedkit, d = 28) | 0.863 ± 0.001 (embedkit-d64, d = 64) | 0.888 ± 0.008 (embedkit_supervised, d = 28) |
| digits | knn_accuracy | 0.981 (raw, d = 64) | 0.972 (center_l2, d = 64) | 0.946 ± 0.023 (embedkit, d = 11) | 0.966 ± 0.011 (embedkit-d32, d = 32) | 0.973 ± 0.016 (embedkit_supervised, d = 11) |
| pets_resnet50 | knn_accuracy | 0.901 (raw, d = 2048) | 0.901 (raw, d = 2048) | 0.894 ± 0.003 (embedkit, d = 22) | 0.900 ± 0.003 (embedkit-d64, d = 64) | 0.915 ± 0.004 (embedkit_supervised, d = 22) |
| stl10_clip | knn_accuracy | 0.983 (raw, d = 512) | 0.976 (abtt-5, d = 512) | 0.978 ± 0.001 (embedkit, d = 24) | 0.979 ± 0.001 (embedkit-d128, d = 128) | 0.985 ± 0.001 (embedkit_supervised, d = 24) |
| stsb_mpnet | sts_spearman | 0.834 (raw, d = 768) | 0.839 (center_l2, d = 768) | 0.781 ± 0.008 (embedkit, d = 26) | 0.797 ± 0.004 (embedkit-d128, d = 128) | n/a |
| synthetic_anisotropic | knn_accuracy | 0.746 (raw, d = 512) | 0.913 (abtt-3, d = 512) | 0.720 ± 0.010 (embedkit, d = 15) | 0.744 ± 0.010 (embedkit-d128, d = 128) | 0.921 ± 0.004 (embedkit_supervised, d = 15) |

## Gate: self-supervised EmbedKit vs the best training-free baseline at the same d

| Dataset | EmbedKit (val-selected) | baseline at same d | difference [95% CI] | verdict | best baseline at lower d | trustworthiness, EmbedKit vs PCA same d |
|---|---|---|---|---|---|---|
| banking77_mpnet | embedkit-d64: 0.917 ± 0.003 | pca-64: 0.919 | -0.002 [-0.008, +0.004] | tie | pca-32: 0.908 | 0.997 vs 1.000 |
| cifar100_dinov2 | embedkit-d64: 0.863 ± 0.001 | pca_whiten-64: 0.866 | -0.003 [-0.008, +0.002] | tie | pca_whiten-32: 0.831 | 0.996 vs 0.993 |
| digits | embedkit-d32: 0.966 ± 0.011 | pca-32: 0.976 | -0.010 [-0.017, -0.002] | loss | pca-16: 0.972 | 0.995 vs 1.000 |
| pets_resnet50 | embedkit-d64: 0.900 ± 0.003 | pca-64: 0.901 | -0.001 [-0.006, +0.005] | tie | pca-32: 0.902 | 0.990 vs 0.997 |
| stl10_clip | embedkit-d128: 0.979 ± 0.001 | pca-128: 0.984 | -0.005 [-0.007, -0.003] | loss | pca-32: 0.984 | 0.995 vs 1.000 |
| stsb_mpnet | embedkit-d128: 0.797 ± 0.004 | pca-128: 0.833 | -0.036 | no interval | pca_whiten-64: 0.818 | 0.978 vs 0.985 |
| synthetic_anisotropic | embedkit-d128: 0.744 ± 0.010 | pca_whiten-128: 0.910 | -0.165 [-0.180, -0.151] | loss | pca_whiten-32: 0.892 | 0.998 vs 1.000 |

Verdicts: 3 tie, 3 loss, 1 no interval over 7 datasets.
