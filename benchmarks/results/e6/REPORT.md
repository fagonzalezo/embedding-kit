# Experiment e6

Every family (``e6_<arm>``) is reduced to its configuration best on validation (auto d or a
fixed d), as in E1. Each candidate family is then compared, per dataset, with:

- the **reference** family (the previous default): paired bootstrap over test points of the
  primary metric. The phase 5 rule accepts a change when it wins (difference above 0, interval
  excluding 0) on at least half the datasets.
- the **best training-free baseline at the same d** (PCA or PCA-whitening, best on validation):
  the phase 3 gate, repeated for the new arm.

## Summary against `e6_mlp` (wins / ties / losses over datasets)

| Family | vs reference | vs best baseline at the same d | accepted |
|---|---|---|---|
| e6_linear | 5 / 1 / 0 (1 n/a) | 1 / 4 / 1 (1 n/a) | yes |
| e6_mlp | (reference) | 0 / 3 / 3 (1 n/a) | |
| e6_residual | 5 / 1 / 0 (1 n/a) | 2 / 2 / 2 (1 n/a) | yes |

A family is accepted when it wins against `e6_mlp` on at least half of the 7 datasets.

## banking77_mpnet (knn_accuracy; raw 0.931)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_linear | linear-d64 | 64 | 0.926 ± 0.001 | +0.006 [+0.001, +0.010] win | pca-64: 0.919 | +0.007 [+0.000, +0.014] win |
| e6_mlp | mlp-d64 | 64 | 0.921 ± 0.002 | reference | pca-64: 0.919 | +0.002 [-0.004, +0.008] tie |
| e6_residual | residual-d64 | 64 | 0.927 ± 0.002 | +0.006 [+0.003, +0.009] win | pca-64: 0.919 | +0.008 [+0.001, +0.014] win |

## cifar100_dinov2 (knn_accuracy; raw 0.879)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_linear | linear-d128 | 128 | 0.875 ± 0.000 | +0.012 [+0.008, +0.015] win | pca_whiten-128: 0.878 | -0.003 [-0.008, +0.001] tie |
| e6_mlp | mlp-d128 | 128 | 0.863 ± 0.002 | reference | pca_whiten-128: 0.878 | -0.015 [-0.019, -0.011] loss |
| e6_residual | residual-d128 | 128 | 0.875 ± 0.002 | +0.012 [+0.009, +0.015] win | pca_whiten-128: 0.878 | -0.003 [-0.007, +0.001] tie |

## digits (knn_accuracy; raw 0.981)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_linear | linear-d32 | 32 | 0.976 ± 0.001 | +0.000 [-0.008, +0.008] tie | pca-32: 0.976 | +0.000 [-0.004, +0.005] tie |
| e6_mlp | mlp-d32 | 32 | 0.976 ± 0.003 | reference | pca-32: 0.976 | -0.000 [-0.007, +0.008] tie |
| e6_residual | residual-d32 | 32 | 0.984 ± 0.001 | +0.008 [+0.002, +0.014] win | pca-32: 0.976 | +0.008 [+0.002, +0.016] win |

## pets_resnet50 (knn_accuracy; raw 0.901)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_linear | linear-d64 | 64 | 0.905 ± 0.002 | +0.009 [+0.003, +0.013] win | pca-64: 0.901 | +0.004 [-0.002, +0.010] tie |
| e6_mlp | mlp-d128 | 128 | 0.896 ± 0.003 | reference | pca-128: 0.901 | -0.005 [-0.011, +0.001] tie |
| e6_residual | residual-d128 | 128 | 0.898 ± 0.002 | +0.001 [-0.003, +0.006] tie | pca-128: 0.901 | -0.003 [-0.010, +0.003] tie |

## stl10_clip (knn_accuracy; raw 0.983)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_linear | linear-d128 | 128 | 0.984 ± 0.000 | +0.005 [+0.003, +0.007] win | pca-128: 0.984 | -0.001 [-0.003, +0.002] tie |
| e6_mlp | mlp-d128 | 128 | 0.979 ± 0.001 | reference | pca-128: 0.984 | -0.006 [-0.008, -0.003] loss |
| e6_residual | residual-d128 | 128 | 0.982 ± 0.000 | +0.003 [+0.002, +0.005] win | pca-128: 0.984 | -0.002 [-0.005, -0.000] loss |

## stsb_mpnet (sts_spearman; raw 0.834)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_linear | linear-d128 | 128 | 0.828 ± 0.000 | +0.030 n/a | pca-128: 0.833 | -0.004 n/a |
| e6_mlp | mlp-d128 | 128 | 0.798 ± 0.004 | reference | pca-128: 0.833 | -0.034 n/a |
| e6_residual | residual-d128 | 128 | 0.822 ± 0.001 | +0.024 n/a | pca-128: 0.833 | -0.010 n/a |

## synthetic_anisotropic (knn_accuracy; raw 0.746)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_linear | linear-d128 | 128 | 0.836 ± 0.001 | +0.075 [+0.064, +0.086] win | pca_whiten-128: 0.910 | -0.073 [-0.087, -0.061] loss |
| e6_mlp | mlp-d128 | 128 | 0.761 ± 0.006 | reference | pca_whiten-128: 0.910 | -0.149 [-0.163, -0.135] loss |
| e6_residual | residual-d32 | 32 | 0.821 ± 0.001 | +0.060 [+0.050, +0.069] win | pca_whiten-32: 0.892 | -0.071 [-0.085, -0.058] loss |
