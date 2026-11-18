# Experiment e6

Every family (``e6_<arm>``) is reduced to its configuration best on validation (auto d or a
fixed d), as in E1. Each candidate family is then compared, per dataset, with:

- the **reference** family (the previous default): paired bootstrap over test points of the
  primary metric. The phase 5 rule accepts a change when it wins (difference above 0, interval
  excluding 0) on at least half the datasets.
- the **best training-free baseline at the same d** (PCA or PCA-whitening, best on validation):
  the phase 3 gate, repeated for the new arm.

Nested prefixes (WP 5.3), when ``matryoshka-d<k>`` methods exist: every prefix of the one nested
model against ``linear-d<k>`` trained at that d, at each fixed d with no selection. The WP 5.3 rule
is non-inferiority: accept unless the prefix loses at some d on more than one dataset.

## Summary against `e6_linear` (wins / ties / losses over datasets)

| Family | vs reference | vs best baseline at the same d | accepted |
|---|---|---|---|
| e6_decoupled | 1 / 3 / 2 (1 n/a) | 0 / 4 / 2 (1 n/a) | no |
| e6_dropout | 1 / 1 / 4 (1 n/a) | 0 / 1 / 5 (1 n/a) | no |
| e6_linear | (reference) | 1 / 4 / 1 (1 n/a) | |
| e6_matryoshka | 0 / 6 / 0 (1 n/a) | 1 / 4 / 1 (1 n/a) | no |
| e6_mlp | 0 / 1 / 5 (1 n/a) | 0 / 3 / 3 (1 n/a) | no |
| e6_mutual | 1 / 3 / 2 (1 n/a) | 0 / 3 / 3 (1 n/a) | no |
| e6_nbrmask | 1 / 3 / 2 (1 n/a) | 0 / 4 / 2 (1 n/a) | no |
| e6_residual | 1 / 2 / 3 (1 n/a) | 2 / 2 / 2 (1 n/a) | no |

A family is accepted when it wins against `e6_linear` on at least half of the 7 datasets.

## Nested prefixes against a model trained at each d (WP 5.3)

| Dataset | d | prefix | trained at d | prefix - trained [95% CI] |
|---|---|---|---|---|
| banking77_mpnet | 32 | 0.924 ± 0.002 | 0.925 ± 0.001 | -0.001 [-0.004, +0.001] tie |
| banking77_mpnet | 64 | 0.926 ± 0.001 | 0.926 ± 0.001 | -0.000 [-0.003, +0.002] tie |
| banking77_mpnet | 128 | 0.928 ± 0.001 | 0.929 ± 0.001 | -0.001 [-0.003, +0.000] tie |
| cifar100_dinov2 | 32 | 0.854 ± 0.001 | 0.857 ± 0.001 | -0.003 [-0.006, -0.001] loss |
| cifar100_dinov2 | 64 | 0.870 ± 0.001 | 0.869 ± 0.001 | +0.001 [-0.002, +0.003] tie |
| cifar100_dinov2 | 128 | 0.875 ± 0.001 | 0.875 ± 0.000 | -0.000 [-0.002, +0.002] tie |
| digits | 32 | 0.976 ± 0.000 | 0.976 ± 0.001 | -0.000 [-0.001, +0.000] tie |
| pets_resnet50 | 32 | 0.894 ± 0.001 | 0.894 ± 0.002 | -0.000 [-0.003, +0.003] tie |
| pets_resnet50 | 64 | 0.904 ± 0.001 | 0.905 ± 0.002 | -0.001 [-0.004, +0.001] tie |
| pets_resnet50 | 128 | 0.905 ± 0.001 | 0.905 ± 0.002 | +0.001 [-0.002, +0.003] tie |
| stl10_clip | 32 | 0.983 ± 0.000 | 0.982 ± 0.000 | +0.001 [+0.000, +0.003] win |
| stl10_clip | 64 | 0.983 ± 0.001 | 0.982 ± 0.001 | +0.000 [-0.001, +0.001] tie |
| stl10_clip | 128 | 0.984 ± 0.000 | 0.984 ± 0.000 | -0.000 [-0.001, +0.001] tie |
| stsb_mpnet | 32 | 0.810 ± 0.002 | 0.810 ± 0.001 | +0.000 n/a |
| stsb_mpnet | 64 | 0.820 ± 0.001 | 0.823 ± 0.001 | -0.002 n/a |
| stsb_mpnet | 128 | 0.827 ± 0.000 | 0.828 ± 0.000 | -0.001 n/a |
| synthetic_anisotropic | 32 | 0.839 ± 0.001 | 0.838 ± 0.002 | +0.001 [-0.001, +0.003] tie |
| synthetic_anisotropic | 64 | 0.837 ± 0.001 | 0.838 ± 0.002 | -0.001 [-0.003, +0.001] tie |
| synthetic_anisotropic | 128 | 0.836 ± 0.001 | 0.836 ± 0.001 | -0.000 [-0.002, +0.001] tie |

Datasets with a loss at some d: 1 (cifar100_dinov2). Accepted (at most one): yes.

## banking77_mpnet (knn_accuracy; raw 0.931)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_decoupled | decoupled-d64 | 64 | 0.923 ± 0.001 | -0.003 [-0.007, +0.001] tie | pca-64: 0.919 | +0.004 [-0.003, +0.012] tie |
| e6_dropout | dropout-d128 | 128 | 0.879 ± 0.002 | -0.047 [-0.056, -0.038] loss | pca-128: 0.929 | -0.050 [-0.059, -0.040] loss |
| e6_linear | linear-d64 | 64 | 0.926 ± 0.001 | reference | pca-64: 0.919 | +0.007 [+0.000, +0.014] win |
| e6_matryoshka | matryoshka-d64 | 64 | 0.926 ± 0.001 | -0.000 [-0.003, +0.002] tie | pca-64: 0.919 | +0.007 [+0.000, +0.014] win |
| e6_mlp | mlp-d64 | 64 | 0.921 ± 0.002 | -0.006 [-0.010, -0.001] loss | pca-64: 0.919 | +0.002 [-0.004, +0.008] tie |
| e6_mutual | mutual-d128 | 128 | 0.924 ± 0.001 | -0.003 [-0.008, +0.002] tie | pca-128: 0.929 | -0.005 [-0.012, +0.002] tie |
| e6_nbrmask | nbrmask-d128 | 128 | 0.929 ± 0.002 | +0.003 [+0.000, +0.006] win | pca-128: 0.929 | +0.000 [-0.006, +0.007] tie |
| e6_residual | residual-d64 | 64 | 0.927 ± 0.002 | +0.001 [-0.002, +0.004] tie | pca-64: 0.919 | +0.008 [+0.001, +0.014] win |

## cifar100_dinov2 (knn_accuracy; raw 0.879)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_decoupled | decoupled-d128 | 128 | 0.876 ± 0.001 | +0.001 [-0.002, +0.003] tie | pca_whiten-128: 0.878 | -0.003 [-0.007, +0.002] tie |
| e6_dropout | dropout-d128 | 128 | 0.807 ± 0.002 | -0.068 [-0.073, -0.063] loss | pca_whiten-128: 0.878 | -0.072 [-0.077, -0.066] loss |
| e6_linear | linear-d128 | 128 | 0.875 ± 0.000 | reference | pca_whiten-128: 0.878 | -0.003 [-0.008, +0.001] tie |
| e6_matryoshka | matryoshka-d128 | 128 | 0.875 ± 0.001 | -0.000 [-0.002, +0.002] tie | pca_whiten-128: 0.878 | -0.003 [-0.008, +0.001] tie |
| e6_mlp | mlp-d128 | 128 | 0.863 ± 0.002 | -0.012 [-0.015, -0.008] loss | pca_whiten-128: 0.878 | -0.015 [-0.019, -0.011] loss |
| e6_mutual | mutual-d128 | 128 | 0.876 ± 0.001 | +0.001 [-0.002, +0.004] tie | pca_whiten-128: 0.878 | -0.002 [-0.007, +0.002] tie |
| e6_nbrmask | nbrmask-d128 | 128 | 0.874 ± 0.001 | -0.000 [-0.001, +0.000] tie | pca_whiten-128: 0.878 | -0.004 [-0.008, +0.001] tie |
| e6_residual | residual-d128 | 128 | 0.875 ± 0.002 | +0.000 [-0.002, +0.003] tie | pca_whiten-128: 0.878 | -0.003 [-0.007, +0.001] tie |

## digits (knn_accuracy; raw 0.981)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_decoupled | decoupled-d32 | 32 | 0.977 ± 0.001 | +0.001 [+0.000, +0.002] tie | pca-32: 0.976 | +0.001 [-0.003, +0.006] tie |
| e6_dropout | dropout-d32 | 32 | 0.975 ± 0.001 | -0.001 [-0.007, +0.005] tie | pca-32: 0.976 | -0.000 [-0.006, +0.005] tie |
| e6_linear | linear-d32 | 32 | 0.976 ± 0.001 | reference | pca-32: 0.976 | +0.000 [-0.004, +0.005] tie |
| e6_matryoshka | matryoshka-d32 | 32 | 0.976 ± 0.000 | -0.000 [-0.001, +0.000] tie | pca-32: 0.976 | -0.000 [-0.004, +0.004] tie |
| e6_mlp | mlp-d32 | 32 | 0.976 ± 0.003 | -0.000 [-0.008, +0.008] tie | pca-32: 0.976 | -0.000 [-0.007, +0.008] tie |
| e6_mutual | mutual-d32 | 32 | 0.977 ± 0.000 | +0.001 [-0.001, +0.003] tie | pca-32: 0.976 | +0.001 [-0.002, +0.006] tie |
| e6_nbrmask | nbrmask-d32 | 32 | 0.978 ± 0.000 | +0.002 [+0.000, +0.005] tie | pca-32: 0.976 | +0.002 [-0.002, +0.007] tie |
| e6_residual | residual-d32 | 32 | 0.984 ± 0.001 | +0.008 [+0.003, +0.014] win | pca-32: 0.976 | +0.008 [+0.002, +0.016] win |

## pets_resnet50 (knn_accuracy; raw 0.901)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_decoupled | decoupled-d64 | 64 | 0.897 ± 0.002 | -0.008 [-0.011, -0.004] loss | pca-64: 0.901 | -0.003 [-0.010, +0.003] tie |
| e6_dropout | dropout-d128 | 128 | 0.793 ± 0.003 | -0.112 [-0.122, -0.101] loss | pca-128: 0.901 | -0.108 [-0.118, -0.097] loss |
| e6_linear | linear-d64 | 64 | 0.905 ± 0.002 | reference | pca-64: 0.901 | +0.004 [-0.002, +0.010] tie |
| e6_matryoshka | matryoshka-d64 | 64 | 0.904 ± 0.001 | -0.001 [-0.004, +0.001] tie | pca-64: 0.901 | +0.003 [-0.004, +0.009] tie |
| e6_mlp | mlp-d128 | 128 | 0.896 ± 0.003 | -0.009 [-0.013, -0.003] loss | pca-128: 0.901 | -0.005 [-0.011, +0.001] tie |
| e6_mutual | mutual-d128 | 128 | 0.880 ± 0.001 | -0.025 [-0.032, -0.017] loss | pca-128: 0.901 | -0.021 [-0.029, -0.013] loss |
| e6_nbrmask | nbrmask-d64 | 64 | 0.901 ± 0.001 | -0.004 [-0.006, -0.001] loss | pca-64: 0.901 | +0.000 [-0.006, +0.007] tie |
| e6_residual | residual-d128 | 128 | 0.898 ± 0.002 | -0.007 [-0.012, -0.002] loss | pca-128: 0.901 | -0.003 [-0.010, +0.003] tie |

## stl10_clip (knn_accuracy; raw 0.983)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_decoupled | decoupled-d128 | 128 | 0.982 ± 0.001 | -0.002 [-0.003, -0.001] loss | pca-128: 0.984 | -0.003 [-0.005, -0.000] loss |
| e6_dropout | dropout-d128 | 128 | 0.954 ± 0.001 | -0.030 [-0.034, -0.026] loss | pca-128: 0.984 | -0.031 [-0.035, -0.026] loss |
| e6_linear | linear-d128 | 128 | 0.984 ± 0.000 | reference | pca-128: 0.984 | -0.001 [-0.003, +0.002] tie |
| e6_matryoshka | matryoshka-d128 | 128 | 0.984 ± 0.000 | -0.000 [-0.001, +0.001] tie | pca-128: 0.984 | -0.001 [-0.003, +0.002] tie |
| e6_mlp | mlp-d128 | 128 | 0.979 ± 0.001 | -0.005 [-0.007, -0.003] loss | pca-128: 0.984 | -0.006 [-0.008, -0.003] loss |
| e6_mutual | mutual-d128 | 128 | 0.977 ± 0.000 | -0.006 [-0.008, -0.004] loss | pca-128: 0.984 | -0.007 [-0.010, -0.004] loss |
| e6_nbrmask | nbrmask-d64 | 64 | 0.982 ± 0.000 | -0.002 [-0.003, -0.001] loss | pca-64: 0.984 | -0.002 [-0.005, -0.000] loss |
| e6_residual | residual-d128 | 128 | 0.982 ± 0.000 | -0.002 [-0.003, -0.000] loss | pca-128: 0.984 | -0.002 [-0.005, -0.000] loss |

## stsb_mpnet (sts_spearman; raw 0.834)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_decoupled | decoupled-d128 | 128 | 0.820 ± 0.000 | -0.008 n/a | pca-128: 0.833 | -0.012 n/a |
| e6_dropout | dropout-d128 | 128 | 0.793 ± 0.002 | -0.036 n/a | pca-128: 0.833 | -0.040 n/a |
| e6_linear | linear-d128 | 128 | 0.828 ± 0.000 | reference | pca-128: 0.833 | -0.004 n/a |
| e6_matryoshka | matryoshka-d128 | 128 | 0.827 ± 0.000 | -0.001 n/a | pca-128: 0.833 | -0.005 n/a |
| e6_mlp | mlp-d128 | 128 | 0.798 ± 0.004 | -0.030 n/a | pca-128: 0.833 | -0.034 n/a |
| e6_mutual | mutual-d128 | 128 | 0.818 ± 0.001 | -0.010 n/a | pca-128: 0.833 | -0.014 n/a |
| e6_nbrmask | nbrmask-d128 | 128 | 0.830 ± 0.000 | +0.001 n/a | pca-128: 0.833 | -0.003 n/a |
| e6_residual | residual-d128 | 128 | 0.822 ± 0.001 | -0.006 n/a | pca-128: 0.833 | -0.010 n/a |

## synthetic_anisotropic (knn_accuracy; raw 0.746)

| Family | selected | d | test | vs reference [95% CI] | best baseline at d | vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| e6_decoupled | decoupled-d128 | 128 | 0.841 ± 0.001 | +0.004 [+0.003, +0.006] win | pca_whiten-128: 0.910 | -0.069 [-0.082, -0.057] loss |
| e6_dropout | dropout-d128 | 128 | 0.856 ± 0.005 | +0.020 [+0.008, +0.033] win | pca_whiten-128: 0.910 | -0.053 [-0.063, -0.042] loss |
| e6_linear | linear-d128 | 128 | 0.836 ± 0.001 | reference | pca_whiten-128: 0.910 | -0.073 [-0.087, -0.061] loss |
| e6_matryoshka | matryoshka-d64 | 64 | 0.837 ± 0.001 | +0.001 [-0.002, +0.004] tie | pca_whiten-64: 0.893 | -0.055 [-0.069, -0.043] loss |
| e6_mlp | mlp-d128 | 128 | 0.761 ± 0.006 | -0.075 [-0.086, -0.064] loss | pca_whiten-128: 0.910 | -0.149 [-0.163, -0.135] loss |
| e6_mutual | mutual-d128 | 128 | 0.843 ± 0.002 | +0.006 [+0.001, +0.013] win | pca_whiten-128: 0.910 | -0.067 [-0.079, -0.054] loss |
| e6_nbrmask | nbrmask-d64 | 64 | 0.838 ± 0.002 | +0.002 [-0.003, +0.007] tie | pca_whiten-64: 0.893 | -0.055 [-0.067, -0.042] loss |
| e6_residual | residual-d32 | 32 | 0.821 ± 0.001 | -0.015 [-0.025, -0.006] loss | pca_whiten-32: 0.892 | -0.071 [-0.085, -0.058] loss |
