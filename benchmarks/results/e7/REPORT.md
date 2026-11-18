# Experiment e7

All methods work at one d per dataset (the d that the default refiner selected in E6), so no
selection on validation is involved. Learning curves: the transform is fitted on ``n`` training
rows (``-n<n>``) or on all of them (``-all``); the kNN gallery and the probe always use the whole
transformed training split. The refiner is compared with the better of PCA and PCA-whitening at
the same ``n`` (chosen on validation), with a paired bootstrap over test points of the primary
metric. Transduction: each transform fitted on train + test rows without labels
(``-transductive``) against the same transform fitted on train rows only (``-all``).

## Summary

| Dataset | d | refiner vs best PCA, smallest n | refiner vs best PCA, all rows | transductive - inductive (refiner) |
|---|---|---|---|---|
| banking77_mpnet | 64 | n=200: -0.052 (loss) | +0.007 (win) | +0.006 (win) |
| cifar100_dinov2 | 128 | n=200: -0.006 (loss) | -0.003 (tie) | +0.001 (tie) |
| digits | 32 | n=200: -0.000 (tie) | +0.000 (tie) | +0.000 (tie) |
| pets_resnet50 | 64 | n=200: +0.006 (win) | +0.004 (tie) | +0.003 (tie) |
| stl10_clip | 128 | n=200: -0.002 (tie) | -0.001 (tie) | -0.001 (tie) |
| stsb_mpnet | 128 | n=200: -0.145 (n/a) | -0.004 (n/a) | +0.003 (n/a) |
| synthetic_anisotropic | 128 | n=200: -0.251 (loss) | -0.073 (loss) | +0.020 (win) |

## banking77_mpnet (knn_accuracy, d = 64; raw 0.931)

| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |
|---|---|---|---|---|
| 200 | 0.868 ± 0.003 | 0.920 ± 0.001 | 0.919 ± 0.004 | -0.052 [-0.058, -0.045] vs pca (loss) |
| 500 | 0.902 ± 0.003 | 0.919 ± 0.001 | 0.916 ± 0.002 | -0.017 [-0.022, -0.012] vs pca (loss) |
| 1000 | 0.914 ± 0.003 | 0.921 ± 0.002 | 0.917 ± 0.001 | -0.007 [-0.012, -0.002] vs pca (loss) |
| 5000 | 0.925 ± 0.001 | 0.920 ± 0.001 | 0.918 ± 0.001 | +0.006 [-0.000, +0.012] vs pca (tie) |
| all (9993) | 0.926 ± 0.001 | 0.919 | 0.917 | +0.007 [+0.000, +0.014] vs pca (win) |

| Transform | inductive | transductive | difference [95% CI] |
|---|---|---|---|
| embedkit | 0.926 ± 0.001 | 0.932 ± 0.002 | +0.006 [+0.001, +0.011] (win) |
| pca | 0.919 | 0.921 | +0.002 [+0.000, +0.005] (tie) |
| pca_whiten | 0.917 | 0.918 | +0.001 [-0.001, +0.004] (tie) |

## cifar100_dinov2 (knn_accuracy, d = 128; raw 0.879)

| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |
|---|---|---|---|---|
| 200 | 0.859 ± 0.001 | 0.864 ± 0.001 | 0.866 ± 0.001 | -0.006 [-0.009, -0.004] vs pca_whiten (loss) |
| 500 | 0.867 ± 0.001 | 0.870 ± 0.001 | 0.873 ± 0.001 | -0.006 [-0.008, -0.004] vs pca_whiten (loss) |
| 1000 | 0.870 ± 0.002 | 0.873 ± 0.001 | 0.875 ± 0.001 | -0.006 [-0.008, -0.003] vs pca_whiten (loss) |
| 5000 | 0.876 ± 0.002 | 0.877 ± 0.002 | 0.880 ± 0.001 | -0.005 [-0.007, -0.002] vs pca_whiten (loss) |
| 20000 | 0.875 ± 0.001 | 0.876 ± 0.001 | 0.879 ± 0.001 | -0.003 [-0.006, -0.000] vs pca_whiten (loss) |
| all (50000) | 0.875 ± 0.000 | 0.875 | 0.878 | -0.003 [-0.008, +0.001] vs pca_whiten (tie) |

| Transform | inductive | transductive | difference [95% CI] |
|---|---|---|---|
| embedkit | 0.875 ± 0.000 | 0.876 ± 0.001 | +0.001 [-0.002, +0.004] (tie) |
| pca | 0.875 | 0.877 | +0.002 [+0.000, +0.003] (win) |
| pca_whiten | 0.878 | 0.879 | +0.001 [-0.000, +0.003] (tie) |

## digits (knn_accuracy, d = 32; raw 0.981)

| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |
|---|---|---|---|---|
| 200 | 0.974 ± 0.001 | 0.975 ± 0.000 | 0.966 ± 0.007 | -0.000 [-0.002, +0.001] vs pca (tie) |
| 500 | 0.973 ± 0.003 | 0.973 ± 0.001 | 0.967 ± 0.002 | -0.000 [-0.002, +0.001] vs pca (tie) |
| all (898) | 0.976 ± 0.001 | 0.976 | 0.966 | +0.000 [-0.004, +0.005] vs pca (tie) |

| Transform | inductive | transductive | difference [95% CI] |
|---|---|---|---|
| embedkit | 0.976 ± 0.001 | 0.976 ± 0.000 | +0.000 [-0.003, +0.004] (tie) |
| pca | 0.976 | 0.973 | -0.002 [-0.006, +0.000] (tie) |
| pca_whiten | 0.966 | 0.970 | +0.004 [-0.003, +0.012] (tie) |

## pets_resnet50 (knn_accuracy, d = 64; raw 0.901)

| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |
|---|---|---|---|---|
| 200 | 0.892 ± 0.002 | 0.899 ± 0.003 | 0.886 ± 0.003 | +0.006 [+0.001, +0.011] vs pca_whiten (win) |
| 500 | 0.904 ± 0.003 | 0.901 ± 0.003 | 0.884 ± 0.003 | +0.002 [-0.001, +0.006] vs pca (tie) |
| 1000 | 0.904 ± 0.002 | 0.903 ± 0.001 | 0.885 ± 0.002 | +0.001 [-0.003, +0.005] vs pca (tie) |
| all (3680) | 0.905 ± 0.002 | 0.901 | 0.881 | +0.004 [-0.002, +0.010] vs pca (tie) |

| Transform | inductive | transductive | difference [95% CI] |
|---|---|---|---|
| embedkit | 0.905 ± 0.002 | 0.908 ± 0.002 | +0.003 [-0.003, +0.009] (tie) |
| pca | 0.901 | 0.903 | +0.002 [-0.001, +0.006] (tie) |
| pca_whiten | 0.881 | 0.878 | -0.003 [-0.008, +0.002] (tie) |

## stl10_clip (knn_accuracy, d = 128; raw 0.983)

| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |
|---|---|---|---|---|
| 200 | 0.982 ± 0.000 | 0.984 ± 0.000 | 0.969 ± 0.002 | -0.002 [-0.003, +0.000] vs pca (tie) |
| 500 | 0.985 ± 0.001 | 0.984 ± 0.001 | 0.969 ± 0.002 | +0.001 [-0.000, +0.003] vs pca (tie) |
| 1000 | 0.985 ± 0.001 | 0.985 ± 0.000 | 0.968 ± 0.001 | +0.000 [-0.001, +0.002] vs pca (tie) |
| all (5000) | 0.984 ± 0.000 | 0.984 | 0.967 | -0.001 [-0.003, +0.002] vs pca (tie) |

| Transform | inductive | transductive | difference [95% CI] |
|---|---|---|---|
| embedkit | 0.984 ± 0.000 | 0.983 ± 0.000 | -0.001 [-0.003, +0.001] (tie) |
| pca | 0.984 | 0.984 | -0.000 [-0.001, +0.000] (tie) |
| pca_whiten | 0.967 | 0.967 | +0.000 [-0.002, +0.003] (tie) |

## stsb_mpnet (sts_spearman, d = 128; raw 0.834)

| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |
|---|---|---|---|---|
| 200 | 0.681 ± 0.018 | 0.826 ± 0.003 | 0.833 ± 0.003 | -0.145 vs pca (n/a) |
| 500 | 0.740 ± 0.008 | 0.829 ± 0.004 | 0.837 ± 0.005 | -0.090 vs pca (n/a) |
| 1000 | 0.771 ± 0.009 | 0.830 ± 0.002 | 0.836 ± 0.003 | -0.059 vs pca (n/a) |
| 5000 | 0.820 ± 0.003 | 0.832 ± 0.001 | 0.837 ± 0.001 | -0.013 vs pca (n/a) |
| all (10566) | 0.828 ± 0.000 | 0.833 | 0.837 | -0.004 vs pca (n/a) |

| Transform | inductive | transductive | difference [95% CI] |
|---|---|---|---|
| embedkit | 0.828 ± 0.000 | 0.832 ± 0.001 | +0.003 (n/a) |
| pca | 0.833 | 0.841 | +0.008 (n/a) |
| pca_whiten | 0.837 | 0.837 | +0.000 (n/a) |

## synthetic_anisotropic (knn_accuracy, d = 128; raw 0.746)

| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |
|---|---|---|---|---|
| 200 | 0.638 ± 0.011 | 0.743 ± 0.002 | 0.889 ± 0.003 | -0.251 [-0.265, -0.237] vs pca_whiten (loss) |
| 500 | 0.733 ± 0.005 | 0.742 ± 0.001 | 0.898 ± 0.002 | -0.165 [-0.178, -0.151] vs pca_whiten (loss) |
| 1000 | 0.781 ± 0.005 | 0.741 ± 0.003 | 0.903 ± 0.002 | -0.122 [-0.135, -0.109] vs pca_whiten (loss) |
| all (3000) | 0.836 ± 0.001 | 0.741 | 0.910 | -0.073 [-0.087, -0.061] vs pca_whiten (loss) |

| Transform | inductive | transductive | difference [95% CI] |
|---|---|---|---|
| embedkit | 0.836 ± 0.001 | 0.857 ± 0.001 | +0.020 [+0.013, +0.027] (win) |
| pca | 0.741 | 0.739 | -0.002 [-0.005, +0.000] (tie) |
| pca_whiten | 0.910 | 0.905 | -0.005 [-0.010, +0.001] (tie) |
