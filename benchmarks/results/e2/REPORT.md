# Experiment e2

Reads ``<results>/e2/<dataset>/c*.json`` written by ``benchmarks/e2.py``. Every monitor is
sign-adjusted by its direction in the registry, so that a positive correlation always means "the
monitor's better values go with higher accuracy".

- **Within a run (stopping signal):** Spearman correlation between the monitor and the test
  metric over the evaluation epochs of one run; median and quartiles over the runs. Regret: the
  test metric's best value in the run minus its value at the epoch where the monitor is best
  (first best); for comparison, the same for the final epoch.
- **Across runs (selection signal):** Spearman correlation over configurations between the
  monitor and the test metric, both at the final epoch. Regret: the best configuration's test
  metric minus that of the configuration the monitor ranks first.

``val_knn_accuracy`` uses labels (the labelled reference); ``train_loss`` is the training loss of
the evaluation epoch.

## Overview: median within-run ρ (stopping) / across-run ρ (selection), test kNN

| Monitor | banking77_mpnet | cifar100_dinov2 | pets_resnet50 | synthetic_anisotropic |
|---|---|---|---|---|
| uniformity | +0.55 / -0.17 | +0.42 / -0.13 | -0.22 / -0.60 | +0.16 / +0.73 |
| isotropy | +0.47 / -0.04 | +0.51 / +0.10 | -0.22 / -0.15 | +0.26 / +0.62 |
| k_skewness | +0.16 / +0.28 | +0.50 / +0.68 | -0.03 / -0.34 | +0.07 / +0.00 |
| knn_overlap | +0.02 / +0.74 | -0.17 / +0.79 | +0.19 / +0.72 | +0.08 / -0.03 |
| effective_rank | -0.21 / +0.20 | +0.02 / +0.13 | +0.07 / -0.19 | -0.17 / +0.49 |
| val_loss | -0.64 / -0.43 | -0.38 / -0.33 | -0.59 / -0.58 | -0.37 / +0.11 |
| train_loss | +0.54 / +0.07 | +0.46 / +0.13 | -0.14 / -0.24 | +0.17 / +0.46 |
| val_knn_accuracy (labels) | +0.42 / +0.89 | +0.52 / +0.95 | +0.20 / +0.87 | +0.49 / +0.95 |

The plan's rule (WP 3.3): the default monitor becomes the label-free statistic with the best within-run correlation; if none has a median ρ above 0.5, early stopping is disabled by default.

## banking77_mpnet

30 runs (0 failed), 100 epochs, evaluated every 5; downstream on 9993 training / 3076 test rows.

| | min | median | max | raw |
|---|---|---|---|---|
| test kNN, final epoch | 0.834 | 0.897 | 0.921 | 0.931 |
| test kNN, best epoch | 0.847 | 0.899 | 0.924 | |
| test probe (C = 1), final epoch | 0.804 | 0.897 | 0.925 | 0.931 |

### Test kNN

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | +0.55 [+0.07, +0.73] | 16/30 | 0.005 (0.039) | -0.17 | 0.081 |
| isotropy | +0.47 [+0.03, +0.73] | 15/30 | 0.003 (0.016) | -0.04 | 0.004 |
| k_skewness | +0.16 [-0.18, +0.40] | 6/30 | 0.005 (0.021) | +0.28 | 0.003 |
| knn_overlap | +0.02 [-0.57, +0.69] | 9/30 | 0.005 (0.021) | +0.74 | 0.002 |
| effective_rank | -0.21 [-0.68, +0.47] | 8/30 | 0.006 (0.021) | +0.20 | 0.081 |
| val_loss | -0.64 [-0.79, +0.03] | 4/30 | 0.009 (0.039) | -0.43 | 0.035 |
| train_loss | +0.54 [+0.04, +0.71] | 16/30 | 0.005 (0.037) | +0.07 | 0.035 |
| val_knn_accuracy (labels) | +0.42 [+0.19, +0.62] | 14/30 | 0.002 (0.008) | +0.89 | 0.001 |
| final epoch (no stopping) | | | 0.005 (0.038) | | |

### Test probe

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | +0.22 [-0.56, +0.63] | 10/30 | 0.007 (0.060) | -0.14 | 0.093 |
| isotropy | +0.31 [-0.37, +0.63] | 9/30 | 0.005 (0.026) | +0.01 | 0.010 |
| k_skewness | +0.00 [-0.42, +0.35] | 5/30 | 0.008 (0.051) | +0.50 | 0.009 |
| knn_overlap | -0.04 [-0.32, +0.72] | 9/30 | 0.005 (0.019) | +0.62 | 0.009 |
| effective_rank | +0.28 [-0.34, +0.78] | 10/30 | 0.006 (0.026) | +0.11 | 0.093 |
| val_loss | -0.29 [-0.80, +0.39] | 6/30 | 0.012 (0.060) | -0.35 | 0.040 |
| train_loss | +0.22 [-0.71, +0.64] | 9/30 | 0.007 (0.060) | +0.02 | 0.040 |
| val_knn_accuracy (labels) | +0.27 [+0.11, +0.58] | 10/30 | 0.003 (0.009) | +0.92 | 0.008 |
| final epoch (no stopping) | | | 0.007 (0.060) | | |

## cifar100_dinov2

30 runs (0 failed), 100 epochs, evaluated every 5; downstream on 10000 training / 5000 test rows.

| | min | median | max | raw |
|---|---|---|---|---|
| test kNN, final epoch | 0.627 | 0.815 | 0.847 | 0.859 |
| test kNN, best epoch | 0.726 | 0.817 | 0.849 | |
| test probe (C = 1), final epoch | 0.619 | 0.837 | 0.862 | 0.878 |

### Test kNN

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | +0.42 [+0.06, +0.66] | 14/30 | 0.008 (0.120) | -0.13 | 0.220 |
| isotropy | +0.51 [+0.26, +0.73] | 15/30 | 0.005 (0.024) | +0.10 | 0.101 |
| k_skewness | +0.50 [+0.01, +0.67] | 15/30 | 0.009 (0.088) | +0.68 | 0.006 |
| knn_overlap | -0.17 [-0.30, +0.64] | 9/30 | 0.007 (0.021) | +0.79 | 0.001 |
| effective_rank | +0.02 [-0.30, +0.65] | 12/30 | 0.009 (0.049) | +0.13 | 0.220 |
| val_loss | -0.38 [-0.67, -0.04] | 5/30 | 0.018 (0.124) | -0.33 | 0.073 |
| train_loss | +0.46 [+0.27, +0.75] | 14/30 | 0.010 (0.121) | +0.13 | 0.073 |
| val_knn_accuracy (labels) | +0.52 [+0.38, +0.67] | 16/30 | 0.003 (0.008) | +0.95 | 0.002 |
| final epoch (no stopping) | | | 0.010 (0.123) | | |

### Test probe

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | +0.30 [-0.34, +0.79] | 12/30 | 0.009 (0.107) | -0.11 | 0.243 |
| isotropy | +0.37 [-0.08, +0.87] | 14/30 | 0.005 (0.030) | +0.23 | 0.091 |
| k_skewness | +0.49 [-0.07, +0.73] | 15/30 | 0.007 (0.070) | +0.76 | 0.003 |
| knn_overlap | +0.14 [-0.30, +0.68] | 12/30 | 0.005 (0.019) | +0.72 | 0.022 |
| effective_rank | +0.45 [-0.01, +0.83] | 15/30 | 0.005 (0.031) | +0.07 | 0.243 |
| val_loss | -0.39 [-0.77, +0.35] | 6/30 | 0.014 (0.110) | -0.35 | 0.064 |
| train_loss | +0.36 [-0.46, +0.85] | 13/30 | 0.010 (0.110) | +0.10 | 0.064 |
| val_knn_accuracy (labels) | +0.37 [+0.15, +0.68] | 12/30 | 0.003 (0.017) | +0.93 | 0.010 |
| final epoch (no stopping) | | | 0.010 (0.111) | | |

## pets_resnet50

30 runs (0 failed), 100 epochs, evaluated every 5; downstream on 3680 training / 3669 test rows.

| | min | median | max | raw |
|---|---|---|---|---|
| test kNN, final epoch | 0.771 | 0.893 | 0.908 | 0.901 |
| test kNN, best epoch | 0.805 | 0.898 | 0.909 | |
| test probe (C = 1), final epoch | 0.835 | 0.906 | 0.916 | 0.916 |

### Test kNN

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | -0.22 [-0.74, +0.49] | 8/30 | 0.012 (0.047) | -0.60 | 0.120 |
| isotropy | -0.22 [-0.74, +0.32] | 5/30 | 0.013 (0.047) | -0.15 | 0.004 |
| k_skewness | -0.03 [-0.27, +0.27] | 3/30 | 0.009 (0.044) | -0.34 | 0.027 |
| knn_overlap | +0.19 [-0.20, +0.76] | 11/30 | 0.005 (0.025) | +0.72 | 0.000 |
| effective_rank | +0.07 [-0.37, +0.66] | 9/30 | 0.009 (0.047) | -0.19 | 0.120 |
| val_loss | -0.59 [-0.82, +0.10] | 2/30 | 0.015 (0.047) | -0.58 | 0.078 |
| train_loss | -0.14 [-0.75, +0.49] | 8/30 | 0.012 (0.047) | -0.24 | 0.078 |
| val_knn_accuracy (labels) | +0.20 [+0.07, +0.49] | 8/30 | 0.003 (0.011) | +0.87 | 0.006 |
| final epoch (no stopping) | | | 0.012 (0.047) | | |

### Test probe

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | -0.17 [-0.87, +0.64] | 11/30 | 0.007 (0.027) | -0.64 | 0.078 |
| isotropy | -0.19 [-0.84, +0.60] | 9/30 | 0.008 (0.026) | -0.16 | 0.000 |
| k_skewness | -0.07 [-0.45, +0.19] | 5/30 | 0.006 (0.023) | -0.14 | 0.021 |
| knn_overlap | -0.03 [-0.51, +0.72] | 10/30 | 0.007 (0.026) | +0.65 | 0.003 |
| effective_rank | +0.17 [-0.56, +0.74] | 11/30 | 0.007 (0.021) | -0.29 | 0.078 |
| val_loss | -0.62 [-0.89, -0.01] | 3/30 | 0.009 (0.027) | -0.58 | 0.043 |
| train_loss | -0.11 [-0.86, +0.67] | 12/30 | 0.007 (0.026) | -0.37 | 0.043 |
| val_knn_accuracy (labels) | +0.18 [-0.08, +0.38] | 5/30 | 0.005 (0.019) | +0.74 | 0.005 |
| final epoch (no stopping) | | | 0.007 (0.026) | | |

## synthetic_anisotropic

30 runs (0 failed), 100 epochs, evaluated every 5; downstream on 3000 training / 3000 test rows.

| | min | median | max | raw |
|---|---|---|---|---|
| test kNN, final epoch | 0.099 | 0.683 | 0.803 | 0.746 |
| test kNN, best epoch | 0.108 | 0.692 | 0.822 | |
| test probe (C = 1), final epoch | 0.155 | 0.879 | 0.913 | 0.933 |

### Test kNN

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | +0.16 [-0.23, +0.70] | 10/30 | 0.031 (0.323) | +0.73 | 0.079 |
| isotropy | +0.26 [-0.33, +0.62] | 9/30 | 0.021 (0.081) | +0.62 | 0.087 |
| k_skewness | +0.07 [-0.25, +0.22] | 2/30 | 0.034 (0.300) | +0.00 | 0.122 |
| knn_overlap | +0.08 [-0.34, +0.71] | 10/30 | 0.023 (0.109) | -0.03 | 0.135 |
| effective_rank | -0.17 [-0.77, +0.50] | 7/30 | 0.025 (0.109) | +0.49 | 0.035 |
| val_loss | -0.37 [-0.82, +0.04] | 3/30 | 0.040 (0.324) | +0.11 | 0.011 |
| train_loss | +0.17 [-0.37, +0.59] | 9/30 | 0.032 (0.323) | +0.46 | 0.011 |
| val_knn_accuracy (labels) | +0.49 [+0.37, +0.71] | 15/30 | 0.006 (0.078) | +0.95 | 0.000 |
| final epoch (no stopping) | | | 0.033 (0.327) | | |

### Test probe

| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, mean (max) | across-run ρ | selection regret |
|---|---|---|---|---|---|
| uniformity | +0.49 [-0.43, +0.83] | 15/30 | 0.049 (0.418) | +0.56 | 0.006 |
| isotropy | +0.37 [-0.64, +0.78] | 12/30 | 0.044 (0.406) | +0.32 | 0.042 |
| k_skewness | +0.23 [-0.05, +0.44] | 6/30 | 0.014 (0.111) | +0.18 | 0.023 |
| knn_overlap | +0.18 [-0.65, +0.82] | 11/30 | 0.038 (0.294) | +0.23 | 0.007 |
| effective_rank | -0.23 [-0.83, +0.80] | 10/30 | 0.039 (0.355) | +0.68 | 0.014 |
| val_loss | -0.23 [-0.73, +0.79] | 11/30 | 0.049 (0.419) | +0.09 | 0.000 |
| train_loss | +0.55 [-0.42, +0.74] | 16/30 | 0.049 (0.419) | +0.43 | 0.000 |
| val_knn_accuracy (labels) | +0.29 [-0.00, +0.64] | 13/30 | 0.023 (0.419) | +0.68 | 0.023 |
| final epoch (no stopping) | | | 0.048 (0.419) | | |
