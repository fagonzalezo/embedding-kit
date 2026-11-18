# E9: time and memory of `fit` (WP 7.3)

Each cell is one `fit` in a fresh process on synthetic float32 data (10 classes, power-law spectrum, generated in chunks; `benchmarks/e9.py`). Entries are "wall time of `fit` / peak resident memory of the process" (GB = 1e9 bytes). The peak includes the interpreter and torch (a few hundred MB), the data array (n d x 4 bytes) and everything `fit` allocates. A cell that is not `ok` shows its status; the messages are listed below the tables.

## analyzer

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 36.2 s / 0.8 GB | 36.8 s / 0.9 GB | 1.3 min / 1.2 GB |
| 10,000 | 56.6 s / 1.0 GB | 1.0 min / 1.1 GB | 2.9 min / 2.0 GB |
| 100,000 | 1.1 min / 1.1 GB | 1.5 min / 2.2 GB | 3.1 min / 7.1 GB |
| 1,000,000 | 55.8 s / 2.6 GB | 1.6 min / 17.0 GB | 4.1 min / 66.1 GB |

## analyzer_fast

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 4.6 s / 0.8 GB | 6.3 s / 0.9 GB | 6.7 s / 1.1 GB |
| 10,000 | 6.7 s / 1.0 GB | 7.1 s / 1.1 GB | 27.0 s / 1.9 GB |
| 100,000 | 5.9 s / 1.1 GB | 10.5 s / 2.2 GB | 41.2 s / 7.1 GB |
| 1,000,000 | 7.9 s / 2.6 GB | 17.4 s / 17.0 GB | 1.2 min / 66.1 GB |

## embedkit

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 0.0 s / 0.6 GB | 0.0 s / 0.6 GB | 0.0 s / 0.7 GB |
| 10,000 | 0.0 s / 0.6 GB | 0.1 s / 0.7 GB | 0.3 s / 1.1 GB |
| 100,000 | 0.1 s / 0.8 GB | 0.7 s / 1.9 GB | 2.6 s / 5.5 GB |
| 1,000,000 | 1.1 s / 2.2 GB | 6.7 s / 12.9 GB | 25.4 s / 49.8 GB |

## embedkit_labels

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 10.3 s / 1.0 GB | 11.4 s / 1.0 GB | 15.0 s / 1.2 GB |
| 10,000 | 27.8 s / 1.2 GB | 36.6 s / 1.4 GB | 2.2 min / 3.4 GB |
| 100,000 | 4.0 min / 3.0 GB | 5.8 min / 5.8 GB | 12.8 min / 20.1 GB |
| 1,000,000 | 1.80 h / 9.7 GB | 3.46 h / 46.0 GB | 9.70 h / 182.0 GB |

## Size of X (float32)

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000 | 0.5 MB | 4.1 MB | 16.4 MB |
| 10,000 | 5.1 MB | 41.0 MB | 163.8 MB |
| 100,000 | 51.2 MB | 409.6 MB | 1.64 GB |
| 1,000,000 | 512.0 MB | 4.10 GB | 16.38 GB |

## Machine

- host: r9nc-48-1, r9nc-48-2, r9nc-48-3, r9nc-48-4
- platform: Linux-5.14.0-611.54.1.el9_7.x86_64-x86_64-with-glibc2.39
- CPUs available to the process: 16
- OMP_NUM_THREADS: 16
- torch threads: 16
- Slurm partition: 48CPUNodes
- Slurm CPUs per task: 16
- Slurm memory per node (MB): 131072, 262144
- CUDA available: False
- MPS available: False
- embedkit: 0.5.0
- commit: 63da93f1, 6bcd7e0d
- torch: 2.11.0a0+eb65b36914.nv26.02

