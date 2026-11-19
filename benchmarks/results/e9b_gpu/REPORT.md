# E9b: time and memory of `fit` after WP 8.2, GPU

Each cell is one `fit` in a fresh process on synthetic float32 data (10 classes, power-law spectrum, generated in chunks; `benchmarks/e9.py`). Entries are "wall time of `fit` / peak resident memory of the process" (GB = 1e9 bytes). The peak includes the interpreter and torch (a few hundred MB), the data array (n d x 4 bytes) and everything `fit` allocates. A cell that is not `ok` shows its status; the messages are listed below the tables.

## embedkit_labels

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000,000 | 1.4 min / 4.0 GB | 2.1 min / 8.2 GB | 5.0 min / 26.6 GB |

## embedkit_labels_all

| n \ d | 128 | 1,024 |
|---|---|---|
| 1,000,000 | 2.76 h / 10.5 GB | 4.18 h / 46.1 GB |

## Size of X (float32)

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 1,000,000 | 512.0 MB | 4.10 GB | 16.38 GB |

## Machine

- host: r9ng-8000-1, r9ng-8000-2
- platform: Linux-5.14.0-611.54.1.el9_7.x86_64-x86_64-with-glibc2.39
- CPUs available to the process: 20
- OMP_NUM_THREADS: 20
- torch threads: 20
- Slurm partition: RTX8000Nodes
- Slurm CPUs per task: 20
- Slurm memory per node (MB): 204800
- CUDA available: True
- MPS available: False
- embedkit: 0.6.0
- commit: 99c82ecc
- torch: 2.11.0a0+eb65b36914.nv26.02

