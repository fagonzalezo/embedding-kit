# E9b: time and memory of `fit` after WP 8.2 (selection on a subsample), CPU

Each cell is one `fit` in a fresh process on synthetic float32 data (10 classes, power-law spectrum, generated in chunks; `benchmarks/e9.py`). Entries are "wall time of `fit` / peak resident memory of the process" (GB = 1e9 bytes). The peak includes the interpreter and torch (a few hundred MB), the data array (n d x 4 bytes) and everything `fit` allocates. A cell that is not `ok` shows its status; the messages are listed below the tables.

## embedkit

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 50,000 | 0.1 s / 0.7 GB | 0.3 s / 1.0 GB | 1.1 s / 2.3 GB |
| 100,000 | 0.1 s / 0.7 GB | 0.6 s / 1.2 GB | 2.2 s / 3.1 GB |
| 1,000,000 | 1.1 s / 1.2 GB | 5.9 s / 4.9 GB | 22.2 s / 17.8 GB |

## embedkit_labels

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 50,000 | 1.9 min / 2.8 GB | 2.6 min / 3.6 GB | 6.4 min / 11.0 GB |
| 100,000 | 1.9 min / 2.8 GB | 2.6 min / 3.8 GB | 6.4 min / 11.8 GB |
| 1,000,000 | 1.9 min / 3.3 GB | 2.7 min / 7.5 GB | 6.7 min / 26.5 GB |

## Size of X (float32)

| n \ d | 128 | 1,024 | 4,096 |
|---|---|---|---|
| 50,000 | 25.6 MB | 204.8 MB | 819.2 MB |
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
- Slurm memory per node (MB): 262144
- CUDA available: False
- MPS available: False
- embedkit: 0.6.0
- commit: 99c82ecc
- torch: 2.11.0a0+eb65b36914.nv26.02

