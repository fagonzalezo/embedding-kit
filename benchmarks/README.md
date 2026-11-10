# Benchmarks

The benchmark harness of `docs/IMPLEMENTATION_PLAN.md` (WP 2.5). It is not part of the installed
package. It scores every method with `embedkit.evaluation.evaluate_transform`: fit on the training
split, score held-out data.

| File | Purpose |
|---|---|
| `bench_datasets.py` | Loaders returning `(X_train, y_train, X_test, y_test, meta)`. Never download. |
| `experiments.py` | Method grids per experiment (`e1`: training-free baselines and `EmbedKit`). |
| `run.py` | Runs an experiment; one JSON file per dataset and method; resumable. |
| `report.py` | Turns the JSON files into Markdown tables, with selection on validation. |
| `extract_features.py` | Downloads a large set and caches its embeddings. **Cluster only.** |
| `cluster/` | Slurm jobs for the OCCIDATA cluster (see "On the cluster" below). |
| `review_2026_11/` | Scripts behind `docs/CRITICAL_REVIEW.md` (pilot, defect reproductions). |

## Running

```bash
# Local datasets: pets_resnet50 (examples/data/), digits, synthetic_anisotropic
python benchmarks/run.py --experiment e1 --datasets pets_resnet50 digits synthetic_anisotropic \
    --seeds 3 --device cpu
python benchmarks/report.py --experiment e1 --out e1.md

python benchmarks/run.py --experiment e1 --datasets digits --list          # show the grid
python benchmarks/run.py --experiment e1 --datasets digits --methods raw pca-32
```

- `--data-root` (or `EMBEDKIT_BENCH_DATA`) is where datasets and cached features live; the default
  is `benchmarks/data/` (git-ignored). `--device` is passed to `EmbedKit`.
- Results go to `benchmarks/results/<experiment>/<dataset>/<method>.json`. A combination whose
  file exists is skipped (`--overwrite` to redo it). Each file records the git commit, whether the
  tree was dirty, versions, host, the configuration, and every (seed, metric) row of two stages:
  - `val`: fitted on 80% of the training split, scored on the other 20% (stratified, seed 0).
    Hyperparameters are selected on these numbers, never on test.
  - `test`: fitted on the whole training split, scored on the test split.
- Deterministic methods run once; stochastic ones (`EmbedKit`, random projection) for `--seeds`
  seeds. Per-test-point outcomes of kNN accuracy, the probe and recall@10 are stored compactly
  so that `compare()` can give paired bootstrap intervals.

## Datasets

| Name | Where it comes from | Task |
|---|---|---|
| `pets_resnet50` | `examples/04_image_classification.py` (local) | 37 classes |
| `digits` | scikit-learn (raw pixels, not a learned embedding) | 10 classes |
| `synthetic_anisotropic` | generated (the review pilot's set) | 10 classes |
| `cifar100_dinov2` | CIFAR-100, DINOv2 ViT-B/14 (cluster) | 100 classes (+20 coarse) |
| `stl10_clip` | STL-10, CLIP ViT-B/32 (cluster) | 10 classes |
| `banking77_mpnet` | Banking77, all-mpnet-base-v2 (cluster) | 77 classes |
| `stsb_mpnet` | STS-B, all-mpnet-base-v2 (cluster) | sentence similarity |
| `glove100`, `sift128`, `gist960` | ANN-Benchmarks (cluster) | unlabelled |

## On the cluster

The jobs follow the OCCIDATA rules: `#SBATCH --account=iris`; no compute on the login node;
environments, caches, logs and `TMPDIR` under `/projects/iris/<user>/`, never `/home` or `/tmp`;
GPU jobs use `singularity exec --nv`; at most 2 GPUs at a time. Code reaches the cluster with
`git push` and `git pull --ff-only` in the clone; every job refuses to run on a dirty tree.

From `benchmarks/cluster/` in the cluster clone: `sbatch setup.slurm.sh` (once),
`sbatch extract.slurm.sh` and `sbatch extract_cpu.slurm.sh` (features), `sbatch test.slurm.sh`
(test suite on CUDA), then `sbatch run.slurm.sh` (E1, one array task per dataset). Copy
`benchmarks/results/` back with rsync and commit the JSON files.
