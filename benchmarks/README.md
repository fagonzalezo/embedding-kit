# Benchmarks

The benchmark harness of `docs/dev/IMPLEMENTATION_PLAN.md` (WP 2.5). It is not part of the installed
package. It scores every method with `embedkit.evaluation.evaluate_transform`: fit on the training
split, score held-out data.

| File | Purpose |
|---|---|
| `bench_datasets.py` | Loaders returning `(X_train, y_train, X_test, y_test, meta)`. Never download. |
| `experiments.py` | Method grids per experiment (`e1`: training-free baselines and `EmbedKit`). |
| `run.py` | Runs an experiment; one JSON file per dataset and method; resumable. |
| `report.py` | Turns the JSON files into Markdown tables, with selection on validation. |
| `e2.py` | Experiment E2: trains ~30 refiner configurations per dataset and logs every monitor next to test kNN and probe accuracy every 5 epochs; resumable. |
| `report_e2.py` | E2 tables: within-run and across-run Spearman correlations of each monitor with test accuracy, and the regret of stopping or selecting by it. |
| `report_gate.py` | The phase 3 gate on E1: self-supervised `EmbedKit` against the best training-free baseline at the same d, with paired intervals. |
| `report_e6.py` | E6 tables: each refiner arm, selected on validation, against the previous default and against PCA at the same d; applies the phase 5 acceptance rule. |
| `report_e7.py` | E7 tables: learning curves of the default refiner against PCA and PCA-whitening over the number of rows fitted, and inductive against transductive fitting. |
| `e4.py`, `e4_data.py`, `report_e4.py` | Experiment E4 (WP 7.1): the report on each natural set (`e4.py`; the interventions run through `run.py --experiment e4`), the tables of diagnostics and gains (`e4_data.py`), and the analysis fixed in the plan before the runs (`report_e4.py`). |
| `report_datasets.py` | `docs/benchmarks/RESULTS_BY_DATASET.md`: E10 per data set, for users (`--collect` first reads the per-run files into `results/e10/summary_datasets.csv`). |
| `extract_features.py` | Downloads a large set and caches its embeddings. **Cluster only.** |
| `cluster/` | Slurm jobs for the OCCIDATA cluster (see "On the cluster" below). |
| `review_2026_11/` | Scripts behind `docs/dev/CRITICAL_REVIEW.md` (pilot, defect reproductions). |

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
| E4 images (12 new: `{cifar10, cifar100, stl10, pets, flowers102}_{resnet50, dinov2, clip}`, minus the three above) | CIFAR-10/100, STL-10, Oxford Pets, Flowers102; ResNet-50 IMAGENET1K_V2 (2,048-d), DINOv2, CLIP (cluster) | 10, 100, 10, 37, 102 classes |
| E4 text (19 new: `{banking77, emotion, massive, agnews, newsgroups}_{mpnet, minilm, bge, e5}`, minus `banking77_mpnet`) | Banking77, emotion, MASSIVE intent (en), AG News (20k / 5k subset), 20 Newsgroups; all-mpnet-base-v2, all-MiniLM-L6-v2, bge-base-en-v1.5, e5-base-v2 (`"query: "`), `Normalize` removed, `max_seq_length` 256 (cluster); JSON records `n_test_in_train` | 77, 6, 60, 4, 20 classes |
| `stsb_mpnet` | STS-B, all-mpnet-base-v2 (cluster) | sentence similarity |
| `glove100`, `sift128`, `gist960` | ANN-Benchmarks (cluster) | unlabelled |

`bench_datasets.E4_SETS` lists the 35 natural sets of E4 (these 31 plus `pets_resnet50`,
`cifar100_dinov2`, `stl10_clip`, `banking77_mpnet`); extract the new ones with
`SETS=E4_GPU_SETS sbatch --array=0-30%2 extract.slurm.sh`.

## On the cluster

The jobs follow the OCCIDATA rules: `#SBATCH --account=iris`; no compute on the login node;
environments, caches, logs and `TMPDIR` under `/projects/iris/<user>/`, never `/home` or `/tmp`;
GPU jobs use `singularity exec --nv`; at most 2 GPUs at a time. Code reaches the cluster with
`git push` and `git pull --ff-only` in the clone; every job refuses to run on a dirty tree.

From `benchmarks/cluster/` in the cluster clone: `sbatch setup.slurm.sh` (once),
`sbatch extract.slurm.sh` and `sbatch extract_cpu.slurm.sh` (features), `sbatch test.slurm.sh`
(test suite on CUDA), then `sbatch run.slurm.sh` (E1, one array task per dataset), `sbatch e1_hub.slurm.sh` (the E1
hubness-reduction methods, on CPU nodes), `sbatch e6.slurm.sh` (E6) and `sbatch e7.slurm.sh` (E7). Copy
`benchmarks/results/` back with rsync and commit the JSON files.
