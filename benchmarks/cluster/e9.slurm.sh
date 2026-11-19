#!/bin/bash
#SBATCH --job-name=embedkit-e9
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e9-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e9-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=48CPUNodes
#SBATCH --array=0-11%4
#
# Experiment E9 (WP 7.3): time and memory of fit for n in {1e3, 1e4, 1e5, 1e6} and
# d in {128, 1,024, 4,096}. One (n, d) cell per array task (E9_N and E9_D in env.sh; task i is
# n = E9_N[i / 3], d = E9_D[i % 3]), for one target (TARGET, default analyzer):
#   sbatch e9.slurm.sh                                  TARGET=analyzer: EmbedKitAnalyzer(), the
#                                                       defaults (calibrated, 20 draws)
#   TARGET=analyzer_fast sbatch e9.slurm.sh             the analyzer without calibration, ID
#                                                       bootstrap and ID scale curve (what
#                                                       EmbedKit.fit runs inside)
#   TARGET=embedkit sbatch e9.slurm.sh                  EmbedKit().fit(X), no labels (default
#                                                       mode, the PCA path)
#   TARGET=embedkit_labels sbatch e9.slurm.sh           EmbedKit().fit(X, y), 10 classes
# Targets are listed in benchmarks/e9.py (TARGETS). Report, once the tasks are done (CPU job,
# the results stay on the cluster):
#   sbatch --account=iris --partition=24CPUNodes --cpus-per-task=2 --mem=8G --time=00:10:00 \
#     --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e9-report-%j.out \
#     --wrap='srun singularity exec /apps/containerCollections/CUDA13/pytorch-NGC-26-02.sif \
#       /projects/iris/fabio.gonzalezosorio/envs/embedkit/bin/python \
#       $PWD/../report_e9.py --results $PWD/../results'     (submitted from this directory)
# Sized for the largest cell, 1e6 x 4096: X alone is 16.4 GB float32, and the code makes float64
# copies of it (the PCA path of EmbedKit centres a float64 copy and runs a full SVD of it), so
# 256G with 16 CPUs; the small cells use a fraction of it. Each cell runs in a child process
# that is killed at --timeout (a little under the job's 24 h), recorded as "timeout"; a child
# killed by the OOM killer is recorded as "oom_or_killed". Finished cells are skipped, so a
# TIMEOUT of the job costs only unfinished work: resubmit the task IDs, for example
# `sbatch --array=11 e9.slurm.sh`; add --force in E9_ARGS to repeat a recorded failure
# (E9_ARGS=--force).
# E9b (WP 8.2), the rerun after selection on a subsample, into results/e9b (CPU) and
# results/e9b_gpu (GPU); E9_NS names the n list in env.sh:
#   E9_NS=E9B_N TARGET=embedkit_labels E9_ARGS="--experiment e9b" sbatch --array=0-8%4 e9.slurm.sh
#   E9_NS=E9B_N TARGET=embedkit E9_ARGS="--experiment e9b" sbatch --array=0-8%4 e9.slurm.sh
#   E9_NS=E9B_GPU_N TARGET=embedkit_labels E9_ARGS="--experiment e9b_gpu" sbatch --array=0-2%2 \
#     --partition=RTX8000Nodes --gres=gpu:1 --cpus-per-task=20 --mem=200G e9.slurm.sh
#   (and TARGET=embedkit_labels_all for the 0.6.0 selection on the GPU; 20 CPUs for 200G there,
#   at 10,200 MB per CPU)
# With a GPU allocated the container runs with --nv.
# Results: $REPO/benchmarks/results/<experiment>/ (they stay on the cluster). Submit from this
# directory.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
TARGET=${TARGET:-analyzer}
declare -n NS=${E9_NS:-E9_N}
N=${NS[$(( SLURM_ARRAY_TASK_ID / ${#E9_D[@]} ))]}
D=${E9_D[$(( SLURM_ARRAY_TASK_ID % ${#E9_D[@]} ))]}
NV=()
[ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ] && NV=(--nv)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# E9 measures the library as a user would run it: all the CPUs for BLAS as well, unlike env.sh's
# half (which protects the many small fits of the linear probe).
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
# shellcheck disable=SC2086
srun singularity exec "${NV[@]}" "$CONTAINER" "$PY" "$REPO/benchmarks/e9.py" --targets "$TARGET" \
  --n "$N" --d "$D" --results "$REPO/benchmarks/results" --timeout 85000 ${E9_ARGS:-}
echo "embedkit-e9 $TARGET n=$N d=$D done"
exit 0
