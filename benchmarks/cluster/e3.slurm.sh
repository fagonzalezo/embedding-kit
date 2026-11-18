#!/bin/bash
#SBATCH --job-name=embedkit-e3
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e3-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e3-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=24CPUNodes
#SBATCH --array=0-39%16
#
# Experiment E3 (WP 4.5): calibration of the diagnostics on data with known answers.
# PART selects the part of benchmarks/e3.py:
#   PART=id        sbatch --array=0-39%16 e3.slurm.sh                       (shard i of 40)
#   PART=gauss     sbatch --array=0-19%16 --time=04:00:00 e3.slurm.sh       (shard i of 20)
#   PART=subsample sbatch --array=0-5 --mem=40G --time=12:00:00 e3.slurm.sh (one set per task)
#   CELLS="name1 name2" PART=id sbatch --array=0 e3.slurm.sh                  (only these cells)
# e3.py skips cells whose file exists, so a TIMEOUT costs only the unfinished cell; resubmit.
# Results: $REPO/benchmarks/results/e3/<part>/<cell>.json (they stay on the cluster).
# Submit from this directory.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
PART=${PART:-id}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ -n "${CELLS:-}" ]; then
  read -r -a CELL_LIST <<< "$CELLS"
  ARGS=(--cells "${CELL_LIST[@]}")
  [ "$PART" = subsample ] && ARGS+=(--datasets "${CELL_LIST[0]%%__*}")
elif [ "$PART" = subsample ]; then
  E3_SUBSAMPLE_SETS=(gist960 cifar100_dinov2 glove100 sift128 yelp_mpnet fluorescence_esm2)
  ARGS=(--datasets "${E3_SUBSAMPLE_SETS[$SLURM_ARRAY_TASK_ID]}")
else
  # Fixed shard counts (not the array size), so resubmitting a few task IDs keeps the split.
  declare -A SHARDS=([id]=40 [gauss]=20)
  ARGS=(--shard "$SLURM_ARRAY_TASK_ID/${SHARDS[$PART]}")
fi
srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/e3.py" --part "$PART" "${ARGS[@]}" \
  --data-root "$DATA"
echo "embedkit-e3 $PART ${ARGS[*]} done"
exit 0
