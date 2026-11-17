#!/bin/bash
#SBATCH --job-name=embedkit-e4
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e4-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e4-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=24CPUNodes
#SBATCH --array=0-34%16
#
# Experiment E4 (WP 7.1): do the report's readings predict the gain of a training-free
# intervention? One data set per array task over E4_SETS (env.sh): 35 sets, tasks 0-34.
#   sbatch --array=0-34%16 e4.slurm.sh                  PART=gains (default): run.py e4, both
#                                                       stages (val and test), the whole grid
#   PART=detect sbatch --array=0-34%16 --time=12:00:00 e4.slurm.sh
#                                                       the report with cosine and Euclidean
#                                                       metric (e4.py), CIFAR-scale sets slower
# 8 CPUs, 32G and 8 h as for E5; the CIFAR sets (50k x 2,048) are the heaviest. Both parts skip
# finished files, so a TIMEOUT costs only unfinished work: resubmit the task IDs, for example
# `sbatch --array=3,17 e4.slurm.sh`.
# Results: $REPO/benchmarks/results/e4/ (they stay on the cluster). Submit from this directory.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
PART=${PART:-gains}
SET=${E4_SETS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ "$PART" = detect ]; then
  srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/e4.py" --datasets "$SET" \
    --data-root "$DATA"
else
  srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" --experiment e4 \
    --datasets "$SET" --data-root "$DATA"
fi
echo "embedkit-e4 $PART $SET done"
exit 0
