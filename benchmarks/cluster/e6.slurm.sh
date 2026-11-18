#!/bin/bash
#SBATCH --job-name=embedkit-e6
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e6-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e6-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-8%2
#
# Experiment E6 (phase 5 refiner ablation), one task per entry of E6_TASKS in env.sh: a dataset,
# or dataset:I/N for shard I of N of its method grid (CIFAR-100 is split so each task fits 12 h).
# Resumable as run.py skips finished methods; resubmit the task IDs after a TIMEOUT.
# Results: $REPO/benchmarks/results/e6/<dataset>/<method>.json, brought back with rsync.
# Submit from this directory:  sbatch e6.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SEEDS=${SEEDS:-5}
ENTRY=${E6_TASKS[$SLURM_ARRAY_TASK_ID]}
SET=${ENTRY%%:*}
SHARD=()
if [[ $ENTRY == *:* ]]; then SHARD=(--shard "${ENTRY#*:}"); fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" \
  --experiment e6 --datasets "$SET" ${SHARD[@]+"${SHARD[@]}"} --seeds "$SEEDS" --device cuda --data-root "$DATA"
echo "embedkit-e6 $ENTRY done"
exit 0
