#!/bin/bash
#SBATCH --job-name=embedkit-e7
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e7-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e7-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-6%2
#
# Experiment E7 (learning curves, inductive against transductive fitting), one task per dataset
# (E7_SETS in env.sh). Resumable as run.py skips finished methods; resubmit the task IDs after a
# TIMEOUT. Results: $REPO/benchmarks/results/e7/<dataset>/<method>.json, brought back with rsync.
# Submit from this directory:  sbatch e7.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SEEDS=${SEEDS:-5}
SET=${E7_SETS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" \
  --experiment e7 --datasets "$SET" --seeds "$SEEDS" --device cuda --data-root "$DATA"
echo "embedkit-e7 $SET done"
exit 0
