#!/bin/bash
#SBATCH --job-name=embedkit-e1
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e1-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e1-%A_%a.err
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
# Experiment E1, one task per dataset (E1_SETS in env.sh). benchmarks/run.py skips methods whose
# result file exists, so a TIMEOUT costs only the unfinished method; resubmit the task IDs.
# Results: $REPO/benchmarks/results/e1/<dataset>/<method>.json, brought back with rsync.
# Submit from this directory:  sbatch run.slurm.sh       (EXPERIMENT=e1 SEEDS=5 by default)
# Size --time from the smoke run first:  sbatch --array=1 --time=01:00:00 run.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
EXPERIMENT=${EXPERIMENT:-e1}
SEEDS=${SEEDS:-5}
SET=${E1_SETS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" \
  --experiment "$EXPERIMENT" --datasets "$SET" --seeds "$SEEDS" --device cuda --data-root "$DATA"
echo "embedkit-$EXPERIMENT $SET done"
exit 0
