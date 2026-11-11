#!/bin/bash
#SBATCH --job-name=embedkit-e2
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e2-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e2-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-3%2
#
# Experiment E2, one task per dataset (E2_SETS in env.sh). benchmarks/e2.py skips configurations
# whose result file exists, so a TIMEOUT costs only the unfinished run; resubmit the task IDs.
# Results: $REPO/benchmarks/results/e2/<dataset>/<config>.json, brought back with rsync.
# Submit from this directory:  sbatch e2.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SET=${E2_SETS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/e2.py" \
  --datasets "$SET" --device cuda --data-root "$DATA"
echo "embedkit-e2 $SET done"
exit 0
