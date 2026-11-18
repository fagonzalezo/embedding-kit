#!/bin/bash
#SBATCH --job-name=embedkit-e1-hub
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e1-hub-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e1-hub-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=24CPUNodes
#SBATCH --array=0-6%4
#
# WP 5.5: the hubness-reduction methods of E1 (raw+ls, raw+mp, raw+csls, raw+dsl), one task per
# dataset (E1_SETS in env.sh). They change the neighbour search, not the coordinates, so they need
# no GPU. Results go next to the other E1 files; existing files are skipped.
# Submit from this directory:  sbatch e1_hub.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SET=${E1_SETS[$SLURM_ARRAY_TASK_ID]}
# Only exact distance computations here (no linear probe): every CPU for BLAS.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" \
  --experiment e1 --datasets "$SET" --methods raw+ls raw+mp raw+csls raw+dsl \
  --device cpu --data-root "$DATA"
echo "embedkit-e1-hub $SET done"
exit 0
