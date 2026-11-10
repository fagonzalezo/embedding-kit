#!/bin/bash
#SBATCH --job-name=embedkit-extract
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-extract-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-extract-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-3%2
#
# One task per GPU feature set (GPU_SETS in env.sh): downloads the raw data and the encoder, and
# writes $DATA/features/<set>.npz. A finished set is skipped, so tasks can be resubmitted.
# Submit from this directory:  sbatch extract.slurm.sh
# The ANN-Benchmarks sets need no GPU:  sbatch extract_cpu.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SET=${GPU_SETS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/extract_features.py" \
  --dataset "$SET" --data-root "$DATA" --device cuda
echo "embedkit-extract $SET done"
exit 0
