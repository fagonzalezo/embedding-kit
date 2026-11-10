#!/bin/bash
#SBATCH --job-name=embedkit-extract-cpu
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-extract-cpu-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-extract-cpu-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --partition=24CPUNodes
#SBATCH --array=0-2
#
# One task per ANN-Benchmarks set (CPU_SETS in env.sh): downloads the HDF5 file and writes
# $DATA/features/<set>.npz. Submit from this directory:  sbatch extract_cpu.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SET=${CPU_SETS[$SLURM_ARRAY_TASK_ID]}
srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/extract_features.py" \
  --dataset "$SET" --data-root "$DATA" --device cpu
echo "embedkit-extract-cpu $SET done"
exit 0
