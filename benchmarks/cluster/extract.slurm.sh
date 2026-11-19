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
#SBATCH --array=0-5%2
#
# One task per GPU feature set (GPU_SETS in env.sh): downloads the raw data and the encoder, and
# writes $DATA/features/<set>.npz. A finished set is skipped, so tasks can be resubmitted.
# Submit from this directory:  sbatch extract.slurm.sh
# E4 (WP 7.1): the 31 new natural sets (E4_GPU_SETS in env.sh) instead; the array bound is the
# number of sets minus 1 (30) and %2 keeps at most 2 GPUs:
#   SETS=E4_GPU_SETS sbatch --array=0-30%2 extract.slurm.sh
# WP 8.3: the 25 extra sets:  SETS=E4X_SETS sbatch --array=0-24%2 extract.slurm.sh
# (SETS names an array of env.sh, GPU_SETS, E4_GPU_SETS or E4X_SETS; the header's 0-5%2 is for
# GPU_SETS.)
# The ANN-Benchmarks sets need no GPU:  sbatch extract_cpu.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
case "${SETS:-GPU_SETS}" in
  GPU_SETS)    ALL=("${GPU_SETS[@]}") ;;
  E4_GPU_SETS) ALL=("${E4_GPU_SETS[@]}") ;;
  E4X_SETS)    ALL=("${E4X_SETS[@]}") ;;
  *) echo "SETS=${SETS} is not GPU_SETS, E4_GPU_SETS or E4X_SETS"; exit 1 ;;
esac
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ALL[@]}" ]; then
  echo "array task $SLURM_ARRAY_TASK_ID is out of range: ${SETS:-GPU_SETS} has ${#ALL[@]} sets"; exit 1
fi
SET=${ALL[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/extract_features.py" \
  --dataset "$SET" --data-root "$DATA" --device cuda
echo "embedkit-extract $SET done"
exit 0
