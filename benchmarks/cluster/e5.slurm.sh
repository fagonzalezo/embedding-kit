#!/bin/bash
#SBATCH --job-name=embedkit-e5
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e5-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e5-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=24CPUNodes
#SBATCH --array=0-51%16
#
# Experiment E5 (WP 4.5): controlled damage of a good embedding.
#   PART=repair sbatch --array=0-51%16 e5.slurm.sh     one (base set, variant) per task: the
#                                                     training-free baselines (run.py e5)
#   PART=detect sbatch --array=0-3 --time=12:00:00 e5.slurm.sh
#                                                     one base set per task: the report on the
#                                                     clean set and its 12 variants, 2 metrics
# Both skip finished files, so a TIMEOUT costs only unfinished work; resubmit the task IDs.
# Results: $REPO/benchmarks/results/e5/ (they stay on the cluster). Submit from this directory.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
PART=${PART:-repair}
E5_SETS=(cifar100_dinov2 banking77_mpnet stl10_clip pets_resnet50)
VARIANTS=("" __dmg_mean1 __dmg_mean5 __dmg_nuis3 __dmg_nuis10 __dmg_aniso100 __dmg_aniso10000
          __dmg_noise1 __dmg_noise3 __dmg_pad1 __dmg_pad4 __dmg_domain1 __dmg_domain3)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ "$PART" = detect ]; then
  SET=${E5_SETS[$SLURM_ARRAY_TASK_ID]}
  srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/e5.py" --datasets "$SET" \
    --data-root "$DATA"
else
  SET=${E5_SETS[$(( SLURM_ARRAY_TASK_ID / ${#VARIANTS[@]} ))]}${VARIANTS[$(( SLURM_ARRAY_TASK_ID % ${#VARIANTS[@]} ))]}
  srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" --experiment e5 \
    --stages test --datasets "$SET" --data-root "$DATA"
fi
echo "embedkit-e5 $PART $SET done"
exit 0
