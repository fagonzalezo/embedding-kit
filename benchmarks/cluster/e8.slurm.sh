#!/bin/bash
#SBATCH --job-name=embedkit-e8
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e8-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e8-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-14%2
#
# Experiment E8 (phase 6: supervised and regression modes), one task per entry of E8_TASKS in
# env.sh: a dataset, datasets joined by "+" (run one after the other), or dataset:I/N for shard I
# of N of its method grid (CIFAR-100 is split so each task fits 12 h), optionally followed by
# "@m1,m2" to run only those methods. TASKS names another task array of env.sh (E8R_TASKS).
# Resumable as run.py skips finished methods; resubmit the task IDs after a TIMEOUT.
# Results: $REPO/benchmarks/results/e8/<dataset>/<method>.json, brought back with rsync.
# Submit from this directory:  sbatch e8.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SEEDS=${SEEDS:-5}
declare -n TASK_LIST=${TASKS:-E8_TASKS}
ENTRY=${TASK_LIST[$SLURM_ARRAY_TASK_ID]}
METHODS=()
if [[ $ENTRY == *@* ]]; then IFS=, read -r -a METHODS <<< "${ENTRY#*@}"; ENTRY=${ENTRY%%@*}; fi
SET=${ENTRY%%:*}
read -r -a SETS <<< "${SET//+/ }"
SHARD=()
if [[ $ENTRY == *:* ]]; then SHARD=(--shard "${ENTRY#*:}"); fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" \
  --experiment e8 --datasets "${SETS[@]}" ${SHARD[@]+"${SHARD[@]}"} \
  ${METHODS[@]+--methods "${METHODS[@]}"} --seeds "$SEEDS" --device cuda --data-root "$DATA"
echo "embedkit-e8 $ENTRY done"
exit 0
