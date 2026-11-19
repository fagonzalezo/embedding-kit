#!/bin/bash
#SBATCH --job-name=embedkit-e10
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e10-%A_%a.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e10-%A_%a.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-34%2
#
# Experiment E10 (WP 8.2 and WP 8.3): EmbedKit() with labels, the default selection, on the 35 E4
# sets (E4_SETS in env.sh, one set per array task, tasks 0-34), with selection_max_rows at the
# default 50,000, at None where that binds, and at 5,000, 10,000 and 20,000 where they bind;
# plus each candidate fitted on its own (benchmarks/experiments.py, e10_methods). Test stage
# only (fit on the training split, score on the test split), one seed: the selection makes its
# own validation split inside the training rows.
#   sbatch e10.slurm.sh
# One GPU per task (the refiner candidate), 2 at a time (the iris GPU limit). The largest sets
# (CIFAR, 50,000 x 2,048; AG News, 120,000 rows) train the refiner up to three times. Finished
# (set, method) files are skipped, so a TIMEOUT costs only unfinished work: resubmit the task
# IDs, for example `sbatch --array=3,17 e10.slurm.sh`.
# Results: $REPO/benchmarks/results/e10/ (they stay on the cluster). Submit from this directory.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SET=${E4_SETS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec --nv "$CONTAINER" "$PY" "$REPO/benchmarks/run.py" --experiment e10 \
  --datasets "$SET" --data-root "$DATA" --stages test --seeds 1 --device cuda
echo "embedkit-e10 $SET done"
exit 0
