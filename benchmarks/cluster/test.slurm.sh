#!/bin/bash
#SBATCH --job-name=embedkit-test-gpu
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-test-gpu-%j.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-test-gpu-%j.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --partition=RTX8000Nodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#
# The test suite on a CUDA node (the plan's WP 1.3 asks for the loss tests to run on CUDA once
# before phase 3). The trainer selects CUDA by default there. Submit from this directory:
#   sbatch test.slurm.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$REPO"
srun singularity exec --nv "$CONTAINER" "$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
srun singularity exec --nv "$CONTAINER" "$PY" -m pytest tests -q -p no:cacheprovider
echo "embedkit-test-gpu done"
exit 0
